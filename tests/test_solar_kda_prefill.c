/* Solar Open 2 CUDA KDA chunked-prefill boundary test.
 *
 * The single-token test already uses Solar's production head_dim=128. This
 * test reduces head_dim to keep six long boundary cases cheap, then verifies
 * the required chunk sizes 128..4096, an arbitrary seven-token remainder,
 * every output, and final recurrent/conv state against a scalar continuation.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    T_HEAD = 2,
    T_DIM = 16,
    T_CONV = 4,
    T_VECTOR = T_HEAD * T_DIM,
    T_STATE = T_HEAD * T_DIM * T_DIM,
    T_CONV_STATE = T_VECTOR * T_CONV,
    T_MAX_CHUNK = 4096,
    T_REMAINDER = 7,
};

typedef struct {
    float state[T_STATE];
    float q_conv[T_CONV_STATE];
    float k_conv[T_CONV_STATE];
    float v_conv[T_CONV_STATE];
} host_state;

static int failures;

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); exit(1); } } while (0)

static float softplus_ref(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float silu_ref(float x) {
    return x / (1.0f + expf(-x));
}

static void make_fixed(float *qw, float *kw, float *vw, float *decay, float *dt) {
    for (uint32_t i = 0; i < T_CONV_STATE; i++) {
        const float x = (float)(i + 1u);
        qw[i] = 0.10f * sinf(0.031f * x) + 0.03f;
        kw[i] = 0.09f * cosf(0.027f * x) - 0.02f;
        vw[i] = 0.08f * sinf(0.023f * x + 0.2f) + 0.01f;
    }
    for (uint32_t h = 0; h < T_HEAD; h++) decay[h] = -expf(-0.4f + 0.2f * h);
    for (uint32_t i = 0; i < T_VECTOR; i++) dt[i] = 0.12f * sinf(0.07f * i) - 0.05f;
}

static void make_token(uint32_t token, float *q, float *k, float *v,
                       float *g, float *beta) {
    const float t = (float)(token + 1u);
    for (uint32_t i = 0; i < T_VECTOR; i++) {
        const float x = (float)(i + 1u);
        q[i] = 0.19f * sinf(0.009f * x * t) + 0.01f * (float)(i % 3u);
        k[i] = 0.17f * cosf(0.011f * x * (t + 0.25f));
        v[i] = 0.15f * sinf(0.013f * x + 0.03f * t);
        g[i] = ((i + token) % 97u == 0u) ? 11.0f : 0.5f * sinf(0.017f * x - 0.02f * t);
    }
    for (uint32_t h = 0; h < T_HEAD; h++) beta[h] = -0.7f + 0.3f * h + 0.001f * t;
}

static void conv(float *out, float *state, const float *raw, const float *weight) {
    for (uint32_t ch = 0; ch < T_VECTOR; ch++) {
        float *row = state + (size_t)ch * T_CONV;
        for (uint32_t i = 0; i + 1u < T_CONV; i++) row[i] = row[i + 1u];
        row[T_CONV - 1u] = raw[ch];
        float sum = 0.0f;
        for (uint32_t i = 0; i < T_CONV; i++) sum += row[i] * weight[(size_t)ch * T_CONV + i];
        out[ch] = silu_ref(sum);
    }
}

static void host_step(float *out, host_state *s,
                      const float *q_raw, const float *k_raw, const float *v_raw,
                      const float *g_raw, const float *beta_logits,
                      const float *qw, const float *kw, const float *vw,
                      const float *decay, const float *dt) {
    float q[T_VECTOR], k[T_VECTOR], v[T_VECTOR];
    conv(q, s->q_conv, q_raw, qw);
    conv(k, s->k_conv, k_raw, kw);
    conv(v, s->v_conv, v_raw, vw);
    for (uint32_t h = 0; h < T_HEAD; h++) {
        const uint32_t base = h * T_DIM;
        float q2 = 1.0e-6f, k2 = 1.0e-6f;
        for (uint32_t d = 0; d < T_DIM; d++) {
            q2 += q[base + d] * q[base + d];
            k2 += k[base + d] * k[base + d];
        }
        const float qs = 1.0f / sqrtf(q2) / sqrtf((float)T_DIM);
        const float ks = 1.0f / sqrtf(k2);
        for (uint32_t d = 0; d < T_DIM; d++) { q[base + d] *= qs; k[base + d] *= ks; }
        const float beta = 2.0f / (1.0f + expf(-beta_logits[h]));
        const size_t sb = (size_t)h * T_DIM * T_DIM;
        for (uint32_t vd = 0; vd < T_DIM; vd++) {
            float memory = 0.0f;
            for (uint32_t kd = 0; kd < T_DIM; kd++) {
                float gate = decay[h] * softplus_ref(g_raw[base + kd] + dt[base + kd]);
                if (gate < -5.0f) gate = -5.0f;
                const size_t ix = sb + (size_t)kd * T_DIM + vd;
                s->state[ix] *= expf(gate);
                memory += s->state[ix] * k[base + kd];
            }
            const float delta = (v[base + vd] - memory) * beta;
            float result = 0.0f;
            for (uint32_t kd = 0; kd < T_DIM; kd++) {
                const size_t ix = sb + (size_t)kd * T_DIM + vd;
                s->state[ix] += k[base + kd] * delta;
                result += s->state[ix] * q[base + kd];
            }
            out[base + vd] = result;
        }
    }
}

static void compare(const char *label, const float *got, const float *want,
                    size_t n, double atol, double rtol) {
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double delta = fabs((double)got[i] - want[i]);
        const double scale = fmax(fabs((double)got[i]), fabs((double)want[i]));
        if (delta > max_abs) max_abs = delta;
        if (scale > 1.0e-8 && delta / scale > max_rel) max_rel = delta / scale;
    }
    const int ok = max_abs <= atol || max_rel <= rtol;
    if (!ok) failures++;
    printf("%-28s abs=%.3e rel=%.3e %s\n", label, max_abs, max_rel, ok ? "ok" : "FAIL");
}

int main(void) {
    CHECK(ds4_gpu_init(), "CUDA init");
    const uint32_t sizes[] = {128, 256, 512, 1024, 2048, 4096};
    const size_t max_vectors = (size_t)(T_MAX_CHUNK + T_REMAINDER) * T_VECTOR;
    const size_t max_betas = (size_t)(T_MAX_CHUNK + T_REMAINDER) * T_HEAD;
    float *q = calloc(max_vectors, sizeof(float));
    float *k = calloc(max_vectors, sizeof(float));
    float *v = calloc(max_vectors, sizeof(float));
    float *g = calloc(max_vectors, sizeof(float));
    float *beta = calloc(max_betas, sizeof(float));
    float *want = calloc(max_vectors, sizeof(float));
    float *got = calloc(max_vectors, sizeof(float));
    CHECK(q && k && v && g && beta && want && got, "host arrays");

    float qw[T_CONV_STATE], kw[T_CONV_STATE], vw[T_CONV_STATE], decay[T_HEAD], dt[T_VECTOR];
    make_fixed(qw, kw, vw, decay, dt);

    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc((uint64_t)T_MAX_CHUNK * T_VECTOR * sizeof(float));
    ds4_gpu_tensor *dk = ds4_gpu_tensor_alloc((uint64_t)T_MAX_CHUNK * T_VECTOR * sizeof(float));
    ds4_gpu_tensor *dv = ds4_gpu_tensor_alloc((uint64_t)T_MAX_CHUNK * T_VECTOR * sizeof(float));
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc((uint64_t)T_MAX_CHUNK * T_VECTOR * sizeof(float));
    ds4_gpu_tensor *db = ds4_gpu_tensor_alloc((uint64_t)T_MAX_CHUNK * T_HEAD * sizeof(float));
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc((uint64_t)T_MAX_CHUNK * T_VECTOR * sizeof(float));
    ds4_gpu_tensor *dstate = ds4_gpu_tensor_alloc((uint64_t)T_STATE * sizeof(float));
    ds4_gpu_tensor *dqc = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    ds4_gpu_tensor *dkc = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    ds4_gpu_tensor *dvc = ds4_gpu_tensor_alloc((uint64_t)T_CONV_STATE * sizeof(float));
    ds4_gpu_tensor *dqw = ds4_gpu_tensor_alloc(sizeof(qw));
    ds4_gpu_tensor *dkw = ds4_gpu_tensor_alloc(sizeof(kw));
    ds4_gpu_tensor *dvw = ds4_gpu_tensor_alloc(sizeof(vw));
    ds4_gpu_tensor *dda = ds4_gpu_tensor_alloc(sizeof(decay));
    ds4_gpu_tensor *ddt = ds4_gpu_tensor_alloc(sizeof(dt));
    CHECK(dq && dk && dv && dg && db && dout && dstate && dqc && dkc && dvc &&
          dqw && dkw && dvw && dda && ddt, "device arrays");
    CHECK(ds4_gpu_tensor_write(dqw, 0, qw, sizeof(qw)), "q weights");
    CHECK(ds4_gpu_tensor_write(dkw, 0, kw, sizeof(kw)), "k weights");
    CHECK(ds4_gpu_tensor_write(dvw, 0, vw, sizeof(vw)), "v weights");
    CHECK(ds4_gpu_tensor_write(dda, 0, decay, sizeof(decay)), "decay");
    CHECK(ds4_gpu_tensor_write(ddt, 0, dt, sizeof(dt)), "dt");

    puts("== Solar Open 2 CUDA KDA chunked prefill ==");
    for (size_t ci = 0; ci < sizeof(sizes) / sizeof(sizes[0]); ci++) {
        const uint32_t chunk = sizes[ci];
        const uint32_t total = chunk + T_REMAINDER;
        host_state hs;
        memset(&hs, 0, sizeof(hs));
        memset(got, 0, (size_t)total * T_VECTOR * sizeof(float));
        for (uint32_t t = 0; t < total; t++) {
            make_token(t, q + (size_t)t * T_VECTOR, k + (size_t)t * T_VECTOR,
                       v + (size_t)t * T_VECTOR, g + (size_t)t * T_VECTOR,
                       beta + (size_t)t * T_HEAD);
            host_step(want + (size_t)t * T_VECTOR, &hs,
                      q + (size_t)t * T_VECTOR, k + (size_t)t * T_VECTOR,
                      v + (size_t)t * T_VECTOR, g + (size_t)t * T_VECTOR,
                      beta + (size_t)t * T_HEAD, qw, kw, vw, decay, dt);
        }

        CHECK(ds4_gpu_tensor_fill_f32(dstate, 0.0f, T_STATE), "state reset");
        CHECK(ds4_gpu_tensor_fill_f32(dqc, 0.0f, T_CONV_STATE), "q conv reset");
        CHECK(ds4_gpu_tensor_fill_f32(dkc, 0.0f, T_CONV_STATE), "k conv reset");
        CHECK(ds4_gpu_tensor_fill_f32(dvc, 0.0f, T_CONV_STATE), "v conv reset");

        uint32_t done = 0;
        while (done < total) {
            const uint32_t n = total - done > chunk ? chunk : total - done;
            const uint64_t vb = (uint64_t)n * T_VECTOR * sizeof(float);
            const uint64_t bb = (uint64_t)n * T_HEAD * sizeof(float);
            CHECK(ds4_gpu_tensor_write(dq, 0, q + (size_t)done * T_VECTOR, vb), "write q chunk");
            CHECK(ds4_gpu_tensor_write(dk, 0, k + (size_t)done * T_VECTOR, vb), "write k chunk");
            CHECK(ds4_gpu_tensor_write(dv, 0, v + (size_t)done * T_VECTOR, vb), "write v chunk");
            CHECK(ds4_gpu_tensor_write(dg, 0, g + (size_t)done * T_VECTOR, vb), "write g chunk");
            CHECK(ds4_gpu_tensor_write(db, 0, beta + (size_t)done * T_HEAD, bb), "write beta chunk");
            CHECK(ds4_gpu_solar_kda_prefill_tensor(
                      dout, dstate, dqc, dkc, dvc, dq, dk, dv, dg, db,
                      dqw, dkw, dvw, dda, ddt, n, T_HEAD, T_DIM, T_CONV, -5.0f),
                  "prefill launch");
            CHECK(ds4_gpu_tensor_read(dout, 0, got + (size_t)done * T_VECTOR, vb),
                  "read output chunk");
            done += n;
        }

        char label[64];
        snprintf(label, sizeof(label), "chunk %u + remainder", chunk);
        compare(label, got, want, (size_t)total * T_VECTOR, 4.0e-5, 5.0e-4);
        float final_state[T_STATE];
        CHECK(ds4_gpu_tensor_read(dstate, 0, final_state, sizeof(final_state)), "read state");
        snprintf(label, sizeof(label), "chunk %u final state", chunk);
        compare(label, final_state, hs.state, T_STATE, 5.0e-5, 8.0e-4);
    }

    ds4_gpu_tensor_free(dq); ds4_gpu_tensor_free(dk); ds4_gpu_tensor_free(dv);
    ds4_gpu_tensor_free(dg); ds4_gpu_tensor_free(db); ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(dstate); ds4_gpu_tensor_free(dqc); ds4_gpu_tensor_free(dkc);
    ds4_gpu_tensor_free(dvc); ds4_gpu_tensor_free(dqw); ds4_gpu_tensor_free(dkw);
    ds4_gpu_tensor_free(dvw); ds4_gpu_tensor_free(dda); ds4_gpu_tensor_free(ddt);
    free(q); free(k); free(v); free(g); free(beta); free(want); free(got);
    ds4_gpu_cleanup();
    puts(failures ? "Solar KDA prefill checks FAILED" : "all Solar KDA prefill checks passed");
    return failures ? 1 : 0;
}
