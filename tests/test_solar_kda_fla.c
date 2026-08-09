#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t tokens;
    uint32_t heads;
    uint32_t dim;
    uint32_t conv;
    float *q;
    float *k;
    float *v;
    float *g;
    float *beta;
    float *qw;
    float *kw;
    float *vw;
    float *decay;
    float *dt;
    float *prefill_out;
    float *prefill_state;
    float *q_prefill_state;
    float *k_prefill_state;
    float *v_prefill_state;
    float *decode_out;
    float *decode_state;
    float *q_decode_state;
    float *k_decode_state;
    float *v_decode_state;
} fla_fixture;

static int failures;

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); exit(1); } } while (0)

static float *read_f32(FILE *fp, size_t count, const char *label) {
    float *data = malloc(count * sizeof(float));
    if (!data || fread(data, sizeof(float), count, fp) != count) {
        fprintf(stderr, "FAIL: fixture array %s\n", label);
        free(data);
        return NULL;
    }
    return data;
}

static int fixture_load(fla_fixture *f, const char *path) {
    memset(f, 0, sizeof(*f));
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;

    char magic[8];
    uint32_t version;
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) ||
        fread(&version, sizeof(version), 1, fp) != 1 ||
        fread(&f->tokens, sizeof(f->tokens), 1, fp) != 1 ||
        fread(&f->heads, sizeof(f->heads), 1, fp) != 1 ||
        fread(&f->dim, sizeof(f->dim), 1, fp) != 1 ||
        fread(&f->conv, sizeof(f->conv), 1, fp) != 1 ||
        memcmp(magic, "SLR2KDA1", sizeof(magic)) != 0 || version != 1u) {
        fclose(fp);
        return 0;
    }
    if (f->tokens == 0u || f->tokens > 4096u || f->heads == 0u || f->heads > 64u ||
        f->dim == 0u || f->dim > 256u || f->conv == 0u || f->conv > 16u) {
        fclose(fp);
        return 0;
    }

    const size_t total = (size_t)f->tokens + 1u;
    const size_t vector = (size_t)f->heads * f->dim;
    const size_t state = (size_t)f->heads * f->dim * f->dim;
    const size_t conv_state = vector * f->conv;
#define READ(field, count) do { f->field = read_f32(fp, (count), #field); if (!f->field) goto fail; } while (0)
    READ(q, total * vector);
    READ(k, total * vector);
    READ(v, total * vector);
    READ(g, total * vector);
    READ(beta, total * f->heads);
    READ(qw, conv_state);
    READ(kw, conv_state);
    READ(vw, conv_state);
    READ(decay, f->heads);
    READ(dt, vector);
    READ(prefill_out, (size_t)f->tokens * vector);
    READ(prefill_state, state);
    READ(q_prefill_state, conv_state);
    READ(k_prefill_state, conv_state);
    READ(v_prefill_state, conv_state);
    READ(decode_out, vector);
    READ(decode_state, state);
    READ(q_decode_state, conv_state);
    READ(k_decode_state, conv_state);
    READ(v_decode_state, conv_state);
#undef READ
    if (fgetc(fp) != EOF) goto fail;
    fclose(fp);
    return 1;

fail:
    fclose(fp);
    return 0;
}

static void fixture_free(fla_fixture *f) {
    free(f->q); free(f->k); free(f->v); free(f->g); free(f->beta);
    free(f->qw); free(f->kw); free(f->vw); free(f->decay); free(f->dt);
    free(f->prefill_out); free(f->prefill_state);
    free(f->q_prefill_state); free(f->k_prefill_state); free(f->v_prefill_state);
    free(f->decode_out); free(f->decode_state);
    free(f->q_decode_state); free(f->k_decode_state); free(f->v_decode_state);
    memset(f, 0, sizeof(*f));
}

static void compare(const char *label, const float *got, const float *want,
                    size_t count, double atol, double rtol) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double delta = fabs((double)got[i] - want[i]);
        const double scale = fmax(fabs((double)got[i]), fabs((double)want[i]));
        if (delta > max_abs) max_abs = delta;
        if (scale > 1.0e-7 && delta / scale > max_rel) max_rel = delta / scale;
    }
    const int ok = max_abs <= atol || max_rel <= rtol;
    if (!ok) failures++;
    printf("%-30s abs=%.6e rel=%.6e %s\n", label, max_abs, max_rel, ok ? "ok" : "FAIL");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s synthetic-fla.bin\n", argv[0]);
        return 2;
    }
    fla_fixture f;
    CHECK(fixture_load(&f, argv[1]), "load official FLA fixture");
    CHECK(ds4_gpu_init(), "CUDA init");

    const size_t total = (size_t)f.tokens + 1u;
    const size_t vector = (size_t)f.heads * f.dim;
    const size_t state = (size_t)f.heads * f.dim * f.dim;
    const size_t conv_state = vector * f.conv;
    const uint64_t vector_bytes = (uint64_t)vector * sizeof(float);

    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc(total * vector_bytes);
    ds4_gpu_tensor *dk = ds4_gpu_tensor_alloc(total * vector_bytes);
    ds4_gpu_tensor *dv = ds4_gpu_tensor_alloc(total * vector_bytes);
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(total * vector_bytes);
    ds4_gpu_tensor *db = ds4_gpu_tensor_alloc((uint64_t)total * f.heads * sizeof(float));
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc((uint64_t)f.tokens * vector_bytes);
    ds4_gpu_tensor *dstate = ds4_gpu_tensor_alloc((uint64_t)state * sizeof(float));
    ds4_gpu_tensor *dqc = ds4_gpu_tensor_alloc((uint64_t)conv_state * sizeof(float));
    ds4_gpu_tensor *dkc = ds4_gpu_tensor_alloc((uint64_t)conv_state * sizeof(float));
    ds4_gpu_tensor *dvc = ds4_gpu_tensor_alloc((uint64_t)conv_state * sizeof(float));
    ds4_gpu_tensor *dqw = ds4_gpu_tensor_alloc((uint64_t)conv_state * sizeof(float));
    ds4_gpu_tensor *dkw = ds4_gpu_tensor_alloc((uint64_t)conv_state * sizeof(float));
    ds4_gpu_tensor *dvw = ds4_gpu_tensor_alloc((uint64_t)conv_state * sizeof(float));
    ds4_gpu_tensor *dda = ds4_gpu_tensor_alloc((uint64_t)f.heads * sizeof(float));
    ds4_gpu_tensor *ddt = ds4_gpu_tensor_alloc(vector_bytes);
    CHECK(dq && dk && dv && dg && db && dout && dstate && dqc && dkc && dvc &&
          dqw && dkw && dvw && dda && ddt, "device allocation");

    const uint64_t prefill_vector_bytes = (uint64_t)f.tokens * vector_bytes;
    CHECK(ds4_gpu_tensor_write(dq, 0, f.q, prefill_vector_bytes), "write q");
    CHECK(ds4_gpu_tensor_write(dk, 0, f.k, prefill_vector_bytes), "write k");
    CHECK(ds4_gpu_tensor_write(dv, 0, f.v, prefill_vector_bytes), "write v");
    CHECK(ds4_gpu_tensor_write(dg, 0, f.g, prefill_vector_bytes), "write gate");
    CHECK(ds4_gpu_tensor_write(db, 0, f.beta, (uint64_t)f.tokens * f.heads * sizeof(float)), "write beta");
    CHECK(ds4_gpu_tensor_write(dqw, 0, f.qw, (uint64_t)conv_state * sizeof(float)), "write q conv");
    CHECK(ds4_gpu_tensor_write(dkw, 0, f.kw, (uint64_t)conv_state * sizeof(float)), "write k conv");
    CHECK(ds4_gpu_tensor_write(dvw, 0, f.vw, (uint64_t)conv_state * sizeof(float)), "write v conv");
    CHECK(ds4_gpu_tensor_write(dda, 0, f.decay, (uint64_t)f.heads * sizeof(float)), "write decay");
    CHECK(ds4_gpu_tensor_write(ddt, 0, f.dt, vector_bytes), "write dt");
    CHECK(ds4_gpu_tensor_fill_f32(dstate, 0.0f, state), "reset state");
    CHECK(ds4_gpu_tensor_fill_f32(dqc, 0.0f, conv_state), "reset q conv");
    CHECK(ds4_gpu_tensor_fill_f32(dkc, 0.0f, conv_state), "reset k conv");
    CHECK(ds4_gpu_tensor_fill_f32(dvc, 0.0f, conv_state), "reset v conv");

    CHECK(ds4_gpu_solar_kda_prefill_tensor(
              dout, dstate, dqc, dkc, dvc, dq, dk, dv, dg, db,
              dqw, dkw, dvw, dda, ddt, f.tokens, f.heads, f.dim, f.conv, -5.0f),
          "official fixture prefill");
    float *scratch = malloc((size_t)f.tokens * vector * sizeof(float));
    float *state_scratch = malloc(state * sizeof(float));
    float *conv_scratch = malloc(conv_state * sizeof(float));
    CHECK(scratch && state_scratch && conv_scratch, "host scratch allocation");
    CHECK(ds4_gpu_tensor_read(dout, 0, scratch, prefill_vector_bytes), "read prefill output");
    compare("FLA chunk prefill output", scratch, f.prefill_out, (size_t)f.tokens * vector, 1.0e-3, 5.0e-2);
    CHECK(ds4_gpu_tensor_read(dstate, 0, state_scratch, (uint64_t)state * sizeof(float)), "read prefill state");
    compare("FLA chunk final state", state_scratch, f.prefill_state, state, 2.0e-2, 5.0e-2);
    CHECK(ds4_gpu_tensor_read(dqc, 0, conv_scratch, (uint64_t)conv_state * sizeof(float)), "read q conv state");
    compare("FLA q conv prefill state", conv_scratch, f.q_prefill_state, conv_state, 0.0, 0.0);
    CHECK(ds4_gpu_tensor_read(dkc, 0, conv_scratch, (uint64_t)conv_state * sizeof(float)), "read k conv state");
    compare("FLA k conv prefill state", conv_scratch, f.k_prefill_state, conv_state, 0.0, 0.0);
    CHECK(ds4_gpu_tensor_read(dvc, 0, conv_scratch, (uint64_t)conv_state * sizeof(float)), "read v conv state");
    compare("FLA v conv prefill state", conv_scratch, f.v_prefill_state, conv_state, 0.0, 0.0);

    CHECK(ds4_gpu_tensor_write(dq, 0, f.q + (size_t)f.tokens * vector, vector_bytes), "write decode q");
    CHECK(ds4_gpu_tensor_write(dk, 0, f.k + (size_t)f.tokens * vector, vector_bytes), "write decode k");
    CHECK(ds4_gpu_tensor_write(dv, 0, f.v + (size_t)f.tokens * vector, vector_bytes), "write decode v");
    CHECK(ds4_gpu_tensor_write(dg, 0, f.g + (size_t)f.tokens * vector, vector_bytes), "write decode gate");
    CHECK(ds4_gpu_tensor_write(db, 0, f.beta + (size_t)f.tokens * f.heads,
                               (uint64_t)f.heads * sizeof(float)), "write decode beta");
    CHECK(ds4_gpu_solar_kda_decode_tensor(
              dout, dstate, dqc, dkc, dvc, dq, dk, dv, dg, db,
              dqw, dkw, dvw, dda, ddt, f.heads, f.dim, f.conv, -5.0f),
          "official fixture decode");
    CHECK(ds4_gpu_tensor_read(dout, 0, scratch, vector_bytes), "read decode output");
    compare("FLA recurrent decode output", scratch, f.decode_out, vector, 1.0e-3, 5.0e-2);
    CHECK(ds4_gpu_tensor_read(dstate, 0, state_scratch, (uint64_t)state * sizeof(float)), "read decode state");
    compare("FLA recurrent decode state", state_scratch, f.decode_state, state, 2.0e-2, 5.0e-2);
    CHECK(ds4_gpu_tensor_read(dqc, 0, conv_scratch, (uint64_t)conv_state * sizeof(float)), "read q decode state");
    compare("FLA q conv decode state", conv_scratch, f.q_decode_state, conv_state, 0.0, 0.0);
    CHECK(ds4_gpu_tensor_read(dkc, 0, conv_scratch, (uint64_t)conv_state * sizeof(float)), "read k decode state");
    compare("FLA k conv decode state", conv_scratch, f.k_decode_state, conv_state, 0.0, 0.0);
    CHECK(ds4_gpu_tensor_read(dvc, 0, conv_scratch, (uint64_t)conv_state * sizeof(float)), "read v decode state");
    compare("FLA v conv decode state", conv_scratch, f.v_decode_state, conv_state, 0.0, 0.0);

    free(scratch); free(state_scratch); free(conv_scratch);
    ds4_gpu_tensor_free(dq); ds4_gpu_tensor_free(dk); ds4_gpu_tensor_free(dv);
    ds4_gpu_tensor_free(dg); ds4_gpu_tensor_free(db); ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(dstate); ds4_gpu_tensor_free(dqc); ds4_gpu_tensor_free(dkc); ds4_gpu_tensor_free(dvc);
    ds4_gpu_tensor_free(dqw); ds4_gpu_tensor_free(dkw); ds4_gpu_tensor_free(dvw);
    ds4_gpu_tensor_free(dda); ds4_gpu_tensor_free(ddt);
    ds4_gpu_cleanup();
    fixture_free(&f);
    puts(failures ? "Solar FLA fixture checks FAILED" : "all Solar FLA fixture checks passed");
    return failures ? 1 : 0;
}
