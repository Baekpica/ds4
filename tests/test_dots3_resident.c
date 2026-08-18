/* dots3-note resident gate: import the VMM weight owner's image, run the
 * CUDA-vs-CPU-reference forward gate, exercise chunked prefill / prefix reuse
 * / decode on the real MQ87 shards, cross the DSA top-2048 boundary, and
 * verify the 256K cache allocates beside the resident model.
 */
#include "../ds4.h"

#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint64_t MIB = 1024ull * 1024ull;

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <dots3 first shard>\n", argv[0]);
        return 2;
    }
    const char *weight_manifest = getenv("DS4_CUDA_WEIGHT_IPC_MANIFEST");
    if (!weight_manifest || !weight_manifest[0]) {
        fprintf(stderr,
                "dots3 residency smoke requires a live VMM weight owner; "
                "set DS4_CUDA_WEIGHT_IPC_MANIFEST\n");
        return 2;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "dots3 residency smoke needs CUDA device 0\n");
        return 1;
    }
    size_t free_before = 0, total = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 1;

    unsetenv("DS4_CUDA_COPY_MODEL");
    unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED");
    unsetenv("DS4_CUDA_NO_MODEL_COPY");
    unsetenv("DS4_CUDA_DIRECT_MODEL");
    setenv("DS4_CUDA_WEIGHT_IPC_SCOPE", "base", 1);

    ds4_engine_options options;
    memset(&options, 0, sizeof(options));
    options.model_path = argv[1];
    options.backend = DS4_BACKEND_CUDA;
    options.n_threads = 1;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0 || !engine) {
        fprintf(stderr, "dots3 resident engine open failed\n");
        return 1;
    }
    size_t free_open = 0, total_open = 0;
    if (cudaMemGetInfo(&free_open, &total_open) != cudaSuccess) {
        ds4_engine_close(engine);
        return 1;
    }
    const uint64_t model_bytes = 86072934272ull; /* unsharded MQ87 payload */
    const uint64_t worker_delta = free_before >= free_open
        ? (uint64_t)(free_before - free_open) : 0;
    const int metadata_ok =
        ds4_engine_model_id(engine) == 5 &&
        ds4_engine_layer_count(engine) == 47 &&
        ds4_engine_vocab_size(engine) == 152064;
    const int no_duplicate_ok = worker_delta + 512ull * MIB < model_bytes;
    printf("dots3 VMM worker open: worker CUDA delta=%" PRIu64 " bytes\n",
           worker_delta);

    /* GPU graph vs the FP32 CPU reference on the official chat prompt. */
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "You are a concise assistant.",
                           "Reply with exactly: OK", DS4_THINK_NONE, &prompt);
    printf("dots3 forward-gate prompt: %d tokens (CPU reference runs once)\n",
           prompt.len);
    const int forward_ok =
        prompt.len > 0 && prompt.len <= 64 &&
        ds4_engine_dots3_forward_test(engine, &prompt) == 0;

    /* Native serial session: sync, decode a few tokens, report them. */
    ds4_session *session = NULL;
    char session_err[256] = "";
    int session_ok = ds4_session_create(&session, engine, 512) == 0 && session;
    if (session_ok) {
        session_ok = ds4_session_sync(
                session, &prompt, session_err, sizeof(session_err)) == 0;
    }
    if (session_ok) {
        fputs("dots3 no-think decode:", stdout);
        int tok = ds4_session_argmax(session);
        for (int i = 0; session_ok && i < 8 && tok >= 0 &&
                        !ds4_token_is_stop(engine, tok); i++) {
            size_t len = 0;
            char *text = ds4_token_text(engine, tok, &len);
            printf(" %.*s", (int)len, text ? text : "");
            free(text);
            session_ok = ds4_session_eval(
                    session, tok, session_err, sizeof(session_err)) == 0;
            tok = ds4_session_argmax(session);
        }
        puts("");
    }
    if (!session_ok) {
        fprintf(stderr, "dots3 native session failed: %s\n",
                session_err[0] ? session_err : "unknown");
    }
    if (session) ds4_session_free(session);

    /* Chunked prefill / SWA ring / prefix reuse parity on a long prompt that
     * crosses the 513-token window several times. */
    ds4_tokens long_prompt = {0};
    long_prompt.len = 1600;
    long_prompt.cap = long_prompt.len;
    long_prompt.v = malloc((size_t)long_prompt.len * sizeof(long_prompt.v[0]));
    for (int i = 0; long_prompt.v && i < long_prompt.len; i++)
        long_prompt.v[i] = prompt.v[i % prompt.len];
    const int vocab = ds4_engine_vocab_size(engine);
    float *one_logits = malloc((size_t)vocab * sizeof(float));
    float *direct_logits = malloc((size_t)vocab * sizeof(float));
    float *chunk_logits = malloc((size_t)vocab * sizeof(float));
    int chunk_ok = one_logits && direct_logits && chunk_logits && long_prompt.v;
    char err[256] = "";
    double one_elapsed = 0.0, decode_elapsed = 0.0;

    ds4_session *one = NULL;
    setenv("DS4_DOTS3_PREFILL_CHUNK", "2048", 1);
    if (chunk_ok) chunk_ok = ds4_session_create(&one, engine, 4096) == 0;
    double t0 = monotonic_seconds();
    if (chunk_ok) chunk_ok = ds4_session_sync(one, &long_prompt, err, sizeof(err)) == 0;
    one_elapsed = monotonic_seconds() - t0;
    if (chunk_ok) chunk_ok = ds4_session_copy_logits(one, one_logits, vocab) == vocab;
    if (one) ds4_session_free(one);

    ds4_session *direct = NULL;
    setenv("DS4_DOTS3_PREFILL_CHUNK", "320", 1);
    if (chunk_ok) chunk_ok = ds4_session_create(&direct, engine, 4096) == 0;
    if (chunk_ok) chunk_ok = ds4_session_sync(direct, &long_prompt, err, sizeof(err)) == 0;
    if (chunk_ok) chunk_ok = ds4_session_copy_logits(direct, direct_logits, vocab) == vocab;
    if (direct) ds4_session_free(direct);

    ds4_session *chunked = NULL;
    if (chunk_ok) chunk_ok = ds4_session_create(&chunked, engine, 4096) == 0;
    ds4_tokens prefix = long_prompt;
    prefix.len = 700;
    if (chunk_ok) chunk_ok = ds4_session_sync(chunked, &prefix, err, sizeof(err)) == 0;
    if (chunk_ok) chunk_ok = ds4_session_sync(chunked, &long_prompt, err, sizeof(err)) == 0;
    if (chunk_ok) chunk_ok = ds4_session_copy_logits(chunked, chunk_logits, vocab) == vocab;

    double err2 = 0.0, ref2 = 0.0, got2 = 0.0, dot = 0.0;
    double cerr2 = 0.0, cref2 = 0.0, cgot2 = 0.0, cdot = 0.0;
    int one_best = -1, chunk_best = -1;
    if (chunk_ok) {
        for (int i = 0; i < vocab; i++) {
            if (!isfinite(one_logits[i]) || !isfinite(direct_logits[i]) ||
                !isfinite(chunk_logits[i])) { chunk_ok = 0; break; }
            const double d = (double)direct_logits[i] - one_logits[i];
            err2 += d * d;
            ref2 += (double)one_logits[i] * one_logits[i];
            got2 += (double)direct_logits[i] * direct_logits[i];
            dot += (double)direct_logits[i] * one_logits[i];
            const double cd = (double)chunk_logits[i] - direct_logits[i];
            cerr2 += cd * cd;
            cref2 += (double)direct_logits[i] * direct_logits[i];
            cgot2 += (double)chunk_logits[i] * chunk_logits[i];
            cdot += (double)chunk_logits[i] * direct_logits[i];
            if (one_best < 0 || one_logits[i] > one_logits[one_best]) one_best = i;
            if (chunk_best < 0 || chunk_logits[i] > chunk_logits[chunk_best]) chunk_best = i;
        }
    }
    const double batch_cos = dot / sqrt(fmax(ref2 * got2, 1.0e-30));
    const double batch_nrmse = sqrt(err2 / fmax(ref2, 1.0e-30));
    const double cache_cos = cdot / sqrt(fmax(cref2 * cgot2, 1.0e-30));
    const double cache_nrmse = sqrt(cerr2 / fmax(cref2, 1.0e-30));
    chunk_ok = chunk_ok && one_best == chunk_best &&
               batch_cos >= 0.98 && batch_nrmse <= 0.2 &&
               cache_cos >= 0.9999 && cache_nrmse <= 0.02;
    if (chunk_ok) {
        const int decode_token = ds4_session_argmax(chunked);
        t0 = monotonic_seconds();
        chunk_ok = ds4_session_eval(chunked, decode_token, err, sizeof(err)) == 0 &&
                   ds4_session_pos(chunked) == long_prompt.len + 1;
        decode_elapsed = monotonic_seconds() - t0;
    }
    printf("dots3 chunk/ring parity: first=%d/%d batch_cos=%.8f "
           "batch_nrmse=%.6g cache_cos=%.8f cache_nrmse=%.6g; "
           "one-chunk %.2f tok/s, decode %.3f tok/s\n",
           chunk_best, one_best, batch_cos, batch_nrmse, cache_cos,
           cache_nrmse,
           one_elapsed > 0.0 ? (double)long_prompt.len / one_elapsed : 0.0,
           decode_elapsed > 0.0 ? 1.0 / decode_elapsed : 0.0);
    if (!chunk_ok) {
        fprintf(stderr, "dots3 chunk/ring session failed: %s\n",
                err[0] ? err : "logit parity mismatch");
    }
    if (chunked) ds4_session_free(chunked);
    unsetenv("DS4_DOTS3_PREFILL_CHUNK");
    free(one_logits); free(direct_logits); free(chunk_logits);

    /* Cross the DSA top-2048 boundary: two identical 2600-token runs must
     * select identically (deterministic scores + top-k) and keep decoding.
     * Exact-parity gates live below 2048 where selection covers the whole
     * prefix; beyond it the official behavior IS the sparse one. */
    int dsa_ok = 1;
    {
        ds4_tokens dsa_prompt = {0};
        dsa_prompt.len = 2600;
        dsa_prompt.cap = dsa_prompt.len;
        dsa_prompt.v = malloc((size_t)dsa_prompt.len * sizeof(dsa_prompt.v[0]));
        for (int i = 0; dsa_prompt.v && i < dsa_prompt.len; i++)
            dsa_prompt.v[i] = prompt.v[i % prompt.len];
        int best[2] = {-1, -1};
        for (int pass = 0; dsa_ok && pass < 2; pass++) {
            ds4_session *s = NULL;
            dsa_ok = dsa_prompt.v &&
                     ds4_session_create(&s, engine, 4096) == 0 &&
                     ds4_session_sync(s, &dsa_prompt, err, sizeof(err)) == 0;
            if (dsa_ok) {
                best[pass] = ds4_session_argmax(s);
                dsa_ok = ds4_session_eval(s, best[pass], err, sizeof(err)) == 0;
            }
            if (s) ds4_session_free(s);
        }
        dsa_ok = dsa_ok && best[0] >= 0 && best[0] == best[1];
        printf("dots3 DSA boundary (2600 > top-2048): argmax %d/%d %s\n",
               best[0], best[1], dsa_ok ? "deterministic" : "MISMATCH");
        free(dsa_prompt.v);
    }

    /* 256K resident cache + graph beside the resident model. */
    size_t free_b = 0, free_a = 0, tot = 0;
    ds4_session *big = NULL;
    int cache_ok = cudaMemGetInfo(&free_b, &tot) == cudaSuccess &&
                   ds4_session_create(&big, engine, 262144) == 0 &&
                   cudaMemGetInfo(&free_a, &tot) == cudaSuccess;
    const uint64_t cache_delta = cache_ok && free_b >= free_a
        ? (uint64_t)(free_b - free_a) : 0;
    printf("dots3 256K resident graph + cache allocation: %.3f GiB\n",
           (double)cache_delta / 1073741824.0);
    cache_ok = cache_ok && cache_delta >= 4ull * 1024ull * MIB &&
               cache_delta <= 24ull * 1024ull * MIB;
    if (big) ds4_session_free(big);
    ds4_tokens_free(&prompt);
    free(long_prompt.v);

    ds4_engine_close(engine);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    size_t free_after = 0, total_after = 0;
    if (cudaMemGetInfo(&free_after, &total_after) != cudaSuccess) return 1;
    const uint64_t unreleased = free_before >= free_after
        ? (uint64_t)(free_before - free_after) : 0;
    const int release_ok = unreleased <= 896ull * MIB;
    printf("dots3 CUDA cleanup remainder: %" PRIu64 " bytes\n", unreleased);

    if (!metadata_ok || !no_duplicate_ok || !forward_ok || !session_ok ||
        !chunk_ok || !dsa_ok || !cache_ok || !release_ok ||
        total != total_after) {
        fprintf(stderr,
                "dots3 residency gate failed: metadata=%d no_duplicate=%d "
                "forward=%d session=%d chunk=%d dsa=%d cache256=%d release=%d\n",
                metadata_ok, no_duplicate_ok, forward_ok, session_ok,
                chunk_ok, dsa_ok, cache_ok, release_ok);
        return 1;
    }
    puts("dots3 VMM-owner worker + latent prefill/decode session: valid");
    return 0;
}
