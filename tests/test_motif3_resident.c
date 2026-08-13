/* Verify the completed Motif-3 image, native graph, and 256K latent cache can
 * remain wholly CUDA-resident without retaining a second CPU copy of the GGUF
 * tensor payload.
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

static uint64_t mapping_rss_kib_for_path(const char *path) {
    FILE *fp = fopen("/proc/self/smaps", "r");
    if (!fp) return UINT64_MAX;

    const char *needle = strrchr(path, '/');
    needle = needle ? needle + 1 : path;
    char line[4096];
    int matched = 0;
    uint64_t total_kib = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long begin = 0, end = 0;
        if (sscanf(line, "%llx-%llx", &begin, &end) == 2) {
            matched = strstr(line, needle) != NULL;
            continue;
        }
        unsigned long long rss_kib = 0;
        if (matched && sscanf(line, "Rss: %llu kB", &rss_kib) == 1)
            total_kib += (uint64_t)rss_kib;
    }
    fclose(fp);
    return total_kib;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <motif3.gguf>\n", argv[0]);
        return 2;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "Motif-3 residency smoke needs CUDA device 0\n");
        return 1;
    }

    size_t free_before = 0, total = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo before engine open failed\n");
        return 1;
    }

    setenv("DS4_CUDA_COPY_MODEL_CHUNKED", "1", 1);
    unsetenv("DS4_CUDA_NO_MODEL_COPY");
    unsetenv("DS4_CUDA_DIRECT_MODEL");
    unsetenv("DS4_CUDA_WEIGHT_CACHE");
    unsetenv("DS4_CUDA_WEIGHT_PRELOAD");

    ds4_engine_options options;
    memset(&options, 0, sizeof(options));
    options.model_path = argv[1];
    options.backend = DS4_BACKEND_CUDA;
    options.n_threads = 1;
    options.context_size = 2048;
    options.placement_ctx_hint = 2048;
    options.prefill_chunk = 64;

    ds4_engine *engine = NULL;
    const int rc = ds4_engine_open(&engine, &options);
    if (rc != 0 || !engine) {
        fprintf(stderr, "Motif-3 resident engine open failed: rc=%d\n", rc);
        return 1;
    }

    size_t free_open = 0, total_open = 0;
    if (cudaMemGetInfo(&free_open, &total_open) != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo after engine open failed\n");
        ds4_engine_close(engine);
        return 1;
    }
    const uint64_t model_bytes = ds4_engine_model_bytes(engine);
    const uint64_t resident_delta = free_before >= free_open
        ? (uint64_t)(free_before - free_open) : 0;
    const int metadata_ok =
        ds4_engine_model_id(engine) == 3 &&
        ds4_engine_layer_count(engine) == 53 &&
        ds4_engine_embd_dim(engine) == 4096 &&
        model_bytes == 94162541472ull;
    const int resident_ok =
        resident_delta + 512ull * MIB >= model_bytes;
    const uint64_t source_rss_kib = mapping_rss_kib_for_path(argv[1]);
    const int source_pages_open_ok =
        source_rss_kib != UINT64_MAX && source_rss_kib <= 256ull * 1024ull;

    printf("Motif-3 resident open: model=%" PRIu64
           " bytes, CUDA delta=%" PRIu64
           " bytes, total=%zu bytes\n",
           model_bytes, resident_delta, total_open);
    if (source_rss_kib == UINT64_MAX) {
        puts("Motif-3 source GGUF mapping RSS: unavailable");
    } else {
        printf("Motif-3 source GGUF mapping RSS after CUDA copy: %" PRIu64
               " KiB (limit 262144 KiB)\n", source_rss_kib);
    }

    const int sparse_ok = ds4_engine_motif3_sparse_test(engine) == 0;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(
            engine,
            "You are a concise assistant.",
            "Reply with exactly: OK",
            DS4_THINK_NONE,
            &prompt);
    const int forward_ok =
        prompt.len > 0 && prompt.len <= 256 &&
        ds4_engine_motif3_forward_test(engine, &prompt) == 0;

    ds4_session *session = NULL;
    char session_err[256] = "";
    const int session_rc = ds4_session_create(&session, engine, 256);
    int session_ok = session_rc == 0 && session != NULL;
    int first = -1;
    if (session_ok) {
        session_ok = ds4_session_sync(
                session, &prompt, session_err, sizeof(session_err)) == 0;
    }
    if (session_ok) {
        first = ds4_session_argmax(session);
        size_t first_len = 0;
        char *first_text = ds4_token_text(engine, first, &first_len);
        session_ok = first_text && first_len == 2 &&
            memcmp(first_text, "OK", 2) == 0;
        free(first_text);
    }
    if (session_ok) {
        session_ok = ds4_session_eval(
                session, first, session_err, sizeof(session_err)) == 0 &&
            ds4_session_pos(session) == prompt.len + 1;
    }
    if (!session_ok) {
        fprintf(stderr, "Motif-3 native session failed: %s\n",
                session_err[0] ? session_err : "unexpected first token");
    }
    if (session) ds4_session_free(session);

    /* Cross the 129-token SWA window and several prefill chunks.  Compare a
     * one-chunk latent run with a 64-token run that first commits a 100-token
     * prefix and then extends it.  This covers ring identity, chunk slack,
     * prefix reuse, and non-replay decode on the actual MQ87 image. */
    ds4_tokens long_prompt = {0};
    long_prompt.len = 200;
    long_prompt.cap = long_prompt.len;
    long_prompt.v = malloc((size_t)long_prompt.len * sizeof(long_prompt.v[0]));
    if (long_prompt.v) {
        for (int i = 0; i < long_prompt.len; i++)
            long_prompt.v[i] = prompt.v[i % prompt.len];
    }
    const int vocab = ds4_engine_vocab_size(engine);
    float *one_logits = malloc((size_t)vocab * sizeof(one_logits[0]));
    float *direct_logits = malloc((size_t)vocab * sizeof(direct_logits[0]));
    float *chunk_logits = malloc((size_t)vocab * sizeof(chunk_logits[0]));
    ds4_session *one = NULL;
    ds4_session *direct = NULL;
    ds4_session *chunked = NULL;
    int chunk_ok = one_logits && direct_logits && chunk_logits && long_prompt.v;
    double one_elapsed = 0.0, chunk_elapsed = 0.0, decode_elapsed = 0.0;
    setenv("DS4_MOTIF3_PREFILL_CHUNK", "256", 1);
    if (chunk_ok) chunk_ok = ds4_session_create(&one, engine, 256) == 0;
    double t0 = monotonic_seconds();
    if (chunk_ok) chunk_ok = ds4_session_sync(
            one, &long_prompt, session_err, sizeof(session_err)) == 0;
    one_elapsed = monotonic_seconds() - t0;
    if (chunk_ok) chunk_ok =
        ds4_session_copy_logits(one, one_logits, vocab) == vocab;
    if (one) ds4_session_free(one);

    setenv("DS4_MOTIF3_PREFILL_CHUNK", "64", 1);
    if (chunk_ok) chunk_ok = ds4_session_create(&direct, engine, 256) == 0;
    if (chunk_ok) chunk_ok = ds4_session_sync(
            direct, &long_prompt, session_err, sizeof(session_err)) == 0;
    if (chunk_ok) chunk_ok =
        ds4_session_copy_logits(direct, direct_logits, vocab) == vocab;
    if (direct) ds4_session_free(direct);

    if (chunk_ok) chunk_ok = ds4_session_create(&chunked, engine, 256) == 0;
    ds4_tokens prefix = long_prompt;
    prefix.len = 128;
    t0 = monotonic_seconds();
    if (chunk_ok) chunk_ok = ds4_session_sync(
            chunked, &prefix, session_err, sizeof(session_err)) == 0;
    if (chunk_ok) chunk_ok = ds4_session_sync(
            chunked, &long_prompt, session_err, sizeof(session_err)) == 0;
    chunk_elapsed = monotonic_seconds() - t0;
    if (chunk_ok) chunk_ok =
        ds4_session_copy_logits(chunked, chunk_logits, vocab) == vocab;
    double err2 = 0.0, ref2 = 0.0, got2 = 0.0, dot = 0.0;
    double cache_err2 = 0.0, cache_ref2 = 0.0;
    double cache_got2 = 0.0, cache_dot = 0.0;
    int one_best = -1, chunk_best = -1;
    int one_top[8], chunk_top[8];
    for (int i = 0; i < 8; i++) one_top[i] = chunk_top[i] = -1;
    if (chunk_ok) {
        for (int i = 0; i < vocab; i++) {
            if (!isfinite(one_logits[i]) || !isfinite(direct_logits[i]) ||
                !isfinite(chunk_logits[i])) {
                chunk_ok = 0;
                break;
            }
            const double d = (double)chunk_logits[i] - one_logits[i];
            const double cd = (double)chunk_logits[i] - direct_logits[i];
            err2 += d * d;
            ref2 += (double)one_logits[i] * one_logits[i];
            got2 += (double)chunk_logits[i] * chunk_logits[i];
            dot += (double)chunk_logits[i] * one_logits[i];
            cache_err2 += cd * cd;
            cache_ref2 += (double)direct_logits[i] * direct_logits[i];
            cache_got2 += (double)chunk_logits[i] * chunk_logits[i];
            cache_dot += (double)chunk_logits[i] * direct_logits[i];
            if (one_best < 0 || one_logits[i] > one_logits[one_best]) one_best = i;
            if (chunk_best < 0 || chunk_logits[i] > chunk_logits[chunk_best]) chunk_best = i;
            for (int j = 0; j < 8; j++) {
                if (one_top[j] < 0 || one_logits[i] > one_logits[one_top[j]]) {
                    for (int k = 7; k > j; k--) one_top[k] = one_top[k - 1];
                    one_top[j] = i;
                    break;
                }
            }
            for (int j = 0; j < 8; j++) {
                if (chunk_top[j] < 0 || chunk_logits[i] > chunk_logits[chunk_top[j]]) {
                    for (int k = 7; k > j; k--) chunk_top[k] = chunk_top[k - 1];
                    chunk_top[j] = i;
                    break;
                }
            }
        }
    }
    const double chunk_nrmse = sqrt(err2 / fmax(ref2, 1.0e-30));
    const double chunk_cos = dot / sqrt(fmax(ref2 * got2, 1.0e-30));
    const double cache_nrmse = sqrt(
            cache_err2 / fmax(cache_ref2, 1.0e-30));
    const double cache_cos = cache_dot /
        sqrt(fmax(cache_ref2 * cache_got2, 1.0e-30));
    int top8_overlap = 0;
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++)
        if (one_top[i] == chunk_top[j]) top8_overlap++;
    chunk_ok = chunk_ok && one_best == chunk_best &&
               chunk_cos >= 0.95 && chunk_nrmse <= 0.35 &&
               top8_overlap >= 4 && cache_cos >= 0.9999 &&
               cache_nrmse <= 0.02;
    if (chunk_ok) {
        const int decode_token = ds4_session_argmax(chunked);
        t0 = monotonic_seconds();
        chunk_ok = ds4_session_eval(
                chunked, decode_token,
                session_err, sizeof(session_err)) == 0 &&
            ds4_session_pos(chunked) == long_prompt.len + 1;
        decode_elapsed = monotonic_seconds() - t0;
    }
    printf("Motif-3 latent chunk/ring parity: first=%d/%d batch_cos=%.8f "
           "batch_nrmse=%.6g top8=%d/8 cache_cos=%.8f cache_nrmse=%.6g; "
           "one-chunk %.2f tok/s, prefix-extend %.2f tok/s, "
           "decode %.3f tok/s\n",
           chunk_best, one_best, chunk_cos, chunk_nrmse, top8_overlap,
           cache_cos, cache_nrmse,
           one_elapsed > 0.0 ? (double)long_prompt.len / one_elapsed : 0.0,
           chunk_elapsed > 0.0 ? (double)long_prompt.len / chunk_elapsed : 0.0,
           decode_elapsed > 0.0 ? 1.0 / decode_elapsed : 0.0);
    if (!chunk_ok) {
        fprintf(stderr, "Motif-3 chunk/ring session failed: %s\n",
                session_err[0] ? session_err : "logit parity mismatch");
    }
    if (chunked) ds4_session_free(chunked);
    unsetenv("DS4_MOTIF3_PREFILL_CHUNK");
    free(one_logits);
    free(direct_logits);
    free(chunk_logits);
    free(long_prompt.v);

    /* Allocate the exact native context cache while the 87.70 GiB model is
     * resident.  This is a physical CUDA measurement, not a GGUF projection. */
    size_t free_256_before = 0, free_256_after = 0, total_256 = 0;
    ds4_session *session_256k = NULL;
    int cache_256k_ok =
        cudaMemGetInfo(&free_256_before, &total_256) == cudaSuccess &&
        ds4_session_create(&session_256k, engine, 262144) == 0 &&
        cudaMemGetInfo(&free_256_after, &total_256) == cudaSuccess;
    const uint64_t cache_256k_delta =
        cache_256k_ok && free_256_before >= free_256_after
            ? (uint64_t)(free_256_before - free_256_after) : 0;
    cache_256k_ok = cache_256k_ok &&
        cache_256k_delta >= 3800ull * MIB &&
        cache_256k_delta <= 5ull * 1024ull * MIB;
    printf("Motif-3 256K resident session allocation: %.3f GiB "
           "(%" PRIu64 " bytes; model remains resident; no offload)\n",
           (double)cache_256k_delta / 1073741824.0,
           cache_256k_delta);
    if (session_256k) ds4_session_free(session_256k);
    ds4_tokens_free(&prompt);

    const uint64_t source_rss_after_kib = mapping_rss_kib_for_path(argv[1]);
    const int source_pages_after_ok =
        source_rss_after_kib != UINT64_MAX &&
        source_rss_after_kib <= 256ull * 1024ull;
    if (source_rss_after_kib == UINT64_MAX) {
        puts("Motif-3 source GGUF mapping RSS after inference: unavailable");
    } else {
        printf("Motif-3 source GGUF mapping RSS after inference: %" PRIu64
               " KiB (limit 262144 KiB)\n", source_rss_after_kib);
    }

    ds4_engine_close(engine);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize after engine close failed\n");
        return 1;
    }
    size_t free_after = 0, total_after = 0;
    if (cudaMemGetInfo(&free_after, &total_after) != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo after engine close failed\n");
        return 1;
    }
    const uint64_t unreleased = free_before >= free_after
        ? (uint64_t)(free_before - free_after) : 0;
    const int release_ok = unreleased <= 512ull * MIB;

    if (!metadata_ok || !resident_ok || !source_pages_open_ok ||
        !source_pages_after_ok ||
        !sparse_ok || !forward_ok ||
        !session_ok || !chunk_ok || !cache_256k_ok ||
        !release_ok || total != total_after) {
        fprintf(stderr,
                "Motif-3 residency gate failed: metadata=%d resident=%d "
                "source_open=%d source_after=%d sparse=%d forward=%d session=%d "
                "chunk=%d cache256=%d "
                "release=%d "
                "unreleased=%" PRIu64 "\n",
                metadata_ok, resident_ok, source_pages_open_ok,
                source_pages_after_ok, sparse_ok, forward_ok, session_ok,
                chunk_ok, cache_256k_ok,
                release_ok, unreleased);
        return 1;
    }
    puts("Motif-3 H200 resident loader + latent prefill/decode session: valid "
         "(no SSD streaming, no CPU offload; chunked prefix reuse)");
    return 0;
}
