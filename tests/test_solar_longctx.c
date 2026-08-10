/* Full-model Solar Open 2 long-context admission and retrieval gate.
 *
 * Prefills one rendered needle fixture end to end, greedily decodes the
 * reserved answer budget, and requires every expected marker string to
 * appear in the completion:
 *
 *   DS4_SOLAR_KV_FORMAT=hybrid ./tests/test_solar_longctx model.gguf \
 *     context-131072.txt 131072 256 SOLAR-D131072-BEGIN-7Q2M ...
 *
 * Reports requested/actual token counts, the context-memory estimate,
 * cold prefill time (TTFT), and the decode rate, so one run yields the
 * ladder numbers the resume plan asks to record.
 */
#include "../ds4.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static char *read_text_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size <= 0) {
        fclose(fp);
        return NULL;
    }
    char *data = (char *)malloc((size_t)size + 1u);
    if (!data || fread(data, 1, (size_t)size, fp) != (size_t)size) {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    data[size] = '\0';
    return data;
}

static uint64_t mem_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long v = 0;
        if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) {
            kb = (uint64_t)v;
            break;
        }
    }
    fclose(f);
    return kb * 1024ull;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s model.gguf rendered.txt ctx_tokens max_new "
                "[expected ...]\n",
                argv[0]);
        return 2;
    }
    const long ctx_tokens = strtol(argv[3], NULL, 10);
    const int max_new = atoi(argv[4]);
    if (ctx_tokens <= 0 || ctx_tokens > (1 << 21) ||
        max_new <= 0 || max_new > 4096) {
        fprintf(stderr, "invalid ctx/max_new: %s %s\n", argv[3], argv[4]);
        return 2;
    }

    char *rendered = read_text_file(argv[2]);
    if (!rendered) {
        fprintf(stderr, "failed to read rendered context: %s\n", argv[2]);
        return 1;
    }

    uint32_t prefill_chunk = 2048u;
    const char *chunk_env = getenv("DS4_SOLAR_PREFILL_CHUNK");
    if (chunk_env) {
        const long parsed = strtol(chunk_env, NULL, 10);
        if (parsed > 0 && parsed <= 65536) prefill_chunk = (uint32_t)parsed;
    }

    int fail = 0;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    char *completion = NULL;

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.context_size = (int)ctx_tokens + max_new + 8;
    opt.prefill_chunk = prefill_chunk;
    opt.share_session_prefill_workspace = true;

    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Solar longctx engine open failed\n");
        free(rendered);
        return 1;
    }
    ds4_tokenize_rendered_chat(engine, rendered, &prompt);
    if (prompt.len <= 0 || prompt.len > (int)ctx_tokens) {
        fprintf(stderr, "fixture tokenized to %d tokens; budget %ld\n",
                prompt.len, ctx_tokens);
        fail = 1;
        goto cleanup;
    }

    const ds4_context_memory memory =
        ds4_context_memory_estimate_with_prefill(
                DS4_BACKEND_CUDA, (uint32_t)opt.context_size,
                prefill_chunk);
    fprintf(stderr,
            "Solar longctx config: prompt=%d max_new=%d ctx=%d chunk=%u "
            "context=%.3f GiB (raw %.3f compressed %.3f scratch %.3f) "
            "mem_avail=%.2f GiB\n",
            prompt.len, max_new, opt.context_size,
            prefill_chunk,
            (double)memory.total_bytes / 1073741824.0,
            (double)memory.raw_bytes / 1073741824.0,
            (double)memory.compressed_bytes / 1073741824.0,
            (double)memory.scratch_bytes / 1073741824.0,
            (double)mem_available_bytes() / 1073741824.0);

    if (ds4_session_create(&session, engine,
                           (uint32_t)opt.context_size) != 0) {
        fprintf(stderr, "Solar longctx session creation failed\n");
        fail = 1;
        goto cleanup;
    }

    char err[256] = "";
    const double prefill_start = now_seconds();
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Solar longctx prefill failed: %s\n", err);
        fail = 1;
        goto cleanup;
    }
    const double prefill_seconds = now_seconds() - prefill_start;

    completion = (char *)calloc((size_t)max_new * 32u + 1u, 1u);
    if (!completion) {
        fail = 1;
        goto cleanup;
    }
    size_t completion_len = 0;
    int generated = 0;
    const double decode_start = now_seconds();
    for (int i = 0; i < max_new; i++) {
        const int token = ds4_session_argmax(session);
        if (token < 0) break;
        if (ds4_token_is_stop(engine, token)) break;
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        if (piece && piece_len > 0 &&
            completion_len + piece_len <= (size_t)max_new * 32u) {
            memcpy(completion + completion_len, piece, piece_len);
            completion_len += piece_len;
            completion[completion_len] = '\0';
            free(piece);
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "Solar longctx decode failed at %d: %s\n", i, err);
            fail = 1;
            break;
        }
        generated++;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) fail = 1;
    const double decode_seconds = now_seconds() - decode_start;

    printf("Solar longctx: prompt=%d generated=%d prefill=%.3fs "
           "prefill_tok_s=%.3f decode=%.3fs decode_tok_s=%.3f "
           "mem_avail_end=%.2f GiB\n",
           prompt.len, generated, prefill_seconds,
           prefill_seconds > 0.0 ? prompt.len / prefill_seconds : 0.0,
           decode_seconds,
           decode_seconds > 0.0 && generated > 0
               ? generated / decode_seconds : 0.0,
           (double)mem_available_bytes() / 1073741824.0);
    printf("Solar longctx completion: %s\n", completion);

    for (int i = 5; i < argc; i++) {
        const int found = strstr(completion, argv[i]) != NULL;
        printf("Solar longctx needle %-28s %s\n", argv[i],
               found ? "ok" : "MISSING");
        if (!found) fail = 1;
    }

cleanup:
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    free(prompt.v);
    free(rendered);
    free(completion);
    puts(fail ? "Solar longctx FAILED" : "Solar longctx passed");
    return fail ? 1 : 0;
}
