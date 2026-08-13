/* End-to-end Motif-3 latent-KV long-context gate on the immutable MQ87 GGUF. */
#include "../ds4.h"

#include <cuda_runtime.h>
#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int next_report;
    double started;
} progress_state;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void progress_cb(void *ud, const char *event, int current, int total) {
    progress_state *state = ud;
    if (strcmp(event, "prefill_chunk") != 0) return;
    if (current < state->next_report && current != total) return;
    const double elapsed = now_seconds() - state->started;
    fprintf(stderr,
            "motif3-long: prefill %d/%d (%.1f%%), %.2f tok/s, %.1f s\n",
            current, total, total ? 100.0 * (double)current / total : 0.0,
            elapsed > 0.0 ? (double)current / elapsed : 0.0, elapsed);
    state->next_report = current + 8192;
}

static int read_npy_i32(const char *path, int **out, int *count) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 0; }
    unsigned char prefix[12] = {0};
    if (fread(prefix, 1, 10, fp) != 10 ||
        memcmp(prefix, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "invalid NumPy file: %s\n", path);
        fclose(fp);
        return 0;
    }
    uint32_t header_len = (uint32_t)prefix[8] | ((uint32_t)prefix[9] << 8u);
    if (prefix[6] >= 2u) {
        if (fread(prefix + 10, 1, 2, fp) != 2) { fclose(fp); return 0; }
        header_len |= (uint32_t)prefix[10] << 16u;
        header_len |= (uint32_t)prefix[11] << 24u;
    }
    if (header_len == 0 || header_len > 65536u) {
        fclose(fp); return 0;
    }
    char *header = calloc((size_t)header_len + 1u, 1u);
    if (!header || fread(header, 1, header_len, fp) != header_len ||
        (!strstr(header, "'<i4'") && !strstr(header, "'|i4'")) ||
        strstr(header, "True")) {
        fprintf(stderr, "unsupported NumPy token array: %s\n", path);
        free(header);
        fclose(fp);
        return 0;
    }
    char *shape = strstr(header, "shape");
    shape = shape ? strchr(shape, '(') : NULL;
    char *end = NULL;
    const unsigned long long n = shape ? strtoull(shape + 1, &end, 10) : 0;
    free(header);
    if (!shape || end == shape + 1 || n == 0 || n > 262144u) {
        fclose(fp); return 0;
    }
    int *tokens = malloc((size_t)n * sizeof(tokens[0]));
    if (!tokens || fread(tokens, sizeof(tokens[0]), (size_t)n, fp) != n) {
        fprintf(stderr, "short NumPy token array: %s\n", path);
        free(tokens);
        fclose(fp);
        return 0;
    }
    fclose(fp);
    *out = tokens;
    *count = (int)n;
    return 1;
}

static int append_bytes(char **out, size_t *len, size_t *cap,
                        const char *text, size_t n) {
    if (*len + n + 1u > *cap) {
        size_t wanted = *cap ? *cap : 256u;
        while (wanted < *len + n + 1u) wanted *= 2u;
        char *grown = realloc(*out, wanted);
        if (!grown) return 0;
        *out = grown;
        *cap = wanted;
    }
    memcpy(*out + *len, text, n);
    *len += n;
    (*out)[*len] = '\0';
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf context-N.tokens.npy\n", argv[0]);
        return 2;
    }
    int *token_data = NULL;
    int source_tokens = 0;
    if (!read_npy_i32(argv[2], &token_data, &source_tokens)) return 1;
    int fixture_tag = source_tokens;
    const char *tag = strstr(argv[2], "context-");
    if (tag) {
        char *tag_end = NULL;
        const long parsed = strtol(tag + strlen("context-"), &tag_end, 10);
        if (tag_end != tag + strlen("context-") && parsed > 0 &&
            parsed <= 262144) fixture_tag = (int)parsed;
    }

    const int generation_budget = 64;
    const int native_ctx = 262144;
    int prompt_tokens = source_tokens;
    int ctx = prompt_tokens + generation_budget;
    if (ctx > native_ctx) {
        /* Every generated fixture has a 20-token question tail.  Remove only
         * filler immediately before it, preserving all records and the exact
         * question while reserving native-context decode room. */
        const int question_tail = 20;
        const int remove = ctx - native_ctx;
        if (remove <= 0 || prompt_tokens <= remove + question_tail) {
            free(token_data); return 1;
        }
        memmove(token_data + prompt_tokens - question_tail - remove,
                token_data + prompt_tokens - question_tail,
                (size_t)question_tail * sizeof(token_data[0]));
        prompt_tokens -= remove;
        ctx = native_ctx;
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
    options.context_size = ctx;
    options.placement_ctx_hint = ctx;
    options.prefill_chunk = 256;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0 || !engine) {
        fprintf(stderr, "motif3-long: engine open failed\n");
        free(token_data);
        return 1;
    }
    ds4_tokens prompt = {
        .v = token_data,
        .len = prompt_tokens,
        .cap = source_tokens,
    };
    int short_reference_ok = 1;
    if (prompt_tokens <= 256) {
        short_reference_ok =
            ds4_engine_motif3_forward_test(engine, &prompt) == 0;
    }
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx) != 0 || !session) {
        fprintf(stderr, "motif3-long: session create failed\n");
        ds4_engine_close(engine);
        free(token_data);
        return 1;
    }
    progress_state progress = {.next_report = 8192, .started = now_seconds()};
    ds4_session_set_progress(session, progress_cb, &progress);
    char err[256] = "";
    const double prefill_t0 = now_seconds();
    const int sync_rc = ds4_session_sync(session, &prompt, err, sizeof(err));
    const double prefill_elapsed = now_seconds() - prefill_t0;
    if (sync_rc != 0) {
        fprintf(stderr, "motif3-long: prefill failed: %s\n", err);
        ds4_session_free(session);
        ds4_engine_close(engine);
        free(token_data);
        return 1;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    int finite = logits && ds4_session_copy_logits(session, logits, vocab) == vocab;
    for (int i = 0; finite && i < vocab; i++) finite = isfinite(logits[i]);

    char *output = NULL;
    size_t output_len = 0, output_cap = 0;
    int generated = 0, evals = 0;
    const double decode_t0 = now_seconds();
    for (int step = 0; finite && step < generation_budget; step++) {
        const int token = ds4_session_argmax(session);
        if (token < 0) { finite = 0; break; }
        if (ds4_token_is_stop(engine, token)) break;
        size_t text_len = 0;
        char *text = ds4_token_text(engine, token, &text_len);
        if (!text || !append_bytes(
                &output, &output_len, &output_cap, text, text_len)) {
            free(text); finite = 0; break;
        }
        free(text);
        generated++;
        if (ds4_session_pos(session) >= ctx ||
            ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            if (ds4_session_pos(session) < ctx)
                fprintf(stderr, "motif3-long: decode failed: %s\n", err);
            break;
        }
        evals++;
    }
    const double decode_elapsed = now_seconds() - decode_t0;
    if (!output) output = strdup("");

    char code1[64], code2[64], code3[64];
    snprintf(code1, sizeof(code1), "MOTIF-%d-BEGIN-7Q2K", fixture_tag);
    snprintf(code2, sizeof(code2), "MOTIF-%d-MIDDLE-9R4V", fixture_tag);
    snprintf(code3, sizeof(code3), "MOTIF-%d-END-3X8P", fixture_tag);
    const char *first = output;
    while (*first && isspace((unsigned char)*first)) first++;
    const int retrieval_ok = *first == '[' && strstr(output, code1) &&
                             strstr(output, code2) && strstr(output, code3);
    printf("Motif-3 long gate: source=%d prompt=%d ctx=%d prefill=%.2f tok/s "
           "decode=%.3f tok/s generated=%d evals=%d finite=%d retrieval=%d\n",
           source_tokens, prompt_tokens, ctx,
           prefill_elapsed > 0.0 ? prompt_tokens / prefill_elapsed : 0.0,
           decode_elapsed > 0.0 ? generated / decode_elapsed : 0.0,
           generated, evals, finite, retrieval_ok);
    printf("Motif-3 long output: %s\n", output);

    const int ok = short_reference_ok && finite && generated > 0 &&
                   evals > 0 && retrieval_ok;
    free(output);
    free(logits);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(token_data);
    return ok ? 0 : 1;
}
