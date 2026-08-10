/* Full-model Solar Open 2 GQA KV format comparison.
 *
 * One resident engine creates and releases one session at a time so BF16,
 * FP8, hybrid, and FP4 compare the same weights and rendered token prefix
 * without multiplying GB10 context allocations. KDA stays on its generic
 * path to isolate KV-format effects.
 *
 * Set DS4_SOLAR_KV_PROFILE_FORMAT to bf16, fp8, hybrid, or fp4 while running
 * under an Nsight Systems cudaProfilerApi capture to trace just that format.
 */
#include "../ds4.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *env_name;
    const char *label;
} format_case;

typedef struct {
    int first;
    int second;
    float first_value;
    float second_value;
} top2_result;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static char *read_text_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    const long size = ftell(fp);
    if (size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *text = (char *)malloc((size_t)size + 1u);
    if (!text) {
        fclose(fp);
        return NULL;
    }
    if (fread(text, 1, (size_t)size, fp) != (size_t)size) {
        free(text);
        fclose(fp);
        return NULL;
    }
    text[size] = '\0';
    fclose(fp);
    return text;
}

static int parse_tokens(const char *value) {
    errno = 0;
    char *end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (errno || !end || *end != '\0' || parsed < 1 || parsed > 65536) {
        return 0;
    }
    return (int)parsed;
}

static int valid_profile_format(const char *value) {
    return !value || !value[0] || strcmp(value, "bf16") == 0 ||
           strcmp(value, "fp8") == 0 || strcmp(value, "hybrid") == 0 ||
           strcmp(value, "fp4") == 0;
}

static top2_result top2(const float *logits, int n) {
    top2_result result = {-1, -1, -INFINITY, -INFINITY};
    for (int i = 0; i < n; i++) {
        if (!isfinite(logits[i])) continue;
        if (logits[i] > result.first_value) {
            result.second = result.first;
            result.second_value = result.first_value;
            result.first = i;
            result.first_value = logits[i];
        } else if (logits[i] > result.second_value) {
            result.second = i;
            result.second_value = logits[i];
        }
    }
    return result;
}

static void report_top2(const char *format, const char *stage,
                        const float *logits, int n) {
    const top2_result result = top2(logits, n);
    printf("Solar KV %s %s top1=%d %.9g top2=%d %.9g margin=%.9g\n",
           format, stage, result.first, result.first_value,
           result.second, result.second_value,
           (double)result.first_value - result.second_value);
}

static int report_diff(const char *format, const char *stage,
                       const float *got, const float *reference, int n) {
    double max_abs = 0.0;
    double err2 = 0.0;
    double ref2 = 0.0;
    double got2 = 0.0;
    double dot = 0.0;
    int max_index = -1;
    int nonfinite = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(got[i]) || !isfinite(reference[i])) {
            nonfinite = 1;
            continue;
        }
        const double g = got[i];
        const double r = reference[i];
        const double delta = g - r;
        if (fabs(delta) > max_abs) {
            max_abs = fabs(delta);
            max_index = i;
        }
        err2 += delta * delta;
        ref2 += r * r;
        got2 += g * g;
        dot += g * r;
    }
    const double rel_rms = ref2 > 0.0 ? sqrt(err2 / ref2) : sqrt(err2);
    const double cosine = ref2 > 0.0 && got2 > 0.0
        ? dot / sqrt(ref2 * got2) : 1.0;
    printf("Solar KV %s %s vs BF16 max_abs=%.9g index=%d "
           "rel_rms=%.9g 1-cos=%.9g nonfinite=%d\n",
           format, stage, max_abs, max_index, rel_rms, 1.0 - cosine,
           nonfinite);
    return nonfinite;
}

static int synchronize_cuda(const char *where) {
    const cudaError_t rc = cudaDeviceSynchronize();
    if (rc == cudaSuccess) return 0;
    fprintf(stderr, "%s: cudaDeviceSynchronize failed: %s\n",
            where, cudaGetErrorString(rc));
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.gguf rendered.txt tokens\n", argv[0]);
        return 2;
    }
    const int tokens = parse_tokens(argv[3]);
    if (tokens == 0) {
        fprintf(stderr, "invalid token count: %s\n", argv[3]);
        return 2;
    }
    char *rendered = read_text_file(argv[2]);
    if (!rendered) {
        fprintf(stderr, "failed to read rendered context: %s\n", argv[2]);
        return 1;
    }
    const char *profile_format = getenv("DS4_SOLAR_KV_PROFILE_FORMAT");
    if (!valid_profile_format(profile_format)) {
        fprintf(stderr,
                "invalid DS4_SOLAR_KV_PROFILE_FORMAT=%s "
                "(expected bf16, fp8, hybrid, or fp4)\n",
                profile_format);
        free(rendered);
        return 2;
    }
    const char *hmma_value = getenv("DS4_SOLAR_KV_PREFILL_HMMA");
    const int hmma_enabled =
        hmma_value && hmma_value[0] && hmma_value[0] != '0';
    printf("Solar KV full-model comparison: tokens=%d prefill_hmma=%d "
           "profile=%s\n",
           tokens, hmma_enabled,
           profile_format && profile_format[0] ? profile_format : "none");
    fflush(stdout);

    /* KDA rides the release path (chunked prefill, recurrent decode) so the
     * KV formats are compared on the configuration a server would run;
     * export DS4_SOLAR_KDA_STATE_PARTS before launch to pin a sequence
     * variant instead. */
    setenv("DS4_SOLAR_KV_FORMAT", "bf16", 1);

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.context_size = (uint32_t)tokens + 8u;
    opt.prefill_chunk = tokens < 2048 ? (uint32_t)tokens : 2048u;
    opt.share_session_prefill_workspace = true;

    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens all_tokens = {0};
    float *bf16_prefill = NULL;
    float *bf16_decode = NULL;
    float *current_prefill = NULL;
    float *current_decode = NULL;
    int fail = 0;
    int capture_active = 0;

    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Solar KV format engine open failed\n");
        fail = 1;
        goto cleanup;
    }
    ds4_tokenize_rendered_chat(engine, rendered, &all_tokens);
    if (all_tokens.len < tokens) {
        fprintf(stderr, "rendered fixture has only %d tokens; need %d\n",
                all_tokens.len, tokens);
        fail = 1;
        goto cleanup;
    }
    ds4_tokens prompt = {
        .v = all_tokens.v,
        .len = tokens,
        .cap = all_tokens.cap,
    };
    const int n_vocab = ds4_engine_vocab_size(engine);
    if (n_vocab <= 0) {
        fprintf(stderr, "Solar KV format model has invalid vocab size %d\n",
                n_vocab);
        fail = 1;
        goto cleanup;
    }
    bf16_prefill = (float *)malloc((size_t)n_vocab * sizeof(float));
    bf16_decode = (float *)malloc((size_t)n_vocab * sizeof(float));
    current_prefill = (float *)malloc((size_t)n_vocab * sizeof(float));
    current_decode = (float *)malloc((size_t)n_vocab * sizeof(float));
    if (!bf16_prefill || !bf16_decode ||
        !current_prefill || !current_decode) {
        fprintf(stderr, "Solar KV format logits allocation failed\n");
        fail = 1;
        goto cleanup;
    }

    const format_case formats[] = {
        {"bf16", "BF16"},
        {"fp8", "FP8"},
        {"hybrid", "HYBRID"},
        {"fp4", "FP4"},
    };
    int reference_token = -1;
    int reference_decode_token = -1;
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
        const format_case *format = &formats[i];
        setenv("DS4_SOLAR_KV_FORMAT", format->env_name, 1);
        if (ds4_session_create(&session, engine, opt.context_size) != 0) {
            fprintf(stderr, "Solar KV %s session creation failed\n",
                    format->label);
            fail = 1;
            break;
        }

        const ds4_context_memory memory =
            ds4_context_memory_estimate_with_prefill(
                    DS4_BACKEND_CUDA, opt.context_size, opt.prefill_chunk);
        const int capture = profile_format &&
            strcmp(profile_format, format->env_name) == 0;
        if (capture) {
            const cudaError_t rc = cudaProfilerStart();
            if (rc != cudaSuccess) {
                fprintf(stderr, "cudaProfilerStart failed: %s\n",
                        cudaGetErrorString(rc));
                fail = 1;
                break;
            }
            capture_active = 1;
        }

        char err[256] = "";
        const double prefill_start = now_seconds();
        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
            synchronize_cuda("Solar KV prefill") != 0) {
            fprintf(stderr, "Solar KV %s prefill failed: %s\n",
                    format->label, err);
            fail = 1;
            break;
        }
        const double prefill_seconds = now_seconds() - prefill_start;
        if (ds4_session_copy_logits(
                    session, current_prefill, n_vocab) != n_vocab) {
            fprintf(stderr, "Solar KV %s prefill logits copy failed\n",
                    format->label);
            fail = 1;
            break;
        }
        const top2_result prefill_top = top2(current_prefill, n_vocab);
        if (prefill_top.first < 0) {
            fprintf(stderr, "Solar KV %s prefill has no finite logits\n",
                    format->label);
            fail = 1;
            break;
        }
        if (reference_token < 0) reference_token = prefill_top.first;

        const double decode_start = now_seconds();
        if (ds4_session_eval(
                    session, reference_token, err, sizeof(err)) != 0 ||
            synchronize_cuda("Solar KV decode") != 0) {
            fprintf(stderr, "Solar KV %s decode failed: %s\n",
                    format->label, err);
            fail = 1;
            break;
        }
        const double decode_seconds = now_seconds() - decode_start;
        if (ds4_session_copy_logits(
                    session, current_decode, n_vocab) != n_vocab) {
            fprintf(stderr, "Solar KV %s decode logits copy failed\n",
                    format->label);
            fail = 1;
            break;
        }
        const top2_result decode_top = top2(current_decode, n_vocab);
        if (decode_top.first < 0) {
            fprintf(stderr, "Solar KV %s decode has no finite logits\n",
                    format->label);
            fail = 1;
            break;
        }
        if (capture) {
            if (synchronize_cuda("Solar KV profiler stop") != 0) {
                fail = 1;
                break;
            }
            const cudaError_t rc = cudaProfilerStop();
            capture_active = 0;
            if (rc != cudaSuccess) {
                fprintf(stderr, "cudaProfilerStop failed: %s\n",
                        cudaGetErrorString(rc));
                fail = 1;
                break;
            }
        }
        if (i == 0u) {
            reference_decode_token = decode_top.first;
        } else if (prefill_top.first != reference_token ||
                   decode_top.first != reference_decode_token) {
            fprintf(stderr,
                    "Solar KV %s top-1 mismatch: prefill=%d expected=%d "
                    "decode=%d expected=%d\n",
                    format->label, prefill_top.first, reference_token,
                    decode_top.first, reference_decode_token);
            fail = 1;
            break;
        }

        printf("Solar KV format=%s tokens=%d context=%.3fGiB raw_kv=%.3fGiB "
               "state=%.3fGiB scratch=%.3fGiB prefill=%.6fs tok/s=%.6f "
               "decode=%.6fs continuation=%d\n",
               format->label, tokens,
               (double)memory.total_bytes / 1073741824.0,
               (double)memory.raw_bytes / 1073741824.0,
               (double)memory.compressed_bytes / 1073741824.0,
               (double)memory.scratch_bytes / 1073741824.0,
               prefill_seconds, tokens / prefill_seconds,
               decode_seconds, reference_token);
        report_top2(format->label, "prefill", current_prefill, n_vocab);
        report_top2(format->label, "decode", current_decode, n_vocab);
        if (i == 0u) {
            memcpy(bf16_prefill, current_prefill,
                   (size_t)n_vocab * sizeof(float));
            memcpy(bf16_decode, current_decode,
                   (size_t)n_vocab * sizeof(float));
        } else {
            fail |= report_diff(format->label, "prefill", current_prefill,
                                bf16_prefill, n_vocab);
            fail |= report_diff(format->label, "decode", current_decode,
                                bf16_decode, n_vocab);
        }
        fflush(stdout);
        ds4_session_free(session);
        session = NULL;
        if (fail) break;
    }

cleanup:
    if (capture_active) {
        cudaDeviceSynchronize();
        cudaProfilerStop();
    }
    ds4_session_free(session);
    free(bf16_prefill);
    free(bf16_decode);
    free(current_prefill);
    free(current_decode);
    ds4_tokens_free(&all_tokens);
    ds4_engine_close(engine);
    free(rendered);
    unsetenv("DS4_SOLAR_KV_FORMAT");
    unsetenv("DS4_SOLAR_KDA_STATE_PARTS");
    printf("Solar KV full-model format comparison: %s\n",
           fail ? "FAILED" : "passed");
    return fail ? 1 : 0;
}
