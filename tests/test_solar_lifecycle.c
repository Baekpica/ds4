/* Full-model Solar Open 2 KDA lifecycle and staged-context regression.
 *
 * Normal mode keeps one model resident while extending a deterministic
 * rendered prompt through 2K, 8K, 32K, and 64K.  Profile mode captures only
 * one prefill plus one decode so Nsight traces stay focused:
 *
 *   DS4_SOLAR_LIFECYCLE_PROFILE=1 DS4_TEST_CUDA_PROFILER_CAPTURE=1 \
 *     ./tests/test_solar_lifecycle model.gguf rendered.txt 2048
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
    int calls;
    int trigger;
} cancel_state;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static bool cancel_after(void *ud) {
    cancel_state *state = (cancel_state *)ud;
    state->calls++;
    return state->calls >= state->trigger;
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

static int copy_logits(ds4_session *session, float *out, int n_vocab,
                       const char *where) {
    if (ds4_session_copy_logits(session, out, n_vocab) != n_vocab) {
        fprintf(stderr, "%s: logits copy failed\n", where);
        return 1;
    }
    return 0;
}

static double max_abs_diff(const float *a, const float *b, int n,
                           int *nonfinite) {
    double worst = 0.0;
    *nonfinite = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            *nonfinite = 1;
            continue;
        }
        const double diff = fabs((double)a[i] - (double)b[i]);
        if (diff > worst) worst = diff;
    }
    return worst;
}

static int synchronize_cuda(const char *where) {
    const cudaError_t rc = cudaDeviceSynchronize();
    if (rc == cudaSuccess) return 0;
    fprintf(stderr, "%s: cudaDeviceSynchronize failed: %s\n",
            where, cudaGetErrorString(rc));
    return 1;
}

static int parse_max_tokens(const char *value) {
    errno = 0;
    char *end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (errno || !end || *end != '\0' ||
        (parsed != 2048 && parsed != 8192 &&
         parsed != 32768 && parsed != 65536)) {
        return 0;
    }
    return (int)parsed;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s model.gguf rendered-context.txt "
                "{2048|8192|32768|65536}\n",
                argv[0]);
        return 2;
    }
    const int max_tokens = parse_max_tokens(argv[3]);
    if (max_tokens == 0) {
        fprintf(stderr, "invalid max token stage: %s\n", argv[3]);
        return 2;
    }
    const char *profile_env = getenv("DS4_SOLAR_LIFECYCLE_PROFILE");
    const int profile_only =
        profile_env && profile_env[0] && strcmp(profile_env, "0") != 0;
    uint32_t prefill_chunk = 2048u;
    const char *chunk_env = getenv("DS4_SOLAR_PREFILL_CHUNK");
    if (chunk_env) {
        const long parsed = strtol(chunk_env, NULL, 10);
        if (parsed > 0 && parsed <= 65536) prefill_chunk = (uint32_t)parsed;
    }

    char *rendered = read_text_file(argv[2]);
    if (!rendered) {
        fprintf(stderr, "failed to read rendered context: %s\n", argv[2]);
        return 1;
    }

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.context_size = (uint32_t)max_tokens + 8u;
    opt.prefill_chunk = prefill_chunk;
    opt.share_session_prefill_workspace = true;

    int fail = 0;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens all_tokens = {0};
    float *baseline = NULL;
    float *cold = NULL;
    float *warm = NULL;
    float *replay = NULL;

    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Solar lifecycle engine open failed\n");
        fail = 1;
        goto cleanup;
    }
    ds4_tokenize_rendered_chat(engine, rendered, &all_tokens);
    if (all_tokens.len < max_tokens) {
        fprintf(stderr, "rendered fixture has only %d tokens; need %d\n",
                all_tokens.len, max_tokens);
        fail = 1;
        goto cleanup;
    }
    if (ds4_session_create(&session, engine, opt.context_size) != 0) {
        fprintf(stderr, "Solar lifecycle session creation failed\n");
        fail = 1;
        goto cleanup;
    }

    const int n_vocab = ds4_engine_vocab_size(engine);
    baseline = (float *)calloc((size_t)n_vocab, sizeof(*baseline));
    cold = (float *)calloc((size_t)n_vocab, sizeof(*cold));
    warm = (float *)calloc((size_t)n_vocab, sizeof(*warm));
    replay = (float *)calloc((size_t)n_vocab, sizeof(*replay));
    if (!baseline || !cold || !warm || !replay) {
        fprintf(stderr, "Solar lifecycle logits allocation failed\n");
        fail = 1;
        goto cleanup;
    }

    const ds4_context_memory memory =
        ds4_context_memory_estimate_with_prefill(
                DS4_BACKEND_CUDA, opt.context_size, prefill_chunk);
    fprintf(stderr,
            "Solar lifecycle config: fixture=%d max=%d ctx=%u chunk=%u "
            "prefill_cap=%u context=%.3f GiB (raw %.3f compressed %.3f "
            "scratch %.3f) profile=%d\n",
            all_tokens.len, max_tokens, opt.context_size, prefill_chunk,
            memory.prefill_cap,
            (double)memory.total_bytes / 1073741824.0,
            (double)memory.raw_bytes / 1073741824.0,
            (double)memory.compressed_bytes / 1073741824.0,
            (double)memory.scratch_bytes / 1073741824.0,
            profile_only);

    if (profile_only) {
        ds4_tokens prompt = {
            .v = all_tokens.v,
            .len = max_tokens,
            .cap = all_tokens.cap,
        };
        char err[256] = "";
        const char *capture_env = getenv("DS4_TEST_CUDA_PROFILER_CAPTURE");
        const int capture =
            capture_env && capture_env[0] && strcmp(capture_env, "0") != 0;
        int capture_active = 0;
        if (capture) {
            const cudaError_t rc = cudaProfilerStart();
            if (rc != cudaSuccess) {
                fprintf(stderr, "cudaProfilerStart failed: %s\n",
                        cudaGetErrorString(rc));
                fail = 1;
                goto cleanup;
            }
            capture_active = 1;
        }
        const double prefill_start = now_seconds();
        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
            synchronize_cuda("profile prefill") != 0) {
            fprintf(stderr, "Solar profile prefill failed: %s\n", err);
            fail = 1;
        }
        const double prefill_seconds = now_seconds() - prefill_start;
        const int first = fail ? -1 : ds4_session_argmax(session);
        const double decode_start = now_seconds();
        if (!fail &&
            (ds4_session_eval(session, first, err, sizeof(err)) != 0 ||
             synchronize_cuda("profile decode") != 0)) {
            fprintf(stderr, "Solar profile decode failed: %s\n", err);
            fail = 1;
        }
        const double decode_seconds = now_seconds() - decode_start;
        printf("Solar profile: tokens=%d prefill=%.6fs tok/s=%.6f "
               "decode=%.6fs first=%d second=%d result=%s\n",
               max_tokens, prefill_seconds,
               prefill_seconds > 0.0 ? max_tokens / prefill_seconds : 0.0,
               decode_seconds, first,
               fail ? -1 : ds4_session_argmax(session),
               fail ? "FAILED" : "passed");
        fflush(stdout);
        if (capture_active) {
            const cudaError_t rc = cudaProfilerStop();
            if (rc != cudaSuccess) {
                fprintf(stderr, "cudaProfilerStop failed: %s\n",
                        cudaGetErrorString(rc));
                fail = 1;
            }
        }
        goto cleanup;
    }

    const int stages[] = {2048, 8192, 32768, 65536};
    const int n_stages = (int)(sizeof(stages) / sizeof(stages[0]));
    int previous_stage = 0;
    for (int stage_index = 0; stage_index < n_stages; stage_index++) {
        const int stage = stages[stage_index];
        if (stage > max_tokens) break;
        ds4_tokens prompt = {
            .v = all_tokens.v,
            .len = stage,
            .cap = all_tokens.cap,
        };
        char err[256] = "";
        const int common = ds4_session_common_prefix(session, &prompt);
        if (common != previous_stage) {
            fprintf(stderr,
                    "stage %d common prefix mismatch: got %d expected %d\n",
                    stage, common, previous_stage);
            fail = 1;
            break;
        }

        const double sync_start = now_seconds();
        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
            synchronize_cuda("lifecycle prefill") != 0) {
            fprintf(stderr, "stage %d prefill failed: %s\n", stage, err);
            fail = 1;
            break;
        }
        const double sync_seconds = now_seconds() - sync_start;
        if (ds4_session_pos(session) != stage ||
            copy_logits(session, baseline, n_vocab, "stage baseline")) {
            fprintf(stderr, "stage %d restored the wrong position/logits\n",
                    stage);
            fail = 1;
            break;
        }

        ds4_session_snapshot snapshot = {0};
        ds4_session_snapshot restored = {0};
        const double snapshot_start = now_seconds();
        if (ds4_session_save_snapshot(
                    session, &snapshot, err, sizeof(err)) != 0) {
            fprintf(stderr, "stage %d snapshot save failed: %s\n", stage, err);
            fail = 1;
            break;
        }
        const double snapshot_seconds = now_seconds() - snapshot_start;
        const int first = ds4_session_argmax(session);
        if (ds4_session_eval(session, first, err, sizeof(err)) != 0 ||
            copy_logits(session, cold, n_vocab, "cold continuation")) {
            fprintf(stderr, "stage %d cold continuation failed: %s\n",
                    stage, err);
            fail = 1;
            ds4_session_snapshot_free(&snapshot);
            break;
        }
        const int second_cold = ds4_session_argmax(session);

        if (ds4_session_load_snapshot(
                    session, &snapshot, err, sizeof(err)) != 0 ||
            copy_logits(session, replay, n_vocab, "restored prefill")) {
            fprintf(stderr, "stage %d snapshot restore failed: %s\n",
                    stage, err);
            fail = 1;
        }
        int nonfinite = 0;
        const double restored_prefill_diff =
            max_abs_diff(baseline, replay, n_vocab, &nonfinite);
        if (!fail && (nonfinite || restored_prefill_diff != 0.0 ||
                      ds4_session_pos(session) != stage ||
                      ds4_session_argmax(session) != first)) {
            fprintf(stderr, "stage %d restored prefill mismatch %.9g\n",
                    stage, restored_prefill_diff);
            fail = 1;
        }
        if (!fail && ds4_session_save_snapshot(
                    session, &restored, err, sizeof(err)) != 0) {
            fprintf(stderr, "stage %d restored snapshot save failed: %s\n",
                    stage, err);
            fail = 1;
        }
        const int snapshot_equal =
            !fail && restored.len == snapshot.len &&
            memcmp(restored.ptr, snapshot.ptr, (size_t)snapshot.len) == 0;
        if (!fail && !snapshot_equal) {
            fprintf(stderr, "stage %d snapshot bytes differ\n", stage);
            fail = 1;
        }
        ds4_session_snapshot_free(&restored);

        if (!fail &&
            (ds4_session_eval(session, first, err, sizeof(err)) != 0 ||
             copy_logits(session, warm, n_vocab, "warm continuation"))) {
            fprintf(stderr, "stage %d warm continuation failed: %s\n",
                    stage, err);
            fail = 1;
        }
        const int second_warm = fail ? -1 : ds4_session_argmax(session);
        const double cold_warm_diff =
            fail ? 0.0 : max_abs_diff(cold, warm, n_vocab, &nonfinite);
        if (!fail && (nonfinite || second_warm != second_cold)) {
            fprintf(stderr,
                    "stage %d cold/warm argmax mismatch %d/%d diff %.9g\n",
                    stage, second_cold, second_warm, cold_warm_diff);
            fail = 1;
        }
        if (!fail &&
            (ds4_session_load_snapshot(
                     session, &snapshot, err, sizeof(err)) != 0 ||
             ds4_session_eval(session, first, err, sizeof(err)) != 0 ||
             copy_logits(session, replay, n_vocab, "replayed continuation"))) {
            fprintf(stderr, "stage %d replay continuation failed: %s\n",
                    stage, err);
            fail = 1;
        }
        const double warm_replay_diff =
            fail ? 0.0 : max_abs_diff(warm, replay, n_vocab, &nonfinite);
        if (!fail && (nonfinite || warm_replay_diff > 1.0e-5 ||
                      ds4_session_argmax(session) != second_warm)) {
            fprintf(stderr, "stage %d warm replay mismatch %.9g\n",
                    stage, warm_replay_diff);
            fail = 1;
        }
        if (!fail && ds4_session_load_snapshot(
                    session, &snapshot, err, sizeof(err)) != 0) {
            fprintf(stderr, "stage %d final restore failed: %s\n", stage, err);
            fail = 1;
        }

        double fork_decode_diff = 0.0;
        if (!fail) {
            ds4_session *branch = NULL;
            if (ds4_session_create(&branch, engine, opt.context_size) != 0 ||
                ds4_session_load_snapshot(
                        branch, &snapshot, err, sizeof(err)) != 0 ||
                copy_logits(branch, replay, n_vocab, "fork prefill")) {
                fprintf(stderr, "stage %d fork restore failed: %s\n", stage, err);
                fail = 1;
            }
            if (!fail) {
                const double fork_prefill_diff =
                    max_abs_diff(baseline, replay, n_vocab, &nonfinite);
                if (nonfinite || fork_prefill_diff != 0.0 ||
                    ds4_session_argmax(branch) != first ||
                    ds4_session_eval(branch, first, err, sizeof(err)) != 0 ||
                    copy_logits(branch, replay, n_vocab, "fork continuation")) {
                    fprintf(stderr, "stage %d fork continuation failed: %s\n",
                            stage, err);
                    fail = 1;
                } else {
                    fork_decode_diff =
                        max_abs_diff(warm, replay, n_vocab, &nonfinite);
                    if (nonfinite || ds4_session_argmax(branch) != second_warm) {
                        fprintf(stderr,
                                "stage %d fork argmax mismatch diff %.9g\n",
                                stage, fork_decode_diff);
                        fail = 1;
                    }
                }
            }
            ds4_session_free(branch);
        }

        if (!fail) {
            ds4_session_rewind(session, stage - 1);
            err[0] = '\0';
            if (ds4_session_eval(session, first, err, sizeof(err)) == 0) {
                fprintf(stderr,
                        "stage %d recurrent rewind unexpectedly decoded\n",
                        stage);
                fail = 1;
            }
        }
        if (!fail && ds4_session_load_snapshot(
                    session, &snapshot, err, sizeof(err)) != 0) {
            fprintf(stderr, "stage %d rewind recovery failed: %s\n", stage, err);
            fail = 1;
        }

        double cold_resync_diff = 0.0;
        int cancel_calls = 0;
        if (!fail && stage_index == 0 && max_tokens > stage) {
            int cancel_len = stage + 2 * ds4_session_prefill_cap(session);
            if (cancel_len > max_tokens) cancel_len = max_tokens;
            ds4_tokens cancel_prompt = {
                .v = all_tokens.v,
                .len = cancel_len,
                .cap = all_tokens.cap,
            };
            cancel_state cancel = {.calls = 0, .trigger = 3};
            ds4_session_set_cancel(session, cancel_after, &cancel);
            const int rc = ds4_session_sync(
                    session, &cancel_prompt, err, sizeof(err));
            ds4_session_set_cancel(session, NULL, NULL);
            cancel_calls = cancel.calls;
            if (rc != DS4_SESSION_SYNC_INTERRUPTED) {
                fprintf(stderr,
                        "stage %d cancellation returned %d after %d calls\n",
                        stage, rc, cancel.calls);
                fail = 1;
            }
            if (!fail && ds4_session_load_snapshot(
                        session, &snapshot, err, sizeof(err)) != 0) {
                fprintf(stderr, "stage %d cancellation recovery failed: %s\n",
                        stage, err);
                fail = 1;
            }

            if (!fail) {
                ds4_session_invalidate(session);
                if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
                    copy_logits(session, replay, n_vocab, "cold resync")) {
                    fprintf(stderr, "stage %d cold resync failed: %s\n",
                            stage, err);
                    fail = 1;
                } else {
                    cold_resync_diff =
                        max_abs_diff(baseline, replay, n_vocab, &nonfinite);
                    if (nonfinite || cold_resync_diff > 1.0e-5 ||
                        ds4_session_argmax(session) != first) {
                        fprintf(stderr,
                                "stage %d cold resync mismatch %.9g\n",
                                stage, cold_resync_diff);
                        fail = 1;
                    }
                }
            }
            if (!fail && ds4_session_load_snapshot(
                        session, &snapshot, err, sizeof(err)) != 0) {
                fprintf(stderr, "stage %d invalidation recovery failed: %s\n",
                        stage, err);
                fail = 1;
            }
        }

        printf("Solar lifecycle stage=%d common=%d chunk=%d sync=%.6fs "
               "tok/s=%.6f snapshot=%llu snapshot_s=%.6f equal=%d "
               "first=%d second=%d cold_warm=%.9g warm_replay=%.9g "
               "fork=%.9g cancel_calls=%d cold_resync=%.9g result=%s\n",
               stage, common, ds4_session_prefill_cap(session), sync_seconds,
               (stage - previous_stage) / sync_seconds,
               (unsigned long long)snapshot.len, snapshot_seconds,
               snapshot_equal, first, second_warm, cold_warm_diff,
               warm_replay_diff, fork_decode_diff, cancel_calls,
               cold_resync_diff, fail ? "FAILED" : "passed");
        fflush(stdout);
        ds4_session_snapshot_free(&snapshot);
        if (fail) break;
        previous_stage = stage;
    }

cleanup:
    free(baseline);
    free(cold);
    free(warm);
    free(replay);
    ds4_session_free(session);
    ds4_tokens_free(&all_tokens);
    ds4_engine_close(engine);
    free(rendered);
    fprintf(stderr, "Solar lifecycle regression: %s\n",
            fail ? "FAILED" : "passed");
    return fail ? 1 : 0;
}
