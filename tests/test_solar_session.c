/* Real-model Solar session-state regression.
 *
 *   DS4_SOLAR_KV_FORMAT=fp8 ./tests/test_solar_session model.gguf
 *
 * This intentionally uses one tiny-context session so the 89 GiB mixed model
 * still fits on one H100. It validates the stateful behaviors most likely to
 * regress for the hybrid GQA/KDA graph: snapshot/restore, decode replay,
 * rewind rejection until recurrent state is restored, and cold invalidation.
 */
#include "../ds4.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.context_size = 128;
    opt.prefill_chunk = 1;
    opt.share_session_prefill_workspace = true;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Solar engine open failed\n");
        return 1;
    }
    if (ds4_engine_layer_count(engine) != 48) {
        fprintf(stderr, "expected 48 Solar layers, got %d\n",
                ds4_engine_layer_count(engine));
        ds4_engine_close(engine);
        return 1;
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(
        engine, NULL,
        "대한민국의 수도를 한 문장으로 답하세요.",
        DS4_THINK_NONE, &prompt);
    if (prompt.len <= 0 || prompt.len >= opt.context_size - 2) {
        fprintf(stderr, "unexpected rendered prompt length: %d\n", prompt.len);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    char err[256] = "";
    const char *capture_env = getenv("DS4_TEST_CUDA_PROFILER_CAPTURE");
    const int capture_requested =
        capture_env && capture_env[0] != '\0' && strcmp(capture_env, "0") != 0;
    int capture_active = 0;

    if (ds4_session_create(&session, engine, opt.context_size) != 0) {
        fprintf(stderr, "Solar session creation failed\n");
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    if (capture_requested) {
        const cudaError_t capture_rc = cudaProfilerStart();
        if (capture_rc != cudaSuccess) {
            fprintf(stderr, "cudaProfilerStart failed: %s\n",
                    cudaGetErrorString(capture_rc));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        capture_active = 1;
    }
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar session setup failed: %s\n", err);
        if (capture_active) cudaProfilerStop();
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    const int n_vocab = ds4_engine_vocab_size(engine);
    float *prefill_logits = calloc((size_t)n_vocab, sizeof(*prefill_logits));
    float *cold_decode_logits = calloc((size_t)n_vocab, sizeof(*cold_decode_logits));
    float *warm_decode_logits = calloc((size_t)n_vocab, sizeof(*warm_decode_logits));
    float *replay_logits = calloc((size_t)n_vocab, sizeof(*replay_logits));
    if (!prefill_logits || !cold_decode_logits || !warm_decode_logits ||
        !replay_logits) {
        fprintf(stderr, "host logits allocation failed\n");
        if (capture_active) cudaProfilerStop();
        free(prefill_logits);
        free(cold_decode_logits);
        free(warm_decode_logits);
        free(replay_logits);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    int fail = copy_logits(session, prefill_logits, n_vocab, "prefill");
    const int first = ds4_session_argmax(session);
    ds4_session_snapshot snapshot = {0};
    if (!fail && ds4_session_save_snapshot(
                     session, &snapshot, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar snapshot save failed: %s\n", err);
        fail = 1;
    }

    if (!fail && ds4_session_eval(session, first, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar first decode failed: %s\n", err);
        fail = 1;
    }
    const int second_cold = fail ? -1 : ds4_session_argmax(session);
    if (!fail) {
        fail = copy_logits(session, cold_decode_logits, n_vocab, "cold decode");
    }

    if (!fail) {
        ds4_session_rewind(session, prompt.len);
        err[0] = '\0';
        if (ds4_session_eval(session, first, err, sizeof(err)) == 0) {
            fprintf(stderr,
                    "Solar decode unexpectedly succeeded after recurrent rewind\n");
            fail = 1;
        }
    }

    if (!fail && ds4_session_load_snapshot(
                     session, &snapshot, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar snapshot restore failed: %s\n", err);
        fail = 1;
    }
    if (!fail && (ds4_session_pos(session) != prompt.len ||
                  ds4_session_argmax(session) != first)) {
        fprintf(stderr, "Solar snapshot restored the wrong position/logits\n");
        fail = 1;
    }

    ds4_session_snapshot restored_snapshot = {0};
    int snapshot_bytes_equal = 0;
    if (!fail && ds4_session_save_snapshot(
                     session, &restored_snapshot, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar restored snapshot re-save failed: %s\n", err);
        fail = 1;
    }
    if (!fail) {
        snapshot_bytes_equal =
            restored_snapshot.len == snapshot.len &&
            memcmp(restored_snapshot.ptr, snapshot.ptr, (size_t)snapshot.len) == 0;
        if (!snapshot_bytes_equal) {
            fprintf(stderr,
                    "Solar restored snapshot bytes differ: %llu/%llu bytes\n",
                    (unsigned long long)restored_snapshot.len,
                    (unsigned long long)snapshot.len);
            fail = 1;
        }
    }

    int nonfinite = 0;
    double restored_prefill_diff = 0.0;
    double cold_warm_decode_diff = 0.0;
    double warm_replay_decode_diff = 0.0;
    double cold_prefill_diff = 0.0;
    if (!fail) {
        fail = copy_logits(session, replay_logits, n_vocab, "restore");
        restored_prefill_diff = max_abs_diff(
            prefill_logits, replay_logits, n_vocab, &nonfinite);
        if (nonfinite || restored_prefill_diff != 0.0) {
            fprintf(stderr,
                    "Solar restored prefill logits differ: max_abs=%.9g\n",
                    restored_prefill_diff);
            fail = 1;
        }
    }

    if (!fail && ds4_session_eval(session, first, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar warm decode failed: %s\n", err);
        fail = 1;
    }
    const int second_warm = fail ? -1 : ds4_session_argmax(session);
    if (!fail) {
        fail = copy_logits(session, warm_decode_logits, n_vocab, "warm decode");
    }
    if (!fail) {
        cold_warm_decode_diff = max_abs_diff(
            cold_decode_logits, warm_decode_logits, n_vocab, &nonfinite);
        if (nonfinite || second_warm != second_cold) {
            fprintf(stderr,
                    "Solar cold/warm decode mismatch: max_abs=%.9g "
                    "argmax=%d/%d\n",
                    cold_warm_decode_diff, second_cold, second_warm);
            fail = 1;
        }
    }

    if (!fail && ds4_session_load_snapshot(
                     session, &snapshot, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar second snapshot restore failed: %s\n", err);
        fail = 1;
    }
    if (!fail && ds4_session_eval(session, first, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar replay decode failed: %s\n", err);
        fail = 1;
    }
    if (!fail) {
        fail = copy_logits(session, replay_logits, n_vocab, "replay decode");
        warm_replay_decode_diff = max_abs_diff(
            warm_decode_logits, replay_logits, n_vocab, &nonfinite);
        if (nonfinite || warm_replay_decode_diff > 1.0e-5 ||
            ds4_session_argmax(session) != second_warm) {
            fprintf(stderr,
                    "Solar warm replay decode mismatch: max_abs=%.9g "
                    "argmax=%d/%d\n",
                    warm_replay_decode_diff, ds4_session_argmax(session),
                    second_warm);
            fail = 1;
        }
    }

    if (!fail) {
        ds4_session_invalidate(session);
        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "Solar cold resync failed: %s\n", err);
            fail = 1;
        }
    }
    if (!fail) {
        fail = copy_logits(session, replay_logits, n_vocab, "cold resync");
        cold_prefill_diff = max_abs_diff(
            prefill_logits, replay_logits, n_vocab, &nonfinite);
        if (nonfinite || cold_prefill_diff > 1.0e-5 ||
            ds4_session_argmax(session) != first) {
            fprintf(stderr,
                    "Solar cold resync mismatch: max_abs=%.9g argmax=%d/%d\n",
                    cold_prefill_diff, ds4_session_argmax(session), first);
            fail = 1;
        }
    }

    if (capture_active) {
        const cudaError_t sync_rc = cudaDeviceSynchronize();
        if (sync_rc != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize before profiler stop failed: %s\n",
                    cudaGetErrorString(sync_rc));
            fail = 1;
        }
    }

    printf("Solar session state: prompt=%d snapshot=%llu bytes "
           "snapshot-bytes-equal=%d first=%d second-cold=%d second-warm=%d\n",
           prompt.len, (unsigned long long)snapshot.len, snapshot_bytes_equal,
           first, second_cold, second_warm);
    printf("max_abs restored-prefill=%.9g cold-vs-warm-decode=%.9g "
           "warm-vs-replay-decode=%.9g cold-prefill=%.9g\n",
           restored_prefill_diff, cold_warm_decode_diff,
           warm_replay_decode_diff, cold_prefill_diff);
    printf("Solar session state regression: %s\n", fail ? "FAILED" : "passed");
    fflush(stdout);

    if (capture_active) {
        const cudaError_t capture_rc = cudaProfilerStop();
        capture_active = 0;
        if (capture_rc != cudaSuccess) {
            fprintf(stderr, "cudaProfilerStop failed: %s\n",
                    cudaGetErrorString(capture_rc));
            fail = 1;
        }
    }

    ds4_session_snapshot_free(&snapshot);
    ds4_session_snapshot_free(&restored_snapshot);
    free(prefill_logits);
    free(cold_decode_logits);
    free(warm_decode_logits);
    free(replay_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return fail ? 1 : 0;
}
