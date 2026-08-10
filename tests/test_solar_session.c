/* Public Solar Open2 session-lifecycle regression.
 *
 *   ./tests/test_solar_session <first-model-shard.gguf>
 *
 * It pins both the live-session contract and exact persistence of Solar's
 * recurrent KDA state plus live GQA KV rows.
 */
#include "../ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int copy_logits(ds4_session *s, float *out, int n_vocab,
                       const char *where) {
    if (ds4_session_copy_logits(s, out, n_vocab) != n_vocab) {
        fprintf(stderr, "%s: logits copy failed\n", where);
        return 1;
    }
    return 0;
}

static int expect_same_logits(const float *want, const float *got,
                              int n_vocab, const char *where) {
    if (memcmp(want, got, (size_t)n_vocab * sizeof(*want)) != 0) {
        fprintf(stderr, "%s: deterministic replay logits differ\n", where);
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
        fprintf(stderr, "usage: %s <first-model-shard.gguf>\n", argv[0]);
        return 2;
    }
    if (setenv("DS4_METAL_PREFILL_CHUNK", "3", 1) != 0 ||
        setenv("DS4_SOLAR_KV_FORMAT", "hybrid", 1) != 0 ||
        setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) != 0 ||
        setenv("DS4_NO_BOOT_PREWARM", "1", 1) != 0) {
        perror("setenv");
        return 1;
    }

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.defer_boot_prewarm = true;

    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    float *initial = NULL;
    float *decoded = NULL;
    float *warm = NULL;
    float *replayed = NULL;
    ds4_session_snapshot snapshot = {0};
    ds4_session_snapshot restored_snapshot = {0};
    char err[256] = "";
    int failed = 0;

    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Solar engine open failed\n");
        return 1;
    }
    if (ds4_engine_layer_count(engine) != 48 ||
        ds4_engine_hidden_f32_values(engine) != 4096u ||
        ds4_engine_n_hc(engine) != 1) {
        fprintf(stderr,
                "unexpected Solar public shape: layers=%d hidden=%llu n_hc=%d\n",
                ds4_engine_layer_count(engine),
                (unsigned long long)ds4_engine_hidden_f32_values(engine),
                ds4_engine_n_hc(engine));
        failed = 1;
        goto cleanup;
    }

    ds4_tokens_push(&prompt, 128);
    ds4_tokens_push(&prompt, 29497);
    ds4_tokens_push(&prompt, 132);
    ds4_tokens_push(&prompt, 4767);
    if (ds4_session_create(&session, engine, 128) != 0) {
        fprintf(stderr, "Solar session creation failed\n");
        failed = 1;
        goto cleanup;
    }
    if (!ds4_session_graph_pending(session) ||
        ds4_session_prefill_cap(session) != 3) {
        fprintf(stderr, "Solar lazy graph/prefill-cap contract is wrong\n");
        failed = 1;
        goto cleanup;
    }

    const int n_vocab = ds4_engine_vocab_size(engine);
    initial = calloc((size_t)n_vocab, sizeof(*initial));
    decoded = calloc((size_t)n_vocab, sizeof(*decoded));
    warm = calloc((size_t)n_vocab, sizeof(*warm));
    replayed = calloc((size_t)n_vocab, sizeof(*replayed));
    if (!initial || !decoded || !warm || !replayed) {
        fprintf(stderr, "host logits allocation failed\n");
        failed = 1;
        goto cleanup;
    }

    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar cold sync failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_graph_pending(session) ||
        ds4_session_pos(session) != prompt.len ||
        copy_logits(session, initial, n_vocab, "cold sync")) {
        fprintf(stderr, "Solar cold-sync state is wrong\n");
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_save_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0 ||
        snapshot.len != ds4_session_payload_bytes(session)) {
        fprintf(stderr, "Solar snapshot save failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    fprintf(stderr,
            "Solar snapshot: bytes=%llu MiB=%.2f tokens=%d\n",
            (unsigned long long)snapshot.len,
            (double)snapshot.len / 1048576.0,
            prompt.len);

    const uint64_t stable_generation = ds4_session_generation(session);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != prompt.len ||
        ds4_session_generation(session) != stable_generation ||
        copy_logits(session, replayed, n_vocab, "unchanged sync") ||
        expect_same_logits(initial, replayed, n_vocab, "unchanged sync")) {
        fprintf(stderr, "Solar unchanged sync was not a no-op: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    const int next = ds4_session_argmax(session);
    if (ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != prompt.len + 1 ||
        copy_logits(session, decoded, n_vocab, "first decode")) {
        fprintf(stderr, "Solar decode failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const int decoded_argmax = ds4_session_argmax(session);

    ds4_session_rewind(session, prompt.len);
    err[0] = '\0';
    if (ds4_session_eval(session, next, err, sizeof(err)) == 0 ||
        ds4_session_pos(session) != prompt.len) {
        fprintf(stderr, "Solar decode unexpectedly accepted a rewound KDA state\n");
        failed = 1;
        goto cleanup;
    }

    if (ds4_session_load_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0 ||
        ds4_session_pos(session) != prompt.len ||
        ds4_session_argmax(session) != next ||
        copy_logits(session, replayed, n_vocab, "snapshot restore") ||
        expect_same_logits(initial, replayed, n_vocab, "snapshot restore")) {
        fprintf(stderr, "Solar snapshot restore failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_save_snapshot(session, &restored_snapshot,
                                  err, sizeof(err)) != 0 ||
        restored_snapshot.len != snapshot.len ||
        memcmp(restored_snapshot.ptr, snapshot.ptr,
               (size_t)snapshot.len) != 0) {
        fprintf(stderr, "Solar restored snapshot bytes differ: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    /* A malformed layout must be rejected before touching GPU state.  Loading
     * failures deliberately invalidate the old logical checkpoint. */
    const size_t layout_magic_offset = 5u * sizeof(uint32_t);
    if (restored_snapshot.len <= layout_magic_offset) {
        fprintf(stderr, "Solar snapshot is shorter than its fixed header\n");
        failed = 1;
        goto cleanup;
    }
    const uint8_t layout_byte = restored_snapshot.ptr[layout_magic_offset];
    restored_snapshot.ptr[layout_magic_offset] ^= UINT8_C(1);
    err[0] = '\0';
    const int corrupt_rc = ds4_session_load_snapshot(
        session, &restored_snapshot, err, sizeof(err));
    restored_snapshot.ptr[layout_magic_offset] = layout_byte;
    if (corrupt_rc == 0 || ds4_session_pos(session) != 0 || err[0] == '\0') {
        fprintf(stderr,
                "Solar corrupted snapshot was not rejected cleanly: %s\n",
                err);
        failed = 1;
        goto cleanup;
    }
    err[0] = '\0';
    if (ds4_session_load_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar recovery after rejected snapshot failed: %s\n",
                err);
        failed = 1;
        goto cleanup;
    }

    err[0] = '\0';
    if (ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        copy_logits(session, warm, n_vocab, "snapshot warm decode")) {
        fprintf(stderr, "Solar snapshot warm decode failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const int warm_argmax = ds4_session_argmax(session);
    int nonfinite = 0;
    const double cold_warm_diff =
        max_abs_diff(decoded, warm, n_vocab, &nonfinite);
    fprintf(stderr,
            "Solar snapshot cold/warm decode: max_abs=%.9g argmax=%d/%d\n",
            cold_warm_diff, decoded_argmax, warm_argmax);
    if (nonfinite || cold_warm_diff > 0.25 ||
        warm_argmax != decoded_argmax) {
        fprintf(stderr,
                "Solar cold/warm snapshot decode drift exceeded its bound\n");
        failed = 1;
        goto cleanup;
    }

    if (ds4_session_load_snapshot(session, &snapshot,
                                  err, sizeof(err)) != 0) {
        fprintf(stderr, "Solar second snapshot restore failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    err[0] = '\0';
    if (ds4_session_eval(session, next, err, sizeof(err)) != 0 ||
        copy_logits(session, replayed, n_vocab,
                    "snapshot deterministic replay")) {
        fprintf(stderr, "Solar snapshot replay decode failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    const double warm_replay_diff =
        max_abs_diff(warm, replayed, n_vocab, &nonfinite);
    fprintf(stderr,
            "Solar snapshot warm/replay decode: max_abs=%.9g argmax=%d/%d\n",
            warm_replay_diff, warm_argmax, ds4_session_argmax(session));
    if (nonfinite || warm_replay_diff > 1.0e-5 ||
        ds4_session_argmax(session) != warm_argmax) {
        fprintf(stderr, "Solar restored decode is not deterministic\n");
        failed = 1;
        goto cleanup;
    }

    ds4_session_invalidate(session);
    if (ds4_session_pos(session) != 0 ||
        ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        copy_logits(session, replayed, n_vocab, "invalidated resync") ||
        expect_same_logits(initial, replayed, n_vocab, "invalidated resync")) {
        fprintf(stderr, "Solar invalidation/resync failed: %s\n", err);
        failed = 1;
    }

cleanup:
    ds4_session_snapshot_free(&restored_snapshot);
    ds4_session_snapshot_free(&snapshot);
    free(replayed);
    free(warm);
    free(decoded);
    free(initial);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    puts(failed ? "Solar public session lifecycle FAILED"
                : "Solar public session lifecycle passed");
    return failed ? 1 : 0;
}
