/* Public Solar Open2 session-lifecycle regression.
 *
 *   ./tests/test_solar_session <first-model-shard.gguf>
 *
 * Snapshot persistence is intentionally covered by a separate test: this one
 * pins the public live-session contract while the Solar graph is integrated
 * into the v0.5.6.2 session skeleton.
 */
#include "../ds4.h"

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
    float *replayed = NULL;
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
    replayed = calloc((size_t)n_vocab, sizeof(*replayed));
    if (!initial || !replayed) {
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
        ds4_session_pos(session) != prompt.len + 1) {
        fprintf(stderr, "Solar decode failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    ds4_session_rewind(session, prompt.len);
    err[0] = '\0';
    if (ds4_session_eval(session, next, err, sizeof(err)) == 0 ||
        ds4_session_pos(session) != prompt.len) {
        fprintf(stderr, "Solar decode unexpectedly accepted a rewound KDA state\n");
        failed = 1;
        goto cleanup;
    }
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        copy_logits(session, replayed, n_vocab, "rewind resync") ||
        expect_same_logits(initial, replayed, n_vocab, "rewind resync")) {
        fprintf(stderr, "Solar rewind resync failed: %s\n", err);
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
    free(replayed);
    free(initial);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    puts(failed ? "Solar public session lifecycle FAILED"
                : "Solar public session lifecycle passed");
    return failed ? 1 : 0;
}
