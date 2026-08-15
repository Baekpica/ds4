/* Motif-3 persistent three-bank regression.
 *
 *   ./tests/test_motif3_batch <model.gguf>
 *
 * The persistent batch path must match three independent greedy sessions.
 */
#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int serial_greedy(ds4_engine *engine, const ds4_tokens *prompt,
                         int budget, int *out, char *err, size_t errlen) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, 256) != 0 ||
        ds4_session_sync(session, prompt, err, errlen) != 0) {
        ds4_session_free(session);
        return 1;
    }
    for (int i = 0; i < budget; i++) {
        out[i] = ds4_session_argmax(session);
        if (i + 1 < budget &&
            ds4_session_eval(session, out[i], err, errlen) != 0) {
            ds4_session_free(session);
            return 1;
        }
    }
    ds4_session_free(session);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    if (setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) != 0 ||
        setenv("DS4_NO_BOOT_PREWARM", "1", 1) != 0 ||
        setenv("DS4_MOTIF3_PREFILL_CHUNK", "16", 1) != 0) {
        perror("setenv");
        return 1;
    }

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.defer_boot_prewarm = true;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Motif-3 engine open failed\n");
        return 1;
    }

    const char *text[3] = {
        "The capital of France is",
        "Two plus two equals",
        "Write one word for the color of grass:",
    };
    ds4_tokens prompt[3] = {0};
    int oracle[3][4] = {{0}};
    char err[256] = {0};
    int failed = 0;
    for (int i = 0; i < 3; i++) {
        ds4_tokenize_text(engine, text[i], &prompt[i]);
        if (prompt[i].len <= 0 ||
            serial_greedy(engine, &prompt[i], 4, oracle[i],
                          err, sizeof(err)) != 0) {
            fprintf(stderr, "Motif-3 scalar oracle %d failed: %s\n", i, err);
            failed = 1;
            goto cleanup;
        }
    }

    if (!ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "Motif-3 batching capability is disabled\n");
        failed = 1;
        goto cleanup;
    }
    ds4_batch_ctx *ctx = NULL;
    if (ds4_batch_ctx_create_fit(
            engine, 256, 3, 48, &ctx, err, sizeof(err)) != 0 ||
        !ctx || ds4_batch_ctx_max_seq(ctx) != 3 ||
        ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "Motif-3 batch context failed: %s\n", err);
        ds4_batch_ctx_destroy(ctx);
        failed = 1;
        goto cleanup;
    }

    /* Four tokens keep bank 0 alive while banks 1 and 2 finish admission;
     * the third scheduler pass therefore executes an actual three-row decode. */
    const int budget[3] = {4, 4, 4};
    const int eos[3] = {
        ds4_token_eos(engine), ds4_token_eos(engine), ds4_token_eos(engine),
    };
    ds4_batch_gen_result got[3] = {0};
    if (ds4_engine_batched_generate_ctx(
            ctx, prompt, 3, budget, eos, got, err, sizeof(err)) != 0) {
        fprintf(stderr, "Motif-3 persistent batch failed: %s\n", err);
        failed = 1;
    }
    for (int i = 0; i < 3; i++) {
        if (!failed && (got[i].n_tokens != 4 ||
            memcmp(got[i].tokens, oracle[i], sizeof(oracle[i])) != 0)) {
            fprintf(stderr,
                    "Motif-3 row %d differs: got=%d,%d want=%d,%d\n",
                    i, got[i].n_tokens > 0 ? got[i].tokens[0] : -1,
                    got[i].n_tokens > 1 ? got[i].tokens[1] : -1,
                    oracle[i][0], oracle[i][1]);
            failed = 1;
        }
        free(got[i].tokens);
    }
    ds4_batch_ctx_destroy(ctx);

cleanup:
    for (int i = 0; i < 3; i++) ds4_tokens_free(&prompt[i]);
    ds4_engine_close(engine);
    printf("%s\n", failed ? "FAILURES" : "all checks passed");
    return failed ? 1 : 0;
}
