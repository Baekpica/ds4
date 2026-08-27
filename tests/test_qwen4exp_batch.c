/* Public Qwen3.8 persistent-bank regression.
 *
 *   ./tests/test_qwen4exp_batch <first-model-shard.gguf>
 *
 * It compares two-bank generation with scalar sessions, then proves an
 * exact-frontier fork.  Partial mid-prefix reuse is a separate capability.
 */
#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int serial_greedy(ds4_engine *engine, const ds4_tokens *prompt,
                         int budget, int *out, char *err, size_t errlen) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, 128) != 0 ||
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

typedef struct {
    const ds4_tokens *prompt;
    int next;
    int cached;
    int computed;
    int bank;
    int token;
    int failed;
} fork_case;

static int fork_admitted(void *ud, void *user, int cached,
                         int computed, int bank) {
    (void)user;
    fork_case *test = ud;
    test->cached = cached;
    test->computed = computed;
    test->bank = bank;
    return 1;
}

static int fork_admit(void *ud, ds4_cont_request *req) {
    fork_case *test = ud;
    if (test->next++) return 0;
    memset(req, 0, sizeof(*req));
    req->tokens = test->prompt->v;
    req->n = test->prompt->len;
    req->max_new = 1;
    req->eos = -1;
    req->n_cached = test->prompt->len;
    req->fork_bank = 1;
    req->place_bank = 2;
    req->on_admitted = fork_admitted;
    req->user = test;
    return 1;
}

static void fork_done(void *ud, void *user, const int *tokens,
                      int n, int finish) {
    (void)user;
    (void)finish;
    fork_case *test = ud;
    if (!tokens || n != 1) {
        test->failed = 1;
        return;
    }
    test->token = tokens[0];
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <first-model-shard.gguf>\n", argv[0]);
        return 2;
    }
    if (setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) != 0 ||
        setenv("DS4_NO_BOOT_PREWARM", "1", 1) != 0 ||
        setenv("DS4_MEMGOV", "observe", 1) != 0 ||
        setenv("DS4_QWEN_BATCH", "1", 1) != 0) {
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
        fprintf(stderr, "Qwen engine open failed\n");
        return 1;
    }

    int failed = 0;
    char err[256] = "";
    ds4_tokens prompt[2] = {0};
    ds4_tokens fork_prompt = {0};
    ds4_batch_ctx *ctx = NULL;
    ds4_tokenize_text(engine, "The capital of France is", &prompt[0]);
    ds4_tokenize_text(engine, "Two plus two equals", &prompt[1]);
    if (prompt[0].len <= 0 || prompt[1].len <= 0 ||
        !ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "Qwen tokenizer or batching capability failed\n");
        failed = 1;
        goto cleanup;
    }

    int oracle[2][2] = {{0}};
    for (int i = 0; i < 2; i++) {
        if (serial_greedy(
                engine, &prompt[i], 2, oracle[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "Qwen scalar oracle %d failed: %s\n", i, err);
            failed = 1;
            goto cleanup;
        }
    }

    if (ds4_batch_ctx_create_fit(
            engine, 128, 2, 32, &ctx, err, sizeof(err)) != 0 ||
        !ctx || ds4_batch_ctx_max_seq(ctx) != 2 ||
        ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "Qwen batch context failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    const int budget[2] = {2, 2};
    const int eos[2] = {ds4_token_eos(engine), ds4_token_eos(engine)};
    ds4_batch_gen_result got[2] = {0};
    if (ds4_engine_batched_generate_ctx(
            ctx, prompt, 2, budget, eos, got, err, sizeof(err)) != 0) {
        fprintf(stderr, "Qwen persistent batch failed: %s\n", err);
        failed = 1;
    }
    for (int i = 0; !failed && i < 2; i++) {
        if (got[i].n_tokens != 2 ||
            memcmp(got[i].tokens, oracle[i], sizeof(oracle[i])) != 0) {
            fprintf(stderr,
                    "Qwen row %d differs from scalar: got=%d,%d want=%d,%d\n",
                    i,
                    got[i].n_tokens > 0 ? got[i].tokens[0] : -1,
                    got[i].n_tokens > 1 ? got[i].tokens[1] : -1,
                    oracle[i][0], oracle[i][1]);
            failed = 1;
        }
    }
    free(got[0].tokens);
    free(got[1].tokens);
    if (failed) goto cleanup;

    for (int i = 0; i < prompt[0].len; i++)
        ds4_tokens_push(&fork_prompt, prompt[0].v[i]);
    ds4_tokens_push(&fork_prompt, oracle[0][0]);
    fork_case fork = {.prompt = &fork_prompt, .cached = -1,
                      .computed = -1, .bank = -1};
    if (ds4_engine_continuous_generate(
            ctx, fork_admit, NULL, fork_done, &fork,
            err, sizeof(err)) != 0 || fork.failed ||
        fork.cached != fork_prompt.len || fork.computed != 0 ||
        fork.bank != 1 || fork.token != oracle[0][1]) {
        fprintf(stderr,
                "Qwen exact fork failed: %s split=%d+%d bank=%d token=%d/%d\n",
                err, fork.cached, fork.computed, fork.bank,
                fork.token, oracle[0][1]);
        failed = 1;
    }

cleanup:
    ds4_batch_ctx_destroy(ctx);
    ds4_tokens_free(&fork_prompt);
    ds4_tokens_free(&prompt[1]);
    ds4_tokens_free(&prompt[0]);
    ds4_engine_close(engine);
    if (!failed)
        printf("Qwen3.8 two-bank parity and exact-frontier fork: PASS\n");
    return failed;
}
