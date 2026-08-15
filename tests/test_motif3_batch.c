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

static int serial_snapshot_roundtrip(ds4_engine *engine,
                                     const ds4_tokens *prompt,
                                     const int oracle[2],
                                     char *err, size_t errlen) {
    ds4_session *session = NULL;
    ds4_session_snapshot snapshot = {0};
    ds4_session_snapshot corrupt = {0};
    int rc = 1;
    if (ds4_session_create(&session, engine, 256) != 0 ||
        ds4_session_sync(session, prompt, err, errlen) != 0 ||
        ds4_session_save_snapshot(session, &snapshot, err, errlen) != 0 ||
        snapshot.len != ds4_session_payload_bytes(session)) {
        goto done;
    }
    corrupt.ptr = malloc((size_t)snapshot.len);
    if (!corrupt.ptr) goto done;
    memcpy(corrupt.ptr, snapshot.ptr, (size_t)snapshot.len);
    corrupt.len = corrupt.cap = snapshot.len;
    corrupt.ptr[5u * sizeof(uint32_t)] ^= UINT8_C(1);
    if (ds4_session_load_snapshot(session, &corrupt, err, errlen) == 0 ||
        ds4_session_pos(session) != 0) {
        goto done;
    }
    if (ds4_session_load_snapshot(session, &snapshot, err, errlen) != 0 ||
        ds4_session_argmax(session) != oracle[0] ||
        ds4_session_eval(session, oracle[0], err, errlen) != 0 ||
        ds4_session_argmax(session) != oracle[1]) {
        goto done;
    }
    rc = 0;
done:
    ds4_session_snapshot_free(&corrupt);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(session);
    return rc;
}

typedef struct {
    const ds4_tokens *prompt;
    int cached;
    int place_bank;
    int next;
    int admitted_cached;
    int admitted_computed;
    int admitted_bank;
    int token;
    int done;
    int failed;
} restore_case;

static int restore_admitted(void *ud, void *user, int cached,
                            int computed, int bank) {
    (void)user;
    restore_case *test = ud;
    test->admitted_cached = cached;
    test->admitted_computed = computed;
    test->admitted_bank = bank;
    return 1;
}

static int restore_admit(void *ud, ds4_cont_request *req) {
    restore_case *test = ud;
    if (test->next++) return 0;
    memset(req, 0, sizeof(*req));
    req->tokens = test->prompt->v;
    req->n = test->prompt->len;
    req->max_new = 1;
    req->eos = -1;
    req->n_cached = test->cached;
    req->place_bank = test->place_bank + 1;
    req->on_admitted = restore_admitted;
    req->user = test;
    return 1;
}

static void restore_done(void *ud, void *user, const int *tokens,
                         int n, int finish) {
    (void)user;
    (void)finish;
    restore_case *test = ud;
    if (!tokens || n != 1) {
        test->failed = 1;
        return;
    }
    test->token = tokens[0];
    test->done = 1;
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
    int oracle[3][5] = {{0}};
    char err[256] = {0};
    int failed = 0;
    ds4_batch_ctx *ctx = NULL;
    ds4_session_payload_file bank_payload = {0};
    ds4_tokens restored_prompt = {0};
    for (int i = 0; i < 3; i++) {
        ds4_tokenize_text(engine, text[i], &prompt[i]);
        if (prompt[i].len <= 0 ||
            serial_greedy(engine, &prompt[i], 5, oracle[i],
                          err, sizeof(err)) != 0) {
            fprintf(stderr, "Motif-3 scalar oracle %d failed: %s\n", i, err);
            failed = 1;
            goto cleanup;
        }
    }
    if (serial_snapshot_roundtrip(
            engine, &prompt[0], oracle[0], err, sizeof(err)) != 0) {
        fprintf(stderr, "Motif-3 serial snapshot round-trip failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    if (!ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "Motif-3 batching capability is disabled\n");
        failed = 1;
        goto cleanup;
    }
    if (ds4_batch_ctx_create_fit(
            engine, 256, 3, 48, &ctx, err, sizeof(err)) != 0 ||
        !ctx || ds4_batch_ctx_max_seq(ctx) != 3 ||
        ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "Motif-3 batch context failed: %s\n", err);
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
            memcmp(got[i].tokens, oracle[i],
                   4u * sizeof(oracle[i][0])) != 0)) {
            fprintf(stderr,
                    "Motif-3 row %d differs: got=%d,%d want=%d,%d\n",
                    i, got[i].n_tokens > 0 ? got[i].tokens[0] : -1,
                    got[i].n_tokens > 1 ? got[i].tokens[1] : -1,
                    oracle[i][0], oracle[i][1]);
            failed = 1;
        }
        free(got[i].tokens);
    }

    const int *committed = NULL;
    const int committed_n = ds4_batch_ctx_bank_committed(ctx, 0, &committed);
    if (!failed &&
        (committed_n != prompt[0].len + 3 ||
         memcmp(committed, prompt[0].v,
                (size_t)prompt[0].len * sizeof(*committed)) != 0 ||
         memcmp(committed + prompt[0].len, oracle[0],
                3u * sizeof(*committed)) != 0 ||
         ds4_cont_bank_payload_bytes(ctx, 0u) == 0u ||
         ds4_cont_bank_stage_payload(
             ctx, 0u, &bank_payload, err, sizeof(err)) != 0)) {
        fprintf(stderr, "Motif-3 durable bank stage failed: %s\n", err);
        failed = 1;
    }
    FILE *bank_fp = NULL;
    if (!failed) bank_fp = fopen(bank_payload.path, "rb");
    if (!failed &&
        (!bank_fp || ds4_cont_bank_restore_payload(
             ctx, 1u, bank_fp, bank_payload.bytes,
             err, sizeof(err)) != 0)) {
        fprintf(stderr, "Motif-3 durable bank restore failed: %s\n", err);
        failed = 1;
    }
    if (bank_fp && fclose(bank_fp) != 0 && !failed) {
        perror("Motif-3 durable bank close");
        failed = 1;
    }

    if (!failed) {
        restored_prompt.len = committed_n + 1;
        restored_prompt.cap = restored_prompt.len;
        restored_prompt.v = malloc(
            (size_t)restored_prompt.len * sizeof(*restored_prompt.v));
        if (!restored_prompt.v) {
            fprintf(stderr, "Motif-3 restored prompt allocation failed\n");
            failed = 1;
        } else {
            memcpy(restored_prompt.v, committed,
                   (size_t)committed_n * sizeof(*restored_prompt.v));
            restored_prompt.v[committed_n] = oracle[0][3];
        }
    }
    restore_case restored = {
        .prompt = &restored_prompt,
        .cached = committed_n,
        .place_bank = 1,
        .admitted_cached = -1,
        .admitted_computed = -1,
        .admitted_bank = -1,
    };
    if (!failed &&
        (ds4_engine_continuous_generate(
             ctx, restore_admit, NULL, restore_done, &restored,
             err, sizeof(err)) != 0 ||
         restored.failed || !restored.done ||
         restored.token != oracle[0][4] ||
         restored.admitted_cached != committed_n ||
         restored.admitted_computed != 1 ||
         restored.admitted_bank != 1)) {
        fprintf(stderr,
                "Motif-3 restored bank continuation failed: %s "
                "token=%d want=%d split=%d+%d bank=%d\n",
                err, restored.token, oracle[0][4],
                restored.admitted_cached, restored.admitted_computed,
                restored.admitted_bank);
        failed = 1;
    }

cleanup:
    ds4_session_payload_file_free(&bank_payload);
    ds4_tokens_free(&restored_prompt);
    ds4_batch_ctx_destroy(ctx);
    for (int i = 0; i < 3; i++) ds4_tokens_free(&prompt[i]);
    ds4_engine_close(engine);
    printf("%s\n", failed ? "FAILURES" : "all checks passed");
    return failed ? 1 : 0;
}
