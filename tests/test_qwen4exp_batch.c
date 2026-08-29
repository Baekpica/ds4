/* Public Qwen3.8 persistent-bank regression.
 *
 *   ./tests/test_qwen4exp_batch <first-model-shard.gguf>
 *
 * It compares two-bank generation with scalar sessions, then proves exact-
 * frontier and partial-prefix forks.
 */
#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

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

static int speculative_greedy(ds4_engine *engine,
                              const ds4_tokens *prompt,
                              int budget, int *out, int *multi_accepts,
                              char *err, size_t errlen) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, 128) != 0 ||
        ds4_session_sync(session, prompt, err, errlen) != 0) {
        ds4_session_free(session);
        return 1;
    }
    int done = 0;
    int multi = 0;
    while (done < budget) {
        const int token = ds4_session_argmax(session);
        int accepted[2] = {-1, -1};
        const int n = ds4_session_eval_speculative_argmax(
            session, token, budget - done, -1, accepted, 2,
            err, errlen);
        if (n <= 0 || n > 2 || n > budget - done) {
            ds4_session_free(session);
            return 1;
        }
        memcpy(out + done, accepted, (size_t)n * sizeof(*out));
        done += n;
        if (n == 2) multi++;
    }
    ds4_session_free(session);
    if (multi_accepts) *multi_accepts = multi;
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
    if (ds4_session_create(&session, engine, 128) != 0 ||
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
    ds4_session_snapshot truncated = snapshot;
    truncated.len--;
    if (ds4_session_load_snapshot(session, &truncated, err, errlen) == 0 ||
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
    int next;
    int n_cached;
    int fork_bank;
    int place_bank;
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
    req->n_cached = test->n_cached;
    req->fork_bank = test->fork_bank;
    req->place_bank = test->place_bank;
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

typedef struct {
    ds4_batch_ctx *ctx;
    const ds4_tokens *prompts;
    int budget[2];
    int next;
    int tokens[2][12];
    int n[2];
    int completed;
    int failed;
    ds4_cont_seq_stats stats[2];
} bank_mtp_case;

static int bank_mtp_admit(void *ud, ds4_cont_request *req) {
    bank_mtp_case *test = ud;
    if (test->next >= 2) return 0;
    const int i = test->next++;
    memset(req, 0, sizeof(*req));
    req->tokens = test->prompts[i].v;
    req->n = test->prompts[i].len;
    req->max_new = test->budget[i];
    req->eos = -1;
    req->user = (void *)(uintptr_t)(i + 1);
    return 1;
}

static void bank_mtp_done(void *ud, void *user, const int *tokens,
                          int n, int finish) {
    (void)finish;
    bank_mtp_case *test = ud;
    const int i = (int)(uintptr_t)user - 1;
    if (i < 0 || i >= 2) {
        test->failed = 1;
        return;
    }
    if (!tokens || n != test->budget[i] ||
        !ds4_cont_last_done_stats(test->ctx, &test->stats[i])) {
        test->failed = 1;
        return;
    }
    memcpy(test->tokens[i], tokens, sizeof(test->tokens[i]));
    test->n[i] = n;
    test->completed++;
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
    opt.mtp_draft_tokens = 2;
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
    ds4_tokens source_prompt = {0};
    ds4_tokens branch_prompt = {0};
    ds4_tokens restored_prompt = {0};
    ds4_session_payload_file bank_payload = {0};
    ds4_batch_ctx *ctx = NULL;
    ds4_tokenize_text(engine, "The capital of France is", &prompt[0]);
    ds4_tokenize_text(engine, "Two plus two equals", &prompt[1]);
    if (prompt[0].len <= 0 || prompt[1].len <= 0 ||
        !ds4_engine_has_mtp(engine) ||
        ds4_engine_mtp_draft_tokens(engine) != 2 ||
        !ds4_engine_supports_batching(engine)) {
        fprintf(stderr, "Qwen tokenizer, MTP, or batching capability failed\n");
        failed = 1;
        goto cleanup;
    }

    int oracle[2][3] = {{0}};
    for (int i = 0; i < 2; i++) {
        if (serial_greedy(
                engine, &prompt[i], 3, oracle[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "Qwen scalar oracle %d failed: %s\n", i, err);
            failed = 1;
            goto cleanup;
        }
    }
    enum { MTP_BUDGET = 12 };
    int mtp_oracle[2][MTP_BUDGET] = {{0}};
    int mtp_got[2][MTP_BUDGET] = {{0}};
    int mtp_multi_accepts[2] = {0};
    if (serial_greedy(
            engine, &prompt[0], MTP_BUDGET, mtp_oracle[0],
            err, sizeof(err)) != 0 ||
        serial_greedy(
            engine, &prompt[1], MTP_BUDGET, mtp_oracle[1],
            err, sizeof(err)) != 0 ||
        speculative_greedy(
            engine, &prompt[0], MTP_BUDGET, mtp_got[0],
            &mtp_multi_accepts[0], err, sizeof(err)) != 0 ||
        speculative_greedy(
            engine, &prompt[1], MTP_BUDGET, mtp_got[1],
            &mtp_multi_accepts[1], err, sizeof(err)) != 0 ||
        memcmp(mtp_got[0], mtp_oracle[0], sizeof(mtp_got[0])) != 0 ||
        memcmp(mtp_got[1], mtp_oracle[1], sizeof(mtp_got[1])) != 0 ||
        mtp_multi_accepts[0] == 0 || mtp_multi_accepts[1] == 0) {
        fprintf(stderr,
                "Qwen target-verified MTP failed: %s multi_accepts=%d/%d\n",
                err, mtp_multi_accepts[0], mtp_multi_accepts[1]);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < MTP_BUDGET; j++) {
                if (mtp_got[i][j] != mtp_oracle[i][j]) {
                    fprintf(stderr,
                            "Qwen MTP prompt %d first mismatch at %d: "
                            "got=%d want=%d\n",
                            i, j, mtp_got[i][j], mtp_oracle[i][j]);
                    break;
                }
            }
        }
        failed = 1;
        goto cleanup;
    }
    if (serial_snapshot_roundtrip(
            engine, &prompt[0], oracle[0], err, sizeof(err)) != 0) {
        fprintf(stderr, "Qwen serial snapshot round-trip failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    if (ds4_batch_ctx_create_fit(
            engine, 128, 2, 32, &ctx, err, sizeof(err)) != 0 ||
        !ctx || ds4_batch_ctx_max_seq(ctx) != 2 ||
        !ds4_batch_ctx_supports_partial_reuse(ctx)) {
        fprintf(stderr, "Qwen batch context failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }

    bank_mtp_case bank_mtp = {
        .ctx = ctx,
        .prompts = prompt,
        .budget = {4, MTP_BUDGET},
    };
    err[0] = '\0';
    if (ds4_engine_continuous_generate(
            ctx, bank_mtp_admit, NULL, bank_mtp_done, &bank_mtp,
            err, sizeof(err)) != 0 || bank_mtp.failed ||
        bank_mtp.completed != 2 ||
        bank_mtp.n[0] != bank_mtp.budget[0] ||
        bank_mtp.n[1] != bank_mtp.budget[1] ||
        memcmp(bank_mtp.tokens[0], mtp_oracle[0],
               (size_t)bank_mtp.budget[0] * sizeof(mtp_oracle[0][0])) != 0 ||
        memcmp(bank_mtp.tokens[1], mtp_oracle[1],
               (size_t)bank_mtp.budget[1] * sizeof(mtp_oracle[1][0])) != 0 ||
        bank_mtp.stats[1].spec_drafts == 0u ||
        bank_mtp.stats[0].spec_hits + bank_mtp.stats[1].spec_hits == 0u) {
        fprintf(stderr,
                "Qwen bank row transition failed: %s completed=%d "
                "tokens=%d/%d drafts=%llu/%llu hits=%llu/%llu\n",
                err, bank_mtp.completed, bank_mtp.n[0], bank_mtp.n[1],
                (unsigned long long)bank_mtp.stats[0].spec_drafts,
                (unsigned long long)bank_mtp.stats[1].spec_drafts,
                (unsigned long long)bank_mtp.stats[0].spec_hits,
                (unsigned long long)bank_mtp.stats[1].spec_hits);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < bank_mtp.budget[i]; j++) {
                if (bank_mtp.tokens[i][j] != mtp_oracle[i][j]) {
                    fprintf(stderr,
                            "Qwen bank row %d first mismatch at %d: "
                            "got=%d want=%d\n",
                            i, j, bank_mtp.tokens[i][j], mtp_oracle[i][j]);
                    break;
                }
            }
        }
        failed = 1;
        goto cleanup;
    }

    if (getenv("DS4_QWEN_ROW_BENCH")) {
        double row_seconds[3] = {0.0};
        double scalar_seconds[3] = {0.0};
        const int order[3][2] = {{1, 0}, {0, 1}, {1, 0}};
        const int perf_budget[2] = {MTP_BUDGET, MTP_BUDGET};
        const int perf_eos[2] = {-1, -1};
        for (int round = 0; round < 3; round++) {
            for (int pass = 0; pass < 2; pass++) {
                const int row_batch = order[round][pass];
                if ((row_batch
                         ? unsetenv("DS4_QWEN_NO_ROW_BATCH")
                         : setenv("DS4_QWEN_NO_ROW_BATCH", "1", 1)) != 0) {
                    perror("row batch benchmark environment");
                    failed = 1;
                    goto cleanup;
                }
                ds4_batch_gen_result perf[2] = {0};
                const double t0 = bench_now();
                const int rc = ds4_engine_batched_generate_ctx(
                    ctx, prompt, 2, perf_budget, perf_eos,
                    perf, err, sizeof(err));
                const double elapsed = bench_now() - t0;
                if (rc != 0 || perf[0].n_tokens != MTP_BUDGET ||
                    perf[1].n_tokens != MTP_BUDGET ||
                    memcmp(perf[0].tokens, mtp_oracle[0],
                           sizeof(mtp_oracle[0])) != 0 ||
                    memcmp(perf[1].tokens, mtp_oracle[1],
                           sizeof(mtp_oracle[1])) != 0) {
                    fprintf(stderr,
                            "Qwen row benchmark parity failed: mode=%s %s\n",
                            row_batch ? "row" : "scalar", err);
                    free(perf[0].tokens);
                    free(perf[1].tokens);
                    failed = 1;
                    goto cleanup;
                }
                free(perf[0].tokens);
                free(perf[1].tokens);
                (row_batch ? row_seconds : scalar_seconds)[round] = elapsed;
            }
        }
        unsetenv("DS4_QWEN_NO_ROW_BATCH");
        double row_total = 0.0, scalar_total = 0.0;
        for (int i = 0; i < 3; i++) {
            row_total += row_seconds[i];
            scalar_total += scalar_seconds[i];
        }
        const double tokens = 2.0 * MTP_BUDGET * 3.0;
        printf("Qwen row A/B: row %.2f tok/s [%.3f %.3f %.3f s], "
               "scalar %.2f tok/s [%.3f %.3f %.3f s], speedup %.2f%%\n",
               tokens / row_total,
               row_seconds[0], row_seconds[1], row_seconds[2],
               tokens / scalar_total,
               scalar_seconds[0], scalar_seconds[1], scalar_seconds[2],
               (scalar_total / row_total - 1.0) * 100.0);
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
            memcmp(got[i].tokens, oracle[i],
                   2u * sizeof(oracle[i][0])) != 0) {
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

    const int *committed = NULL;
    const int committed_n = ds4_batch_ctx_bank_committed(ctx, 0, &committed);
    if (committed_n != prompt[0].len + 1 ||
        ds4_cont_bank_payload_bytes(ctx, 0u) == 0u ||
        ds4_cont_bank_stage_payload(
            ctx, 0u, &bank_payload, err, sizeof(err)) != 0) {
        fprintf(stderr, "Qwen durable bank stage failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    FILE *bank_fp = fopen(bank_payload.path, "rb");
    if (!bank_fp || ds4_cont_bank_restore_payload(
            ctx, 1u, bank_fp, bank_payload.bytes,
            err, sizeof(err)) != 0) {
        fprintf(stderr, "Qwen durable bank restore failed: %s\n", err);
        if (bank_fp) fclose(bank_fp);
        failed = 1;
        goto cleanup;
    }
    if (fclose(bank_fp) != 0) {
        perror("Qwen durable bank close");
        failed = 1;
        goto cleanup;
    }
    restored_prompt.len = committed_n + 1;
    restored_prompt.cap = restored_prompt.len;
    restored_prompt.v = malloc(
        (size_t)restored_prompt.len * sizeof(*restored_prompt.v));
    if (!restored_prompt.v) {
        fprintf(stderr, "Qwen restored prompt allocation failed\n");
        failed = 1;
        goto cleanup;
    }
    memcpy(restored_prompt.v, committed,
           (size_t)committed_n * sizeof(*restored_prompt.v));
    restored_prompt.v[committed_n] = oracle[0][1];
    fork_case restored = {
        .prompt = &restored_prompt,
        .n_cached = committed_n,
        .place_bank = 2,
        .cached = -1,
        .computed = -1,
        .bank = -1,
    };
    if (ds4_engine_continuous_generate(
            ctx, fork_admit, NULL, fork_done, &restored,
            err, sizeof(err)) != 0 || restored.failed ||
        restored.cached != committed_n || restored.computed != 1 ||
        restored.bank != 1 || restored.token != oracle[0][2]) {
        fprintf(stderr,
                "Qwen restored bank continuation failed: %s "
                "split=%d+%d bank=%d token=%d/%d\n",
                err, restored.cached, restored.computed, restored.bank,
                restored.token, oracle[0][2]);
        failed = 1;
        goto cleanup;
    }

    for (int i = 0; i < prompt[0].len; i++)
        ds4_tokens_push(&fork_prompt, prompt[0].v[i]);
    ds4_tokens_push(&fork_prompt, oracle[0][0]);
    fork_case fork = {.prompt = &fork_prompt,
                      .n_cached = fork_prompt.len,
                      .fork_bank = 1,
                      .place_bank = 2,
                      .cached = -1,
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
    if (failed) goto cleanup;

    /* Capture a semantic checkpoint at the first request boundary, extend
     * the source by two tokens, then diverge after one shared suffix token.
     * The partial fork must restore the older checkpoint, replay two rows,
     * and match a cold serial oracle without mutating the source bank. */
    fork_case source_base = {
        .prompt = &prompt[0],
        .place_bank = 1,
        .cached = -1,
        .computed = -1,
        .bank = -1,
    };
    if (ds4_engine_continuous_generate(
            ctx, fork_admit, NULL, fork_done, &source_base,
            err, sizeof(err)) != 0 || source_base.failed ||
        source_base.cached != 0 || source_base.computed != prompt[0].len ||
        source_base.bank != 0) {
        fprintf(stderr,
                "Qwen partial source checkpoint failed: %s "
                "split=%d+%d bank=%d\n",
                err, source_base.cached, source_base.computed,
                source_base.bank);
        failed = 1;
        goto cleanup;
    }
    for (int i = 0; i < prompt[0].len; i++)
        ds4_tokens_push(&source_prompt, prompt[0].v[i]);
    ds4_tokens_push(&source_prompt, source_base.token);
    ds4_tokens_push(&source_prompt, prompt[1].v[0]);
    fork_case source_tail = {
        .prompt = &source_prompt,
        .n_cached = prompt[0].len,
        .place_bank = 1,
        .cached = -1,
        .computed = -1,
        .bank = -1,
    };
    if (ds4_engine_continuous_generate(
            ctx, fork_admit, NULL, fork_done, &source_tail,
            err, sizeof(err)) != 0 || source_tail.failed ||
        source_tail.cached != prompt[0].len || source_tail.computed != 2 ||
        source_tail.bank != 0) {
        fprintf(stderr,
                "Qwen partial source extension failed: %s "
                "split=%d+%d bank=%d\n",
                err, source_tail.cached, source_tail.computed,
                source_tail.bank);
        failed = 1;
        goto cleanup;
    }
    for (int i = 0; i < prompt[0].len + 1; i++)
        ds4_tokens_push(&branch_prompt, source_prompt.v[i]);
    ds4_tokens_push(
        &branch_prompt,
        (source_prompt.v[source_prompt.len - 1] + 1) %
            ds4_engine_vocab_size(engine));
    int branch_oracle = -1;
    if (serial_greedy(
            engine, &branch_prompt, 1, &branch_oracle,
            err, sizeof(err)) != 0) {
        fprintf(stderr, "Qwen partial cold oracle failed: %s\n", err);
        failed = 1;
        goto cleanup;
    }
    fork_case partial = {
        .prompt = &branch_prompt,
        .n_cached = prompt[0].len + 1,
        .fork_bank = 1,
        .place_bank = 2,
        .cached = -1,
        .computed = -1,
        .bank = -1,
    };
    if (ds4_engine_continuous_generate(
            ctx, fork_admit, NULL, fork_done, &partial,
            err, sizeof(err)) != 0 || partial.failed ||
        partial.cached != prompt[0].len || partial.computed != 2 ||
        partial.bank != 1 || partial.token != branch_oracle ||
        ds4_batch_ctx_bank_committed(ctx, 0, NULL) != source_prompt.len ||
        ds4_batch_ctx_bank_committed(ctx, 1, NULL) != branch_prompt.len) {
        fprintf(stderr,
                "Qwen partial fork failed: %s split=%d+%d bank=%d "
                "token=%d/%d source=%d destination=%d\n",
                err, partial.cached, partial.computed, partial.bank,
                partial.token, branch_oracle,
                ds4_batch_ctx_bank_committed(ctx, 0, NULL),
                ds4_batch_ctx_bank_committed(ctx, 1, NULL));
        failed = 1;
    }
    if (failed) goto cleanup;

    /* An idle Qwen bank can fund a serial graph, then rebuild cold on demand. */
    const uint32_t victim[] = {0u};
    ds4_reclaim_plan reclaim = {0};
    ds4_reclaim_result released = {0};
    if (ds4_batch_ctx_reclaim_prepare(
            ctx, victim, 1u, 1u, &reclaim) != DS4_RECLAIM_OK ||
        reclaim.n != 1u || reclaim.banks[0].bank != 0u ||
        ds4_batch_ctx_reclaim_commit(
            ctx, &reclaim, &released) != DS4_RECLAIM_OK ||
        released.banks_reclaimed != 1u || released.bytes_released == 0u ||
        released.bytes_released != reclaim.est_bytes ||
        ds4_batch_ctx_bank_committed(ctx, 0, NULL) != 0) {
        fprintf(stderr,
                "Qwen idle graph reclaim failed: status=%s planned=%u "
                "reclaimed=%u released=%.2f MiB\n",
                ds4_reclaim_status_str(released.status), reclaim.n,
                released.banks_reclaimed,
                (double)released.bytes_released / 1048576.0);
        failed = 1;
        goto cleanup;
    }
    fork_case rebuilt = {
        .prompt = &prompt[0],
        .place_bank = 1,
        .cached = -1,
        .computed = -1,
        .bank = -1,
    };
    if (ds4_engine_continuous_generate(
            ctx, fork_admit, NULL, fork_done, &rebuilt,
            err, sizeof(err)) != 0 || rebuilt.failed ||
        rebuilt.cached != 0 || rebuilt.computed != prompt[0].len ||
        rebuilt.bank != 0 || rebuilt.token != oracle[0][0]) {
        fprintf(stderr,
                "Qwen graph rebuild failed: %s split=%d+%d bank=%d "
                "token=%d/%d\n",
                err, rebuilt.cached, rebuilt.computed, rebuilt.bank,
                rebuilt.token, oracle[0][0]);
        failed = 1;
    }

cleanup:
    ds4_session_payload_file_free(&bank_payload);
    ds4_tokens_free(&restored_prompt);
    ds4_batch_ctx_destroy(ctx);
    ds4_tokens_free(&branch_prompt);
    ds4_tokens_free(&source_prompt);
    ds4_tokens_free(&fork_prompt);
    ds4_tokens_free(&prompt[1]);
    ds4_tokens_free(&prompt[0]);
    ds4_engine_close(engine);
    if (!failed)
        printf("Qwen3.8 two-bank parity, disk KV, partial fork, and graph "
               "lifecycle: PASS (MTP drafts=%llu/%llu hits=%llu/%llu)\n",
               (unsigned long long)bank_mtp.stats[0].spec_drafts,
               (unsigned long long)bank_mtp.stats[1].spec_drafts,
               (unsigned long long)bank_mtp.stats[0].spec_hits,
               (unsigned long long)bank_mtp.stats[1].spec_hits);
    return failed;
}
