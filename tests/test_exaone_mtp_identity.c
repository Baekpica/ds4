#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int generate_plain(ds4_session *s, ds4_engine *e,
                          int *out, int cap, double *elapsed) {
    char err[256] = "";
    const double t0 = monotonic_sec();
    int n = 0;
    while (n < cap) {
        const int token = ds4_session_argmax(s);
        out[n++] = token;
        if (ds4_token_is_stop(e, token)) break;
        if (ds4_session_eval(s, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "plain decode failed: %s\n", err);
            return -1;
        }
    }
    *elapsed = monotonic_sec() - t0;
    return n;
}

static int generate_mtp(ds4_session *s, ds4_engine *e,
                        int *out, int cap, double *elapsed) {
    char err[256] = "";
    const int eos = ds4_token_eos(e);
    const double t0 = monotonic_sec();
    int n = 0;
    while (n < cap) {
        const int first = ds4_session_argmax(s);
        if (ds4_token_is_stop(e, first)) {
            out[n++] = first;
            break;
        }
        int accepted[2] = { -1, -1 };
        const int got = ds4_session_eval_speculative_argmax(
            s, first, cap - n, eos, accepted, 2, err, sizeof(err));
        if (got < 0) {
            fprintf(stderr, "MTP decode failed: %s\n", err);
            return -1;
        }
        for (int i = 0; i < got && n < cap; i++) {
            out[n++] = accepted[i];
            if (ds4_token_is_stop(e, accepted[i])) {
                *elapsed = monotonic_sec() - t0;
                return n;
            }
        }
    }
    *elapsed = monotonic_sec() - t0;
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL.gguf [TOKENS]\n", argv[0]);
        return 2;
    }
    int max_tokens = argc == 3 ? atoi(argv[2]) : 64;
    if (max_tokens < 16 || max_tokens > 512) {
        fprintf(stderr, "TOKENS must be in [16, 512]\n");
        return 2;
    }

    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = argv[1],
        .backend = DS4_BACKEND_CUDA,
        .prefill_chunk = 128,
        .power_percent = 100,
        .exaone_mtp = true,
        .exaone_mtp_timing = true,
    };
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "failed to open K-EXAONE engine\n");
        return 1;
    }
    if (!ds4_engine_has_mtp(engine) || ds4_engine_mtp_draft_tokens(engine) != 2) {
        fprintf(stderr, "integrated K-EXAONE MTP was not enabled\n");
        ds4_engine_close(engine);
        return 1;
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(
        engine, NULL,
        "Write every integer from 6 through 200 inclusive, one integer per "
        "line, without explanation.",
        DS4_THINK_NONE, &prompt);
    const int ctx = prompt.len + max_tokens + 32;
    ds4_session *plain = NULL;
    ds4_session *mtp = NULL;
    char err[256] = "";
    if (ds4_session_create(&plain, engine, ctx) != 0 ||
        ds4_session_create(&mtp, engine, ctx) != 0 ||
        ds4_session_sync(plain, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_sync(mtp, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "session setup failed: %s\n", err);
        ds4_session_free(plain);
        ds4_session_free(mtp);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    int *plain_tokens = calloc((size_t)max_tokens, sizeof(*plain_tokens));
    int *mtp_tokens = calloc((size_t)max_tokens, sizeof(*mtp_tokens));
    double plain_sec = 0.0, mtp_sec = 0.0;
    const int n_plain = generate_plain(plain, engine, plain_tokens,
                                       max_tokens, &plain_sec);
    const int n_mtp = generate_mtp(mtp, engine, mtp_tokens,
                                   max_tokens, &mtp_sec);
    uint32_t verify_cycles = 0;
    uint32_t accepted_drafts = 0;
    bool quenched = false;
    const int stats_rc = ds4_test_exaone_mtp_stats(
        mtp, &verify_cycles, &accepted_drafts, &quenched);
    int mismatch = -1;
    if (n_plain != n_mtp) {
        mismatch = n_plain < n_mtp ? n_plain : n_mtp;
    } else {
        for (int i = 0; i < n_plain; i++) {
            if (plain_tokens[i] != mtp_tokens[i]) {
                mismatch = i;
                break;
            }
        }
    }

    printf("exaone MTP greedy identity: plain=%d MTP=%d mismatch=%d\n",
           n_plain, n_mtp, mismatch);
    printf("MTP verifier: cycles=%u accepted=%u quenched=%d\n",
           verify_cycles, accepted_drafts, quenched ? 1 : 0);
    if (n_plain > 0 && plain_sec > 0.0) {
        printf("plain: %.3f s, %.3f tok/s\n", plain_sec,
               (double)n_plain / plain_sec);
    }
    if (n_mtp > 0 && mtp_sec > 0.0) {
        printf("MTP:   %.3f s, %.3f tok/s\n", mtp_sec,
               (double)n_mtp / mtp_sec);
    }
    if (mismatch >= 0 && mismatch < n_plain && mismatch < n_mtp) {
        fprintf(stderr, "first mismatch at %d: plain=%d MTP=%d\n",
                mismatch, plain_tokens[mismatch], mtp_tokens[mismatch]);
    }

    free(plain_tokens);
    free(mtp_tokens);
    ds4_session_free(plain);
    ds4_session_free(mtp);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    if (stats_rc != 0 || verify_cycles == 0) {
        fprintf(stderr, "MTP verifier was not exercised\n");
        return 1;
    }
    return n_plain < 0 || n_mtp < 0 || mismatch >= 0 ? 1 : 0;
}
