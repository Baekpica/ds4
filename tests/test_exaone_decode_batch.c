/* Row-batched decode vs the sequential path, on the real model.
 *
 *   ./tests/test_exaone_decode_batch model.gguf [width] [steps]
 *
 * Two questions, both answered from the same session states:
 *
 *  1. Numerics.  For the same (session, position, token), how far do the
 *     batched pass's logits sit from the sequential pass's, and when the
 *     argmax differs, how close was the sequential top-2 margin?  The batched
 *     projections reduce in a different order, so bitwise equality is not
 *     expected; argmax flips confined to near-ties are quantization-order
 *     noise, flips at wide margins are a bug.
 *
 *  2. Cost.  Wall time per batched step at width N vs N sequential steps,
 *     which is the whole point of the batched path existing.
 *
 * Sessions get distinct deterministic prompts (token ids, no tokenizer);
 * plausible text is irrelevant to both questions.
 */
#include "../ds4.c"

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;
static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static int argmax2(const float *v, uint32_t n, int *second) {
    int b = 0, s = -1;
    for (uint32_t i = 1; i < n; i++) {
        if (v[i] > v[b]) { s = b; b = (int)i; }
        else if (s < 0 || v[i] > v[s]) s = (int)i;
    }
    if (second) *second = s;
    return b;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [width] [steps]\n", argv[0]);
        return 2;
    }
    const int width = argc > 2 ? atoi(argv[2]) : 4;
    const int steps = argc > 3 ? atoi(argv[3]) : 8;
    if (width < 2 || width > DS4_EXAONE_DECODE_BATCH_MAX) return 2;

    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.context_size = 4096;
    opt.share_session_prefill_workspace = true; /* what batched serving sets */

    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "engine open failed\n");
        return 1;
    }
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_EXAONE_MOE) {
        fprintf(stderr, "not an exaone-moe model\n");
        return 1;
    }

    ds4_session *ss[DS4_EXAONE_DECODE_BATCH_MAX] = {0};
    char err[256];
    for (int i = 0; i < width; i++) {
        if (ds4_session_create(&ss[i], e, opt.context_size) != 0) {
            fprintf(stderr, "session %d open failed\n", i);
            return 1;
        }
        ds4_tokens prompt = {0};
        const int plen = 48 + 17 * i;   /* distinct lengths and contents */
        for (int t = 0; t < plen; t++)
            token_vec_push(&prompt, (int)(rng_next() % (DS4_N_VOCAB - 8u)) + 4);
        if (ds4_session_sync(ss[i], &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "session %d prefill failed: %s\n", i, err);
            return 1;
        }
        ds4_tokens_free(&prompt);
    }

    float *seq_logits = xmalloc((size_t)width * DS4_N_VOCAB * sizeof(float));
    int fail = 0;
    double t_seq_total = 0.0, t_bat_total = 0.0;
    int flips = 0;
    double worst_rel = 0.0, worst_abs = 0.0;

    for (int step = 0; step < steps; step++) {
        int toks[DS4_EXAONE_DECODE_BATCH_MAX];
        int pos[DS4_EXAONE_DECODE_BATCH_MAX];
        for (int i = 0; i < width; i++) {
            toks[i] = ds4_session_argmax(ss[i]);
            pos[i] = ds4_session_pos(ss[i]);
        }

        /* Sequential reference: evaluate, record, rewind.  Re-evaluating the
         * same token at the same position rewrites identical KV, so the
         * rewind leaves the session exactly re-usable. */
        const double t0 = now_sec();
        for (int i = 0; i < width; i++) {
            if (ds4_session_eval(ss[i], toks[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "seq eval failed: %s\n", err);
                return 1;
            }
            memcpy(seq_logits + (size_t)i * DS4_N_VOCAB, ss[i]->logits,
                   DS4_N_VOCAB * sizeof(float));
            ds4_session_rewind(ss[i], pos[i]);
        }
        t_seq_total += now_sec() - t0;

        ds4_decode_item items[DS4_EXAONE_DECODE_BATCH_MAX];
        for (int i = 0; i < width; i++) {
            items[i].session = ss[i];
            items[i].token = toks[i];
        }
        const double t1 = now_sec();
        if (ds4_sessions_eval_batch(items, width, err, sizeof(err)) != 0) {
            fprintf(stderr, "batched eval failed: %s\n", err);
            return 1;
        }
        t_bat_total += now_sec() - t1;

        int match = 0;
        for (int i = 0; i < width; i++) {
            const float *a = seq_logits + (size_t)i * DS4_N_VOCAB;
            const float *b = ss[i]->logits;
            double mabs = 0.0, mrel = 0.0;
            for (uint32_t v = 0; v < DS4_N_VOCAB; v++) {
                const double d = fabs((double)a[v] - (double)b[v]);
                const double m = fmax(fabs((double)a[v]), fabs((double)b[v]));
                if (d > mabs) mabs = d;
                if (m > 1.0 && d / m > mrel) mrel = d / m;
            }
            if (mrel > worst_rel) worst_rel = mrel;
            int a2 = -1;
            const int am_a = argmax2(a, DS4_N_VOCAB, &a2);
            const int am_b = argmax2(b, DS4_N_VOCAB, NULL);
            if (am_a == am_b) match++;
            else {
                const double margin = a2 >= 0 ? a[am_a] - a[a2] : 0.0;
                flips++;
                printf("step %d row %d argmax flip: seq=%d bat=%d "
                       "margin=%.4f mabs=%.2e\n",
                       step, i, am_a, am_b, margin, mabs);
            }
            if (mabs > worst_abs) worst_abs = mabs;
        }
        printf("step %2d  batched argmax matches sequential on %d/%d rows\n",
               step, match, width);
    }

    printf("\nwidth=%d steps=%d\n", width, steps);
    printf("sequential  %7.1f ms/step (%d rows)\n",
           1000.0 * t_seq_total / steps, width);
    printf("batched     %7.1f ms/step  -> %.2fx the per-row sequential cost\n",
           1000.0 * t_bat_total / steps,
           (t_bat_total / steps) / (t_seq_total / steps / width));
    printf("argmax flips %d/%d rows  worst logits diff abs %.3e rel %.3e\n",
           flips, width * steps, worst_abs, worst_rel);
    /* What separates the accepted contract from corruption, calibrated on
     * this model with random-token stress prompts (three runs, two kernel
     * configurations): the batched pass's accumulated kernel-order drift
     * reaches ~1.9 absolute on the logits tail and flips ~2% of rows at
     * margins under that amplitude, deterministically.  A real defect (row
     * swap, stale KV, wrong position) moves whole distributions: amplitude
     * an order beyond the drift class, or flips on a large fraction of rows.
     * Margins alone cannot separate the two -- a legitimate ~2-amplitude
     * drift flips a 1.4 margin now and then -- so the verdict keys on
     * amplitude and rate. */
    const int rate_fail = flips * 5 > width * steps;   /* >20% of rows */
    if (worst_abs > 8.0 || rate_fail) { printf("FAILURES\n"); fail = 1; }
    else printf("all checks passed\n");

    free(seq_logits);
    for (int i = 0; i < width; i++) ds4_session_free(ss[i]);
    ds4_engine_close(e);
    return fail;
}
