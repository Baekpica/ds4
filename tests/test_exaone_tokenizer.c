/* K-EXAONE tokenizer check.
 *
 *   ./tests/test_exaone_tokenizer <model.gguf> [--file cases.txt]
 *
 * Prints one line of token ids per input case, for diffing against
 * `llama-tokenize --ids` on the same model. A pre-tokenizer that is subtly
 * wrong still produces plausible text, so "it generated Korean" is not
 * evidence -- the only useful check is that the id stream is identical to the
 * reference implementation's.
 *
 * Only the GGUF metadata is read, so this does not touch the weights and runs
 * in a second even on an 86 GiB artifact.
 */
#include "../ds4.c"

static void emit(const ds4_vocab *vocab, const char *text) {
    token_vec out;
    memset(&out, 0, sizeof(out));
    bpe_tokenize_text(vocab, text, &out);
    for (int i = 0; i < out.len; i++) printf("%s%d", i ? " " : "", out.v[i]);
    printf("\n");
    free(out.v);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--file cases.txt]\n", argv[0]);
        return 2;
    }
    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    ds4_vocab vocab;
    vocab_load(&vocab, &model);
    fprintf(stderr, "vocab %d, bos=%d eos=%d user=%d assistant=%d system=%d\n",
            vocab.n_vocab, vocab.bos_id, vocab.eos_id, vocab.user_id,
            vocab.assistant_id, vocab.system_id);

    if (argc >= 4 && strcmp(argv[2], "--file") == 0) {
        FILE *f = fopen(argv[3], "r");
        if (!f) { perror(argv[3]); return 1; }
        char *line = NULL;
        size_t cap = 0;
        ssize_t n;
        while ((n = getline(&line, &cap, f)) != -1) {
            /* One case per line; \n and \t in the case become real control
             * characters so the whitespace branches can be exercised from a
             * line-oriented file. Must match however the reference side feeds
             * the same file, or a harness difference reads as a tokenizer bug. */
            if (n > 0 && line[n - 1] == '\n') line[n - 1] = '\0';
            char *dst = line;
            for (char *src = line; *src; src++) {
                if (src[0] == '\\' && src[1] == 'n')      { *dst++ = '\n'; src++; }
                else if (src[0] == '\\' && src[1] == 't') { *dst++ = '\t'; src++; }
                else *dst++ = *src;
            }
            *dst = '\0';
            emit(&vocab, line);
        }
        free(line);
        fclose(f);
    } else {
        for (int i = 2; i < argc; i++) emit(&vocab, argv[i]);
    }

    vocab_free(&vocab);
    model_close(&model);
    return 0;
}
