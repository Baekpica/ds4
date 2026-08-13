/* Official-final Motif-3 tokenizer regex, byte-BPE, and chat-template parity. */
#include "../ds4.c"

typedef struct {
    uint32_t kind;
    char *name;
    char *text;
    uint64_t n_token;
    int32_t *token;
} motif3_token_case;

static void token_read_exact(FILE *fp, void *dst, size_t n, const char *what) {
    if (fread(dst, 1, n, fp) != n) {
        fprintf(stderr, "short read while loading %s\n", what);
        exit(1);
    }
}

static motif3_token_case *load_cases(const char *path, uint32_t *count) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        exit(1);
    }
    char magic[8];
    uint32_t version = 0;
    token_read_exact(fp, magic, sizeof(magic), "tokenizer fixture magic");
    token_read_exact(fp, &version, sizeof(version), "tokenizer fixture version");
    token_read_exact(fp, count, sizeof(*count), "tokenizer fixture count");
    if (memcmp(magic, "DS4TOK1\0", 8) != 0 || version != 1 || *count > 1024) {
        fprintf(stderr, "unsupported tokenizer fixture: %s\n", path);
        exit(1);
    }

    motif3_token_case *cases = calloc(*count, sizeof(cases[0]));
    if (!cases) abort();
    for (uint32_t i = 0; i < *count; i++) {
        uint32_t name_len = 0;
        uint64_t text_len = 0;
        token_read_exact(fp, &cases[i].kind, sizeof(uint32_t), "case kind");
        token_read_exact(fp, &name_len, sizeof(uint32_t), "case name length");
        token_read_exact(fp, &text_len, sizeof(uint64_t), "case text length");
        token_read_exact(fp, &cases[i].n_token, sizeof(uint64_t), "case token count");
        if (cases[i].kind > 1 || name_len == 0 || name_len > 255 ||
            text_len > (16u << 20) || cases[i].n_token > (16u << 20)) {
            fprintf(stderr, "invalid tokenizer case descriptor in %s\n", path);
            exit(1);
        }
        cases[i].name = calloc((size_t)name_len + 1u, 1);
        cases[i].text = calloc((size_t)text_len + 1u, 1);
        cases[i].token = malloc((size_t)cases[i].n_token * sizeof(cases[i].token[0]));
        if (!cases[i].name || !cases[i].text ||
            (cases[i].n_token && !cases[i].token)) abort();
        token_read_exact(fp, cases[i].name, name_len, "case name");
        token_read_exact(fp, cases[i].text, (size_t)text_len, "case text");
        token_read_exact(fp, cases[i].token,
                         (size_t)cases[i].n_token * sizeof(cases[i].token[0]),
                         "case tokens");
    }
    fclose(fp);
    return cases;
}

static void free_cases(motif3_token_case *cases, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        free(cases[i].name);
        free(cases[i].text);
        free(cases[i].token);
    }
    free(cases);
}

static void assert_tokens(const char *name, const token_vec *got,
                          const int32_t *want, uint64_t n_want) {
    if ((uint64_t)got->len != n_want) {
        fprintf(stderr, "%s token count mismatch: got %d want %" PRIu64 "\n",
                name, got->len, n_want);
        fprintf(stderr, "got:");
        for (int i = 0; i < got->len; i++) fprintf(stderr, " %d", got->v[i]);
        fprintf(stderr, "\nwant:");
        for (uint64_t i = 0; i < n_want; i++) fprintf(stderr, " %d", want[i]);
        fprintf(stderr, "\n");
        exit(1);
    }
    for (uint64_t i = 0; i < n_want; i++) {
        if (got->v[i] != want[i]) {
            fprintf(stderr, "%s token mismatch at %" PRIu64 ": got %d want %d\n",
                    name, i, got->v[i], want[i]);
            exit(1);
        }
    }
}

static const motif3_token_case *find_case(
        const motif3_token_case *cases, uint32_t count, const char *name) {
    for (uint32_t i = 0; i < count; i++)
        if (strcmp(cases[i].name, name) == 0) return &cases[i];
    fprintf(stderr, "tokenizer case not found: %s\n", name);
    exit(1);
}

static void test_builder(const ds4_vocab *vocab, const motif3_token_case *want,
                         const char *system, const char *user,
                         ds4_think_mode think_mode) {
    token_vec got = {0};
    encode_chat_prompt(vocab, system, user, think_mode, &got);
    assert_tokens(want->name, &got, want->token, want->n_token);
    token_vec_free(&got);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <motif3.gguf> <tokenizer-chat.ds4tok>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    ds4_vocab vocab;
    vocab_load(&vocab, &model);
    if (vocab.bos_id != 1 || vocab.eos_id != 0 ||
        vocab.start_of_turn_id != 5 || vocab.end_of_turn_id != 6 ||
        vocab.think_start_id != 11 || vocab.think_end_id != 12 ||
        vocab.tool_call_start_id != 13 || vocab.tool_response_start_id != 15) {
        fprintf(stderr, "Motif-3 special-token ids do not match the official tokenizer\n");
        return 1;
    }

    uint32_t count = 0;
    motif3_token_case *cases = load_cases(argv[2], &count);
    uint32_t raw_count = 0;
    uint32_t chat_count = 0;
    for (uint32_t i = 0; i < count; i++) {
        token_vec got = {0};
        if (cases[i].kind == 0) {
            bpe_tokenize_text(&vocab, cases[i].text, &got);
            raw_count++;
        } else {
            tokenize_rendered_chat_vocab(&vocab, cases[i].text, &got);
            chat_count++;
        }
        assert_tokens(cases[i].name, &got, cases[i].token, cases[i].n_token);
        token_vec_free(&got);
    }

    test_builder(&vocab, find_case(cases, count, "user_thinking"),
                 NULL, "Hello", DS4_THINK_HIGH);
    test_builder(&vocab, find_case(cases, count, "system_user_no_thinking"),
                 "You are precise.", "Hello", DS4_THINK_NONE);

    printf("Motif-3 official-final tokenizer/chat parity: valid "
           "(%u raw + %u rendered fixtures)\n", raw_count, chat_count);
    free_cases(cases, count);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}
