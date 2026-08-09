/* Solar Open 2 tokenizer/chat-template parity diagnostic.
 *
 *   ./tests/test_solar_tokenizer <vocab.gguf> --text <text>
 *   ./tests/test_solar_tokenizer <vocab.gguf> --chat-think <system> <user>
 *   ./tests/test_solar_tokenizer <vocab.gguf> --chat-nothink <system> <user>
 *
 * Emits one space-separated id stream for direct diffing with the pinned
 * Upstage Transformers tokenizer or llama-tokenize.
 */
#include "../ds4.c"

static void print_ids(const token_vec *tokens) {
    for (int i = 0; i < tokens->len; i++) {
        printf("%s%d", i ? " " : "", tokens->v[i]);
    }
    putchar('\n');
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <vocab.gguf> --text <text> | "
                "--chat-{think,nothink} <system> <user>\n",
                argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_SOLAR_OPEN2) {
        fprintf(stderr, "not a Solar Open 2 GGUF\n");
        model_close(&model);
        return 1;
    }

    ds4_vocab vocab;
    vocab_load(&vocab, &model);
    token_vec tokens = {0};

    if (!strcmp(argv[2], "--text") && argc == 4) {
        bpe_tokenize_text(&vocab, argv[3], &tokens);
    } else if ((!strcmp(argv[2], "--chat-think") ||
                !strcmp(argv[2], "--chat-nothink")) && argc == 5) {
        encode_chat_prompt_solar(&vocab,
                                 argv[3][0] ? argv[3] : NULL,
                                 argv[4],
                                 !strcmp(argv[2], "--chat-think")
                                     ? DS4_THINK_HIGH
                                     : DS4_THINK_NONE,
                                 &tokens);
    } else {
        fprintf(stderr, "invalid tokenizer diagnostic arguments\n");
        vocab_free(&vocab);
        model_close(&model);
        return 2;
    }

    print_ids(&tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}
