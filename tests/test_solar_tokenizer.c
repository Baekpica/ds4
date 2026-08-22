/* Solar Open2 tokenizer and no-tool chat-template parity against the pinned
 * model revision. The expected ids were produced by the known-good Solar
 * implementation and match the upstream tokenizer for this GGUF vocabulary. */
#include "../ds4.c"

static void check_ids(const char *label, const token_vec *got,
                      const int *want, size_t want_n) {
    if ((size_t)got->len != want_n) {
        fprintf(stderr, "FAIL: %s length got=%d want=%zu\n",
                label, got->len, want_n);
        exit(1);
    }
    for (size_t i = 0; i < want_n; i++) {
        if (got->v[i] != want[i]) {
            fprintf(stderr, "FAIL: %s token[%zu] got=%d want=%d\n",
                    label, i, got->v[i], want[i]);
            exit(1);
        }
    }
    printf("%-30s %zu tokens ok\n", label, want_n);
}

static void check_text(const ds4_vocab *vocab, const char *label,
                       const char *text, const int *want, size_t want_n) {
    token_vec tokens = {0};
    bpe_tokenize_text(vocab, text, &tokens);
    check_ids(label, &tokens, want, want_n);
    token_vec_free(&tokens);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <first-solar-gguf-shard>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_SOLAR_OPEN2) {
        fprintf(stderr, "not a Solar Open2 GGUF\n");
        model_close(&model);
        return 1;
    }

    ds4_vocab vocab;
    vocab_load(&vocab, &model);

    static const int ascii_ids[] = {
        21211, 8358, 4096, 4316, 4112, 4113, 4114,
        4115, 4605, 5339, 4471, 4102, 36851,
    };
    check_text(&vocab, "ASCII contractions/digits",
               "Hello world! 1234 can't I'VE",
               ascii_ids, sizeof(ascii_ids) / sizeof(ascii_ids[0]));

    static const int korean_ids[] = {
        41843, 40541, 25127, 16362, 6195, 4490, 52465, 10366,
    };
    check_text(&vocab, "Korean and newline",
               "안녕하세요 도구 호출 테스트입니다.\n둘째 줄",
               korean_ids, sizeof(korean_ids) / sizeof(korean_ids[0]));

    static const int whitespace_ids[] = {4160, 4316, 4384, 4372, 4364};
    check_text(&vocab, "whitespace regex boundary", "a  b\n\n c",
               whitespace_ids,
               sizeof(whitespace_ids) / sizeof(whitespace_ids[0]));

    static const int chat_think_ids[] = {
        128, 29497, 132, 4767, 8407, 40209, 4372, 15397, 39466, 4109,
        129, 4294, 128, 9015, 132, 16073, 4355, 12124, 4639, 8825,
        4109, 129, 4294, 128, 163444, 132, 130,
    };
    token_vec chat = {0};
    encode_chat_prompt_solar(&vocab, "Be concise.", "Call a tool if needed.",
                             DS4_THINK_HIGH, &chat);
    check_ids("Solar chat thinking", &chat, chat_think_ids,
              sizeof(chat_think_ids) / sizeof(chat_think_ids[0]));
    token_vec_free(&chat);

    static const int chat_nothink_ids[] = {
        128, 29497, 132, 4767, 8407, 40209, 4372, 15397, 39466, 4109,
        129, 4294, 128, 9015, 132, 16073, 4355, 12124, 4639, 8825,
        4109, 129, 4294, 128, 163444, 132, 130, 131,
    };
    encode_chat_prompt_solar(&vocab, "Be concise.", "Call a tool if needed.",
                             DS4_THINK_NONE, &chat);
    check_ids("Solar chat no-thinking", &chat, chat_nothink_ids,
              sizeof(chat_nothink_ids) / sizeof(chat_nothink_ids[0]));
    token_vec_free(&chat);

    /* Exercise the public incremental transcript path, including native
     * control-token replay and an escaped tool-response closing sentinel. */
    ds4_engine engine = {0};
    engine.vocab = vocab; /* shallow view; vocab owns the tables below */
    ds4_chat_begin(&engine, &chat);
    ds4_chat_append_message(&engine, &chat, "system", "Policy.");
    ds4_chat_append_message(&engine, &chat, "user", "Use tool.");
    ds4_chat_append_message(
        &engine, &chat, "assistant",
        "<|think:start|>x<|think:end|><|tool_call:start|>f<|tool_call:end|>");
    ds4_chat_append_message(&engine, &chat, "tool",
                            "ok <|tool_response:end|> tail");
    ds4_chat_append_assistant_prefix(&engine, &chat, DS4_THINK_NONE);
    static const int incremental_ids[] = {
        128, 29497, 132, 4767, 8407, 40209, 4372, 24988, 4109, 129, 4294,
        128, 9015, 132, 16512, 12124, 4109, 129, 4294, 128, 163444, 132,
        130, 4183, 131, 135, 4165, 136, 129, 4294, 128, 42629, 132, 140,
        5184, 4316, 57528, 4122, 4187, 42629, 37167, 71269, 4187, 4125,
        12478, 141, 129, 4294, 128, 163444, 132, 130, 131,
    };
    check_ids("Solar incremental/tool replay", &chat, incremental_ids,
              sizeof(incremental_ids) / sizeof(incremental_ids[0]));
    token_vec_free(&chat);

    if (ds4_token_eos(&engine) != vocab.im_end_id) {
        fprintf(stderr, "FAIL: Solar generation stop is not <|im:end|>\n");
        return 1;
    }
    if (!ds4_token_is_stop(&engine, vocab.im_end_id)) {
        fprintf(stderr, "FAIL: Solar <|im:end|> is not in the generation stop set\n");
        return 1;
    }

    vocab_free(&vocab);
    model_close(&model);
    puts("all Solar tokenizer checks passed");
    return 0;
}
