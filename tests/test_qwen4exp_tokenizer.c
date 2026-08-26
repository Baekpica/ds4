/* Qwen3.8 tokenizer/chat parity against the pinned official tokenizer.json. */
#include "../ds4.c"

static int failures;

static void check_ids(const char *name, const token_vec *got,
                      const int *want, size_t want_len) {
    if (got->len == (int)want_len) {
        size_t i = 0;
        while (i < want_len && got->v[i] == want[i]) i++;
        if (i == want_len) return;
    }
    fprintf(stderr, "FAIL %s: got [", name);
    for (int i = 0; i < got->len; i++)
        fprintf(stderr, "%s%d", i ? ", " : "", got->v[i]);
    fprintf(stderr, "]\n");
    failures++;
}

static void check_text(const ds4_vocab *vocab, const char *name,
                       const char *text, const int *want, size_t want_len) {
    token_vec got = {0};
    bpe_tokenize_text(vocab, text, &got);
    check_ids(name, &got, want, want_len);
    token_vec_free(&got);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s QWEN4EXP_GGUF\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_QWEN4EXP) return 1;

    ds4_vocab vocab;
    vocab_load(&vocab, &model);
    if (vocab.bos_id != 248044 || vocab.eos_id != 248046 ||
        vocab.eot_id != 248044 || vocab.im_start_id != 248045 ||
        vocab.im_end_id != 248046 || vocab.think_start_id != 248068 ||
        vocab.think_end_id != 248069 ||
        vocab.tool_call_start_id != 248058 ||
        vocab.tool_call_end_id != 248059 ||
        vocab.tool_response_start_id != 248066 ||
        vocab.tool_response_end_id != 248067) {
        fprintf(stderr, "FAIL Qwen special-token ids\n");
        failures++;
    }

    static const int ascii[] = {
        9419, 1814, 0, 220, 16, 17, 18, 19, 628, 914, 353, 6, 4343,
    };
    check_text(&vocab, "ASCII", "Hello world! 1234 can't I'VE",
               ascii, sizeof(ascii) / sizeof(ascii[0]));
    static const int korean[] = {
        148924, 154982, 88005, 212283, 92198, 187792,
        151396, 13, 198, 169070, 80780, 159490,
    };
    check_text(&vocab, "Korean", "안녕하세요 도구 호출 테스트입니다.\n둘째 줄",
               korean, sizeof(korean) / sizeof(korean[0]));
    static const int markers[] = {220, 248068, 15131, 220, 248069};
    check_text(&vocab, "added markers", " <think> hi </think>",
               markers, sizeof(markers) / sizeof(markers[0]));

    static const int chat_thinking[] = {
        248045, 8678, 198, 3320, 61446, 13, 248046, 198,
        248045, 846, 198, 6994, 264, 5224, 413, 4221, 13, 248046, 198,
        248045, 74455, 198, 248068, 198,
    };
    token_vec chat = {0};
    encode_chat_prompt(&vocab, "Be concise.", "Call a tool if needed.",
                       DS4_THINK_LOW, &chat);
    check_ids("chat thinking", &chat, chat_thinking,
              sizeof(chat_thinking) / sizeof(chat_thinking[0]));
    token_vec_free(&chat);

    static const int chat_no_thinking[] = {
        248045, 8678, 198, 3320, 61446, 13, 248046, 198,
        248045, 846, 198, 6994, 264, 5224, 413, 4221, 13, 248046, 198,
        248045, 74455, 198, 248068, 271, 248069, 271,
    };
    encode_chat_prompt(&vocab, "Be concise.", "Call a tool if needed.",
                       DS4_THINK_NONE, &chat);
    check_ids("chat no thinking", &chat, chat_no_thinking,
              sizeof(chat_no_thinking) / sizeof(chat_no_thinking[0]));
    token_vec_free(&chat);

    ds4_engine engine = {0};
    engine.vocab = vocab;
    if (!vocab_token_is_generation_stop(&vocab, 248044) ||
        !vocab_token_is_generation_stop(&vocab, 248046) ||
        vocab_token_is_generation_stop(&vocab, 248045) ||
        ds4_token_eos(&engine) != 248046) {
        fprintf(stderr, "FAIL Qwen generation stop set\n");
        failures++;
    }

    vocab_free(&vocab);
    model_close(&model);
    if (failures) return 1;
    puts("Qwen3.8 tokenizer: official text/chat goldens and stop set passed");
    return 0;
}
