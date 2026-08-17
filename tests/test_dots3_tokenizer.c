/* dots3-note-prev tokenizer parity against the official HF tokenizer.
 *
 *   ./tests/test_dots3_tokenizer <dots3 first shard or merged gguf>
 *
 * Golden ids in dots3_tokenizer_goldens.inc were produced by the HF
 * `tokenizers` runtime on the official tokenizer.json whose SHA-256 the GGUF
 * pins as dots3-note.source.tokenizer_sha256, so passing here means byte
 * parity with the reference implementation for these cases.  Inputs are NFC;
 * like the other byte-BPE families this engine does not re-normalize. */
#include "../ds4.c"

#include "dots3_tokenizer_goldens.inc"

static int check_ids(const char *name, const token_vec *got,
                     const int *want, size_t want_len) {
    bool ok = got->len == (int)want_len;
    for (int i = 0; ok && i < got->len; i++) ok = got->v[i] == want[i];
    if (ok) return 0;
    fprintf(stderr, "FAIL %s: got %d tokens [", name, got->len);
    for (int i = 0; i < got->len; i++)
        fprintf(stderr, "%s%d", i ? ", " : "", got->v[i]);
    fprintf(stderr, "], want %zu [", want_len);
    for (size_t i = 0; i < want_len; i++)
        fprintf(stderr, "%s%d", i ? ", " : "", want[i]);
    fprintf(stderr, "]\n");
    return 1;
}

#define CHECK_TEXT(name) do { \
        token_vec got = {0}; \
        bpe_tokenize_text(&vocab, name##_text, &got); \
        failures += check_ids(#name, &got, name##_ids, \
                              sizeof(name##_ids) / sizeof(name##_ids[0])); \
        free(got.v); \
    } while (0)

#define CHECK_CHAT(name) do { \
        token_vec got = {0}; \
        tokenize_rendered_chat_vocab(&vocab, name##_text, &got); \
        failures += check_ids(#name, &got, name##_ids, \
                              sizeof(name##_ids) / sizeof(name##_ids[0])); \
        free(got.v); \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <dots3.gguf>\n", argv[0]);
        return 2;
    }

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DOTS3_NOTE) {
        fprintf(stderr, "not a dots3-note GGUF\n");
        return 1;
    }

    ds4_vocab vocab;
    vocab_load(&vocab, &model);

    int failures = 0;

    /* The official special-token ids are load-bearing for the server. */
    if (vocab.bos_id != 151643 || vocab.eos_id != 151668 ||
        vocab.dots3_endoftext_id != 151643 ||
        vocab.system_id != 151650 || vocab.dots3_endofsystem_id != 151651 ||
        vocab.user_id != 151665 || vocab.dots3_endofuser_id != 151666 ||
        vocab.assistant_id != 151667 ||
        vocab.think_start_id != 151721 || vocab.think_end_id != 151722 ||
        vocab.tool_call_start_id != 151724 || vocab.tool_call_end_id != 151725 ||
        vocab.tool_response_start_id != 151726 ||
        vocab.tool_response_end_id != 151727) {
        fprintf(stderr, "dots3-note special-token ids are wrong\n");
        failures++;
    }
    if (!vocab_token_is_generation_stop(&vocab, 151668) ||
        !vocab_token_is_generation_stop(&vocab, 151643) ||
        vocab_token_is_generation_stop(&vocab, 151667)) {
        fprintf(stderr, "dots3-note generation stop set is wrong\n");
        failures++;
    }

    CHECK_TEXT(ascii_basic);
    CHECK_TEXT(contractions);
    CHECK_TEXT(digits);
    CHECK_TEXT(korean);
    CHECK_TEXT(code_indent);
    CHECK_TEXT(space_runs);
    CHECK_TEXT(url);
    CHECK_TEXT(emoji);
    CHECK_TEXT(mixed);
    CHECK_TEXT(think_markers);
    CHECK_TEXT(newline_runs);
    CHECK_TEXT(cjk_punct);
    CHECK_CHAT(chat_thinking);
    CHECK_CHAT(chat_no_think);
    CHECK_CHAT(chat_tool_response);

    /* The token-level chat builder must agree byte-for-byte with the
     * template-rendered goldens for the same message sets. */
    {
        token_vec got = {0};
        encode_chat_prompt(&vocab, "You are a helpful assistant.",
                           "\354\225\210\353\205\225! \354\235\264 \353\254\270\354\236\245\354\235\204 "
                           "\355\206\240\355\201\260\355\231\224\355\225\264 \354\244\230.",
                           DS4_THINK_HIGH, &got);
        failures += check_ids("builder_thinking", &got, chat_thinking_ids,
                              sizeof(chat_thinking_ids) / sizeof(chat_thinking_ids[0]));
        free(got.v);
    }
    {
        token_vec got = {0};
        encode_chat_prompt(&vocab, "SYS", "hi", DS4_THINK_NONE, &got);
        failures += check_ids("builder_no_think", &got, chat_no_think_ids,
                              sizeof(chat_no_think_ids) / sizeof(chat_no_think_ids[0]));
        free(got.v);
    }

    /* Detokenization round-trip for plain text. */
    {
        token_vec got = {0};
        bpe_tokenize_text(&vocab, korean_text, &got);
        char buf[1024];
        size_t off = 0;
        for (int i = 0; i < got.len; i++) {
            size_t n = 0;
            char *piece = vocab_token_text(&vocab, got.v[i], &n);
            if (off + n < sizeof(buf)) {
                memcpy(buf + off, piece, n);
                off += n;
            }
            free(piece);
        }
        buf[off] = '\0';
        if (strcmp(buf, korean_text) != 0) {
            fprintf(stderr, "FAIL detok round-trip: %s\n", buf);
            failures++;
        }
        free(got.v);
    }

    if (failures) {
        fprintf(stderr, "dots3-note tokenizer: %d failure(s)\n", failures);
        model_close(&model);
        return 1;
    }
    printf("dots3-note tokenizer: all 15 golden cases, builder parity, "
           "stop set, and round-trip passed\n");
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}
