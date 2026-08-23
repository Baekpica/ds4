/* Link stubs so tests/parity/kv_c_oracle can use ds4_kvstore.o without
 * the CUDA engine.  The oracle only calls format/policy entry points;
 * these abort if a store-live path is reached by mistake. */

#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>

#define STUB(name) \
    do { \
        fprintf(stderr, "kv_c_oracle stub called: %s\n", name); \
        abort(); \
    } while (0)

int ds4_engine_model_id(ds4_engine *e) { (void)e; STUB("ds4_engine_model_id"); }
int ds4_engine_routed_quant_bits(ds4_engine *e) { (void)e; STUB("ds4_engine_routed_quant_bits"); }
int ds4_session_ctx(ds4_session *s) { (void)s; STUB("ds4_session_ctx"); }
void ds4_session_invalidate(ds4_session *s) { (void)s; STUB("ds4_session_invalidate"); }
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes,
                             char *err, size_t errlen) {
    (void)s; (void)fp; (void)payload_bytes; (void)err; (void)errlen;
    STUB("ds4_session_load_payload");
}
void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    (void)payload;
    STUB("ds4_session_payload_file_free");
}
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    (void)s; (void)out; (void)err; (void)errlen;
    STUB("ds4_session_stage_payload");
}
const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    (void)s;
    STUB("ds4_session_tokens");
}
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    (void)payload; (void)fp; (void)err; (void)errlen;
    STUB("ds4_session_write_staged_payload");
}
char *ds4_token_text(ds4_engine *e, int token, size_t *out_len) {
    (void)e; (void)token; (void)out_len;
    STUB("ds4_token_text");
}
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; (void)out;
    STUB("ds4_tokenize_rendered_chat");
}
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    (void)dst; (void)src;
    STUB("ds4_tokens_copy");
}
void ds4_tokens_free(ds4_tokens *tv) { (void)tv; STUB("ds4_tokens_free"); }
void ds4_tokens_push(ds4_tokens *tv, int token) {
    (void)tv; (void)token;
    STUB("ds4_tokens_push");
}
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    (void)tokens; (void)prefix;
    STUB("ds4_tokens_starts_with");
}
int ds4_cont_bank_restore_payload(ds4_batch_ctx *ctx, uint32_t bank,
                                  FILE *fp, uint64_t payload_bytes,
                                  char *err, size_t errlen) {
    (void)ctx; (void)bank; (void)fp; (void)payload_bytes; (void)err; (void)errlen;
    STUB("ds4_cont_bank_restore_payload");
}
int ds4_batch_ctx_bank_committed(const ds4_batch_ctx *ctx, int bank,
                                 const int **toks) {
    (void)ctx; (void)bank; (void)toks;
    STUB("ds4_batch_ctx_bank_committed");
}
