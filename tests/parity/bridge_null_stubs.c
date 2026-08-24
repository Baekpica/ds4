/* Link stubs so tests/parity/bridge_null_oracle can use ds4_bridge.o
 * without the CUDA engine. NULL-handle tests never reach these. */

#include "ds4.h"
#include "native/bridge/ds4_host_load.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUB(name) \
    do { \
        fprintf(stderr, "bridge_null_oracle stub called: %s\n", name); \
        abort(); \
    } while (0)

void ds4_host_tensor_dir_install(const ds4_host_tensor_dir *d) { (void)d; }
void ds4_host_tensor_dir_clear(void) {}
void ds4_host_shape_install(const ds4_host_shape *s) { (void)s; }
void ds4_host_shape_clear(void) {}
void ds4_host_vocab_install(const ds4_host_vocab *v) { (void)v; }
void ds4_host_vocab_clear(void) {}
void ds4_host_bind_map_install(const ds4_host_bind_map *m) { (void)m; }
void ds4_host_bind_map_clear(void) {}
void ds4_host_mtp_bind_map_install(const ds4_host_bind_map *m) { (void)m; }
void ds4_host_mtp_bind_map_clear(void) {}
void ds4_host_dspark_bind_map_install(const ds4_host_bind_map *m) { (void)m; }
void ds4_host_dspark_bind_map_clear(void) {}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    (void)out; (void)opt; STUB("ds4_engine_open");
}
void ds4_engine_close(ds4_engine *e) { (void)e; STUB("ds4_engine_close"); }
int ds4_engine_model_id(ds4_engine *e) { (void)e; STUB("ds4_engine_model_id"); }
int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    (void)out; (void)e; (void)ctx_size; STUB("ds4_session_create");
}
void ds4_session_free(ds4_session *s) { (void)s; STUB("ds4_session_free"); }
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    (void)s; (void)prompt; (void)err; (void)errlen; STUB("ds4_session_sync");
}
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    (void)s; (void)token; (void)err; (void)errlen; STUB("ds4_session_eval");
}
int ds4_session_argmax(ds4_session *s) { (void)s; STUB("ds4_session_argmax"); }
int ds4_session_pos(ds4_session *s) { (void)s; STUB("ds4_session_pos"); }
int ds4_session_ctx(ds4_session *s) { (void)s; STUB("ds4_session_ctx"); }
void ds4_session_rewind(ds4_session *s, int pos) {
    (void)s; (void)pos; STUB("ds4_session_rewind");
}
void ds4_session_invalidate(ds4_session *s) { (void)s; STUB("ds4_session_invalidate"); }
uint64_t ds4_session_generation(const ds4_session *s) {
    (void)s; STUB("ds4_session_generation");
}
int ds4_session_prefill_cap(ds4_session *s) { (void)s; STUB("ds4_session_prefill_cap"); }
int ds4_session_exaone_rewind_span(ds4_session *s) {
    (void)s; STUB("ds4_session_exaone_rewind_span");
}
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p,
                       float min_p, uint64_t *rng) {
    (void)s; (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng;
    STUB("ds4_session_sample");
}
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    (void)s; (void)fp; (void)err; (void)errlen; STUB("ds4_session_save_payload");
}
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes,
                             char *err, size_t errlen) {
    (void)s; (void)fp; (void)payload_bytes; (void)err; (void)errlen;
    STUB("ds4_session_load_payload");
}
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; (void)out; STUB("ds4_tokenize_text");
}
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; (void)out; STUB("ds4_tokenize_rendered_chat");
}
void ds4_tokens_free(ds4_tokens *tv) { (void)tv; STUB("ds4_tokens_free"); }
char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e; (void)token; (void)len; STUB("ds4_token_text");
}
int ds4_token_eos(ds4_engine *e) { (void)e; STUB("ds4_token_eos"); }
bool ds4_token_is_stop(ds4_engine *e, int token) {
    (void)e; (void)token; STUB("ds4_token_is_stop");
}

int ds4_gpu_mem_census_read(int consumer_class, int domain, ds4_mem_cell *out) {
    (void)consumer_class; (void)domain; (void)out;
    return 1;
}
uint64_t ds4_gpu_mem_census_faults(void) { return 0; }
uint64_t ds4_gpu_mem_census_epoch_begin(void) { return 0; }
int ds4_gpu_mem_census_epoch_verify(uint64_t began) { return began == 0; }
int ds4_gpu_mem_observe(ds4_mem_observation *out) {
    if (out) {
        memset(out, 0, sizeof(*out));
        out->status = DS4_MEMOBS_UNSUPPORTED;
        out->source = DS4_MEMOBS_SRC_NONE;
    }
    return 1;
}
uint64_t ds4_gpu_substrate_outstanding(void) { return 0; }
