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

unsigned bridge_payload_load_calls;
int64_t bridge_payload_load_offset;
uint64_t bridge_payload_load_bytes;
int bridge_routed_quant_bits;
unsigned bridge_boot_prewarm_calls;
int bridge_sync_rc;
unsigned bridge_sync_calls;
unsigned bridge_progress_sets;
unsigned bridge_progress_clears;
int bridge_progress_active;
int bridge_batch_max_seq;
int bridge_bank_committed;
int bridge_bank_tokens[8];
uint64_t bridge_bank_generation;
unsigned bridge_bank_save_calls;
int bridge_bank_save_result;
unsigned bridge_bank_load_calls;
int64_t bridge_bank_load_offset;
uint64_t bridge_bank_load_bytes;

static ds4_session_progress_fn bridge_progress;
static void *bridge_progress_ud;

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
void ds4_engine_boot_prewarm(ds4_engine *e) {
    if (!e) STUB("ds4_engine_boot_prewarm");
    bridge_boot_prewarm_calls++;
}
int ds4_engine_model_id(ds4_engine *e) { (void)e; STUB("ds4_engine_model_id"); }
int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) STUB("ds4_engine_routed_quant_bits");
    return bridge_routed_quant_bits;
}
int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    (void)out; (void)e; (void)ctx_size; STUB("ds4_session_create");
}
void ds4_session_free(ds4_session *s) { (void)s; STUB("ds4_session_free"); }
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    bridge_sync_calls++;
    if (bridge_progress) {
        bridge_progress(bridge_progress_ud, "prefill_display", 1024, prompt->len);
        bridge_progress(bridge_progress_ud, "prefill_chunk", 0, prompt->len);
        bridge_progress(bridge_progress_ud, "prefill_chunk", 4096, prompt->len);
        bridge_progress(bridge_progress_ud, "prefill_chunk", prompt->len, prompt->len);
    }
    return bridge_sync_rc;
}
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    (void)s;
    bridge_progress = fn;
    bridge_progress_ud = ud;
    bridge_progress_active = fn != NULL;
    if (fn) {
        bridge_progress_sets++;
    } else {
        bridge_progress_clears++;
    }
}
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    (void)s; (void)token; (void)err; (void)errlen; STUB("ds4_session_eval");
}
int ds4_session_argmax(ds4_session *s) { (void)s; STUB("ds4_session_argmax"); }
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    (void)s; (void)excluded_id; STUB("ds4_session_argmax_excluding");
}
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
    (void)s; (void)err; (void)errlen;
    bridge_payload_load_calls++;
    bridge_payload_load_offset = (int64_t)ftello(fp);
    bridge_payload_load_bytes = payload_bytes;
    return 0;
}
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    (void)s; (void)snap; (void)err; (void)errlen;
    STUB("ds4_session_save_snapshot");
}
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    (void)s; (void)snap; (void)err; (void)errlen;
    STUB("ds4_session_load_snapshot");
}
void ds4_session_snapshot_free(ds4_session_snapshot *snap) { (void)snap; }
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; (void)out; STUB("ds4_tokenize_text");
}
void ds4_encode_chat_prompt(ds4_engine *e, const char *system, const char *prompt,
                            ds4_think_mode think_mode, ds4_tokens *out) {
    (void)e; (void)system; (void)prompt; (void)think_mode; (void)out;
    STUB("ds4_encode_chat_prompt");
}
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    (void)s; (void)out; (void)k; STUB("ds4_session_top_logprobs");
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

int ds4_batch_ctx_create_fit(ds4_engine *e, int ctx_size, int max_seq,
                             int max_total_tokens, ds4_batch_ctx **out,
                             char *err, size_t errlen) {
    (void)e; (void)ctx_size; (void)max_seq; (void)max_total_tokens;
    (void)out; (void)err; (void)errlen;
    STUB("ds4_batch_ctx_create_fit");
}
void ds4_batch_ctx_destroy(ds4_batch_ctx *ctx) { (void)ctx; }
int ds4_batch_ctx_max_seq(const ds4_batch_ctx *ctx) {
    (void)ctx;
    return bridge_batch_max_seq;
}
int ds4_batch_ctx_seq_cap(const ds4_batch_ctx *ctx) { (void)ctx; return 0; }
int ds4_batch_ctx_bank_committed(const ds4_batch_ctx *ctx, int bank,
                                 const int **tokens) {
    (void)ctx;
    if (bank < 0 || bank >= bridge_batch_max_seq) {
        if (tokens) *tokens = NULL;
        return 0;
    }
    if (tokens) {
        *tokens = bridge_bank_committed > 0 ? bridge_bank_tokens : NULL;
    }
    return bridge_bank_committed;
}
uint64_t ds4_batch_ctx_bank_generation(const ds4_batch_ctx *ctx, int bank) {
    (void)ctx;
    return bank >= 0 && bank < bridge_batch_max_seq ? bridge_bank_generation : 0;
}
int ds4_cont_bank_save_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp,
                               char *err, size_t errlen) {
    (void)ctx; (void)bank;
    bridge_bank_save_calls++;
    if (bridge_bank_save_result != 0) {
        if (err && errlen) snprintf(err, errlen, "bank save failed");
        return bridge_bank_save_result;
    }
    return fwrite("BANK", 1, 4, fp) == 4 ? 0 : 1;
}
int ds4_cont_bank_restore_payload(ds4_batch_ctx *ctx, uint32_t bank, FILE *fp,
                                  uint64_t payload_bytes,
                                  char *err, size_t errlen) {
    (void)ctx; (void)bank; (void)err; (void)errlen;
    bridge_bank_load_calls++;
    bridge_bank_load_offset = (int64_t)ftello(fp);
    bridge_bank_load_bytes = payload_bytes;
    return 0;
}
int ds4_engine_continuous_generate(ds4_batch_ctx *ctx,
                                   int (*admit)(void *ud, ds4_cont_request *req),
                                   int (*on_token)(void *ud, void *user, int token),
                                   void (*on_done)(void *ud, void *user,
                                                   const int *tokens, int n, int finish),
                                   void *ud, char *err, size_t errlen) {
    (void)ctx; (void)admit; (void)on_token; (void)on_done; (void)ud;
    (void)err; (void)errlen;
    STUB("ds4_engine_continuous_generate");
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
