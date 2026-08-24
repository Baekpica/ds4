#ifndef DS4_BRIDGE_H
#define DS4_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "ds4_host_load.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Narrow Rust ↔ native ABI.  Do not include ds4.h from Rust.  Handles are
 * opaque; the C structs below may contain ds4_engine / ds4_session pointers
 * but those layouts are not part of this contract. */

typedef struct ds4_bridge_model ds4_bridge_model;
typedef struct ds4_bridge_session ds4_bridge_session;
typedef struct ds4_bridge_snapshot ds4_bridge_snapshot;

enum {
    DS4_BRIDGE_BACKEND_CUDA = 0,
    DS4_BRIDGE_BACKEND_METAL = 1,
    DS4_BRIDGE_BACKEND_CPU = 2
};

#define DS4_BRIDGE_MAX_DIMS 8

typedef struct {
    const char *name;         /* borrowed for the call */
    uint32_t required;        /* 1 = weights_bind required_tensor */
    uint32_t ndim;
    uint64_t dim[DS4_BRIDGE_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t bytes;
    uint32_t shard;
    uint32_t found;
} ds4_bridge_bind_slot;

typedef struct {
    const char *path;         /* borrowed for the call */
    uint64_t size;
    uint64_t base;
} ds4_bridge_shard;

/* Host tensor inventory + weights_bind name table.  Native bind consumes
 * this plan: check before ds4_engine_open, match after C parse when an
 * engine is open. */
typedef struct {
    uint32_t n_slots;
    const ds4_bridge_bind_slot *slots;
    uint32_t n_shards;
    const ds4_bridge_shard *shards;
    uint64_t data_pos;
    uint64_t alignment;
    uint64_t page;
} ds4_bridge_bind_plan;

typedef struct {
    const char *model_path;
    int backend;              /* DS4_BRIDGE_BACKEND_* */
    int n_threads;
    int defer_boot_prewarm;   /* nonzero => skip boot prewarm inside open */
    const ds4_bridge_bind_plan *plan; /* optional; borrowed for the call */
    const ds4_host_tensor_dir *tensors; /* optional full inventory; borrowed */
    const ds4_host_shape *shape; /* optional; skip C validate when set */
    const ds4_host_vocab *vocab; /* optional; skip C vocab_load when set */
    const ds4_host_bind_map *bind; /* optional; skip C name walk when set */
    /* DeepSeek-only sibling support models.  Paths open through the same
     * native model_open; the optional maps are host-resolved name->index
     * tables for THAT sibling's tensor dir, and when installed native
     * skips that sibling's C layout check. */
    const char *mtp_path;                 /* optional; borrowed */
    const char *dspark_path;              /* optional; borrowed */
    const ds4_host_bind_map *mtp_bind;    /* optional; borrowed */
    const ds4_host_bind_map *dspark_bind; /* optional; borrowed */
} ds4_bridge_model_open_options;

/* All functions: 0 on success, nonzero on failure.  err is optional; when
 * provided it is NUL-terminated on failure.  Token pointers are borrowed
 * for the duration of the call only. */

int ds4_bridge_bind_plan_check(const ds4_bridge_bind_plan *plan,
                               char *err, size_t errlen);
int ds4_bridge_bind_plan_match(const ds4_bridge_bind_plan *host,
                               const ds4_bridge_bind_plan *native,
                               char *err, size_t errlen);

int ds4_bridge_model_open(ds4_bridge_model **out,
                          const ds4_bridge_model_open_options *opt,
                          char *err, size_t errlen);
void ds4_bridge_model_free(ds4_bridge_model *m);

int ds4_bridge_session_create(ds4_bridge_session **out,
                              ds4_bridge_model *m,
                              int ctx_size,
                              char *err, size_t errlen);
void ds4_bridge_session_free(ds4_bridge_session *s);

int ds4_bridge_session_sync(ds4_bridge_session *s,
                            const int32_t *tokens, int n_tokens,
                            char *err, size_t errlen);
int ds4_bridge_eval(ds4_bridge_session *s, int32_t token,
                    char *err, size_t errlen);
int ds4_bridge_session_argmax(ds4_bridge_session *s);
int ds4_bridge_session_pos(ds4_bridge_session *s);
int ds4_bridge_session_ctx(ds4_bridge_session *s);
void ds4_bridge_session_rewind(ds4_bridge_session *s, int pos);
void ds4_bridge_session_invalidate(ds4_bridge_session *s);
uint64_t ds4_bridge_session_generation(ds4_bridge_session *s);
int ds4_bridge_session_prefill_cap(ds4_bridge_session *s);
int ds4_bridge_session_exaone_rewind_span(ds4_bridge_session *s);
int ds4_bridge_session_sample(ds4_bridge_session *s,
                              float temperature, int top_k, float top_p, float min_p,
                              uint64_t *rng);
int ds4_bridge_session_save_payload(ds4_bridge_session *s, const char *path,
                                    char *err, size_t errlen);
int ds4_bridge_session_load_payload(ds4_bridge_session *s, const char *path,
                                    char *err, size_t errlen);

int ds4_bridge_snapshot_create(ds4_bridge_snapshot **out,
                               char *err, size_t errlen);
void ds4_bridge_snapshot_free(ds4_bridge_snapshot *snap);
uint64_t ds4_bridge_snapshot_len(const ds4_bridge_snapshot *snap);
int ds4_bridge_session_save_snapshot(ds4_bridge_session *s,
                                     ds4_bridge_snapshot *snap,
                                     char *err, size_t errlen);
int ds4_bridge_session_load_snapshot(ds4_bridge_session *s,
                                     const ds4_bridge_snapshot *snap,
                                     char *err, size_t errlen);

/* Caller-owned token / text buffers.  n_out is always written on a
 * successful length discovery, including the "buffer too small" error. */
int ds4_bridge_tokenize_text(ds4_bridge_model *m, const char *text,
                             int32_t *out, int cap, int *n_out,
                             char *err, size_t errlen);
int ds4_bridge_tokenize_rendered_chat(ds4_bridge_model *m, const char *text,
                                      int32_t *out, int cap, int *n_out,
                                      char *err, size_t errlen);
int ds4_bridge_token_text(ds4_bridge_model *m, int32_t token,
                          char *out, size_t cap, size_t *n_out,
                          char *err, size_t errlen);
int ds4_bridge_token_eos(ds4_bridge_model *m);
int ds4_bridge_token_is_stop(ds4_bridge_model *m, int32_t token);
int ds4_bridge_model_id(ds4_bridge_model *m);

/* CLI chat-template encode (ds4_encode_chat_prompt): system may be NULL,
 * think_mode is ds4_think_mode 0..3.  Same buffer contract as tokenize. */
int ds4_bridge_encode_chat_prompt(ds4_bridge_model *m, const char *system,
                                  const char *prompt, int think_mode,
                                  int32_t *out, int cap, int *n_out,
                                  char *err, size_t errlen);

/* Post-prefill distribution head (proof harness --dump-logprobs).
 * Copies up to k entries; returns the count, -1 on a NULL session. */
typedef struct {
    int32_t id;
    float logit;
    float logprob;
} ds4_bridge_token_score;

int ds4_bridge_session_top_logprobs(ds4_bridge_session *s,
                                    ds4_bridge_token_score *out, int k);

/* Live CUDA memgov census.  Process-global after backend init; no model
 * handle.  Counts match ds4_mem_census.h (DS4_MEMC__COUNT x DS4_MEMD__COUNT).
 * supported=0 means the backend keeps no census (Metal/CPU/stubs): porcelain
 * renders ABSENCE, never a zero family. */
#define DS4_BRIDGE_MEMC_COUNT 17
#define DS4_BRIDGE_MEMD_COUNT 2

typedef struct {
    uint64_t requested;
    uint64_t committed;
    uint64_t freed_requested;
    uint64_t freed_committed;
    uint64_t alloc_calls;
    uint64_t free_calls;
} ds4_bridge_mem_cell;

typedef struct {
    int32_t supported;          /* 1 = coherent CUDA census image */
    uint64_t faults;
    uint64_t epoch;
    uint64_t torn_fallbacks;
    ds4_bridge_mem_cell cells[DS4_BRIDGE_MEMC_COUNT][DS4_BRIDGE_MEMD_COUNT];
} ds4_bridge_mem_census;

typedef struct {
    int32_t status;             /* 0 ok, 1 unsupported, 2 query_error */
    int32_t source;             /* 0 none, 1 cuda_free, 2 meminfo_available */
    uint64_t free_bytes;
    uint64_t total_bytes;
    uint64_t cuda_free_bytes;
    uint64_t meminfo_avail_bytes;
} ds4_bridge_mem_observe;

int ds4_bridge_mem_census_snap(ds4_bridge_mem_census *out);
int ds4_bridge_mem_observe_snap(ds4_bridge_mem_observe *out);
uint64_t ds4_bridge_mem_substrate_outstanding(void);

/* Continuous batching (mid-flight admit/evict) over a persistent batch
 * context.  Mirrors ds4_batch_ctx / ds4_engine_continuous_generate with
 * a narrow request struct; the engine's rolling scheduler stays native.
 * All callbacks run on the calling thread.  `user` is an opaque
 * per-request handle echoed back verbatim; `ud` is the caller context
 * given to ds4_bridge_continuous_generate. */
typedef struct ds4_bridge_batch_ctx ds4_bridge_batch_ctx;

int ds4_bridge_batch_ctx_create_fit(ds4_bridge_model *m, int ctx_size,
                                    int max_seq, int max_total_tokens,
                                    ds4_bridge_batch_ctx **out,
                                    char *err, size_t errlen);
void ds4_bridge_batch_ctx_destroy(ds4_bridge_batch_ctx *c);
int ds4_bridge_batch_ctx_max_seq(ds4_bridge_batch_ctx *c);
int ds4_bridge_batch_ctx_seq_cap(ds4_bridge_batch_ctx *c);

typedef struct {
    const int32_t *tokens;  /* caller-owned; keep alive until on_done */
    int32_t n;
    int32_t max_new;
    int32_t eos;            /* < 0 => engine default */
    void *user;
    float temperature;      /* <= 0 => greedy argmax */
    int32_t top_k;
    float top_p;
    float min_p;
    uint64_t seed;
    /* Optional (NULL disables).  Same contracts as ds4_cont_request:
     * sample_override returns DS4_SAMPLE_OVERRIDE_* encoding; alive
     * returns 0 to abandon a pending admission; on_admitted returns 0
     * to cancel before prefill (n_cached + n_computed == n). */
    int (*sample_override)(void *ud, void *user);
    int (*alive)(void *ud, void *user);
    int (*on_admitted)(void *ud, void *user, int n_cached, int n_computed,
                       int bank);
    int32_t place_bank;     /* bank id + 1; 0 = engine's choice */
    int32_t n_cached;       /* committed prefix length; 0 = cold */
    int32_t *bank_used;     /* OUT (optional): placed bank id */
    int32_t fork_bank;      /* source bank id + 1; 0 = no fork */
} ds4_bridge_cont_request;

int ds4_bridge_continuous_generate(
    ds4_bridge_batch_ctx *c,
    int (*admit)(void *ud, ds4_bridge_cont_request *req),
    int (*on_token)(void *ud, void *user, int32_t token),
    void (*on_done)(void *ud, void *user, const int32_t *tokens, int32_t n,
                    int32_t finish),
    void *ud, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
