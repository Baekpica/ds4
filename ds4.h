#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ds4_backend backend;
    int n_threads;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    bool warm_weights;
    bool quality;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);

/* Phase 2 W1: batched greedy generation.  Ragged-prefills `n` prompts in one
 * forward, then batch-decodes all sequences with compact-on-finish.  Each
 * out[i].tokens is a malloc'd stream the CALLER must free(); out[i].finish is 1
 * when the sequence hit EOS, 0 when it hit the token budget. */
typedef struct {
    int max_new_tokens;   /* per-sequence decode budget (>=1) */
    int eos_id;           /* sequence-ending token; <0 => engine default */
} ds4_batch_gen_options;

typedef struct {
    int *tokens;          /* malloc'd generated tokens; caller frees */
    int  n_tokens;        /* count generated (<= max_new_tokens) */
    int  finish;          /* 1 = hit EOS, 0 = hit budget */
} ds4_batch_gen_result;

int ds4_engine_batched_generate(ds4_engine *e, const ds4_tokens *prompts, int n,
                                int ctx_size, const ds4_batch_gen_options *opts,
                                ds4_batch_gen_result *out,
                                char *err, size_t errlen);
/* Per-sequence variant: max_new_tokens[i]/eos_ids[i] are length-n arrays
 * (max_new_tokens entry <=0 => 1; eos_ids may be NULL or an entry <0 => engine
 * default EOS).  Used by the server's request-coalescing path. */
int ds4_engine_batched_generate_ex(ds4_engine *e, const ds4_tokens *prompts, int n,
                                   int ctx_size,
                                   const int *max_new_tokens, const int *eos_ids,
                                   ds4_batch_gen_result *out,
                                   char *err, size_t errlen);

/* Phase 2 W4: persistent batched-generation context.  Allocates the graph + N KV
 * bank slabs ONCE (sized for up to max_seq sequences and max_total_tokens packed
 * prompt tokens at the given ctx_size) and reuses them across batches, removing
 * the per-batch graph/slab alloc from the server's hot path.  Opaque handle. */
typedef struct ds4_batch_ctx ds4_batch_ctx;
int  ds4_batch_ctx_create(ds4_engine *e, int ctx_size, int max_seq, int max_total_tokens,
                          ds4_batch_ctx **out, char *err, size_t errlen);
void ds4_batch_ctx_destroy(ds4_batch_ctx *ctx);
/* Per-sequence prompt+budget bound of the persistent ctx (the SWA raw ring): a
 * single sequence's prompt length must be <= this for the batch/continuous path
 * (no ring wrap).  Returns 0 if ctx is NULL. */
int  ds4_batch_ctx_raw_cap(const ds4_batch_ctx *ctx);
/* Batched generation over a persistent context (W4); same semantics as
 * ds4_engine_batched_generate_ex but reuses ctx's graph + slabs.  n <= ctx max_seq,
 * Σ(prompt len) <= ctx max_total_tokens. Returns 0 on success. */
int  ds4_engine_batched_generate_ctx(ds4_batch_ctx *ctx, const ds4_tokens *prompts, int n,
                                     const int *max_new_tokens, const int *eos_ids,
                                     ds4_batch_gen_result *out, char *err, size_t errlen);

/* Phase 2 W5: continuous batching (mid-flight admit/evict) over a persistent ctx.
 * The scheduler maintains a rolling active set of up to ctx max_seq sequences: each
 * step it admits waiting requests into freed KV banks (ragged-prefill the prompt)
 * and evicts finished ones, so short requests don't wait for long ones.  CUDA
 * backend only (the Metal path ignores per-seq bank ids).
 * W7: per-sequence sampling -- each request carries its own temperature/top-k/
 * top-p/min-p/seed, sampled with an independent RNG stream so concurrent rows in
 * one batch do not perturb each other.  A zeroed sampling block (temperature<=0)
 * is greedy argmax, bit-identical to the W5/W6 default. */
typedef struct {
    const int *tokens;   /* prompt tokens (caller-owned; must outlive the admit) */
    int        n;        /* prompt length (>0, <= ctx raw_cap) */
    int        max_new;  /* per-seq decode budget (>=1) */
    int        eos;      /* per-seq EOS token; <0 => engine default */
    void      *user;     /* opaque handle echoed back to on_done */
    /* W7 per-seq sampling (zeroed => greedy argmax, the W5/W6 default):       */
    float      temperature; /* <= 0 => greedy argmax (ignores the rest)        */
    int        top_k;       /* <= 0 => full vocab                              */
    float      top_p;       /* nucleus; <=0 or >1 treated as 1.0               */
    float      min_p;       /* relative floor; <0 => 0                         */
    uint64_t   seed;        /* per-seq RNG seed (caller resolves 0 if it wants */
                            /* distinct streams; 0 is a fixed, valid sequence) */
    /* A2a warm start.  Zero-init = engine-managed cold admit (the W5..W7
     * behavior, unchanged).  place_bank is a bank id + 1 placement directive
     * (0 = engine picks the first free bank); it lets the caller route a
     * request to a specific FREE bank -- warm continuation, or directed cold
     * placement away from valuable retired banks.  n_cached > 0 requests a
     * WARM admit into bank place_bank-1: tokens[0..n_cached) must equal that
     * bank's committed history exactly (ENGINE-VALIDATED against its own
     * per-bank record; any mismatch degrades to a cold reset, never reuses a
     * non-matching cache), and only tokens[n_cached..n) are prefilled. */
    int        place_bank;  /* bank id + 1; 0 = engine's choice               */
    int        n_cached;    /* committed prefix length in bank place_bank-1;  */
                            /* 0 = cold admit                                  */
    int       *bank_used;   /* OUT (optional): engine writes the bank id this */
                            /* request was placed in, at admit time           */
    /* A2b fork-by-copy.  fork_bank = source bank id + 1 (0 = no fork): the
     * request's tokens[0..n_cached) must equal the SOURCE bank's committed
     * history (ENGINE-VALIDATED, like warm; the source must also be idle --
     * not generating).  The engine D2D-copies the source bank's committed
     * state into the target bank (place_bank directive or engine's pick) and
     * prefills only tokens[n_cached..n), leaving the source bank untouched --
     * N requests sharing a long prefix pay one prefill + N cheap copies.
     * Any validation failure degrades to a cold admit.  When fork_bank > 0,
     * n_cached describes the SOURCE bank (warm matching is skipped); if the
     * target resolves to the source itself the fork becomes a plain warm
     * admit (no copy). */
    int        fork_bank;   /* source bank id + 1; 0 = no fork                */
} ds4_cont_request;
/* A2a: a bank's committed token history (engine-authoritative bookkeeping for
 * warm start).  *toks points at ctx-owned storage, valid until the next admit
 * or reset that touches the bank; returns the committed length, 0 when the
 * bank is out of range or its state is not reuse-trustworthy (engine failure,
 * static-path reuse of the slabs, deferred-commit MTP path). */
int  ds4_batch_ctx_bank_committed(const ds4_batch_ctx *ctx, int bank,
                                  const int **toks);
/* admit: fill *req for the next waiting request and return 1; return 0 when none is
 *   available right now (the loop keeps decoding the active set and ends once the
 *   active set is empty AND admit returns 0).
 * on_token (may be NULL): called once per newly sampled NON-EOS token, in order,
 *   for the sequence identified by `user` (seed token then each decode step).
 *   Return 1 to keep generating, 0 to ABORT that sequence now (e.g. its client
 *   disconnected) -- the engine evicts it this step and still calls on_done
 *   (finish=0).  NULL disables streaming (pure buffer-then-on_done, the W5 path).
 * on_done: a sequence finished -- tokens[0..n) is its full generation (caller must
 *   NOT free; valid only during the call), finish=1 if it hit EOS (0 = budget/abort).
 * Returns 0 on success. */
int  ds4_engine_continuous_generate(ds4_batch_ctx *ctx,
                                    int (*admit)(void *ud, ds4_cont_request *req),
                                    int (*on_token)(void *ud, void *user, int token),
                                    void (*on_done)(void *ud, void *user,
                                                    const int *tokens, int n, int finish),
                                    void *ud, char *err, size_t errlen);

/* Phase 2 S1.1: deterministic MTP gate.  Drives the continuous engine over a fixed
 * set of synthetic prompts (deterministic admission) and asserts the per-seq output
 * tokens are identical with the per-bank MTP draft path off vs on -- the clean
 * non-invasiveness/exactness proof (no server-timing / batch-composition confound).
 * Requires a ctx created with --mtp.  Returns 0 PASS, 1 token MISMATCH, 2 setup error. */
int  ds4_cont_mtp_gate(ds4_batch_ctx *ctx, char *err, size_t errlen);

/* Phase 2 A2a: deterministic warm-start gate.  Drives the continuous engine with
 * fixed REAL-TEXT prompts (confident greedy margins, so cross-packing token
 * comparison is meaningful) and asserts (a) STRUCTURAL: an isolated warm suffix
 * prefill leaves the committed compressed-cache frontier (per-layer counts)
 * exactly equal to a cold full prefill, at two group alignments; (b) a warm
 * admit's token stream matches a cold full prefill of the same effective prompt,
 * including a chained second warm turn and a LONG suffix; (c) a non-matching
 * cached prefix is rejected and degrades to a byte-identical cold run; (d) two
 * banks warm in one run with out-of-order placement directives.  A2b adds the
 * fork-by-copy phases: (e) a fork admit (D2D bank copy + suffix prefill) is
 * STRUCTURALLY frontier-exact vs a cold prefill and token-matches it, including
 * a second fork from the same source (fan-out reuse); (f) the source bank still
 * warm-continues byte-identically after serving two forks; (g) a fork with a
 * mutated cached token is rejected and degrades to cold.  Needs only a batch
 * ctx (no --mtp).  Returns 0 PASS, 1 MISMATCH, 2 setup error. */
int  ds4_cont_warm_gate(ds4_batch_ctx *ctx, char *err, size_t errlen);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);
int ds4_session_output_head_bench(ds4_session *s, int iters, FILE *fp, char *err, size_t errlen);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
