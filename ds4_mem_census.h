#ifndef DS4_MEM_CENSUS_H
#define DS4_MEM_CENSUS_H

#include <stdint.h>

/* =========================================================================
 * memgov D0a-1: allocation-census core (types + checked arithmetic).
 *
 * A deliberately dependency-free leaf header: ds4_cuda.cu carries its own
 * signatures instead of including ds4.h/ds4_gpu.h (the v0.5.6.1 Metal-PR
 * convention), so the census types it shares with the engine and the unit
 * suite live here, the same way cuda/mmq/ds4_mmq.h is shared.  ds4.h
 * includes this header and layers the backend API declarations on top.
 *
 * Census: local/docs/v057/d0a_census_table.md classifies every CUDA
 * allocation site into the fixed consumer classes below.  The CUDA backend
 * owns one cell per class x domain and records {requested, committed} at
 * each allocation/release site.  NO decision path reads these cells --
 * D0a's contract is zero decision change; enforcement arrives with the
 * governor (D0b/D2), which subsumes these classes as lease consumers.
 *
 * Vocabulary (plan sec 4): `requested` = what the consumer asked for,
 * including any bump-allocator alignment it consumed; `committed` = what
 * the allocator physically committed, including chunk/granularity rounding.
 * slack = live committed - live requested: rounding + arena tail the box
 * pays for but no consumer asked for.  Frees are recorded on the same
 * cell; over-free clamps and counts a fault rather than wrapping (checked
 * saturating arithmetic, plan sec 6.2 -- an impossible state must be
 * visible, never a negative ledger). */
typedef enum {                       /* FIXED cardinality -- census sec 6.
                                        ENGINE_OTHER is deliberately the
                                        ZERO value: a zero-initialized
                                        class tag reads as "untagged", and
                                        the close gate asserts that class
                                        settles to zero live bytes. */
    DS4_MEMC_ENGINE_OTHER = 0,       /* untagged funnel fallback           */
    DS4_MEMC_WEIGHT_ARENA,           /* VMM/cudaMalloc weight arena chunks */
    DS4_MEMC_WEIGHT_SPAN,            /* direct per-span device copies      */
    DS4_MEMC_WEIGHT_WHOLE,           /* whole-model device copies          */
    DS4_MEMC_WEIGHT_DERIVED,         /* q8->f16/f32 dequant caches         */
    DS4_MEMC_WEIGHT_IMPORT,          /* VMM/IPC weight-server imports      */
    DS4_MEMC_WEIGHT_ARTIFACT,        /* in-process aligned artifacts       */
    DS4_MEMC_WEIGHT_HOST_PIN,        /* model-map host registrations       */
    DS4_MEMC_STAGE_PIN,              /* pinned staging pools               */
    DS4_MEMC_SCALARS_MIRROR,         /* decode/layer/row scalar mirrors    */
    DS4_MEMC_SESSION_TENSORS,        /* engine graph activations/state     */
    DS4_MEMC_KV_PRIMARY,             /* raw ring + compressed KV           */
    DS4_MEMC_BATCH_BANK,             /* bank comp/index slabs (VMM demand) */
    DS4_MEMC_SCRATCH_STICKY,         /* grow-only sticky scratches         */
    DS4_MEMC_KERNEL_PARTIALS,        /* boot-fixed kernel buffers + ws     */
    DS4_MEMC_GRAPH_EXEC,             /* instantiated graphs (empirical)    */
    DS4_MEMC_DIAG,                   /* env-gated selftest/verify buffers  */
    DS4_MEMC__COUNT
} ds4_mem_consumer_class;

typedef enum {
    DS4_MEMD_UNIFIED_DEVICE = 0,     /* CUDA allocs; aliases system RAM on
                                        integrated (GB10)                  */
    DS4_MEMD_PINNED_HOST,            /* pinned allocs + registrations      */
    DS4_MEMD__COUNT
} ds4_mem_domain;

typedef struct {
    uint64_t requested;              /* monotonic sum of consumer asks     */
    uint64_t committed;              /* monotonic sum of allocator commits */
    uint64_t freed_requested;        /* monotonic sum of released asks     */
    uint64_t freed_committed;        /* monotonic sum of released commits  */
    uint64_t alloc_calls, free_calls;
} ds4_mem_cell;

/* Saturating add: on overflow, pin at UINT64_MAX and count a fault.  A
 * saturated monotonic counter stays ordered (stale reads err HIGH, the
 * tripwire convention) instead of wrapping into a plausible small lie. */
static inline uint64_t ds4_mem_sat_add(uint64_t a, uint64_t b,
                                       uint64_t *faults) {
    if (b > UINT64_MAX - a) {
        if (faults) (*faults)++;
        return UINT64_MAX;
    }
    return a + b;
}

static inline void ds4_mem_cell_note_alloc(ds4_mem_cell *c,
                                           uint64_t requested,
                                           uint64_t committed,
                                           uint64_t *faults) {
    if (!c) return;
    c->requested = ds4_mem_sat_add(c->requested, requested, faults);
    c->committed = ds4_mem_sat_add(c->committed, committed, faults);
    c->alloc_calls++;
}

/* Free clamps to the outstanding balance: releasing more than is live is
 * an impossible state (double-free or misattributed class) -- record the
 * fault and clamp so `live` bottoms at zero instead of wrapping. */
static inline void ds4_mem_cell_note_free(ds4_mem_cell *c,
                                          uint64_t requested,
                                          uint64_t committed,
                                          uint64_t *faults) {
    if (!c) return;
    uint64_t live_req = c->requested - c->freed_requested;
    uint64_t live_com = c->committed - c->freed_committed;
    if (requested > live_req) { if (faults) (*faults)++; requested = live_req; }
    if (committed > live_com) { if (faults) (*faults)++; committed = live_com; }
    c->freed_requested = ds4_mem_sat_add(c->freed_requested, requested, faults);
    c->freed_committed = ds4_mem_sat_add(c->freed_committed, committed, faults);
    c->free_calls++;
}

static inline uint64_t ds4_mem_cell_live(const ds4_mem_cell *c) {
    return c ? c->committed - c->freed_committed : 0;
}

/* Live slack: committed-but-unasked bytes (chunk tails, granularity
 * rounding, import over-allocation, trim pages that unmapped but failed
 * to release).  Floors at 0: requested can briefly exceed committed only
 * in a fault state, never a real ledger. */
static inline uint64_t ds4_mem_cell_slack(const ds4_mem_cell *c) {
    if (!c) return 0;
    const uint64_t live_com = c->committed - c->freed_committed;
    const uint64_t live_req = c->requested - c->freed_requested;
    return live_com > live_req ? live_com - live_req : 0;
}

#endif /* DS4_MEM_CENSUS_H */
