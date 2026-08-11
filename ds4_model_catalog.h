#ifndef DS4_MODEL_CATALOG_H
#define DS4_MODEL_CATALOG_H

#include <stdint.h>

/* =========================================================================
 * memgov D1a-2: semantic tensor catalog (pure classification).
 *
 * A dependency-free leaf header in the ds4_mem_census.h convention: the
 * engine builds one catalog per model source at model_open, and the unit
 * suite drives the exact production classifier without a model file.
 *
 * The traits encode the engine's EXACT-NAME binding knowledge (weights_bind
 * / mtp_weights_bind / dspark_weights_bind suffixes) and the repack
 * candidacy rules (cuda/mmq/ds4_repack.cu ds4_repack_*_candidate), so the
 * residency planner can reason about tensor classes without substring
 * heuristics.  The legacy memmem(name, "_exps.") predicate at the HBM
 * pre-cache site survives ONE stage as a cross-check tripwire (mismatch =
 * census fault + gate FAIL) and then dies — scoping sec 5: the heuristic
 * is retired by cross-checked replacement, not deletion.
 *
 * Trait bits (0 = ALWAYS-HOT, the pre-cacheable default):
 * - ROUTED_EXPERT: the 3-D ffn_{gate,up,down}_exps stacks — top-K of N
 *   experts fire per token, so pre-caching them starves hot tensors.
 *   Deliberately matches the BINDER's names (blk.N./mtp.0./dspark.N.
 *   prefixes all share these suffixes); the 2-D shared-expert *_shexp
 *   tensors and exp_probs_b bias are NOT routed.
 * - ARTIFACT_REPLACED: an aligned repack artifact REPLACES the raw range
 *   when built/imported (IQ2_XXS gate/up, Q2_K down) — byte-neutral
 *   layouts, raw consumers fall back to the host mmap.
 * - ARTIFACT_ADDITIVE: an artifact may shadow the raw range for specific
 *   consumers while the raw stays served (Q8_0 aligned dense, Q8_0->f16
 *   colmajor prebuild).
 * - OPTIONAL: bound via the optional lookup (exp_probs_b.bias on the base
 *   model) — absence is legal.
 *
 * The type/dims/bytes gates on the artifact traits MIRROR the repack
 * candidate functions exactly; units pin both the suffix sets and the
 * mirror (a drifted repack rule fails the unit, not the field). */

enum {
    DS4_TCAT_ROUTED_EXPERT     = 1u << 0,
    DS4_TCAT_ARTIFACT_REPLACED = 1u << 1,
    DS4_TCAT_ARTIFACT_ADDITIVE = 1u << 2,
    DS4_TCAT_OPTIONAL          = 1u << 3
};

static inline int ds4_tcat_has_suffix(const char *name, uint64_t len,
                                      const char *sfx) {
    uint64_t sl = 0, i;
    if (!name || !sfx) return 0;
    while (sfx[sl]) sl++;
    if (len < sl) return 0;
    for (i = 0; i < sl; i++)
        if (name[len - sl + i] != sfx[i]) return 0;
    return 1;
}

static inline int ds4_tcat_contains(const char *name, uint64_t len,
                                    const char *needle) {
    uint64_t nl = 0, i, j;
    if (!name || !needle) return 0;
    while (needle[nl]) nl++;
    if (nl == 0 || len < nl) return 0;
    for (i = 0; i + nl <= len; i++) {
        for (j = 0; j < nl; j++)
            if (name[i + j] != needle[j]) break;
        if (j == nl) return 1;
    }
    return 0;
}

/* GGML type ids as the repack rules spell them (leaf header: no ggml
 * include, same literals-with-comments convention as ds4_repack.cu). */
#define DS4_TCAT_GGML_Q8_0    8u
#define DS4_TCAT_GGML_Q2_K    10u
#define DS4_TCAT_GGML_IQ2_XXS 16u

static inline uint32_t ds4_tensor_catalog_classify(const char *name,
                                                   uint64_t name_len,
                                                   uint32_t ndim,
                                                   uint32_t ggml_type,
                                                   const uint64_t *dims,
                                                   uint64_t bytes) {
    uint32_t traits = 0;
    const int gate = ds4_tcat_has_suffix(name, name_len, ".ffn_gate_exps.weight");
    const int up   = ds4_tcat_has_suffix(name, name_len, ".ffn_up_exps.weight");
    const int down = ds4_tcat_has_suffix(name, name_len, ".ffn_down_exps.weight");

    if (ndim == 3 && (gate || up || down)) {
        traits |= DS4_TCAT_ROUTED_EXPERT;
        /* ds4_repack_iq2_candidate mirror: IQ2_XXS gate/up stacks. */
        if (ggml_type == DS4_TCAT_GGML_IQ2_XXS && (gate || up) && dims &&
            dims[0] != 0 && dims[1] != 0 && dims[2] != 0 &&
            dims[2] <= UINT32_MAX && dims[0] % 1024u == 0 &&
            bytes != 0 && bytes % 66u == 0) {
            traits |= DS4_TCAT_ARTIFACT_REPLACED;
        }
        /* ds4_repack_q2k_candidate mirror: Q2_K down stacks. */
        if (ggml_type == DS4_TCAT_GGML_Q2_K && down && dims &&
            dims[0] != 0 && dims[1] != 0 && dims[2] != 0 &&
            dims[2] <= UINT32_MAX && dims[0] % 256u == 0 &&
            dims[1] % 2u == 0 && bytes != 0 && bytes % 84u == 0) {
            traits |= DS4_TCAT_ARTIFACT_REPLACED;
        }
    }

    if (ggml_type == DS4_TCAT_GGML_Q8_0 && ndim == 2 && dims &&
        dims[0] != 0 && dims[1] != 0 && bytes % 34u == 0 && bytes != 0) {
        /* ds4_repack_q8_candidate mirror: aligned dense (2 MiB floor,
         * token_embd excluded). */
        if (dims[0] % 1024u == 0 && bytes >= 2u * 1024u * 1024u &&
            !ds4_tcat_contains(name, name_len, "token_embd")) {
            traits |= DS4_TCAT_ARTIFACT_ADDITIVE;
        }
        /* ds4_repack_q8_f16_candidate mirror: f16 colmajor prebuild. */
        if (dims[0] % 32u == 0 &&
            ds4_tcat_contains(name, name_len, "attn_output_a.weight")) {
            traits |= DS4_TCAT_ARTIFACT_ADDITIVE;
        }
    }

    if (ds4_tcat_has_suffix(name, name_len, ".exp_probs_b.bias"))
        traits |= DS4_TCAT_OPTIONAL;

    return traits;
}

/* Per-catalog census: how many tensors carry each trait (the boot line
 * and the D1a gate's positive engagement signal render from this). */
typedef struct {
    uint64_t tensors;
    uint64_t routed;
    uint64_t replaced;
    uint64_t additive;
    uint64_t optional;
} ds4_model_catalog_counts;

static inline void ds4_model_catalog_count(const uint8_t *traits, uint64_t n,
                                           ds4_model_catalog_counts *out) {
    uint64_t i;
    if (!out) return;
    out->tensors = n;
    out->routed = out->replaced = out->additive = out->optional = 0;
    if (!traits) return;
    for (i = 0; i < n; i++) {
        if (traits[i] & DS4_TCAT_ROUTED_EXPERT)     out->routed++;
        if (traits[i] & DS4_TCAT_ARTIFACT_REPLACED) out->replaced++;
        if (traits[i] & DS4_TCAT_ARTIFACT_ADDITIVE) out->additive++;
        if (traits[i] & DS4_TCAT_OPTIONAL)          out->optional++;
    }
}

#endif /* DS4_MODEL_CATALOG_H */
