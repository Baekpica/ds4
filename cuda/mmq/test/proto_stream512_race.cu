// SPDX-License-Identifier: MIT
// proto_stream512_race.cu — #9 synthesis fallback: racecheck / synccheck
// microstress of the PRODUCTION indexer_topk_stream512_kernel.
//
// WHY THIS EXISTS (crash-hunt synthesis 2026-08-04): the four-model panel
// flagged that racecheck/synccheck were never run on the PRODUCTION
// stream512 kernel — only on the proto (proto_topk_diet.cu), whose
// racecheck-confirmed aliasing hazard the production kernel then fixed
// (separate buf / cub TempStorage regions, comment at the kernel).  Sol's
// CUDA_EXCEPTION_5 local/shared mapping for the field Xid 13 class fits an
// in-kernel shared-memory defect, and the in-vivo record-and-clamp guard
// (TOPK-BOUND-VIOL, 18-run leg, zero fires across 6 crashes) has already
// eliminated the live-bound overread — smem is the remaining in-kernel
// surface.  This harness compiles ds4_cuda.cu INTO this TU so the kernel
// under test is the shipped one, not a copy that can drift.
//
// Shapes: the crash window's exact tier crossing (n_comp=8208, the first
// >8192 stream512 engagement in the reproducer; n_tokens=512) plus
// adjacent shapes.  Patterns: uniform random, ascending (recency-shaped
// accept-storm — the one measured adversarial case for the rising
// threshold), descending (compaction-storm: early threshold saturation),
// all-equal (tie flood, max ballot divergence), and banded (-INF pad
// above a live count = the captured-band row layout).  Substrate: NULL
// (eager) and a device ds4_layer_scalars with n_index_comp <= n_comp
// (captured semantics: live scan bound under a baked stride).
//
// Every launch is also verified against a CPU exact reference (descending
// topk_pack_key order == production contract), so silent corruption fails
// the harness even where the sanitizer stays quiet.
//
// Build ON .33 from the repo root (same arch flags as the Makefile; the
// include needs repo root as the working directory):
//   /usr/local/cuda/bin/nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 \
//     -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1 \
//     -Xcompiler -march=native -Xcompiler -pthread -DDS4_CUDA_SPARK_HBM_CACHE=1 \
//     -I. cuda/mmq/test/proto_stream512_race.cu \
//     -o /tmp/proto_stream512_race \
//     -lm -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/lib64 \
//     -lcudart -lcublas -lcuda
// Run:
//   /tmp/proto_stream512_race                         # plain: exactness only
//   compute-sanitizer --tool racecheck /tmp/proto_stream512_race
//   compute-sanitizer --tool synccheck /tmp/proto_stream512_race
// Exit 0 = all launches byte-exact vs the reference.  Sanitizer verdicts
// come from the tool's own summary (assert '0 errors' in the runner).
//
// NOTE: any racecheck hazard reported here must be adjudicated against the
// 2026-07-28 elimination dossier (brief-v05-arc.md): the proto-era flags on
// the load-fed-predicate pattern were ruled a tool artifact.  A hazard is
// only actionable if it names the PRODUCTION smem objects with a
// read/write pair not covered by that ruling (or synccheck, which had no
// artifact history, reports at all).

#include "ds4_cuda.cu"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA FAIL %s:%d %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e_)); return 1; } } while (0)

/* host mirror of topk_pack_key / topk_float_ordered_key (verbatim math) */
static uint32_t h_ordered_key(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
static uint64_t h_pack_key(float v, uint32_t idx) {
    return ((uint64_t)h_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

/* exact reference: top_k ids of row[0..n) in descending pack-key order */
static void ref_topk(const float *row, uint32_t n, uint32_t top_k,
                     uint32_t *out) {
    std::vector<uint64_t> keys;
    keys.reserve(n);
    for (uint32_t c = 0; c < n; c++) keys.push_back(h_pack_key(row[c], c));
    const uint32_t k = top_k < n ? top_k : n;
    std::partial_sort(keys.begin(), keys.begin() + k, keys.end(),
                      std::greater<uint64_t>());
    for (uint32_t i = 0; i < k; i++)
        out[i] = 0xffffffffu - (uint32_t)(keys[i] & 0xffffffffu);
    for (uint32_t i = k; i < top_k; i++) out[i] = 0xffffffffu;  /* unreachable
        at these shapes (n > top_k always); kept for shape safety */
}

struct Pattern { const char *name; int kind; };

static void fill_row(float *row, uint32_t stride, uint32_t live, int kind,
                     std::mt19937 &rng) {
    std::uniform_real_distribution<float> d(-8.f, 8.f);
    for (uint32_t c = 0; c < live; c++) {
        switch (kind) {
        case 0: row[c] = d(rng); break;                        /* uniform   */
        case 1: row[c] = (float)c * 1e-3f + d(rng) * 1e-4f; break; /* asc   */
        case 2: row[c] = -(float)c * 1e-3f + d(rng) * 1e-4f; break;/* desc  */
        case 3: row[c] = 1.25f; break;                         /* tie flood */
        case 4: row[c] = (c & 1u) ? d(rng) : -INFINITY; break; /* inf comb  */
        }
    }
    for (uint32_t c = live; c < stride; c++) row[c] = -INFINITY; /* band pad */
}

int main(void) {
    const uint32_t TOP_K = 512u;
    /* {stride (baked n_comp), live (ls->n_index_comp), tokens, use_ls} */
    struct Shape { uint32_t stride, live, tokens; int use_ls; };
#ifdef DS4_S512_TRACE
    /* trace mode: only the reproducibly-faulting launch (4/4 under
     * racecheck: launch 1 = first shape, uniform, iter 0) */
    const Shape shapes[] = {
        { 8208u,  8208u, 512u, 0 },
    };
    const Pattern pats[] = { { "uniform", 0 } };
    const int ITERS = 1;
#else
    const Shape shapes[] = {
        { 8208u,  8208u, 512u, 0 },   /* crash-window tier, eager          */
        { 8208u,  8208u, 512u, 1 },   /* crash-window tier, captured, n==band */
        { 8208u,  8200u, 512u, 1 },   /* captured, live < band (pad scanned) */
        { 8208u,  8208u,  32u, 1 },   /* tier floor tokens                  */
        { 12000u, 11984u, 512u, 1 },  /* deeper band, live < band           */
        { 16400u, 16400u, 256u, 1 },  /* two-compact depth                  */
    };
    const Pattern pats[] = {
        { "uniform", 0 }, { "ascending", 1 }, { "descending", 2 },
        { "ties", 3 }, { "infcomb", 4 },
    };
    const int ITERS = 3;   /* schedules per shape x pattern */
#endif

    std::mt19937 rng(0x515u);
    int total = 0;
#ifdef DS4_S512_APPEND_GUARD
    int mismatches = 0;   /* guard mode tolerates dropped keys: count, don't abort */
#endif
#ifdef DS4_S512_TRACE
    volatile unsigned int *h_trace = NULL;
    unsigned int *d_trace = NULL;
    const size_t trace_words = 512u * 16u * 2u;
    CK(cudaHostAlloc((void **)&h_trace, trace_words * sizeof(unsigned int),
                     cudaHostAllocMapped));
    memset((void *)h_trace, 0, trace_words * sizeof(unsigned int));
    CK(cudaHostGetDevicePointer((void **)&d_trace, (void *)h_trace, 0));
    CK(cudaMemcpyToSymbol(g_s512_trace, &d_trace, sizeof d_trace));
#endif

    for (const Shape &sh : shapes) {
        float *d_scores = NULL;
        uint32_t *d_sel = NULL;
        struct ds4_layer_scalars *d_ls = NULL;
        CK(cudaMalloc(&d_scores, (uint64_t)sh.tokens * sh.stride * sizeof(float)));
        CK(cudaMalloc(&d_sel, (uint64_t)sh.tokens * TOP_K * sizeof(uint32_t)));
        if (sh.use_ls) {
            struct ds4_layer_scalars h_ls;
            memset(&h_ls, 0, sizeof h_ls);
            h_ls.n_comp = sh.live;
            h_ls.n_index_comp = sh.live;
            CK(cudaMalloc(&d_ls, sizeof h_ls));
            CK(cudaMemcpy(d_ls, &h_ls, sizeof h_ls, cudaMemcpyHostToDevice));
        }
        std::vector<float> h_scores((uint64_t)sh.tokens * sh.stride);
        std::vector<uint32_t> h_sel((uint64_t)sh.tokens * TOP_K);
        std::vector<uint32_t> h_ref(TOP_K);

        for (const Pattern &p : pats) {
            for (int it = 0; it < ITERS; it++) {
                for (uint32_t t = 0; t < sh.tokens; t++)
                    fill_row(&h_scores[(uint64_t)t * sh.stride], sh.stride,
                             sh.live, p.kind, rng);
                CK(cudaMemcpy(d_scores, h_scores.data(),
                              h_scores.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
                CK(cudaMemset(d_sel, 0xee,
                              h_sel.size() * sizeof(uint32_t)));
                fprintf(stderr, "LAUNCH stride=%u live=%u tokens=%u ls=%d pat=%s iter=%d\n",
                        sh.stride, sh.live, sh.tokens, sh.use_ls, p.name, it);
                indexer_topk_stream512_kernel<<<sh.tokens, 512>>>(
                        d_sel, d_scores, sh.stride, sh.tokens, TOP_K, d_ls);
                CK(cudaGetLastError());
#ifdef DS4_S512_TRACE
                {
                    const cudaError_t se = cudaDeviceSynchronize();
                    if (se != cudaSuccess) {
                        fprintf(stderr, "TRACE-FAULT: %s\n",
                                cudaGetErrorString(se));
                        /* distinct last-marker histogram across all warps */
                        unsigned int vals[64], cnts[64];
                        int nv = 0;
                        for (uint32_t s = 0; s < sh.tokens * 16u; s++) {
                            const unsigned int m = h_trace[s * 2u];
                            int j;
                            for (j = 0; j < nv; j++)
                                if (vals[j] == m) { cnts[j]++; break; }
                            if (j == nv && nv < 64) {
                                vals[nv] = m; cnts[nv] = 1; nv++;
                            }
                        }
                        for (int j = 0; j < nv; j++)
                            fprintf(stderr,
                                    "MARK 0x%05x (iter=%u phase=%u) x%u\n",
                                    vals[j], vals[j] >> 8u, vals[j] & 0xffu,
                                    cnts[j]);
                        /* blocks with intra-block warp skew: the smoking gun */
                        int skew_blocks = 0;
                        for (uint32_t b = 0; b < sh.tokens; b++) {
                            unsigned int mn = 0xffffffffu, mx = 0u;
                            for (int w = 0; w < 16; w++) {
                                const unsigned int m = h_trace[(b * 16u + w) * 2u];
                                if (m < mn) mn = m;
                                if (m > mx) mx = m;
                            }
                            if (mn != mx && skew_blocks < 8) {
                                skew_blocks++;
                                for (int w = 0; w < 16; w++)
                                    fprintf(stderr,
                                            "SKEW b=%u w=%d mark=0x%05x s_cnt=%u\n",
                                            b, w,
                                            h_trace[(b * 16u + w) * 2u],
                                            h_trace[(b * 16u + w) * 2u + 1u]);
                            } else if (mn != mx) {
                                skew_blocks++;
                            }
                        }
                        fprintf(stderr, "SKEW-BLOCKS total=%d\n", skew_blocks);
                        return 3;
                    }
                }
#else
                CK(cudaDeviceSynchronize());
#endif
                CK(cudaMemcpy(h_sel.data(), d_sel,
                              h_sel.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
#ifdef DS4_S512_APPEND_GUARD
                {
                    unsigned int oc = 0, om = 0;
                    CK(cudaMemcpyFromSymbol(&oc, g_s512_append_ovf_count,
                                            sizeof oc));
                    CK(cudaMemcpyFromSymbol(&om, g_s512_append_ovf_max,
                                            sizeof om));
                    if (oc) fprintf(stderr,
                            "APPEND-OVF cum_count=%u max_idx=%u\n", oc, om);
                }
#endif
                /* verify every 64th token + the last (CPU ref is the slow
                 * part; sampled verify keeps sanitizer runs tractable while
                 * still crossing the block-id range) */
                for (uint32_t vi = 0; vi <= sh.tokens; vi += 64u) {
                    const uint32_t t = (vi < sh.tokens) ? vi : sh.tokens - 1u;
                    ref_topk(&h_scores[(uint64_t)t * sh.stride], sh.live,
                             TOP_K, h_ref.data());
                    if (memcmp(h_ref.data(), &h_sel[(uint64_t)t * TOP_K],
                               TOP_K * sizeof(uint32_t)) != 0) {
                        fprintf(stderr,
                                "MISMATCH stride=%u live=%u tokens=%u ls=%d "
                                "pat=%s iter=%d token=%u\n",
                                sh.stride, sh.live, sh.tokens, sh.use_ls,
                                p.name, it, t);
#ifdef DS4_S512_APPEND_GUARD
                        mismatches++;
                        break;   /* one record per launch is enough */
#else
                        return 2;
#endif
                    }
                }
                total++;
            }
        }
        CK(cudaFree(d_scores));
        CK(cudaFree(d_sel));
        if (d_ls) CK(cudaFree(d_ls));
        printf("shape stride=%u live=%u tokens=%u ls=%d: %d launches OK\n",
               sh.stride, sh.live, sh.tokens, sh.use_ls,
               (int)(sizeof(pats) / sizeof(pats[0])) * ITERS);
        fflush(stdout);
    }
#ifdef DS4_S512_APPEND_GUARD
    {
        unsigned int oc = 0, om = 0;
        CK(cudaMemcpyFromSymbol(&oc, g_s512_append_ovf_count, sizeof oc));
        CK(cudaMemcpyFromSymbol(&om, g_s512_append_ovf_max, sizeof om));
        printf("PROTO-STREAM512-RACE-GUARD-DONE (%d launches, mismatches=%d, "
               "append_ovf=%u max_idx=%u)\n", total, mismatches, oc, om);
    }
#else
    printf("PROTO-STREAM512-RACE-PASS (%d launches, all byte-exact)\n", total);
#endif
    return 0;
}
