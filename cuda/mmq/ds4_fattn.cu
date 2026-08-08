/* exaone prefill attention on tensor cores (flash-attention style).
 *
 * The anchor kernel scans the whole KV history once per (query, head) block
 * with warp-shuffle dot products; a shared-memory tiled variant cut the KV
 * traffic 16x and measured a WASH, because the per-(query, key) shuffle
 * reductions became the bound.  This kernel removes the reduction entirely:
 * Q·K^T and P·V are m16n8k16 HMMA (f16 in, f32 out for scores; f32
 * accumulators for the output), with the standard flash-attention online
 * softmax between them.
 *
 * Layout (head_dim 128 only -- the shape everything here is sized for):
 *   grid  (ceil(n_tokens/64), n_head), block 128 threads = 4 warps.
 *   Each warp owns 16 query rows; the block stages its 64 Q rows (converted
 *   to f16) once, then walks 16-key K/V tiles staged in shared memory, all
 *   four warps consuming each tile.
 *
 * Per key tile and warp: scores S[16x16] from 8+8 QK mmas, online-softmax
 * update, then P·V into 16 f32 accumulator tiles covering head_dim.  The
 * PTX fragment layouts line up so P's score fragments re-pack into the next
 * mma's A operand entirely in-lane (no shuffles) -- that alignment is the
 * whole reason the m16n8k16 C and A layouts look the way they do.
 *
 * Numerics: scores come out of an f16 multiply (f32 accumulate), so this is
 * not bit-identical to the f32 anchor; the kernel-suite gate is agreement
 * with the CPU reference at the same tolerances as the anchor.  Positions
 * enter the mask logic as absolute values; the ring modulo is applied when
 * staging, exactly as the anchor applies it when reading.
 */
#include "common.cuh"
#include "mma.cuh"
#include "ds4_mmq.h"

#include <cuda_fp16.h>

using namespace ggml_cuda_mma;

namespace {

enum {
    FA_HD      = 128,   /* head_dim this kernel is built for            */
    FA_WQ      = 16,    /* query rows per warp                          */
    FA_WARPS   = 4,
    FA_TQ      = FA_WQ * FA_WARPS,   /* query rows per block            */
    FA_TK      = 16,    /* key rows per staged tile                     */
    FA_PAD     = 8,     /* half padding per smem row against bank camp  */
    FA_ROW     = FA_HD + FA_PAD,
};

typedef tile<16, 8, half2> tile_a;      /* Q rows / P rows              */
typedef tile< 8, 8, half2> tile_b;      /* K keys / V columns           */
typedef tile<16, 8, float> tile_c;      /* scores / output accumulators */

__global__ void exaone_fattn_hmma_kernel(
        float * __restrict__ heads,
        const float * __restrict__ q,
        const __half * __restrict__ kv,
        const uint32_t n_tokens,
        const uint32_t pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const uint32_t kv_cap,
        const uint32_t window,
        const float scale) {
    const uint32_t tq0 = blockIdx.x * FA_TQ;
    const uint32_t h   = blockIdx.y;
    if (tq0 >= n_tokens || h >= n_head) return;
    const uint32_t group  = n_head / n_head_kv;
    const uint32_t kvh    = h / group;
    const uint32_t kv_dim = n_head_kv * FA_HD;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;

    __shared__ __half s_q[FA_TQ][FA_ROW];
    __shared__ __half s_k[FA_TK][FA_ROW];
    __shared__ __half s_v[FA_TK][FA_ROW];

    /* Stage and convert this block's Q rows once.  Dead rows (past n_tokens)
     * stage row 0 so the mma operands stay finite; their stores are skipped
     * and their mask kills every key. */
    for (uint32_t idx = threadIdx.x; idx < FA_TQ * FA_HD; idx += blockDim.x) {
        const uint32_t r = idx / FA_HD, c = idx - r * FA_HD;
        const uint32_t t = tq0 + r < n_tokens ? tq0 + r : 0u;
        s_q[r][c] = __float2half(q[((size_t)t * n_head + h) * FA_HD + c]);
    }
    __syncthreads();

    /* Per-warp query bookkeeping.  Lane owns C-tile rows tid/4 and tid/4+8
     * of the warp's 16; masks and stats are per those two rows. */
    const uint32_t qrow[2] = { warp * FA_WQ + lane / 4,
                               warp * FA_WQ + lane / 4 + 8u };
    uint32_t qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r]  = tq0 + qrow[r] < n_tokens;
        qpos[r]   = alive[r] ? pos0 + tq0 + qrow[r] : pos0;   /* absolute */
        qfirst[r] = (window && qpos[r] + 1u > window) ? qpos[r] + 1u - window : 0u;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    /* Q fragments for all 8 k-chunks, loaded once.
     *
     * Fragment coordinates are written out lane-based rather than through
     * tile::get_i/get_j: those helpers read threadIdx.x raw and are only
     * meaningful inside a 32-thread block (mmq launches them that way), and
     * this kernel runs four warps.  The formulas are the m16n8k16 PTX
     * layouts:  A row (l%2)*8 + lane/4, A k-half2 (l/2)*4 + lane%4;
     * B col lane/4, B k-half2 l*4 + lane%4;  C row (l/2)*8 + lane/4,
     * C col (lane%4)*2 + l%2. */
    tile_a qa[FA_HD / 16];
#pragma unroll
    for (int kc = 0; kc < FA_HD / 16; kc++) {
#pragma unroll
        for (int l = 0; l < tile_a::ne; l++) {
            const int i = (l % 2) * 8 + (int)(lane / 4);
            const int j = (l / 2) * 4 + (int)(lane % 4);
            qa[kc].x[l] = *(const half2 *)&s_q[warp * FA_WQ + i][kc * 16 + 2 * j];
        }
    }

    /* Output accumulators: 16 col-blocks of 8 across head_dim. */
    tile_c oc[FA_HD / 8];

    /* Block-wide key range: from the earliest window start among live rows
     * to the latest query position. */
    uint32_t blk_first = 0xffffffffu, blk_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!alive[r]) continue;
        if (qfirst[r] < blk_first) blk_first = qfirst[r];
        if (qpos[r]   > blk_last)  blk_last  = qpos[r];
    }
    /* Every lane knows two rows; fold the warp before one lane represents it
     * in the block hull. */
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        blk_first = min(blk_first, __shfl_xor_sync(0xffffffffu, blk_first, off));
        blk_last  = max(blk_last,  __shfl_xor_sync(0xffffffffu, blk_last,  off));
    }
    __shared__ uint32_t s_first, s_last;
    if (threadIdx.x == 0u) { s_first = 0xffffffffu; s_last = 0u; }
    __syncthreads();
    if (lane == 0u) { atomicMin(&s_first, blk_first); atomicMax(&s_last, blk_last); }
    __syncthreads();
    blk_first = s_first;
    blk_last  = s_last;
    if (blk_first == 0xffffffffu) return;   /* whole block past n_tokens */

    for (uint32_t kt0 = blk_first; kt0 <= blk_last; kt0 += FA_TK) {
        const uint32_t tk_len = (blk_last - kt0 + 1u < FA_TK)
                              ? blk_last - kt0 + 1u : FA_TK;
        for (uint32_t idx = threadIdx.x; idx < FA_TK * FA_HD; idx += blockDim.x) {
            const uint32_t r = idx / FA_HD, c = idx - r * FA_HD;
            const uint32_t src = r < tk_len ? kt0 + r : kt0;   /* clamp */
            const __half *row = kv + (size_t)(src % kv_cap) * kv_dim * 2u
                                   + (size_t)kvh * FA_HD;
            s_k[r][c] = row[c];
            s_v[r][c] = row[kv_dim + c];
        }
        __syncthreads();

        /* Scores: two n-8 key blocks, 8 k-chunks each. */
        tile_c sc[2];
#pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            tile_c z;                    /* zero-initialized accumulator */
            sc[nb] = z;
#pragma unroll
            for (int kc = 0; kc < FA_HD / 16; kc++) {
                tile_b kb;
#pragma unroll
                for (int l = 0; l < tile_b::ne; l++) {
                    const int i = (int)(lane / 4);           /* key in block  */
                    const int j = l * 4 + (int)(lane % 4);   /* half2 in chnk */
                    kb.x[l] = *(const half2 *)&s_k[nb * 8 + i][kc * 16 + 2 * j];
                }
                mma(sc[nb], qa[kc], kb);
            }
        }

        /* Mask + online softmax.  Lane's score cols: j = (tid%4)*2 + l%2 in
         * each n-8 block; absolute key = kt0 + nb*8 + j. */
        float tile_max[2] = { -INFINITY, -INFINITY };
#pragma unroll
        for (int nb = 0; nb < 2; nb++) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; l++) {
                const int r = l / 2;
                const uint32_t p = kt0 + nb * 8u + (lane % 4u) * 2u + (l % 2u);
                float s = sc[nb].x[l] * scale;
                if (!alive[r] || p > qpos[r] || p < qfirst[r] || p >= kt0 + tk_len)
                    s = -INFINITY;
                sc[nb].x[l] = s;
                tile_max[r] = fmaxf(tile_max[r], s);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
        }

        float es[2], tile_l[2] = { 0.0f, 0.0f };
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const float mn = fmaxf(row_m[r], tile_max[r]);
            es[r] = (row_m[r] == -INFINITY) ? 0.0f : __expf(row_m[r] - mn);
            row_m[r] = mn;
        }
#pragma unroll
        for (int nb = 0; nb < 2; nb++) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; l++) {
                const int r = l / 2;
                const float w = (sc[nb].x[l] == -INFINITY || row_m[r] == -INFINITY)
                              ? 0.0f : __expf(sc[nb].x[l] - row_m[r]);
                sc[nb].x[l] = w;
                tile_l[r] += w;
            }
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            tile_l[r] += __shfl_xor_sync(0xffffffffu, tile_l[r], 1);
            tile_l[r] += __shfl_xor_sync(0xffffffffu, tile_l[r], 2);
            row_l[r] = row_l[r] * es[r] + tile_l[r];
        }

        /* Rescale the running output, repack P into the next mma's A operand
         * (in-lane by construction), and accumulate P·V. */
        tile_a pa;
#pragma unroll
        for (int l = 0; l < tile_a::ne; l++) {
            /* A row block (l%2) matches C row block (lc/2); A k-halfs match
             * C's col pair; nb = l/2 selects the 8-key score tile. */
            pa.x[l] = __floats2half2_rn(sc[l / 2].x[(l % 2) * 2 + 0],
                                        sc[l / 2].x[(l % 2) * 2 + 1]);
        }
#pragma unroll
        for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; l++)
                oc[cb].x[l] *= es[l / 2];
            tile_b vb;
#pragma unroll
            for (int l = 0; l < tile_b::ne; l++) {
                const int i = (int)(lane / 4);       /* output col in block */
                const int j = l * 4 + (int)(lane % 4);   /* half2 over keys */
                vb.x[l] = __halves2half2(s_v[2 * j + 0][cb * 8 + i],
                                         s_v[2 * j + 1][cb * 8 + i]);
            }
            mma(oc[cb], pa, vb);
        }
        __syncthreads();
    }

    /* Normalize and store. */
#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r] || row_l[r] <= 0.0f) continue;
            const uint32_t t = tq0 + qrow[r];
            const int cj = (int)(lane % 4) * 2 + (l % 2);
            heads[((size_t)t * n_head + h) * FA_HD + cb * 8 + cj] =
                oc[cb].x[l] / row_l[r];
        }
    }
}

}   // namespace

extern "C" int ds4_mmq_exaone_prefill_attn_hmma(
        float *heads, const float *q, const void *kv,
        int n_tokens, int pos0, int n_head, int n_head_kv, int head_dim,
        int kv_cap, int window, float scale, cudaStream_t stream) {
    if (!heads || !q || !kv || n_tokens <= 0 || n_head <= 0 ||
        n_head_kv <= 0 || head_dim != FA_HD || kv_cap <= 0 ||
        n_head % n_head_kv != 0) {
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[dev].cc < GGML_CUDA_CC_AMPERE) return -1;
    dim3 grid((n_tokens + FA_TQ - 1) / FA_TQ, n_head, 1);
    exaone_fattn_hmma_kernel<<<grid, FA_WARPS * 32, 0, stream>>>(
            heads, q, (const __half *)kv,
            (uint32_t)n_tokens, (uint32_t)pos0, (uint32_t)n_head,
            (uint32_t)n_head_kv, (uint32_t)kv_cap, (uint32_t)window, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
