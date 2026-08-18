/* Prefill attention on tensor cores (flash-attention style).
 *
 * Layout (head_dim 128 only): grid (ceil(n_tokens/64), n_head), block 128
 * threads. Each warp owns 16 query rows. The block stages 64 Q rows and
 * walks 32-key K/V tiles (two 16-key HMMA consume steps). Q.K^T and P.V
 * use m16n8k16 HMMA with online softmax between them. Compressed Solar
 * K/V is decoded once per shared tile with the per-row scale reused
 * across the 128 dims, so all 64 query rows share one dequant.
 *
 * When n_head/n_head_kv is even and at least 2, the GQA-pair kernel
 * launches instead: grid.y = n_head/2, block 256. Warps 0-3 and 4-7
 * own consecutive Q heads that share one KV head, so the same 32-key
 * tile is dequantized once for both. Q stays in qa[] registers so the
 * extra head does not push static shared memory past 48 KiB. Eight
 * Q heads in one block still does not fit registers or shared memory.
 */
#include "common.cuh"
#include "mma.cuh"
#include "ds4_mmq.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

using namespace ggml_cuda_mma;

namespace {

enum {
    FA_HD      = 128,
    FA_WQ      = 16,
    FA_WARPS   = 4,
    FA_TQ      = FA_WQ * FA_WARPS,
    FA_TK      = 32,
    FA_CONSUME = 16,
    FA_PAD     = 8,
    FA_ROW     = FA_HD + FA_PAD,
    SOLAR_KV_BF16 = 0,
    SOLAR_KV_FP8 = 1,
    SOLAR_KV_FP4 = 2,
    SOLAR_KV_KFP8_VFP4 = 3,
};

typedef tile<16, 8, half2> tile_a;
typedef tile< 8, 8, half2> tile_b;
typedef tile<16, 8, float> tile_c;

__device__ __forceinline__ float solar_fattn_e4m3(uint8_t code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return (float)value;
}

__device__ __forceinline__ float solar_fattn_e2m1(uint8_t code) {
    __nv_fp4_e2m1 value;
    value.__x = code & 0x0fu;
    return (float)value;
}

template <int FORMAT, bool VALUE>
__device__ __forceinline__ float solar_fattn_kv_load(
        const uint8_t *row,
        uint32_t       n_head_kv,
        uint32_t       kvh,
        uint32_t       dim) {
    const uint64_t kv_dim = (uint64_t)n_head_kv * FA_HD;
    const uint64_t elem = (uint64_t)kvh * FA_HD + dim;
    const uint64_t k_bytes = FORMAT == SOLAR_KV_FP4 ? kv_dim / 2u : kv_dim;
    const uint64_t v_bytes = FORMAT == SOLAR_KV_FP8 ? kv_dim : kv_dim / 2u;
    const __half *scales = (const __half *)(row + k_bytes + v_bytes);
    const float scale = __half2float(
        scales[(VALUE ? n_head_kv : 0u) + kvh]);
    const uint8_t *data = VALUE ? row + k_bytes : row;
    if constexpr ((VALUE && FORMAT == SOLAR_KV_FP8) ||
                  (!VALUE && FORMAT != SOLAR_KV_FP4)) {
        return solar_fattn_e4m3(data[elem]) * scale;
    } else {
        const uint8_t packed = data[elem >> 1u];
        const uint8_t code = (elem & 1u) ? packed >> 4u : packed & 0x0fu;
        return solar_fattn_e2m1(code) * scale;
    }
}

template <int FORMAT>
__device__ __forceinline__ void solar_fattn_kv_bytes(
        uint32_t  n_head_kv,
        uint64_t *k_bytes,
        uint64_t *v_bytes) {
    const uint64_t kv_dim = (uint64_t)n_head_kv * FA_HD;
    *k_bytes = FORMAT == SOLAR_KV_FP4 ? kv_dim / 2u : kv_dim;
    *v_bytes = FORMAT == SOLAR_KV_FP8 ? kv_dim : kv_dim / 2u;
}

/* Decode one 32-key (or shorter) K/V tile.  Per-row K/V scales are read
 * once and reused across the 128 dims.  Compressed formats convert four
 * consecutive dims per thread so the packed row is touched with aligned
 * 4-byte / 2-byte loads. */
template <int FORMAT>
__device__ __forceinline__ void solar_fattn_fill_kv_tile(
        __half        s_k[][FA_ROW],
        __half        s_v[][FA_ROW],
        const void   *kv,
        uint64_t      row_bytes,
        uint32_t      n_head_kv,
        uint32_t      kvh,
        uint32_t      kv_dim,
        uint32_t      kv_cap,
        uint32_t      kt0,
        uint32_t      tile_len) {
    __shared__ float scale_k[FA_TK];
    __shared__ float scale_v[FA_TK];
    if (threadIdx.x < FA_TK) {
        const uint32_t r = threadIdx.x;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        if constexpr (FORMAT == SOLAR_KV_BF16) {
            scale_k[r] = 1.0f;
            scale_v[r] = 1.0f;
        } else {
            uint64_t k_bytes = 0, v_bytes = 0;
            solar_fattn_kv_bytes<FORMAT>(n_head_kv, &k_bytes, &v_bytes);
            const uint8_t *row = (const uint8_t *)kv +
                (uint64_t)(src % kv_cap) * row_bytes;
            const __half *scales = (const __half *)(row + k_bytes + v_bytes);
            scale_k[r] = __half2float(scales[kvh]);
            scale_v[r] = __half2float(scales[n_head_kv + kvh]);
        }
    }
    __syncthreads();

    constexpr uint32_t NGRP = FA_HD / 4u;
    for (uint32_t idx = threadIdx.x; idx < FA_TK * NGRP; idx += blockDim.x) {
        const uint32_t r = idx / NGRP;
        const uint32_t g = idx - r * NGRP;
        const uint32_t c = g * 4u;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        if constexpr (FORMAT == SOLAR_KV_BF16) {
            const __half *row = (const __half *)kv +
                (size_t)(src % kv_cap) * kv_dim * 2u +
                (size_t)kvh * FA_HD;
            const float2 k2 = *reinterpret_cast<const float2 *>(row + c);
            const float2 v2 = *reinterpret_cast<const float2 *>(
                row + kv_dim + c);
            *reinterpret_cast<float2 *>(&s_k[r][c]) = k2;
            *reinterpret_cast<float2 *>(&s_v[r][c]) = v2;
        } else {
            uint64_t k_bytes = 0, v_bytes = 0;
            solar_fattn_kv_bytes<FORMAT>(n_head_kv, &k_bytes, &v_bytes);
            const uint8_t *row = (const uint8_t *)kv +
                (uint64_t)(src % kv_cap) * row_bytes;
            const float sk = scale_k[r];
            const float sv = scale_v[r];
            const uint64_t ke = (uint64_t)kvh * FA_HD + c;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s_k[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_kv_load<FORMAT, false>(
                        row, n_head_kv, kvh, c + (uint32_t)i));
                s_v[r][c + (uint32_t)i] = __float2half(
                    solar_fattn_kv_load<FORMAT, true>(
                        row, n_head_kv, kvh, c + (uint32_t)i));
                (void)sk;
                (void)sv;
                (void)ke;
            }
        }
    }
}

/* Specialized 4-wide dequant for the production K-FP8/V-FP4 row so the
 * per-row scale is applied from registers and the packed bytes are read
 * with aligned word loads.  Other formats keep the scalar helper above. */
template <>
__device__ __forceinline__ void solar_fattn_fill_kv_tile<SOLAR_KV_KFP8_VFP4>(
        __half        s_k[][FA_ROW],
        __half        s_v[][FA_ROW],
        const void   *kv,
        uint64_t      row_bytes,
        uint32_t      n_head_kv,
        uint32_t      kvh,
        uint32_t      kv_dim,
        uint32_t      kv_cap,
        uint32_t      kt0,
        uint32_t      tile_len) {
    (void)kv_dim;
    __shared__ float scale_k[FA_TK];
    __shared__ float scale_v[FA_TK];
    const uint64_t k_bytes = (uint64_t)n_head_kv * FA_HD;
    const uint64_t v_bytes = k_bytes / 2u;
    if (threadIdx.x < FA_TK) {
        const uint32_t r = threadIdx.x;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        const uint8_t *row = (const uint8_t *)kv +
            (uint64_t)(src % kv_cap) * row_bytes;
        const __half *scales = (const __half *)(row + k_bytes + v_bytes);
        scale_k[r] = __half2float(scales[kvh]);
        scale_v[r] = __half2float(scales[n_head_kv + kvh]);
    }
    __syncthreads();

    constexpr uint32_t NGRP = FA_HD / 4u;
    for (uint32_t idx = threadIdx.x; idx < FA_TK * NGRP; idx += blockDim.x) {
        const uint32_t r = idx / NGRP;
        const uint32_t g = idx - r * NGRP;
        const uint32_t c = g * 4u;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        const uint8_t *row = (const uint8_t *)kv +
            (uint64_t)(src % kv_cap) * row_bytes;
        const float sk = scale_k[r];
        const float sv = scale_v[r];
        const uint64_t kbase = (uint64_t)kvh * FA_HD + c;
        const uint32_t kpack = *reinterpret_cast<const uint32_t *>(
            row + kbase);
        const uint16_t vpack = *reinterpret_cast<const uint16_t *>(
            row + k_bytes + (kbase >> 1u));
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint8_t kc = (uint8_t)(kpack >> (8 * i));
            const uint8_t vc = (uint8_t)((vpack >> (4 * i)) & 0x0fu);
            s_k[r][c + (uint32_t)i] = __float2half(
                solar_fattn_e4m3(kc) * sk);
            s_v[r][c + (uint32_t)i] = __float2half(
                solar_fattn_e2m1(vc) * sv);
        }
    }
}

template <>
__device__ __forceinline__ void solar_fattn_fill_kv_tile<SOLAR_KV_FP8>(
        __half        s_k[][FA_ROW],
        __half        s_v[][FA_ROW],
        const void   *kv,
        uint64_t      row_bytes,
        uint32_t      n_head_kv,
        uint32_t      kvh,
        uint32_t      kv_dim,
        uint32_t      kv_cap,
        uint32_t      kt0,
        uint32_t      tile_len) {
    (void)kv_dim;
    __shared__ float scale_k[FA_TK];
    __shared__ float scale_v[FA_TK];
    const uint64_t kv_elems = (uint64_t)n_head_kv * FA_HD;
    if (threadIdx.x < FA_TK) {
        const uint32_t r = threadIdx.x;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        const uint8_t *row = (const uint8_t *)kv +
            (uint64_t)(src % kv_cap) * row_bytes;
        const __half *scales = (const __half *)(row + kv_elems + kv_elems);
        scale_k[r] = __half2float(scales[kvh]);
        scale_v[r] = __half2float(scales[n_head_kv + kvh]);
    }
    __syncthreads();

    constexpr uint32_t NGRP = FA_HD / 4u;
    for (uint32_t idx = threadIdx.x; idx < FA_TK * NGRP; idx += blockDim.x) {
        const uint32_t r = idx / NGRP;
        const uint32_t g = idx - r * NGRP;
        const uint32_t c = g * 4u;
        const uint32_t src = r < tile_len ? kt0 + r : kt0;
        const uint8_t *row = (const uint8_t *)kv +
            (uint64_t)(src % kv_cap) * row_bytes;
        const float sk = scale_k[r];
        const float sv = scale_v[r];
        const uint64_t e = (uint64_t)kvh * FA_HD + c;
        const uint32_t kpack = *reinterpret_cast<const uint32_t *>(row + e);
        const uint32_t vpack = *reinterpret_cast<const uint32_t *>(
            row + kv_elems + e);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            s_k[r][c + (uint32_t)i] = __float2half(
                solar_fattn_e4m3((uint8_t)(kpack >> (8 * i))) * sk);
            s_v[r][c + (uint32_t)i] = __float2half(
                solar_fattn_e4m3((uint8_t)(vpack >> (8 * i))) * sv);
        }
    }
}

__device__ __forceinline__ void solar_fattn_consume_16(
        tile_c          output[FA_HD / 8],
        float           row_m[2],
        float           row_l[2],
        const tile_a    qa[FA_HD / 16],
        const __half  (*s_k)[FA_ROW],
        const __half  (*s_v)[FA_ROW],
        const bool      alive[2],
        const uint32_t  qpos[2],
        const uint32_t  qfirst[2],
        uint32_t        kt0,
        uint32_t        tile_len,
        uint32_t        lane,
        float           scale) {
    if (tile_len == 0u) return;
    tile_c scores[2];
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
        tile_c zero;
        scores[nb] = zero;
#pragma unroll
        for (int kc = 0; kc < FA_HD / 16; kc++) {
            tile_b keys;
#pragma unroll
            for (int l = 0; l < tile_b::ne; l++) {
                const int i = (int)(lane / 4);
                const int j = l * 4 + (int)(lane % 4);
                keys.x[l] = *(const half2 *)&s_k[
                    nb * 8 + i][kc * 16 + 2 * j];
            }
            mma(scores[nb], qa[kc], keys);
        }
    }

    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const uint32_t p =
                kt0 + nb * 8u + (lane % 4u) * 2u + (l % 2u);
            float score = scores[nb].x[l] * scale;
            if (!alive[r] || p > qpos[r] || p < qfirst[r] ||
                p >= kt0 + tile_len) {
                score = -INFINITY;
            }
            scores[nb].x[l] = score;
            tile_max[r] = fmaxf(tile_max[r], score);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_max[r] = fmaxf(
            tile_max[r],
            __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
        tile_max[r] = fmaxf(
            tile_max[r],
            __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
    }

    float rescale[2];
    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const float next_max = fmaxf(row_m[r], tile_max[r]);
        rescale[r] = row_m[r] == -INFINITY
            ? 0.0f : __expf(row_m[r] - next_max);
        row_m[r] = next_max;
    }
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            const float weight =
                scores[nb].x[l] == -INFINITY || row_m[r] == -INFINITY
                    ? 0.0f : __expf(scores[nb].x[l] - row_m[r]);
            scores[nb].x[l] = weight;
            tile_sum[r] += weight;
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        tile_sum[r] +=
            __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
        tile_sum[r] +=
            __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
        row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
    }

    tile_a probabilities;
#pragma unroll
    for (int l = 0; l < tile_a::ne; l++) {
        probabilities.x[l] = __floats2half2_rn(
            scores[l / 2].x[(l % 2) * 2],
            scores[l / 2].x[(l % 2) * 2 + 1]);
    }
#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            output[cb].x[l] *= rescale[l / 2];
        }
        tile_b values;
#pragma unroll
        for (int l = 0; l < tile_b::ne; l++) {
            const int i = (int)(lane / 4);
            const int j = l * 4 + (int)(lane % 4);
            values.x[l] = __halves2half2(
                s_v[2 * j][cb * 8 + i],
                s_v[2 * j + 1][cb * 8 + i]);
        }
        mma(output[cb], probabilities, values);
    }
}

template <int KV_FORMAT>
__global__ void ds4_fattn_hmma_kernel(
        float * __restrict__ heads,
        const float * __restrict__ q,
        const void * __restrict__ kv,
        const uint64_t row_bytes,
        const uint32_t n_tokens,
        const uint32_t pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const uint32_t kv_cap,
        const uint32_t window,
        const float scale) {
    const uint32_t tq0 = blockIdx.x * FA_TQ;
    const uint32_t h = blockIdx.y;
    if (tq0 >= n_tokens || h >= n_head) return;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = h / group;
    const uint32_t kv_dim = n_head_kv * FA_HD;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;

    __shared__ __half s_q[FA_TQ][FA_ROW];
    __shared__ __half s_k[FA_TK][FA_ROW];
    __shared__ __half s_v[FA_TK][FA_ROW];

    for (uint32_t idx = threadIdx.x; idx < FA_TQ * FA_HD;
         idx += blockDim.x) {
        const uint32_t r = idx / FA_HD;
        const uint32_t c = idx - r * FA_HD;
        const uint32_t t = tq0 + r < n_tokens ? tq0 + r : 0u;
        s_q[r][c] = __float2half(
            q[((size_t)t * n_head + h) * FA_HD + c]);
    }
    __syncthreads();

    const uint32_t qrow[2] = {
        warp * FA_WQ + lane / 4,
        warp * FA_WQ + lane / 4 + 8u,
    };
    uint32_t qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = tq0 + qrow[r] < n_tokens;
        qpos[r] = alive[r] ? pos0 + tq0 + qrow[r] : pos0;
        qfirst[r] = window && qpos[r] + 1u > window
            ? qpos[r] + 1u - window : 0u;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    tile_a qa[FA_HD / 16];
#pragma unroll
    for (int kc = 0; kc < FA_HD / 16; kc++) {
#pragma unroll
        for (int l = 0; l < tile_a::ne; l++) {
            const int i = (l % 2) * 8 + (int)(lane / 4);
            const int j = (l / 2) * 4 + (int)(lane % 4);
            qa[kc].x[l] = *(const half2 *)&s_q[
                warp * FA_WQ + i][kc * 16 + 2 * j];
        }
    }

    tile_c output[FA_HD / 8];
    uint32_t block_first = 0xffffffffu;
    uint32_t block_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!alive[r]) continue;
        block_first = min(block_first, qfirst[r]);
        block_last = max(block_last, qpos[r]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_first = min(
            block_first,
            __shfl_xor_sync(0xffffffffu, block_first, offset));
        block_last = max(
            block_last,
            __shfl_xor_sync(0xffffffffu, block_last, offset));
    }
    __shared__ uint32_t shared_first;
    __shared__ uint32_t shared_last;
    if (threadIdx.x == 0u) {
        shared_first = 0xffffffffu;
        shared_last = 0u;
    }
    __syncthreads();
    if (lane == 0u) {
        atomicMin(&shared_first, block_first);
        atomicMax(&shared_last, block_last);
    }
    __syncthreads();
    block_first = shared_first;
    block_last = shared_last;
    if (block_first == 0xffffffffu) return;

    for (uint32_t kt0 = block_first; kt0 <= block_last; kt0 += FA_TK) {
        const uint32_t remaining = block_last - kt0 + 1u;
        const uint32_t tile_len = remaining < (uint32_t)FA_TK
            ? remaining : (uint32_t)FA_TK;
        solar_fattn_fill_kv_tile<KV_FORMAT>(
            s_k, s_v, kv, row_bytes, n_head_kv, kvh, kv_dim, kv_cap,
            kt0, tile_len);
        __syncthreads();
        const uint32_t first_len = tile_len < (uint32_t)FA_CONSUME
            ? tile_len : (uint32_t)FA_CONSUME;
        solar_fattn_consume_16(
            output, row_m, row_l, qa, s_k, s_v, alive, qpos, qfirst,
            kt0, first_len, lane, scale);
        if (tile_len > (uint32_t)FA_CONSUME) {
            solar_fattn_consume_16(
                output, row_m, row_l, qa, s_k + FA_CONSUME, s_v + FA_CONSUME,
                alive, qpos, qfirst, kt0 + (uint32_t)FA_CONSUME,
                tile_len - (uint32_t)FA_CONSUME, lane, scale);
        }
        __syncthreads();
    }

#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r] || row_l[r] <= 0.0f) continue;
            const uint32_t token = tq0 + qrow[r];
            const int col = (int)(lane % 4) * 2 + (l % 2);
            heads[((size_t)token * n_head + h) * FA_HD + cb * 8 + col] =
                output[cb].x[l] / row_l[r];
        }
    }
}

__device__ __forceinline__ void solar_fattn_load_qa(
        tile_a        qa[FA_HD / 16],
        const float  *q,
        uint32_t      tq0,
        uint32_t      n_tokens,
        uint32_t      n_head,
        uint32_t      h,
        uint32_t      warp,
        uint32_t      lane) {
#pragma unroll
    for (int kc = 0; kc < FA_HD / 16; kc++) {
#pragma unroll
        for (int l = 0; l < tile_a::ne; l++) {
            const int i = (l % 2) * 8 + (int)(lane / 4);
            const int j = (l / 2) * 4 + (int)(lane % 4);
            const uint32_t row = warp * FA_WQ + (uint32_t)i;
            const uint32_t col = (uint32_t)kc * 16u + 2u * (uint32_t)j;
            const uint32_t t = tq0 + row < n_tokens ? tq0 + row : 0u;
            const float2 xy = *reinterpret_cast<const float2 *>(
                q + ((size_t)t * n_head + h) * FA_HD + col);
            qa[kc].x[l] = __floats2half2_rn(xy.x, xy.y);
        }
    }
}

/* Two even-grouped Q heads share one dequantized K/V tile.  Eight warps:
 * 0-3 own head 2*by, 4-7 own head 2*by+1.  Fragment loads use the local
 * lane (threadIdx.x % 32), not raw threadIdx.x / 4, so warps 4-7 keep
 * the same HMMA layout as warps 0-3. */
template <int KV_FORMAT>
__global__ void ds4_fattn_hmma_gqa2_kernel(
        float * __restrict__ heads,
        const float * __restrict__ q,
        const void * __restrict__ kv,
        const uint64_t row_bytes,
        const uint32_t n_tokens,
        const uint32_t pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const uint32_t kv_cap,
        const uint32_t window,
        const float scale) {
    constexpr uint32_t N_Q = 2u;
    constexpr uint32_t GROUP_THREADS = (uint32_t)FA_WARPS * 32u;
    const uint32_t tq0 = blockIdx.x * FA_TQ;
    const uint32_t h0 = blockIdx.y * N_Q;
    if (tq0 >= n_tokens || h0 + 1u >= n_head) return;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = h0 / group;
    const uint32_t kv_dim = n_head_kv * FA_HD;
    const uint32_t local = threadIdx.x % GROUP_THREADS;
    const uint32_t hi = threadIdx.x / GROUP_THREADS;
    const uint32_t h = h0 + hi;
    const uint32_t warp = local >> 5;
    const uint32_t lane = local & 31u;

    __shared__ __half s_k[FA_TK][FA_ROW];
    __shared__ __half s_v[FA_TK][FA_ROW];
    static_assert(sizeof(s_k) + sizeof(s_v) + 2u * sizeof(uint32_t) <=
                      49152u,
                  "GQA-pair FATTN must stay under 48 KiB static shared");

    tile_a qa[FA_HD / 16];
    solar_fattn_load_qa(qa, q, tq0, n_tokens, n_head, h, warp, lane);

    const uint32_t qrow[2] = {
        warp * FA_WQ + lane / 4,
        warp * FA_WQ + lane / 4 + 8u,
    };
    uint32_t qpos[2], qfirst[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = tq0 + qrow[r] < n_tokens;
        qpos[r] = alive[r] ? pos0 + tq0 + qrow[r] : pos0;
        qfirst[r] = window && qpos[r] + 1u > window
            ? qpos[r] + 1u - window : 0u;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    tile_c output[FA_HD / 8];
    uint32_t block_first = 0xffffffffu;
    uint32_t block_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!alive[r]) continue;
        block_first = min(block_first, qfirst[r]);
        block_last = max(block_last, qpos[r]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_first = min(
            block_first,
            __shfl_xor_sync(0xffffffffu, block_first, offset));
        block_last = max(
            block_last,
            __shfl_xor_sync(0xffffffffu, block_last, offset));
    }
    __shared__ uint32_t shared_first;
    __shared__ uint32_t shared_last;
    if (threadIdx.x == 0u) {
        shared_first = 0xffffffffu;
        shared_last = 0u;
    }
    __syncthreads();
    if (lane == 0u) {
        atomicMin(&shared_first, block_first);
        atomicMax(&shared_last, block_last);
    }
    __syncthreads();
    block_first = shared_first;
    block_last = shared_last;
    if (block_first == 0xffffffffu) return;

    for (uint32_t kt0 = block_first; kt0 <= block_last; kt0 += FA_TK) {
        const uint32_t remaining = block_last - kt0 + 1u;
        const uint32_t tile_len = remaining < (uint32_t)FA_TK
            ? remaining : (uint32_t)FA_TK;
        solar_fattn_fill_kv_tile<KV_FORMAT>(
            s_k, s_v, kv, row_bytes, n_head_kv, kvh, kv_dim, kv_cap,
            kt0, tile_len);
        __syncthreads();
        const uint32_t first_len = tile_len < (uint32_t)FA_CONSUME
            ? tile_len : (uint32_t)FA_CONSUME;
        solar_fattn_consume_16(
            output, row_m, row_l, qa, s_k, s_v, alive, qpos, qfirst,
            kt0, first_len, lane, scale);
        if (tile_len > (uint32_t)FA_CONSUME) {
            solar_fattn_consume_16(
                output, row_m, row_l, qa, s_k + FA_CONSUME, s_v + FA_CONSUME,
                alive, qpos, qfirst, kt0 + (uint32_t)FA_CONSUME,
                tile_len - (uint32_t)FA_CONSUME, lane, scale);
        }
        __syncthreads();
    }

#pragma unroll
    for (int cb = 0; cb < FA_HD / 8; cb++) {
#pragma unroll
        for (int l = 0; l < tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r] || row_l[r] <= 0.0f) continue;
            const uint32_t token = tq0 + qrow[r];
            const int col = (int)(lane % 4) * 2 + (l % 2);
            heads[((size_t)token * n_head + h) * FA_HD + cb * 8 + col] =
                output[cb].x[l] / row_l[r];
        }
    }
}

enum {
    M3_FA_QK     = 192,
    M3_FA_V      = 128,
    M3_FA_WQ     = 16,
    M3_FA_WARPS  = 4,
    M3_FA_TQ     = M3_FA_WQ * M3_FA_WARPS,
    M3_FA_TK     = 16,
    M3_FA_PAD    = 8,
    M3_FA_QK_ROW = M3_FA_QK + M3_FA_PAD,
    M3_FA_V_ROW  = M3_FA_V + M3_FA_PAD,
};

typedef tile<16, 8, nv_bfloat162> motif_tile_a;
typedef tile< 8, 8, nv_bfloat162> motif_tile_b;
typedef tile<16, 8, float> motif_tile_c;

/* Motif-3 full-attention prefill follows the compute-friendly MLA path from
 * the official Motif vLLM port: W_UK/W_UV are materialized for one bounded KV
 * chunk and attention runs at QK=192, V=128.  A caller can merge several KV
 * chunks exactly from the returned log-sum-exp values, so no expanded 256K KV
 * cache is required. */
/* GB10 has 100 KiB shared / SM. Staging Q+K+V was 36 KiB and capped the
 * kernel at two CTAs; Q already lives in qa[] for the whole K walk, so
 * loading it once from global frees enough shared memory for three CTAs. */
__global__ __launch_bounds__(M3_FA_WARPS * 32, 3)
void motif3_fattn_hmma_kernel(
        float * __restrict__ heads,
        float * __restrict__ lse,
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const uint32_t n_query,
        const uint32_t query_pos0,
        const uint32_t n_kv,
        const uint32_t kv_pos0,
        const uint32_t n_head,
        const uint32_t n_head_kv,
        const float scale,
        const uint32_t window) {
    const uint32_t tq0 = blockIdx.x * M3_FA_TQ;
    const uint32_t h = blockIdx.y;
    if (tq0 >= n_query || h >= n_head) return;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = h / group;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;

    __shared__ __nv_bfloat16 s_k[M3_FA_TK][M3_FA_QK_ROW];
    __shared__ __nv_bfloat16 s_v[M3_FA_TK][M3_FA_V_ROW];
    static_assert(3u * (sizeof(s_k) + sizeof(s_v) + sizeof(uint32_t)) <=
                      102400u,
                  "Motif FATTN shared memory no longer fits three GB10 CTAs");

    const uint32_t qrow[2] = {
        warp * M3_FA_WQ + lane / 4,
        warp * M3_FA_WQ + lane / 4 + 8u,
    };
    uint32_t qpos[2];
    bool alive[2];
    float row_m[2], row_l[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        alive[r] = tq0 + qrow[r] < n_query;
        qpos[r] = alive[r] ? query_pos0 + tq0 + qrow[r] : query_pos0;
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    motif_tile_a qa[M3_FA_QK / 16];
#pragma unroll
    for (int kc = 0; kc < M3_FA_QK / 16; kc++) {
#pragma unroll
        for (int l = 0; l < motif_tile_a::ne; l++) {
            const int i = (l % 2) * 8 + (int)(lane / 4);
            const int j = (l / 2) * 4 + (int)(lane % 4);
            const uint32_t row = warp * M3_FA_WQ + (uint32_t)i;
            const uint32_t col = (uint32_t)kc * 16u + 2u * (uint32_t)j;
            const uint32_t t = tq0 + row < n_query ? tq0 + row : 0u;
            const float2 xy = *reinterpret_cast<const float2 *>(
                q + ((size_t)t * n_head + h) * M3_FA_QK + col);
            qa[kc].x[l] = __floats2bfloat162_rn(xy.x, xy.y);
        }
    }

    motif_tile_c output[M3_FA_V / 8];
    const uint32_t kv_last = kv_pos0 + n_kv - 1u;
    uint32_t block_last = 0u;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (alive[r]) block_last = max(block_last, min(qpos[r], kv_last));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        block_last = max(
            block_last,
            __shfl_xor_sync(0xffffffffu, block_last, offset));
    __shared__ uint32_t shared_last;
    if (threadIdx.x == 0u) shared_last = 0u;
    __syncthreads();
    if (lane == 0u) atomicMax(&shared_last, block_last);
    __syncthreads();
    block_last = shared_last;

    uint32_t block_first = kv_pos0;
    if (window > 0u) {
        const uint32_t first_qpos = query_pos0 + tq0;
        const uint32_t history = window - 1u;
        if (first_qpos > history)
            block_first = max(block_first, first_qpos - history);
    }
    if (block_last >= block_first) {
        for (uint32_t kt0 = block_first; kt0 <= block_last; kt0 += M3_FA_TK) {
            const uint32_t remaining = block_last - kt0 + 1u;
            const uint32_t tile_len = remaining < (uint32_t)M3_FA_TK
                ? remaining : (uint32_t)M3_FA_TK;
            for (uint32_t idx = threadIdx.x * 4u;
                 idx < M3_FA_TK * M3_FA_QK; idx += blockDim.x * 4u) {
                const uint32_t r = idx / M3_FA_QK;
                const uint32_t c = idx - r * M3_FA_QK;
                const uint32_t src = r < tile_len ? kt0 + r : kt0;
                const uint32_t local = src - kv_pos0;
                const float4 x = *reinterpret_cast<const float4 *>(
                    k + ((size_t)local * n_head_kv + kvh) * M3_FA_QK + c);
                *reinterpret_cast<nv_bfloat162 *>(&s_k[r][c]) =
                    __floats2bfloat162_rn(x.x, x.y);
                *reinterpret_cast<nv_bfloat162 *>(&s_k[r][c + 2]) =
                    __floats2bfloat162_rn(x.z, x.w);
            }
            for (uint32_t idx = threadIdx.x * 4u;
                 idx < M3_FA_TK * M3_FA_V; idx += blockDim.x * 4u) {
                const uint32_t r = idx / M3_FA_V;
                const uint32_t c = idx - r * M3_FA_V;
                const uint32_t src = r < tile_len ? kt0 + r : kt0;
                const uint32_t local = src - kv_pos0;
                const float4 x = *reinterpret_cast<const float4 *>(
                    v + ((size_t)local * n_head_kv + kvh) * M3_FA_V + c);
                *reinterpret_cast<nv_bfloat162 *>(&s_v[r][c]) =
                    __floats2bfloat162_rn(x.x, x.y);
                *reinterpret_cast<nv_bfloat162 *>(&s_v[r][c + 2]) =
                    __floats2bfloat162_rn(x.z, x.w);
            }
            __syncthreads();

            motif_tile_c scores[2];
#pragma unroll
            for (int nb = 0; nb < 2; nb++) {
                motif_tile_c zero;
                scores[nb] = zero;
#pragma unroll
                for (int kc = 0; kc < M3_FA_QK / 16; kc++) {
                    motif_tile_b keys;
#pragma unroll
                    for (int l = 0; l < motif_tile_b::ne; l++) {
                        const int i = (int)(lane / 4);
                        const int j = l * 4 + (int)(lane % 4);
                        keys.x[l] = *(const nv_bfloat162 *)&s_k[
                            nb * 8 + i][kc * 16 + 2 * j];
                    }
                    mma(scores[nb], qa[kc], keys);
                }
            }

            float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int nb = 0; nb < 2; nb++) {
#pragma unroll
                for (int l = 0; l < motif_tile_c::ne; l++) {
                    const int r = l / 2;
                    const uint32_t p =
                        kt0 + nb * 8u + (lane % 4u) * 2u + (l % 2u);
                    float score = scores[nb].x[l] * scale;
                    if (!alive[r] || p > qpos[r] ||
                        (window > 0u && p + window <= qpos[r]) ||
                        p >= kt0 + tile_len || p > kv_last) {
                        score = -INFINITY;
                    }
                    scores[nb].x[l] = score;
                    tile_max[r] = fmaxf(tile_max[r], score);
                }
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                tile_max[r] = fmaxf(
                    tile_max[r],
                    __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
                tile_max[r] = fmaxf(
                    tile_max[r],
                    __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            }

            float rescale[2];
            float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
            for (int r = 0; r < 2; r++) {
                const float next_max = fmaxf(row_m[r], tile_max[r]);
                rescale[r] = row_m[r] == -INFINITY
                    ? 0.0f : __expf(row_m[r] - next_max);
                row_m[r] = next_max;
            }
#pragma unroll
            for (int nb = 0; nb < 2; nb++) {
#pragma unroll
                for (int l = 0; l < motif_tile_c::ne; l++) {
                    const int r = l / 2;
                    const float weight =
                        scores[nb].x[l] == -INFINITY || row_m[r] == -INFINITY
                            ? 0.0f : __expf(scores[nb].x[l] - row_m[r]);
                    scores[nb].x[l] = weight;
                    tile_sum[r] += weight;
                }
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                tile_sum[r] +=
                    __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
                tile_sum[r] +=
                    __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
                row_l[r] = row_l[r] * rescale[r] + tile_sum[r];
            }

            motif_tile_a probabilities;
#pragma unroll
            for (int l = 0; l < motif_tile_a::ne; l++) {
                probabilities.x[l] = __floats2bfloat162_rn(
                    scores[l / 2].x[(l % 2) * 2],
                    scores[l / 2].x[(l % 2) * 2 + 1]);
            }
#pragma unroll
            for (int cb = 0; cb < M3_FA_V / 8; cb++) {
#pragma unroll
                for (int l = 0; l < motif_tile_c::ne; l++)
                    output[cb].x[l] *= rescale[l / 2];
                motif_tile_b values;
#pragma unroll
                for (int l = 0; l < motif_tile_b::ne; l++) {
                    const int i = (int)(lane / 4);
                    const int j = l * 4 + (int)(lane % 4);
                    values.x[l] = __halves2bfloat162(
                        s_v[2 * j][cb * 8 + i],
                        s_v[2 * j + 1][cb * 8 + i]);
                }
                mma(output[cb], probabilities, values);
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int cb = 0; cb < M3_FA_V / 8; cb++) {
#pragma unroll
        for (int l = 0; l < motif_tile_c::ne; l++) {
            const int r = l / 2;
            if (!alive[r] || row_l[r] <= 0.0f) continue;
            const uint32_t token = tq0 + qrow[r];
            const int col = (int)(lane % 4) * 2 + (l % 2);
            heads[((size_t)token * n_head + h) * M3_FA_V +
                  cb * 8 + col] = output[cb].x[l] / row_l[r];
        }
    }
    if (lse && (lane & 3u) == 0u) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            if (!alive[r] || row_l[r] <= 0.0f) continue;
            const uint32_t token = tq0 + qrow[r];
            lse[(size_t)token * n_head + h] = row_m[r] + logf(row_l[r]);
        }
    }
}

}  // namespace

static int solar_fattn_gqa_pair(int n_head, int n_head_kv) {
    if (n_head_kv <= 0 || n_head % n_head_kv != 0) return 0;
    const int group = n_head / n_head_kv;
    if (group < 2 || (group % 2) != 0) return 0;
    /* Diagnostic only: DS4_SOLAR_FATTN_GQA2=0 restores the one-head
     * kernel so tests can compare the pair path against it. */
    const char *value = getenv("DS4_SOLAR_FATTN_GQA2");
    if (value && value[0] == '0') return 0;
    return 1;
}

template <int FORMAT>
static void solar_fattn_launch(
        float *heads, const float *q, const void *kv, uint64_t row_bytes,
        int n_tokens, int pos0, int n_head, int n_head_kv, int kv_cap,
        int window, float scale, cudaStream_t stream) {
    const int tiles = (n_tokens + FA_TQ - 1) / FA_TQ;
    if (solar_fattn_gqa_pair(n_head, n_head_kv)) {
        const dim3 grid(tiles, n_head / 2, 1);
        ds4_fattn_hmma_gqa2_kernel<FORMAT>
            <<<grid, FA_WARPS * 32 * 2, 0, stream>>>(
                heads, q, kv, row_bytes, (uint32_t)n_tokens,
                (uint32_t)pos0, (uint32_t)n_head, (uint32_t)n_head_kv,
                (uint32_t)kv_cap, (uint32_t)window, scale);
        return;
    }
    const dim3 grid(tiles, n_head, 1);
    ds4_fattn_hmma_kernel<FORMAT>
        <<<grid, FA_WARPS * 32, 0, stream>>>(
            heads, q, kv, row_bytes, (uint32_t)n_tokens, (uint32_t)pos0,
            (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)kv_cap,
            (uint32_t)window, scale);
}

extern "C" int ds4_mmq_exaone_prefill_attn_hmma(
        float *heads, const float *q, const void *kv,
        int n_tokens, int pos0, int n_head, int n_head_kv, int head_dim,
        int kv_cap, int window, float scale, cudaStream_t stream) {
    if (!heads || !q || !kv || n_tokens <= 0 || pos0 < 0 || n_head <= 0 ||
        n_head_kv <= 0 || head_dim != FA_HD || kv_cap <= 0 ||
        n_head % n_head_kv != 0) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    solar_fattn_launch<SOLAR_KV_BF16>(
        heads, q, kv, 0u, n_tokens, pos0, n_head, n_head_kv, kv_cap,
        window, scale, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_mmq_solar_prefill_attn_hmma(
        float *heads, const float *q, const void *kv,
        int format, size_t row_bytes,
        int n_tokens, int pos0, int n_head, int n_head_kv, int head_dim,
        int kv_cap, int window, float scale, cudaStream_t stream) {
    if (!heads || !q || !kv || format < SOLAR_KV_FP8 ||
        format > SOLAR_KV_KFP8_VFP4 || row_bytes == 0u || n_tokens <= 0 ||
        pos0 < 0 || n_head <= 0 || n_head_kv <= 0 || head_dim != FA_HD ||
        kv_cap <= 0 || n_head % n_head_kv != 0) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    switch (format) {
    case SOLAR_KV_FP8:
        solar_fattn_launch<SOLAR_KV_FP8>(
            heads, q, kv, (uint64_t)row_bytes, n_tokens, pos0, n_head,
            n_head_kv, kv_cap, window, scale, stream);
        break;
    case SOLAR_KV_FP4:
        solar_fattn_launch<SOLAR_KV_FP4>(
            heads, q, kv, (uint64_t)row_bytes, n_tokens, pos0, n_head,
            n_head_kv, kv_cap, window, scale, stream);
        break;
    case SOLAR_KV_KFP8_VFP4:
        solar_fattn_launch<SOLAR_KV_KFP8_VFP4>(
            heads, q, kv, (uint64_t)row_bytes, n_tokens, pos0, n_head,
            n_head_kv, kv_cap, window, scale, stream);
        break;
    default:
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_mmq_motif3_prefill_attn_hmma(
        float *heads, float *lse, const float *q,
        const float *k, const float *v,
        int n_query, int query_pos0, int n_kv, int kv_pos0,
        int n_head, int n_head_kv, int qk_dim, int v_dim,
        float scale, int window, cudaStream_t stream) {
    if (!heads || !q || !k || !v || n_query <= 0 || query_pos0 < 0 ||
        n_kv <= 0 || kv_pos0 < 0 || n_head <= 0 || n_head_kv <= 0 ||
        n_head % n_head_kv != 0 || qk_dim != M3_FA_QK ||
        v_dim != M3_FA_V || kv_pos0 > query_pos0 || window < 0) {
        return -1;
    }
    const int device = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) return -1;
    const dim3 grid((n_query + M3_FA_TQ - 1) / M3_FA_TQ, n_head, 1);
    motif3_fattn_hmma_kernel<<<grid, M3_FA_WARPS * 32, 0, stream>>>(
        heads, lse, q, k, v,
        (uint32_t)n_query, (uint32_t)query_pos0,
        (uint32_t)n_kv, (uint32_t)kv_pos0,
        (uint32_t)n_head, (uint32_t)n_head_kv, scale, (uint32_t)window);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
