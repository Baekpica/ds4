/* proto_mm_ids.cu — P4 prototype: mm_ids_helper<1> fast path for the
 * routed-MoE down matmul (ds4_mmq.cu ds4_mmq_moe_impl: n_expert_used=1,
 * si1=1, sis1=1, nchannels_y=1, n_experts=256).
 *
 * Without a case-1 dispatch the down matmul falls to the generic <0>
 * template: one warp per expert scans all assignment rows with a single
 * active lane (n_expert_used=1 leaves lanes 1..31 idle in the iex loop) —
 * 22.5 ms/launch at W4096 prefill (129 launches = 2.90 s of a 12k
 * admission, sm_121 trace 2026-07-07).  The optimized template at
 * neu_padded=1 covers 32 assignment rows per warp iteration.
 *
 * Gate: ids_src1 / ids_dst / expert_bounds BIT-IDENTICAL to <0> on every
 * leg (uniform routing, skewed routing, ~2% -1 invalid ids mirroring the
 * router NaN path, tiny decode shapes) with the production pre-zero of
 * both id maps; then timing at production shapes.
 *
 * Bit-exactness argument: at n_expert_used_template=1, neu_padded=1, so
 * warp_reduce_any<1> is the identity (its xor-shuffle loop starts at
 * offset 1/2=0), each lane owns exactly one (token, iex=0) probe, the
 * shuffle scan accumulates hit counts from lower-token lanes only, and
 * store[] receives the same (it, iex_used) tuples in the same token order
 * as the generic sequential loop; nex_prev sums the same per-token
 * predicate over disjoint lanes.  Output formulas are shared.
 *
 * Build (GB10):
 *   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 proto_mm_ids.cu -o proto_mm_ids
 */
#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cassert>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static constexpr int WARP_SIZE = 32;

/* ------------------------------------------------------------------ */
/* Vendored from cuda/mmq/mmid.cu + common.cuh (verbatim semantics)     */

template <int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == WARP_SIZE) {
        return __any_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

struct mm_ids_helper_store {
    uint32_t data;
    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }
    __device__ uint32_t it() const { return data & 0x003FFFFF; }
    __device__ uint32_t iex_used() const { return data >> 22; }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

template <int n_expert_used_template>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = WARP_SIZE;
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0;
    int it_compact = 0;

    if constexpr (n_expert_used_template == 0) {
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1;
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }
            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }
            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        static_assert(n_expert_used_template == 6 || warp_size % n_expert_used_template == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used;
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded;
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            const int it_compact_add_self = warp_reduce_any<1>(iex_used != -1);

            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    __syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }
    expert_bounds[expert] = nex_prev;
    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }
    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

/* NOTE: the proto's <1> instantiation uses warp_reduce_any<1> where the
 * vendored kernel writes warp_reduce_any<neu_padded>; neu_padded is a
 * (const) runtime int there but the template arg must be constexpr — the
 * vendored code relies on n_expert_used_template making neu_padded
 * effectively constexpr per instantiation.  Semantics identical at 1. */

/* ------------------------------------------------------------------ */

static constexpr int N_EXPERTS = 256;

struct leg {
    const char *name;
    int n_tokens;      // assignment rows
    int invalid_pct;   // % of rows with -1 (router NaN path)
    bool skew;         // hot-expert distribution
};

int main() {
    int smpbo = 0;
    CHECK(cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CHECK(cudaFuncSetAttribute((const void*)mm_ids_helper<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smpbo));
    CHECK(cudaFuncSetAttribute((const void*)mm_ids_helper<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smpbo));
    printf("smpbo=%d\n", smpbo);

    const leg legs[] = {
        {"W4096x6 uniform",   4096*6, 0, false},
        {"W4096x6 2%invalid", 4096*6, 2, false},
        {"W4096x6 skewed",    4096*6, 1, true},
        {"W2048x6 uniform",   2048*6, 2, false},
        {"decode w1x6",       6,      0, false},
        {"decode w8x6",       48,     0, false},
    };

    const int si1 = 1, sis1 = 1, nchannels_y = 1, n_expert_used = 1;
    int failures = 0;

    for (const leg &L : legs) {
        const int nt = L.n_tokens;
        const size_t smem = (size_t)nt * 4;
        assert(smem <= (size_t)smpbo);

        int32_t *h_ids = (int32_t*)malloc(nt * sizeof(int32_t));
        srand(20260707);
        for (int i = 0; i < nt; i++) {
            if (L.invalid_pct && rand() % 100 < L.invalid_pct) { h_ids[i] = -1; continue; }
            h_ids[i] = L.skew ? (rand() % 100 < 60 ? rand() % 8 : rand() % N_EXPERTS)
                              : rand() % N_EXPERTS;
        }

        int32_t *d_ids, *d_src1_a, *d_dst_a, *d_eb_a, *d_src1_b, *d_dst_b, *d_eb_b;
        CHECK(cudaMalloc(&d_ids, nt * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_src1_a, nt * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_dst_a,  nt * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_eb_a,   (N_EXPERTS+1) * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_src1_b, nt * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_dst_b,  nt * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_eb_b,   (N_EXPERTS+1) * sizeof(int32_t)));
        CHECK(cudaMemcpy(d_ids, h_ids, nt * sizeof(int32_t), cudaMemcpyHostToDevice));

        const dim3 grid(N_EXPERTS, 1, 1), block(WARP_SIZE, 1, 1);
        // production pre-zeroes both id maps (dropped -1 rows leave tails unwritten)
        CHECK(cudaMemset(d_src1_a, 0, nt * sizeof(int32_t)));
        CHECK(cudaMemset(d_dst_a,  0, nt * sizeof(int32_t)));
        CHECK(cudaMemset(d_src1_b, 0, nt * sizeof(int32_t)));
        CHECK(cudaMemset(d_dst_b,  0, nt * sizeof(int32_t)));

        mm_ids_helper<0><<<grid, block, smem>>>(d_ids, d_src1_a, d_dst_a, d_eb_a, nt, n_expert_used, nchannels_y, si1, sis1);
        mm_ids_helper<1><<<grid, block, smem>>>(d_ids, d_src1_b, d_dst_b, d_eb_b, nt, n_expert_used, nchannels_y, si1, sis1);
        CHECK(cudaDeviceSynchronize());

        int32_t *h_a = (int32_t*)malloc(nt * sizeof(int32_t));
        int32_t *h_b = (int32_t*)malloc(nt * sizeof(int32_t));
        int32_t h_eba[N_EXPERTS+1], h_ebb[N_EXPERTS+1];
        bool ok = true;
        CHECK(cudaMemcpy(h_a, d_src1_a, nt*4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_b, d_src1_b, nt*4, cudaMemcpyDeviceToHost));
        ok = ok && memcmp(h_a, h_b, nt*4) == 0;
        CHECK(cudaMemcpy(h_a, d_dst_a, nt*4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_b, d_dst_b, nt*4, cudaMemcpyDeviceToHost));
        ok = ok && memcmp(h_a, h_b, nt*4) == 0;
        CHECK(cudaMemcpy(h_eba, d_eb_a, (N_EXPERTS+1)*4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_ebb, d_eb_b, (N_EXPERTS+1)*4, cudaMemcpyDeviceToHost));
        ok = ok && memcmp(h_eba, h_ebb, (N_EXPERTS+1)*4) == 0;

        // timing (50 reps each after 5 warmups)
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        float ms_a = 0, ms_b = 0;
        for (int w = 0; w < 5; w++) mm_ids_helper<0><<<grid, block, smem>>>(d_ids, d_src1_a, d_dst_a, d_eb_a, nt, n_expert_used, nchannels_y, si1, sis1);
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 50; r++) mm_ids_helper<0><<<grid, block, smem>>>(d_ids, d_src1_a, d_dst_a, d_eb_a, nt, n_expert_used, nchannels_y, si1, sis1);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms_a, t0, t1));
        for (int w = 0; w < 5; w++) mm_ids_helper<1><<<grid, block, smem>>>(d_ids, d_src1_b, d_dst_b, d_eb_b, nt, n_expert_used, nchannels_y, si1, sis1);
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 50; r++) mm_ids_helper<1><<<grid, block, smem>>>(d_ids, d_src1_b, d_dst_b, d_eb_b, nt, n_expert_used, nchannels_y, si1, sis1);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms_b, t0, t1));

        printf("%-18s nt=%-6d %s  <0> %8.3f ms  <1> %8.3f ms  speedup %.1fx\n",
               L.name, nt, ok ? "PARITY-OK " : "MISMATCH!!", ms_a/50, ms_b/50, ms_a/ms_b);
        if (!ok) failures++;

        free(h_ids); free(h_a); free(h_b);
        cudaFree(d_ids);
        cudaFree(d_src1_a); cudaFree(d_dst_a); cudaFree(d_eb_a);
        cudaFree(d_src1_b); cudaFree(d_dst_b); cudaFree(d_eb_b);
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    }

    printf(failures ? "PROTO FAIL (%d legs)\n" : "PROTO PASS\n", failures);
    return failures ? 1 : 0;
}
