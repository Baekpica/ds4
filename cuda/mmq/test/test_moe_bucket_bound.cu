// test_moe_bucket_bound.cu — routed-MoE expert-bucket launch-bound gate.
//
// K-EXAONE's down projection flattens [token, top-k] into one single-slot
// row per assignment.  The generic MMQ entry therefore sees
// n_tokens=forward_tokens*top_k even though top-k is sampled without
// replacement, so no expert can own more than forward_tokens rows.  This
// test compares the established unbounded Q3_K/Q4_K entries with the bounded
// entries that receive that tighter, caller-proven maximum.
//
// Build from the repository root (GB10):
//   nvcc -O3 --use_fast_math -std=c++17 \
//        -gencode arch=compute_121a,code=sm_121a -Icuda/mmq \
//        cuda/mmq/test/test_moe_bucket_bound.cu \
//        cuda/mmq/{ds4_ggml_stubs,ds4_mmq,ds4_mmq_d2r,quantize,mmid,mmvq}.o \
//        -lcudart -lcublas -lcuda -o test_moe_bucket_bound

#include "ds4_mmq.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *src, uint64_t in_dim, const void **out) {
    (void)src;
    (void)in_dim;
    (void)out;
    return 0;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA ERR %s @%d: %s\n", #x, __LINE__, \
                 cudaGetErrorString(e_)); \
    std::exit(1); \
} } while (0)

using moe_fn = int (*)(const void *, const float *, const int32_t *, float *,
                       int, int, int, int, int, cudaStream_t);
using bounded_moe_fn = int (*)(const void *, const float *, const int32_t *,
                               float *, int, int, int, int, int, int,
                               cudaStream_t);

static void make_finite_blocks(std::vector<uint8_t> &w, size_t block_bytes,
                               size_t scale_offset, bool has_min,
                               std::mt19937 &rng) {
    for (uint8_t &v : w) v = (uint8_t)(rng() & 0xffu);
    const uint16_t d = 0x2400u; // finite fp16 scale
    for (size_t off = 0; off < w.size(); off += block_bytes) {
        std::memcpy(w.data() + off + scale_offset, &d, sizeof(d));
        if (has_min) {
            std::memcpy(w.data() + off + scale_offset + sizeof(d),
                        &d, sizeof(d));
        }
    }
}

static bool run_case(const char *name, size_t block_bytes,
                     size_t scale_offset, bool has_min,
                     moe_fn reference_fn, bounded_moe_fn bounded_fn,
                     int iters, std::mt19937 &rng) {
    constexpr int M = 512;
    constexpr int K = 2048;
    constexpr int n_experts = 128;
    // Match one production prefill chunk: 2048 forward tokens become 16384
    // flattened down rows, while the largest expert bucket is ~128 rows for
    // this balanced deterministic router pattern.
    constexpr int forward_tokens = 2048;
    constexpr int top_k = 8;
    constexpr int assignment_rows = forward_tokens * top_k;

    const size_t n_weight_blocks =
        (size_t)n_experts * M * (K / 256);
    std::vector<uint8_t> w(n_weight_blocks * block_bytes);
    make_finite_blocks(w, block_bytes, scale_offset, has_min, rng);

    std::vector<float> x((size_t)assignment_rows * K);
    for (float &v : x) {
        v = ((float)(rng() % 2001u) - 1000.0f) / 4000.0f;
    }

    // Each original token chooses eight distinct experts.  Use a fixed-seed
    // random base and an odd stride so the aggregate bucket widths have the
    // same small imbalance as a real router instead of landing at exactly
    // 128 rows/expert.  Flattening keeps the same order as production's
    // routed-mid buffer and selected-id table.
    std::vector<int32_t> ids(assignment_rows);
    std::vector<int> counts(n_experts, 0);
    for (int t = 0; t < forward_tokens; ++t) {
        const int base = (int)(rng() % n_experts);
        const int stride = 2 * (int)(rng() % (n_experts / 2)) + 1;
        for (int s = 0; s < top_k; ++s) {
            const int expert = (base + s * stride) % n_experts;
            ids[t * top_k + s] = expert;
            counts[expert]++;
        }
    }
    const int actual_max = *std::max_element(counts.begin(), counts.end());
    if (actual_max > forward_tokens) {
        std::fprintf(stderr, "%s generator violated bucket contract: %d > %d\n",
                     name, actual_max, forward_tokens);
        return false;
    }

    uint8_t *dw = nullptr;
    float *dx = nullptr, *dref = nullptr, *dbounded = nullptr;
    int32_t *dids = nullptr;
    const size_t out_bytes =
        (size_t)assignment_rows * M * sizeof(float);
    CK(cudaMalloc(&dw, w.size()));
    CK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CK(cudaMalloc(&dids, ids.size() * sizeof(int32_t)));
    CK(cudaMalloc(&dref, out_bytes));
    CK(cudaMalloc(&dbounded, out_bytes));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dids, ids.data(), ids.size() * sizeof(int32_t),
                  cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaEvent_t begin, end;
    CK(cudaStreamCreate(&stream));
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));

    int rc = reference_fn(dw, dx, dids, dref, M, K, assignment_rows,
                          n_experts, 1, stream);
    if (rc == 0) {
        rc = bounded_fn(dw, dx, dids, dbounded, M, K, assignment_rows,
                        n_experts, 1, forward_tokens, stream);
    }
    CK(cudaStreamSynchronize(stream));
    if (rc != 0) {
        std::fprintf(stderr, "%s entry returned rc=%d\n", name, rc);
        return false;
    }

    std::vector<float> ref((size_t)assignment_rows * M);
    std::vector<float> bounded(ref.size());
    CK(cudaMemcpy(ref.data(), dref, out_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bounded.data(), dbounded, out_bytes,
                  cudaMemcpyDeviceToHost));

    long double err2 = 0.0L, ref2 = 0.0L;
    double max_abs = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i], b = bounded[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            if (!(std::isnan(a) && std::isnan(b))) bad++;
            continue;
        }
        const double d = a - b;
        err2 += (long double)d * d;
        ref2 += (long double)a * a;
        max_abs = std::max(max_abs, std::fabs(d));
        if (std::fabs(d) > 2e-4 * std::max(1.0, std::fabs(a))) bad++;
    }
    const double rel_rms = std::sqrt((double)(err2 / std::max(ref2, 1e-30L)));

    auto time_ms = [&](int bucket_bound) {
        CK(cudaEventRecord(begin, stream));
        for (int i = 0; i < iters; ++i) {
            const int trc = bucket_bound > 0
                ? bounded_fn(dw, dx, dids, dbounded, M, K, assignment_rows,
                             n_experts, 1, bucket_bound, stream)
                : reference_fn(dw, dx, dids, dref, M, K, assignment_rows,
                               n_experts, 1, stream);
            if (trc != 0) std::exit(2);
        }
        CK(cudaEventRecord(end, stream));
        CK(cudaEventSynchronize(end));
        float total = 0.0f;
        CK(cudaEventElapsedTime(&total, begin, end));
        return total / iters;
    };
    const float ref_ms = time_ms(0);
    const float bounded_ms = time_ms(forward_tokens);
    const float exact_ms = time_ms(actual_max);

    std::printf("%-4s rows=%d actual_max=%d bound=%d reference=%.3f ms "
                "bounded=%.3f ms speedup=%.2fx exact=%.3f ms exact_speedup=%.2fx "
                "rel_rms=%.3e max_abs=%.3e "
                "bad=%zu\n",
                name, assignment_rows, actual_max, forward_tokens,
                ref_ms, bounded_ms, ref_ms / bounded_ms,
                exact_ms, ref_ms / exact_ms,
                rel_rms, max_abs, bad);

    CK(cudaEventDestroy(end));
    CK(cudaEventDestroy(begin));
    CK(cudaStreamDestroy(stream));
    CK(cudaFree(dbounded));
    CK(cudaFree(dref));
    CK(cudaFree(dids));
    CK(cudaFree(dx));
    CK(cudaFree(dw));
    return bad == 0 && rel_rms <= 2e-5;
}

int main() {
    if (ds4_mmq_init(0) != 0) return 1;
    const char *iters_env = std::getenv("DS4_TEST_ITERS");
    const int iters = iters_env && std::atoi(iters_env) > 0
        ? std::atoi(iters_env) : 20;
    std::mt19937 rng(0x4d4f4542u);

    const bool q3_ok = run_case(
        "Q3_K", 110, 108, false,
        ds4_mmq_q3_K_moe, ds4_mmq_q3_K_moe_bounded,
        iters, rng);
    const bool q4_ok = run_case(
        "Q4_K", 144, 0, true,
        ds4_mmq_q4_K_moe, ds4_mmq_q4_K_moe_bounded,
        iters, rng);
    std::printf("moe bucket bound: %s\n", q3_ok && q4_ok ? "PASS" : "FAIL");
    return q3_ok && q4_ok ? 0 : 1;
}
