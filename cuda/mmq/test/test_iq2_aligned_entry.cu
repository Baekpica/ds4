// test_iq2_aligned_entry.cu — M1-Inc1 integration parity gate.
//
// A/B of the PRODUCTION aligned entry (ds4_mmq_iq2_xxs_aligned_moe_vec, fed
// the artifact layout the weight server builds under --repack-iq2-aligned)
// against the production raw-layout vec entry (ds4_mmq_iq2_xxs_moe_vec) at
// the decode shape.  Both entries quantize the activation through
// quantize_row_q8_1_cuda, so only the float accumulation order differs.
//
// The kernel-level A/B with hand-rolled quantize lives in
// proto_iq2_aligned.cu (the original +12% proof); this test locks the
// integrated entry + artifact layout contract.
//
// Build (box):
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 -I.. \
//        test_iq2_aligned_entry.cu ../*.o -lcudart -lcuda -o test_iq2_aligned_entry

#include "ds4_mmq.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

int main(int argc, char **argv) {
    const int M = 2048, K = 4096;
    const int n_experts = 256;
    const int n_slots = 6;
    const int n_idsets = 32;
    const int iters = 2000;
    const int nb = K / 256;
    const long long row_bytes = (long long)nb * 66;
    const double slot_gb = (double)n_slots * M * row_bytes / 1e9;

    if (ds4_mmq_init(0) != 0) { printf("ds4_mmq_init failed\n"); return 1; }

    std::mt19937 rng(argc > 1 ? (uint32_t)atoi(argv[1]) : 1234u);

    // Random bytes are valid IQ2_XXS content; keep block scales sane.
    std::vector<uint8_t> W((size_t)n_experts * M * row_bytes);
    for (auto &b : W) b = (uint8_t)(rng() & 0xff);
    const long long nblk = (long long)n_experts * M * nb;
    for (long long blk = 0; blk < nblk; blk++) {
        uint16_t h = (uint16_t)(0x2c00 | (rng() & 0xff));
        memcpy(&W[blk * 66], &h, 2);
    }

    std::vector<float> X(K);
    for (auto &v : X) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;

    std::vector<int32_t> ids(n_idsets * n_slots);
    for (int s = 0; s < n_idsets; s++)
        for (int j = 0; j < n_slots; j++)
            ids[s * n_slots + j] = (s * n_slots + j) % n_experts;

    // Artifact layout, exactly as the weight server repack kernel builds it:
    //   [__half dq[nblk]][pad to 64B][uint2 qs[nblk*8]]
    const uint64_t expect_bytes = ds4_mmq_iq2_xxs_aligned_bytes(M, K, n_experts);
    const uint64_t dq_bytes = ((uint64_t)nblk * 2u + 63u) & ~63ull;
    const uint64_t art_bytes = dq_bytes + (uint64_t)nblk * 64u;
    if (expect_bytes != art_bytes) {
        printf("aligned_bytes mismatch: helper=%llu local=%llu\n",
               (unsigned long long)expect_bytes, (unsigned long long)art_bytes);
        return 1;
    }
    std::vector<uint8_t> ART(art_bytes);
    for (long long blk = 0; blk < nblk; blk++) {
        memcpy(&ART[blk * 2], &W[blk * 66], 2);
        memcpy(&ART[dq_bytes + (uint64_t)blk * 64u], &W[blk * 66 + 2], 64);
    }

    uint8_t *dW, *dArt; float *dX, *dOutBase, *dOutAl; int32_t *dIds;
    CK(cudaMalloc(&dW, W.size()));
    CK(cudaMalloc(&dArt, ART.size()));
    CK(cudaMalloc(&dX, sizeof(float) * K));
    CK(cudaMalloc(&dOutBase, sizeof(float) * n_slots * M));
    CK(cudaMalloc(&dOutAl, sizeof(float) * n_slots * M));
    CK(cudaMalloc(&dIds, sizeof(int32_t) * ids.size()));
    CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dArt, ART.data(), ART.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, X.data(), sizeof(float) * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dIds, ids.data(), sizeof(int32_t) * ids.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // ---- baseline: raw-layout production vec entry -------------------------
    int rc = ds4_mmq_iq2_xxs_moe_vec(dW, dX, dIds, dOutBase, M, K, 1, n_experts, n_slots, stream);
    if (rc != 0) { printf("baseline moe_vec rc=%d\n", rc); return 1; }
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        (void)ds4_mmq_iq2_xxs_moe_vec(dW, dX, dIds + (i % n_idsets) * n_slots, dOutBase,
                                      M, K, 1, n_experts, n_slots, stream);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_base = 0.0f; CK(cudaEventElapsedTime(&ms_base, e0, e1));
    ms_base /= iters;

    // ---- aligned production entry ------------------------------------------
    rc = ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds, dOutAl, M, K, 1, n_experts, n_slots, stream);
    if (rc != 0) { printf("aligned entry rc=%d\n", rc); return 1; }
    // width != 1 must be rejected so the caller can fall back
    if (ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds, dOutAl, M, K, 2, n_experts, n_slots, stream) == 0) {
        printf("aligned entry accepted n_tokens=2 (must reject)\n");
        return 1;
    }
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        (void)ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds + (i % n_idsets) * n_slots, dOutAl,
                                              M, K, 1, n_experts, n_slots, stream);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_al = 0.0f; CK(cudaEventElapsedTime(&ms_al, e0, e1));
    ms_al /= iters;

    // ---- parity on idset 0 --------------------------------------------------
    (void)ds4_mmq_iq2_xxs_moe_vec(dW, dX, dIds, dOutBase, M, K, 1, n_experts, n_slots, stream);
    (void)ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds, dOutAl, M, K, 1, n_experts, n_slots, stream);
    CK(cudaStreamSynchronize(stream));
    std::vector<float> ob(n_slots * M), oa(n_slots * M);
    CK(cudaMemcpy(ob.data(), dOutBase, sizeof(float) * ob.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(oa.data(), dOutAl, sizeof(float) * oa.size(), cudaMemcpyDeviceToHost));
    double max_rel = 0.0, max_abs = 0.0; int bad = 0;
    for (size_t i = 0; i < ob.size(); i++) {
        const double a = ob[i], b = oa[i];
        const double ad = fabs(a - b);
        const double rd = ad / (fabs(a) > 1e-3 ? fabs(a) : 1e-3);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > 2e-2 && ad > 1e-2) bad++;
    }

    printf("TEST_IQ2_ALIGNED_ENTRY  M=%d K=%d slots=%d experts=%d iters=%d (weights/iter %.1f MB)\n",
           M, K, n_slots, n_experts, iters, slot_gb * 1000.0);
    printf("  baseline moe_vec : %.4f ms  -> %6.1f GB/s\n", ms_base, slot_gb / (ms_base / 1e3));
    printf("  aligned  entry   : %.4f ms  -> %6.1f GB/s   (%+.1f%%)\n",
           ms_al, slot_gb / (ms_al / 1e3), 100.0 * (ms_base / ms_al - 1.0));
    printf("  parity: max_rel=%.3e max_abs=%.3e bad=%d -> %s\n",
           max_rel, max_abs, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 2;
}
