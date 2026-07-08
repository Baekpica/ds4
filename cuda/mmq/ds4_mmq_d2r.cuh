// SPDX-License-Identifier: MIT
// Internal launcher for the gated D2R Q2_K MoE down-GEMM path.

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

bool ds4_mmq_q2_K_moe_d2r_available(int cc);

size_t ds4_mmq_q2_K_moe_d2r_scratch_bytes(int64_t ncols_max, int n_experts);

int ds4_mmq_q2_K_moe_d2r_launch(
    const void    * W_soa,
    int64_t         soa_blocks,
    const void    * q8,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out,
    int             M,
    int             K,
    int64_t         ne_get_rows,
    int             n_experts,
    void          * worklist_scratch,
    size_t          worklist_scratch_bytes,
    cudaStream_t    stream);
