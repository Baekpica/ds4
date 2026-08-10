/* Solar Open 2 GB10 residency admission accounting.
 *
 * The CUDA-owned model copy and the context graph compete for the same
 * unified-memory pool.  Before choosing the faster resident copy, admission
 * must price Solar's 12 full-attention KV layers plus KDA state and scratch;
 * treating Solar like an unknown family (zero KV bytes) can turn a clean
 * direct-map fallback into a late 1M-context OOM.
 */

#include "../ds4.h"

#include <stdint.h>
#include <stdio.h>

uint64_t ds4_test_solar_weight_residency_context_bytes(
        uint32_t ctx_size,
        uint32_t prefill_chunk,
        const char *kv_format);
int ds4_test_weight_residency_should_copy(
        uint64_t model_bytes,
        uint64_t context_bytes,
        uint64_t floor_bytes,
        uint64_t mem_available_bytes,
        uint64_t cuda_free_bytes,
        uint64_t physical_pool_bytes);

static int failures;
static int checks;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        checks++;                                                             \
        if (!(condition)) {                                                   \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);   \
            failures++;                                                       \
        }                                                                     \
    } while (0)

static uint64_t raw_kv_bytes(uint64_t row_bytes, uint32_t ctx_size) {
    return row_bytes * (uint64_t)ctx_size * 12u;
}

int main(void) {
    const uint32_t ctx = 1048576u;
    const uint64_t bf16 = ds4_test_solar_weight_residency_context_bytes(
            ctx, 0u, "bf16");
    const uint64_t fp8 = ds4_test_solar_weight_residency_context_bytes(
            ctx, 0u, "fp8");
    const uint64_t hybrid = ds4_test_solar_weight_residency_context_bytes(
            ctx, 0u, "hybrid");
    const uint64_t fp4 = ds4_test_solar_weight_residency_context_bytes(
            ctx, 0u, "fp4");

    const uint64_t bf16_kv = raw_kv_bytes(4096u, ctx);
    const uint64_t fp8_kv = raw_kv_bytes(2080u, ctx);
    const uint64_t hybrid_kv = raw_kv_bytes(1568u, ctx);
    const uint64_t fp4_kv = raw_kv_bytes(1056u, ctx);

    CHECK(fp4 > fp4_kv,
          "residency admission includes Solar KDA state and scratch");
    CHECK(bf16 > fp8 && fp8 > hybrid && hybrid > fp4,
          "Solar residency bytes follow BF16 > FP8 > hybrid > FP4");
    CHECK(bf16 - fp8 == bf16_kv - fp8_kv,
          "BF16/FP8 delta equals the exact 12-layer KV delta");
    CHECK(fp8 - hybrid == fp8_kv - hybrid_kv,
          "FP8/hybrid delta equals the exact 12-layer KV delta");
    CHECK(hybrid - fp4 == hybrid_kv - fp4_kv,
          "hybrid/FP4 delta equals the exact 12-layer KV delta");

    const uint64_t fp4_p1024 =
        ds4_test_solar_weight_residency_context_bytes(ctx, 1024u, "fp4");
    const uint64_t fp4_p4096 =
        ds4_test_solar_weight_residency_context_bytes(ctx, 4096u, "fp4");
    CHECK(fp4_p1024 < fp4 && fp4 < fp4_p4096,
          "explicit prefill chunk changes Solar scratch admission bytes");

    const uint64_t gib = 1ull << 30;
    CHECK(ds4_test_weight_residency_should_copy(
                  89u * gib, 14u * gib, 8u * gib,
                  29u * gib, 1u * gib, 121u * gib),
          "physical UMA pool admits resident copy after a GB10 restart");
    CHECK(ds4_test_weight_residency_should_copy(
                  89u * gib, 14u * gib, 8u * gib,
                  118u * gib, 0u, 0u),
          "Linux MemAvailable remains a valid first-boot fallback");
    CHECK(!ds4_test_weight_residency_should_copy(
                  89u * gib, 14u * gib, 8u * gib,
                  29u * gib, 1u * gib, 110u * gib),
          "resident copy is refused when neither UMA view has headroom");
    CHECK(!ds4_test_weight_residency_should_copy(
                  UINT64_MAX, 1u, 1u,
                  UINT64_MAX, UINT64_MAX, UINT64_MAX),
          "resident-copy requirement saturates instead of wrapping");

    fprintf(stderr,
            "Solar 1M residency: BF16 %.3f, FP8 %.3f, hybrid %.3f, "
            "FP4 %.3f GiB\n",
            (double)bf16 / 1073741824.0,
            (double)fp8 / 1073741824.0,
            (double)hybrid / 1073741824.0,
            (double)fp4 / 1073741824.0);
    fprintf(stderr,
            "test_solar_memory: %d/%d checks passed (%d failed)\n",
            checks - failures, checks, failures);
    return failures ? 1 : 0;
}
