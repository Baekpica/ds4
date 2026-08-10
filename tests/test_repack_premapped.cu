/* Hermetic regression for split-GGUF aligned-artifact input.
 *
 * The engine maps every shard into one virtual address range and hands the
 * repacker a merged tensor catalog.  Builders must therefore read source
 * bytes from that mapping rather than reopening shard 00001.  Exercise the
 * contract with one Q8_0 tensor at a non-zero merged offset and verify the
 * resulting aligned-SoA bytes exactly.
 */
#include "../cuda/mmq/ds4_repack.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int fail(const char *what) {
    std::fprintf(stderr, "test_repack_premapped: %s\n", what);
    return 1;
}

int main() {
    constexpr uint64_t kOffset = 4096;
    constexpr uint64_t kIn = 1024;
    constexpr uint64_t kOut = 2048;
    constexpr uint64_t kBlocks = (kIn / 32u) * kOut;
    constexpr uint64_t kRawBytes = kBlocks * 34u;
    constexpr uint64_t kScaleBytes = kBlocks * 2u;
    constexpr uint64_t kArtifactBytes = kScaleBytes + kBlocks * 32u;

    std::vector<uint8_t> source(kOffset + kRawBytes + 64u, 0xa5u);
    for (uint64_t block = 0; block < kBlocks; block++) {
        uint8_t *raw = source.data() + kOffset + block * 34u;
        raw[0] = 0x00u;
        raw[1] = 0x3cu;  // IEEE fp16 1.0
        for (uint32_t i = 0; i < 32u; i++) {
            raw[2u + i] = static_cast<uint8_t>((block + i * 17u) & 0xffu);
        }
    }

    ds4_repack_tensor tensor;
    tensor.name = "blk.0.attn_q.weight";
    tensor.type = 8u;
    tensor.ndim = 2u;
    tensor.dims[0] = kIn;
    tensor.dims[1] = kOut;
    tensor.elements = kIn * kOut;
    tensor.off = kOffset;
    tensor.bytes = kRawBytes;
    std::vector<ds4_repack_tensor> records{tensor};

    ds4_repack_build_args args;
    args.log_prefix = "test_repack_premapped";
    args.model_id = "fixture";
    args.records = &records;
    args.copy_chunk_bytes = kRawBytes;
    args.source_data = source.data();
    args.source_size = source.size();

    std::vector<ds4_repack_artifact> artifacts;
    uint64_t built = 0;
    if (!ds4_repack_build_q8_aligned(args, artifacts, &built)) {
        return fail("premapped Q8 build failed");
    }
    if (artifacts.size() != 1u || built != kArtifactBytes ||
        artifacts[0].bytes != kArtifactBytes) {
        return fail("unexpected artifact geometry");
    }

    std::vector<uint8_t> actual(kArtifactBytes);
    cudaError_t err = cudaMemcpy(actual.data(), artifacts[0].dev,
                                 actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return fail(cudaGetErrorString(err));

    for (uint64_t block = 0; block < kBlocks; block++) {
        if (actual[block * 2u] != 0x00u ||
            actual[block * 2u + 1u] != 0x3cu) {
            return fail("scale plane mismatch");
        }
        const uint8_t *want = source.data() + kOffset + block * 34u + 2u;
        const uint8_t *got = actual.data() + kScaleBytes + block * 32u;
        if (std::memcmp(got, want, 32u) != 0) {
            return fail("quant plane mismatch");
        }
    }

    (void)cudaFree(artifacts[0].dev);
    std::puts("premapped repack: ok");
    return 0;
}
