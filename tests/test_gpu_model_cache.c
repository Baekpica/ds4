/* Unit test for the per-device selective model cache
 * (mgpu-selective-model-cache).
 *
 * Exercises:
 *   - ds4_gpu_device_cache_tensors with disjoint ranges on device 0
 *   - ds4_gpu_lookup_cache at range bases and at interior offsets
 *     (proves the subrange pointer offset arithmetic is right)
 *   - device-id resolution
 *   - on multi-GPU boxes: caching on device 1 and active-device
 *     preference in lookup */

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);   \
            return 1;                                                   \
        }                                                               \
    } while (0)

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_gpu_model_cache: %d CUDA devices visible\n",
            dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices\n");
        return 0;
    }

    CHECK(ds4_gpu_init(), "ds4_gpu_init");

    /* Build a synthetic 1-MiB file-mapping-like model. Temporarily protect
     * one page so the initial whole-map cudaHostRegister is rejected, matching
     * the aligned-artifact startup path which deliberately leaves the full
     * GGUF mmap unpinned. Individual promotion ranges remain registerable. */
    const size_t total = 1024 * 1024;
    char model_path[] = "/tmp/ds4-model-map-XXXXXX";
    int model_fd = mkstemp(model_path);
    CHECK(model_fd >= 0, "mkstemp synthetic model");
    (void)unlink(model_path);
    CHECK(ftruncate(model_fd, (off_t)total) == 0,
          "truncate synthetic model");
    void *host = mmap(NULL, total, PROT_READ | PROT_WRITE,
                      MAP_SHARED, model_fd, 0);
    CHECK(host != MAP_FAILED, "mmap synthetic model");
    unsigned char *bytes = (unsigned char *)host;
    for (size_t i = 0; i < total; i++) bytes[i] = (unsigned char)(i & 0xff);
    const long page_size = sysconf(_SC_PAGESIZE);
    CHECK(page_size > 0, "page size");
    CHECK(mprotect(host, (size_t)page_size, PROT_NONE) == 0,
          "protect page during whole-map registration");
    CHECK(ds4_gpu_set_model_map(host, total), "set_model_map");
    CHECK(mprotect(host, (size_t)page_size, PROT_READ | PROT_WRITE) == 0,
          "restore page after whole-map registration");

    /* A split GGUF is one virtual mapping backed by several files. The
     * legacy single-FD fast path only describes shard 0, so an offset beyond
     * that FD must fall back before allocating an arena or staging pool. */
    char split_fd_path[] = "/tmp/ds4-split-fd-XXXXXX";
    int split_fd = mkstemp(split_fd_path);
    CHECK(split_fd >= 0, "mkstemp split fd");
    (void)unlink(split_fd_path);
    CHECK(ftruncate(split_fd, (off_t)(total / 2)) == 0, "truncate split fd");
    CHECK(ds4_gpu_set_model_fd_for_map(split_fd, host), "set short shard fd");
    (void)unsetenv("DS4_CUDA_NO_FD_CACHE");
    CHECK(setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1) == 0,
          "set minimum arena chunk");

    size_t free_before = 0, free_after = 0, total_mem = 0;
    CHECK(cudaMemGetInfo(&free_before, &total_mem) == cudaSuccess,
          "mem info before split fallback");
    const uint64_t cached_before = ds4_gpu_model_range_cached_bytes();
    CHECK(ds4_gpu_cache_model_range(host, total, 3 * total / 4, 64 * 1024,
                                    "split_fd_fallback"),
          "range beyond shard fd is promoted from the mapping");
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "sync split fallback");
    const uint64_t cached_after = ds4_gpu_model_range_cached_bytes();
    CHECK(cached_after - cached_before == 64 * 1024,
          "explicit split-shard promotion owns device-cache payload");
    CHECK(cudaMemGetInfo(&free_after, &total_mem) == cudaSuccess,
          "mem info after split fallback");
    CHECK(free_before >= free_after, "split fallback memory accounting");
    CHECK(free_before - free_after < 128ull * 1024ull * 1024ull,
          "split fallback does not leak a model arena");
    CHECK(ds4_gpu_set_model_fd(-1), "clear short shard fd");
    (void)close(split_fd);
    (void)unsetenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");

    /* Three disjoint ranges on device 0. */
    ds4_tensor_range ranges[3];
    ranges[0].source_offset = 0;          ranges[0].bytes = 256 * 1024; ranges[0].target_device = 0;
    ranges[1].source_offset = 384 * 1024; ranges[1].bytes = 128 * 1024; ranges[1].target_device = 0;
    ranges[2].source_offset = 768 * 1024; ranges[2].bytes = 256 * 1024; ranges[2].target_device = 0;

    CHECK(ds4_gpu_device_cache_tensors(0, ranges, 3) == 0,
          "device_cache_tensors dev 0 (3 ranges)");

    /* Base lookups + interior offset arithmetic. */
    int dev = -1; void *base0 = NULL, *interior0 = NULL;
    CHECK(ds4_gpu_lookup_cache(0, 1024, &dev, &base0) == 1, "lookup range 0 base");
    CHECK(dev == 0, "range 0 device");
    CHECK(base0 != NULL, "range 0 ptr");
    /* An interior offset must return base0 + delta. */
    CHECK(ds4_gpu_lookup_cache(100, 1024, &dev, &interior0) == 1, "lookup range 0 interior");
    CHECK(dev == 0, "range 0 interior device");
    CHECK(interior0 == (char *)base0 + 100, "interior offset arithmetic");

    void *base1 = NULL;
    CHECK(ds4_gpu_lookup_cache(384 * 1024, 1024, &dev, &base1) == 1, "lookup range 1 base");
    CHECK(dev == 0, "range 1 device");
    /* Interior of range 1 at +200 bytes should be base1 + 200. */
    void *interior1 = NULL;
    CHECK(ds4_gpu_lookup_cache(384 * 1024 + 200, 1024, &dev, &interior1) == 1, "lookup range 1 interior");
    CHECK(interior1 == (char *)base1 + 200, "range 1 interior offset");

    void *base2 = NULL;
    CHECK(ds4_gpu_lookup_cache(900 * 1024, 1024, &dev, &base2) == 1, "lookup range 2");
    CHECK(dev == 0 && base2 != NULL, "range 2 device+ptr");

    /* Convenience wrapper. */
    CHECK(ds4_gpu_lookup_cache_device(0, 1024) == 0, "lookup_device range 0");

    /* Lookup must be overflow-safe: a query with bytes=UINT64_MAX must
     * not wrap around into a false hit. */
    int dev_overflow = -1; void *ptr_overflow = NULL;
    int hit = ds4_gpu_lookup_cache(100, UINT64_MAX, &dev_overflow, &ptr_overflow);
    /* Either miss (preferred), or hit but the path must NOT have wrapped.
     * Accept miss only — a wrap-induced hit would be a bug. */
    CHECK(hit == 0, "lookup with bytes=UINT64_MAX does not wrap into a false hit");

    /* Bounds-check: ranges that overflow the model must be rejected
     * before any allocation. */
    ds4_tensor_range bad_overflow = { 0, total + 1, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_overflow, 1) != 0,
          "overflow range rejected");
    ds4_tensor_range bad_offset = { total + 1, 16, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_offset, 1) != 0,
          "out-of-range offset rejected");
    ds4_tensor_range bad_wrap = { total - 4, UINT64_MAX, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_wrap, 1) != 0,
          "wrap-around range rejected");

    /* Gap not covered by selective ranges. The legacy chunked path may
     * happen to cover it (it caches the whole model span); accept either
     * outcome, but if it returns 1 the device must be 0. */
    int dev_gap = -1; void *ptr_gap = NULL;
    int gap_hit = ds4_gpu_lookup_cache(300 * 1024, 1024, &dev_gap, &ptr_gap);
    if (gap_hit) {
        CHECK(dev_gap == 0, "gap fallback device");
    }

    if (dev_count >= 2) {
        /* Cache a different range on device 1. */
        ds4_tensor_range r2;
        r2.source_offset = 256 * 1024;
        r2.bytes         = 128 * 1024;
        r2.target_device = 1;
        CHECK(ds4_gpu_device_cache_tensors(1, &r2, 1) == 0, "cache dev 1");

        /* With cudaGetDevice() == 1, the lookup should resolve to dev 1
         * for this range (the only selective entry covering it). */
        (void)cudaSetDevice(1);
        int dd = -1; void *pp = NULL;
        CHECK(ds4_gpu_lookup_cache(256 * 1024 + 10, 1024, &dd, &pp) == 1,
              "lookup dev 1");
        CHECK(dd == 1, "lookup resolves to dev 1");
        CHECK(pp != NULL, "lookup ptr non-null");
        (void)cudaSetDevice(0);
    }

    ds4_gpu_cleanup();
    (void)munmap(host, total);
    (void)close(model_fd);
    fprintf(stderr, "test_gpu_model_cache PASS (devs=%d)\n", dev_count);
    return 0;
}
