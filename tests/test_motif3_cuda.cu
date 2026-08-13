/* H200 CUDA parity for Motif-3-only control and expanded-attention kernels. */
#include "ds4_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct array_value {
    uint32_t dtype;
    std::vector<uint64_t> dim;
    std::vector<uint8_t> bytes;
};

using fixture = std::unordered_map<std::string, array_value>;

static void read_exact(FILE *fp, void *dst, size_t n, const char *what) {
    if (fread(dst, 1, n, fp) != n) {
        fprintf(stderr, "short read: %s\n", what);
        std::exit(1);
    }
}

static fixture load_fixture(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); std::exit(1); }
    char magic[8];
    uint32_t version, count;
    read_exact(fp, magic, 8, "magic");
    read_exact(fp, &version, 4, "version");
    read_exact(fp, &count, 4, "count");
    if (memcmp(magic, "DS4FX1\0\0", 8) || version != 1 || count > 64) {
        fprintf(stderr, "bad fixture: %s\n", path);
        std::exit(1);
    }
    fixture result;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t name_len, dtype, ndim, reserved;
        uint64_t dim[4], nbytes;
        read_exact(fp, &name_len, 4, "name length");
        read_exact(fp, &dtype, 4, "dtype");
        read_exact(fp, &ndim, 4, "ndim");
        read_exact(fp, &reserved, 4, "reserved");
        read_exact(fp, dim, sizeof(dim), "dimensions");
        read_exact(fp, &nbytes, 8, "nbytes");
        if (!name_len || name_len > 255 || ndim > 4 || (dtype != 1 && dtype != 2)) {
            fprintf(stderr, "bad array descriptor\n"); std::exit(1);
        }
        std::string name(name_len, '\0');
        read_exact(fp, name.data(), name_len, "name");
        array_value value;
        value.dtype = dtype;
        value.dim.assign(dim, dim + ndim);
        value.bytes.resize((size_t)nbytes);
        read_exact(fp, value.bytes.data(), (size_t)nbytes, "data");
        result.emplace(std::move(name), std::move(value));
    }
    fclose(fp);
    return result;
}

static array_value &get(fixture &f, const char *name) {
    auto it = f.find(name);
    if (it == f.end()) { fprintf(stderr, "missing fixture array: %s\n", name); std::exit(1); }
    return it->second;
}

static const float *f32(fixture &f, const char *name) {
    auto &a = get(f, name);
    if (a.dtype != 1 || a.bytes.size() % sizeof(float)) std::exit(1);
    return reinterpret_cast<const float *>(a.bytes.data());
}

static const int32_t *i32(fixture &f, const char *name) {
    auto &a = get(f, name);
    if (a.dtype != 2 || a.bytes.size() % sizeof(int32_t)) std::exit(1);
    return reinterpret_cast<const int32_t *>(a.bytes.data());
}

struct gpu_tensor {
    ds4_gpu_tensor *p;
    explicit gpu_tensor(uint64_t bytes) : p(ds4_gpu_tensor_alloc(bytes)) {
        if (!p) { fprintf(stderr, "GPU allocation failed\n"); std::exit(1); }
    }
    ~gpu_tensor() { ds4_gpu_tensor_free(p); }
    gpu_tensor(const gpu_tensor &) = delete;
    gpu_tensor &operator=(const gpu_tensor &) = delete;
};

static void upload(gpu_tensor &dst, const array_value &src) {
    if (!ds4_gpu_tensor_write(dst.p, 0, src.bytes.data(), src.bytes.size())) {
        fprintf(stderr, "GPU write failed\n"); std::exit(1);
    }
}

static std::vector<float> download_f32(gpu_tensor &src, uint64_t count) {
    std::vector<float> out((size_t)count);
    if (!ds4_gpu_tensor_read(src.p, 0, out.data(), count * sizeof(float))) {
        fprintf(stderr, "GPU read failed\n"); std::exit(1);
    }
    return out;
}

static std::vector<int32_t> download_i32(gpu_tensor &src, uint64_t count) {
    std::vector<int32_t> out((size_t)count);
    if (!ds4_gpu_tensor_read(src.p, 0, out.data(), count * sizeof(int32_t))) {
        fprintf(stderr, "GPU read failed\n"); std::exit(1);
    }
    return out;
}

static std::vector<uint16_t> download_u16(gpu_tensor &src, uint64_t count) {
    std::vector<uint16_t> out((size_t)count);
    if (!ds4_gpu_tensor_read(src.p, 0, out.data(), count * sizeof(uint16_t))) {
        fprintf(stderr, "GPU read failed\n"); std::exit(1);
    }
    return out;
}

static void assert_close(const char *name, const float *got, const float *want,
                         uint64_t n, float atol, float rtol) {
    float worst = 0.0f;
    uint64_t wi = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float err = std::fabs(got[i] - want[i]);
        const float limit = atol + rtol * std::fabs(want[i]);
        const float ratio = limit > 0 ? err / limit : err;
        if (ratio > worst) { worst = ratio; wi = i; }
    }
    if (worst > 1.0f) {
        fprintf(stderr, "%s mismatch at %llu: got %.9g want %.9g (%.2fx)\n",
                name, (unsigned long long)wi, got[wi], want[wi], worst);
        std::exit(1);
    }
}

static std::string path(const char *dir, const char *name) {
    return std::string(dir) + "/" + name;
}

static uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(rounded >> 16u);
}

static float bf16_bits_to_f32(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static void test_bf16_projection() {
    /* Authentic Motif mHC proj_res shape: [16, 4 * hidden_size]. */
    constexpr uint64_t in_dim = 4u * 4096u;
    constexpr uint64_t out_dim = 16u;
    constexpr uint64_t rows = 3u;
    std::vector<uint16_t> weight((size_t)(in_dim * out_dim));
    std::vector<uint16_t> model_storage(weight.size() + 1u);
    std::vector<float> input((size_t)(rows * in_dim));
    std::vector<float> input_bf16(input.size());
    std::vector<float> expected((size_t)(rows * out_dim), 0.0f);

    for (uint64_t o = 0; o < out_dim; o++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            const int32_t raw = (int32_t)((i * 17u + o * 29u) % 257u) - 128;
            weight[(size_t)(o * in_dim + i)] =
                f32_to_bf16_bits((float)raw / 2048.0f);
        }
    }
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            const int32_t raw = (int32_t)((i * 11u + r * 37u) % 251u) - 125;
            const float value = (float)raw / 1024.0f;
            input[(size_t)(r * in_dim + i)] = value;
            input_bf16[(size_t)(r * in_dim + i)] =
                bf16_bits_to_f32(f32_to_bf16_bits(value));
        }
    }
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < in_dim; i++) {
                sum += bf16_bits_to_f32(weight[(size_t)(o * in_dim + i)]) *
                       input_bf16[(size_t)(r * in_dim + i)];
            }
            expected[(size_t)(r * out_dim + o)] = sum;
        }
    }

    /* Reinstall a different-sized model at the same host address.  Short-lived
     * auxiliary mappings can reuse a freed allocator address; a stale range
     * keyed only by that address must not shadow the current full device copy. */
    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    std::fill(model_storage.begin(), model_storage.end(),
              f32_to_bf16_bits(0.5f));
    if (!ds4_gpu_set_model_map(model_storage.data(),
                               model_storage.size() * sizeof(uint16_t))) {
        fprintf(stderr, "could not install stale BF16 projection test map\n");
        std::exit(1);
    }
    std::copy(weight.begin(), weight.end(), model_storage.begin());
    if (!ds4_gpu_set_model_map(model_storage.data(),
                               weight.size() * sizeof(uint16_t))) {
        fprintf(stderr, "could not install BF16 projection test map\n");
        std::exit(1);
    }
    unsetenv("DS4_CUDA_COPY_MODEL");
    gpu_tensor x(input.size() * sizeof(float));
    gpu_tensor out(expected.size() * sizeof(float));
    if (!ds4_gpu_tensor_write(x.p, 0, input.data(), input.size() * sizeof(float)) ||
        !ds4_gpu_matmul_bf16_tensor(out.p, model_storage.data(),
                                    weight.size() * sizeof(uint16_t), 0,
                                    in_dim, out_dim, x.p, rows)) {
        fprintf(stderr, "BF16 projection dispatch failed\n");
        std::exit(1);
    }
    auto got = download_f32(out, expected.size());
    assert_close("CUDA BF16 mHC projection", got.data(), expected.data(),
                 expected.size(), 2e-3f, 8e-4f);
}

static void test_router(const char *dir) {
    fixture f = load_fixture(path(dir, "router-layer2.ds4fx").c_str());
    gpu_tensor logits(get(f, "logits").bytes.size()); upload(logits, get(f, "logits"));
    gpu_tensor bias(get(f, "expert_bias").bytes.size()); upload(bias, get(f, "expert_bias"));
    gpu_tensor selected(8u * 8u * sizeof(int32_t));
    gpu_tensor weights(8u * 8u * sizeof(float));
    gpu_tensor probs(8u * 384u * sizeof(float));
    if (!ds4_gpu_motif3_router_select_batch_tensor(selected.p, weights.p, probs.p,
                                                    logits.p, bias.p, 8, 384, 8, 2.0f)) std::exit(1);
    auto ids = download_i32(selected, 64);
    auto route = download_f32(weights, 64);
    const int32_t *want_ids = i32(f, "selected_experts");
    for (uint32_t i = 0; i < 64; i++) if (ids[i] != want_ids[i]) {
        fprintf(stderr, "router id mismatch at %u: %d != %d\n", i, ids[i], want_ids[i]);
        std::exit(1);
    }
    assert_close("CUDA router", route.data(), f32(f, "route_weights"), 64, 3e-7f, 3e-6f);
}

static void test_polynorm(const char *dir) {
    fixture f = load_fixture(path(dir, "polynorm-layer2-expert173.ds4fx").c_str());
    gpu_tensor gate(get(f, "gate").bytes.size()); upload(gate, get(f, "gate"));
    gpu_tensor up(get(f, "up").bytes.size()); upload(up, get(f, "up"));
    gpu_tensor coeff(get(f, "raw_coeff").bytes.size()); upload(coeff, get(f, "raw_coeff"));
    gpu_tensor bias(get(f, "raw_bias").bytes.size()); upload(bias, get(f, "raw_bias"));
    gpu_tensor out(4u * 1280u * sizeof(float));
    if (!ds4_gpu_motif3_polynorm_mul_tensor(out.p, gate.p, up.p, coeff.p, bias.p,
                                            4, 1280, 1e6f, .5f, .5f, 1e-6f)) std::exit(1);
    auto got = download_f32(out, 4u * 1280u);
    assert_close("CUDA PolyNorm", got.data(), f32(f, "activated_fp32"),
                 4u * 1280u, 4e-5f, 4e-5f);
}

static void test_mhc(const char *dir) {
    fixture f = load_fixture(path(dir, "mhc-layer0-attn.ds4fx").c_str());
    const char *names[] = {"projected_pre", "projected_post", "projected_res",
                           "alpha_pre", "alpha_post", "alpha_res",
                           "bias_pre", "bias_post", "bias_res", "hidden"};
    std::vector<gpu_tensor *> tensors;
    for (const char *name : names) {
        auto *t = new gpu_tensor(get(f, name).bytes.size());
        upload(*t, get(f, name)); tensors.push_back(t);
    }
    gpu_tensor h_pre(16u * sizeof(float)), h_post(16u * sizeof(float)), h_res(64u * sizeof(float));
    if (!ds4_gpu_motif3_mhc_controls_tensor(
            h_pre.p, h_post.p, h_res.p,
            tensors[0]->p, tensors[1]->p, tensors[2]->p,
            tensors[3]->p, tensors[4]->p, tensors[5]->p,
            tensors[6]->p, tensors[7]->p, tensors[8]->p,
            4, 4, 20, 1.0f)) std::exit(1);
    auto pre = download_f32(h_pre, 16), post = download_f32(h_post, 16), res = download_f32(h_res, 64);
    assert_close("CUDA mHC pre", pre.data(), f32(f, "h_pre"), 16, 3e-7f, 3e-6f);
    assert_close("CUDA mHC post", post.data(), f32(f, "h_post"), 16, 3e-7f, 3e-6f);
    assert_close("CUDA mHC Sinkhorn", res.data(), f32(f, "h_res"), 64, 4e-6f, 4e-5f);
    gpu_tensor reduced(4u * 4096u * sizeof(float));
    gpu_tensor mixed(4u * 4u * 4096u * sizeof(float));
    if (!ds4_gpu_motif3_mhc_apply_pre_tensor(reduced.p, tensors[9]->p, h_pre.p, 4, 4, 4096) ||
        !ds4_gpu_motif3_mhc_apply_res_tensor(mixed.p, tensors[9]->p, h_res.p, 4, 4, 4096)) std::exit(1);
    auto reduced_h = download_f32(reduced, 4u * 4096u);
    auto mixed_h = download_f32(mixed, 4u * 4u * 4096u);
    assert_close("CUDA mHC pre apply", reduced_h.data(), f32(f, "reduced_input"), 4u * 4096u, 3e-6f, 3e-5f);
    assert_close("CUDA mHC residual apply", mixed_h.data(), f32(f, "residual_mixed"), 4u * 4u * 4096u, 3e-6f, 3e-5f);
    for (gpu_tensor *t : tensors) delete t;
}

static void test_gdla(const char *dir) {
    fixture f = load_fixture(path(dir, "gdla-expanded-layer0.ds4fx").c_str());
    gpu_tensor positions(get(f, "positions").bytes.size()); upload(positions, get(f, "positions"));
    gpu_tensor probes(get(f, "probe_positions").bytes.size()); upload(probes, get(f, "probe_positions"));
    gpu_tensor inv(get(f, "yarn_inv_freq").bytes.size()); upload(inv, get(f, "yarn_inv_freq"));
    gpu_tensor qpe(get(f, "q_pe_before").bytes.size()); upload(qpe, get(f, "q_pe_before"));
    gpu_tensor kpe(get(f, "k_pe_before").bytes.size()); upload(kpe, get(f, "k_pe_before"));
    gpu_tensor qrot(get(f, "q_pe_before").bytes.size());
    gpu_tensor krot(get(f, "k_pe_before").bytes.size());
    if (!ds4_gpu_motif3_rope_tensor(qrot.p, qpe.p, positions.p, inv.p, 8, 80, 64) ||
        !ds4_gpu_motif3_rope_tensor(krot.p, kpe.p, positions.p, inv.p, 8, 1, 64)) std::exit(1);
    auto qr = download_f32(qrot, 8u * 80u * 64u), kr = download_f32(krot, 8u * 64u);
    assert_close("CUDA GDLA q RoPE", qr.data(), f32(f, "q_pe_after_fp32"), qr.size(), 3e-5f, 3e-5f);
    assert_close("CUDA GDLA k RoPE", kr.data(), f32(f, "k_pe_after_fp32"), kr.size(), 3e-5f, 3e-5f);
    if (!ds4_gpu_motif3_rope_tensor(qrot.p, qpe.p, probes.p, inv.p, 8, 80, 64) ||
        !ds4_gpu_motif3_rope_tensor(krot.p, kpe.p, probes.p, inv.p, 8, 1, 64)) std::exit(1);
    qr = download_f32(qrot, 8u * 80u * 64u); kr = download_f32(krot, 8u * 64u);
    assert_close("CUDA GDLA 256K q RoPE", qr.data(), f32(f, "q_pe_probe_fp32"), qr.size(), 8e-4f, 6e-5f);
    assert_close("CUDA GDLA 256K k RoPE", kr.data(), f32(f, "k_pe_probe_fp32"), kr.size(), 8e-4f, 6e-5f);

    gpu_tensor q(get(f, "q_full").bytes.size()); upload(q, get(f, "q_full"));
    gpu_tensor k(get(f, "k_full").bytes.size()); upload(k, get(f, "k_full"));
    gpu_tensor v(get(f, "value").bytes.size()); upload(v, get(f, "value"));
    gpu_tensor attention(8u * 80u * 128u * sizeof(float));
    if (!ds4_gpu_motif3_expanded_attention_tensor(attention.p, q.p, k.p, v.p,
                                                  8, 80, 16, 192, 128,
                                                  f32(f, "attention_scale")[0], true)) std::exit(1);
    auto attn = download_f32(attention, 8u * 80u * 128u);
    assert_close("CUDA expanded GDLA", attn.data(), f32(f, "attention_fp32"), attn.size(), 3e-4f, 3e-5f);

    gpu_tensor lambda(get(f, "lambda").bytes.size()); upload(lambda, get(f, "lambda"));
    gpu_tensor gate(get(f, "gate_score").bytes.size()); upload(gate, get(f, "gate_score"));
    gpu_tensor diff(8u * 64u * 128u * sizeof(float));
    if (!ds4_gpu_motif3_differential_tensor(diff.p, attention.p, lambda.p, gate.p,
                                             8, 16, 5, 128)) std::exit(1);
    auto d = download_f32(diff, 8u * 64u * 128u);
    assert_close("CUDA differential GDLA", d.data(), f32(f, "diff_attention_fp32"), d.size(), 4e-4f, 4e-5f);
}

static void test_latent_gdla() {
    constexpr uint32_t rows = 6u;
    constexpr uint32_t q_heads = 80u;
    constexpr uint32_t kv_heads = 16u;
    constexpr uint32_t group = 5u;
    constexpr uint32_t latent_dim = 512u;
    constexpr uint32_t qk_nope = 128u;
    constexpr uint32_t rope_dim = 64u;
    constexpr uint32_t key_dim = qk_nope + rope_dim;
    constexpr uint32_t value_dim = 128u;
    constexpr uint32_t kv_raw_dim = latent_dim + rope_dim;
    constexpr uint64_t row_bytes = (latent_dim / 32u) * 34u;
    constexpr uint32_t weight_rows = kv_heads * (qk_nope + value_dim);
    constexpr float weight_scale = 1.0f / 64.0f;

    /* A deterministic authentic-shape Q8_0 kv_b matrix lets this test prove
     * the MLA identity used by production: q W_k C followed by attention and
     * C W_v is equivalent to (W_k^T q) C, latent accumulation, then W_v. */
    std::vector<uint8_t> model((size_t)weight_rows * row_bytes, 0u);
    for (uint32_t r = 0; r < weight_rows; r++) {
        uint8_t *row = model.data() + (uint64_t)r * row_bytes;
        for (uint32_t b = 0; b < latent_dim / 32u; b++) {
            uint8_t *block = row + (uint64_t)b * 34u;
            const uint16_t half_scale = 0x2400u; /* IEEE F16 2^-6. */
            std::memcpy(block, &half_scale, sizeof(half_scale));
            for (uint32_t j = 0; j < 32u; j++) {
                const uint32_t col = b * 32u + j;
                const int8_t q = (int8_t)((r * 17u + col * 13u + 3u) % 9u) - 4;
                std::memcpy(block + 2u + j, &q, sizeof(q));
            }
        }
    }
    auto weight = [&](uint32_t r, uint32_t c) -> float {
        const uint8_t *block = model.data() + (uint64_t)r * row_bytes +
                               (uint64_t)(c / 32u) * 34u;
        int8_t q;
        std::memcpy(&q, block + 2u + (c % 32u), sizeof(q));
        return weight_scale * (float)q;
    };

    std::vector<float> q_raw((size_t)rows * q_heads * key_dim);
    std::vector<float> kv_norm((size_t)rows * latent_dim);
    std::vector<float> kv_raw((size_t)rows * kv_raw_dim);
    std::vector<float> inv(rope_dim / 2u);
    std::vector<int32_t> positions(rows);
    for (size_t i = 0; i < q_raw.size(); i++)
        q_raw[i] = (float)((int32_t)((i * 29u + 11u) % 257u) - 128) / 256.0f;
    for (size_t i = 0; i < kv_norm.size(); i++)
        kv_norm[i] = (float)((int32_t)((i * 31u + 7u) % 251u) - 125) / 512.0f;
    for (uint32_t r = 0; r < rows; r++) {
        positions[r] = (int32_t)r;
        std::memcpy(kv_raw.data() + (uint64_t)r * kv_raw_dim,
                    kv_norm.data() + (uint64_t)r * latent_dim,
                    latent_dim * sizeof(float));
        for (uint32_t d = 0; d < rope_dim; d++) {
            const uint64_t i = (uint64_t)r * rope_dim + d;
            kv_raw[(uint64_t)r * kv_raw_dim + latent_dim + d] =
                (float)((int32_t)((i * 19u + 5u) % 193u) - 96) / 256.0f;
        }
    }
    for (uint32_t i = 0; i < rope_dim / 2u; i++)
        inv[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)rope_dim);

    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    if (!ds4_gpu_set_model_map(model.data(), model.size())) {
        fprintf(stderr, "could not install latent GDLA test map\n");
        std::exit(1);
    }
    unsetenv("DS4_CUDA_COPY_MODEL");

    gpu_tensor q_raw_gpu(q_raw.size() * sizeof(float));
    gpu_tensor q_full_gpu(q_raw.size() * sizeof(float));
    gpu_tensor kv_norm_gpu(kv_norm.size() * sizeof(float));
    gpu_tensor kv_raw_gpu(kv_raw.size() * sizeof(float));
    gpu_tensor positions_gpu(positions.size() * sizeof(int32_t));
    gpu_tensor inv_gpu(inv.size() * sizeof(float));
    gpu_tensor latent_cache((uint64_t)rows * latent_dim * sizeof(uint16_t));
    gpu_tensor k_pe_cache((uint64_t)rows * rope_dim * sizeof(uint16_t));
    gpu_tensor q_absorbed((uint64_t)rows * q_heads * latent_dim * sizeof(float));
    gpu_tensor latent_out((uint64_t)rows * q_heads * latent_dim * sizeof(float));
    gpu_tensor heads((uint64_t)rows * q_heads * value_dim * sizeof(float));
    if (!ds4_gpu_tensor_write(q_raw_gpu.p, 0, q_raw.data(), q_raw.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(kv_norm_gpu.p, 0, kv_norm.data(), kv_norm.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(kv_raw_gpu.p, 0, kv_raw.data(), kv_raw.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(positions_gpu.p, 0, positions.data(), positions.size() * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(inv_gpu.p, 0, inv.data(), inv.size() * sizeof(float))) {
        fprintf(stderr, "latent GDLA upload failed\n"); std::exit(1);
    }
    if (!ds4_gpu_motif3_prepare_q_tensor(q_full_gpu.p, q_raw_gpu.p,
                                          positions_gpu.p, inv_gpu.p,
                                          rows, q_heads, key_dim, rope_dim) ||
        !ds4_gpu_motif3_round_bf16_tensor(q_full_gpu.p, q_full_gpu.p,
                                           (uint64_t)rows * q_heads * key_dim) ||
        !ds4_gpu_motif3_store_latent_kv_bf16_tensor(
                latent_cache.p, k_pe_cache.p, kv_norm_gpu.p, kv_raw_gpu.p,
                positions_gpu.p, inv_gpu.p, rows, rows, kv_raw_dim,
                latent_dim, rope_dim, false) ||
        !ds4_gpu_motif3_qk_absorb_q8_0_tensor(
                q_absorbed.p, q_full_gpu.p, model.data(), model.size(), 0,
                rows, q_heads, kv_heads, group, latent_dim, qk_nope,
                key_dim, value_dim)) {
        fprintf(stderr, "latent GDLA preparation failed\n"); std::exit(1);
    }

    auto q_full = download_f32(q_full_gpu, (uint64_t)rows * q_heads * key_dim);
    auto latent_bits = download_u16(latent_cache, (uint64_t)rows * latent_dim);
    auto k_pe_bits = download_u16(k_pe_cache, (uint64_t)rows * rope_dim);
    auto q_abs = download_f32(q_absorbed, (uint64_t)rows * q_heads * latent_dim);
    std::vector<float> latent((size_t)rows * latent_dim);
    std::vector<float> k_pe((size_t)rows * rope_dim);
    for (size_t i = 0; i < latent.size(); i++) {
        latent[i] = bf16_bits_to_f32(latent_bits[i]);
        const float want = bf16_bits_to_f32(f32_to_bf16_bits(kv_norm[i]));
        if (latent[i] != want) {
            fprintf(stderr, "latent BF16 cache mismatch at %zu\n", i);
            std::exit(1);
        }
    }
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t d = 0; d < rope_dim; d++) {
            const uint32_t half = rope_dim / 2u;
            const uint32_t freq = d % half;
            const float angle = (float)positions[r] * inv[freq];
            const float first = kv_raw[(uint64_t)r * kv_raw_dim + latent_dim + freq];
            const float second = kv_raw[(uint64_t)r * kv_raw_dim + latent_dim + half + freq];
            const float rotated = d < half
                ? first * (float)std::cos((double)angle) - second * (float)std::sin((double)angle)
                : second * (float)std::cos((double)angle) + first * (float)std::sin((double)angle);
            const size_t i = (size_t)r * rope_dim + d;
            k_pe[i] = bf16_bits_to_f32(k_pe_bits[i]);
            if (k_pe_bits[i] != f32_to_bf16_bits(rotated)) {
                fprintf(stderr, "k_pe BF16 cache mismatch at %zu\n", i);
                std::exit(1);
            }
        }
    }

    std::vector<float> q_abs_want(q_abs.size(), 0.0f);
    for (uint32_t t = 0; t < rows; t++) for (uint32_t h = 0; h < q_heads; h++) {
        const uint32_t kh = h / group;
        const float *q = q_full.data() + ((uint64_t)t * q_heads + h) * key_dim;
        float *dst = q_abs_want.data() + ((uint64_t)t * q_heads + h) * latent_dim;
        for (uint32_t j = 0; j < latent_dim; j++)
            for (uint32_t d = 0; d < qk_nope; d++)
                dst[j] += q[d] * weight(kh * (qk_nope + value_dim) + d, j);
    }
    assert_close("CUDA absorbed Motif Q/K", q_abs.data(), q_abs_want.data(),
                 q_abs.size(), 2e-5f, 2e-5f);

    const float scale = 1.0f / std::sqrt((float)key_dim);
    if (!ds4_gpu_motif3_latent_attention_bf16_tensor(
                latent_out.p, q_full_gpu.p, q_absorbed.p,
                latent_cache.p, k_pe_cache.p, rows, 0, rows, 0,
                q_heads, latent_dim, qk_nope, rope_dim, scale) ||
        !ds4_gpu_motif3_value_project_q8_0_tensor(
                heads.p, latent_out.p, model.data(), model.size(), 0,
                rows, q_heads, kv_heads, group, latent_dim,
                qk_nope, value_dim)) {
        fprintf(stderr, "latent GDLA attention failed\n"); std::exit(1);
    }

    /* Independent expanded K/V reference, using the exact dequantized Q8_0
     * weights and BF16 persistent state consumed by the latent kernels. */
    std::vector<float> expanded_k((size_t)rows * kv_heads * qk_nope, 0.0f);
    std::vector<float> expanded_v((size_t)rows * kv_heads * value_dim, 0.0f);
    for (uint32_t t = 0; t < rows; t++) for (uint32_t kh = 0; kh < kv_heads; kh++) {
        const float *c = latent.data() + (uint64_t)t * latent_dim;
        for (uint32_t d = 0; d < qk_nope; d++)
            for (uint32_t j = 0; j < latent_dim; j++)
                expanded_k[((uint64_t)t * kv_heads + kh) * qk_nope + d] +=
                    weight(kh * (qk_nope + value_dim) + d, j) * c[j];
        for (uint32_t d = 0; d < value_dim; d++)
            for (uint32_t j = 0; j < latent_dim; j++)
                expanded_v[((uint64_t)t * kv_heads + kh) * value_dim + d] +=
                    weight(kh * (qk_nope + value_dim) + qk_nope + d, j) * c[j];
    }
    std::vector<float> heads_want((size_t)rows * q_heads * value_dim, 0.0f);
    std::vector<float> scores(rows), probs(rows);
    for (uint32_t t = 0; t < rows; t++) for (uint32_t h = 0; h < q_heads; h++) {
        const uint32_t kh = h / group;
        const float *q = q_full.data() + ((uint64_t)t * q_heads + h) * key_dim;
        float max_score = -INFINITY;
        for (uint32_t k = 0; k <= t; k++) {
            float dot = 0.0f;
            const float *ek = expanded_k.data() + ((uint64_t)k * kv_heads + kh) * qk_nope;
            for (uint32_t d = 0; d < qk_nope; d++) dot += q[d] * ek[d];
            for (uint32_t d = 0; d < rope_dim; d++)
                dot += q[qk_nope + d] * k_pe[(uint64_t)k * rope_dim + d];
            scores[k] = dot * scale;
            if (scores[k] > max_score) max_score = scores[k];
        }
        float denom = 0.0f;
        for (uint32_t k = 0; k <= t; k++) {
            probs[k] = std::exp(scores[k] - max_score);
            denom += probs[k];
        }
        float *dst = heads_want.data() + ((uint64_t)t * q_heads + h) * value_dim;
        for (uint32_t k = 0; k <= t; k++) {
            const float p = probs[k] / denom;
            const float *ev = expanded_v.data() + ((uint64_t)k * kv_heads + kh) * value_dim;
            for (uint32_t d = 0; d < value_dim; d++) dst[d] += p * ev[d];
        }
    }
    auto heads_got = download_f32(heads, heads_want.size());
    assert_close("CUDA latent GDLA vs expanded identity",
                 heads_got.data(), heads_want.data(), heads_want.size(),
                 2e-4f, 2e-4f);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE_DIR\n", argv[0]);
        return 2;
    }
    if (!ds4_gpu_init()) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    test_router(argv[1]);
    test_polynorm(argv[1]);
    test_mhc(argv[1]);
    test_gdla(argv[1]);
    test_latent_gdla();
    test_bf16_projection();
    ds4_gpu_cleanup();
    printf("Motif-3 H200 CUDA fixtures: BF16, router, PolyNorm, mHC, expanded/latent GDLA valid\n");
    return 0;
}
