/* Compare ds4's plain-C Motif-3 control paths against fixtures generated from
 * the pinned official checkpoint on H200. */
#include "../ds4.c"

typedef struct {
    char *name;
    uint32_t dtype;
    uint32_t ndim;
    uint64_t dim[4];
    uint64_t nbytes;
    void *data;
} fixture_array;

typedef struct {
    uint32_t count;
    fixture_array *array;
} fixture_file;

static void read_exact(FILE *fp, void *dst, size_t n, const char *what) {
    if (fread(dst, 1, n, fp) != n) {
        fprintf(stderr, "short read while loading %s\n", what);
        exit(1);
    }
}

static fixture_file fixture_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        exit(1);
    }
    char magic[8];
    uint32_t version = 0;
    fixture_file out = {0};
    read_exact(fp, magic, sizeof(magic), "fixture magic");
    read_exact(fp, &version, sizeof(version), "fixture version");
    read_exact(fp, &out.count, sizeof(out.count), "fixture array count");
    if (memcmp(magic, "DS4FX1\0\0", 8) != 0 || version != 1 || out.count > 64) {
        fprintf(stderr, "unsupported fixture container: %s\n", path);
        exit(1);
    }
    out.array = calloc(out.count, sizeof(out.array[0]));
    if (!out.array) abort();
    for (uint32_t i = 0; i < out.count; i++) {
        uint32_t name_len = 0;
        uint32_t reserved = 0;
        read_exact(fp, &name_len, sizeof(name_len), "array name length");
        read_exact(fp, &out.array[i].dtype, sizeof(uint32_t), "array dtype");
        read_exact(fp, &out.array[i].ndim, sizeof(uint32_t), "array ndim");
        read_exact(fp, &reserved, sizeof(reserved), "array reserved field");
        read_exact(fp, out.array[i].dim, sizeof(out.array[i].dim), "array dimensions");
        read_exact(fp, &out.array[i].nbytes, sizeof(uint64_t), "array byte count");
        if (name_len == 0 || name_len > 255 || out.array[i].ndim > 4 ||
            (out.array[i].dtype != 1 && out.array[i].dtype != 2)) {
            fprintf(stderr, "invalid array descriptor in %s\n", path);
            exit(1);
        }
        out.array[i].name = calloc((size_t)name_len + 1u, 1);
        out.array[i].data = malloc((size_t)out.array[i].nbytes);
        if (!out.array[i].name || !out.array[i].data) abort();
        read_exact(fp, out.array[i].name, name_len, "array name");
        read_exact(fp, out.array[i].data, (size_t)out.array[i].nbytes, "array data");
    }
    fclose(fp);
    return out;
}

static void fixture_free(fixture_file *file) {
    for (uint32_t i = 0; i < file->count; i++) {
        free(file->array[i].data);
        free(file->array[i].name);
    }
    free(file->array);
    memset(file, 0, sizeof(*file));
}

static fixture_array *fixture_find(fixture_file *file, const char *name) {
    for (uint32_t i = 0; i < file->count; i++) {
        if (strcmp(file->array[i].name, name) == 0) return &file->array[i];
    }
    fprintf(stderr, "fixture array not found: %s\n", name);
    exit(1);
}

static float *fixture_f32(fixture_file *file, const char *name) {
    fixture_array *a = fixture_find(file, name);
    if (a->dtype != 1 || a->nbytes % sizeof(float) != 0) {
        fprintf(stderr, "fixture array is not f32: %s\n", name);
        exit(1);
    }
    return a->data;
}

static int32_t *fixture_i32(fixture_file *file, const char *name) {
    fixture_array *a = fixture_find(file, name);
    if (a->dtype != 2 || a->nbytes % sizeof(int32_t) != 0) {
        fprintf(stderr, "fixture array is not i32: %s\n", name);
        exit(1);
    }
    return a->data;
}

static void assert_close(const char *name, const float *got, const float *want,
                         uint64_t n, float atol, float rtol) {
    float worst = 0.0f;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float err = fabsf(got[i] - want[i]);
        const float limit = atol + rtol * fabsf(want[i]);
        const float ratio = limit > 0.0f ? err / limit : err;
        if (ratio > worst) {
            worst = ratio;
            worst_i = i;
        }
    }
    if (worst > 1.0f) {
        fprintf(stderr, "%s mismatch at %" PRIu64 ": got %.9g want %.9g (%.2fx tolerance)\n",
                name, worst_i, got[worst_i], want[worst_i], worst);
        exit(1);
    }
}

static void path_join(char *out, size_t cap, const char *dir, const char *name) {
    if (snprintf(out, cap, "%s/%s", dir, name) >= (int)cap) {
        fprintf(stderr, "fixture path is too long\n");
        exit(1);
    }
}

static void test_router(const char *dir) {
    char path[1024];
    path_join(path, sizeof(path), dir, "router-layer2.ds4fx");
    fixture_file file = fixture_load(path);
    int32_t selected[8 * 8];
    float weights[8 * 8];
    motif3_router_from_logits_reference(selected, weights,
        fixture_f32(&file, "logits"), fixture_f32(&file, "expert_bias"),
        8, 384, 8, 2.0f);
    int32_t *want_selected = fixture_i32(&file, "selected_experts");
    for (uint32_t i = 0; i < 64; i++) {
        if (selected[i] != want_selected[i]) {
            fprintf(stderr, "router selected_experts mismatch at %u: %d != %d\n",
                    i, selected[i], want_selected[i]);
            exit(1);
        }
    }
    assert_close("router weights", weights, fixture_f32(&file, "route_weights"),
                 64, 3.0e-7f, 3.0e-6f);
    fixture_free(&file);
}

static void test_polynorm(const char *dir) {
    char path[1024];
    path_join(path, sizeof(path), dir, "polynorm-layer2-expert173.ds4fx");
    fixture_file file = fixture_load(path);
    float *out = calloc(4u * 1280u, sizeof(float));
    float *coeff = fixture_f32(&file, "raw_coeff");
    motif3_polynorm_mul_reference(out,
        fixture_f32(&file, "gate"), fixture_f32(&file, "up"), coeff,
        fixture_f32(&file, "raw_bias")[0], 4, 1280, 1000000.0f,
        0.5f, 0.5f, 1.0e-6f);
    assert_close("expert PolyNorm", out, fixture_f32(&file, "activated_fp32"),
                 4u * 1280u, 3.0e-5f, 3.0e-5f);
    free(out);
    fixture_free(&file);
}

static void test_mhc(const char *dir) {
    char path[1024];
    path_join(path, sizeof(path), dir, "mhc-layer0-attn.ds4fx");
    fixture_file file = fixture_load(path);
    float h_pre[4 * 4];
    float h_post[4 * 4];
    float h_res[4 * 4 * 4];
    motif3_mhc_controls_reference(h_pre, h_post, h_res,
        fixture_f32(&file, "projected_pre"),
        fixture_f32(&file, "projected_post"),
        fixture_f32(&file, "projected_res"),
        fixture_f32(&file, "alpha_pre")[0],
        fixture_f32(&file, "alpha_post")[0],
        fixture_f32(&file, "alpha_res")[0],
        fixture_f32(&file, "bias_pre"),
        fixture_f32(&file, "bias_post"),
        fixture_f32(&file, "bias_res"), 4, 4, 20, 1.0f);
    assert_close("mHC h_pre", h_pre, fixture_f32(&file, "h_pre"), 16,
                 3.0e-7f, 3.0e-6f);
    assert_close("mHC h_post", h_post, fixture_f32(&file, "h_post"), 16,
                 3.0e-7f, 3.0e-6f);
    assert_close("mHC Sinkhorn", h_res, fixture_f32(&file, "h_res"), 64,
                 3.0e-6f, 3.0e-5f);

    float *reduced = calloc(4u * 4096u, sizeof(float));
    float *mixed = calloc(4u * 4u * 4096u, sizeof(float));
    motif3_mhc_apply_pre_reference(reduced, fixture_f32(&file, "hidden"),
                                   h_pre, 4, 4, 4096);
    motif3_mhc_apply_res_reference(mixed, fixture_f32(&file, "hidden"),
                                   h_res, 4, 4, 4096);
    assert_close("mHC pre mix", reduced, fixture_f32(&file, "reduced_input"),
                 4u * 4096u, 2.0e-6f, 2.0e-5f);
    assert_close("mHC residual mix", mixed, fixture_f32(&file, "residual_mixed"),
                 4u * 4u * 4096u, 2.0e-6f, 2.0e-5f);
    free(mixed);
    free(reduced);
    fixture_free(&file);
}

static void test_gdla(const char *dir) {
    char path[1024];
    path_join(path, sizeof(path), dir, "gdla-expanded-layer0.ds4fx");
    fixture_file file = fixture_load(path);
    const uint64_t q_rope_n = 8u * 80u * 64u;
    const uint64_t k_rope_n = 8u * 64u;
    float *q_rope = calloc((size_t)q_rope_n, sizeof(float));
    float *k_rope = calloc((size_t)k_rope_n, sizeof(float));
    motif3_neox_rope_reference(q_rope, fixture_f32(&file, "q_pe_before"),
        fixture_i32(&file, "positions"), fixture_f32(&file, "yarn_inv_freq"),
        8, 80, 64);
    motif3_neox_rope_reference(k_rope, fixture_f32(&file, "k_pe_before"),
        fixture_i32(&file, "positions"), fixture_f32(&file, "yarn_inv_freq"),
        8, 1, 64);
    assert_close("GDLA q YaRN", q_rope, fixture_f32(&file, "q_pe_after_fp32"),
                 q_rope_n, 2.0e-5f, 2.0e-5f);
    assert_close("GDLA k YaRN", k_rope, fixture_f32(&file, "k_pe_after_fp32"),
                 k_rope_n, 2.0e-5f, 2.0e-5f);

    motif3_neox_rope_reference(q_rope, fixture_f32(&file, "q_pe_before"),
        fixture_i32(&file, "probe_positions"), fixture_f32(&file, "yarn_inv_freq"),
        8, 80, 64);
    motif3_neox_rope_reference(k_rope, fixture_f32(&file, "k_pe_before"),
        fixture_i32(&file, "probe_positions"), fixture_f32(&file, "yarn_inv_freq"),
        8, 1, 64);
    assert_close("GDLA q 256K YaRN probe", q_rope,
                 fixture_f32(&file, "q_pe_probe_fp32"), q_rope_n,
                 3.0e-4f, 3.0e-4f);
    assert_close("GDLA k 256K YaRN probe", k_rope,
                 fixture_f32(&file, "k_pe_probe_fp32"), k_rope_n,
                 3.0e-4f, 3.0e-4f);
    free(k_rope);
    free(q_rope);

    const uint64_t attention_n = 8u * 80u * 128u;
    float *attention = calloc((size_t)attention_n, sizeof(float));
    motif3_expanded_attention_reference(attention,
        fixture_f32(&file, "q_full"), fixture_f32(&file, "k_full"),
        fixture_f32(&file, "value"), 8, 80, 16, 192, 128,
        fixture_f32(&file, "attention_scale")[0]);
    assert_close("GDLA expanded causal attention", attention,
                 fixture_f32(&file, "attention_fp32"), attention_n,
                 1.5e-4f, 2.0e-4f);

    const uint64_t diff_n = 8u * 64u * 128u;
    float *diff = calloc((size_t)diff_n, sizeof(float));
    motif3_differential_attention_reference(diff, attention,
        fixture_f32(&file, "lambda"), fixture_f32(&file, "gate_score"),
        8, 16, 4, 128);
    assert_close("GDLA differential lambda/output gate", diff,
                 fixture_f32(&file, "diff_attention_fp32"), diff_n,
                 1.5e-4f, 2.0e-4f);
    free(diff);
    free(attention);
    fixture_free(&file);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <official-fixture-dir>\n", argv[0]);
        return 2;
    }
    test_router(argv[1]);
    test_polynorm(argv[1]);
    test_mhc(argv[1]);
    test_gdla(argv[1]);
    puts("Motif-3 official fixtures: router, expert PolyNorm, mHC, and expanded GDLA valid");
    return 0;
}
