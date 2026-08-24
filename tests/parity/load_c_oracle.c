/* Host tensor-dir consume oracle from native/bridge/ds4_host_load.h.
 * Do not include ds4.c. */

#include "native/bridge/ds4_host_load.h"

#include <stdio.h>
#include <string.h>

static void fill_native(ds4_host_native_tensor *n, const char *name)
{
    memset(n, 0, sizeof(*n));
    n->name = name;
    n->name_len = (uint32_t)strlen(name);
    n->ndim = 2;
    n->dim[0] = 4;
    n->dim[1] = 8;
    n->type = 0;
    n->rel_offset = 0;
    n->abs_offset = 32;
    n->bytes = 128;
}

static void fill_host(ds4_host_tensor *h, const char *name)
{
    memset(h, 0, sizeof(*h));
    h->name = name;
    h->ndim = 2;
    h->dim[0] = 4;
    h->dim[1] = 8;
    h->type = 0;
    h->rel_offset = 0;
    h->abs_offset = 32;
    h->bytes = 128;
}

static const char *token(int rc, const char *err)
{
    return rc == 0 ? "ok" : err;
}

static void consume_tapes(void)
{
    ds4_host_native_tensor native[2];
    ds4_host_tensor host[2];
    ds4_host_tensor_dir dir;
    char err[64];
    int rc;

    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    fill_host(&host[0], "tok_embd");
    fill_host(&host[1], "output");

    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, NULL, err, sizeof err);
    printf("absent %s\n", token(rc, err));

    dir.n = 2;
    dir.v = host;
    dir.data_pos = 32;
    dir.alignment = 32;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("ok %s\n", token(rc, err));

    dir.n = 2;
    dir.v = NULL;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("tensors-null %s\n", token(rc, err));

    dir.n = 1;
    dir.v = host;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("count %s\n", token(rc, err));

    fill_host(&host[1], "other");
    dir.n = 2;
    dir.v = host;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("name %s\n", token(rc, err));
    fill_host(&host[1], "output");

    fill_host(&host[0], "");
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("name-empty %s\n", token(rc, err));
    fill_host(&host[0], "tok_embd");

    host[0].type = 1;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("type %s\n", token(rc, err));
    host[0].type = 0;

    host[0].dim[0] = 2;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("dim %s\n", token(rc, err));
    host[0].dim[0] = 4;

    host[0].rel_offset = 64;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("offset %s\n", token(rc, err));
    host[0].rel_offset = 0;

    host[0].bytes = 64;
    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("bytes %s\n", token(rc, err));
    host[0].bytes = 128;

    fill_native(&native[0], "tok_embd");
    fill_native(&native[1], "output");
    dir.data_pos = 64;
    rc = ds4_host_tensor_dir_consume(native, 2, 32, 32, &dir, err, sizeof err);
    printf("data %s\n", token(rc, err));
}

static void apply_tapes(void)
{
    ds4_host_native_tensor out[2];
    ds4_host_tensor host[2];
    ds4_host_tensor_dir dir;
    char err[64];
    int rc;

    fill_host(&host[0], "tok_embd");
    fill_host(&host[1], "output");
    memset(out, 0, sizeof out);

    rc = ds4_host_tensor_dir_apply(out, 2, NULL, err, sizeof err);
    printf("dir-null %s\n", token(rc, err));

    dir.n = 2;
    dir.v = NULL;
    dir.data_pos = 32;
    dir.alignment = 32;
    rc = ds4_host_tensor_dir_apply(out, 2, &dir, err, sizeof err);
    printf("tensors-null %s\n", token(rc, err));

    dir.n = 1;
    dir.v = host;
    rc = ds4_host_tensor_dir_apply(out, 2, &dir, err, sizeof err);
    printf("count %s\n", token(rc, err));

    dir.n = 2;
    dir.v = host;
    rc = ds4_host_tensor_dir_apply(NULL, 2, &dir, err, sizeof err);
    printf("out-null %s\n", token(rc, err));

    fill_host(&host[0], "");
    rc = ds4_host_tensor_dir_apply(out, 2, &dir, err, sizeof err);
    printf("name-empty %s\n", token(rc, err));
    fill_host(&host[0], "tok_embd");

    host[0].ndim = 0;
    rc = ds4_host_tensor_dir_apply(out, 2, &dir, err, sizeof err);
    printf("dim %s\n", token(rc, err));
    host[0].ndim = 2;

    memset(out, 0, sizeof out);
    rc = ds4_host_tensor_dir_apply(out, 2, &dir, err, sizeof err);
    printf("ok %s\n", token(rc, err));
    printf("row0 %s %u %u %llu %llu %llu\n",
           out[0].name, out[0].ndim, out[0].type,
           (unsigned long long)out[0].rel_offset,
           (unsigned long long)out[0].abs_offset,
           (unsigned long long)out[0].bytes);
    printf("row1 %s %u %u %llu %llu %llu\n",
           out[1].name, out[1].ndim, out[1].type,
           (unsigned long long)out[1].rel_offset,
           (unsigned long long)out[1].abs_offset,
           (unsigned long long)out[1].bytes);

    rc = ds4_host_tensor_dir_consume(out, 2, 32, 32, &dir, err, sizeof err);
    printf("then-consume %s\n", token(rc, err));
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: load_c_oracle consume-tapes|apply-tapes\n");
        return 2;
    }
    if (strcmp(argv[1], "consume-tapes") == 0) {
        consume_tapes();
        return 0;
    }
    if (strcmp(argv[1], "apply-tapes") == 0) {
        apply_tapes();
        return 0;
    }
    fprintf(stderr, "usage: load_c_oracle consume-tapes|apply-tapes\n");
    return 2;
}
