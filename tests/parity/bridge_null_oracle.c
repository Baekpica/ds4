/* NULL-handle FFI checks for the Phase 7 generation ABI. Links
 * ds4_bridge.o against stubs so it does not load CUDA or a GGUF. */

#include "native/bridge/ds4_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *m) {
    fprintf(stderr, "bridge_null_oracle: %s\n", m);
    exit(1);
}

int main(void) {
    char err[64];
    int32_t toks[4];
    int n = -1;
    char text[8];
    size_t tn = 99;
    uint64_t rng = 1;

    memset(err, 0, sizeof(err));
    if (ds4_bridge_tokenize_text(NULL, "hi", toks, 4, &n, err, sizeof(err)) == 0)
        fail("tokenize_text NULL model");
    if (n != 0) fail("tokenize_text n_out");
    if (!strstr(err, "NULL")) fail("tokenize_text err");

    memset(err, 0, sizeof(err));
    n = -1;
    if (ds4_bridge_tokenize_rendered_chat(NULL, "hi", toks, 4, &n, err, sizeof(err)) == 0)
        fail("tokenize_chat NULL model");
    if (n != 0) fail("tokenize_chat n_out");

    memset(err, 0, sizeof(err));
    if (ds4_bridge_token_text(NULL, 1, text, sizeof(text), &tn, err, sizeof(err)) == 0)
        fail("token_text NULL model");
    if (tn != 0) fail("token_text n_out");

    if (ds4_bridge_token_eos(NULL) != -1) fail("token_eos");
    if (ds4_bridge_token_is_stop(NULL, 1) != 0) fail("token_is_stop");
    if (ds4_bridge_model_id(NULL) != -1) fail("model_id");
    if (ds4_bridge_session_sample(NULL, 1.0f, 0, 1.0f, 0.05f, &rng) != -1)
        fail("sample");
    if (ds4_bridge_session_ctx(NULL) != -1) fail("ctx");
    if (ds4_bridge_session_argmax(NULL) != -1) fail("argmax");
    if (ds4_bridge_session_pos(NULL) != -1) fail("pos");

    if (ds4_bridge_mem_census_snap(NULL) == 0) fail("census NULL");
    if (ds4_bridge_mem_observe_snap(NULL) == 0) fail("observe NULL");
    {
        ds4_bridge_mem_census img;
        ds4_bridge_mem_observe o;
        memset(&img, 0xff, sizeof(img));
        memset(&o, 0xff, sizeof(o));
        if (ds4_bridge_mem_census_snap(&img) != 0) fail("census rc");
        if (img.supported != 0) fail("census supported");
        if (img.faults != 0) fail("census faults");
        if (img.epoch != 0) fail("census epoch");
        if (img.torn_fallbacks != 0) fail("census torn");
        if (img.cells[0][0].requested != 0) fail("census cell");
        if (sizeof(ds4_bridge_mem_cell) != 48) fail("cell sizeof");
        if (ds4_bridge_mem_observe_snap(&o) != 0) fail("observe rc");
        if (o.status != 1) fail("observe unsupported");
        if (o.source != 0) fail("observe source");
        if (o.free_bytes != 0) fail("observe free");
        if (ds4_bridge_mem_substrate_outstanding() != 0) fail("substrate");
    }

    {
        ds4_bridge_batch_ctx *bc = (ds4_bridge_batch_ctx *)0;
        memset(err, 0, sizeof(err));
        if (ds4_bridge_batch_ctx_create_fit(NULL, 2048, 4, 8192, &bc,
                                            err, sizeof(err)) == 0)
            fail("batch_ctx NULL model");
        if (bc != NULL) fail("batch_ctx out");
        if (!strstr(err, "NULL")) fail("batch_ctx err");
        ds4_bridge_batch_ctx_destroy(NULL);
        if (ds4_bridge_batch_ctx_max_seq(NULL) != 0) fail("batch max_seq");
        if (ds4_bridge_batch_ctx_seq_cap(NULL) != 0) fail("batch seq_cap");
        memset(err, 0, sizeof(err));
        if (ds4_bridge_continuous_generate(NULL, NULL, NULL, NULL, NULL,
                                           err, sizeof(err)) == 0)
            fail("cont NULL ctx");
        if (!strstr(err, "NULL")) fail("cont err");
    }

    printf("ok\n");
    return 0;
}
