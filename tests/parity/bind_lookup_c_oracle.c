/* Host bind-map lookup oracle from native/bridge/ds4_host_load.h.
 * Do not include ds4.c. */

#include "native/bridge/ds4_host_load.h"

#include <stdio.h>
#include <string.h>

static void print_lookup(const char *label, int rc, const char *err, uint32_t idx)
{
    if (rc == 0) {
        if (idx == DS4_HOST_BIND_MISS) printf("%s miss\n", label);
        else printf("%s %u\n", label, idx);
    } else if (rc == 2) {
        printf("%s unknown\n", label);
    } else {
        printf("%s %s\n", label, err);
    }
}

static void fill_ok(ds4_host_bind_map *m, ds4_host_bind_look *v)
{
    memset(v, 0, 3 * sizeof(*v));
    v[0].name = "token_embd.weight";
    v[0].required = 1;
    v[0].found = 1;
    v[0].index = 3;
    v[1].name = "exp_probs_b.bias";
    v[1].required = 0;
    v[1].found = 0;
    v[1].index = DS4_HOST_BIND_MISS;
    v[2].name = "output.weight";
    v[2].required = 1;
    v[2].found = 0;
    v[2].index = DS4_HOST_BIND_MISS;
    m->n = 3;
    m->v = v;
}

int main(void)
{
    ds4_host_bind_map m;
    ds4_host_bind_look v[3];
    char err[64];
    uint32_t idx;
    int rc;

    rc = ds4_host_bind_lookup(NULL, "token_embd.weight", 8, &idx, err, sizeof err);
    print_lookup("map-null", rc, err, idx);

    fill_ok(&m, v);
    rc = ds4_host_bind_lookup(&m, NULL, 8, &idx, err, sizeof err);
    print_lookup("name-empty", rc, err, idx);

    fill_ok(&m, v);
    m.n = 3;
    m.v = NULL;
    rc = ds4_host_bind_lookup(&m, "token_embd.weight", 8, &idx, err, sizeof err);
    print_lookup("looks-null", rc, err, idx);

    fill_ok(&m, v);
    rc = ds4_host_bind_lookup(&m, "not.in.plan", 8, &idx, err, sizeof err);
    print_lookup("unknown", rc, err, idx);

    fill_ok(&m, v);
    rc = ds4_host_bind_lookup(&m, "output.weight", 8, &idx, err, sizeof err);
    print_lookup("missing", rc, err, idx);

    fill_ok(&m, v);
    rc = ds4_host_bind_lookup(&m, "exp_probs_b.bias", 8, &idx, err, sizeof err);
    print_lookup("miss", rc, err, idx);

    fill_ok(&m, v);
    v[0].index = 8;
    rc = ds4_host_bind_lookup(&m, "token_embd.weight", 8, &idx, err, sizeof err);
    print_lookup("index-range", rc, err, idx);

    fill_ok(&m, v);
    v[0].index = DS4_HOST_BIND_MISS;
    rc = ds4_host_bind_lookup(&m, "token_embd.weight", 8, &idx, err, sizeof err);
    print_lookup("index-miss", rc, err, idx);

    fill_ok(&m, v);
    v[0].name = NULL;
    rc = ds4_host_bind_lookup(&m, "output.weight", 8, &idx, err, sizeof err);
    print_lookup("slot-empty", rc, err, idx);

    fill_ok(&m, v);
    rc = ds4_host_bind_lookup(&m, "token_embd.weight", 8, &idx, err, sizeof err);
    print_lookup("ok", rc, err, idx);

    return 0;
}
