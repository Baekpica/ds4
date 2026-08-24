/* Host vocab apply oracle from native/bridge/ds4_host_load.h.
 * Do not include ds4.c. */

#include "native/bridge/ds4_host_load.h"

#include <stdio.h>
#include <string.h>

static const char *token(int rc, const char *err)
{
    return rc == 0 ? "ok" : err;
}

static void fill_ok(ds4_host_vocab *h, ds4_host_str *tokens, ds4_host_str *merges,
                    int32_t *ud)
{
    memset(h, 0, sizeof(*h));
    tokens[0].ptr = "a";
    tokens[0].len = 1;
    tokens[1].ptr = "bb";
    tokens[1].len = 2;
    merges[0].ptr = "a b";
    merges[0].len = 3;
    ud[0] = 1;
    h->n_vocab = 2;
    h->tokens = tokens;
    h->n_merges = 1;
    h->merges = merges;
    h->n_user_defined = 1;
    h->user_defined = ud;
    h->user_defined_max_len = 2;
    h->bos_id = 0;
    h->eos_id = 1;
}

int main(void)
{
    ds4_host_vocab h;
    ds4_host_str tokens[2];
    ds4_host_str merges[1];
    int32_t ud[1];
    char err[64];
    int rc;

    rc = ds4_host_vocab_apply(NULL, err, sizeof err);
    printf("vocab-null %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    h.n_vocab = 1;
    h.tokens = NULL;
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("tokens-null %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    h.n_merges = 1;
    h.merges = NULL;
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("merges-null %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    h.n_user_defined = 1;
    h.user_defined = NULL;
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("ud-null %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    tokens[0].ptr = NULL;
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("token-empty %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    merges[0].ptr = NULL;
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("merge-empty %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    ud[0] = 9;
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("ud-range %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    tokens[0].ptr = "";
    tokens[0].len = 0;
    ud[0] = 0;
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("ud-empty %s\n", token(rc, err));

    fill_ok(&h, tokens, merges, ud);
    rc = ds4_host_vocab_apply(&h, err, sizeof err);
    printf("ok %s\n", token(rc, err));
    printf("ok-row n_vocab=2 n_merges=1 n_ud=1 max_ud=2 bos=0 eos=1 token0=61 token1=6262 merge0=612062 ud=1\n");
    return 0;
}
