#include "ds4_bridge.h"

#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Thin wrappers over the existing engine/session API.  No inference logic
 * lives here; this file exists so Rust never includes ds4.h. */

struct ds4_bridge_model {
    ds4_engine *engine;
};

struct ds4_bridge_session {
    ds4_bridge_model *model;
    ds4_session *session;
};

static void set_err(char *err, size_t errlen, const char *msg)
{
    if (!err || errlen == 0) return;
    snprintf(err, errlen, "%s", msg ? msg : "unknown error");
}

static int map_backend(int bridge_backend, ds4_backend *out, char *err, size_t errlen)
{
    switch (bridge_backend) {
    case DS4_BRIDGE_BACKEND_CUDA:
        *out = DS4_BACKEND_CUDA;
        return 0;
    case DS4_BRIDGE_BACKEND_METAL:
        *out = DS4_BACKEND_METAL;
        return 0;
    case DS4_BRIDGE_BACKEND_CPU:
        *out = DS4_BACKEND_CPU;
        return 0;
    default:
        set_err(err, errlen, "unknown backend");
        return 1;
    }
}

int ds4_bridge_model_open(ds4_bridge_model **out,
                          const ds4_bridge_model_open_options *opt,
                          char *err, size_t errlen)
{
    ds4_engine_options eopt;
    ds4_engine *engine = NULL;
    ds4_bridge_model *m;
    int rc;

    if (out) *out = NULL;
    if (!out) {
        set_err(err, errlen, "out is NULL");
        return 1;
    }
    if (!opt || !opt->model_path || !opt->model_path[0]) {
        set_err(err, errlen, "model_path is required");
        return 1;
    }

    memset(&eopt, 0, sizeof(eopt));
    eopt.model_path = opt->model_path;
    eopt.n_threads = opt->n_threads;
    eopt.defer_boot_prewarm = opt->defer_boot_prewarm != 0;
    if (map_backend(opt->backend, &eopt.backend, err, errlen) != 0) return 1;

    rc = ds4_engine_open(&engine, &eopt);
    if (rc != 0 || !engine) {
        set_err(err, errlen, "ds4_engine_open failed");
        return rc != 0 ? rc : 1;
    }

    m = calloc(1, sizeof(*m));
    if (!m) {
        ds4_engine_close(engine);
        set_err(err, errlen, "out of memory");
        return 1;
    }
    m->engine = engine;
    *out = m;
    return 0;
}

void ds4_bridge_model_free(ds4_bridge_model *m)
{
    if (!m) return;
    ds4_engine_close(m->engine);
    free(m);
}

int ds4_bridge_session_create(ds4_bridge_session **out,
                              ds4_bridge_model *m,
                              int ctx_size,
                              char *err, size_t errlen)
{
    ds4_session *session = NULL;
    ds4_bridge_session *s;
    int rc;

    if (out) *out = NULL;
    if (!out) {
        set_err(err, errlen, "out is NULL");
        return 1;
    }
    if (!m || !m->engine) {
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    if (ctx_size <= 0) {
        set_err(err, errlen, "ctx_size must be positive");
        return 1;
    }

    rc = ds4_session_create(&session, m->engine, ctx_size);
    if (rc != 0 || !session) {
        set_err(err, errlen, "ds4_session_create failed");
        return rc != 0 ? rc : 1;
    }

    s = calloc(1, sizeof(*s));
    if (!s) {
        ds4_session_free(session);
        set_err(err, errlen, "out of memory");
        return 1;
    }
    s->model = m;
    s->session = session;
    *out = s;
    return 0;
}

void ds4_bridge_session_free(ds4_bridge_session *s)
{
    if (!s) return;
    ds4_session_free(s->session);
    free(s);
}

int ds4_bridge_session_sync(ds4_bridge_session *s,
                            const int32_t *tokens, int n_tokens,
                            char *err, size_t errlen)
{
    ds4_tokens prompt;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (n_tokens < 0) {
        set_err(err, errlen, "n_tokens is negative");
        return 1;
    }
    if (n_tokens > 0 && !tokens) {
        set_err(err, errlen, "tokens is NULL");
        return 1;
    }

    memset(&prompt, 0, sizeof(prompt));
    /* Borrowed view: ds4_session_sync must not retain prompt.v. */
    prompt.v = (int *)(void *)tokens;
    prompt.len = n_tokens;
    prompt.cap = n_tokens;
    return ds4_session_sync(s->session, &prompt, err, errlen);
}

int ds4_bridge_eval(ds4_bridge_session *s, int32_t token,
                    char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    return ds4_session_eval(s->session, (int)token, err, errlen);
}

int ds4_bridge_session_argmax(ds4_bridge_session *s)
{
    if (!s || !s->session) return -1;
    return ds4_session_argmax(s->session);
}

int ds4_bridge_session_pos(ds4_bridge_session *s)
{
    if (!s || !s->session) return -1;
    return ds4_session_pos(s->session);
}
