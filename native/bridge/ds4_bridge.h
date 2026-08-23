#ifndef DS4_BRIDGE_H
#define DS4_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Narrow Rust ↔ native ABI.  Do not include ds4.h from Rust.  Handles are
 * opaque; the C structs below may contain ds4_engine / ds4_session pointers
 * but those layouts are not part of this contract. */

typedef struct ds4_bridge_model ds4_bridge_model;
typedef struct ds4_bridge_session ds4_bridge_session;

enum {
    DS4_BRIDGE_BACKEND_CUDA = 0,
    DS4_BRIDGE_BACKEND_METAL = 1,
    DS4_BRIDGE_BACKEND_CPU = 2
};

typedef struct {
    const char *model_path;
    int backend;              /* DS4_BRIDGE_BACKEND_* */
    int n_threads;
    int defer_boot_prewarm;   /* nonzero => skip boot prewarm inside open */
} ds4_bridge_model_open_options;

/* All functions: 0 on success, nonzero on failure.  err is optional; when
 * provided it is NUL-terminated on failure.  Token pointers are borrowed
 * for the duration of the call only. */

int ds4_bridge_model_open(ds4_bridge_model **out,
                          const ds4_bridge_model_open_options *opt,
                          char *err, size_t errlen);
void ds4_bridge_model_free(ds4_bridge_model *m);

int ds4_bridge_session_create(ds4_bridge_session **out,
                              ds4_bridge_model *m,
                              int ctx_size,
                              char *err, size_t errlen);
void ds4_bridge_session_free(ds4_bridge_session *s);

int ds4_bridge_session_sync(ds4_bridge_session *s,
                            const int32_t *tokens, int n_tokens,
                            char *err, size_t errlen);
int ds4_bridge_eval(ds4_bridge_session *s, int32_t token,
                    char *err, size_t errlen);
int ds4_bridge_session_argmax(ds4_bridge_session *s);
int ds4_bridge_session_pos(ds4_bridge_session *s);

#ifdef __cplusplus
}
#endif

#endif
