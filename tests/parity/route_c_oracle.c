/* C route_decide / request_compute_needs oracle. Copied from ds4_server.c
 * at v0.6.3-dfm so Rust can compare lane+reason without linking the server. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DS4_NEED_STREAMING            = 1u << 0,
    DS4_NEED_PER_ROW_SAMPLING     = 1u << 1,
    DS4_NEED_THINKING             = 1u << 2,
    DS4_NEED_STOP_SCAN            = 1u << 3,
    DS4_NEED_TOOL_SCAN            = 1u << 4,
    DS4_NEED_TOKEN_IDS            = 1u << 5,
    DS4_NEED_LIVE_FRONTIER        = 1u << 6,
    DS4_NEED_CONTINUATION_PUBLISH = 1u << 7,
    DS4_NEED_CORRECTIVE_RECOVERY  = 1u << 8,
    DS4_NEED_DURABLE_RESPONSE     = 1u << 9,
    DS4_NEED_PREFILL_ONLY         = 1u << 10,
    DS4_NEED_BANK_FRONTIER        = 1u << 11,
};

#define ROUTE_CONT_MASK  (DS4_NEED_STREAMING | DS4_NEED_PER_ROW_SAMPLING | \
                          DS4_NEED_THINKING | DS4_NEED_STOP_SCAN |         \
                          DS4_NEED_TOOL_SCAN | DS4_NEED_TOKEN_IDS)
#define ROUTE_STATIC_MASK ((uint32_t)0)
#define ROUTE_LANE_NONE 0xFF
enum { ROUTE_LANE_SERIAL = 0, ROUTE_LANE_CONTINUOUS, ROUTE_LANE_STATIC };

enum {
    DS4_WIRE_OPENAI_CHAT,
    DS4_WIRE_OPENAI_COMPLETION,
    DS4_WIRE_ANTHROPIC,
    DS4_WIRE_RESPONSES,
};

enum {
    ROUTE_REASON_CONT = 0,
    ROUTE_REASON_STATIC_NO_CONT,
    ROUTE_REASON_STATIC_PROMPT_BOUNDS,
    ROUTE_REASON_COALESCE_OFF,
    ROUTE_REASON_SURFACE,
    ROUTE_REASON_NEED_LIVE_FRONTIER,
    ROUTE_REASON_NEED_CONTINUATION_PUBLISH,
    ROUTE_REASON_NEED_CORRECTIVE_RECOVERY,
    ROUTE_REASON_NEED_DURABLE,
    ROUTE_REASON_NEED_PREFILL_ONLY,
    ROUTE_REASON_TOKEN_IDS_PROJECTION,
    ROUTE_REASON_TOOLS_COMPLETION,
    ROUTE_REASON_CONT_UNAVAILABLE,
    ROUTE_REASON_CONT_BANK,
};

typedef struct {
    bool coalesce, have_cont, cont_anthropic, cont_responses;
    bool cont_tools_anthropic, cont_tools_responses;
    int seq_cap, prompt_len;
} ds4_route_env;

typedef struct { uint8_t lane; uint8_t reason; } ds4_route_decision;

static ds4_route_decision route_decide(uint32_t needs, int surf,
                                       const ds4_route_env *env) {
    ds4_route_decision d = { ROUTE_LANE_SERIAL, 0 };
    if (needs & DS4_NEED_DURABLE_RESPONSE) {
        d.lane = ROUTE_LANE_NONE; d.reason = ROUTE_REASON_NEED_DURABLE; return d;
    }
    const bool tools_promoted =
        (env->cont_anthropic && env->cont_tools_anthropic &&
         surf == DS4_WIRE_ANTHROPIC) ||
        (env->cont_responses && env->cont_tools_responses &&
         surf == DS4_WIRE_RESPONSES);
    if (needs & DS4_NEED_LIVE_FRONTIER) {
        d.reason = ROUTE_REASON_NEED_LIVE_FRONTIER; return d;
    }
    if ((needs & DS4_NEED_BANK_FRONTIER) && !tools_promoted) {
        d.reason = ROUTE_REASON_NEED_LIVE_FRONTIER; return d;
    }
    if (needs & DS4_NEED_CONTINUATION_PUBLISH) {
        if (!((needs & DS4_NEED_STREAMING) && tools_promoted)) {
            d.reason = ROUTE_REASON_NEED_CONTINUATION_PUBLISH; return d;
        }
    }
    if (needs & DS4_NEED_CORRECTIVE_RECOVERY) {
        d.reason = ROUTE_REASON_NEED_CORRECTIVE_RECOVERY; return d;
    }
    if (needs & DS4_NEED_PREFILL_ONLY) {
        d.reason = ROUTE_REASON_NEED_PREFILL_ONLY; return d;
    }
    if ((needs & DS4_NEED_TOKEN_IDS) &&
        !((needs & DS4_NEED_STREAMING) && surf == DS4_WIRE_OPENAI_CHAT)) {
        d.reason = ROUTE_REASON_TOKEN_IDS_PROJECTION; return d;
    }
    if ((needs & DS4_NEED_TOOL_SCAN) && surf == DS4_WIRE_OPENAI_COMPLETION) {
        d.reason = ROUTE_REASON_TOOLS_COMPLETION; return d;
    }
    uint32_t cont_mask = ROUTE_CONT_MASK;
    if (tools_promoted) {
        if (needs & DS4_NEED_STREAMING)
            cont_mask |= DS4_NEED_CONTINUATION_PUBLISH;
        cont_mask |= DS4_NEED_BANK_FRONTIER;
    }
    if (needs & ~cont_mask) {
        d.reason = ROUTE_REASON_CONT_UNAVAILABLE; return d;
    }
    const bool cont_promoted =
        (env->cont_anthropic && surf == DS4_WIRE_ANTHROPIC) ||
        (env->cont_responses && surf == DS4_WIRE_RESPONSES);
    if (surf != DS4_WIRE_OPENAI_CHAT && surf != DS4_WIRE_OPENAI_COMPLETION &&
        !cont_promoted) {
        d.reason = ROUTE_REASON_SURFACE; return d;
    }
    if (!env->coalesce) { d.reason = ROUTE_REASON_COALESCE_OFF; return d; }
    if (env->have_cont && env->prompt_len > 0 && env->prompt_len <= env->seq_cap) {
        d.lane = ROUTE_LANE_CONTINUOUS;
        d.reason = (needs & DS4_NEED_BANK_FRONTIER) ? ROUTE_REASON_CONT_BANK
                                                    : ROUTE_REASON_CONT;
        return d;
    }
    if ((needs & ~ROUTE_STATIC_MASK) == 0 &&
        (surf == DS4_WIRE_OPENAI_CHAT || surf == DS4_WIRE_OPENAI_COMPLETION ||
         cont_promoted)) {
        d.lane = ROUTE_LANE_STATIC;
        d.reason = env->have_cont ? ROUTE_REASON_STATIC_PROMPT_BOUNDS
                                  : ROUTE_REASON_STATIC_NO_CONT;
        return d;
    }
    d.reason = ROUTE_REASON_CONT_UNAVAILABLE;
    return d;
}

enum { API_OPENAI = 0, API_ANTHROPIC, API_RESPONSES };

static uint32_t compute_needs(int api, int stream, float temperature, int think,
                              int stop_count, int has_tools, int echo,
                              int resp_live_tool, int resp_live_reason,
                              int anth_live_tool, int bank_owned,
                              int max_set, int max_tokens) {
    uint32_t n = 0;
    if (stream) n |= DS4_NEED_STREAMING;
    if (temperature > 0.0f) n |= DS4_NEED_PER_ROW_SAMPLING;
    if (think) n |= DS4_NEED_THINKING;
    if (stop_count > 0) n |= DS4_NEED_STOP_SCAN;
    if (has_tools) n |= DS4_NEED_TOOL_SCAN;
    if (echo) n |= DS4_NEED_TOKEN_IDS;
    if (resp_live_tool || resp_live_reason || anth_live_tool) {
        const int bank = bank_owned && !resp_live_reason;
        n |= bank ? DS4_NEED_BANK_FRONTIER : DS4_NEED_LIVE_FRONTIER;
    }
    if (has_tools && api != API_OPENAI) {
        n |= DS4_NEED_CONTINUATION_PUBLISH;
        if (!stream) n |= DS4_NEED_CORRECTIVE_RECOVERY;
    }
    if (api == API_ANTHROPIC && max_set && max_tokens <= 0)
        n |= DS4_NEED_PREFILL_ONLY;
    return n;
}

static void die(const char *m) { fprintf(stderr, "route_c_oracle: %s\n", m); exit(2); }
static int need_i(const char *s) { return (int)strtol(s, NULL, 0); }
static uint32_t need_u(const char *s) { return (uint32_t)strtoul(s, NULL, 0); }

int main(int argc, char **argv) {
    if (argc < 2) die("usage");
    if (!strcmp(argv[1], "decide")) {
        if (argc < 12) die("decide NEEDS SURF coal have ca cr ta tr seq plen");
        ds4_route_env env = {
            .coalesce = need_i(argv[4]) != 0,
            .have_cont = need_i(argv[5]) != 0,
            .cont_anthropic = need_i(argv[6]) != 0,
            .cont_responses = need_i(argv[7]) != 0,
            .cont_tools_anthropic = need_i(argv[8]) != 0,
            .cont_tools_responses = need_i(argv[9]) != 0,
            .seq_cap = need_i(argv[10]),
            .prompt_len = need_i(argv[11]),
        };
        ds4_route_decision d = route_decide(need_u(argv[2]), need_i(argv[3]), &env);
        printf("lane=%u reason=%u", d.lane, d.reason);
    } else if (!strcmp(argv[1], "needs")) {
        if (argc < 15) die("needs api stream temp think stops tools echo rlt rlr alt bank maxset maxt");
        float temp = (float)strtod(argv[4], NULL);
        uint32_t n = compute_needs(
            need_i(argv[2]), need_i(argv[3]), temp, need_i(argv[5]),
            need_i(argv[6]), need_i(argv[7]), need_i(argv[8]),
            need_i(argv[9]), need_i(argv[10]), need_i(argv[11]),
            need_i(argv[12]), need_i(argv[13]), need_i(argv[14]));
        printf("%u", n);
    } else {
        die("unknown command");
    }
    return 0;
}
