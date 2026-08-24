/* C admission + job-id + host-metrics fragment oracle from ds4_server.c
 * at v0.6.3-dfm. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ENQ_OK = 0, ENQ_STOPPING, ENQ_SHED_QUEUE_DEPTH, ENQ_SHED_QUEUE_BYTES };
enum { REQ_CHAT = 0, REQ_COMPLETION = 1 };

static const char *shed_names[] = {
    "clients", "queue_depth", "queue_bytes", "queue_age", "slow_reader",
    "continuation_hold",
};
static const char *route_surface_names[] = {
    "openai_chat", "openai_completion", "anthropic_messages", "openai_responses",
};
static const char *route_lane_names[] = { "serial", "continuous", "static" };
static const char *route_reason_names[] = {
    "continuous", "static_no_cont", "static_prompt_bounds", "coalesce_off",
    "surface", "need_live_frontier", "need_continuation_publish",
    "need_corrective_recovery", "need_durable_response", "need_prefill_only",
    "token_ids_projection", "tools_completion_kind", "cont_unavailable",
    "continuous_bank_continuation",
};
static const char *think_names[] = { "none", "low", "high", "max" };

typedef struct {
    int stopping, queued, max_queue, max_clients, clients;
    unsigned long long inflight, max_queue_bytes, seq;
} admit;

static int enqueue(admit *s, unsigned long long body) {
    if (s->stopping) return ENQ_STOPPING;
    if (s->max_queue > 0 && s->queued >= s->max_queue) return ENQ_SHED_QUEUE_DEPTH;
    if (s->max_queue_bytes > 0 && s->inflight + body > s->max_queue_bytes)
        return ENQ_SHED_QUEUE_BYTES;
    s->queued++;
    s->inflight += body;
    return ENQ_OK;
}

static const char *verdict_name(int v) {
    switch (v) {
    case ENQ_OK: return "ok";
    case ENQ_STOPPING: return "stopping";
    case ENQ_SHED_QUEUE_DEPTH: return "shed_queue_depth";
    case ENQ_SHED_QUEUE_BYTES: return "shed_queue_bytes";
    default: return "unknown";
    }
}

static void mint_id(int kind, unsigned long long seq) {
    printf("%s-%llu\n", kind == REQ_CHAT ? "chatcmpl" : "cmpl", seq);
}

static void preparse(admit *s, int inference, int generation, unsigned long long incoming) {
    if (generation && s->max_clients > 0 && s->clients > s->max_clients) {
        printf("shed clients 503 10 server connection capacity reached; retry later\n");
        return;
    }
    if (inference && s->max_queue > 0 && s->queued >= s->max_queue) {
        printf("shed queue_depth 429 5 request queue is full; retry later\n");
        return;
    }
    if (inference && s->max_queue_bytes > 0 && s->inflight + incoming > s->max_queue_bytes) {
        printf("shed queue_bytes 429 5 server request-body budget exhausted; retry later\n");
        return;
    }
    printf("ok\n");
}

static void metrics_zero(void) {
    fputs("# TYPE ds4_route_requests_total counter\n", stdout);
    for (int si = 0; si < 4; si++)
        for (int li = 0; li < 3; li++)
            printf("ds4_route_requests_total{surface=\"%s\",lane=\"%s\"} 0\n",
                   route_surface_names[si], route_lane_names[li]);
    fputs("# TYPE ds4_route_decisions_total counter\n", stdout);
    for (int ri = 0; ri < 14; ri++)
        printf("ds4_route_decisions_total{reason=\"%s\"} 0\n", route_reason_names[ri]);
    fputs("# TYPE ds4_requests_shed_total counter\n", stdout);
    for (int ri = 0; ri < 6; ri++)
        printf("ds4_requests_shed_total{reason=\"%s\"} 0\n", shed_names[ri]);
    fputs("# TYPE ds4_requests_think_total counter\n", stdout);
    for (int ti = 0; ti < 4; ti++)
        printf("ds4_requests_think_total{mode=\"%s\"} 0\n", think_names[ti]);
    fputs("# TYPE ds4_clients_connected gauge\n"
          "ds4_clients_connected 0\n"
          "# TYPE ds4_queue_depth gauge\n"
          "ds4_queue_depth 0\n"
          "# TYPE ds4_inflight_body_bytes gauge\n"
          "ds4_inflight_body_bytes 0\n", stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: admit_c_oracle <cmd> [...]\n");
        return 2;
    }
    if (!strcmp(argv[1], "enqueue")) {
        if (argc < 8) return 2;
        admit s = {0};
        s.stopping = atoi(argv[2]);
        s.queued = atoi(argv[3]);
        s.max_queue = atoi(argv[4]);
        s.inflight = strtoull(argv[5], NULL, 10);
        s.max_queue_bytes = strtoull(argv[6], NULL, 10);
        unsigned long long body = strtoull(argv[7], NULL, 10);
        int v = enqueue(&s, body);
        printf("%s queued=%d inflight=%llu\n", verdict_name(v), s.queued, s.inflight);
        return 0;
    }
    if (!strcmp(argv[1], "mint")) {
        mint_id(atoi(argv[2]), strtoull(argv[3], NULL, 10));
        return 0;
    }
    if (!strcmp(argv[1], "preparse")) {
        if (argc < 11) return 2;
        admit s = {0};
        s.max_clients = atoi(argv[2]);
        s.clients = atoi(argv[3]);
        s.max_queue = atoi(argv[4]);
        s.queued = atoi(argv[5]);
        s.max_queue_bytes = strtoull(argv[6], NULL, 10);
        s.inflight = strtoull(argv[7], NULL, 10);
        unsigned long long incoming = strtoull(argv[8], NULL, 10);
        int inference = atoi(argv[9]);
        int generation = atoi(argv[10]);
        preparse(&s, inference, generation, incoming);
        return 0;
    }
    if (!strcmp(argv[1], "metrics-zero")) {
        metrics_zero();
        return 0;
    }
    fprintf(stderr, "unknown cmd\n");
    return 2;
}
