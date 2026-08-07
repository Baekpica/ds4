#!/bin/bash
# cont_registry_gate.sh — v0.5.6 API Inc 5a+5b live gate: generation-checked
# continuation registry (plan §4.6 record shape + retention policy + §5
# Inc 5 gate line, the serial-owner slice; bank owners land in 5c).
#
# What 5a changed: serial tool turns publish ONE continuation record binding
# the turn's call ids to the engine execution reference (session generation +
# committed frontier); T2 admission resolves through the registry with pure
# equality (the old singletons compared only the position -- ABA); the two
# continuation-state refusals are now REGISTRY MISSES with counters; exact
# DSML for a LIVE frontier is pinned against the tool-memory LRU.
#
# Legs (one zero-config boot, all traffic serial BY NEEDS -- tools /
# live-frontier / token-ids never left serial in Inc 3/4):
#   anth_t1           /v1/messages tool turn: stop_reason tool_use +
#                     ds4_continuation_records_published_total +1 +
#                     records_live 1 + need_continuation_publish decision
#   anth_t2_live      OUTPUT-ONLY tool_result (no assistant replay): 200,
#                     resolved_total +1, srv.log "anthropic live
#                     continuation match=tool-output-ids", usage
#                     cache_read_input_tokens > 0 (the live KV was kept)
#   anth_t2_replayonly the SAME output-only T2 again: the record is
#                     REPLAY_ONLY now -> parse-side 400 with the
#                     continuation-state message in the anthropic envelope
#   anth_t2_replay    full-history T2 (assistant tool_use replayed): 200 --
#                     stateless replay stays exact after demotion
#   anth_t2_race409   T1' publishes; a SERIAL unrelated request (token-ids
#                     blocker, deep prompt) is queued FIRST; the output-only
#                     T2' parses while the record is LIVE, then admission
#                     finds it demoted -> native 409 invalid_request_error +
#                     missed_total +1 (the worker-side registry miss)
#   resp_t1           /v1/responses function-call turn: published +1
#   resp_t2_live      function_call_output-only: 200, resolved +1, srv.log
#                     "responses live continuation"
#   resp_t2_replayonly same T2 again -> 400 "retry by replaying the full
#                     input history"
#   resp_t2_replay    full-history T2 (function_call item replayed): 200
#   metrics_final     all five ds4_continuation_* series present and the
#                     ledger adds up (published >= 3, resolved == 2,
#                     missed == 1)
#
# Boot A runs with the retention policy ZEROED (DS4_CONT_GRACE_S=0,
# DS4_CONT_PIN_DEADLINE_S=0): the 5a legs prove the registry's resolution
# semantics, and the race409 leg NEEDS the unrelated blocker to win the
# session -- under default grace it would shed instead (which is boot B's
# grace_shed leg proving exactly that).
#
# Boot B legs (Inc 5b, DS4_CONT_GRACE_S=6 DS4_CONT_TTL_S=45, default pin):
#   grace_shed        an unrelated SERIAL request inside the grace window
#                     sheds 503+Retry-After with shed{continuation_hold} +1
#                     and the record STAYS LIVE; the T2 then resolves 200
#                     (grace protected the frontier for its owner)
#   postgrace_demote  after the grace window the same unrelated request
#                     proceeds (200) and demotes; the T2 refuses natively
#   ttl_expire        an unclaimed record lazily demotes after the soft TTL (45s)
#                     (records_live 1 -> 0, output-only T2 refused)
#   pin_wins          a cont request occupies the worker; an unrelated
#                     serial blocker queues; the T2 parses AFTER it (FIFO
#                     behind it) and pins the record -- at admission the
#                     blocker sheds on the PIN (grace long over) and the T2
#                     resolves 200: the plan's "admission ... sheds instead
#                     of violating a pin", live
#
# Runs FROM the Mac over SSH like the other gates.  Boots kill any running
# ds4-server on $R.  End state: ds4-server killed, box left free.
#
# Env overrides: R (sync-192_168_88_33) BINDIR (/home/ent/code/ds4-phase0)
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
set -uo pipefail

R=${R:-sync-192_168_88_33}
BINDIR=${BINDIR:-/home/ent/code/ds4-phase0}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/cont_registry_gate
OUT=${OUT:-/tmp/cont_registry_gate_$$}
mkdir -p "$OUT"
BASE="http://127.0.0.1:$TUNNEL_PORT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; exit 1; }

tunnel_up(){
  curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 && return 0
  ssh -f -N -L "$TUNNEL_PORT:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "$BASE/v1/models" >/dev/null 2>&1
}

wait_mem(){ # $1=min MemAvailable GiB
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}

boot(){
  SRV=$RWORK/srv.log
  log "boot: killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server -c $CTX --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot timeout"
  done
  tunnel_up || fail "tunnel :$TUNNEL_PORT unreachable"
  log "boot: up"
}

post(){
  curl -s -m 240 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
post_file(){
  curl -s -m 240 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d @"$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1.json"))"; }
# METRICS-HELPER TRAP: the # TYPE line matches the fixed-string grep but
# carries no trailing number, so extract the number BEFORE head -1.
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
srv_count(){ ssh "$R" "grep -cF \"$1\" $RWORK/srv.log" 2>/dev/null; }

TOOLS_ANTH='[{"name":"list_files","description":"List the files in a directory","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'
TOOLS_RESP='[{"type":"function","name":"list_files","description":"List the files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'

BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot

# ==================== boot A: anthropic block (5a semantics) ================
pub0=$(m ds4_continuation_records_published_total)
res0=$(m ds4_continuation_resolved_total)
c=$(post anth_t1 /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":"Use the list_files tool to list the files in /tmp. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is anth_t1 "$c" 200
has anth_t1 '"stop_reason":"tool_use"'
has anth_t1 '"type":"tool_use"'
pub1=$(m ds4_continuation_records_published_total)
live1=$(m ds4_continuation_records_live)
dec1=$(m 'ds4_route_decisions_total{reason="need_continuation_publish"}')
[ "${pub1:-0}" -gt "${pub0:-0}" ] || fail "anth_t1: no continuation record published (${pub0:-?} -> ${pub1:-?})"
[ "${live1:-0}" -eq 1 ] || fail "anth_t1: records_live ${live1:-?}, want 1"
[ "${dec1:-0}" -ge 1 ] || fail "anth_t1: need_continuation_publish decision never recorded"
ANTH_ID=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); print(next(b["id"] for b in t["content"] if b.get("type")=="tool_use"))' "$OUT/anth_t1.json") || fail "anth_t1: no tool_use id"
log "anth_t1 PASS (tool turn published a LIVE record; id=$ANTH_ID)"

# anth_t2_live: OUTPUT-ONLY -- no assistant replay in the request.  Resolution
# must come from the registry record (generation + frontier), and the live KV
# is kept: the whole prior turn reads back as cache.
T2_BODY='{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"'"$ANTH_ID"'","content":"file1.txt\nfile2.txt"}]}],"tools":'"$TOOLS_ANTH"'}'
c=$(post anth_t2_live /v1/messages "$T2_BODY")
code_is anth_t2_live "$c" 200
res1=$(m ds4_continuation_resolved_total)
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "anth_t2_live: registry never resolved (${res0:-?} -> ${res1:-?})"
[ "$(srv_count 'anthropic live continuation match=tool-output-ids')" -ge 1 ] || fail "anth_t2_live: live-continuation log line missing"
python3 - "$OUT/anth_t2_live.json" <<'PY' || fail "anth_t2_live: cache_read_input_tokens not > 0 (live KV was not kept)"
import json, sys
u = json.load(open(sys.argv[1]))["usage"]
assert u.get("cache_read_input_tokens", 0) > 0, u
PY
log "anth_t2_live PASS (output-only T2 continued the live frontier; resolved ${res0:-0} -> $res1)"

# anth_t2_replayonly: the record was consumed/demoted by the T2 turn.  The
# SAME output-only request now has no live binding and no replayable prefix:
# the parser refuses 400 with the continuation-state message (REPLAY_ONLY is
# honest at parse time; the 409 twin is the admission race below).
c=$(post anth_t2_replayonly /v1/messages "$T2_BODY")
code_is anth_t2_replayonly "$c" 400
has anth_t2_replayonly 'continuation state is not available'
has anth_t2_replayonly '"type":"invalid_request_error"'
log "anth_t2_replayonly PASS (demoted record refuses output-only at parse, native envelope)"

# anth_t2_replay: full-history T2 replays the assistant tool_use -- the
# REPLAY_ONLY promise: exact DSML replay still serves after demotion.
python3 - "$OUT/anth_t1.json" "$ANTH_ID" > "$OUT/anth_replay_req.json" <<'PY' || fail "anth_t2_replay: request build failed"
import json, sys
t1 = json.load(open(sys.argv[1]))
tid = sys.argv[2]
tool_use = next(b for b in t1["content"] if b.get("type") == "tool_use")
msgs = [
    {"role": "user", "content": "Use the list_files tool to list the files in /tmp. Call the tool."},
    {"role": "assistant", "content": [b for b in t1["content"] if b.get("type") in ("text", "tool_use")]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": "file1.txt\nfile2.txt"}]},
]
tools = [{"name": "list_files", "description": "List the files in a directory",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
print(json.dumps({"model": "m", "max_tokens": 1200, "temperature": 0, "messages": msgs, "tools": tools}))
PY
c=$(post_file anth_t2_replay /v1/messages "$OUT/anth_replay_req.json")
code_is anth_t2_replay "$c" 200
has anth_t2_replay '"role":"assistant"'
log "anth_t2_replay PASS (full-history replay serves after demotion)"

# anth_t2_race409: the worker-side registry miss.  T1' publishes a fresh
# record; a SERIAL unrelated request (token-ids buffered = serial by needs,
# deep prompt = a long race window) queues FIRST; the output-only T2' parses
# while the record is still LIVE, then admission finds it demoted -> 409.
c=$(post anth_race_t1 /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":"Use the list_files tool to list the files in /home. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is anth_race_t1 "$c" 200
has anth_race_t1 '"type":"tool_use"'
RACE_ID=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); print(next(b["id"] for b in t["content"] if b.get("type")=="tool_use"))' "$OUT/anth_race_t1.json") || fail "anth_race_t1: no tool_use id"
miss0=$(m ds4_continuation_missed_total)
python3 - > "$OUT/blocker_req.json" <<'PY'
import json
sent = "The registry keeps one record per turn and checks the generation before continuing. "
print(json.dumps({"model": "m", "max_tokens": 16, "temperature": 0, "return_token_ids": True,
                  "messages": [{"role": "user", "content": sent * 500}]}))
PY
curl -s -m 240 -o "$OUT/blocker.json" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @"$OUT/blocker_req.json" &
BLOCKER_PID=$!
sleep 1
c=$(post anth_t2_race409 /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"'"$RACE_ID"'","content":"a b c"}]}],"tools":'"$TOOLS_ANTH"'}')
wait "$BLOCKER_PID" 2>/dev/null
code_is anth_t2_race409 "$c" 409
has anth_t2_race409 'continuation state is not available'
has anth_t2_race409 '"type":"invalid_request_error"'
miss1=$(m ds4_continuation_missed_total)
[ "${miss1:-0}" -gt "${miss0:-0}" ] || fail "anth_t2_race409: missed_total never moved (${miss0:-?} -> ${miss1:-?})"
log "anth_t2_race409 PASS (LIVE at parse, demoted at admission -> native 409; missed ${miss0:-0} -> $miss1)"

# ==================== responses block =======================================
pubA=$(m ds4_continuation_records_published_total)
resA=$(m ds4_continuation_resolved_total)
c=$(post resp_t1 /v1/responses '{"model":"m","max_output_tokens":1200,"temperature":0,"input":[{"role":"user","content":"Use the list_files tool to list the files in /tmp. Call the tool."}],"tools":'"$TOOLS_RESP"'}')
code_is resp_t1 "$c" 200
has resp_t1 '"type":"function_call"'
pubB=$(m ds4_continuation_records_published_total)
[ "${pubB:-0}" -gt "${pubA:-0}" ] || fail "resp_t1: no continuation record published (${pubA:-?} -> ${pubB:-?})"
RESP_ID=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); print(next(o["call_id"] for o in t["output"] if o.get("type")=="function_call"))' "$OUT/resp_t1.json") || fail "resp_t1: no function call_id"
log "resp_t1 PASS (function-call turn published; call_id=$RESP_ID)"

RESP_T2_BODY='{"model":"m","max_output_tokens":1200,"temperature":0,"input":[{"type":"function_call_output","call_id":"'"$RESP_ID"'","output":"file1.txt\nfile2.txt"}],"tools":'"$TOOLS_RESP"'}'
c=$(post resp_t2_live /v1/responses "$RESP_T2_BODY")
code_is resp_t2_live "$c" 200
resB=$(m ds4_continuation_resolved_total)
[ "${resB:-0}" -gt "${resA:-0}" ] || fail "resp_t2_live: registry never resolved (${resA:-?} -> ${resB:-?})"
[ "$(srv_count 'responses live continuation')" -ge 1 ] || fail "resp_t2_live: live-continuation log line missing"
log "resp_t2_live PASS (output-only T2 continued; resolved ${resA:-0} -> $resB)"

c=$(post resp_t2_replayonly /v1/responses "$RESP_T2_BODY")
code_is resp_t2_replayonly "$c" 400
has resp_t2_replayonly 'retry by replaying the full input history'
log "resp_t2_replayonly PASS (demoted record refuses output-only at parse)"

python3 - "$OUT/resp_t1.json" "$RESP_ID" > "$OUT/resp_replay_req.json" <<'PY' || fail "resp_t2_replay: request build failed"
import json, sys
t1 = json.load(open(sys.argv[1]))
cid = sys.argv[2]
fc = next(o for o in t1["output"] if o.get("type") == "function_call")
inp = [
    {"role": "user", "content": "Use the list_files tool to list the files in /tmp. Call the tool."},
    {"type": "function_call", "call_id": cid, "name": fc["name"], "arguments": fc["arguments"]},
    {"type": "function_call_output", "call_id": cid, "output": "file1.txt\nfile2.txt"},
]
tools = [{"type": "function", "name": "list_files", "description": "List the files in a directory",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
print(json.dumps({"model": "m", "max_output_tokens": 1200, "temperature": 0, "input": inp, "tools": tools}))
PY
c=$(post_file resp_t2_replay /v1/responses "$OUT/resp_replay_req.json")
code_is resp_t2_replay "$c" 200
has resp_t2_replay '"status":"completed"'
log "resp_t2_replay PASS (full-history replay serves after demotion)"

# ==================== boot A ledger =========================================
pubF=$(m ds4_continuation_records_published_total)
resF=$(m ds4_continuation_resolved_total)
misF=$(m ds4_continuation_missed_total)
demF=$(m ds4_continuation_demoted_total)
livF=$(m ds4_continuation_records_live)
[ -n "$pubF" ] && [ -n "$resF" ] && [ -n "$misF" ] && [ -n "$demF" ] && [ -n "$livF" ] || \
  fail "metrics_final: a ds4_continuation_* series is missing from /metrics"
[ "$pubF" -ge 3 ] || fail "metrics_final: published $pubF, want >= 3"
[ "$resF" -eq 2 ] || fail "metrics_final: resolved $resF, want exactly 2"
[ "$misF" -eq 1 ] || fail "metrics_final: missed $misF, want exactly 1"
# Ledger identity: every publish increments live, every demote decrements,
# nothing else touches it -- so live == published - demoted, always.
[ "$livF" -eq $((pubF - demF)) ] || \
  fail "metrics_final: ledger identity broken: live $livF != published $pubF - demoted $demF"
log "metrics_final PASS (published=$pubF resolved=$resF missed=$misF demoted=$demF live=$livF)"

# ==================== boot B: Inc 5b retention policy =======================
# grace 6s (short enough to wait out in a leg), soft TTL 45s (LONGER than
# the pin leg's grace-wait + occupier window ~27s -- a 20s TTL demoted the
# pinned record mid-leg on the first run, found live 17:30), pin deadline
# default 60s.  All traffic below is serial by needs except the cont
# occupier in pin_wins.
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=6 DS4_CONT_TTL_S=45" boot

t1(){ # $1=name $2=user text -> publishes a tool turn; sets T1_ID.
      # Precondition: no leftover LIVE record (a prior T2 may have chained).
  drain_frontier
  local c
  c=$(post "$1" /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":"'"$2"'"}],"tools":'"$TOOLS_ANTH"'}')
  code_is "$1" "$c" 200
  has "$1" '"type":"tool_use"'
  T1_ID=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); print(next(b["id"] for b in t["content"] if b.get("type")=="tool_use"))' "$OUT/$1.json") || fail "$1: no tool_use id"
}
t2_body(){ # $1=id  (a MEANINGFUL result: "ok" invites the model to retry
           # the tool, chaining a new call -> a fresh record -> a fresh
           # grace window that sheds the NEXT leg's T1; found live 17:24)
  echo '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"'"$1"'","content":"file1.txt\nfile2.txt"}]}],"tools":'"$TOOLS_ANTH"'}'
}
BLOCKER_BODY=$(python3 -c 'import json; print(json.dumps({"model":"m","max_tokens":16,"temperature":0,"return_token_ids":True,"messages":[{"role":"user","content":"Say hello."}]}))')

# A T2 answer may still legally end in ANOTHER tool call (the model's
# choice), leaving a fresh LIVE record + grace window behind a leg.  Drain
# deterministically: outwait the grace and let an unrelated serial win
# demote, bounded at 3 rounds.
drain_frontier(){
  local n=0
  while [ "$(m ds4_continuation_records_live)" -gt 0 ]; do
    n=$((n+1))
    [ $n -gt 3 ] && fail "drain_frontier: records_live stuck > 0 after 3 rounds"
    sleep 7
    curl -s -m 240 -o "$OUT/drain_$n.json" "$BASE/v1/chat/completions" \
         -H 'Content-Type: application/json' -d "$BLOCKER_BODY" >/dev/null
  done
}

# grace_shed: unrelated serial work inside the grace window sheds and the
# frontier survives for its continuation.
t1 grace_t1 "Use the list_files tool to list the files in /var. Call the tool."
shed0=$(m 'ds4_requests_shed_total{reason="continuation_hold"}')
c=$(post grace_blocked /v1/chat/completions "$BLOCKER_BODY")
code_is grace_blocked "$c" 503
grep -qi "reserved for a live tool continuation" "$OUT/grace_blocked.json" || \
  fail "grace_blocked: shed message missing ($(head -c 200 "$OUT/grace_blocked.json"))"
shed1=$(m 'ds4_requests_shed_total{reason="continuation_hold"}')
[ "${shed1:-0}" -gt "${shed0:-0}" ] || fail "grace_shed: continuation_hold counter never moved (${shed0:-?} -> ${shed1:-?})"
[ "$(m ds4_continuation_records_live)" -eq 1 ] || fail "grace_shed: record did not stay LIVE"
resg0=$(m ds4_continuation_resolved_total)
c=$(post grace_t2 /v1/messages "$(t2_body "$T1_ID")")
code_is grace_t2 "$c" 200
resg1=$(m ds4_continuation_resolved_total)
[ "${resg1:-0}" -gt "${resg0:-0}" ] || fail "grace_shed: T2 did not resolve after the shed (${resg0:-?} -> ${resg1:-?})"
log "grace_shed PASS (unrelated serial shed 503 inside grace; T2 then resolved; shed ${shed0:-0} -> $shed1)"
drain_frontier

# postgrace_demote: after the window the same unrelated work proceeds and
# demotes; the output-only T2 refuses natively.
t1 pg_t1 "Use the list_files tool to list the files in /etc. Call the tool."
sleep 7
c=$(post pg_blocker /v1/chat/completions "$BLOCKER_BODY")
code_is pg_blocker "$c" 200
[ "$(m ds4_continuation_records_live)" -eq 0 ] || fail "postgrace_demote: record still LIVE after an unrelated serial win"
c=$(post pg_t2 /v1/messages "$(t2_body "$T1_ID")")
[ "$c" = 400 ] || [ "$c" = 409 ] || fail "postgrace_demote: T2 got HTTP $c, want 400/409"
has pg_t2 'continuation state is not available'
log "postgrace_demote PASS (post-grace serial win demoted; T2 refused $c)"

# ttl_expire: unclaimed LIVE record lazily demotes after the soft TTL.
t1 ttl_t1 "Use the list_files tool to list the files in /opt. Call the tool."
[ "$(m ds4_continuation_records_live)" -eq 1 ] || fail "ttl_expire: no LIVE record after t1"
demt0=$(m ds4_continuation_demoted_total)
sleep 46
c=$(post ttl_t2 /v1/messages "$(t2_body "$T1_ID")")
[ "$c" = 400 ] || [ "$c" = 409 ] || fail "ttl_expire: T2 got HTTP $c, want 400/409 after ttl"
has ttl_t2 'continuation state is not available'
demt1=$(m ds4_continuation_demoted_total)
[ "${demt1:-0}" -gt "${demt0:-0}" ] || fail "ttl_expire: demoted counter never moved (${demt0:-?} -> ${demt1:-?})"
[ "$(m ds4_continuation_records_live)" -eq 0 ] || fail "ttl_expire: record still LIVE past ttl"
log "ttl_expire PASS (soft TTL lazily demoted the unclaimed record; T2 refused $c)"

# pin_wins: FIFO order blocker-before-T2, but the T2 parses while queued and
# pins the record -- the blocker's admission sheds on the pin (grace long
# over), the T2 resolves.  A cont request occupies the worker to create the
# queue window; grace is waited out first so ONLY the pin can explain the
# shed.
t1 pin_t1 "Use the list_files tool to list the files in /usr. Call the tool."
sleep 7   # grace (6s) fully elapsed; record still LIVE
[ "$(m ds4_continuation_records_live)" -eq 1 ] || fail "pin_wins: record not LIVE after grace wait"
curl -s -m 240 -o "$OUT/pin_occupier.json" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":900,"temperature":0,"messages":[{"role":"user","content":"Count from 1 to 200, one number per line."}]}' &
OCC_PID=$!
sleep 1
shedp0=$(m 'ds4_requests_shed_total{reason="continuation_hold"}')
curl -s -m 240 -o "$OUT/pin_blocker.json" -w '%{http_code}' "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d "$BLOCKER_BODY" > "$OUT/pin_blocker.code" &
BLK_PID=$!
sleep 0.5
c=$(post pin_t2 /v1/messages "$(t2_body "$T1_ID")")
wait "$BLK_PID" 2>/dev/null
wait "$OCC_PID" 2>/dev/null
code_is pin_t2 "$c" 200
bcode=$(cat "$OUT/pin_blocker.code")
[ "$bcode" = 503 ] || fail "pin_wins: blocker got HTTP $bcode, want 503 (pin should shed it)"
shedp1=$(m 'ds4_requests_shed_total{reason="continuation_hold"}')
[ "${shedp1:-0}" -gt "${shedp0:-0}" ] || fail "pin_wins: continuation_hold counter never moved (${shedp0:-?} -> ${shedp1:-?})"
log "pin_wins PASS (queued T2's pin shed the FIFO-earlier blocker; T2 resolved 200)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT"
