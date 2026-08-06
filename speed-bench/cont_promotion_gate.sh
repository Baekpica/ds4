#!/bin/bash
# cont_promotion_gate.sh — v0.5.6 API Inc 3 live gate: batched-lane surface
# promotions (plan §5 Inc 3, §6.3 route engagement + semantic equivalence).
#
# Inc 3a legs (boot A = zero-config, cont on):
#   anth_cont      buffered /v1/messages rides the CONTINUOUS lane: native
#                  message schema AND route_requests{anthropic_messages,
#                  continuous} increments (the promotion, observed)
#   anth_stop_cont stop_sequences match on the CONT scan sites: stop_reason
#                  stop_sequence + the matched text, lane continuous
#   anth_warm_cont turn-2 continuation built from the SERVED t1 answer
#                  (GATE-DESIGN LAW: t1 repeats never warm-match): usage
#                  cache_read_input_tokens > 0 AND == timings prefill_cached
#                  -- the Inc 2d engine-verdict law on the promoted surface
#   anth_zero_serial   explicit max_tokens:0 (prewarm) STAYS SERIAL:
#                  need_prefill_only decision + serial lane entry
#   anth_tools_serial  tool-bearing buffered request STAYS SERIAL:
#                  need_continuation_publish decision recorded
#   oa_cont_sanity an OpenAI chat request still rides continuous
#
# Kill switch (boot B = DS4_SERVER_CONT_ANTHROPIC=0, plan §7):
#   anth_kill      the same plain request runs SERIAL (surface decision
#                  recorded, zero new cont entries) and still serves the
#                  native 200 -- semantic equivalence with boot A asserted
#                  loosely (same stop_reason, nonempty text; cont temp-0 is
#                  documented not run-to-run deterministic, so no byte diff)
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
RWORK=/tmp/cont_promotion_gate
OUT=${OUT:-/tmp/cont_promotion_gate_$$}
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
  curl -s -m 180 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1.json"))"; }
# route_requests / route_decisions cells from /metrics
lane(){ curl -s -m 10 "$BASE/metrics" | grep -F "surface=\"$1\",lane=\"$2\"" | grep -oE '[0-9]+$'; }
decision(){ curl -s -m 10 "$BASE/metrics" | grep -F "ds4_route_decisions_total{reason=\"$1\"}" | grep -oE '[0-9]+$'; }

# ============================ boot A: promotion ============================
boot

# anth_cont: the promotion itself, schema + lane observed.
ac0=$(lane anthropic_messages continuous)
c=$(post anth_cont /v1/messages '{"model":"m","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":"Name one ocean."}]}')
code_is anth_cont "$c" 200
has anth_cont '"type":"message"'
has anth_cont '"role":"assistant"'
has anth_cont '"stop_reason"'
has anth_cont '"input_tokens"'
ac1=$(lane anthropic_messages continuous)
[ "${ac1:-0}" -gt "${ac0:-0}" ] || fail "anth_cont: anthropic continuous lane never incremented (${ac0:-?} -> ${ac1:-?})"
log "anth_cont PASS (native message on the continuous lane; lane counter ${ac0:-0} -> $ac1)"

# anth_stop_cont: the cont scan sites produce native stop honesty (2c pt2
# plumbed matched_stop on cont for exactly this increment).
sc0=$(lane anthropic_messages continuous)
c=$(post anth_stop_cont /v1/messages '{"model":"m","max_tokens":128,"temperature":0,"stop_sequences":["beta"],"messages":[{"role":"user","content":"Repeat exactly: alpha beta gamma"}]}')
code_is anth_stop_cont "$c" 200
has anth_stop_cont '"stop_reason":"stop_sequence"'
has anth_stop_cont '"stop_sequence":"beta"'
sc1=$(lane anthropic_messages continuous)
[ "${sc1:-0}" -gt "${sc0:-0}" ] || fail "anth_stop_cont: stop leg did not ride continuous (${sc0:-?} -> ${sc1:-?})"
log "anth_stop_cont PASS (stop_sequence + matched text from the cont scan)"

# anth_warm_cont: engine-verdict usage (2d) on the promoted surface.
c=$(post anth_warm_t1 /v1/messages '{"model":"m","max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one planet."}]}')
code_is anth_warm_t1 "$c" 200
python3 - "$OUT/anth_warm_t1.json" > "$OUT/anth_warm_t2_req.json" <<'PY' || fail "anth_warm_cont: t2 build failed"
import json, sys
t1 = json.load(open(sys.argv[1]))
ans = "".join(b.get("text", "") for b in t1["content"] if b.get("type") == "text")
assert ans.strip(), "t1 answer empty"
msgs = [{"role": "user", "content": "Name one planet."},
        {"role": "assistant", "content": ans},
        {"role": "user", "content": "Name another."}]
print(json.dumps({"model": "m", "max_tokens": 48, "temperature": 0, "messages": msgs}))
PY
c=$(curl -s -m 180 -o "$OUT/anth_warm_cont.json" -w '%{http_code}' "$BASE/v1/messages" \
     -H 'Content-Type: application/json' -d @"$OUT/anth_warm_t2_req.json")
code_is anth_warm_cont "$c" 200
# Inc 3a frame law (found HERE, first run): the engine verdict counts
# EFFECTIVE-prompt positions -- a thinking-warm admit replays the bank's
# committed reasoning, so the verdict frame exceeds the client's terse t2
# prompt (measured: 16-token prompt vs 87 cached + 6 computed).  Usage is
# rendered in the CLIENT frame; the structural tie to engine stats is:
# input + cache_creation == timings prefill_tokens (the uncached portion is
# exactly what the engine computed fresh), with cache_read > 0 proving warm
# engagement.
ur=$(grep -oE '"cache_read_input_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
ui=$(grep -oE '"input_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
uw=$(grep -oE '"cache_creation_input_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
pf=$(grep -oE '"prefill_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
[ -n "$ur" ] && [ -n "$ui" ] && [ -n "$uw" ] && [ -n "$pf" ] \
  || fail "anth_warm_cont: usage/timings fields missing (read=${ur:-?} input=${ui:-?} creation=${uw:-?} prefill=${pf:-?})"
[ $((ui + uw)) -eq "$pf" ] \
  || fail "anth_warm_cont: uncached portion $((ui + uw)) != engine computed $pf (read=$ur)"
[ "${ur:-0}" -gt 0 ] || fail "anth_warm_cont: t2 continuation reported zero cache (no warm engagement)"
log "anth_warm_cont PASS (client frame: read=$ur input=$ui creation=$uw; uncached==computed=$pf)"

# anth_zero_serial: the prewarm contract stays serial (route_decide sends
# NEED_PREFILL_ONLY to serial before the surface stage).
pz0=$(decision need_prefill_only)
as0=$(lane anthropic_messages serial)
c=$(post anth_zero_serial /v1/messages '{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"prewarm this prompt"}]}')
code_is anth_zero_serial "$c" 200
has anth_zero_serial '"output_tokens":0'
pz1=$(decision need_prefill_only)
as1=$(lane anthropic_messages serial)
[ "${pz1:-0}" -gt "${pz0:-0}" ] || fail "anth_zero_serial: need_prefill_only decision missing (${pz0:-?} -> ${pz1:-?})"
[ "${as1:-0}" -gt "${as0:-0}" ] || fail "anth_zero_serial: serial lane entry missing (${as0:-?} -> ${as1:-?})"
log "anth_zero_serial PASS (prewarm stays serial; decision + lane recorded)"

# anth_tools_serial: tool generation keeps the serial continuation contract.
tp0=$(decision need_continuation_publish)
c=$(post anth_tools_serial /v1/messages '{"model":"m","max_tokens":128,"temperature":0,"tools":[{"name":"get_time","description":"Get the current time.","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"What time is it?"}]}')
code_is anth_tools_serial "$c" 200
tp1=$(decision need_continuation_publish)
[ "${tp1:-0}" -gt "${tp0:-0}" ] || fail "anth_tools_serial: need_continuation_publish decision missing (${tp0:-?} -> ${tp1:-?})"
log "anth_tools_serial PASS (tool gen stays serial with the continuation-publish reason)"

# oa_cont_sanity: the incumbent surface still rides continuous.
oc0=$(lane openai_chat continuous)
c=$(post oa_cont_sanity /v1/chat/completions '{"max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"Say ok."}]}')
code_is oa_cont_sanity "$c" 200
oc1=$(lane openai_chat continuous)
[ "${oc1:-0}" -gt "${oc0:-0}" ] || fail "oa_cont_sanity: openai chat lost the continuous lane (${oc0:-?} -> ${oc1:-?})"
log "oa_cont_sanity PASS"

# ===================== boot B: the §7 kill switch ==========================
BOOT_ENV="DS4_SERVER_CONT_ANTHROPIC=0" boot
ks0=$(decision surface)
kc0=$(lane anthropic_messages continuous)
c=$(post anth_kill /v1/messages '{"model":"m","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":"Name one ocean."}]}')
code_is anth_kill "$c" 200
has anth_kill '"type":"message"'
has anth_kill '"stop_reason"'
ks1=$(decision surface)
kc1=$(lane anthropic_messages continuous)
[ "${ks1:-0}" -gt "${ks0:-0}" ] || fail "anth_kill: surface decision missing under the kill switch (${ks0:-?} -> ${ks1:-?})"
[ "${kc1:-0}" -eq "${kc0:-0}" ] || fail "anth_kill: request rode continuous DESPITE the kill switch (${kc0:-?} -> ${kc1:-?})"
# Loose semantic equivalence vs boot A (documented cont temp-0 jitter: no
# byte compare; the lane must not change the protocol result class).
sr_a=$(grep -oE '"stop_reason":"[a-z_]+"' "$OUT/anth_cont.json" | head -1)
sr_b=$(grep -oE '"stop_reason":"[a-z_]+"' "$OUT/anth_kill.json" | head -1)
[ -n "$sr_a" ] && [ "$sr_a" = "$sr_b" ] || fail "anth_kill: stop_reason diverged across lanes ($sr_a vs $sr_b)"
grep -q '"type":"text"' "$OUT/anth_kill.json" || fail "anth_kill: no text block"
grep -q '"type":"text"' "$OUT/anth_cont.json" || fail "anth_kill: boot A response had no text block"
log "anth_kill PASS (kill switch = serial with the surface reason; same result class)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
