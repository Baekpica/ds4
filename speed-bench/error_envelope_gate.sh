#!/bin/bash
# error_envelope_gate.sh — v0.5.6 API Inc 2b live gate: endpoint-native error
# envelopes + the per-surface decode-budget matrix (plan §5 Inc 2 / §6.2;
# surface-matrix defect 3).
#
# ONE default zero-config boot serves every leg (all legs are HTTP-level):
#
#   envelope legs
#     anth_parse   POST /v1/messages garbage body -> HTTP 400 + the documented
#                  {"type":"error","error":{"type":"invalid_request_error",...
#                  envelope (NO OpenAI shape on the Anthropic surface)
#     anth_neg     /v1/messages max_tokens:-1 -> 400 native + field message
#     oa_neg       /v1/chat/completions max_tokens:-1 -> 400 OpenAI envelope
#     oa_neg2      chat max_completion_tokens:-2 -> the message names the
#                  field THE CLIENT SENT
#     comp_neg     /v1/completions max_tokens:-1 -> 400
#     resp_neg     /v1/responses max_output_tokens:-3 -> 400 names the field
#     not_found    GET /nope -> 404 OpenAI envelope (historical owner)
#
#   client cancel (Inc 2c)
#     cancel       a live stream whose client disconnects mid-decode (curl
#                  -m 4 on a 3000-token stream) must count
#                  requests_total{outcome="canceled"} -- never completed
#                  (the pre-2c behavior counted every mid-flight cancel as
#                  a completed request).
#
#   failure injection (Inc 2c)
#     strand       reboot with DS4_GATE_CONT_FAIL_AFTER_STEPS=8 (engine
#                  gate-teeth: strands every live bank mid-decode exactly as
#                  a real engine failure).  A live chat stream must get real
#                  deltas, then the NATIVE stream-error event carrying the
#                  forced-failure message -- and NO fabricated
#                  finish="length" / [DONE] (the pre-2c dishonesty, plan
#                  §2.1 / matrix defect 2).  Counters: the request counts
#                  FAILED, never completed; cont_batch_failures increments.
#
#   budget matrix (omitted is every other gate's default; negative above)
#     anth_zero    anthropic max_tokens:0 (prewarm contract) -> 200 +
#                  "output_tokens":0
#     resp_zero    responses max_output_tokens:0 -> 200 (serial zero-decode)
#     oa_zero      chat max_tokens:0 -> 200, completion_tokens <= 1 (batched
#                  lanes floor the seed token -- documented at
#                  request_decode_budget until Inc 3 prefill-only routing;
#                  the gate RECORDS the observed value)
#     oa_one       chat max_tokens:1 -> 200, completion_tokens == 1, finish
#                  "length"
#     oa_clamped   chat max_tokens:10000000 -> 200, finish "stop" (a huge
#                  budget must serve; EOS ends it long before the bound)
#
# Runs FROM the Mac over SSH like the other gates.  NOTE: the boot kills any
# running ds4-server on $R.  End state: ds4-server killed, box left free.
#
# Env overrides: R (sync-192_168_88_33) BINDIR (/home/ent/code/ds4-phase0)
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
set -uo pipefail

R=${R:-sync-192_168_88_33}
BINDIR=${BINDIR:-/home/ent/code/ds4-phase0}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/error_envelope_gate
OUT=${OUT:-/tmp/error_envelope_gate_$$}
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

# post <leg> <path> <json>  -> $OUT/<leg>.json, echoes HTTP code
post(){
  curl -s -m 180 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
lacks(){ grep -q "$2" "$OUT/$1.json" && fail "$1: unexpected [$2] in $(head -c 300 "$OUT/$1.json")"; true; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1.json"))"; }
metric(){ curl -s -m 10 "$BASE/metrics" | grep -oE "^$1 [0-9]+" | awk '{print $2}'; }
lmetric(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$'; }

boot

# ---- envelope legs ---------------------------------------------------------
c=$(post anth_parse /v1/messages '{not json')
code_is anth_parse "$c" 400
has  anth_parse '"type":"error","error":{"type":"invalid_request_error"'
lacks anth_parse '"error":{"message":'
log "anth_parse PASS (native envelope on parse failure)"

c=$(post anth_neg /v1/messages '{"model":"m","max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}')
code_is anth_neg "$c" 400
has anth_neg '"type":"error","error":{"type":"invalid_request_error"'
has anth_neg 'max_tokens must be >= 0'
log "anth_neg PASS"

c=$(post oa_neg /v1/chat/completions '{"max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_neg "$c" 400
has oa_neg '"error":{"message":"max_tokens must be >= 0"'
log "oa_neg PASS"

c=$(post oa_neg2 /v1/chat/completions '{"max_completion_tokens":-2,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_neg2 "$c" 400
has oa_neg2 'max_completion_tokens must be >= 0'
log "oa_neg2 PASS (field name echoed)"

c=$(post comp_neg /v1/completions '{"max_tokens":-1,"prompt":"hi"}')
code_is comp_neg "$c" 400
has comp_neg 'max_tokens must be >= 0'
log "comp_neg PASS"

c=$(post resp_neg /v1/responses '{"max_output_tokens":-3,"input":"hi"}')
code_is resp_neg "$c" 400
has resp_neg 'max_output_tokens must be >= 0'
log "resp_neg PASS (field name echoed)"

c=$(curl -s -m 20 -o "$OUT/not_found.json" -w '%{http_code}' "$BASE/nope")
code_is not_found "$c" 404
has not_found '"error":{"message":'
log "not_found PASS (historical-owner envelope)"

# ---- budget matrix ----------------------------------------------------------
c=$(post anth_zero /v1/messages '{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"prewarm this prompt"}]}')
code_is anth_zero "$c" 200
has anth_zero '"output_tokens":0'
log "anth_zero PASS (prewarm contract, zero decode)"

c=$(post resp_zero /v1/responses '{"max_output_tokens":0,"input":"prewarm this too"}')
code_is resp_zero "$c" 200
log "resp_zero PASS"

c=$(post oa_zero /v1/chat/completions '{"max_tokens":0,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_zero "$c" 200
ct=$(grep -oE '"completion_tokens":[0-9]+' "$OUT/oa_zero.json" | head -1 | cut -d: -f2)
[ -n "$ct" ] && [ "$ct" -le 1 ] || fail "oa_zero: completion_tokens=${ct:-absent}, want <= 1"
log "oa_zero PASS (completion_tokens=$ct; batched seed floor documented)"

c=$(post oa_one /v1/chat/completions '{"max_tokens":1,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_one "$c" 200
has oa_one '"completion_tokens":1'
has oa_one '"finish_reason":"length"'
log "oa_one PASS"

c=$(post oa_clamped /v1/chat/completions '{"max_tokens":10000000,"temperature":0,"messages":[{"role":"user","content":"Reply with exactly: envelope gate ok"}]}')
code_is oa_clamped "$c" 200
has oa_clamped '"finish_reason":"stop"'
log "oa_clamped PASS (huge budget served, EOS ended it)"

# ---- client cancel (Inc 2c): a mid-stream disconnect counts CANCELED -------
can0=$(lmetric 'ds4_requests_total{outcome="canceled"}')
ccomp0=$(lmetric 'ds4_requests_total{outcome="completed"}')
curl -s -m 4 -o "$OUT/cancel.sse" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"temperature":0,"max_tokens":3000,"messages":[{"role":"user","content":"Write a very long story about the sea."}]}' || true
n=0
while :; do
  can1=$(lmetric 'ds4_requests_total{outcome="canceled"}')
  [ "${can1:-0}" -gt "${can0:-0}" ] && break
  n=$((n+1)); [ $n -ge 24 ] && fail "cancel: canceled counter never incremented (${can0:-?} -> ${can1:-?})"
  sleep 5
done
ccomp1=$(lmetric 'ds4_requests_total{outcome="completed"}')
[ "${ccomp1:-0}" -eq "${ccomp0:-0}" ] || fail "cancel: canceled stream counted COMPLETED ($ccomp0 -> $ccomp1)"
log "cancel PASS (mid-stream disconnect counted canceled, not completed)"

# ---- failure injection: engine-stranded live stream (Inc 2c) ---------------
BOOT_ENV="DS4_GATE_CONT_FAIL_AFTER_STEPS=8" boot
comp0=$(lmetric 'ds4_requests_total{outcome="completed"}')
fail0=$(lmetric 'ds4_requests_total{outcome="failed"}')
c=$(curl -s -m 90 -o "$OUT/strand.sse" -w '%{http_code}' "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"temperature":0,"max_tokens":400,"messages":[{"role":"user","content":"Count slowly from one to fifty in words."}]}')
code_is strand "$c" 200   # headers + deltas went out before the failure
grep -qE '"delta":\{"(content|reasoning_content)":' "$OUT/strand.sse" \
  || fail "strand: no deltas before the failure"
grep -q 'event: error' "$OUT/strand.sse" || fail "strand: native stream-error event missing"
grep -q '"type":"server_error"' "$OUT/strand.sse" || fail "strand: error payload missing"
grep -q 'gate-forced continuous failure' "$OUT/strand.sse" \
  || fail "strand: forced-failure cause not surfaced"
grep -q '"finish_reason":"length"' "$OUT/strand.sse" \
  && fail "strand: fabricated length finish still present"
grep -q 'data: \[DONE\]' "$OUT/strand.sse" \
  && fail "strand: fabricated DONE after failure"
comp1=$(lmetric 'ds4_requests_total{outcome="completed"}')
fail1=$(lmetric 'ds4_requests_total{outcome="failed"}')
cbf=$(metric ds4_cont_batch_failures_total)
[ "${fail1:-0}" -gt "${fail0:-0}" ] || fail "strand: requests failed counter did not increment ($fail0 -> ${fail1:-?})"
[ "${comp1:-0}" -eq "${comp0:-0}" ] || fail "strand: stranded stream counted COMPLETED ($comp0 -> $comp1)"
[ -n "$cbf" ] && [ "$cbf" -ge 1 ] || fail "strand: cont_batch_failures=${cbf:-absent}"
log "strand PASS (deltas -> native error event, counted failed not completed, cont_batch_failures=$cbf)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
