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
  ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./ds4-server -c $CTX --port $PORT \
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

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
