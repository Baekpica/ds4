#!/bin/bash
# launch_defaults_gate.sh — functional gate for the in-engine launch defaults.
#
# Verifies that on a standard install layout the server resolves models and
# arms speculation by itself, that the opt-outs really opt out, and that the
# strict preset boots the same stack:
#
#   leg zero_config   ./ds4-server -c $CTX --port $PORT      (no -m, no env)
#       asserts: one launch-defaults boot line naming model+mtp+dspark and
#       arming DS4_CONT_MTP_MODE=2 + DS4_CONT_DSPARK=1; MTP + DSpark loaded;
#       aligned fast-path artifacts present; a completion succeeds; the
#       request drove speculative drafts (ds4_spec_drafts_total > 0).
#   leg no_spec       ./ds4-server --no-spec -c $CTX --port $PORT
#       asserts: NO mtp/dspark load lines, NO spec arming in the boot line;
#       a completion still succeeds on the plain continuous path.
#   leg preset_spark  ./ds4-server --preset spark -c $CTX --port $PORT
#       asserts: same full-stack tells as zero_config.
#
# Runs FROM the Mac over SSH like the other gates. NOTE: each boot kills any
# running ds4-server on $R. End state: ds4-server killed, box left free.
#
# Env overrides: R (sync-192_168_88_33) BINDIR (/home/ent/code/ds4-phase0)
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
set -uo pipefail

R=${R:-sync-192_168_88_33}
BINDIR=${BINDIR:-/home/ent/code/ds4-phase0}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/launch_gate
OUT=${OUT:-/tmp/launch_defaults_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; exit 1; }

tunnel_up(){
  curl -s -m 5 "http://127.0.0.1:$TUNNEL_PORT/v1/models" >/dev/null 2>&1 && return 0
  ssh -f -N -L "$TUNNEL_PORT:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "http://127.0.0.1:$TUNNEL_PORT/v1/models" >/dev/null 2>&1
}

# boot <leg> <extra server args...>
boot(){
  local leg=$1; shift
  SRV=$RWORK/srv_${leg}.log
  log "boot($leg): killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  sleep 3
  ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./ds4-server -c $CTX --port $PORT $* \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3   # a too-eager liveness poll can false-negative mid-exec
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "boot($leg) BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot($leg) timeout"
  done
  tunnel_up || fail "boot($leg): tunnel :$TUNNEL_PORT unreachable"
  ssh "$R" "cat $SRV" > "$OUT/srv_${leg}.log"
  log "boot($leg): up"
}

srv_has(){ grep -q "$2" "$OUT/srv_$1.log"; }

completion(){
  local leg=$1
  curl -s -m 120 "http://127.0.0.1:$TUNNEL_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Reply with exactly: launch gate ok"}],"max_tokens":48,"temperature":0}' \
    > "$OUT/resp_${leg}.json"
  grep -q '"finish_reason"' "$OUT/resp_${leg}.json" || fail "$leg: completion malformed: $(head -c 300 "$OUT/resp_${leg}.json")"
}

metric(){ curl -s -m 10 "http://127.0.0.1:$TUNNEL_PORT/metrics" | grep -oE "^$1 [0-9]+" | awk '{print $2}'; }

# ---- leg 1: zero-config ---------------------------------------------------
# v0.2.4 MTP-droppable default: with a drafter beside the base model the
# launch defaults arm DSpark-only spec and DROP the MTP head (~3.55 GiB;
# teb fast MTP-less is counter-identical and slightly faster, 240K deep
# within noise). --mtp / --preset spark still load it (leg 3 covers that).
boot zero_config
srv_has zero_config 'launch defaults:.*model=.*mtp=dropped.*dspark=.*DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1' \
  || fail "zero_config: launch-defaults boot line missing/incomplete (want mtp=dropped)"
srv_has zero_config 'MTP support model loaded'  && fail "zero_config: MTP loaded despite armed drafter (MTP-droppable default)"
srv_has zero_config 'DSpark drafter loaded'     || fail "zero_config: drafter not loaded"
[ "$(ssh "$R" "grep -c aligned $RWORK/srv_zero_config.log")" -gt 0 ] \
  || fail "zero_config: no aligned fast-path artifacts (perf-cliff tell)"
completion zero_config
d=$(metric ds4_spec_drafts_total); [ -n "$d" ] && [ "$d" -gt 0 ] \
  || fail "zero_config: spec not engaged (ds4_spec_drafts_total=${d:-absent})"
log "zero_config PASS (drafts=$d)"

# ---- leg 2: --no-spec ------------------------------------------------------
boot no_spec --no-spec
srv_has no_spec 'MTP support model loaded' && fail "no_spec: MTP loaded despite --no-spec"
srv_has no_spec 'DSpark drafter loaded'    && fail "no_spec: drafter loaded despite --no-spec"
srv_has no_spec 'DS4_CONT_MTP_MODE=2'      && fail "no_spec: spec armed despite --no-spec"
completion no_spec
log "no_spec PASS"

# ---- leg 3: --preset spark -------------------------------------------------
boot preset_spark --preset spark
srv_has preset_spark 'MTP support model loaded' || fail "preset_spark: MTP not loaded"
srv_has preset_spark 'DSpark drafter loaded'    || fail "preset_spark: drafter not loaded"
completion preset_spark
log "preset_spark PASS"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
