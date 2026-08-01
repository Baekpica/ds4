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
# v0.5.1 inc4 — MTP accept guard legs.  The ggufs carry NO generation
# metadata, so support-model health is judged by MEASUREMENT (spec decode is
# lossless; a bad module costs pure speed).  MEASURED VERDICT (2026-08-01,
# .33, 400-tok essays, warm): the feared cross-generation pairing is BENIGN
# -- 0731 base + legacy MTP drafts at 51.9% accept / 22.3 tok/s vs plain
# 21.9 and matched-pairing 24.4 (66.1%): D=1 break-even sits at ~50% accept,
# same-lineage checkpoints transfer.  The guard is therefore a FLOOR against
# genuinely broken support models (foreign/corrupt files drafting ~random),
# with the trip threshold far below any same-lineage pairing:
#   leg cross_gen     0731 base + legacy MTP, --no-dspark (the field shape).
#       asserts: 'accept guard armed' at load; guard does NOT trip (benign
#       pairing; a future refresh that makes this harmful drops accept and
#       flips this leg -> re-derive); completion healthy.
#   leg guard_trip    same boot + DS4_MTP_FORCE_DRAFT_TOK=77 (gate hook:
#       force every draft wrong through the REAL serving path).
#       asserts: guard TRIPS ('Speculative MTP decode DISABLED'); the
#       completion still finishes; a second completion drives ZERO new
#       drafts (spec stays off for the process).
#   leg right_pairing legacy base + legacy MTP, --no-dspark (control).
#       asserts: NO trip line; drafts flow with healthy accept.
#   leg guard_off     forced-wrong drafts + DS4_MTP_ACCEPT_GUARD=0 (escape).
#       asserts: NO trip line; drafts keep flowing past the guard window.
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

# Boot-time memory guard (standing law; see bank_persist_gate.sh): back-to-
# back boots race the dying server's reclaim.
wait_mem(){ # $1=min MemAvailable GiB
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}

# boot <leg> <extra server args...>  (BOOT_ENV: extra env assignments)
boot(){
  local leg=$1; shift
  SRV=$RWORK/srv_${leg}.log
  log "boot($leg): killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server -c $CTX --port $PORT $* \
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

# ---- v0.5.1 inc4: MTP accept guard legs -----------------------------------
GDIR=${GDIR:-/home/ent/gguf}
BASE_0731=$GDIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
BASE_LEGACY=$GDIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP_LEGACY=$GDIR/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf

# The guard window is 256 DRAFTS at draft depth 1 = 256 decode STEPS.  At
# accept ~0 steps == tokens (400 suffices); at healthy accept ~66% each step
# emits ~1.67 tokens, so proving the guard EVALUATED (and did not fire)
# needs >= 256/0.6 ~ 430+ tokens -- healthy legs pass 520.
long_completion(){ # $1=leg $2=max_tokens (default 400)
  local leg=$1 mt=${2:-400}
  curl -s -m 300 "http://127.0.0.1:$TUNNEL_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Write a detailed 600-word essay about the history of container shipping. Do not stop early."}],"max_tokens":'"$mt"',"temperature":0}' \
    > "$OUT/resp_${leg}.json"
  grep -q '"finish_reason"' "$OUT/resp_${leg}.json" || fail "$leg: long completion malformed: $(head -c 300 "$OUT/resp_${leg}.json")"
}

# ---- leg 4: cross-generation pairing is benign (measured) -----------------
boot cross_gen -m "$BASE_0731" --mtp "$MTP_LEGACY" --no-dspark
srv_has cross_gen 'MTP support model loaded.*accept guard armed' \
  || fail "cross_gen: guard-armed load line missing"
long_completion cross_gen 520
ssh "$R" "cat $SRV" > "$OUT/srv_cross_gen.log"
srv_has cross_gen 'Speculative MTP decode DISABLED' \
  && fail "cross_gen: guard tripped on the measured-benign cross-generation pairing (accept regressed? re-derive)"
d=$(metric ds4_spec_drafts_total); h=$(metric ds4_spec_hits_total)
[ -n "$d" ] && [ "$d" -ge 256 ] || fail "cross_gen: guard window not reached (drafts=${d:-absent})"
log "cross_gen PASS (benign: drafts=$d hits=$h, no trip)"

# ---- leg 4b: forced-wrong drafts trip the guard ---------------------------
BOOT_ENV="DS4_MTP_FORCE_DRAFT_TOK=77" boot guard_trip -m "$BASE_0731" --mtp "$MTP_LEGACY" --no-dspark
long_completion guard_trip
ssh "$R" "cat $SRV" > "$OUT/srv_guard_trip.log"
srv_has guard_trip 'Speculative MTP decode DISABLED' \
  || fail "guard_trip: guard never tripped on forced-wrong drafts (accept line: $(grep -ao 'MTP accept [0-9.]*%' "$OUT/srv_guard_trip.log" | tail -1))"
d1=$(metric ds4_spec_drafts_total)
completion guard_trip_after
d2=$(metric ds4_spec_drafts_total)
[ -n "$d1" ] && [ -n "$d2" ] && [ "$d2" -eq "$d1" ] \
  || fail "guard_trip: drafts still flowing after trip (before=$d1 after=$d2)"
log "guard_trip PASS (guard tripped, post-trip drafts frozen at $d1)"

# ---- leg 5: right pairing must NOT trip (control) -------------------------
boot right_pairing -m "$BASE_LEGACY" --mtp "$MTP_LEGACY" --no-dspark
srv_has right_pairing 'MTP support model loaded.*accept guard armed' \
  || fail "right_pairing: guard-armed load line missing"
long_completion right_pairing 520
ssh "$R" "cat $SRV" > "$OUT/srv_right_pairing.log"
srv_has right_pairing 'Speculative MTP decode DISABLED' \
  && fail "right_pairing: guard FALSE-FIRED on a matched pairing"
d=$(metric ds4_spec_drafts_total); h=$(metric ds4_spec_hits_total)
[ -n "$d" ] && [ "$d" -ge 256 ] || fail "right_pairing: guard window not reached (drafts=${d:-absent})"
python3 -c "import sys; sys.exit(0 if int('$h') * 100 >= int('$d') * 30 else 1)" \
  || fail "right_pairing: accept unhealthily low (hits=$h drafts=$d)"
log "right_pairing PASS (drafts=$d hits=$h, no false fire)"

# ---- leg 6: kill switch disarms the guard ---------------------------------
BOOT_ENV="DS4_MTP_ACCEPT_GUARD=0 DS4_MTP_FORCE_DRAFT_TOK=77" \
  boot guard_off -m "$BASE_0731" --mtp "$MTP_LEGACY" --no-dspark
srv_has guard_off 'accept guard armed' \
  && fail "guard_off: guard armed despite DS4_MTP_ACCEPT_GUARD=0"
long_completion guard_off
ssh "$R" "cat $SRV" > "$OUT/srv_guard_off.log"
srv_has guard_off 'Speculative MTP decode DISABLED' \
  && fail "guard_off: guard tripped despite kill switch"
d=$(metric ds4_spec_drafts_total)
[ -n "$d" ] && [ "$d" -ge 256 ] \
  || fail "guard_off: drafting stopped without the guard (drafts=${d:-absent})"
log "guard_off PASS (drafts=$d kept flowing on forced-wrong drafts, no trip)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
