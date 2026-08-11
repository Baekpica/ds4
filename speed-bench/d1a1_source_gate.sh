#!/bin/bash
# d1a1_source_gate.sh — memgov D1a-1 close gate (spec: local/docs/v057/
# d1a_scoping_2026-08-11.md §3 "D1a-1").  Runs the D1a-1-SPECIFIC legs;
# the increment close ALSO requires the standing D0 batteries green on the
# same tree: d0a_census_gate.sh (oracles + reconciliation + ABBA vs the
# cebaff7 control), d0b_shadow_gate.sh, serial_reclaim_gate.sh (INJECT).
#
# WHAT IT PROVES
#   (s) SOURCE LEDGER: a stock zero-config boot (base + drafter on this
#       box -- launch defaults DROP mtp when the drafter is armed) prints
#       one residency line per source with the expected roles, the
#       per-source weight ledger reconciles EXACTLY against the 2-D census
#       cells (the boot report recomputes Σ rows == cell and counts a
#       census fault on divergence), no UNATTRIBUTED weight bytes exist,
#       and census faults are 0 at settle.  A decode wave then re-checks
#       faults: post-boot lazy weight traffic (q8 caches, cold pins) must
#       attribute cleanly too.
#   (s2) AUXILIARY ROLE: a --no-dspark boot re-arms the default MTP head;
#       assert base [primary] + mtp [auxiliary] lines + reconcile.  With
#       (s) this covers all three roles live in one gate.
#       Both boot legs also assert the D1a-4 funnel porcelain: one
#       "ds4: range plan" line per source with planned == total for the
#       model/derived/q8 families and refused == 0, and no
#       RANGE-PUBLISH REFUSED line anywhere in the boot.
#   (p) LOOKUP MICRO-PARITY: the same boot under
#       DS4_CUDA_RANGE_LOOKUP_PARITY=1 recomputes EVERY token-path range
#       resolve against the legacy flat-keyed algorithm (a shadow index
#       maintained with the old last-write-wins semantics).  After real
#       decode traffic + graceful shutdown: checked > 0, divergent == 0,
#       and the decode output is a terminal message.
#
# Usage: bash speed-bench/d1a1_source_gate.sh   (from the repo root)
set -uo pipefail
HOST=${HOST:-sync-192_168_88_33}
TEST_TREE='~/code/ds4-phase0'
CTX=${CTX:-32768}
PORT=8000
TS=$(date +%s); NONCE="d1a1src-$TS-$$"
LOCK=/tmp/ds4_box_lock
R() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }
SPID=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
cleanup() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    R "rm -rf $LOCK" 2>/dev/null
}
die() { log "FAIL: $*"; cleanup; exit 1; }

R "ps -eo comm | grep -q '^ds4-server' && echo BUSY || echo IDLE" | grep -q IDLE \
    || die "a ds4-server is already running on $HOST"
R "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held: $(R cat $LOCK/nonce 2>/dev/null || echo unknown)"
trap cleanup EXIT
log "lock acquired ($NONCE)"

boot_tree() { # $1 log, $2 extra env prefix (may be empty), $3 extra flags
    ssh -o ConnectTimeout=10 "$HOST" \
        "cd $TEST_TREE && setsid nohup env $2 ./ds4-server -c $CTX ${3:-} > $1 2>&1 < /dev/null & exit 0" &
    local lp=$!; sleep 5; kill $lp 2>/dev/null || true
    for i in $(seq 240); do
        R "grep -q 'listening on http' $1" && { SPID=$(R "pgrep -x ds4-server | head -1"); return 0; }
        R "ps -eo comm | grep -q '^ds4-server'" || { R "tail -5 $1"; return 1; }
        sleep 2
    done
    return 1
}
met() { R "curl -s -m 20 localhost:$PORT/metrics" | awk -v k="$1" '$1==k{print $2}'; }
decode_one() { # $1 out json, $2 tag
    R "curl -s -m 600 -X POST localhost:$PORT/v1/messages -H 'content-type: application/json' \
        -d '{\"model\":\"ds4\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Gate $2 $TS: name three rivers and one ledger.\"}]}' -o $1"
    R "grep -q '\"type\":\"message\"' $1"
}

log "(s) source ledger: stock boot (base+drafter) + reconcile + decode"
BOOTLOG=/tmp/d1a1_boot_s.log
boot_tree "$BOOTLOG" "" "" || die "leg (s) boot failed"
R "grep -q 'ds4: model source base \[primary\]' $BOOTLOG" \
    || die "no base [primary] source line"
R "grep -q 'ds4: model source drafter \[drafter\]' $BOOTLOG" \
    || die "no drafter [drafter] source line"
R "grep -q 'ds4: model-source ledger reconciled:' $BOOTLOG" \
    || die "no ledger reconcile line"
R "grep -q 'source ledger mismatch' $BOOTLOG" && die "ledger mismatch at boot"
R "grep -q 'model source UNATTRIBUTED' $BOOTLOG" && die "unattributed weight bytes at boot"
# D1a-2 (tripwire retired in D1a-4b): the catalog lines ARE the positive
# engagement signal; the classifier units pin the name relation.
R "grep -q 'ds4: model catalog base:' $BOOTLOG" || die "no base catalog line"
R "grep -q 'ds4: model catalog drafter:' $BOOTLOG" || die "no drafter catalog line"
# D1a-3: unit tables compile + verify clean on every source.
R "grep -q 'ds4: model units base: .* (verify=0)' $BOOTLOG" || die "no clean base unit table"
R "grep -q 'ds4: model units drafter: .* (verify=0)' $BOOTLOG" || die "no clean drafter unit table"
R "grep -q 'UNIT-TABLE FAULT' $BOOTLOG" && die "unit-table fault at boot"
# D1a-4: publication funnel -- every live range plan-known on a full boot
# (model/derived/q8 planned == total per source), zero refusals.
R "grep -q 'ds4: range plan base:' $BOOTLOG" || die "no base range-plan line"
R "grep -q 'ds4: range plan drafter:' $BOOTLOG" || die "no drafter range-plan line"
R "grep -q 'RANGE-PUBLISH REFUSED' $BOOTLOG" && die "range publication refused at boot"
BAD=$(R "grep 'ds4: range plan ' $BOOTLOG" \
    | sed -E 's/.*model ([0-9]+)\/([0-9]+) planned, derived ([0-9]+)\/([0-9]+), q8 ([0-9]+)\/([0-9]+), refused ([0-9]+).*/\1 \2 \3 \4 \5 \6 \7/' \
    | awk '$1!=$2 || $3!=$4 || $5!=$6 || $7!=0 {print}')
[ -z "$BAD" ] || die "range plan incomplete/refused: $BAD"
CF0=$(met ds4_memory_census_faults_total)
[ "$CF0" = "0" ] || die "census faults=$CF0 at boot settle"
decode_one /tmp/d1a1_s_dec.json s || die "leg (s) decode failed"
sleep 3
CF1=$(met ds4_memory_census_faults_total)
[ "$CF1" = "0" ] || die "census faults=$CF1 after decode (lazy weight attribution)"
SRCLINES=$(R "grep -c 'ds4: model source ' $BOOTLOG")
log "(s) PASS (sources=$SRCLINES reconciled, faults 0 at boot + after decode)"
R "kill $SPID" 2>/dev/null; SPID=""
sleep 8
R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill -9' 2>/dev/null
sleep 5

log "(s2) auxiliary role: --no-dspark boot re-arms default MTP + reconcile"
BOOTLOG=/tmp/d1a1_boot_s2.log
boot_tree "$BOOTLOG" "" "--no-dspark" || die "leg (s2) boot failed"
R "grep -q 'ds4: model source base \[primary\]' $BOOTLOG" \
    || die "(s2) no base [primary] source line"
R "grep -q 'ds4: model source mtp \[auxiliary\]' $BOOTLOG" \
    || die "(s2) no mtp [auxiliary] source line"
R "grep -q 'ds4: model-source ledger reconciled:' $BOOTLOG" \
    || die "(s2) no ledger reconcile line"
R "grep -q 'source ledger mismatch' $BOOTLOG" && die "(s2) ledger mismatch"
R "grep -q 'model source UNATTRIBUTED' $BOOTLOG" && die "(s2) unattributed weight bytes"
# D1a-2 (tripwire retired in D1a-4b): catalog engagement on the mtp shape.
R "grep -q 'ds4: model catalog mtp:' $BOOTLOG" || die "(s2) no mtp catalog line"
# D1a-3: mtp unit table clean.
R "grep -q 'ds4: model units mtp: .* (verify=0)' $BOOTLOG" || die "(s2) no clean mtp unit table"
R "grep -q 'UNIT-TABLE FAULT' $BOOTLOG" && die "(s2) unit-table fault"
# D1a-4: funnel plan-known on the base+mtp shape too.
R "grep -q 'ds4: range plan mtp:' $BOOTLOG" || die "(s2) no mtp range-plan line"
R "grep -q 'RANGE-PUBLISH REFUSED' $BOOTLOG" && die "(s2) range publication refused"
BAD=$(R "grep 'ds4: range plan ' $BOOTLOG" \
    | sed -E 's/.*model ([0-9]+)\/([0-9]+) planned, derived ([0-9]+)\/([0-9]+), q8 ([0-9]+)\/([0-9]+), refused ([0-9]+).*/\1 \2 \3 \4 \5 \6 \7/' \
    | awk '$1!=$2 || $3!=$4 || $5!=$6 || $7!=0 {print}')
[ -z "$BAD" ] || die "(s2) range plan incomplete/refused: $BAD"
CF2=$(met ds4_memory_census_faults_total)
[ "$CF2" = "0" ] || die "(s2) census faults=$CF2"
log "(s2) PASS (base+mtp reconciled, faults 0)"
R "kill $SPID" 2>/dev/null; SPID=""
sleep 8
R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill -9' 2>/dev/null
sleep 5

log "(p) lookup micro-parity: parity boot + decode + graceful shutdown"
BOOTLOG=/tmp/d1a1_boot_p.log
boot_tree "$BOOTLOG" "DS4_CUDA_RANGE_LOOKUP_PARITY=1" "" || die "leg (p) boot failed"
decode_one /tmp/d1a1_p_dec.json p || die "leg (p) decode failed"
R "grep -q 'RANGE-PARITY DIVERGENT' $BOOTLOG" && die "range parity divergence during run"
R "kill $SPID" 2>/dev/null
for i in $(seq 60); do
    R "grep -q 'ds4: range-lookup parity:' $BOOTLOG" && break
    R "ps -eo comm | grep -q '^ds4-server'" || break
    sleep 2
done
SPID=""
PLINE=$(R "grep 'ds4: range-lookup parity:' $BOOTLOG | tail -1")
[ -n "$PLINE" ] || die "no parity summary after graceful shutdown"
CHECKED=$(echo "$PLINE" | sed -E 's/.*checked=([0-9]+).*/\1/')
DIVERGENT=$(echo "$PLINE" | sed -E 's/.*divergent=([0-9]+).*/\1/')
[ -n "$CHECKED" ] && [ "$CHECKED" -gt 0 ] || die "parity checked=$CHECKED (no coverage)"
[ "$DIVERGENT" = "0" ] || die "parity divergent=$DIVERGENT"
log "(p) PASS ($PLINE)"

log "D1A1 SOURCE GATE: ALL LEGS PASS"
cleanup
trap - EXIT
