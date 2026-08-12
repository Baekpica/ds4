#!/bin/bash
# d1a_residency_gate.sh — memgov D1a STAGE CLOSE gate (stage spec:
# local/docs/v057/d1a_scoping_2026-08-11.md §4; authority plan §12).
# The equivalence matrix: every supported boot shape must produce a
# reconciled per-source ledger, clean semantic catalogs + unit tables,
# and a fully plan-known range registry (D1a-4 funnel), with boot
# decision families byte-identical (normalized) to the pre-D1a control
# tree and golden decode vectors in band per residency mode.
#
# LEGS
#   (l1) base-only          --no-spec
#   (l2) base+mtp           --no-dspark
#   (l3) base+mtp+drafter   --mtp <gguf>   (launch defaults drop mtp when
#                                           the drafter arms; the flag
#                                           overrides — all 3 roles live)
#   (l4) SLICE boot         --role coordinator --listen 127.0.0.1 9411
#                           --layers 20:40 --no-spec
#        (coordinator, not worker: a worker validates --coordinator HOST
#        PORT before anything runs — l4 first-run finding.  The
#        coordinator opens its own engine slice [20:40]+output with no
#        peer required, which is exactly the boot shape this leg gates.)
#        Boot-shape only (no workers -> no serving): catalog + unit
#        tables + reconcile + zero faults/refusals asserted from the boot
#        log.  planned==total is NOT asserted here: the pre-cache walk is
#        slice-blind until D1b adopts the unit table — the leg RECORDS
#        the plan-coverage numbers as the slice-blindness measurement.
#   (l5) raw                DS4_CUDA_NO_DERIVED_WEIGHTS=1
#                           DS4_CUDA_NO_HBM_CACHE=1 (+ pinned vmm budget
#                           for the control pair, d0a nohbm law)
#        + golden vectors under the same env.
#   (l6) self-load          stock zero-config; asserts the BUILT artifact
#        banner + golden vectors under stock env.
#   (l7) manifest           .33-local ds4_weight_server --scope base,
#        DS4_CUDA_WEIGHT_IPC_MANIFEST import boot --no-spec + golden
#        vectors under the manifest env.  SKIP only with a receipt note.
#
# Each server leg (l1,l2,l3,l5,l6,l7): (a) one serial decode returns a
# terminal message with census faults 0 before AND after; (b) decision-
# family parity vs the control tree (~/code/ds4-cebaff7base) with the
# d0a normalization + fit-jitter adjudication; (c) Σ-sources reconcile;
# (d) zero UNIT-TABLE FAULT / RANGE-PUBLISH REFUSED / MEM-CENSUS FAULT
# lines and fully-planned range-plan lines.
#
# Usage: bash speed-bench/d1a_residency_gate.sh   (from the repo root)
set -uo pipefail
HOST=${HOST:-sync-192_168_88_33}
TEST_TREE='~/code/ds4-phase0'
CTRL_TREE='~/code/ds4-cebaff7base'
CTX=${CTX:-32768}
PORT=8000
BASE_GGUF=/home/ent/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP_GGUF=/home/ent/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
GOLDEN_VEC=tests/test-vectors/local-golden-0731.vec
WS_MANIFEST=/tmp/d1a_ws.manifest
TS=$(date +%s); NONCE="d1ares-$TS-$$"
LOCK=/tmp/ds4_box_lock
R() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }
SPID=""; WSPID=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
cleanup() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    [ -n "$WSPID" ] && R "kill $WSPID" 2>/dev/null
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4_weight_serv"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    R "rm -rf $LOCK" 2>/dev/null
}
die() { log "FAIL: $*"; cleanup; exit 1; }

R "ps -eo comm | grep -q '^ds4-server' && echo BUSY || echo IDLE" | grep -q IDLE \
    || die "a ds4-server is already running on $HOST"
R "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held: $(R cat $LOCK/nonce 2>/dev/null || echo unknown)"
trap cleanup EXIT
log "lock acquired ($NONCE)"

boot_tree() { # $1 tree  $2 env  $3 flags  $4 log  [$5 ready-regex, default HTTP]
    local ready=${5:-'listening on http'}
    ssh -o ConnectTimeout=10 "$HOST" \
        "cd $1 && setsid nohup env $2 ./ds4-server -c $CTX $3 > $4 2>&1 < /dev/null & exit 0" &
    local lp=$!; sleep 5; kill $lp 2>/dev/null || true
    for i in $(seq 300); do
        R "grep -q '$ready' $4" && { SPID=$(R "pgrep -x ds4-server | head -1"); return 0; }
        R "ps -eo comm | grep -q '^ds4-server'" || { R "tail -5 $4"; return 1; }
        sleep 2
    done
    return 1
}
kill_server() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    for i in $(seq 20); do R "ps -eo comm | grep -q '^ds4-server'" || { SPID=""; sleep 5; return 0; }; sleep 2; done
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill -9'; SPID=""; sleep 5
}
met() { R "curl -s -m 20 localhost:$PORT/metrics" | awk -v k="$1" '$1==k{print $2}'; }
decode_one() { # $1 out json, $2 tag
    R "curl -s -m 600 -X POST localhost:$PORT/v1/messages -H 'content-type: application/json' \
        -d '{\"model\":\"ds4\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Gate $2 $TS: name three rivers and one ledger.\"}]}' -o $1"
    R "grep -q '\"type\":\"message\"' $1"
}
avail_mb() { R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }
stable_avail() { # two reads within 150M (d0a ABORT-on-unstable law)
    local a b
    for t in 1 2 3 4 5; do
        a=$(avail_mb); sleep 5; b=$(avail_mb)
        local d=$((a - b)); [ ${d#-} -lt 150 ] && { echo "$b"; return 0; }
    done
    return 1
}
# Decision families, normalized exactly like d0a_census_gate.sh: strip
# timestamps, mask the free-derived inputs; formulas + verdict shapes
# must byte-match.  The test tree's D1a lines are excluded by grep
# construction (the control cannot print them) -- the D1a-4 range-plan
# porcelain ends in "refused N" and would otherwise leak in through the
# 'refus' family pattern (l1 first-run finding, 2026-08-12).
extract_decisions() { # $1 remote log  $2 local out
    R "grep -E 'batch fit|batch vmm|mem floor|boot ledger|refus|reject|no graph fits|admission|serial fit' $1" \
        | grep -v 'ds4: range plan ' \
        | sed -E 's/^[0-9]{4} [0-9:]{8} //; s/free=[0-9.]+ GiB/free=# GiB/g; s/MemAvailable[= ][0-9.]+/MemAvailable=#/g; s/capacity [0-9.]+ GiB/capacity # GiB/g; s/max_seq [0-9]+/max_seq #/g; s/x [0-9]+ banks/x # banks/g; s/budget=[0-9.]+ GiB/budget=# GiB/g; s/\[plan [0-9.]+, capacity [0-9.]+\]/[plan #, capacity #]/g' \
        > "$2"
}
jitter_ok() { # $1 name  $2 val_T  $3 val_C  $4 free_T  $5 free_C
    python3 -c "
t,c,ft,fc=int($2),int($3),float($4),float($5)
if t==c: print('EQUAL'); exit(0)
if abs(t-c)==1 and ((t>c)==(ft>fc)): print('JITTER'); exit(0)
print('DIVERGENT'); exit(1)"
}
fit_free_gib() { R "grep 'batch fit' $1 | head -1" | grep -o 'free=[0-9.]*' | cut -d= -f2; }
fit_max_seq()  { R "grep 'batch fit' $1 | head -1" | grep -o 'max_seq [0-9]*' | awk '{print $2}'; }
vmm_banks()    { R "grep 'batch vmm' $1 | head -1" | grep -o 'x [0-9]* banks' | grep -o '[0-9]*'; }

# ---- shared assert batteries ----
plan_asserts() { # $1 log  $2.. expected source names
    local L=$1; shift
    R "grep -q 'ds4: model-source ledger reconciled:' $L" || die "$L: no reconcile line"
    R "grep -q 'source ledger mismatch' $L" && die "$L: ledger mismatch"
    R "grep -q 'model source UNATTRIBUTED' $L" && die "$L: unattributed weight bytes"
    R "grep -q 'UNIT-TABLE FAULT' $L" && die "$L: unit-table fault"
    R "grep -q 'RANGE-PUBLISH REFUSED' $L" && die "$L: range publication refused"
    R "grep -q 'MEM-CENSUS FAULT' $L" && die "$L: census fault line"
    local s
    for s in "$@"; do
        R "grep -q 'ds4: model source $s \[' $L" || die "$L: no $s source line"
        R "grep -q 'ds4: model catalog $s:' $L" || die "$L: no $s catalog line"
        R "grep -q 'ds4: model units $s: .* (verify=0)' $L" || die "$L: no clean $s unit table"
        R "grep -q 'ds4: range plan $s:' $L" || die "$L: no $s range-plan line"
    done
}
range_plan_full() { # $1 log — planned == total + refused 0 on EVERY line
    local BAD
    BAD=$(R "grep 'ds4: range plan ' $1" \
        | sed -E 's/.*model ([0-9]+)\/([0-9]+) planned, derived ([0-9]+)\/([0-9]+), q8 ([0-9]+)\/([0-9]+), refused ([0-9]+).*/\1 \2 \3 \4 \5 \6 \7/' \
        | awk '$1!=$2 || $3!=$4 || $5!=$6 || $7!=0 {print}')
    [ -z "$BAD" ] || die "$1: range plan incomplete/refused: $BAD"
}
faults_zero() { # $1 tag
    local CF; CF=$(met ds4_memory_census_faults_total)
    [ "$CF" = "0" ] || die "$1: census faults=$CF"
}

# ---- paired server leg: test asserts + decode, then control parity ----
pair_leg() { # $1 name  $2 env  $3 flags  $4.. expected sources
    local name=$1 env=$2 flags=$3; shift 3
    local attempt TLOG=/tmp/d1ares_${name}_T.log CLOG=/tmp/d1ares_${name}_C.log
    for attempt in 1 2; do
        log "($name) attempt $attempt: test boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable pre-boot (ABORT)"
        boot_tree "$TEST_TREE" "$env" "$flags" "$TLOG" || die "$name test boot failed"
        plan_asserts "$TLOG" "$@"
        range_plan_full "$TLOG"
        faults_zero "$name boot"
        decode_one "/tmp/d1ares_${name}_dec.json" "$name" || die "$name decode failed"
        sleep 3
        faults_zero "$name after decode"
        kill_server
        log "($name) attempt $attempt: control boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable between pair (ABORT)"
        boot_tree "$CTRL_TREE" "$env" "$flags" "$CLOG" || die "$name control boot failed"
        kill_server
        extract_decisions "$TLOG" "/tmp/d1ares_${name}_T.dec"
        extract_decisions "$CLOG" "/tmp/d1ares_${name}_C.dec"
        if ! diff -q "/tmp/d1ares_${name}_T.dec" "/tmp/d1ares_${name}_C.dec" >/dev/null; then
            diff "/tmp/d1ares_${name}_T.dec" "/tmp/d1ares_${name}_C.dec" | head -12
            die "$name: decision families differ (normalized byte-diff)"
        fi
        local ft fc mt mc bt bc mv bv band
        ft=$(fit_free_gib "$TLOG"); fc=$(fit_free_gib "$CLOG")
        mt=$(fit_max_seq  "$TLOG"); mc=$(fit_max_seq  "$CLOG")
        bt=$(vmm_banks    "$TLOG"); bc=$(vmm_banks    "$CLOG")
        log "($name) inputs: free T=$ft C=$fc GiB; max_seq T=$mt C=$mc; vmm_banks T=${bt:--} C=${bc:--}"
        band=$(python3 -c "print(1 if abs($ft-$fc)<1.0 else 0)")
        if [ "$band" != "1" ]; then
            log "($name): free inputs outside the 1.0 GiB drift band -- retrying pair once"
            continue
        fi
        mv=$(jitter_ok "$name/max_seq" "$mt" "$mc" "$ft" "$fc") || { log "($name) max_seq $mt vs $mc DIVERGENT"; mv=FAIL; }
        bv=EQUAL
        if [ -n "$bt" ] && [ -n "$bc" ]; then
            bv=$(jitter_ok "$name/vmm_banks" "$bt" "$bc" "$ft" "$fc") || { log "($name) vmm_banks $bt vs $bc DIVERGENT"; bv=FAIL; }
        fi
        if [ "$mv" != "FAIL" ] && [ "$bv" != "FAIL" ]; then
            log "($name) PASS (max_seq=$mv vmm_banks=$bv)"
            sleep 10
            return 0
        fi
        log "($name): derived decision divergent -- retrying pair once"
    done
    die "$name: derived decisions diverge or inputs drift on both attempts -- NOT jitter"
}

golden_leg() { # $1 name  $2 env
    local name=$1 env=$2 GLOG=/tmp/d1ares_golden_${name}.log RC
    log "(golden:$name) ds4_test --local-golden-vectors under $name env"
    RC=$(R "cd $TEST_TREE && env DS4_TEST_MODEL=$BASE_GGUF DS4_TEST_LOCAL_GOLDEN_FILE=$GOLDEN_VEC $env ./ds4_test --local-golden-vectors > $GLOG 2>&1; echo \$?")
    R "grep 'ds4-test: local golden' $GLOG" || true
    [ "$RC" = "0" ] || { R "tail -15 $GLOG"; die "golden:$name RC=$RC"; }
    R "grep -q 'ds4-test: local golden' $GLOG" || die "golden:$name: no golden summary line"
    log "(golden:$name) PASS"
    sleep 10
}

# ---- (l1..l3) source-combination legs ----
pair_leg l1_baseonly "" "--no-spec" base
pair_leg l2_basemtp  "" "--no-dspark" base mtp
pair_leg l3_allthree "" "--mtp $MTP_GGUF" base mtp drafter

# ---- (l4) slice boot: worker-shape, boot asserts only ----
log "(l4) slice boot: coordinator --layers 20:40 (boot-shape leg)"
SLOG=/tmp/d1ares_l4_slice.log
boot_tree "$TEST_TREE" "" "--role coordinator --listen 127.0.0.1 9411 --layers 20:40 --no-spec" \
    "$SLOG" "backend initialized for graph diagnostics" \
    || die "l4 slice coordinator boot failed (never reached backend init)"
plan_asserts "$SLOG" base
SLICELINE=$(R "grep 'ds4: range plan base' $SLOG" | tail -1)
echo "$SLICELINE" | grep -q 'refused 0' || die "l4: slice refused != 0 ($SLICELINE)"
UNITLINE=$(R "grep 'ds4: model units base' $SLOG" | tail -1)
log "(l4) slice unit table: $UNITLINE"
if echo "$SLICELINE" | sed -E 's/.*model ([0-9]+)\/([0-9]+) planned.*/\1 \2/' | awk '$1!=$2{exit 0} {exit 1}'; then
    log "(l4) SLICE-BLINDNESS RECORDED (expected until D1b adopts the table): $SLICELINE"
else
    log "(l4) slice fully planned: $SLICELINE"
fi
kill_server
sleep 10
log "(l4) PASS (boot-shape asserts; serving needs a coordinator — receipt notes the scope)"

# ---- (l5) raw + goldens ----
pair_leg l5_raw "DS4_CUDA_NO_DERIVED_WEIGHTS=1 DS4_CUDA_NO_HBM_CACHE=1 DS4_BATCH_VMM_BUDGET_MB=8000" "" base drafter
golden_leg raw "DS4_CUDA_NO_DERIVED_WEIGHTS=1 DS4_CUDA_NO_HBM_CACHE=1"

# ---- (l6) self-load + goldens ----
pair_leg l6_selfload "" "" base drafter
R "grep -q 'ds4: aligned artifacts built in-process' /tmp/d1ares_l6_selfload_T.log" \
    || die "l6: no BUILT artifact banner on the self-load boot"
golden_leg selfload ""

# ---- (l7) manifest via .33-local weight server + goldens ----
log "(l7) starting ds4_weight_server (scope=base) -- repacks take minutes"
R "rm -f $WS_MANIFEST"
ssh -o ConnectTimeout=10 "$HOST" \
    "cd $TEST_TREE && setsid nohup ./ds4_weight_server --base $BASE_GGUF --scope base --manifest $WS_MANIFEST > /tmp/d1a_ws.log 2>&1 < /dev/null & exit 0" &
WLP=$!; sleep 5; kill $WLP 2>/dev/null || true
WS_OK=0
for i in $(seq 450); do
    R "grep -q 'ds4_weight_server: ready manifest=' /tmp/d1a_ws.log" && { WS_OK=1; break; }
    R "ps -eo comm | grep -q '^ds4_weight_serv'" || break
    sleep 2
done
if [ "$WS_OK" = "1" ]; then
    WSPID=$(R "pgrep -x ds4_weight_serv | head -1")
    log "(l7) weight server ready (pid $WSPID)"
    pair_leg l7_manifest "DS4_CUDA_WEIGHT_IPC_MANIFEST=$WS_MANIFEST" "--no-spec" base
    R "grep -q 'ds4: CUDA imported shared' /tmp/d1ares_l7_manifest_T.log" \
        || die "l7: no import engagement line"
    golden_leg manifest "DS4_CUDA_WEIGHT_IPC_MANIFEST=$WS_MANIFEST"
    R "kill $WSPID" 2>/dev/null; WSPID=""
    sleep 5
else
    R "tail -10 /tmp/d1a_ws.log" || true
    log "(l7) SKIPPED: weight server never reached ready -- RECEIPT NOTE REQUIRED"
fi

log "D1A RESIDENCY GATE: ALL LEGS COMPLETE"
cleanup
trap - EXIT
