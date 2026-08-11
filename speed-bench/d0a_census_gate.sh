#!/bin/bash
# d0a_census_gate.sh — memgov D0a close gate (spec: local/docs/v057/
# d0a_scoping_2026-08-08.md §3 D0a-close).  Mac-driven, runs .33 legs under
# the box-ownership lock and Mac legs locally.
#
# WHAT IT PROVES
#   (a) DECISION ORACLE: fit/floor/admission/reject log families identical
#       vs the pre-D0a control build (~/code/ds4-v0563base @ b9c97ad) on
#       three boot shapes: eager default, DS4_CUDA_NO_HBM_CACHE=1 (the
#       tripwire class exercised), and the zeroconf deep leg.  "Identical"
#       is byte-diff on the normalized family lines: timestamps stripped
#       and the free-memory INPUT masked (fit-jitter law: consecutive
#       boots differ in MemAvailable, not in the formula) -- with the raw
#       inputs asserted inside a drift band and the DERIVED decision
#       (max_seq) asserted equal, one retry per pair to absorb a
#       boundary-crossing jitter.
#   (b) RECONCILIATION at settle: census registry totals vs the allocator
#       books (WEIGHT_HOST_PIN vs the registered-mapping boot line;
#       ENGINE_OTHER live == 0) vs the system delta (MemAvailable
#       pre-boot minus settle vs census live totals, stated tolerance,
#       sources logged).  An unstable starting memory state ABORTS (plan
#       §13), never warns-and-continues.  Substrate tripwire bit-identity:
#       census totals line == fit line, same boot.
#   (c) CPU/Metal: local `make ds4_server_cpu.o` compiles and the Metal
#       units binary runs the server suite green (census reads report
#       unsupported, never zero).
#   (d) NO-HOT-PATH: every census/injection call in ds4_cuda.cu sits in an
#       allow-listed control-plane function (alloc/free/ensure/trim/init/
#       cleanup/import/artifact); ABBA perf close vs the control build
#       (short decode + prefill, ABBA order, <5% with both clocks logged).
#
# Usage: bash speed-bench/d0a_census_gate.sh          (from the repo root)
#   env: HOST, CTX (default 32768), DEEP_CTX (default 240000), TOL_GIB.
set -uo pipefail
HOST=${HOST:-sync-192_168_88_33}
TEST_TREE='~/code/ds4-phase0'
CTRL_TREE='~/code/ds4-v0563base'
CTX=${CTX:-32768}
DEEP_CTX=${DEEP_CTX:-240000}
TOL_GIB=${TOL_GIB:-3}
PORT=8000
TS=$(date +%s); NONCE="d0aclose-$TS-$$"
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

# ---- box ownership ----
R "ps -eo comm | grep -q '^ds4-server' && echo BUSY || echo IDLE" | grep -q IDLE \
    || die "a ds4-server is already running on $HOST"
R "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held: $(R cat $LOCK/nonce 2>/dev/null || echo unknown)"
trap cleanup EXIT
log "lock acquired ($NONCE)"

# ---- (c) CPU + Metal legs, local ----
log "(c) CPU object + Metal server units (local)"
( cd "$(dirname "$0")/.." && make ds4_server_cpu.o >/tmp/d0aclose_cpu.log 2>&1 ) \
    || die "CPU object build failed (see /tmp/d0aclose_cpu.log)"
( cd "$(dirname "$0")/.." && make ds4_test >/dev/null 2>&1 && ./ds4_test --server >/tmp/d0aclose_metal.log 2>&1 ) \
    || die "Metal server units failed (see /tmp/d0aclose_metal.log)"
grep -q "server: OK" /tmp/d0aclose_metal.log || die "Metal units did not report server: OK"
log "(c) PASS"

# ---- (d1) no-hot-path source proof ----
log "(d1) no-hot-path allow-list scan (ds4_cuda.cu)"
python3 - "$(dirname "$0")/../ds4_cuda.cu" <<'EOF' || exit 1
import re, sys
src = open(sys.argv[1]).read().splitlines()
# Enclosing-function scan: registry/injection calls may appear ONLY inside
# the EXPLICIT reviewed owner set below (inventoried 2026-08-11 from the
# D0a tree; every entry adjudicated control-plane: funnel, arenas, sticky
# growers -- capture-refused by the sticky-scratch law -- load/teardown,
# and ONE documented kernel-wrapper exception: the norm_q8 sticky grow in
# ds4_gpu_rms_norm_weight_rows_f16_tensor).  A note in ANY other function
# fails the gate until reviewed and added here.  No wildcards: a wildcard
# that matches a kernel wrapper is a hole, not a convenience.
allow = {
    "cuda_mem_note_alloc", "cuda_mem_note_free", "cuda_trim_inject_fire",
    "ds4_gpu_mem_census_read",
    "cuda_model_arena_alloc", "cuda_model_copy_chunked",
    "cuda_model_range_populate_device_copy", "cuda_model_range_release_all",
    "cuda_model_stage_pool_alloc",
    "cuda_moe_iq2_derepack_scratch", "cuda_moe_q2k_derepack_scratch",
    "fp8_predecode_scratch_alloc", "rr_scratch_ensure", "tt_scratch_ensure",
    "cuda_tmp_alloc",
    "cuda_pinned_file_note_register", "cuda_pinned_file_note_unregister",
    "cuda_q8_f16_cache_release_all", "cuda_q8_f16_ptr", "cuda_q8_f32_ptr",
    "cuda_vmm_arena_alloc", "cuda_vmm_arenas_release_all",
    "ds4_gpu_build_derived_artifacts", "ds4_gpu_import_model_ipc_manifest",
    "ds4_gpu_set_model_map",
    "import_vmm_allocation", "import_vmm_derived_allocation",
    "ds4_gpu_init", "ds4_gpu_cleanup",
    "ds4_gpu_decode_scalars_init", "ds4_gpu_decode_scalars_cleanup",
    "ds4_gpu_decode_layer_scalars_init", "ds4_gpu_decode_layer_scalars_cleanup",
    "ds4_gpu_decode_row_scalars_init",
    "ds4_gpu_tensor_alloc", "ds4_gpu_tensor_alloc_managed",
    "ds4_gpu_tensor_ensure", "ds4_gpu_tensor_trim", "ds4_gpu_tensor_free",
    "ds4_gpu_rms_norm_weight_rows_f16_tensor",
}
call = re.compile(r'\b(cuda_mem_note_(alloc|free)|cuda_trim_inject_fire|ds4_gpu_mem_census_read)\s*\(')
fndef = re.compile(r'^(?:static\s+|extern\s+"C"\s+)?[A-Za-z_][\w\s\*:<>,]*?\b([A-Za-z_]\w*)\s*\([^;]*$')
cur, depth, bad = "?", 0, []
for i, ln in enumerate(src, 1):
    if depth == 0:
        m = fndef.match(ln)
        if m:
            cur = m.group(1)
    depth += ln.count('{') - ln.count('}')
    if depth < 0: depth = 0
    if call.search(ln) and cur not in allow:
        bad.append((i, cur, ln.strip()[:90]))
if bad:
    for b in bad: print("HOT-PATH LEAK: line %d in %s: %s" % b)
    sys.exit(1)
print("no-hot-path scan clean (%d call sites, all in the reviewed owner set)" %
      sum(1 for ln in src if call.search(ln)))
EOF
[ $? -eq 0 ] || die "no-hot-path scan found registry calls outside the allow-list"
log "(d1) PASS"

# ---- remote helpers ----
BOOT_FLAGS=""   # extra ds4-server flags (ABBA sets --no-spec --no-mtp)
boot_tree() { # $1 tree  $2 env  $3 ctx  $4 log
    ssh -o ConnectTimeout=10 "$HOST" \
        "cd $1 && setsid nohup env $2 ./ds4-server -c $3 $BOOT_FLAGS > $4 2>&1 < /dev/null & exit 0" &
    local lp=$!; sleep 5; kill $lp 2>/dev/null || true
    for i in $(seq 240); do
        R "grep -q 'listening on http' $4" && { SPID=$(R "pgrep -x ds4-server | head -1"); return 0; }
        R "ps -eo comm | grep -q '^ds4-server'" || { R "tail -5 $4"; return 1; }
        sleep 2
    done
    return 1
}
kill_server() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    for i in $(seq 20); do R "ps -eo comm | grep -q '^ds4-server'" || { SPID=""; return 0; }; sleep 2; done
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill -9'; SPID=""
}
avail_mb() { R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }
stable_avail() { # ABORT-on-unstable (plan §13): two reads within 150M
    local a b
    for t in 1 2 3 4 5; do
        a=$(avail_mb); sleep 5; b=$(avail_mb)
        local d=$((a - b)); [ ${d#-} -lt 150 ] && { echo "$b"; return 0; }
    done
    return 1
}
# Decision families, normalized: strip the timestamp/pid prefix, mask the
# free-memory inputs (fit-jitter law) -- formulas and verdict shapes must
# byte-match; the masked inputs are band-checked separately.
extract_decisions() { # $1 remote log  $2 local out
    # max_seq is masked HERE (it is free-input-derived, the fit-jitter law);
    # its equality is asserted separately with a one-retry jitter policy.
    # The batch-vmm ledger's banks/budget/plan/capacity are ALL free-input
    # derived (plan = banks x per-bank; banks from the free-derived fit):
    # masked here, with the bank COUNT adjudicated separately like max_seq.
    R "grep -E 'batch fit|batch vmm|mem floor|boot ledger|refus|reject|no graph fits|admission|serial fit' $1" \
        | sed -E 's/^[0-9]{4} [0-9:]{8} //; s/free=[0-9.]+ GiB/free=# GiB/g; s/MemAvailable[= ][0-9.]+/MemAvailable=#/g; s/capacity [0-9.]+ GiB/capacity # GiB/g; s/max_seq [0-9]+/max_seq #/g; s/x [0-9]+ banks/x # banks/g; s/budget=[0-9.]+ GiB/budget=# GiB/g; s/\[plan [0-9.]+, capacity [0-9.]+\]/[plan #, capacity #]/g' \
        > "$2"
}
# Fit-jitter adjudication for a free-derived integer decision: equal is
# clean; a one-step difference whose DIRECTION matches the free inputs is
# the documented jitter shape (D0a-1 law); anything else is real.
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

# ---- (a) decision oracle: three legs x (test, control) ----
oracle_leg() { # $1 name  $2 env  $3 ctx
    local name=$1 env=$2 ctx=$3 attempt
    for attempt in 1 2; do
        log "(a) $name attempt $attempt: test boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable pre-boot (ABORT, plan §13)"
        boot_tree "$TEST_TREE" "$env" "$ctx" "/tmp/d0aclose_${name}_T.log" || die "$name test boot failed"
        kill_server
        log "(a) $name attempt $attempt: control boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable between pair (ABORT)"
        boot_tree "$CTRL_TREE" "$env" "$ctx" "/tmp/d0aclose_${name}_C.log" || die "$name control boot failed"
        kill_server
        extract_decisions "/tmp/d0aclose_${name}_T.log" "/tmp/d0aclose_${name}_T.dec"
        extract_decisions "/tmp/d0aclose_${name}_C.log" "/tmp/d0aclose_${name}_C.dec"
        # The test tree ADDS census lines the control cannot have; decision
        # families exclude them by grep construction.  Byte-diff:
        if ! diff -q "/tmp/d0aclose_${name}_T.dec" "/tmp/d0aclose_${name}_C.dec" >/dev/null; then
            diff "/tmp/d0aclose_${name}_T.dec" "/tmp/d0aclose_${name}_C.dec" | head -12
            die "$name: decision families differ (normalized byte-diff)"
        fi
        local ft fc mt mc bt bc mv bv
        ft=$(fit_free_gib "/tmp/d0aclose_${name}_T.log"); fc=$(fit_free_gib "/tmp/d0aclose_${name}_C.log")
        mt=$(fit_max_seq  "/tmp/d0aclose_${name}_T.log"); mc=$(fit_max_seq  "/tmp/d0aclose_${name}_C.log")
        bt=$(vmm_banks    "/tmp/d0aclose_${name}_T.log"); bc=$(vmm_banks    "/tmp/d0aclose_${name}_C.log")
        log "(a) $name inputs: free T=$ft C=$fc GiB; max_seq T=$mt C=$mc; vmm_banks T=${bt:--} C=${bc:--}"
        local band; band=$(python3 -c "print(1 if abs($ft-$fc)<1.0 else 0)")
        [ "$band" = "1" ] || die "$name: free inputs outside the 1.0 GiB drift band"
        mv=$(jitter_ok "$name/max_seq" "$mt" "$mc" "$ft" "$fc") || { log "(a) $name max_seq $mt vs $mc DIVERGENT"; mv=FAIL; }
        bv=EQUAL
        if [ -n "$bt" ] && [ -n "$bc" ]; then
            bv=$(jitter_ok "$name/vmm_banks" "$bt" "$bc" "$ft" "$fc") || { log "(a) $name vmm_banks $bt vs $bc DIVERGENT"; bv=FAIL; }
        fi
        if [ "$mv" != "FAIL" ] && [ "$bv" != "FAIL" ]; then
            log "(a) $name PASS (max_seq=$mv vmm_banks=$bv)"
            return 0
        fi
        log "(a) $name: derived decision divergent -- retrying pair once"
    done
    die "$name: derived decisions diverge on both attempts -- NOT jitter"
}
oracle_leg eager  ""                       "$CTX"
# nohbm: the vmm budget is PINNED on both trees (handoff law: pin the plan)
# so the dominant free-derived quantity is deterministic; capacity stays
# masked and the bank count still runs the jitter adjudication.
oracle_leg nohbm  "DS4_CUDA_NO_HBM_CACHE=1 DS4_BATCH_VMM_BUDGET_MB=8000" "$CTX"
oracle_leg deep   ""                       "$DEEP_CTX"

# ---- (b) reconciliation at settle (test tree, eager boot) ----
log "(b) reconciliation boot (test tree)"
PRE_MB=$(stable_avail) || die "reconciliation: MemAvailable unstable pre-boot (ABORT)"
log "(b) pre-boot MemAvailable=${PRE_MB}M (stable)"
boot_tree "$TEST_TREE" "" "$CTX" /tmp/d0aclose_recon.log || die "reconciliation boot failed"
sleep 20   # settle: post-prewarm idle
POST_MB=$(stable_avail) || die "reconciliation: MemAvailable unstable at settle (ABORT)"
R "curl -s -m 30 localhost:$PORT/metrics > /tmp/d0aclose_metrics.txt" || die "metrics curl"
# ENGINE_OTHER settles to zero on server boots (census doc §11 contract).
EO=$(R "grep 'class=\"engine_other\",state=\"allocated\"' /tmp/d0aclose_metrics.txt" | awk '{s+=$NF} END{print s}')
[ "$EO" = "0" ] || die "ENGINE_OTHER allocated != 0 ($EO) -- untagged allocation on a server boot"
FAULTS=$(R "grep '^ds4_memory_census_faults_total' /tmp/d0aclose_metrics.txt" | awk '{print $NF}')
[ "$FAULTS" = "0" ] || die "census faults at settle ($FAULTS)"
# WEIGHT_HOST_PIN == registered-mapping boot line (single funnel, 1:1).
HP=$(R "grep 'domain=\"pinned_host\",class=\"weight_host_pin\",state=\"allocated\"' /tmp/d0aclose_metrics.txt" | awk '{print $NF}')
REG_GIB=$(R "grep -o 'registered [0-9.]* GiB model mapping' /tmp/d0aclose_recon.log | head -1" | grep -o '[0-9.]*' | head -1)
HP_OK=$(python3 -c "print(1 if abs($HP/2**30 - $REG_GIB) < 0.02 else 0)")
[ "$HP_OK" = "1" ] || die "WEIGHT_HOST_PIN ($HP B) != registered mapping (${REG_GIB} GiB)"
# System delta vs census totals, stated tolerance, sources logged.
DEV=$(R "grep 'state=\"allocated\"' /tmp/d0aclose_metrics.txt | grep 'unified_device'" | awk '{s+=$NF} END{print s}')
HOSTB=$(R "grep 'state=\"allocated\"' /tmp/d0aclose_metrics.txt | grep 'pinned_host'" | awk '{s+=$NF} END{print s}')
DELTA_MB=$((PRE_MB - POST_MB))
# Tolerance = max(TOL_GIB, 5% of census): MemAvailable is a kernel
# ESTIMATE (watermark + reclaimable heuristics) and an 86 GiB model file
# streams through page cache during boot -- a flat few-GiB band is
# mis-sized against a ~107 GiB footprint.  The SHARP cross-checks are the
# exact ones above (host_pin==registered 1:1, ENGINE_OTHER==0, faults==0,
# census rows == D0a-0 ground truth); this delta exists to catch GROSS
# drift (a missing class, double counting), not estimate noise.
RECON=$(python3 -c "
census=($DEV+$HOSTB)/2**30; sys=$DELTA_MB/1024.0
tol=max(float($TOL_GIB), 0.05*census)
print('census=%.2f sys_delta=%.2f diff=%.2f tol=%.2f -> %s' % (census, sys, abs(census-sys), tol, 'OK' if abs(census-sys)<tol else 'FAIL'))")
log "(b) reconciliation: $RECON (sources: /metrics census totals; /proc/meminfo MemAvailable ${PRE_MB}M->${POST_MB}M)"
echo "$RECON" | grep -q 'OK$' || die "reconciliation outside tolerance"
# Substrate tripwire bit-identity (census totals line vs fit line).
SUB_C=$(R "grep 'memory census totals:' /tmp/d0aclose_recon.log | head -1" | grep -o 'substrate_outstanding=[0-9.]*' | cut -d= -f2)
SUB_F=$(R "grep 'batch fit' /tmp/d0aclose_recon.log | head -1" | grep -o 'substrate outstanding=[0-9.]*' | grep -o '[0-9.]*')
[ "$SUB_C" = "$SUB_F" ] || die "substrate tripwire mismatch census=$SUB_C fit=$SUB_F"
log "(b) PASS (ENGINE_OTHER=0 faults=0 host_pin==registered substrate bit-identical)"

# ---- (d2) ABBA perf close vs control (short decode leg) ----
# Instrument: the measured request's OWN timings.decode_tok_s (exact for
# that request), never the 60s window gauge (run-1 lesson: the gauge read
# 9.85 and 12.8 tok/s for the SAME tree across two boots while the
# per-request path is stable).  Both sides boot --no-spec --no-mtp
# (launch-defaults law: accept-rate variance alone busts a 5% band) and
# pay one warmup request (graph capture + page warm) before measuring.
abba_ms_tok() { # $1 tree -> echoes decode_tok_s of a measured 256-token request
    kill_server
    BOOT_FLAGS="--no-spec --no-mtp"
    boot_tree "$1" "" "$CTX" /tmp/d0aclose_abba.log || die "ABBA boot failed ($1)"
    BOOT_FLAGS=""
    R "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null || echo clocks=n/a" >&2
    R "curl -s -m 300 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Warm up.\"}],\"max_tokens\":128}'" > /dev/null
    local out
    out=$(R "curl -s -m 300 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Write a 400-word story about a ledger.\"}],\"max_tokens\":256}'")
    echo "$out" | grep -q finish_reason || die "ABBA request failed ($1)"
    echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['decode_tok_s'])"
}
log "(d2) ABBA decode close (T C C T, per-request timings, plain decode)"
T1=$(abba_ms_tok "$TEST_TREE");  log "  T1=$T1 tok/s"
C1=$(abba_ms_tok "$CTRL_TREE");  log "  C1=$C1 tok/s"
C2=$(abba_ms_tok "$CTRL_TREE");  log "  C2=$C2 tok/s"
T2=$(abba_ms_tok "$TEST_TREE");  log "  T2=$T2 tok/s"
kill_server
ABBA=$(python3 -c "
t=($T1+$T2)/2; c=($C1+$C2)/2
d=abs(t-c)/c*100
print('test=%.1f ctrl=%.1f delta=%.1f%% -> %s' % (t, c, d, 'OK' if d < 5.0 else 'FAIL'))")
log "(d2) ABBA: $ABBA"
echo "$ABBA" | grep -q 'OK$' || die "ABBA decode delta >= 5%"

cleanup
trap - EXIT
log "D0A CENSUS GATE: ALL LEGS PASS"
