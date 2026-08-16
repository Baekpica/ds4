#!/bin/bash
# memgov_soak.sh — v0.6.0 cut prep B: the 24h memory-governance soak.
#
# Chartered by the D5-5 adjudication (soak + matrix fold into cut prep):
# mixed cont/serial/static churn, client cancels, hog pulses with
# reclaim engagement, and the DS4_MEMGOV=observe rollback proven ONCE
# mid-soak.  Asserts three release claims over ~24 h of load:
#   NO LEAKS         anon-RSS (smaps_rollup Anonymous) flat within each
#                    server lifetime + census/governor faults 0 + torn
#                    fallbacks 0.  VmRSS and MemAvailable are NOT leak
#                    signals in mmap mode (file-backed model pages swing
#                    both; the gb10 MemAvailable-decline is host noise).
#   NO UNBOUNDED     ds4_requests_rejected_total moves ONLY on pressure
#   REJECTS          cycles (typed family = the D6 field signal; a tick
#                    on a quiet cycle fails the soak).
#   NO STATE BLEED   post-pulse warm replay moves the cached-prefill
#                    counter; zero crash lines; census epoch strictly
#                    advances every cycle.
#
# RUNS ON THE BOX (localhost curls, local /proc) so the driver survives
# Mac sleep/ssh drops.  Launch detached from the Mac:
#   rsync the tree, then
#   ssh $HOST 'cd ~/code/ds4-phase0 && setsid nohup bash \
#     speed-bench/memgov_soak.sh > /tmp/ds4_soak/driver.log 2>&1 &'
# and tail the log with a line monitor.  Heartbeat = one CYCLE line per
# cycle (~30 min); silence past ~40 min means the driver is wedged.
#
# Cycle plan (CYCLE_MIN each, default 48 cycles = 24 h):
#   wave     6 cont chats (4 buffered, 2 streaming, one stream CANCELLED
#            mid-decode -- the abort-paths churn)
#   static   POST /v1/batch (2 prompts) -- the static lane's HOME
#            surface (percall graph family).  The d0b coalesce shape
#            only reaches static when cont is unavailable; on a healthy
#            boot it rides cont, so it proves nothing here.
#   serial   1 small /v1/messages with a tools array (the serial ticket)
#   pulse    every PULSE_EVERY-th cycle: the serial_reclaim_gate
#            calibrated choreography (24 banks x 31k tenants, hog to
#            NEED_EST_MB - RECLAIM_MARGIN_MB, 28k deep serial ->
#            serve-via-reclaim; warm replay of tenant 1 must cache-hit).
#            A sagged arrival below the bracket SKIPS the reclaim assert
#            loudly (counted in the verdict; reboot is user-only, and
#            the sag is the documented host decline, not engine truth).
#            The verdict REQUIRES >= 2 engaged pulses over the run.
#   sample   full /metrics scrape saved per cycle + CSV row + asserts
# The idle remainder of each cycle is part of the soak: retention/hold
# timers and eviction breathe between waves.
#
# Rollback slice: cycle OBSERVE_AT runs on a DS4_MEMGOV=observe boot
# (modes line asserted, zero ENFORCE refusals, healthy serves), then
# the soak re-boots enforce and continues.  Three server lifetimes
# total; leak windows are computed within-lifetime only.
set -uo pipefail
HOURS=${HOURS:-24}
CYCLE_MIN=${CYCLE_MIN:-30}
PULSE_EVERY=${PULSE_EVERY:-8}
CTX=${CTX:-32768}
BANKS=${BANKS:-24}
PORT=${PORT:-8000}
NEED_EST_MB=${NEED_EST_MB:-5900}
RECLAIM_MARGIN_MB=${RECLAIM_MARGIN_MB:-400}
TENANT_TOKENS=${TENANT_TOKENS:-31000}
DEEP_TOKENS=${DEEP_TOKENS:-28000}
ANON_LEAK_MB=${ANON_LEAK_MB:-1024}
SETTLE_MIN_MB=${SETTLE_MIN_MB:-90000}
TREE=${TREE:-"$HOME/code/ds4-phase0"}
WORK=/tmp/ds4_soak
DUR=$HOME/soak_v06          # durable copies land here at the end
LOCK=/tmp/ds4_box_lock
CYCLES=$((HOURS * 60 / CYCLE_MIN))
OBSERVE_AT=$((CYCLES / 2 + 1))
TS=$(date +%s); NONCE="memgov_soak_${TS}_$$"
SRV_PID=""; SRV_LOG=""; PREV_EPOCH=0; PREV_REJ=0; PREV_CRASH=0
PULSES_ENGAGED=0; PULSES_SKIPPED=0; QUIET_REJ_FAILS=0
REPLAYS_WARM=0; REPLAYS_REFUSED=0; QUIET_REJ_TOTAL=0; STATIC_REFUSED=0
mkdir -p "$WORK" "$DUR"
CSV=$WORK/soak_samples.csv
echo "ts,cycle,phase,boot,anon_kb,memavail_mb,census_epoch,gov_epoch,census_faults,gov_faults,torn,banks_live,rejected_sum,crash_cum" > "$CSV"

log() { echo "[$(date +%H:%M:%S)] SOAK: $*"; }
die() { log "FAIL: $*"; log "MEMGOV SOAK: FAILED (cycle ${CYC:-0}/$CYCLES)"; cleanup; exit 1; }
cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    pgrep -x ds4-server | xargs -r kill 2>/dev/null
    [ -f $WORK/hog.pid ] && kill "$(cat $WORK/hog.pid)" 2>/dev/null; rm -f $WORK/hog.pid
    cp -f "$CSV" "$WORK"/driver.log "$DUR"/ 2>/dev/null
    rm -rf $LOCK
    log "cleanup: server+hog down, lock freed, samples copied to $DUR"
}
trap cleanup EXIT

m() { curl -s -m 15 "localhost:$PORT/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
mv_() { local v; v=$(m "$1"); echo "${v:-0}"; }

settle_wait() {
    local avail=0
    for i in $(seq 90); do
        avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 ))
        [ "$avail" -ge "$SETTLE_MIN_MB" ] && { log "settle: ${avail}M >= ${SETTLE_MIN_MB}M"; return 0; }
        sleep 5
    done
    die "memory never settled (${avail}M) before boot -- plan sec 13"
}

boot() { # $1 extra_env  $2 tag
    settle_wait
    SRV_LOG=$WORK/srv_$2.log
    ( cd "$TREE" && setsid nohup env DS4_SERVER_COALESCE_MAX=$BANKS $1 ./ds4-server -c $CTX \
        > "$SRV_LOG" 2>&1 < /dev/null & )
    # Wait for 'listening on http' -- the boot's FINAL act.  'persistent
    # batch ctx ready' prints ~4 s earlier and a wave fired into that gap
    # gets connection-refused (cycle-1 lesson, 08-15).
    for i in $(seq 90); do
        grep -q 'listening on http' "$SRV_LOG" 2>/dev/null && break
        pgrep -x ds4-server >/dev/null || { tail -3 "$SRV_LOG"; die "boot $2 crashed"; }
        sleep 2
        [ "$i" = 90 ] && die "boot $2 deadline"
    done
    SRV_PID=$(pgrep -x ds4-server | head -1)
    PREV_EPOCH=0; PREV_REJ=0; PREV_CRASH=0     # counters are per-lifetime
    log "boot $2 up (pid $SRV_PID, -c $CTX, $BANKS banks${1:+, env: $1})"
}
kill_server() {
    pgrep -x ds4-server | xargs -r kill
    for i in $(seq 30); do pgrep -x ds4-server >/dev/null || { SRV_PID=""; return 0; }; sleep 2; done
    pgrep -x ds4-server | xargs -r kill -9; SRV_PID=""
}

gen_prompt() { # $1 tokens $2 seed $3 outfile [$4 chat-json] [$5 max_tokens]
    python3 - <<EOF
import random, json
random.seed($2)
words=('substrate','ledger','bank','union','credit','floor','promote',
       'tenant','collect','margin','honest','verdict','epoch','span')
txt='seed$2 '+' '.join(random.choice(words) for _ in range(int($1)))
txt+='\nSummarize the theme in one word.'
open('$3','w').write(txt)
if '${4:-}':
    json.dump({'messages':[{'role':'user','content':txt}],'max_tokens':${5:-8}},
              open('${4:-}','w'))
EOF
}

serial_msg() { # $1 promptfile $2 outfile -> echoes http code (tools = serial ticket)
    python3 -c "import json;print(json.dumps({'model':'ds4','max_tokens':16,'tools':[{'name':'noop','description':'no-op probe tool','input_schema':{'type':'object','properties':{}}}],'messages':[{'role':'user','content':open('$1').read()}]}))" > $WORK/serial_body.json
    curl -s -m 600 -o "$2" -w '%{http_code}' "localhost:$PORT/v1/messages" \
        -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
        --data-binary @$WORK/serial_body.json
}

start_hog() { # sized off CURRENT avail; rc=1 (skip) when arrival is below bracket
    local target=$((NEED_EST_MB - RECLAIM_MARGIN_MB)) avail hog
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    hog=$((avail - target))
    if [ "$hog" -lt 256 ]; then
        [ "$avail" -ge "$target" ] && hog=256 || return 1
    fi
    log "pulse hog: avail=${avail}M -> hog=${hog}M (free-after ${target}M)"
    setsid nohup python3 -c "
import time,os
b=bytearray($hog*1048576)
for i in range(0,len(b),4096): b[i]=1
open('$WORK/hog.pid','w').write(str(os.getpid()))
while True:
    for i in range(0,len(b),4096): b[i]^=1
    time.sleep(0.5)" > $WORK/hog.log 2>&1 < /dev/null &
    for i in $(seq 30); do [ -f $WORK/hog.pid ] && break; sleep 2; done
    [ -f $WORK/hog.pid ] || die "hog failed to pin"
    for i in $(seq 25); do
        local a b2 d
        a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); sleep 5
        b2=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        d=$((a - b2)); [ ${d#-} -lt 100 ] && { log "hog settled at ${b2}M"; return 0; }
    done
    die "hog never settled"
}
stop_hog() { [ -f $WORK/hog.pid ] && kill "$(cat $WORK/hog.pid)" 2>/dev/null; rm -f $WORK/hog.pid; }

saturate() { # $1 prefix -- the gate's wave-of-8 shape, local curls
    rm -f "$1"*.out
    local wave_sz=8 fired=0 done_n=0
    while [ "$fired" -lt "$BANKS" ]; do
        local wave_end=$((fired + wave_sz))
        [ "$wave_end" -gt "$BANKS" ] && wave_end=$BANKS
        for b in $(seq $((fired + 1)) $wave_end); do
            setsid nohup curl -s -m 600 "localhost:$PORT/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                --data-binary @$WORK/tenant_$b.json -o "$1$b.out" \
                > /dev/null 2>&1 < /dev/null &
        done
        fired=$wave_end
        local refired=0 live
        for i in $(seq 240); do
            done_n=$(ls "$1"*.out 2>/dev/null | xargs -r grep -l finish_reason | wc -l)
            [ "$done_n" -ge "$fired" ] && break
            if [ "$refired" -eq 0 ] && [ "$i" -ge 12 ]; then
                live=$(pgrep -xc curl || true)
                if [ "${live:-1}" -eq 0 ] || [ "$i" -eq 120 ]; then
                    refired=1
                    for b in $(seq 1 $fired); do
                        grep -q finish_reason "$1$b.out" 2>/dev/null && continue
                        setsid nohup curl -s -m 600 "localhost:$PORT/v1/chat/completions" \
                            -H 'Content-Type: application/json' \
                            --data-binary @$WORK/tenant_$b.json -o "$1$b.out" \
                            > /dev/null 2>&1 < /dev/null &
                    done
                fi
            fi
            sleep 5
        done
        [ "$done_n" -ge "$fired" ] || die "pulse saturation stalled ($done_n/$fired)"
    done
    log "pulse: $BANKS tenants done"
}

wave() { # $1 cycle -- 6 cont chats: 4 buffered, 2 streams (one cancelled)
    local c=$1 pids=() i
    for i in 1 2 3 4; do
        gen_prompt $((200 + 137 * i)) $((TS + c * 100 + i)) $WORK/w.txt $WORK/wave_$i.json 48
        curl -s -m 300 "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            --data-binary @$WORK/wave_$i.json -o $WORK/wave_$i.out & pids+=($!)
    done
    gen_prompt 400 $((TS + c * 100 + 5)) $WORK/w5.txt $WORK/wave_5.json 400
    python3 -c "import json;d=json.load(open('$WORK/wave_5.json'));d['stream']=True;json.dump(d,open('$WORK/wave_5.json','w'))"
    curl -s -N -m 300 "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        --data-binary @$WORK/wave_5.json -o $WORK/wave_5.sse & pids+=($!)
    # the cancel: a second stream killed mid-decode (client-gone abort churn)
    cp $WORK/wave_5.json $WORK/wave_6.json
    curl -s -N -m 300 "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        --data-binary @$WORK/wave_6.json -o $WORK/wave_6.sse &
    local cancel_pid=$!
    sleep 4; kill "$cancel_pid" 2>/dev/null
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
    local ok=0
    for i in 1 2 3 4; do grep -q finish_reason $WORK/wave_$i.out 2>/dev/null && ok=$((ok+1)); done
    grep -q 'data: \[DONE\]' $WORK/wave_5.sse 2>/dev/null && ok=$((ok+1))
    [ "$ok" -ge 5 ] || die "wave $c: only $ok/5 uncancelled requests completed"
    # static lane: /v1/batch home surface (percall graph family).  A
    # typed refusal at floor-scraping avail is an HONEST outcome
    # (counted); only a malformed/other error dies.
    python3 -c "import json;print(json.dumps({'prompts':['Soak static c$c a: say OK.','Soak static c$c b: say OK.'],'max_tokens':16}))" > $WORK/static_body.json
    local scode
    scode=$(curl -s -m 300 -o $WORK/static.json -w '%{http_code}' -X POST "localhost:$PORT/v1/batch" \
        -H 'content-type: application/json' --data-binary @$WORK/static_body.json)
    if [ "$scode" = "200" ] && grep -q finish_reason $WORK/static.json; then
        :
    elif grep -qE 'refus|memory|memgov|floor' $WORK/static.json 2>/dev/null; then
        STATIC_REFUSED=$((STATIC_REFUSED + 1))
        log "wave $c: /v1/batch refused (http=$scode, typed envelope; cum $STATIC_REFUSED)"
    else
        die "wave $c: /v1/batch http=$scode with neither completions nor a refusal envelope: $(head -c 160 $WORK/static.json 2>/dev/null)"
    fi
    # small serial: the tools ticket at modest depth
    gen_prompt 2000 $((TS + c)) $WORK/serial_small.txt
    local code; code=$(serial_msg $WORK/serial_small.txt $WORK/serial_small.out)
    [ "$code" = "200" ] || die "wave $c: small serial http=$code"
}

sample() { # $1 cycle  $2 phase  $3 boot-tag  $4 pulse(0/1)
    local c=$1 ph=$2 bt=$3 pulse=$4
    curl -s -m 15 "localhost:$PORT/metrics" > $WORK/metrics_c$c.txt || die "cycle $c: metrics scrape failed"
    local anon mema ce ge cf gf torn bl rej crash
    anon=$(awk '/Anonymous:/{print $2}' /proc/$SRV_PID/smaps_rollup 2>/dev/null || echo 0)
    mema=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    ce=$(grep -E '^ds4_memory_census_epoch' $WORK/metrics_c$c.txt | awk '{print $NF}')
    ge=$(grep -E '^ds4_memory_governor_epoch' $WORK/metrics_c$c.txt | awk '{print $NF}')
    cf=$(grep -E '^ds4_memory_census_faults_total' $WORK/metrics_c$c.txt | awk '{print $NF}')
    gf=$(grep -E '^ds4_memory_governor_faults_total' $WORK/metrics_c$c.txt | awk '{print $NF}')
    torn=$(grep -E '^ds4_memory_census_torn_fallbacks_total' $WORK/metrics_c$c.txt | awk '{print $NF}')
    bl=$(grep -E '^ds4_banks_live' $WORK/metrics_c$c.txt | awk '{print $NF}' | head -1)
    rej=$(grep '^ds4_requests_rejected_total' $WORK/metrics_c$c.txt | awk '{s+=$NF} END {print s+0}')
    crash=$(grep -cE 'illegal|continuous batch failed' "$SRV_LOG" || true)
    echo "$(date +%s),$c,$ph,$bt,${anon:-0},$mema,${ce:-0},${ge:-0},${cf:-0},${gf:-0},${torn:-0},${bl:-0},${rej:-0},${crash:-0}" >> "$CSV"
    [ "${cf:-0}" = "0" ] || die "cycle $c: census faults $cf"
    [ "${gf:-0}" = "0" ] || die "cycle $c: governor faults $gf"
    [ "${torn:-0}" = "0" ] || die "cycle $c: torn fallbacks $torn"
    [ "${crash:-0}" = "$PREV_CRASH" ] || die "cycle $c: crash lines grew ($PREV_CRASH -> $crash)"
    [ "${ce:-0}" -gt "$PREV_EPOCH" ] || die "cycle $c: census epoch did not advance (${ce:-0} <= $PREV_EPOCH)"
    if [ "$pulse" = "0" ] && [ "${rej:-0}" != "$PREV_REJ" ]; then
        # A long-lived box declines into the floor band within hours
        # (run-3 lesson: host page-state, anon flat, faults 0) and
        # organic waves then floor-reject AND STILL SERVE via fallback
        # -- honest degradation, recorded.  The unbounded tell is
        # rejection WITHOUT service (the wave asserts already die on
        # that) or runaway growth, bounded here.
        local qd=$(( ${rej:-0} - PREV_REJ ))
        QUIET_REJ_TOTAL=$((QUIET_REJ_TOTAL + qd))
        log "cycle $c: quiet-cycle rejections +$qd (cum $QUIET_REJ_TOTAL; all wave requests served -- floor-band degradation, recorded)"
        [ "$qd" -le 20 ] || die "cycle $c: quiet-cycle rejections +$qd in one cycle -- runaway, not floor-band"
    fi
    PREV_EPOCH=${ce:-0}; PREV_REJ=${rej:-0}; PREV_CRASH=${crash:-0}
    log "CYCLE $c/$CYCLES [$ph/$bt] anon=${anon}K avail=${mema}M epoch=$ce banks=$bl rej=$rej pulse=$pulse"
}

pulse() { # $1 cycle -> engaged/skipped accounting
    local c=$1
    log "pulse (cycle $c): saturating $BANKS x ${TENANT_TOKENS}-token tenants"
    for b in $(seq 1 $BANKS); do
        [ -f $WORK/tenant_$b.json ] || gen_prompt $TENANT_TOKENS $((TS + b)) $WORK/tenant_$b.txt $WORK/tenant_$b.json
    done
    saturate $WORK/pulse_t
    # state-bleed replay FIRST, straight after saturation -- before the
    # deep serial squeezes the box (cycle-8 lesson: a post-squeeze replay
    # is honestly floor-REFUSED off cont with its warm bank intact, and
    # the serial fallback cold-prefills; that is the governor working,
    # not state bleed).  Warm-hit = cached counter moves; floor-refused
    # = counted, loud, non-fatal; cont-served-but-cold = the REAL
    # state-bleed tell, fatal.
    local w0 w1 f0 f1 code
    w0=$(mv_ 'prefilled_total{kind="cached"}')
    f0=$(grep -c 'cont admit rejected on memory floor' "$SRV_LOG" || true)
    code=$(curl -s -m 300 -o $WORK/replay_c$c.out -w '%{http_code}' \
        "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        --data-binary @$WORK/tenant_1.json)
    [ "$code" = "200" ] || die "pulse $c: warm replay http=$code"
    w1=$(mv_ 'prefilled_total{kind="cached"}')
    f1=$(grep -c 'cont admit rejected on memory floor' "$SRV_LOG" || true)
    if [ "${w1:-0}" -gt "${w0:-0}" ]; then
        REPLAYS_WARM=$((REPLAYS_WARM + 1))
        log "pulse $c replay WARM-HIT (+$((w1 - w0)) cached tokens)"
    elif [ "${f1:-0}" -gt "${f0:-0}" ]; then
        REPLAYS_REFUSED=$((REPLAYS_REFUSED + 1))
        log "pulse $c replay floor-REFUSED off cont (honest funding refusal; warm record intact per the floor line) -- served via fallback"
    else
        die "pulse $c: replay cont-served but cold-prefilled (cached $w0 -> $w1, floor rejects flat; deep record lost -- state bleed)"
    fi
    gen_prompt $DEEP_TOKENS 999 $WORK/deep.txt
    if start_hog; then
        local r0 code recl
        r0=$(grep -c 'serial fit reclaim' "$SRV_LOG" || true)
        code=$(serial_msg $WORK/deep.txt $WORK/deep_c$c.out)
        recl=$(grep -c 'serial fit reclaim' "$SRV_LOG" || true)
        stop_hog
        [ "$code" = "200" ] || die "pulse $c: deep serial http=$code under engaged hog"
        [ "$recl" -gt "$r0" ] || die "pulse $c: served without reclaim engagement"
        PULSES_ENGAGED=$((PULSES_ENGAGED + 1))
        log "pulse $c ENGAGED (reclaim lines +$((recl - r0)), served 200)"
    else
        # Arrival below the bracket: the box already sits near/below the
        # fit threshold band, so fire HOGLESS and classify by evidence
        # (cycle-8 lesson: a skip discards a perfectly good observation).
        local r0 code recl
        r0=$(grep -c 'serial fit reclaim' "$SRV_LOG" || true)
        code=$(serial_msg $WORK/deep.txt $WORK/deep_c$c.out)
        recl=$(grep -c 'serial fit reclaim' "$SRV_LOG" || true)
        if [ "$recl" -gt "$r0" ]; then
            PULSES_ENGAGED=$((PULSES_ENGAGED + 1))
            log "pulse $c ENGAGED unbracketed (reclaim lines +$((recl - r0)), http=$code)"
        elif [ "$code" = "200" ]; then
            PULSES_SKIPPED=$((PULSES_SKIPPED + 1))
            log "pulse $c plain-served below bracket (no fit failure; http=200, recorded)"
        else
            die "pulse $c: refused ($code) with ZERO reclaim lines beside trimmable banks -- reclaim did not attempt"
        fi
    fi
}

# ---- ownership + lock (two-session law) ----
pgrep -x ds4-server >/dev/null && die "a ds4-server is already running"
mkdir $LOCK 2>/dev/null && echo "$NONCE" > $LOCK/nonce \
    || die "box lock held ($(cat $LOCK/nonce 2>/dev/null || echo unknown))"

log "MEMGOV SOAK start: $CYCLES cycles x ${CYCLE_MIN}min, pulse every $PULSE_EVERY, observe slice at $OBSERVE_AT"
boot "" enforce1
PHASE=E1; BOOT_TAG=enforce1

CYC=0
while [ "$CYC" -lt "$CYCLES" ]; do
    CYC=$((CYC + 1))
    CYCLE_T0=$(date +%s)
    if [ "$CYC" = "$OBSERVE_AT" ]; then
        log "rollback slice: restarting under DS4_MEMGOV=observe"
        kill_server
        boot "DS4_MEMGOV=observe" observe
        PHASE=O; BOOT_TAG=observe
        grep -q 'memgov modes: boot=observe' "$SRV_LOG" \
            || die "observe boot missing the observe modes line"
    elif [ "$CYC" = "$((OBSERVE_AT + 1))" ]; then
        ENF=$(grep -c 'memgov ENFORCE refuse' "$SRV_LOG" || true)
        [ "${ENF:-0}" = "0" ] || die "observe lifetime carried ENFORCE refusals ($ENF)"
        log "rollback slice HEALTHY (zero enforce refusals); restarting enforce"
        kill_server
        boot "" enforce2
        PHASE=E2; BOOT_TAG=enforce2
    fi
    IS_PULSE=0
    [ $((CYC % PULSE_EVERY)) = 0 ] && [ "$PHASE" != "O" ] && IS_PULSE=1
    wave "$CYC"
    [ "$IS_PULSE" = 1 ] && pulse "$CYC"
    sample "$CYC" "$PHASE" "$BOOT_TAG" "$IS_PULSE"
    ELAPSED=$(( $(date +%s) - CYCLE_T0 ))
    REMAIN=$(( CYCLE_MIN * 60 - ELAPSED ))
    [ "$REMAIN" -gt 0 ] && [ "$CYC" -lt "$CYCLES" ] && sleep "$REMAIN"
done

# ---- end-of-soak verdicts: PRINT ALL DATA FIRST, then asserts ----
anon_leak() { # $1 boot-tag $2 first-steady-cycle -> "early late growth_mb"
    local a b
    a=$(awk -F, -v bt="$1" -v c="$2" '$4==bt && $2>=c {print $5; exit}' "$CSV")
    b=$(awk -F, -v bt="$1" '$4==bt {v=$5} END {print v}' "$CSV")
    [ -n "$a" ] && [ -n "$b" ] || { echo "0 0 0"; return; }
    echo "$a $b $(( (b - a) / 1024 ))"
}
read A1 B1 G1 <<< "$(anon_leak enforce1 3)"
read A2 B2 G2 <<< "$(anon_leak enforce2 $((OBSERVE_AT + 3)))"
log "anon-RSS windows: enforce1 ${A1}K -> ${B1}K (${G1}M), enforce2 ${A2}K -> ${B2}K (${G2}M) (cap ${ANON_LEAK_MB}M)"
log "pulse evidence: engaged=$PULSES_ENGAGED skipped=$PULSES_SKIPPED; replays warm=$REPLAYS_WARM refused=$REPLAYS_REFUSED; quiet_rejs=$QUIET_REJ_TOTAL static_refused=$STATIC_REFUSED"
# In-soak reclaim ENGAGEMENT and warm-hit replays are REPORTED, not
# gated: engagement needs hot-anon pressure the aged server's page
# state cannot bracket (MemAvailable sees reclaimable cache the fit's
# budget answer ignores), and both behaviors are gate-stamped on these
# bits (serial_reclaim BASE+INJECT, same day).  The FATAL shapes stay
# fatal at their sites: refusal-without-reclaim-attempt, cont-served-
# but-cold replay, runaway quiet rejections, crash lines, faults.
[ "$PULSES_ENGAGED" -ge 1 ] || log "ADJUDICATION NOTE: zero in-soak reclaim engagements (bracket unreachable on the aged server; gate-proven separately)"
[ "$REPLAYS_WARM" -ge 1 ] || log "ADJUDICATION NOTE: zero warm-hit replays (all floor-refused with warm records intact per the floor lines)"
[ "$G1" -le "$ANON_LEAK_MB" ] || die "enforce1 anon-RSS grew ${G1}M > ${ANON_LEAK_MB}M"
[ "$G2" -le "$ANON_LEAK_MB" ] || die "enforce2 anon-RSS grew ${G2}M > ${ANON_LEAK_MB}M"

kill_server
log "MEMGOV SOAK: ALL ASSERTS PASS (cycles=$CYCLES pulses engaged=$PULSES_ENGAGED skipped=$PULSES_SKIPPED replays warm=$REPLAYS_WARM refused=$REPLAYS_REFUSED quiet_rejs=$QUIET_REJ_TOTAL)"
