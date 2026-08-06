#!/bin/bash
# serial_reclaim_gate.sh — deepmem lite-4: the serial lane COLLECTS from the
# commons instead of 503ing beside idle cache.
#
# Chartered by ds4-memory-governance-design-2026-08-03.md (RECLAIM section);
# ships with commit 33ac89f.  Mechanism: serial_session_ensure_fit's refusal
# path calls ds4_batch_ctx_trim_free (context-owned, NULL-tolerant victims),
# re-runs the settle poll, refuses only if the graph still cannot fit.
#
# LANE: the serial route is driven by API FLAVOR (Anthropic /v1/messages
# routes serial at any length -- job_is_batchable is OpenAI-only until API
# plan Inc 3+), NOT by prompt length: at -c 32768 every prompt that exceeds
# seq_cap also exceeds the serial session, so there is no oversized-but-
# servable length (first-run lesson).  Lane oracle: the serial 'prompt
# start' log marker.
#
# SIZING (measured 08-06 on .33, zero-config -c 32768 boot, MTP+DSpark
# armed -- the fit estimate includes their scaffolding and the probe path
# is hard-quiet loud=false, so the threshold was bracketed BEHAVIORALLY
# with SETTLED hot-hog probes): need_min(28K prompt) threshold T is in
# (6.02, 6.87) GiB -- refused at settled 6.02, served at settled 6.87.
# Hog settle drift is +-0.5 GiB, so the window is sized to be drift-proof
# with 24 saturated banks (~1.9 GiB trimmable, measured 78.75 MiB/bank at
# ~31K-token tenants):
#     free-after = NEED_EST_MB - RECLAIM_MARGIN_MB = 5.5 GiB
#     drift band (5.0, 6.0)  -> always BELOW T  -> fit fails
#     post-trim  (6.9, 7.9)  -> always ABOVE T  -> re-poll serves
# For the fit probe to FAIL the sign is:
#     hog = MemAvailable - NEED_EST_MB + RECLAIM_MARGIN_MB
# start_hog WAITS FOR SETTLE (two stable reads) before returning -- the
# box has 16 GiB swap and MemAvailable keeps moving for ~30 s after the
# pin; firing unsettled makes the verdict a coin flip (runs 3-5).
# Re-bracket with two settled probes on a measurement boot if the config
# changes (the 503 refusal line carries need_min in TOKENS only).
#
# Legs:
#   A (reclaim): Anthropic request under hog pressure SERVES (200) with
#      engagement proven: +1 'prompt start' (lane), +1 'serial fit reclaim',
#      >=1 trim line carrying 'for the serial lane (reclaim)'.
#   C (recovery): a cont chat request then admits clean on trimmed banks.
#   B (control): DS4_BATCH_VMM_TRIM=0, same shape -> the old 503 +
#      'no graph fits', ZERO reclaim lines.  Causality.
#
# Remote-wait discipline: every detached remote command goes through
# fire_remote() (local-background ssh + grace kill -- the bare
# `ssh '... & exit 0'` pattern hangs the local ssh nondeterministically).
set -uo pipefail
HOST=${HOST:-sync-192_168_88_33}
PORT=${PORT:-8000}
NEED_EST_MB=${NEED_EST_MB:-5900}
RECLAIM_MARGIN_MB=${RECLAIM_MARGIN_MB:-400}
TENANT_TOKENS=${TENANT_TOKENS:-31000}
PROMPT_TOKENS=${PROMPT_TOKENS:-28000}
CTX=32768
BANKS=24
TS=$(date +%s)
LOCK="/tmp/ds4_box_lock"
NONCE="reclaim_${TS}_$$"
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "FAIL: $*"; cleanup; exit 1; }

remote() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }

# Detached remote command that cannot wedge the driver: fire the ssh in
# LOCAL background, give it a grace window, then kill the local ssh (the
# remote setsid child survives -- selfload law).
fire_remote() { # $1 remote command string
    ssh -o ConnectTimeout=10 "$HOST" "$1" > /dev/null 2>&1 &
    local pid=$!
    for i in $(seq 12); do kill -0 $pid 2>/dev/null || return 0; sleep 1; done
    kill $pid 2>/dev/null
    return 0
}

cleanup() {
    remote 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    remote '[ -f /tmp/reclaim_hog.pid ] && kill $(cat /tmp/reclaim_hog.pid) 2>/dev/null; rm -f /tmp/reclaim_hog.pid' 2>/dev/null
    remote "rm -rf $LOCK" 2>/dev/null
    log "cleanup: server + hog killed on $HOST, lock removed"
}

# ---- box ownership (two-session law) ----
remote "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held ($(remote cat $LOCK/nonce 2>/dev/null || echo unknown))"
remote 'ps -eo comm | grep -q "^ds4-server" && exit 1 || exit 0' \
    || die "a ds4-server is already running on $HOST"
trap cleanup EXIT

gen_prompt() { # $1 tokens  $2 seed  $3 outfile(remote)  [$4 chat-json outfile]
    # Also pre-writes the full chat JSON payload when $4 is given: detached
    # curls must post @file -- a process substitution @<(python3 ...) dies
    # with the remote bash session at teardown and the half-fed curl aborts
    # mid-prefill (run-11 lesson: tenants vanish with no .out and the
    # server logs client-gone aborts).
    remote "python3 - <<EOF
import random, json
random.seed($2)
words=('substrate','ledger','bank','union','credit','floor','promote',
       'tenant','collect','margin','honest','verdict','epoch','span')
txt='seed$2 '+' '.join(random.choice(words) for _ in range(int($1)))
txt+='\nSummarize the theme in one word.'
open('$3','w').write(txt)
if '${4:-}':
    json.dump({'messages':[{'role':'user','content':txt}],'max_tokens':8},
              open('${4:-}','w'))
EOF"
}

boot() { # $1 extra_env  $2 log
    fire_remote "cd ~/code/ds4-phase0 && setsid nohup env DS4_SERVER_COALESCE_MAX=$BANKS $1 ./ds4-server -c $CTX > $2 2>&1 < /dev/null & exit 0"
    for i in $(seq 60); do
        remote "grep -q 'persistent batch ctx ready' $2" && return 0
        remote 'ps -eo comm | grep -q "^ds4-server"' || { remote "tail -3 $2"; die "boot crashed"; }
        sleep 2
    done
    die "boot deadline ($2)"
}

kill_server() {
    remote 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill'
    for i in $(seq 30); do
        remote 'ps -eo comm | grep -q "^ds4-server"' || return 0
        sleep 2
    done
    log "WARN: server still draining after 60s"
}

start_hog() { # $1 log-for-sizing-message
    local avail hog
    avail=$(remote "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
    hog=$((avail - NEED_EST_MB + RECLAIM_MARGIN_MB))
    [ "$hog" -gt 1000 ] || die "hog sizing degenerate (avail=$avail need_est=$NEED_EST_MB)"
    log "MemAvailable=${avail}M -> hog=${hog}M (free-after ~= need_est - margin)"
    # The box carries 16 GiB of swap and kswapd evicts a cold slept-on hog
    # within ~1 min (MemAvailable creeps back over the threshold and the
    # fit probe passes -- run-4 lesson).  Keep the pages HOT: continuous
    # re-touch loop, ~1-2 s/pass at 4K stride, so swap-out never sticks.
    fire_remote "setsid nohup python3 -c \"
import time,os
b=bytearray($hog*1048576)
for i in range(0,len(b),4096): b[i]=1
open('/tmp/reclaim_hog.pid','w').write(str(os.getpid()))
while True:
    for i in range(0,len(b),4096): b[i]^=1
    time.sleep(0.5)\" > /tmp/reclaim_hog.log 2>&1 < /dev/null & exit 0"
    for i in $(seq 30); do
        remote '[ -f /tmp/reclaim_hog.pid ]' && break
        sleep 2
    done
    remote '[ -f /tmp/reclaim_hog.pid ]' || die "hog failed to pin"
    # settle: fire only once MemAvailable is stable (two reads +-100 MiB)
    for i in $(seq 25); do
        local a b2 d
        a=$(remote "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"); sleep 5
        b2=$(remote "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
        d=$((a - b2))
        [ ${d#-} -lt 100 ] && { log "hog settled at ${b2}M free"; return 0; }
    done
    die "hog never settled"
}

stop_hog() {
    remote '[ -f /tmp/reclaim_hog.pid ] && kill $(cat /tmp/reclaim_hog.pid) 2>/dev/null; rm -f /tmp/reclaim_hog.pid'
}

fire_tenant() { # $1 outfile-prefix  $2 tenant-index
    # THREE rules: fire_remote protects the LOCAL ssh from wedging;
    # setsid nohup detaches the REMOTE curl from session teardown
    # (0b-4 law); the payload is a pre-written FILE because a
    # process substitution dies with the remote bash and the
    # half-fed curl aborts mid-prefill (run-11 lesson).
    fire_remote "setsid nohup curl -s -m 600 localhost:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' \
      --data-binary @/tmp/reclaim_t$2.json \
      -o $1$2.out > /dev/null 2>&1 < /dev/null & exit 0"
}

saturate() { # $1 outfile-prefix
    # Waves of 8: prefill aggregation past width 8 hits the small16-tier
    # cliff (knobs-k5 law; measured here as ~350 tok/s at width 24 vs
    # ~2400 at width 8) -- 24-wide saturation runs 3x slower than three
    # 8-wide waves reaching the same final bank state.
    remote "rm -f $1*.out"   # stale outs from a prior run corrupt arrival counting
    local wave_sz=8 fired=0 done_n=0
    while [ "$fired" -lt "$BANKS" ]; do
        local wave_end=$((fired + wave_sz))
        [ "$wave_end" -gt "$BANKS" ] && wave_end=$BANKS
        for b in $(seq $((fired + 1)) $wave_end); do
            fire_tenant "$1" "$b"
        done
        fired=$wave_end
        # Waiter with ONE refire: a detached curl can die client-side
        # mid-run (run 13: one vanished ~4 min into decode -- no OOM, no
        # .out, server logged the client-gone abort; ~1/32 firings).  The
        # bank keeps the prompt as a warm record, so a refire warm-hits
        # and completes in seconds.  Refire when the box runs out of live
        # curls early (>=60 s in, launches are long past) or at the old
        # 600 s deadline, whichever comes first.
        local refired=0 live
        for i in $(seq 240); do
            done_n=$(remote "ls $1*.out 2>/dev/null | xargs -r grep -l finish_reason | wc -l")
            [ "$done_n" -ge "$fired" ] && break
            if [ "$refired" -eq 0 ] && [ "$i" -ge 12 ]; then
                live=$(remote "ps -eo comm | grep -c '^curl'" || true)
                if [ "${live:-1}" -eq 0 ] || [ "$i" -eq 120 ]; then
                    refired=1
                    for b in $(seq 1 $fired); do
                        remote "grep -l finish_reason $1$b.out >/dev/null 2>&1" && continue
                        log "refire tenant $b ($done_n/$fired complete, live_curls=${live:-?})"
                        fire_tenant "$1" "$b"
                    done
                fi
            fi
            sleep 5
        done
        [ "$done_n" -ge "$fired" ] || die "wave stalled: $done_n/$fired tenants completed (arrival, refires=$refired)"
        log "wave done ($done_n/$BANKS)"
    done
    log "tenants done ($done_n/$BANKS)"
}

anthropic_deep() { # $1 outfile -> echoes http code
    remote "curl -s -m 600 -o $1 -w '%{http_code}' localhost:$PORT/v1/messages \
      -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
      --data-binary @<(python3 -c \"import json;print(json.dumps({'model':'ds4','max_tokens':16,'messages':[{'role':'user','content':open('/tmp/reclaim_deep.txt').read()}]}))\")"
}

count() { remote "grep -c \"$1\" $2 || true"; }

for b in $(seq 1 $BANKS); do
    gen_prompt $TENANT_TOKENS $((TS+b)) /tmp/reclaim_t$b.txt /tmp/reclaim_t$b.json
done
gen_prompt $PROMPT_TOKENS 999 /tmp/reclaim_deep.txt

# ---- LEG A ----
log "boot A (stock, $BANKS banks pinned)"
boot "" /tmp/reclaim_A.log
log "saturating $BANKS banks with ${TENANT_TOKENS}-token tenants"
saturate /tmp/reclaim_ta
start_hog A
log "firing Anthropic serial request (${PROMPT_TOKENS} tokens; lane oracle = 'prompt start')"
CODE=$(anthropic_deep /tmp/reclaim_deep.out)
PSTART=$(count "prompt start" /tmp/reclaim_A.log)
RECL=$(count "serial fit reclaim" /tmp/reclaim_A.log)
TRIM=$(count "for the serial lane (reclaim)" /tmp/reclaim_A.log)
log "leg A: http=$CODE prompt_start=$PSTART reclaim_lines=$RECL trim_lines=$TRIM"
# Assert ORDER matters (run-5 lesson): reclaim engagement first.  'prompt
# start' prints only inside generate_job AFTER a successful fit, so it is
# the SUCCESS-side lane oracle, not a routing check -- a reclaim line
# already proves the serial path ran ensure_fit.
[ "$RECL" -ge 1 ] || die "leg A: fit never failed (no reclaim line) -- lower NEED_EST_MB / raise the hog (lane proof: a cont-served request leaves reclaim impossible AND prompt_start=0)"
[ "$TRIM" -ge 1 ] || die "leg A: reclaim line without trim receipt"
[ "$CODE" = "200" ] || die "leg A: reclaim fired (trim receipt in the A log shows released MiB) but the re-poll still missed -- post-trim free must exceed the need_min threshold; raise TENANT_TOKENS (more trimmable) or NEED_EST_MB (higher free-after)"
[ "$PSTART" -ge 1 ] || die "leg A: served without the serial success marker (unexpected lane)"
log "A PASS (served via reclaim: prompt_start=$PSTART reclaim=$RECL trim=$TRIM)"

# ---- LEG C ----
gen_prompt 200 777 /tmp/reclaim_c.txt
CODE=$(remote "curl -s -m 120 -o /tmp/reclaim_c.out -w '%{http_code}' localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @<(python3 -c \"import json;print(json.dumps({'messages':[{'role':'user','content':open('/tmp/reclaim_c.txt').read()}],'max_tokens':8}))\")")
[ "$CODE" = "200" ] || die "leg C: post-reclaim cont request failed (http $CODE)"
ILL=$(count "illegal\|continuous batch failed" /tmp/reclaim_A.log)
[ "$ILL" = "0" ] || die "leg C: crash lines present ($ILL)"
log "C PASS (cont admits clean on trimmed banks)"
stop_hog
kill_server

# ---- LEG B (control) ----
log "boot B (DS4_BATCH_VMM_TRIM=0 control)"
boot "DS4_BATCH_VMM_TRIM=0" /tmp/reclaim_B.log
saturate /tmp/reclaim_tb
start_hog B
CODE=$(anthropic_deep /tmp/reclaim_bdeep.out)
NOFIT=$(count "no graph fits" /tmp/reclaim_B.log)
RECL=$(count "serial fit reclaim" /tmp/reclaim_B.log)
log "leg B: http=$CODE nofit=$NOFIT reclaim_lines=$RECL"
[ "$CODE" = "503" ] || die "leg B control: expected 503, got $CODE"
[ "$NOFIT" -ge 1 ] || die "leg B control: no 'no graph fits' line"
[ "$RECL" = "0" ] || die "leg B control: reclaim fired with trim disabled"
log "B PASS (control reproduces the old 503, zero reclaim lines)"
stop_hog

log "SERIAL-RECLAIM GATE: ALL LEGS PASS"
