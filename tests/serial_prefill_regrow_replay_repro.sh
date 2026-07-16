#!/usr/bin/env bash
# Captured-graph dangling-scratch regression repro (2026-06-11 serial-conc crash).
#
# Bug class: the captured decode graphs (layer / dense-vec / routed-MoE)
# bake the cuda_tmp_alloc sticky scratch pointer into their kernel-node
# args at capture time.  cuda_tmp_alloc grows by cudaFree(old) +
# cudaMalloc(new), and prefill-side requests scale with prompt length --
# so a full re-prefill LONGER than every prior prefill (after a live-KV
# token-mismatch reset) freed the pointer every cached graph still held,
# and the FIRST decode after that prefill died with a CUDA illegal
# memory access.  Fix: ds4_cuda_invalidate_captured_graphs() drops all
# cached execs whenever a sticky scratch buffer is about to be freed for
# growth (see cuda_tmp_alloc / fp8_predecode_scratch_alloc).
#
# Minimal trigger (single-threaded; concurrency and --mtp are NOT needed,
# both were red herrings in the original report):
#   1. conversation A turn 1   -> prefill N tokens, decode a few (graphs
#                                 are captured here, baking the scratch
#                                 sized by the N-token prefill)
#   2. conversation B turn 1   -> live-KV token mismatch -> reset ->
#                                 re-prefill N (same size: no resize)
#   3. conversation A turn 2   -> reset -> re-prefill M > N (scratch
#                                 RESIZES) -> first decode replays the
#                                 captured graphs.  Broken build: illegal
#                                 memory access.  Fixed build: one
#                                 "ds4: invalidated ... captured graph(s)"
#                                 line, then a normal generation.
#
# PASS = all three turns finish=stop and the server log carries no
# "illegal memory" line.  Needs a real model; defaults match the GB10
# weight-server setup (see memory note gb10-ds4-build-run-loop).
#
# Usage:
#   DS4_GGUF=... [DS4_MANIFEST=...] [PORT=...] tests/serial_prefill_regrow_replay_repro.sh
set -uo pipefail

GGUF=${DS4_GGUF:-/home/ent/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
MAN=${DS4_MANIFEST:-/tmp/ds4_weights.manifest}
PORT=${PORT:-8099}
LOG=${LOG:-/tmp/serial_regrow_repro.log}

rm -f "$LOG" /tmp/ds4.lock
ENV=(DS4_SERVER_CONTINUOUS=0)
[ -f "$MAN" ] && ENV+=(DS4_CUDA_WEIGHT_IPC_SCOPE=base DS4_CUDA_WEIGHT_IPC_MANIFEST="$MAN")
env "${ENV[@]}" ./ds4-server -m "$GGUF" --no-spec --cuda -c 4096 --port "$PORT" >"$LOG" 2>&1 &
SPID=$!
n=0
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
  sleep 2; n=$((n+1))
  if [ $n -ge 150 ] || ! kill -0 "$SPID" 2>/dev/null; then
    echo "FAIL: server did not come up"; tail -5 "$LOG"
    kill -9 "$SPID" 2>/dev/null; exit 1
  fi
done

python3 - "$PORT" <<'EOF'
import json, sys, urllib.request
URL = "http://127.0.0.1:%s/v1/chat/completions" % sys.argv[1]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]

def turn(messages):
    body = {"model": "ds4", "temperature": 0, "max_tokens": 64, "stream": True,
            "reasoning_effort": "none", "tools": TOOLS, "messages": messages}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content, finish = [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                obj = json.loads(line[5:].strip())
            except ValueError:
                continue
            for ch in obj.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    content.append(ch["delta"]["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return "".join(content), finish

A = [{"role": "user", "content": "Conversation 1. Count from 10 to 15, "
     "comma-separated, one line, no other text."}]
c, fin = turn(A)
print("A:t1 finish=%s" % fin)
ok = fin == "stop"
A.append({"role": "assistant", "content": c})

B = [{"role": "user", "content": "Conversation 2. Count from 20 to 25, "
     "comma-separated, one line, no other text."}]
c, fin = turn(B)
print("B:t1 finish=%s" % fin)
ok = ok and fin == "stop"

A.append({"role": "user", "content": "Continue with the next 6 numbers, same format."})
c, fin = turn(A)   # re-prefill longer than every prior prefill, then decode
print("A:t2 finish=%s" % fin)
ok = ok and fin == "stop"
sys.exit(0 if ok else 1)
EOF
GRC=$?
kill -TERM "$SPID" 2>/dev/null; sleep 2; kill -9 "$SPID" 2>/dev/null
wait "$SPID" 2>/dev/null

ILL=$(grep -acE "illegal memory" "$LOG" || true)
if [ "$GRC" -eq 0 ] && [ "$ILL" -eq 0 ]; then
  echo "PASS: regrow + replay survived (invalidations: $(grep -ac 'invalidated.*captured graph' "$LOG" || true))"
  exit 0
fi
echo "FAIL: client_rc=$GRC illegal_lines=$ILL (log: $LOG)"
grep -aE "illegal memory|finish=error" "$LOG" | head -4
exit 1
