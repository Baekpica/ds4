#!/usr/bin/env python3
"""Static lint for the two CUDA concurrency/determinism bug classes ds4 has hit.

  Class A (mmid mm_ids_helper, fixed 2026-06-30): a __shared__ array written by some
          warp lanes and read by OTHER lanes with NO __syncwarp()/__syncthreads()
          between -> post-Volta independent-thread-scheduling visibility hazard.
  Class B (moe_scatter / moe_down, 2026-06-30): a reduction/compaction whose order is
          scheduling-dependent -> nondeterministic FP/order. float atomicAdd, or an
          atomicAdd-cursor scatter feeding an order-sensitive consumer.

Heuristic, so triage the output:
  A!!  cross-lane __shared__ write+read, NO barrier at all   -> highest suspicion (mmid)
  A?   __shared__ + cross-lane __shfl, no __syncwarp          -> review single-warp regions
  B!   atomicAdd(cursor) scatter                              -> nondeterministic order
  B    float atomicAdd                                        -> scheduling-dependent FP order

Usage:  python3 tools/kernel_scan.py          # scans ds4_cuda.cu + cuda/mmq/*
Exit 0 always; this is an advisory lint, not a gate. Validate real hits with
compute-sanitizer racecheck/synccheck (the authoritative dynamic check); see the
weight-server-client recipe in the cont-server-nondeterminism postmortem.
"""
import re, glob, os

# Walk up from this script until we find the repo root (the dir holding ds4_cuda.cu),
# so the lint works regardless of where it is invoked from.
ROOT = os.path.abspath(os.path.dirname(__file__))
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, "ds4_cuda.cu")):
    ROOT = os.path.dirname(ROOT)
FILES = (["ds4_cuda.cu"] +
         sorted(os.path.relpath(p, ROOT) for p in
                glob.glob(os.path.join(ROOT, "cuda/mmq/*.cu")) +
                glob.glob(os.path.join(ROOT, "cuda/mmq/*.cuh"))))

def kernels(text):
    """Split into __global__ kernels by brace counting (handles template/launch_bounds)."""
    out = []
    for m in re.finditer(r'__global__[\s\w\(\),*<>:]*?\bvoid\s+([A-Za-z_]\w*)\s*\(', text):
        i = text.find('{', m.end())
        if i < 0:
            continue
        depth, j = 0, i
        while j < len(text):
            c = text[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((m.group(1), m.start(), text[i:j + 1]))
    return out

def has(body, pat):
    return re.search(pat, body) is not None

FLOAT_ATOMIC = re.compile(r'atomicAdd\s*\(\s*([A-Za-z_]\w*)[^,]*,\s*(?!1u\b|1\b|\(uint)')
INT_COUNTER  = re.compile(r'atomicAdd\s*\(\s*(cursors?|counts?|count|n_\w*|\w*_count\w*|\w*cnt\w*)\b')

flagged, coverage = [], []
for rel in FILES:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        continue
    text = open(path, encoding='utf-8', errors='replace').read()
    ks = kernels(text)
    coverage.append((rel, len(ks)))
    for name, pos, body in ks:
        ln = text.count('\n', 0, pos) + 1
        S  = has(body, r'__shared__')
        W  = has(body, r'__syncwarp')
        T  = has(body, r'__syncthreads')
        Fx = has(body, r'__shfl_(up|down|xor)_sync')
        crosslane = False
        for s in set(re.findall(r'__shared__[^;]*?\b([A-Za-z_]\w*)\s*\[', body)):
            if has(body, re.escape(s) + r'\s*\[[^\]]*\]\s*=') and has(body, r'=\s*[^;]*\b' + re.escape(s) + r'\s*\['):
                crosslane = True
        # strip comments so a comment mentioning atomicAdd doesn't false-positive
        code = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
        code = re.sub(r'//[^\n]*', '', code)
        fa = [a for a in FLOAT_ATOMIC.findall(code) if not INT_COUNTER.search('atomicAdd(' + a + ',')]
        intcur = has(code, r'atomicAdd\s*\(\s*cursors?\b')
        tags = []
        if crosslane and not W and not T:
            tags.append("A!! cross-lane __shared__ write+read, NO barrier")
        elif S and Fx and not W:
            tags.append("A?  __shared__ + cross-lane __shfl, no __syncwarp (review single-warp regions)")
        if intcur:
            tags.append("B!  atomicAdd(cursor) scatter (nondeterministic compaction order)")
        if fa:
            tags.append(f"B   float atomicAdd x{len(fa)} (scheduling-dependent FP order)")
        if tags:
            flagged.append((rel, ln, name, tags, (S, W, T, Fx)))

def rank(f):
    t = ' '.join(f[3])
    return (0 if 'A!!' in t else 1 if 'B!' in t else 2 if 'A?' in t else 3, f[0], f[1])

for rel, ln, name, tags, (S, W, T, Fx) in sorted(flagged, key=rank):
    print(f"{rel}:{ln}  {name}")
    for t in tags:
        print(f"     - {t}")
    print(f"     [shared={int(S)} syncwarp={int(W)} syncthreads={int(T)} shfl={int(Fx)}]")
print(f"\n== {len(flagged)} flagged of {sum(n for _, n in coverage)} kernels "
      f"across {len([c for c in coverage if c[1]])} files ==")
