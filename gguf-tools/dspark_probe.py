#!/usr/bin/env python3
"""Offline DSpark accept-length probe (DSpark D2/D3 GO/NO-GO gate).

Consumes a hidden-state trace produced by `ds4-eval` with DS4_DSPARK_DUMP set
(target layers 40/41/42 mean-HC hidden + the gold greedy token per position on
the 2-bit target), plus the ds4 DSpark drafter GGUF and the base GGUF (for the
shared lm_head + token_embd), and reports:

  --mode fusion : main_x = main_norm(main_proj(concat(hc40,hc41,hc42))) stats.
                  Validates D2 (target capture + fusion) -- finite, sane scale.
  --mode accept : the full drafter block forward + Markov refine, measuring the
                  accept length against the target's gold continuation (D3 GO/NO-GO).

The drafter's 3 layers are dequantized to f32 and run in numpy (q8/q4 dequant is
near-lossless, so this slightly *over*-states drafter quality -- conservative
direction is the 2-bit TARGET hidden, which is captured faithfully from ds4).

Usage:
  python dspark_probe.py --trace trace.bin --drafter dspark.gguf --base base.gguf \
        --mode fusion
"""
import argparse, struct, sys
import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT
import gguf.quants as gq

RMS_EPS = 1.0e-6

# ----------------------------------------------------------------- trace reader
class Trace:
    """DS4DSPK1 hidden-state trace: per record (prompt_len, total_len, tokens,
    hc40, hc41, hc42), each hc* is [total_len, n_embd] f32 (mean over HC lanes)."""
    def __init__(self, path):
        with open(path, "rb") as f:
            buf = f.read()
        if buf[:8] != b"DS4DSPK1":
            raise ValueError("bad trace magic")
        off = 8
        n_embd, n_cap, n_rec = struct.unpack_from("<III", buf, off); off += 12
        if n_cap != 3:
            raise ValueError(f"expected 3 capture layers, got {n_cap}")
        self.n_embd = n_embd
        self.records = []
        for _ in range(n_rec):
            prompt_len, total_len = struct.unpack_from("<II", buf, off); off += 8
            toks = np.frombuffer(buf, np.int32, total_len, off); off += 4 * total_len
            hcs = []
            for _ in range(3):
                cnt = total_len * n_embd
                a = np.frombuffer(buf, np.float32, cnt, off).reshape(total_len, n_embd)
                off += 4 * cnt
                hcs.append(a)
            self.records.append(dict(prompt_len=int(prompt_len),
                                     total_len=int(total_len),
                                     tokens=toks.copy(),
                                     hc=[h.copy() for h in hcs]))
        print(f"trace: {n_rec} records, n_embd={n_embd}, "
              f"positions={sum(r['total_len'] for r in self.records)}")

# --------------------------------------------------------------- gguf tensor db
class GGUF:
    """Loads named tensors from a GGUF as f32 numpy arrays in ds4 row-major
    (out-major) layout: a 2D weight W maps x->W@x; a 3D expert stack is
    [n_expert, out, in]."""
    def __init__(self, path, only=None):
        self.r = gguf.GGUFReader(path, "r")
        self.t = {}
        for t in self.r.tensors:
            if only is not None and t.name not in only:
                continue
            self.t[t.name] = t
        self.cache = {}
        self._exp_cache = []  # LRU order of expert keys in self.cache
        self.exp_cache_max = 2400  # ~80GB f32; holds the full 256x3layx3mat working set

    def meta_u32(self, key):
        f = self.r.fields.get(key)
        if f is None:
            return None
        return int(f.parts[f.data[0]][0])

    def has(self, name):
        return name in self.t

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        t = self.t.get(name)
        if t is None:
            raise KeyError(name)
        qt = t.tensor_type
        shape = tuple(int(s) for s in reversed(t.shape))  # ds4 row-major (out-major)
        if qt == QT.F32:
            a = np.asarray(t.data, np.float32)
        elif qt == QT.F16:
            a = np.asarray(t.data, np.float32)
        else:
            a = gq.dequantize(t.data, qt).astype(np.float32)
        a = a.reshape(shape)
        self.cache[name] = a
        return a

    def get_expert(self, name, e):
        """Dequantize ONLY expert slab e of a [n_expert,out,in] quantized stack ->
        [out,in] f32. Lazy (full f32 stack would be ~77 GB). LRU-cached."""
        key = (name, e)
        if key in self.cache:
            return self.cache[key]
        t = self.t[name]
        qt = t.tensor_type
        ne = [int(s) for s in t.shape]                 # gguf order: [in, out, n_expert]
        in_dim, out_dim = ne[0], ne[1]
        block_elems, type_size = gguf.GGML_QUANT_SIZES[qt]
        row_bytes = (in_dim // block_elems) * type_size
        expert_bytes = out_dim * row_bytes
        raw = np.asarray(t.data).view(np.uint8).reshape(-1)
        sl = raw[e * expert_bytes:(e + 1) * expert_bytes].reshape(out_dim, row_bytes)
        w = gq.dequantize(sl, qt).astype(np.float32).reshape(out_dim, in_dim)
        if len(self._exp_cache) >= self.exp_cache_max:
            self.cache.pop(self._exp_cache.pop(0), None)
        self._exp_cache.append(key)
        self.cache[key] = w
        return w

# ------------------------------------------------------------------ primitives
def rms_norm(x, w, eps=RMS_EPS):
    # x: [..., n], w: [n]
    ss = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ss + eps))).astype(np.float32) * w

# ------------------------------------------------------------------- D2: fusion
def compute_main_x(draft, hc40, hc41, hc42):
    """main_x = main_norm(main_proj @ concat(hc40,hc41,hc42)).  Each hc* is
    [n,n_embd]; returns [n,n_embd]."""
    W = draft.get("dspark.main_proj.weight")   # [n_embd, 3*n_embd]
    norm_w = draft.get("dspark.main_norm.weight")
    concat = np.concatenate([hc40, hc41, hc42], axis=-1)  # [n, 3*n_embd]
    proj = concat @ W.T                          # [n, n_embd]
    return rms_norm(proj, norm_w)

def mode_fusion(args, trace, draft):
    W = draft.get("dspark.main_proj.weight")
    print(f"main_proj shape (out,in) = {W.shape}; main_norm = "
          f"{draft.get('dspark.main_norm.weight').shape}")
    all_rms = []
    for ri, rec in enumerate(trace.records):
        mx = compute_main_x(draft, rec["hc"][0], rec["hc"][1], rec["hc"][2])
        finite = np.isfinite(mx).all()
        per_pos_rms = np.sqrt(np.mean(mx ** 2, axis=-1))
        all_rms.append(per_pos_rms)
        # input scale for context
        h_rms = [float(np.sqrt(np.mean(rec["hc"][k] ** 2))) for k in range(3)]
        print(f"  rec {ri}: pos={rec['total_len']} finite={finite} "
              f"main_x rms mean={per_pos_rms.mean():.4f} "
              f"min={per_pos_rms.min():.4f} max={per_pos_rms.max():.4f} | "
              f"hc rms={h_rms[0]:.3f}/{h_rms[1]:.3f}/{h_rms[2]:.3f}")
    rms = np.concatenate(all_rms)
    print(f"\nFUSION: positions={rms.size} finite_all={np.isfinite(rms).all()} "
          f"main_x rms p50={np.median(rms):.4f} p5={np.percentile(rms,5):.4f} "
          f"p95={np.percentile(rms,95):.4f}")
    ok = np.isfinite(rms).all() and rms.min() > 1e-3 and rms.max() < 1e3
    print("D2 FUSION", "OK" if ok else "SUSPECT")
    return 0 if ok else 1

# =========================================================================
# D3: faithful drafter forward (ds4 dense MoE layer math) + Markov + accept
# =========================================================================
N_HEAD = 64; HEAD_DIM = 512; N_ROT = 64; N_NOPE = HEAD_DIM - N_ROT
N_LORA_Q = 1024; N_EMBD = 4096; N_HC = 4; N_EXPERT = 256; N_EXP_USED = 6
N_FF = 2048; N_GROUP = 8; GROUP_HEADS = N_HEAD // N_GROUP; GROUP_DIM = HEAD_DIM * GROUP_HEADS
GROUP_RANK = 1024; EXPERT_WEIGHT_SCALE = 1.5; SWIGLU_CLAMP = 10.0
ROPE_FREQ_BASE = 10000.0; NOISE_TOKEN = 128799
KQ_SCALE = 1.0 / np.sqrt(HEAD_DIM)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def silu(x): return x * sigmoid(x)
def softplus(x): return np.where(x > 20.0, x, np.where(x < -20.0, np.exp(x), np.log1p(np.exp(x))))

def rms_nw(x, eps=RMS_EPS):  # rms-norm, no weight; x[...,n]
    return x * (1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)).astype(np.float32)

# rope theta_scale for adjacent pairs (interleaved NeoX), dense layers
_THETA = ROPE_FREQ_BASE ** (-2.0 / N_ROT)
_THETA_POW = _THETA ** np.arange(N_ROT // 2)             # [32]

def rope_inplace(x, pos, inverse=False):
    """x: [..., n_head, head_dim] rotate last N_ROT dims, adjacent pairs, at position pos."""
    tail = x[..., N_NOPE:]                                # [..., head_dim_rot=64]
    theta = float(pos) * _THETA_POW                       # [32]
    c = np.cos(theta).astype(np.float32)
    s = (np.sin(theta) * (-1.0 if inverse else 1.0)).astype(np.float32)
    x0 = tail[..., 0::2]; x1 = tail[..., 1::2]
    tail[..., 0::2] = x0 * c - x1 * s
    tail[..., 1::2] = x0 * s + x1 * c

def hc_split_sinkhorn(mix, scale, base, iters=20, eps=1e-6):
    """mix[24],scale[3],base[24] -> split[24]: [0:4]=pre,[4:8]=post,[8:24]=comb(idx src+dst*4)."""
    out = np.empty(24, np.float32)
    out[0:4] = sigmoid(mix[0:4] * scale[0] + base[0:4]) + eps
    out[4:8] = 2.0 / (1.0 + np.exp(-(mix[4:8] * scale[1] + base[4:8])))
    C = (mix[8:24] * scale[2] + base[8:24]).reshape(4, 4).astype(np.float64)  # C[dst,src], idx=src+dst*4
    C = np.exp(C - C.max(axis=1, keepdims=True)); C = C / C.sum(axis=1, keepdims=True) + eps   # softmax/src per dst
    C = C / (C.sum(axis=0, keepdims=True) + eps)                                                # 1 col-norm (over dst)
    for _ in range(iters - 1):
        C = C / (C.sum(axis=1, keepdims=True) + eps)      # row: per dst over src
        C = C / (C.sum(axis=0, keepdims=True) + eps)      # col: per src over dst
    out[8:24] = C.reshape(-1).astype(np.float32)
    return out

def hc_pre(fn, scale, base, residual_hc):
    """residual_hc[4,4096] -> (attn_cur[4096], post[4], comb[16]). fn[24,16384]."""
    flat = rms_nw(residual_hc.reshape(-1))                # [16384]
    mix = fn @ flat                                       # [24]
    split = hc_split_sinkhorn(mix, scale, base)
    cur = (residual_hc * split[0:4, None]).sum(axis=0)    # hc_weighted_sum over lanes
    return cur, split[4:8].copy(), split[8:24].copy()

def hc_post(block_out, residual_hc, post, comb):
    """-> out_hc[4,4096]; out[dst]=block_out*post[dst]+sum_src comb[dst+src*4]*residual[src]."""
    out = block_out[None, :] * post[:, None]              # [4,4096]
    M = comb.reshape(4, 4)                                # M[a]=comb[a*4 ..]; need comb[dst+src*4]
    # comb[dst+src*4] => index a=src,b=dst of M (M[src,dst]); contribution sum_src M[src,dst]*residual[src]
    out += np.einsum('sd,se->de', M, residual_hc)         # de: dst,embd ; M[src,dst]*residual[src,embd]
    return out

class Drafter:
    def __init__(self, draft, base):
        self.d = draft; self.b = base
        self.L = [self._layer(i) for i in range(3)]
        self.main_proj = draft.get("dspark.main_proj.weight")     # [4096,12288]
        self.main_norm = draft.get("dspark.main_norm.weight")
        self.markov_w1 = draft.get("dspark.markov_w1.weight")     # [vocab,256]
        self.markov_w2 = draft.get("dspark.markov_w2.weight")     # [vocab,256]
        self.hc_head_fn = draft.get("dspark.hc_head_fn.weight")   # [4,16384]
        self.hc_head_base = draft.get("dspark.hc_head_base.weight")
        self.hc_head_scale = draft.get("dspark.hc_head_scale.weight")
        self.out_norm = draft.get("dspark.norm.weight")
        self.lm_head = base.get("output.weight")                  # [vocab,4096]
        self.token_embd = base.get("token_embd.weight")           # [vocab,4096]
        self.vocab = self.lm_head.shape[0]

    def _layer(self, i):
        g = self.d; p = f"dspark.{i}."
        d = {}
        for k in ("hc_attn_fn","hc_attn_scale","hc_attn_base","attn_norm","attn_q_a",
                  "attn_q_a_norm","attn_q_b","attn_kv","attn_kv_a_norm","attn_sinks",
                  "attn_output_a","attn_output_b","hc_ffn_fn","hc_ffn_scale","hc_ffn_base",
                  "ffn_norm","ffn_gate_inp","exp_probs_b","ffn_gate_shexp","ffn_up_shexp",
                  "ffn_down_shexp"):
            nm = p + ("exp_probs_b.bias" if k == "exp_probs_b" else k + ".weight")
            d[k] = g.get(nm)
        d["idx"] = i
        return d

    def embed_hc(self, token):                                    # -> [4,4096]
        e = self.token_embd[token]
        return np.broadcast_to(e, (N_HC, N_EMBD)).copy()

    # --- KV from injected hidden (main_x) for one layer: [n,4096] -> [n,512] (rope'd at pos) ---
    def kv_from_main_x(self, lw, main_x, kv_norm_input):
        x = rms_nw(main_x) * lw["attn_norm"] if kv_norm_input else main_x
        kv = x @ lw["attn_kv"].T                                  # [n,512]
        kv = rms_nw(kv) * lw["attn_kv_a_norm"]
        for q in range(kv.shape[0]):
            rope_inplace(kv[q], q, inverse=False)
        return kv

    def attn_norm_cur(self, lw, residual_hc):
        cur, post, comb = hc_pre(lw["hc_attn_fn"], lw["hc_attn_scale"], lw["hc_attn_base"], residual_hc)
        return rms_nw(cur) * lw["attn_norm"], post, comb

    def q_proj(self, lw, normed):                                 # normed[4096] -> q[64,512]
        qr = normed @ lw["attn_q_a"].T                            # [1024]
        qr = rms_nw(qr) * lw["attn_q_a_norm"]
        q = (qr @ lw["attn_q_b"].T).reshape(N_HEAD, HEAD_DIM)     # [64,512]
        return rms_nw(q)                                          # per-head rms (no weight)

    def kv_proj(self, lw, normed):                                # normed[4096] -> kv[512]
        kv = normed @ lw["attn_kv"].T
        return rms_nw(kv) * lw["attn_kv_a_norm"]

    def grouped_out(self, lw, heads):                             # heads[64,512] -> [4096]
        Wa = lw["attn_output_a"].reshape(N_GROUP, GROUP_RANK, GROUP_DIM)   # [8,1024,4096]
        hg = heads.reshape(N_GROUP, GROUP_DIM)                    # [8,4096]
        low = np.einsum('gor,gr->go', Wa, hg).reshape(-1)        # [8192]
        return lw["attn_output_b"] @ low                          # [4096]

    def moe(self, lw, x):                                         # x[4096] -> [4096]
        logits = lw["ffn_gate_inp"] @ x
        probs = np.sqrt(softplus(logits))
        sel = np.argsort(-(probs + lw["exp_probs_b"]), kind="stable")[:N_EXP_USED]
        wsum = max(probs[sel].sum(), 6.103515625e-5)
        w = probs[sel] / wsum * EXPERT_WEIGHT_SCALE
        out = np.zeros(N_EMBD, np.float32)
        i = lw["idx"]
        for j, e in enumerate(sel):
            gate = self.d.get_expert(f"dspark.{i}.ffn_gate_exps.weight", int(e)) @ x
            up = self.d.get_expert(f"dspark.{i}.ffn_up_exps.weight", int(e)) @ x
            g = np.minimum(gate, SWIGLU_CLAMP)               # gate: upper clamp only
            u = np.clip(up, -SWIGLU_CLAMP, SWIGLU_CLAMP)
            mid = silu(g) * u * w[j]
            out += self.d.get_expert(f"dspark.{i}.ffn_down_exps.weight", int(e)) @ mid
        # shared expert (Q8_0, dequant'd whole)
        sg = lw["ffn_gate_shexp"] @ x; su = lw["ffn_up_shexp"] @ x
        smid = silu(np.minimum(sg, SWIGLU_CLAMP)) * np.clip(su, -SWIGLU_CLAMP, SWIGLU_CLAMP)
        out += lw["ffn_down_shexp"] @ smid
        return out

    def ffn(self, lw, after_attn_hc):                            # [4,4096] -> [4,4096]
        cur, post, comb = hc_pre(lw["hc_ffn_fn"], lw["hc_ffn_scale"], lw["hc_ffn_base"], after_attn_hc)
        normed = rms_nw(cur) * lw["ffn_norm"]
        moe = self.moe(lw, normed)
        return hc_post(moe, after_attn_hc, post, comb)

    def out_head(self, block_hc):                                # [4,4096] -> logits[vocab]
        flat = rms_nw(block_hc.reshape(-1))
        pre = self.hc_head_fn @ flat                             # [4]
        w = sigmoid(pre * self.hc_head_scale[0] + self.hc_head_base) + RMS_EPS
        embd = (block_hc * w[:, None]).sum(axis=0)               # [4096]
        normed = rms_nw(embd) * self.out_norm
        return self.lm_head @ normed                             # [vocab]

def block_forward(drf, main_x_kv, P, bonus, window=128):
    """Run the 5-row drafter block at positions P..P+4 over target-fused prefix KV.
    main_x_kv: list of 3 arrays [total_len,512] (rope'd per-position KV per draft layer).
    Returns base_logits [5, vocab]."""
    B = 5
    toks = [bonus] + [NOISE_TOKEN] * (B - 1)
    block_hc = np.stack([drf.embed_hc(t) for t in toks])         # [5,4,4096]
    lo = max(0, P + (B - 1) - (window - 1))                      # last-row window start
    for li in range(3):
        lw = drf.L[li]
        prefix_kv = main_x_kv[li][lo:P]                          # [<=window, 512]
        # per-row attn pre + norm + q + block kv
        posts = []; combs = []; qs = []; blk_kv = np.empty((B, HEAD_DIM), np.float32)
        for r in range(B):
            normed, post, comb = drf.attn_norm_cur(lw, block_hc[r])
            posts.append(post); combs.append(comb)
            qs.append(drf.q_proj(lw, normed))                    # [64,512] rope'd next
            kvr = drf.kv_proj(lw, normed)
            blk_kv[r] = kvr
        for r in range(B):
            rope_inplace(qs[r], P + r, inverse=False)
            rope_inplace(blk_kv[r], P + r, inverse=False)
        keys = np.concatenate([prefix_kv, blk_kv], axis=0)       # [K,512]  (full intra-block + window)
        sinks = lw["attn_sinks"]
        after_attn = np.empty((B, N_HC, N_EMBD), np.float32)
        for r in range(B):
            q = qs[r]                                            # [64,512]
            scores = (q @ keys.T) * KQ_SCALE                     # [64,K]
            m = np.maximum(scores.max(axis=1), sinks)            # [64]
            w = np.exp(scores - m[:, None])                      # [64,K]
            denom = np.exp(sinks - m) + w.sum(axis=1)            # [64]
            heads = (w @ keys) / denom[:, None]                  # [64,512]
            rope_inplace(heads, P + r, inverse=True)
            attn_out = drf.grouped_out(lw, heads)                # [4096]
            after_attn[r] = hc_post(attn_out, block_hc[r], posts[r], combs[r])
        nb = np.empty_like(block_hc)
        for r in range(B):
            nb[r] = drf.ffn(lw, after_attn[r])
        block_hc = nb
    return np.stack([drf.out_head(block_hc[r]) for r in range(B)])  # [5,vocab]

def markov_refine(drf, base_logits, bonus):
    """base_logits[5,vocab] + bonus -> candidates[5] = [bonus, d1..d4]."""
    out = [bonus]
    for i in range(4):
        prev_embed = drf.markov_w1[out[i]]                       # [256]
        bias = drf.markov_w2 @ prev_embed                        # [vocab]
        out.append(int(np.argmax(base_logits[i] + bias)))
    return out

def mode_accept(args, trace, draft, base):
    drf = Drafter(draft, base)
    kv_norm_input = bool(args.kv_attn_norm)
    max_rec = args.max_records if args.max_records > 0 else len(trace.records)
    accept_lens = []; pos0_hits = 0; pos0_tot = 0
    for ri, rec in enumerate(trace.records[:max_rec]):
        hc = rec["hc"]; toks = rec["tokens"]; total = rec["total_len"]; plen = rec["prompt_len"]
        main_x = compute_main_x(draft, hc[0], hc[1], hc[2])      # [total,4096]
        kv = [drf.kv_from_main_x(drf.L[li], main_x, kv_norm_input) for li in range(3)]
        lo = max(1, plen)                                        # measure on generated region
        hi = total - 2
        step = max(1, args.stride)
        cnt = 0
        for P in range(lo, hi + 1, step):
            if args.max_steps and cnt >= args.max_steps: break
            bonus = int(toks[P])
            base_logits = block_forward(drf, kv, P, bonus)
            cand = markov_refine(drf, base_logits, bonus)
            al = 0
            for j in range(1, 5):
                if P + j < total and cand[j] == int(toks[P + j]): al += 1
                else: break
            accept_lens.append(al)
            pos0_tot += 1; pos0_hits += (1 if (P + 1 < total and cand[1] == int(toks[P + 1])) else 0)
            cnt += 1
        print(f"  rec {ri}: steps={cnt} mean_accept={np.mean(accept_lens[-cnt:]):.3f} "
              f"pos0={pos0_hits}/{pos0_tot}")
    a = np.array(accept_lens, float)
    print(f"\nACCEPT: steps={a.size} pos0_accept={pos0_hits/max(1,pos0_tot):.3f} "
          f"mean_accept_len={a.mean():.3f} mean_commit={a.mean()+1:.3f} "
          f"(dist {[int((a==k).sum()) for k in range(5)]})")
    print(f"D3 {'GO' if a.mean()+1 >= 2.3 else 'NO-GO'} "
          f"(commit {a.mean()+1:.2f} vs break-even 2.3 / ref 3.36)")
    return 0

# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--base", default=None, help="base GGUF (shared lm_head + token_embd)")
    ap.add_argument("--mode", default="fusion", choices=["fusion", "accept"])
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    ap.add_argument("--max-steps", type=int, default=0, help="per-record step cap (0=all)")
    ap.add_argument("--stride", type=int, default=1, help="measure every Nth position")
    ap.add_argument("--kv-attn-norm", action="store_true",
                    help="apply layer attn_norm to main_x before wkv (vs feeding main_x direct)")
    args = ap.parse_args()

    trace = Trace(args.trace)
    draft = GGUF(args.drafter)
    if trace.n_embd * 3 != draft.get("dspark.main_proj.weight").shape[1]:
        print("WARN: main_proj in-dim != 3*n_embd", file=sys.stderr)

    if args.mode == "fusion":
        return mode_fusion(args, trace, draft)
    if not args.base:
        print("--base (base GGUF) required for --mode accept", file=sys.stderr)
        return 2
    base = GGUF(args.base, only={"output.weight", "token_embd.weight"})
    return mode_accept(args, trace, draft, base)

if __name__ == "__main__":
    sys.exit(main())
