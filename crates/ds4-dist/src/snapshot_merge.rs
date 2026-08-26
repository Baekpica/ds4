//! Coordinator DSV4 assembly from `ds4_dist_session_save_payload`.
//!
//! Host owns layer-header parse, layout match, tensor-size arithmetic, and
//! write order. Layer GPU bytes and logits stay opaque blobs.

use crate::snapshot_gather::ERR_EMPTY_SHARD;

pub const LAYER_MAGIC: u32 = 0x4C56_5344; /* "DSVL" */
pub const LAYER_VERSION: u32 = 1;
pub const LAYER_U32_FIELDS: usize = 14;
pub const SESSION_MAGIC: u32 = 0x3456_5344; /* "DSV4" */
pub const SESSION_VERSION: u32 = 3;
pub const SESSION_U32_FIELDS: usize = 13;

pub const ERR_UNSUPPORTED_LAYER: &str = "unsupported distributed KV layer payload";
pub const ERR_RANGE: &str = "distributed KV shard range mismatch";
pub const ERR_LAYOUT: &str = "distributed KV shard layout is invalid";
pub const ERR_LAYOUTS_DIFFER: &str = "distributed KV shards use different layouts";
pub const ERR_COMP_ROWS: &str = "distributed KV shard has invalid compressed row count";
pub const ERR_INDEX_ROWS: &str = "distributed KV shard has invalid indexer row count";
pub const ERR_TENSOR_OVERFLOW: &str = "distributed KV shard tensor size overflow";
pub const ERR_TENSOR_BYTES: &str = "distributed KV shard tensor byte count mismatch";
pub const ERR_METADATA: &str = "distributed KV shard metadata mismatch";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistKvLayout {
    pub ctx: u32,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub raw_window: u32,
    pub comp_cap: u32,
    pub token_count: u32,
    pub n_layers: u32,
    pub head_dim: u32,
    pub indexer_head_dim: u32,
    pub vocab: u32,
    pub raw_live: u32,
}

impl DistKvLayout {
    pub fn matches_core(&self, other: &Self) -> bool {
        self.ctx == other.ctx
            && self.prefill_cap == other.prefill_cap
            && self.raw_cap == other.raw_cap
            && self.raw_window == other.raw_window
            && self.comp_cap == other.comp_cap
            && self.token_count == other.token_count
            && self.n_layers == other.n_layers
            && self.head_dim == other.head_dim
            && self.indexer_head_dim == other.indexer_head_dim
            && self.raw_live == other.raw_live
    }

    pub fn raw_live_valid(&self) -> bool {
        if self.raw_window == 0 || self.raw_cap == 0 {
            return false;
        }
        let expected = self.token_count.min(self.raw_window);
        self.raw_live == expected && self.raw_live <= self.raw_cap
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedLayerShard {
    pub layout: DistKvLayout,
    pub layer_start: u32,
    pub layer_end: u32,
    pub n_comp: Vec<u32>,
    pub n_index_comp: Vec<u32>,
    pub tensor: Vec<u8>,
}

fn get_u32(bytes: &[u8]) -> Option<u32> {
    bytes.get(..4)?.try_into().ok().map(u32::from_le_bytes)
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// C `dist_kv_state_bytes`.
pub fn kv_state_bytes(ratio: u32, head_dim: u32) -> Option<u64> {
    let coff = if ratio == 4 { 2u64 } else { 1u64 };
    coff.checked_mul(u64::from(head_dim))?
        .checked_mul(coff)?
        .checked_mul(u64::from(ratio))?
        .checked_mul(4)
}

/// C `dist_kv_layer_tensor_bytes` once the host compress ratio is known.
pub fn layer_tensor_bytes(
    layout: &DistKvLayout,
    ratio: u32,
    n_comp: u32,
    n_index_comp: u32,
) -> Option<u64> {
    let mut bytes = u64::from(layout.raw_live)
        .checked_mul(u64::from(layout.head_dim))?
        .checked_mul(4)?;
    if ratio == 0 {
        return Some(bytes);
    }
    bytes = bytes.checked_add(
        u64::from(n_comp)
            .checked_mul(u64::from(layout.head_dim))?
            .checked_mul(4)?,
    )?;
    let attn = kv_state_bytes(ratio, layout.head_dim)?;
    bytes = bytes.checked_add(attn)?.checked_add(attn)?;
    if ratio == 4 {
        bytes = bytes.checked_add(
            u64::from(n_index_comp)
                .checked_mul(u64::from(layout.indexer_head_dim))?
                .checked_mul(4)?,
        )?;
        let index = kv_state_bytes(ratio, layout.indexer_head_dim)?;
        bytes = bytes.checked_add(index)?.checked_add(index)?;
    }
    Some(bytes)
}

pub fn parse_layer_payload(
    bytes: &[u8],
    expected_start: u32,
    expected_end: u32,
    compress: &[u32],
) -> Result<ParsedLayerShard, String> {
    if bytes.len() < LAYER_U32_FIELDS * 4 {
        return Err(ERR_UNSUPPORTED_LAYER.into());
    }
    let mut h = [0u32; LAYER_U32_FIELDS];
    for (i, slot) in h.iter_mut().enumerate() {
        *slot = get_u32(&bytes[i * 4..]).ok_or(ERR_UNSUPPORTED_LAYER)?;
    }
    if h[0] != LAYER_MAGIC || h[1] != LAYER_VERSION {
        return Err(ERR_UNSUPPORTED_LAYER.into());
    }
    let layout = DistKvLayout {
        ctx: h[2],
        prefill_cap: h[3],
        raw_cap: h[4],
        raw_window: h[5],
        comp_cap: h[6],
        token_count: h[7],
        n_layers: h[8],
        head_dim: h[9],
        indexer_head_dim: h[10],
        vocab: 0,
        raw_live: h[13],
    };
    let layer_start = h[11];
    let layer_end = h[12];
    if layer_start != expected_start
        || layer_end != expected_end
        || layer_start > layer_end
        || layer_end >= layout.n_layers
    {
        return Err(ERR_RANGE.into());
    }
    if !layout.raw_live_valid() {
        return Err(ERR_LAYOUT.into());
    }
    let slice = (layer_end - layer_start + 1) as usize;
    let mut off = LAYER_U32_FIELDS * 4;
    let mut n_comp = vec![0u32; layout.n_layers as usize];
    let mut n_index_comp = vec![0u32; layout.n_layers as usize];
    for i in 0..slice {
        let il = (layer_start as usize) + i;
        let v = get_u32(bytes.get(off..).unwrap_or(&[])).ok_or(ERR_UNSUPPORTED_LAYER)?;
        if v > layout.comp_cap {
            return Err(ERR_COMP_ROWS.into());
        }
        n_comp[il] = v;
        off += 4;
    }
    for i in 0..slice {
        let il = (layer_start as usize) + i;
        let v = get_u32(bytes.get(off..).unwrap_or(&[])).ok_or(ERR_UNSUPPORTED_LAYER)?;
        if v > layout.comp_cap {
            return Err(ERR_INDEX_ROWS.into());
        }
        n_index_comp[il] = v;
        off += 4;
    }
    let mut expected_tensor = 0u64;
    for il in layer_start..=layer_end {
        let ratio = compress.get(il as usize).copied().unwrap_or(0);
        let layer_bytes = layer_tensor_bytes(
            &layout,
            ratio,
            n_comp[il as usize],
            n_index_comp[il as usize],
        )
        .ok_or(ERR_TENSOR_OVERFLOW)?;
        expected_tensor = expected_tensor
            .checked_add(layer_bytes)
            .ok_or(ERR_TENSOR_OVERFLOW)?;
    }
    let tensor = bytes.get(off..).unwrap_or(&[]).to_vec();
    if tensor.len() as u64 != expected_tensor {
        return Err(ERR_TENSOR_BYTES.into());
    }
    Ok(ParsedLayerShard {
        layout,
        layer_start,
        layer_end,
        n_comp,
        n_index_comp,
        tensor,
    })
}

/// C save merge after shards are gathered: header, tokens, logits,
/// n_comp, n_index_comp, then shard tensor tails in walk order.
pub fn merge_session_payload(
    tokens: &[i32],
    logits: &[f32],
    n_layers: u32,
    shards: &[ParsedLayerShard],
    compress: &[u32],
) -> Result<Vec<u8>, String> {
    if shards.is_empty() {
        return Err(ERR_EMPTY_SHARD.into());
    }
    let token_count = u32::try_from(tokens.len()).map_err(|_| ERR_METADATA.to_string())?;
    let vocab = u32::try_from(logits.len()).map_err(|_| ERR_METADATA.to_string())?;
    let mut layout = shards[0].layout;
    layout.vocab = vocab;
    if layout.token_count != token_count || layout.n_layers != n_layers || !layout.raw_live_valid()
    {
        return Err(ERR_METADATA.into());
    }
    let mut n_comp = vec![0u32; n_layers as usize];
    let mut n_index_comp = vec![0u32; n_layers as usize];
    let mut covered = vec![false; n_layers as usize];
    for shard in shards {
        if !layout.matches_core(&shard.layout) {
            return Err(ERR_LAYOUTS_DIFFER.into());
        }
        let mut shard_bytes = 0u64;
        for il in shard.layer_start..=shard.layer_end {
            let i = il as usize;
            if covered[i] {
                return Err(ERR_RANGE.into());
            }
            covered[i] = true;
            n_comp[i] = shard.n_comp[i];
            n_index_comp[i] = shard.n_index_comp[i];
            let ratio = compress.get(i).copied().unwrap_or(0);
            let layer_bytes = layer_tensor_bytes(&layout, ratio, n_comp[i], n_index_comp[i])
                .ok_or(ERR_TENSOR_OVERFLOW)?;
            shard_bytes = shard_bytes
                .checked_add(layer_bytes)
                .ok_or(ERR_TENSOR_OVERFLOW)?;
        }
        if shard.tensor.len() as u64 != shard_bytes {
            return Err(ERR_TENSOR_BYTES.into());
        }
    }
    if covered.iter().any(|c| !*c) {
        return Err(ERR_METADATA.into());
    }
    let mut out = Vec::new();
    for f in [
        SESSION_MAGIC,
        SESSION_VERSION,
        layout.ctx,
        layout.prefill_cap,
        layout.raw_cap,
        layout.raw_window,
        layout.comp_cap,
        layout.token_count,
        layout.n_layers,
        layout.head_dim,
        layout.indexer_head_dim,
        layout.vocab,
        layout.raw_live,
    ] {
        put_u32(&mut out, f);
    }
    for &tok in tokens {
        put_u32(&mut out, tok as u32);
    }
    for &logit in logits {
        out.extend_from_slice(&logit.to_le_bytes());
    }
    for v in &n_comp {
        put_u32(&mut out, *v);
    }
    for v in &n_index_comp {
        put_u32(&mut out, *v);
    }
    for shard in shards {
        out.extend_from_slice(&shard.tensor);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> DistKvLayout {
        DistKvLayout {
            ctx: 16,
            prefill_cap: 8,
            raw_cap: 16,
            raw_window: 4,
            comp_cap: 8,
            token_count: 2,
            n_layers: 2,
            head_dim: 4,
            indexer_head_dim: 2,
            vocab: 0,
            raw_live: 2,
        }
    }

    fn encode_layer(
        start: u32,
        end: u32,
        n_comp: &[u32],
        n_index: &[u32],
        tensor: &[u8],
    ) -> Vec<u8> {
        let l = layout();
        let mut out = Vec::new();
        for f in [
            LAYER_MAGIC,
            LAYER_VERSION,
            l.ctx,
            l.prefill_cap,
            l.raw_cap,
            l.raw_window,
            l.comp_cap,
            l.token_count,
            l.n_layers,
            l.head_dim,
            l.indexer_head_dim,
            start,
            end,
            l.raw_live,
        ] {
            put_u32(&mut out, f);
        }
        for il in start..=end {
            put_u32(&mut out, n_comp[il as usize]);
        }
        for il in start..=end {
            put_u32(&mut out, n_index[il as usize]);
        }
        out.extend_from_slice(tensor);
        out
    }

    #[test]
    fn layer_tensor_bytes_ratio_zero_is_raw_only() {
        let l = layout();
        assert_eq!(layer_tensor_bytes(&l, 0, 3, 9), Some(2 * 4 * 4));
    }

    #[test]
    fn parse_and_merge_two_uncompressed_shards() {
        let compress = [0u32, 0];
        let t0 = vec![1u8; 32];
        let t1 = vec![2u8; 32];
        let s0 = parse_layer_payload(&encode_layer(0, 0, &[0, 0], &[0, 0], &t0), 0, 0, &compress)
            .unwrap();
        let s1 = parse_layer_payload(&encode_layer(1, 1, &[0, 0], &[0, 0], &t1), 1, 1, &compress)
            .unwrap();
        let tokens = [7i32, 8];
        let logits = [0.5f32, -1.0];
        let merged = merge_session_payload(&tokens, &logits, 2, &[s0, s1], &compress).unwrap();
        assert_eq!(&merged[..4], &SESSION_MAGIC.to_le_bytes());
        assert_eq!(&merged[4..8], &SESSION_VERSION.to_le_bytes());
        let header = SESSION_U32_FIELDS * 4;
        assert_eq!(&merged[header..header + 4], &7u32.to_le_bytes());
        assert_eq!(&merged[header + 4..header + 8], &8u32.to_le_bytes());
        let after_tokens = header + 8;
        assert_eq!(
            &merged[after_tokens..after_tokens + 4],
            &0.5f32.to_le_bytes()
        );
        let after_logits = after_tokens + 8;
        let after_counts = after_logits + 16;
        assert_eq!(&merged[after_counts..after_counts + 32], &t0[..]);
        assert_eq!(&merged[after_counts + 32..after_counts + 64], &t1[..]);
    }

    #[test]
    fn parse_refuses_tensor_byte_mismatch() {
        let compress = [0u32, 0];
        let err = parse_layer_payload(
            &encode_layer(0, 0, &[0, 0], &[0, 0], &[1, 2, 3]),
            0,
            0,
            &compress,
        )
        .unwrap_err();
        assert_eq!(err, ERR_TENSOR_BYTES);
    }

    #[test]
    fn merge_refuses_layout_mismatch() {
        let compress = [0u32, 0];
        let t = vec![1u8; 32];
        let s0 = parse_layer_payload(&encode_layer(0, 0, &[0, 0], &[0, 0], &t), 0, 0, &compress)
            .unwrap();
        let mut other = encode_layer(1, 1, &[0, 0], &[0, 0], &t);
        other[2 * 4..2 * 4 + 4].copy_from_slice(&32u32.to_le_bytes());
        let s1 = parse_layer_payload(&other, 1, 1, &compress).unwrap();
        let err = merge_session_payload(&[7, 8], &[0.0, 0.0], 2, &[s0, s1], &compress).unwrap_err();
        assert_eq!(err, ERR_LAYOUTS_DIFFER);
    }
}
