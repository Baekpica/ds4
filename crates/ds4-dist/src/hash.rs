//! FNV-1a over little-endian token IDs. Not a security primitive.

pub const TOKEN_HASH_INIT: u64 = 1_469_598_103_934_665_603;
pub const TOKEN_HASH_PRIME: u64 = 1_099_511_628_211;

pub fn token_hash_update(mut h: u64, token: i32) -> u64 {
    let t = token as u32;
    for i in 0..4 {
        h ^= u64::from((t >> (i * 8)) & 0xff);
        h = h.wrapping_mul(TOKEN_HASH_PRIME);
    }
    h
}

pub fn token_hash_update_span(mut h: u64, tokens: &[i32]) -> u64 {
    for &t in tokens {
        h = token_hash_update(h, t);
    }
    h
}

pub fn token_hash_prefix(tokens: &[i32]) -> u64 {
    token_hash_update_span(TOKEN_HASH_INIT, tokens)
}
