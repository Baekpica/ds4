//! Byte-identical port of the SHA-1 in `ds4_kvstore.c`.

pub fn sha1(bytes: &[u8]) -> [u8; 20] {
    let mut c = Ctx::new();
    c.update(bytes);
    c.finish()
}

pub fn sha1_hex(bytes: &[u8]) -> String {
    hex20(&sha1(bytes))
}

struct Ctx {
    h: [u32; 5],
    bytes: u64,
    block: [u8; 64],
    used: usize,
}

impl Ctx {
    fn new() -> Self {
        Self {
            h: [
                0x6745_2301,
                0xefcd_ab89,
                0x98ba_dcfe,
                0x1032_5476,
                0xc3d2_e1f0,
            ],
            bytes: 0,
            block: [0; 64],
            used: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.bytes += input.len() as u64;
        while !input.is_empty() {
            let n = (64 - self.used).min(input.len());
            self.block[self.used..self.used + n].copy_from_slice(&input[..n]);
            self.used += n;
            input = &input[n..];
            if self.used == 64 {
                self.transform();
                self.used = 0;
            }
        }
    }

    fn finish(&mut self) -> [u8; 20] {
        let bits = self.bytes * 8;
        self.update(&[0x80]);
        let zero = [0u8; 1];
        while self.used != 56 {
            self.update(&zero);
        }
        let mut len = [0u8; 8];
        for i in 0..8 {
            len[7 - i] = (bits >> (8 * i)) as u8;
        }
        self.update(&len);
        let mut out = [0u8; 20];
        for i in 0..5 {
            out[i * 4] = (self.h[i] >> 24) as u8;
            out[i * 4 + 1] = (self.h[i] >> 16) as u8;
            out[i * 4 + 2] = (self.h[i] >> 8) as u8;
            out[i * 4 + 3] = self.h[i] as u8;
        }
        out
    }

    fn transform(&mut self) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = (u32::from(self.block[i * 4]) << 24)
                | (u32::from(self.block[i * 4 + 1]) << 16)
                | (u32::from(self.block[i * 4 + 2]) << 8)
                | u32::from(self.block[i * 4 + 3]);
        }
        for i in 16..80 {
            w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }

        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut cc = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        for i in 0..80 {
            let (f, k) = if i < 20 {
                ((b & cc) | ((!b) & d), 0x5a82_7999)
            } else if i < 40 {
                (b ^ cc ^ d, 0x6ed9_eba1)
            } else if i < 60 {
                ((b & cc) | (b & d) | (cc & d), 0x8f1b_bcdc)
            } else {
                (b ^ cc ^ d, 0xca62_c1d6)
            };
            let tmp = rol32(a, 5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = cc;
            cc = rol32(b, 30);
            b = a;
            a = tmp;
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(cc);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
    }
}

fn rol32(v: u32, n: i32) -> u32 {
    v.rotate_left(n as u32)
}

fn hex20(digest: &[u8; 20]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(40);
    for &b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 15) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_abc_matches_fips() {
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }
}
