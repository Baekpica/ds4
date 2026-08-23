//! Activation packing. 32-bit payloads are native IEEE754 memcpy (LE on GB10).
//! 16/8-bit packers match `ds4_distributed.c` bit-for-bit.

pub const BITS_DEFAULT: u32 = 32;

pub fn bits_or_default(bits: u32) -> u32 {
    if bits == 0 { BITS_DEFAULT } else { bits }
}

pub fn bits_valid(bits: u32) -> bool {
    let bits = bits_or_default(bits);
    bits == 32 || bits == 16 || bits == 8
}

pub fn wire_bytes(bits: u32, values: u64) -> Option<u32> {
    let bits = bits_or_default(bits);
    if !bits_valid(bits) || bits % 8 != 0 {
        return None;
    }
    let bytes = values.checked_mul(u64::from(bits / 8))?;
    u32::try_from(bytes).ok()
}

pub fn values_from_wire_bytes(bits: u32, bytes: u32) -> Option<u64> {
    let bits = bits_or_default(bits);
    if !bits_valid(bits) || bits % 8 != 0 {
        return None;
    }
    let bpv = bits / 8;
    if bpv == 0 || bytes % bpv != 0 {
        return None;
    }
    Some(u64::from(bytes / bpv))
}

pub fn wire_bytes_from_f32_bytes(bits: u32, f32_bytes: u32) -> Option<u32> {
    if f32_bytes % 4 != 0 {
        return None;
    }
    wire_bytes(bits, u64::from(f32_bytes / 4))
}

pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mut mant = bits & 0x7f_ffff;

    if exp <= 0 {
        if exp < -10 {
            return sign as u16;
        }
        mant |= 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut half_mant = mant >> shift;
        let round_bit = (mant >> (shift - 1)) & 1;
        let sticky = mant & ((1u32 << (shift - 1)) - 1);
        if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return (sign | half_mant) as u16;
    }

    if exp >= 31 {
        if ((bits >> 23) & 0xff) == 0xff && mant != 0 {
            return (sign | 0x7e00) as u16;
        }
        return (sign | 0x7c00) as u16;
    }

    let mut half = sign | ((exp as u32) << 10) | (mant >> 13);
    let round = mant & 0x1fff;
    if round > 0x1000 || (round == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    half as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h & 0x8000)) << 16;
    let mut exp = i32::from((h >> 10) & 0x1f);
    let mut mant = u32::from(h & 0x03ff);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 1;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f8_e4m3(f: f32) -> u8 {
    let sign = if f.is_sign_negative() { 0x80u8 } else { 0 };
    let a = f.abs();
    if a == 0.0 {
        return sign;
    }
    if !a.is_finite() || a >= 240.0 {
        return sign | 0x77;
    }
    if a < 0.001953125 {
        let mut mant = (a * 512.0 + 0.5).floor() as i32;
        if mant <= 0 {
            return sign;
        }
        if mant > 7 {
            mant = 7;
        }
        return sign | (mant as u8);
    }

    let (_frac, exp2) = frexp(a);
    let mut exp = exp2 - 1 + 7;
    if exp <= 0 {
        let mut mant = (a * 512.0 + 0.5).floor() as i32;
        if mant <= 0 {
            return sign;
        }
        if mant > 7 {
            mant = 7;
        }
        return sign | (mant as u8);
    }
    let base = libm_ldexp(1.0, exp2 - 1);
    let mut mant = (((a / base) - 1.0) * 8.0 + 0.5).floor() as i32;
    if mant >= 8 {
        mant = 0;
        exp += 1;
    }
    if exp >= 15 {
        return sign | 0x77;
    }
    sign | ((exp as u8) << 3) | (mant as u8)
}

fn frexp(x: f32) -> (f32, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32;
    if e == 0 {
        // subnormal
        let mut v = x.abs();
        let mut exp = 0;
        while v < 0.5 {
            v *= 2.0;
            exp -= 1;
        }
        return (x.signum() * v, exp + 1);
    }
    e -= 126; // so mantissa in [0.5, 1)
    let frac_bits = (bits & 0x807f_ffff) | 0x3f00_0000;
    (f32::from_bits(frac_bits), e)
}

fn libm_ldexp(x: f32, exp: i32) -> f32 {
    x * 2f32.powi(exp)
}

pub fn f8_e4m3_to_f32(h: u8) -> f32 {
    let sign = if (h & 0x80) != 0 { -1.0f32 } else { 1.0 };
    let exp = (h >> 3) & 0x0f;
    let mant = h & 0x07;
    if exp == 0 {
        return sign * (mant as f32) * 0.001953125;
    }
    if exp >= 15 {
        return sign * 240.0;
    }
    sign * (1.0 + (mant as f32) / 8.0) * 2f32.powi(exp as i32 - 7)
}

/// Pack host f32 values to the C wire layout (LE f32 / LE f16 / raw f8).
pub fn encode_activation(src: &[f32], bits: u32) -> Option<Vec<u8>> {
    let bits = bits_or_default(bits);
    if !bits_valid(bits) {
        return None;
    }
    match bits {
        32 => {
            let mut out = Vec::with_capacity(src.len() * 4);
            for &f in src {
                out.extend_from_slice(&f.to_bits().to_le_bytes());
            }
            Some(out)
        }
        16 => {
            let mut out = Vec::with_capacity(src.len() * 2);
            for &f in src {
                out.extend_from_slice(&f32_to_f16(f).to_le_bytes());
            }
            Some(out)
        }
        8 => Some(src.iter().copied().map(f32_to_f8_e4m3).collect()),
        _ => None,
    }
}

pub fn decode_activation(wire: &[u8], bits: u32) -> Option<Vec<f32>> {
    let bits = bits_or_default(bits);
    let n = values_from_wire_bytes(bits, wire.len() as u32)? as usize;
    match bits {
        32 => {
            let mut out = Vec::with_capacity(n);
            for chunk in wire.chunks_exact(4) {
                out.push(f32::from_bits(u32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
            Some(out)
        }
        16 => {
            let mut out = Vec::with_capacity(n);
            for chunk in wire.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
            Some(out)
        }
        8 => Some(wire.iter().copied().map(f8_e4m3_to_f32).collect()),
        _ => None,
    }
}
