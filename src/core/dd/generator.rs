use super::{quick_two_sum, DD};

#[derive(Clone, Copy)]
pub(super) struct OperandPair {
    pub a: DD,
    pub b: DD,
}

pub(super) fn mix32(value: u32) -> u32 {
    let mut x = value.wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

fn normal_from_bits(state: u32, exponent: i32) -> f32 {
    let sign = (state & 1) << 31;
    let mantissa = (state >> 1) & 0x007f_ffff;
    let biased = (exponent + 127).clamp(1, 254) as u32;
    f32::from_bits(sign | (biased << 23) | mantissa)
}

pub(super) fn generated_pair(index: u32, op: u32) -> OperandPair {
    let s0 = mix32(index.wrapping_mul(4));
    let s1 = mix32(index.wrapping_mul(4).wrapping_add(1));
    let s2 = mix32(index.wrapping_mul(4).wrapping_add(2));
    let s3 = mix32(index.wrapping_mul(4).wrapping_add(3));
    let exponent_a = (s0 % 121) as i32 - 60;
    let mut exponent_b = (s1 % 121) as i32 - 60;
    if op == 1 {
        exponent_b = -exponent_a;
    }
    if op == 2 {
        exponent_b = exponent_a + (s1 % 7) as i32 - 3;
    }
    let a_hi = normal_from_bits(s0, exponent_a);
    let mut b_hi = normal_from_bits(s1, exponent_b);
    if op == 0 && index & 3 == 0 {
        b_hi = -a_hi;
    }
    let a_lo = normal_from_bits(s2, exponent_a - 25);
    let mut b_lo = normal_from_bits(s3, exponent_b - 25);
    if op == 0 && index & 3 == 0 {
        b_lo = -a_lo + normal_from_bits(s3, exponent_a - 48);
    }
    OperandPair {
        a: quick_two_sum(a_hi, a_lo),
        b: quick_two_sum(b_hi, b_lo),
    }
}

pub(super) fn adversarial_pair(index: u32, _op: u32) -> OperandPair {
    let family = index & 7;
    let salt = mix32(index);
    let pair = match family {
        0 => (
            DD {
                hi: 1.0,
                lo: f32::from_bits(0x3380_0000 | (salt & 0x7f_ffff)),
            },
            DD {
                hi: -1.0,
                lo: f32::from_bits(0x3300_0000 | (salt & 0x7f_ffff)),
            },
        ),
        1 => (
            DD {
                hi: f32::from_bits(0x5880_0000 | (salt & 0x7f_ffff)),
                lo: 1.0,
            },
            DD {
                hi: f32::from_bits(0x2680_0000 | ((salt >> 1) & 0x7f_ffff)),
                lo: f32::from_bits(0x1a00_0000 | (salt & 0x7f_ffff)),
            },
        ),
        2 => (
            DD {
                hi: f32::from_bits(0x2b80_0000 | (salt & 0x7f_ffff)),
                lo: f32::from_bits(0x1f00_0000 | (salt & 0x7f_ffff)),
            },
            DD {
                hi: 2.0,
                lo: f32::from_bits(0x3380_0000),
            },
        ),
        3 => (
            DD {
                hi: f32::from_bits(0x5e00_0000 | (salt & 0x7f_ffff)),
                lo: f32::from_bits(0x5180_0000 | (salt & 0x7f_ffff)),
            },
            DD {
                hi: f32::from_bits(0x2000_0000 | ((salt >> 2) & 0x7f_ffff)),
                lo: f32::from_bits(0x1380_0000 | (salt & 0x7f_ffff)),
            },
        ),
        4 => (
            DD {
                hi: f32::from_bits(0x3f80_0000 | (salt & 0x7f_ffff)),
                lo: f32::from_bits(0x3300_0000 | ((salt >> 1) & 0x7f_ffff)),
            },
            DD {
                hi: -f32::from_bits(0x3f80_0000 | (salt & 0x7f_ffff)),
                lo: f32::from_bits(0x3280_0000 | ((salt >> 2) & 0x7f_ffff)),
            },
        ),
        5 => (
            DD {
                hi: f32::from_bits(0x5880_0000 | (salt & 0x7f_ffff)),
                lo: f32::from_bits(0x4c00_0000 | ((salt >> 1) & 0x7f_ffff)),
            },
            DD {
                hi: f32::from_bits(0x2680_0000 | ((salt >> 2) & 0x7f_ffff)),
                lo: f32::from_bits(0x1a00_0000 | ((salt >> 3) & 0x7f_ffff)),
            },
        ),
        6 => (
            DD {
                hi: f32::from_bits(0x3f00_0000 | (salt & 0x7f_ffff)),
                lo: f32::from_bits(0x3280_0000 | ((salt >> 1) & 0x7f_ffff)),
            },
            DD {
                hi: f32::from_bits(0x3f80_0000 | ((salt >> 2) & 0x7f_ffff)),
                lo: -f32::from_bits(0x3300_0000 | ((salt >> 3) & 0x7f_ffff)),
            },
        ),
        _ => (
            DD {
                hi: -f32::from_bits(0x4f00_0000 | (salt & 0x7f_ffff)),
                lo: f32::from_bits(0x4200_0000 | ((salt >> 1) & 0x7f_ffff)),
            },
            DD {
                hi: f32::from_bits(0x2f00_0000 | ((salt >> 2) & 0x7f_ffff)),
                lo: f32::from_bits(0x2300_0000 | ((salt >> 3) & 0x7f_ffff)),
            },
        ),
    };
    OperandPair {
        a: pair.0,
        b: pair.1,
    }
}

pub(super) fn canary_pair(index: u32) -> (f32, f32) {
    match index {
        0 => (1.0, f32::from_bits(0x3380_0000)),
        1 => (1.0, -1.0),
        2 => (f32::from_bits(0x007f_ffff), f32::from_bits(1)),
        3 => (f32::from_bits(0x7e80_1234), f32::from_bits(0x0080_4321)),
        4 => (f32::from_bits(0x0001_2345), f32::from_bits(0x5d12_3456)),
        5 => (-f32::from_bits(0x7d55_4321), f32::from_bits(0x0180_0123)),
        6 => (f32::from_bits(0x3f80_0001), f32::from_bits(0x3f7f_ffff)),
        7 => (12_345.5, -12_344.75),
        8 => (0.0, -0.0),
        _ => (-0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dd::{two_prod_fma, two_prod_split, two_sum};

    #[test]
    fn denormal_canary_has_nonzero_exact_sum_and_product() {
        let (a, b) = canary_pair(4);
        assert!(a.is_subnormal());
        let sum = two_sum(a, b);
        assert_eq!(sum.to_f64(), a as f64 + b as f64);
        let product = two_prod_fma(a, b);
        assert_ne!(product.to_f64(), 0.0);
        assert_eq!(product.to_f64(), a as f64 * b as f64);
        assert_eq!(two_prod_split(a, b).to_f64(), a as f64 * b as f64);
    }

    #[test]
    fn subnormal_plus_signed_zero_stays_normalized() {
        let tiny = f32::from_bits(1);
        for zero in [0.0, -0.0] {
            assert_eq!(two_sum(tiny, zero).hi.to_bits(), tiny.to_bits());
            assert_eq!(two_sum(zero, tiny).hi.to_bits(), tiny.to_bits());
            assert_eq!(two_sum(tiny, zero).lo.to_bits(), 0.0_f32.to_bits());
            assert_eq!(two_sum(zero, tiny).lo.to_bits(), 0.0_f32.to_bits());
        }
    }
}
