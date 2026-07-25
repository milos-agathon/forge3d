//! Minimal exact dyadic-integer oracle for predicate adjudication.
//!
//! This is intentionally dependency-free and test-facing.  Every finite f64 is
//! an integer mantissa times a power of two.  Addition aligns the powers by a
//! limb shift; multiplication multiplies the `Vec<u64>` magnitudes.  Only the
//! signed add/subtract/multiply/compare operations needed by the determinants
//! are implemented.

use core::cmp::Ordering;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BigInt {
    sign: i8,
    limbs: Vec<u64>, // little-endian magnitude
}

impl BigInt {
    fn from_u64(value: u64, sign: i8) -> Self {
        if value == 0 {
            Self::default()
        } else {
            Self {
                sign: sign.signum(),
                limbs: vec![value],
            }
        }
    }

    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
        if self.limbs.is_empty() {
            self.sign = 0;
        }
    }

    fn magnitude_cmp(&self, other: &Self) -> Ordering {
        self.limbs
            .len()
            .cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }

    fn add_magnitudes(left: &[u64], right: &[u64]) -> Vec<u64> {
        let mut out = Vec::with_capacity(left.len().max(right.len()) + 1);
        let mut carry = 0u128;
        for index in 0..left.len().max(right.len()) {
            let sum = left.get(index).copied().unwrap_or(0) as u128
                + right.get(index).copied().unwrap_or(0) as u128
                + carry;
            out.push(sum as u64);
            carry = sum >> 64;
        }
        if carry != 0 {
            out.push(carry as u64);
        }
        out
    }

    fn sub_magnitudes(larger: &[u64], smaller: &[u64]) -> Vec<u64> {
        let mut out = Vec::with_capacity(larger.len());
        let mut borrow = 0u128;
        for (index, &value) in larger.iter().enumerate() {
            let subtrahend = smaller.get(index).copied().unwrap_or(0) as u128 + borrow;
            let value = value as u128;
            if value >= subtrahend {
                out.push((value - subtrahend) as u64);
                borrow = 0;
            } else {
                out.push(((1u128 << 64) + value - subtrahend) as u64);
                borrow = 1;
            }
        }
        debug_assert_eq!(borrow, 0);
        out
    }

    fn add(&self, other: &Self) -> Self {
        if self.sign == 0 {
            return other.clone();
        }
        if other.sign == 0 {
            return self.clone();
        }
        let mut out = if self.sign == other.sign {
            Self {
                sign: self.sign,
                limbs: Self::add_magnitudes(&self.limbs, &other.limbs),
            }
        } else {
            match self.magnitude_cmp(other) {
                Ordering::Greater => Self {
                    sign: self.sign,
                    limbs: Self::sub_magnitudes(&self.limbs, &other.limbs),
                },
                Ordering::Less => Self {
                    sign: other.sign,
                    limbs: Self::sub_magnitudes(&other.limbs, &self.limbs),
                },
                Ordering::Equal => Self::default(),
            }
        };
        out.normalize();
        out
    }

    fn negated(&self) -> Self {
        let mut out = self.clone();
        out.sign = -out.sign;
        out
    }

    fn sub(&self, other: &Self) -> Self {
        self.add(&other.negated())
    }

    fn mul(&self, other: &Self) -> Self {
        if self.sign == 0 || other.sign == 0 {
            return Self::default();
        }
        let mut limbs = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &left) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &right) in other.limbs.iter().enumerate() {
                let index = i + j;
                let product = left as u128 * right as u128 + limbs[index] as u128 + carry;
                limbs[index] = product as u64;
                carry = product >> 64;
            }
            let mut index = i + other.limbs.len();
            while carry != 0 {
                let sum = limbs[index] as u128 + carry;
                limbs[index] = sum as u64;
                carry = sum >> 64;
                index += 1;
                if index == limbs.len() && carry != 0 {
                    limbs.push(0);
                }
            }
        }
        let mut out = Self {
            sign: self.sign * other.sign,
            limbs,
        };
        out.normalize();
        out
    }

    fn shl(&self, bits: usize) -> Self {
        if self.sign == 0 || bits == 0 {
            return self.clone();
        }
        let words = bits / 64;
        let offset = bits % 64;
        let mut limbs = vec![0; words];
        let mut carry = 0u64;
        for &limb in &self.limbs {
            limbs.push((limb << offset) | carry);
            carry = if offset == 0 {
                0
            } else {
                limb >> (64 - offset)
            };
        }
        if carry != 0 {
            limbs.push(carry);
        }
        Self {
            sign: self.sign,
            limbs,
        }
    }

    fn sign_ordering(&self) -> Ordering {
        self.sign.cmp(&0)
    }
}

#[derive(Clone, Debug)]
struct Dyadic {
    coefficient: BigInt,
    exponent: i32,
}

impl Dyadic {
    fn from_f64(value: f64) -> Self {
        assert!(value.is_finite(), "oracle accepts only finite f64 values");
        if value == 0.0 {
            return Self {
                coefficient: BigInt::default(),
                exponent: 0,
            };
        }
        let bits = value.to_bits();
        let sign = if bits >> 63 == 0 { 1 } else { -1 };
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        let (mantissa, exponent) = if exponent_bits == 0 {
            (fraction, -1074)
        } else {
            ((1u64 << 52) | fraction, exponent_bits - 1023 - 52)
        };
        Self {
            coefficient: BigInt::from_u64(mantissa, sign),
            exponent,
        }
    }

    fn add(&self, other: &Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        let left = self.coefficient.shl((self.exponent - exponent) as usize);
        let right = other.coefficient.shl((other.exponent - exponent) as usize);
        Self {
            coefficient: left.add(&right),
            exponent,
        }
    }

    fn sub(&self, other: &Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        let left = self.coefficient.shl((self.exponent - exponent) as usize);
        let right = other.coefficient.shl((other.exponent - exponent) as usize);
        Self {
            coefficient: left.sub(&right),
            exponent,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            coefficient: self.coefficient.mul(&other.coefficient),
            exponent: self.exponent + other.exponent,
        }
    }

    fn square(&self) -> Self {
        self.mul(self)
    }
}

fn point(value: [f64; 2]) -> [Dyadic; 2] {
    [Dyadic::from_f64(value[0]), Dyadic::from_f64(value[1])]
}

fn cross(a: &[Dyadic; 2], b: &[Dyadic; 2]) -> Dyadic {
    a[0].mul(&b[1]).sub(&a[1].mul(&b[0]))
}

fn difference(a: &[Dyadic; 2], b: &[Dyadic; 2]) -> [Dyadic; 2] {
    [a[0].sub(&b[0]), a[1].sub(&b[1])]
}

fn lift(value: &[Dyadic; 2]) -> Dyadic {
    value[0].square().add(&value[1].square())
}

/// Exact orientation ordering for finite binary64 inputs.
pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Ordering {
    let a = point(a);
    let b = point(b);
    let c = point(c);
    let ac = difference(&a, &c);
    let bc = difference(&b, &c);
    cross(&ac, &bc).coefficient.sign_ordering()
}

/// Exact incircle ordering for finite binary64 inputs.
pub fn incircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Ordering {
    let a = point(a);
    let b = point(b);
    let c = point(c);
    let d = point(d);
    let ad = difference(&a, &d);
    let bd = difference(&b, &d);
    let cd = difference(&c, &d);
    let determinant = lift(&ad)
        .mul(&cross(&bd, &cd))
        .add(&lift(&bd).mul(&cross(&cd, &ad)))
        .add(&lift(&cd).mul(&cross(&ad, &bd)));
    determinant.coefficient.sign_ordering()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigint_carry_borrow_and_multiply_are_exact() {
        let max = BigInt::from_u64(u64::MAX, 1);
        let one = BigInt::from_u64(1, 1);
        assert_eq!(max.add(&one).limbs, vec![0, 1]);
        assert_eq!(max.add(&one).sub(&one), max);
        let square = max.mul(&max);
        assert_eq!(square.limbs, vec![1, u64::MAX - 1]);
    }

    #[test]
    fn dyadic_signs_cover_subnormals_and_large_exponents() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            orient2d([0.0, 0.0], [tiny, 0.0], [0.0, tiny]),
            Ordering::Greater
        );
        assert_eq!(
            orient2d([1.0e300, 1.0e300], [1.0e300, 1.0e300], [0.0, 0.0]),
            Ordering::Equal
        );
    }
}
