//! DUPLA double-float arithmetic.

mod generator;
mod gpu;
mod gpu_exec;
mod gpu_report;
mod jitter;
mod jitter_model;
mod jitter_pipeline;
mod product;
mod proof;
mod types;
mod vector;

pub(crate) use gpu::{harness, initialize_for_context, selftest};
#[cfg(test)]
pub(crate) use gpu::{harness_for_test, harness_window_for_test};
#[cfg(test)]
pub(crate) use gpu_report::DdOperation;
pub(crate) use jitter::jitter_demo;
pub use product::{two_prod, two_prod_fma, two_prod_split};
pub use types::{
    DDVec3, DD, DD_ADD_BOUND_U2, DD_DIV_BOUND_U2, DD_MUL_BOUND_U2, DD_SQRT_BOUND_U2, DD_U,
};
#[allow(unused_imports)] // Complete Rust mirror surface; exercised by core/dd_tests.rs.
pub use vector::{dd_dot3, dd_length3, dd_sub_vec3};

// BEGIN GENERATED DD MIRROR
impl DD {
    /// Split a finite, binary32-range f64 into a normalized pair.
    pub fn from_f64(value: f64) -> Self {
        debug_assert!(value.is_finite() && value.abs() <= f32::MAX as f64);
        let hi = value as f32;
        let lo = (value - hi as f64) as f32;
        dd_renorm(Self { hi, lo })
    }
}

#[inline]
fn dd_barrier(value: f32) -> f32 {
    f32::from_bits(value.to_bits())
}

/// Knuth two_sum: six flops, exact under round-to-nearest barring overflow.
fn two_sum_subnormal(a: f32, b: f32, a_bits: u32, b_bits: u32, a_sub: bool, b_sub: bool) -> DD {
    if a_sub && b_sub {
        let mut a_signed = (a_bits & 0x007f_ffff) as i32;
        if a_bits & 0x8000_0000 != 0 {
            a_signed = -a_signed;
        }
        let mut b_signed = (b_bits & 0x007f_ffff) as i32;
        if b_bits & 0x8000_0000 != 0 {
            b_signed = -b_signed;
        }
        let total = a_signed + b_signed;
        if total == 0 {
            return DD::ZERO;
        }
        let mut sign = 0;
        let mut magnitude = total;
        if magnitude < 0 {
            sign = 0x8000_0000;
            magnitude = -magnitude;
        }
        return DD {
            hi: f32::from_bits(sign | magnitude as u32),
            lo: 0.0,
        };
    }
    if a_sub && b_bits & 0x7fff_ffff == 0 {
        return DD { hi: a, lo: 0.0 };
    }
    if b_sub && a_bits & 0x7fff_ffff == 0 {
        return DD { hi: b, lo: 0.0 };
    }
    if a_sub {
        return DD { hi: b, lo: a };
    }
    DD { hi: a, lo: b }
}

pub fn two_sum(a: f32, b: f32) -> DD {
    let a_bits = a.to_bits();
    let b_bits = b.to_bits();
    let a_sub = a_bits & 0x007f_ffff != 0 && a_bits & 0x7f80_0000 == 0;
    let b_sub = b_bits & 0x007f_ffff != 0 && b_bits & 0x7f80_0000 == 0;
    if a_sub || b_sub {
        return two_sum_subnormal(a, b, a_bits, b_bits, a_sub, b_sub);
    }
    let hi = dd_barrier(a + b);
    let b_virtual = dd_barrier(hi - a);
    let a_virtual = dd_barrier(hi - b_virtual);
    let b_roundoff = dd_barrier(b - b_virtual);
    let a_roundoff = dd_barrier(a - a_virtual);
    let lo = dd_barrier(a_roundoff + b_roundoff);
    DD { hi, lo }
}

/// Dekker quick_two_sum: exact when |a| >= |b| under round-to-nearest.
pub fn quick_two_sum(a: f32, b: f32) -> DD {
    let hi = dd_barrier(a + b);
    let lo = dd_barrier(b - dd_barrier(hi - a));
    DD { hi, lo }
}

pub fn dd_renorm(value: DD) -> DD {
    quick_two_sum(value.hi, value.lo)
}

fn dd_canonicalize_tie(value: DD) -> DD {
    let bits = value.hi.to_bits();
    let biased = (bits >> 23) & 0xff;
    if biased <= 24 || bits & 1 == 0 {
        return value;
    }
    let half_ulp = f32::from_bits((biased - 24) << 23);
    let lo_bits = value.lo.abs().to_bits();
    let half_bits = half_ulp.to_bits();
    if lo_bits > half_bits || half_bits - lo_bits > 1 {
        return value;
    }
    let increment = (value.lo > 0.0) != (value.hi < 0.0);
    let mut adjacent_bits = bits - 1;
    if increment {
        adjacent_bits = bits + 1;
    }
    DD {
        hi: f32::from_bits(adjacent_bits),
        lo: -value.lo,
    }
}

/// Joldeș-Muller-Popescu (2017): accurate double-word add <= 3u².
pub fn dd_add(a: DD, b: DD) -> DD {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    let merged = quick_two_sum(s.hi, dd_barrier(s.lo + t.hi));
    quick_two_sum(merged.hi, dd_barrier(merged.lo + t.lo))
}

pub fn dd_sub(a: DD, b: DD) -> DD {
    dd_add(
        a,
        DD {
            hi: -b.hi,
            lo: -b.lo,
        },
    )
}

/// Joldeș-Muller-Popescu (2017): double-word multiply <= 7u².
pub fn dd_mul(a: DD, b: DD) -> DD {
    let p = two_prod(a.hi, b.hi);
    let cross0 = dd_barrier(a.hi * b.lo);
    let cross1 = dd_barrier(a.lo * b.hi);
    let correction = dd_barrier(dd_barrier(p.lo + cross0) + cross1);
    quick_two_sum(p.hi, correction)
}

fn dd_mul_split_fixed(a: DD, b: DD) -> DD {
    let p = two_prod_split(a.hi, b.hi);
    let cross0 = dd_barrier(a.hi * b.lo);
    let cross1 = dd_barrier(a.lo * b.hi);
    let correction = dd_barrier(dd_barrier(p.lo + cross0) + cross1);
    quick_two_sum(p.hi, correction)
}

/// Newton-Raphson reciprocal from multiplication/addition only (JMP 2017 division setup).
pub fn dd_reciprocal_refine(x: f32) -> f32 {
    let ax = x.abs();
    let mut y = f32::from_bits(0x7ef3_11c3_u32 - ax.to_bits());
    y = dd_barrier(y * (2.0 - dd_barrier(ax * y)));
    y = dd_barrier(y * (2.0 - dd_barrier(ax * y)));
    y = dd_barrier(y * (2.0 - dd_barrier(ax * y)));
    y = dd_barrier(y * (2.0 - dd_barrier(ax * y)));
    if x < 0.0 {
        return -y;
    }
    y
}

/// Joldeș-Muller-Popescu (2017): corrected double-word divide <= 15u².
pub fn dd_div(a: DD, b: DD) -> DD {
    debug_assert!(b.hi.is_finite() && b.hi != 0.0);
    let reciprocal = dd_reciprocal_refine(b.hi);
    let q_hi = dd_barrier(a.hi * reciprocal);
    let q = DD { hi: q_hi, lo: 0.0 };
    let remainder = dd_sub(a, dd_mul(b, q));
    let q_lo = dd_barrier((remainder.hi + remainder.lo) * reciprocal);
    let first = dd_add(q, DD { hi: q_lo, lo: 0.0 });
    let final_remainder = dd_sub(a, dd_mul(b, first));
    let final_lo = dd_barrier((final_remainder.hi + final_remainder.lo) * reciprocal);
    dd_add(
        first,
        DD {
            hi: final_lo,
            lo: 0.0,
        },
    )
}

/// Hardware inverse-sqrt is a seed only; no error-free step depends on its accuracy.
pub fn dd_inverse_sqrt_seed(x: f32) -> f32 {
    x.sqrt().recip()
}

fn inverse_sqrt_residual(x: f32, y: f32) -> DD {
    let yy = dd_mul_split_fixed(DD { hi: y, lo: 0.0 }, DD { hi: y, lo: 0.0 });
    let xyy = dd_mul_split_fixed(DD { hi: x, lo: 0.0 }, yy);
    dd_sub(DD { hi: 1.0, lo: 0.0 }, xyy)
}

fn inverse_sqrt_residual_better(candidate: DD, best: DD) -> bool {
    f32::abs(candidate.hi) < f32::abs(best.hi)
        || (f32::abs(candidate.hi) == f32::abs(best.hi)
            && f32::abs(candidate.lo) < f32::abs(best.lo))
}

fn inverse_sqrt_residual_equal(candidate: DD, best: DD) -> bool {
    candidate.hi == best.hi && candidate.lo == best.lo
}

/// Newton inverse-square-root refinement using multiplication/addition only.
pub fn dd_inverse_sqrt_refine(x: f32, seed: f32) -> f32 {
    let half_x = dd_barrier(0.5 * x);
    let mut y = seed;
    y = dd_barrier(y * (1.5 - dd_barrier(half_x * dd_barrier(y * y))));
    y = dd_barrier(y * (1.5 - dd_barrier(half_x * dd_barrier(y * y))));
    y = dd_barrier(y * (1.5 - dd_barrier(half_x * dd_barrier(y * y))));
    y = dd_barrier(y * (1.5 - dd_barrier(half_x * dd_barrier(y * y))));
    let mut best = y;
    let mut best_bits = y.to_bits();
    let mut best_error = inverse_sqrt_residual(x, y);
    let below_bits = best_bits - 1;
    let below = f32::from_bits(below_bits);
    let below_error = inverse_sqrt_residual(x, below);
    if inverse_sqrt_residual_better(below_error, best_error)
        || (inverse_sqrt_residual_equal(below_error, best_error) && below_bits < best_bits)
    {
        best = below;
        best_bits = below_bits;
        best_error = below_error;
    }
    let above_bits = y.to_bits() + 1;
    let above = f32::from_bits(above_bits);
    let above_error = inverse_sqrt_residual(x, above);
    if inverse_sqrt_residual_better(above_error, best_error)
        || (inverse_sqrt_residual_equal(above_error, best_error) && above_bits < best_bits)
    {
        best = above;
    }
    best
}

/// Newton inverse-square-root refinement plus two residual corrections; <= 15u² gate.
pub fn dd_sqrt(a: DD) -> DD {
    debug_assert!(a.hi.is_finite() && a.hi >= 0.0);
    if a.hi == 0.0 {
        return DD::ZERO;
    }
    let inverse = dd_inverse_sqrt_refine(a.hi, dd_inverse_sqrt_seed(a.hi));
    let root_hi = dd_barrier(a.hi * inverse);
    let root = DD {
        hi: root_hi,
        lo: 0.0,
    };
    let remainder = dd_sub(a, dd_mul(root, root));
    let inverse_two_root = dd_reciprocal_refine(dd_barrier(2.0 * root_hi));
    let root_lo = dd_barrier(dd_barrier(remainder.hi + remainder.lo) * inverse_two_root);
    let first = dd_add(
        root,
        DD {
            hi: root_lo,
            lo: 0.0,
        },
    );
    let canonical_root = DD {
        hi: first.hi,
        lo: 0.0,
    };
    let canonical_remainder = dd_sub(a, dd_mul(canonical_root, canonical_root));
    let canonical_inverse = dd_reciprocal_refine(dd_barrier(2.0 * first.hi));
    let canonical_lo =
        dd_barrier(dd_barrier(canonical_remainder.hi + canonical_remainder.lo) * canonical_inverse);
    dd_canonicalize_tie(dd_add(
        canonical_root,
        DD {
            hi: canonical_lo,
            lo: 0.0,
        },
    ))
}

// END GENERATED DD MIRROR
