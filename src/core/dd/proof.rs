use super::generator::{adversarial_pair, generated_pair, OperandPair};
use super::gpu_exec::HarnessOutput;
use super::gpu_report::{DdOperation, TwoProdVariant};
use super::*;

pub(super) fn bits_equal(a: DD, b: DD) -> bool {
    a.hi.to_bits() == b.hi.to_bits() && a.lo.to_bits() == b.lo.to_bits()
}

pub(super) fn selected_product(a: f32, b: f32, variant: TwoProdVariant) -> DD {
    match variant {
        TwoProdVariant::Fma => two_prod_fma(a, b),
        TwoProdVariant::Split => two_prod_split(a, b),
    }
}

fn mul_selected(a: DD, b: DD, variant: TwoProdVariant) -> DD {
    let p = selected_product(a.hi, b.hi, variant);
    let correction =
        dd_barrier(dd_barrier(p.lo + dd_barrier(a.hi * b.lo)) + dd_barrier(a.lo * b.hi));
    quick_two_sum(p.hi, correction)
}

fn eval_selected(op: DdOperation, pair: OperandPair, variant: TwoProdVariant) -> (DD, f64) {
    match op {
        DdOperation::Add => (dd_add(pair.a, pair.b), pair.a.to_f64() + pair.b.to_f64()),
        DdOperation::Mul => (
            mul_selected(pair.a, pair.b, variant),
            pair.a.to_f64() * pair.b.to_f64(),
        ),
        DdOperation::Div => {
            let reciprocal = dd_reciprocal_refine(pair.b.hi);
            let q = DD {
                hi: dd_barrier(pair.a.hi * reciprocal),
                lo: 0.0,
            };
            let remainder = dd_sub(pair.a, mul_selected(pair.b, q, variant));
            let q_lo = dd_barrier((remainder.hi + remainder.lo) * reciprocal);
            let first = dd_add(q, DD { hi: q_lo, lo: 0.0 });
            let final_remainder = dd_sub(pair.a, mul_selected(pair.b, first, variant));
            let final_lo = dd_barrier((final_remainder.hi + final_remainder.lo) * reciprocal);
            (
                dd_add(
                    first,
                    DD {
                        hi: final_lo,
                        lo: 0.0,
                    },
                ),
                pair.a.to_f64() / pair.b.to_f64(),
            )
        }
        DdOperation::Sqrt => {
            let value = DD {
                hi: pair.a.hi.abs(),
                lo: pair.a.lo.abs(),
            };
            if value.hi == 0.0 {
                return (DD::ZERO, 0.0);
            }
            let inverse = dd_inverse_sqrt_refine(value.hi, dd_inverse_sqrt_seed(value.hi));
            let root = DD {
                hi: dd_barrier(value.hi * inverse),
                lo: 0.0,
            };
            let remainder = dd_sub(value, mul_selected(root, root, variant));
            let inverse_two_root = dd_reciprocal_refine(dd_barrier(2.0 * root.hi));
            let lo = dd_barrier(dd_barrier(remainder.hi + remainder.lo) * inverse_two_root);
            let first = dd_add(root, DD { hi: lo, lo: 0.0 });
            let canonical_root = DD {
                hi: first.hi,
                lo: 0.0,
            };
            let canonical_remainder =
                dd_sub(value, mul_selected(canonical_root, canonical_root, variant));
            let canonical_inverse = dd_reciprocal_refine(dd_barrier(2.0 * first.hi));
            let canonical_lo = dd_barrier(
                dd_barrier(canonical_remainder.hi + canonical_remainder.lo) * canonical_inverse,
            );
            (
                dd_canonicalize_tie(dd_add(
                    canonical_root,
                    DD {
                        hi: canonical_lo,
                        lo: 0.0,
                    },
                )),
                value.to_f64().sqrt(),
            )
        }
    }
}

pub(super) fn reduce_chunk(
    op: DdOperation,
    variant: TwoProdVariant,
    phase: u32,
    offset: u64,
    outputs: &[HarnessOutput],
    mismatch_count: &mut u64,
    max_err_u2: &mut f64,
) {
    for (local, output) in outputs.iter().enumerate() {
        let index = (offset + local as u64) as u32;
        let pair = if phase == 0 {
            adversarial_pair(index, op.code())
        } else {
            generated_pair(index, op.code())
        };
        let (mirror, exact) = eval_selected(op, pair, variant);
        if !output.primary.hi.is_finite()
            || !output.primary.lo.is_finite()
            || !mirror.hi.is_finite()
            || !mirror.lo.is_finite()
            || !exact.is_finite()
        {
            *mismatch_count += 1;
            *max_err_u2 = f64::INFINITY;
            continue;
        }
        if !bits_equal(output.primary, mirror) {
            *mismatch_count += 1;
        }
        let scale = exact.abs().max(f64::MIN_POSITIVE);
        let err = (output.primary.to_f64() - exact).abs() / scale / DD_U.powi(2);
        if err.is_finite() {
            *max_err_u2 = max_err_u2.max(err);
        } else {
            *mismatch_count += 1;
            *max_err_u2 = f64::INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_gpu_output_forces_proof_failure() {
        let output = HarnessOutput {
            primary: DD {
                hi: f32::NAN,
                lo: 0.0,
            },
            product: DD::ZERO,
        };
        let mut mismatches = 0;
        let mut maximum = 0.0;
        reduce_chunk(
            DdOperation::Add,
            TwoProdVariant::Fma,
            1,
            0,
            &[output],
            &mut mismatches,
            &mut maximum,
        );
        assert_eq!(mismatches, 1);
        assert!(maximum.is_infinite());
    }
}
