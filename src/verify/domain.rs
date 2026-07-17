//! Conservative scalar intervals used component-wise for WGSL values.
#![cfg_attr(not(test), allow(dead_code))]
// ponytail: the follow-on verifier interpreter will consume this module; remove the allowance then.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    pub(crate) lo: f32,
    pub(crate) hi: f32,
    pub(crate) may_nan: bool,
    pub(crate) may_pos_inf: bool,
    pub(crate) may_neg_inf: bool,
    pub(crate) may_neg_zero: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Comparison {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Interval {
    pub(crate) fn new(lo: f32, hi: f32) -> Self {
        assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
        Self {
            lo,
            hi,
            may_nan: false,
            may_pos_inf: false,
            may_neg_inf: false,
            may_neg_zero: (lo == 0.0 && lo.is_sign_negative()) || (lo < 0.0 && hi >= 0.0),
        }
    }

    pub(crate) fn constant(value: f32) -> Self {
        if value.is_nan() {
            return Self::exceptional(true, false, false, false);
        }
        if value == f32::INFINITY {
            return Self::exceptional(false, true, false, false);
        }
        if value == f32::NEG_INFINITY {
            return Self::exceptional(false, false, true, false);
        }
        Self {
            lo: value,
            hi: value,
            may_nan: false,
            may_pos_inf: false,
            may_neg_inf: false,
            may_neg_zero: value == 0.0 && value.is_sign_negative(),
        }
    }

    fn exceptional(nan: bool, pos_inf: bool, neg_inf: bool, neg_zero: bool) -> Self {
        Self {
            lo: f32::INFINITY,
            hi: f32::NEG_INFINITY,
            may_nan: nan,
            may_pos_inf: pos_inf,
            may_neg_inf: neg_inf,
            may_neg_zero: neg_zero,
        }
    }

    fn top(nan: bool) -> Self {
        Self {
            lo: -f32::MAX,
            hi: f32::MAX,
            may_nan: nan,
            may_pos_inf: true,
            may_neg_inf: true,
            may_neg_zero: true,
        }
    }

    fn has_finite(self) -> bool {
        self.lo <= self.hi
    }

    fn has_infinity(self) -> bool {
        self.may_pos_inf || self.may_neg_inf
    }

    fn may_zero(self) -> bool {
        self.may_neg_zero || (self.has_finite() && self.lo <= 0.0 && self.hi >= 0.0)
    }

    pub(crate) fn has_exceptional_values(&self) -> bool {
        self.may_nan || self.may_pos_inf || self.may_neg_inf || self.may_neg_zero
    }

    pub(crate) fn contains(self, value: f32) -> bool {
        if value.is_nan() {
            self.may_nan
        } else if value == f32::INFINITY {
            self.may_pos_inf
        } else if value == f32::NEG_INFINITY {
            self.may_neg_inf
        } else if value == 0.0 && value.is_sign_negative() {
            self.may_neg_zero
        } else {
            self.has_finite() && self.lo <= value && value <= self.hi
        }
    }

    pub(crate) fn add(self, rhs: Self) -> Self {
        if self.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        bounds(
            self.lo as f64 + rhs.lo as f64,
            self.hi as f64 + rhs.hi as f64,
            self.may_nan || rhs.may_nan,
        )
    }

    pub(crate) fn sub(self, rhs: Self) -> Self {
        if self.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        bounds(
            self.lo as f64 - rhs.hi as f64,
            self.hi as f64 - rhs.lo as f64,
            self.may_nan || rhs.may_nan,
        )
    }

    pub(crate) fn mul(self, rhs: Self) -> Self {
        if self.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        let products = [
            self.lo as f64 * rhs.lo as f64,
            self.lo as f64 * rhs.hi as f64,
            self.hi as f64 * rhs.lo as f64,
            self.hi as f64 * rhs.hi as f64,
        ];
        from_candidates(&products, self.may_nan || rhs.may_nan)
    }

    pub(crate) fn div(self, rhs: Self) -> Self {
        if rhs.may_zero() {
            return Self::top(true);
        }
        if self.has_infinity() || rhs.has_infinity() {
            return Self::top(true);
        }
        let quotients = [
            self.lo as f64 / rhs.lo as f64,
            self.lo as f64 / rhs.hi as f64,
            self.hi as f64 / rhs.lo as f64,
            self.hi as f64 / rhs.hi as f64,
        ];
        from_candidates(&quotients, self.may_nan || rhs.may_nan)
    }

    pub(crate) fn sqrt(self) -> Self {
        let may_nan = self.may_nan || self.may_neg_inf || (self.has_finite() && self.lo < 0.0);
        let mut result = if self.has_finite() && self.hi >= 0.0 {
            bounds(self.lo.max(0.0) as f64, self.hi as f64, may_nan).map_bounds(f64::sqrt)
        } else {
            Self::exceptional(may_nan, false, false, false)
        };
        result.may_pos_inf |= self.may_pos_inf;
        result.may_neg_zero |= self.may_neg_zero;
        result
    }

    pub(crate) fn inverse_sqrt(self) -> Self {
        Self::constant(1.0).div(self.sqrt())
    }

    pub(crate) fn min(self, rhs: Self) -> Self {
        if self.has_infinity() || rhs.has_infinity() {
            return Self::top(self.may_nan || rhs.may_nan);
        }
        if !self.has_finite() || !rhs.has_finite() {
            return self.join(rhs);
        }
        bounds(
            self.lo.min(rhs.lo) as f64,
            self.hi.min(rhs.hi) as f64,
            self.may_nan || rhs.may_nan,
        )
    }

    pub(crate) fn max(self, rhs: Self) -> Self {
        if self.has_infinity() || rhs.has_infinity() {
            return Self::top(self.may_nan || rhs.may_nan);
        }
        if !self.has_finite() || !rhs.has_finite() {
            return self.join(rhs);
        }
        bounds(
            self.lo.max(rhs.lo) as f64,
            self.hi.max(rhs.hi) as f64,
            self.may_nan || rhs.may_nan,
        )
    }

    pub(crate) fn clamp(self, low: Self, high: Self) -> Self {
        if !low.has_finite() || !high.has_finite() || low.hi > high.lo {
            return Self::top(true);
        }
        self.max(low).min(high)
    }

    pub(crate) fn mix(self, rhs: Self, factor: Self) -> Self {
        self.mul(Self::constant(1.0).sub(factor))
            .add(rhs.mul(factor))
    }

    pub(crate) fn select(on_false: Self, on_true: Self, condition: Option<bool>) -> Self {
        match condition {
            Some(false) => on_false,
            Some(true) => on_true,
            None => on_false.join(on_true),
        }
    }

    pub(crate) fn fma(self, multiplier: Self, addend: Self) -> Self {
        if self.has_infinity() || multiplier.has_infinity() || addend.has_infinity() {
            return Self::top(true);
        }
        let mut candidates = [0.0; 8];
        let mut index = 0;
        for left in [self.lo, self.hi] {
            for right in [multiplier.lo, multiplier.hi] {
                for addend in [addend.lo, addend.hi] {
                    candidates[index] = left as f64 * right as f64 + addend as f64;
                    index += 1;
                }
            }
        }
        from_candidates(
            &candidates,
            self.may_nan || multiplier.may_nan || addend.may_nan,
        )
    }

    pub(crate) fn join(self, rhs: Self) -> Self {
        let (lo, hi) = match (self.has_finite(), rhs.has_finite()) {
            (true, true) => (self.lo.min(rhs.lo), self.hi.max(rhs.hi)),
            (true, false) => (self.lo, self.hi),
            (false, true) => (rhs.lo, rhs.hi),
            (false, false) => (f32::INFINITY, f32::NEG_INFINITY),
        };
        Self {
            lo,
            hi,
            may_nan: self.may_nan || rhs.may_nan,
            may_pos_inf: self.may_pos_inf || rhs.may_pos_inf,
            may_neg_inf: self.may_neg_inf || rhs.may_neg_inf,
            may_neg_zero: self.may_neg_zero || rhs.may_neg_zero || (lo < 0.0 && hi >= 0.0),
        }
    }

    pub(crate) fn widen(self, next: Self) -> Self {
        if !self.has_finite() {
            return next;
        }
        let joined = self.join(next);
        Self {
            lo: if next.has_finite() && next.lo < self.lo {
                -f32::MAX
            } else {
                joined.lo
            },
            hi: if next.has_finite() && next.hi > self.hi {
                f32::MAX
            } else {
                joined.hi
            },
            ..joined
        }
    }

    pub(crate) fn refine(
        self,
        rhs: Self,
        comparison: Comparison,
        truth: bool,
    ) -> Option<(Self, Self)> {
        if !truth
            && (self.may_nan || rhs.may_nan)
            && matches!(
                comparison,
                Comparison::Lt | Comparison::Le | Comparison::Gt | Comparison::Ge | Comparison::Eq
            )
        {
            return Some((self, rhs));
        }
        match (comparison, truth) {
            (Comparison::Lt, true) => refine_ordered(self, rhs, true),
            (Comparison::Lt, false) => refine_ordered(rhs, self, false).map(swap),
            (Comparison::Le, true) => refine_ordered(self, rhs, false),
            (Comparison::Le, false) => refine_ordered(rhs, self, true).map(swap),
            (Comparison::Gt, truth) => rhs.refine(self, Comparison::Lt, truth).map(swap),
            (Comparison::Ge, truth) => rhs.refine(self, Comparison::Le, truth).map(swap),
            (Comparison::Eq, true) | (Comparison::Ne, false) => {
                let lo = self.lo.max(rhs.lo);
                let hi = self.hi.min(rhs.hi);
                let mut left = self.with_bounds(lo, hi);
                let mut right = rhs.with_bounds(lo, hi);
                left.may_nan = false;
                right.may_nan = false;
                viable(left, right)
            }
            (Comparison::Eq, false) | (Comparison::Ne, true) => Some((self, rhs)),
        }
    }

    fn map_bounds(self, map: impl Fn(f64) -> f64) -> Self {
        if !self.has_finite() {
            return self;
        }
        let mut result = bounds(map(self.lo as f64), map(self.hi as f64), self.may_nan);
        result.may_pos_inf |= self.may_pos_inf;
        result.may_neg_inf |= self.may_neg_inf;
        result.may_neg_zero |= self.may_neg_zero;
        result
    }

    fn with_bounds(mut self, lo: f32, hi: f32) -> Self {
        self.lo = lo;
        self.hi = hi;
        self.may_neg_zero &= lo <= 0.0 && hi >= 0.0;
        self
    }
}

pub(crate) fn dot(left: &[Interval], right: &[Interval]) -> Interval {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(Interval::constant(0.0), |sum, (&left, &right)| {
            sum.add(left.mul(right))
        })
}

pub(crate) fn normalize(value: &[Interval]) -> Vec<Interval> {
    let length = dot(value, value).sqrt();
    value
        .iter()
        .map(|component| component.div(length))
        .collect()
}

fn bounds(lo: f64, hi: f64, may_nan: bool) -> Interval {
    if lo.is_nan() || hi.is_nan() || lo > hi {
        return Interval::exceptional(true, false, false, false);
    }
    let may_neg_inf = lo < -(f32::MAX as f64);
    let may_pos_inf = hi > f32::MAX as f64;
    let lo = lower_f32(lo.max(-(f32::MAX as f64)));
    let hi = upper_f32(hi.min(f32::MAX as f64));
    Interval {
        lo,
        hi,
        may_nan,
        may_pos_inf,
        may_neg_inf,
        may_neg_zero: lo <= 0.0 && hi >= 0.0,
    }
}

fn from_candidates(values: &[f64], may_nan: bool) -> Interval {
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    bounds(lo, hi, may_nan)
}

fn lower_f32(value: f64) -> f32 {
    let rounded = value as f32;
    if rounded as f64 > value {
        next_down(rounded)
    } else {
        rounded
    }
}

fn upper_f32(value: f64) -> f32 {
    let rounded = value as f32;
    if (rounded as f64) < value {
        next_up(rounded)
    } else {
        rounded
    }
}

fn next_down(value: f32) -> f32 {
    if value == f32::NEG_INFINITY {
        value
    } else if value == 0.0 {
        -f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits() - 1)
    } else {
        f32::from_bits(value.to_bits() + 1)
    }
}

fn next_up(value: f32) -> f32 {
    if value == f32::INFINITY {
        value
    } else if value == 0.0 {
        f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits() + 1)
    } else {
        f32::from_bits(value.to_bits() - 1)
    }
}

fn refine_ordered(left: Interval, right: Interval, strict: bool) -> Option<(Interval, Interval)> {
    let left_hi = if strict {
        left.hi.min(next_down(right.hi))
    } else {
        left.hi.min(right.hi)
    };
    let right_lo = if strict {
        right.lo.max(next_up(left.lo))
    } else {
        right.lo.max(left.lo)
    };
    let mut left = left.with_bounds(left.lo, left_hi);
    let mut right = right.with_bounds(right_lo, right.hi);
    left.may_nan = false;
    right.may_nan = false;
    viable(left, right)
}

fn viable(left: Interval, right: Interval) -> Option<(Interval, Interval)> {
    let left_exists = left.has_finite() || left.has_infinity() || left.may_neg_zero;
    let right_exists = right.has_finite() || right.has_infinity() || right.may_neg_zero;
    (left_exists && right_exists).then_some((left, right))
}

fn swap((left, right): (Interval, Interval)) -> (Interval, Interval) {
    (right, left)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Rng(u64);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 as u32
        }

        fn finite(&mut self, low: f32, high: f32) -> f32 {
            let unit = self.next_u32() as f64 / u32::MAX as f64;
            (low as f64 + unit * (high - low) as f64) as f32
        }
    }

    fn assert_contains(interval: Interval, concrete: f32) {
        assert!(
            interval.contains(concrete),
            "{interval:?} does not contain {concrete:?}"
        );
    }

    #[test]
    fn tracks_exceptional_values_and_negative_zero() {
        let nan = Interval::constant(f32::NAN);
        assert!(nan.may_nan);
        assert!(Interval::constant(f32::INFINITY).may_pos_inf);
        assert!(Interval::constant(f32::NEG_INFINITY).may_neg_inf);
        assert!(Interval::constant(-0.0).may_neg_zero);
        assert!(Interval::new(-0.0, 1.0).may_neg_zero);
        assert!(!Interval::new(1.0, 2.0).has_exceptional_values());
        assert_contains(nan.min(Interval::constant(2.0)), 2.0);
        assert_contains(nan.max(Interval::constant(2.0)), 2.0);
    }

    #[test]
    fn scalar_transfers_cover_the_required_operations() {
        let x = Interval::new(2.0, 4.0);
        let y = Interval::new(5.0, 8.0);
        assert_contains(x.add(y), 12.0);
        assert_contains(x.sub(y), -6.0);
        assert_contains(x.mul(y), 32.0);
        assert_contains(x.div(y), 0.25);
        assert_contains(x.sqrt(), 2.0);
        assert_contains(x.inverse_sqrt(), 1.0 / 2.0_f32.sqrt());
        assert_contains(x.min(y), 4.0);
        assert_contains(x.max(y), 8.0);
        assert_contains(
            y.clamp(Interval::constant(6.0), Interval::constant(7.0)),
            7.0,
        );
        assert_contains(x.mix(y, Interval::constant(0.25)), 5.0);
        assert_eq!(Interval::select(x, y, Some(false)), x);
        assert_eq!(Interval::select(x, y, Some(true)), y);
        assert_eq!(Interval::select(x, y, None), x.join(y));
        assert_contains(x.fma(y, Interval::constant(1.0)), 33.0);
    }

    #[test]
    fn dot_and_normalize_are_component_wise() {
        let vector = [
            Interval::constant(3.0),
            Interval::constant(4.0),
            Interval::constant(0.0),
        ];
        assert_contains(dot(&vector, &vector), 25.0);
        let normalized = normalize(&vector);
        assert_contains(normalized[0], 0.6);
        assert_contains(normalized[1], 0.8);
        assert_contains(normalized[2], 0.0);
    }

    #[test]
    fn branch_refinement_join_and_widen_are_conservative() {
        let left = Interval::new(-10.0, 10.0);
        let right = Interval::new(0.0, 20.0);
        let (left_true, right_true) = left.refine(right, Comparison::Lt, true).unwrap();
        assert!(left_true.hi < right.hi);
        assert!(right_true.lo > left.lo);
        assert_contains(left_true, 9.0);
        assert_contains(right_true, 10.0);

        let (left_false, right_false) = left.refine(right, Comparison::Lt, false).unwrap();
        assert_contains(left_false, 5.0);
        assert_contains(right_false, 5.0);

        let (equal_left, equal_right) = left.refine(right, Comparison::Eq, true).unwrap();
        assert_eq!((equal_left.lo, equal_left.hi), (0.0, 10.0));
        assert_eq!((equal_right.lo, equal_right.hi), (0.0, 10.0));
        assert!(equal_left.may_neg_zero);
        assert!(!equal_right.may_neg_zero);
        assert!(Interval::new(2.0, 3.0)
            .refine(Interval::new(0.0, 1.0), Comparison::Le, true)
            .is_none());
        assert!(left.refine(right, Comparison::Gt, true).is_some());
        assert!(left.refine(right, Comparison::Ge, false).is_some());
        assert!(left.refine(right, Comparison::Ne, true).is_some());

        let (nan_left, _) = Interval::constant(f32::NAN)
            .refine(Interval::constant(1.0), Comparison::Lt, false)
            .expect("NaN makes an ordered comparison false");
        assert!(nan_left.may_nan);

        assert_eq!(left.join(right), Interval::new(-10.0, 20.0));
        assert!(
            Interval::constant(-1.0)
                .join(Interval::constant(1.0))
                .may_neg_zero
        );
        let widened = Interval::new(-1.0, 1.0).widen(Interval::new(-2.0, 2.0));
        assert_eq!((widened.lo, widened.hi), (-f32::MAX, f32::MAX));
    }

    #[test]
    fn unsafe_operations_remain_unproven() {
        let division = Interval::new(-1.0, 1.0).div(Interval::new(-1.0, 1.0));
        assert!(division.may_nan && division.may_pos_inf && division.may_neg_inf);
        assert!(Interval::new(-1.0, 4.0).sqrt().may_nan);
        assert!(Interval::new(0.0, 4.0).inverse_sqrt().may_pos_inf);
        assert!(normalize(&[Interval::constant(0.0); 3])
            .iter()
            .all(Interval::has_exceptional_values));
    }

    #[test]
    fn one_million_deterministic_samples_are_contained() {
        let mut rng = Rng(0x4d59_5df4_d0f3_3173);
        for _ in 0..1_000_000 {
            let x = rng.finite(-100.0, 100.0);
            let y = rng.finite(0.25, 100.0);
            let t = rng.finite(0.0, 1.0);
            let ax = Interval::constant(x);
            let ay = Interval::constant(y);
            let at = Interval::constant(t);

            assert_contains(ax.add(ay), x + y);
            assert_contains(ax.sub(ay), x - y);
            assert_contains(ax.mul(ay), x * y);
            assert_contains(ax.div(ay), x / y);
            assert_contains(ay.sqrt(), y.sqrt());
            assert_contains(ay.inverse_sqrt(), 1.0 / y.sqrt());
            assert_contains(ax.min(ay), x.min(y));
            assert_contains(ax.max(ay), x.max(y));
            assert_contains(
                ax.clamp(Interval::constant(-5.0), Interval::constant(5.0)),
                x.clamp(-5.0, 5.0),
            );
            assert_contains(ax.mix(ay, at), x * (1.0 - t) + y * t);
            assert_contains(
                Interval::select(ax, ay, None),
                if rng.next_u32() & 1 == 0 { x } else { y },
            );
            assert_contains(ax.fma(ay, at), x.mul_add(y, t));

            let concrete = [x, y, t];
            let abstracted = concrete.map(Interval::constant);
            let concrete_dot = concrete.iter().map(|v| v * v).sum::<f32>();
            assert_contains(dot(&abstracted, &abstracted), concrete_dot);
            let concrete_length = concrete_dot.sqrt();
            for (component, concrete) in normalize(&abstracted)
                .into_iter()
                .zip(concrete.map(|v| v / concrete_length))
            {
                assert_contains(component, concrete);
            }
        }
    }
}
