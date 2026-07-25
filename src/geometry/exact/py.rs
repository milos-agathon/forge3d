//! Test-facing exhaustive predicate adjudication harness.

use super::{oracle, predicates};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn integer(&mut self) -> f64 {
        (self.next() as i64 % 2_000_001) as f64 - 1_000_000.0
    }
}

#[derive(Default)]
struct Counts {
    exact_errors: u64,
    ablation_errors: u64,
    filter: u64,
    adaptive: u64,
    full: u64,
    orientation: u64,
    incircle: u64,
}

fn stage(counts: &mut Counts, value: predicates::PredicateStage) {
    match value {
        predicates::PredicateStage::Filter => counts.filter += 1,
        predicates::PredicateStage::Adaptive => counts.adaptive += 1,
        predicates::PredicateStage::Full => counts.full += 1,
    }
}

fn orientation(counts: &mut Counts, a: [f64; 2], b: [f64; 2], c: [f64; 2]) {
    let expected = oracle::orient2d(a, b, c);
    let (actual, used) = predicates::orient2d_with_stage(a, b, c);
    counts.exact_errors += u64::from(predicates::sign_ordering(actual) != expected);
    counts.ablation_errors +=
        u64::from(predicates::sign_ordering(predicates::orient2d_fast(a, b, c)) != expected);
    counts.orientation += 1;
    stage(counts, used);
}

fn circle(counts: &mut Counts, points: [[f64; 2]; 4]) {
    let [a, b, c, d] = points;
    let expected = oracle::incircle(a, b, c, d);
    let (actual, used) = predicates::incircle_with_stage(a, b, c, d);
    counts.exact_errors += u64::from(predicates::sign_ordering(actual) != expected);
    counts.ablation_errors +=
        u64::from(predicates::sign_ordering(predicates::incircle_fast(a, b, c, d)) != expected);
    counts.incircle += 1;
    stage(counts, used);
}

fn next_up(value: f64, steps: u64) -> f64 {
    f64::from_bits(value.to_bits() + steps)
}

fn run(random_cases: u64, adversarial_cases: u64, seed: u64) -> Counts {
    let mut rng = Rng(seed.max(1));
    let mut counts = Counts::default();
    for index in 0..random_cases {
        if index & 1 == 0 {
            orientation(
                &mut counts,
                [rng.integer(), rng.integer()],
                [rng.integer(), rng.integer()],
                [rng.integer(), rng.integer()],
            );
        } else {
            circle(
                &mut counts,
                [
                    [rng.integer(), rng.integer()],
                    [rng.integer(), rng.integer()],
                    [rng.integer(), rng.integer()],
                    [rng.integer(), rng.integer()],
                ],
            );
        }
    }

    // Integer products around 2^54 lose their low two bits in ordinary f64.
    // The determinant below is exactly one before the 0--4 ULP perturbation,
    // providing a reproducible cancellation family rather than hoping random
    // inputs happen to reach the adaptive path.
    for index in 0..adversarial_cases {
        let ulps = index % 5;
        if index & 1 == 0 {
            let n = 134_217_728.0 + (rng.next() % 4096) as f64;
            orientation(
                &mut counts,
                [0.0, 0.0],
                [n + 1.0, n],
                [next_up(n + 2.0, ulps), n + 1.0],
            );
        } else {
            let radius = 134_217_728.0 + (rng.next() % 4096) as f64;
            circle(
                &mut counts,
                [
                    [radius, 0.0],
                    [0.0, radius],
                    [-radius, 0.0],
                    [0.0, -next_up(radius, ulps)],
                ],
            );
        }
    }
    counts
}

/// Execute the CPU-only hard gate against the independent exact-dyadic oracle.
#[pyfunction(
    name = "_euclidea_predicate_report",
    signature = (random_cases=10_000_000, adversarial_cases=1_000_000, seed=0x32e0_c1de_u64)
)]
pub fn euclidea_predicate_report_py(
    py: Python<'_>,
    random_cases: u64,
    adversarial_cases: u64,
    seed: u64,
) -> PyResult<PyObject> {
    let counts = py.allow_threads(|| run(random_cases, adversarial_cases, seed));
    let report = PyDict::new_bound(py);
    report.set_item("random_cases", random_cases)?;
    report.set_item("adversarial_cases", adversarial_cases)?;
    report.set_item("orientation_cases", counts.orientation)?;
    report.set_item("incircle_cases", counts.incircle)?;
    report.set_item("exact_errors", counts.exact_errors)?;
    report.set_item("ablation_errors", counts.ablation_errors)?;
    report.set_item("filter_accepts", counts.filter)?;
    report.set_item("filter_rejects", counts.adaptive + counts.full)?;
    report.set_item("adaptive_accepts", counts.adaptive)?;
    report.set_item("full_expansions", counts.full)?;
    report.set_item("seed", seed)?;
    Ok(report.into_py(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_family_proves_the_fast_path_is_not_robust() {
        let counts = run(10_000, 20_000, 0x32e0_c1de);
        assert_eq!(counts.exact_errors, 0);
        assert!(
            counts.ablation_errors >= 1_000,
            "{}",
            counts.ablation_errors
        );
        assert!(counts.adaptive + counts.full > 0);
    }
}
