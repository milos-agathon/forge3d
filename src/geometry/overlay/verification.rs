//! Test-facing deterministic overlay fuzz and conservation gate.

use std::panic::{catch_unwind, AssertUnwindSafe};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sha2::{Digest, Sha256};

use super::{is_valid_polygonal, overlay, BooleanOp, MultiPolygon, Point, Polygon, Ring};

#[derive(Clone, Copy)]
pub(super) struct Rng(pub(super) u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn coordinate(&mut self) -> f64 {
        (self.next() as i64 % 1024) as f64 / 16.0
    }
}

fn polygon(ring: Ring) -> MultiPolygon {
    MultiPolygon(vec![Polygon {
        exterior: ring,
        holes: Vec::new(),
    }])
}

fn rectangle(x: f64, y: f64, width: f64, height: f64) -> MultiPolygon {
    polygon(vec![
        Point::new(x, y),
        Point::new(x + width, y),
        Point::new(x + width, y + height),
        Point::new(x, y + height),
        Point::new(x, y),
    ])
}

pub(super) fn pair(index: u64, rng: &mut Rng) -> (MultiPolygon, MultiPolygon) {
    let x = rng.coordinate();
    let y = rng.coordinate();
    let width = 1.0 + (rng.next() % 64) as f64 / 16.0;
    let height = 1.0 + (rng.next() % 64) as f64 / 16.0;
    let left = rectangle(x, y, width, height);
    let right = match index % 6 {
        0 => rectangle(rng.coordinate(), rng.coordinate(), width, height),
        1 => rectangle(x + width, y, width, height),
        2 => rectangle(
            f64::from_bits((x + width).to_bits() + index % 2),
            y + height * 0.25,
            width,
            height,
        ),
        3 => polygon(vec![
            Point::new(x + width * 0.5, y - height),
            Point::new(x + width * 1.5, y - height),
            Point::new(x + width * 1.5, y),
            Point::new(x + width, y),
            Point::new(x + width * 0.5, y),
            Point::new(x + width * 0.5, y - height),
        ]),
        4 => polygon(vec![
            Point::new(x - width, y + height * 0.5),
            Point::new(x + width * 2.0, y + height * 0.5),
            Point::new(x + width * 2.0, y + height * 0.5 + f64::EPSILON * 64.0),
            Point::new(x - width, y + height * 0.5),
        ]),
        _ => polygon(vec![
            Point::new(x - width, y - height),
            Point::new(x + width * 2.0, y - height),
            Point::new(x + width * 2.0, y + height * 2.0),
            Point::new(x + width * 0.5, y + height * 0.75),
            Point::new(x - width, y + height * 2.0),
            Point::new(x - width, y - height),
        ]),
    };
    (left, right)
}

fn ring_area(ring: &Ring) -> f64 {
    ring.windows(2)
        .map(|edge| edge[0].x * edge[1].y - edge[0].y * edge[1].x)
        .sum::<f64>()
        * 0.5
}

pub(super) fn area(value: &MultiPolygon) -> f64 {
    value
        .0
        .iter()
        .map(|polygon| {
            ring_area(&polygon.exterior).abs()
                - polygon
                    .holes
                    .iter()
                    .map(|ring| ring_area(ring).abs())
                    .sum::<f64>()
        })
        .sum()
}

pub(super) fn perimeter(value: &MultiPolygon) -> f64 {
    value
        .0
        .iter()
        .flat_map(|polygon| std::iter::once(&polygon.exterior).chain(&polygon.holes))
        .flat_map(|ring| ring.windows(2))
        .map(|edge| (edge[1].x - edge[0].x).hypot(edge[1].y - edge[0].y))
        .sum()
}

fn hash_geometry(hasher: &mut Sha256, operation: BooleanOp, value: &MultiPolygon) {
    hasher.update([operation as u8]);
    hasher.update((value.0.len() as u64).to_le_bytes());
    for polygon in &value.0 {
        hasher.update((polygon.holes.len() as u64).to_le_bytes());
        for ring in std::iter::once(&polygon.exterior).chain(&polygon.holes) {
            hasher.update((ring.len() as u64).to_le_bytes());
            for point in ring {
                hasher.update(point.x.to_bits().to_le_bytes());
                hasher.update(point.y.to_bits().to_le_bytes());
            }
        }
    }
}

#[derive(Default)]
struct Report {
    operations: u64,
    errors: u64,
    panics: u64,
    invalid_outputs: u64,
    snap_bound_violations: u64,
    conservation_violations: u64,
    max_snap_motion: f64,
    max_snap_bound: f64,
    max_conservation_error: f64,
    max_conservation_bound: f64,
    first_error: Option<String>,
}

fn run(cases: u64, seed: u64, collect: bool) -> (Report, String) {
    let mut rng = Rng(seed.max(1));
    let mut report = Report::default();
    let mut hash = Sha256::new();
    for index in 0..cases {
        let (left, right) = pair(index, &mut rng);
        let mut union = None;
        let mut intersection = None;
        for operation in [
            BooleanOp::Union,
            BooleanOp::Intersection,
            BooleanOp::Difference,
            BooleanOp::SymmetricDifference,
        ] {
            report.operations += 1;
            let result = catch_unwind(AssertUnwindSafe(|| overlay(&left, &right, operation)));
            let Ok(result) = result else {
                report.panics += 1;
                continue;
            };
            let Ok(result) = result else {
                report.errors += 1;
                if report.first_error.is_none() {
                    report.first_error = Some(format!("case={index} op={operation:?}: {result:?}"));
                }
                continue;
            };
            hash_geometry(&mut hash, operation, &result.geometry);
            if collect {
                report.invalid_outputs += u64::from(!is_valid_polygonal(&result.geometry).valid);
                report.snap_bound_violations +=
                    u64::from(result.max_snap_motion > result.snap_motion_bound);
                report.max_snap_motion = report.max_snap_motion.max(result.max_snap_motion);
                report.max_snap_bound = report.max_snap_bound.max(result.snap_motion_bound);
            }
            match operation {
                BooleanOp::Union => union = Some(result),
                BooleanOp::Intersection => intersection = Some(result),
                _ => {}
            }
        }
        if collect {
            if let (Some(union), Some(intersection)) = (union, intersection) {
                let conservation = (area(&union.geometry) + area(&intersection.geometry)
                    - area(&left)
                    - area(&right))
                .abs();
                // A closed curve displaced by at most delta changes area by
                // at most perimeter*delta + pi*delta^2.  Applying that bound
                // to both inputs and both outputs yields this four-boundary
                // snap-rounding conservation envelope.
                let delta = union.snap_motion_bound.max(intersection.snap_motion_bound);
                let bound = delta
                    * (perimeter(&left)
                        + perimeter(&right)
                        + perimeter(&union.geometry)
                        + perimeter(&intersection.geometry))
                    + 4.0 * std::f64::consts::PI * delta * delta;
                report.conservation_violations += u64::from(conservation > bound);
                report.max_conservation_error = report.max_conservation_error.max(conservation);
                report.max_conservation_bound = report.max_conservation_bound.max(bound);
            }
        }
    }
    (report, format!("{:x}", hash.finalize()))
}

/// Run the deterministic 100k-pair boolean topology hard gate twice.
#[pyfunction(
    name = "_euclidea_boolean_fuzz_report",
    signature = (cases=100_000, seed=0x4555_434c_4944_4541_u64)
)]
pub fn euclidea_boolean_fuzz_report_py(
    py: Python<'_>,
    cases: u64,
    seed: u64,
) -> PyResult<PyObject> {
    let ((report, hash_a), (_, hash_b)) =
        py.allow_threads(|| (run(cases, seed, true), run(cases, seed, false)));
    #[cfg(feature = "geos-topology")]
    let oracle = py.allow_threads(|| super::verification_oracle::run(2_000, seed));
    let output = PyDict::new_bound(py);
    output.set_item("cases", cases)?;
    output.set_item("operations", report.operations)?;
    output.set_item("errors", report.errors)?;
    output.set_item("panics", report.panics)?;
    output.set_item("invalid_outputs", report.invalid_outputs)?;
    output.set_item("snap_bound_violations", report.snap_bound_violations)?;
    output.set_item("conservation_violations", report.conservation_violations)?;
    output.set_item("max_snap_motion", report.max_snap_motion)?;
    output.set_item("max_snap_bound", report.max_snap_bound)?;
    output.set_item("max_conservation_error", report.max_conservation_error)?;
    output.set_item("max_conservation_bound", report.max_conservation_bound)?;
    output.set_item("first_error", report.first_error)?;
    output.set_item("hash_a", hash_a)?;
    output.set_item("hash_b", hash_b)?;
    output.set_item("seed", seed)?;
    #[cfg(feature = "geos-topology")]
    {
        output.set_item("oracle_enabled", true)?;
        output.set_item("oracle_cases", oracle.cases)?;
        output.set_item("oracle_disagreements", oracle.disagreements)?;
        output.set_item("exact_benchmark_ms", oracle.exact_ms)?;
        output.set_item("geo_benchmark_ms", oracle.geo_ms)?;
        output.set_item("benchmark_ratio", oracle.ratio)?;
    }
    #[cfg(not(feature = "geos-topology"))]
    output.set_item("oracle_enabled", false)?;
    Ok(output.into_py(py))
}
