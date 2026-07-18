//! Deterministic snap-rounded polygon overlay.
//!
//! Segment discovery follows the Bentley--Ottmann sweep model with a fixed
//! lexicographic event order.  This correctness-first v1 checks every active
//! segment rather than only neighboring status entries, retaining the sweep's
//! completeness while accepting O(n^2) worst-case work.  Intersections are
//! represented as exact expansion ratios until the Hobby/Guibas--Marimont
//! hot-pixel snap-rounding step.  Every segment crossing a hot pixel is routed
//! through its center; therefore vertices move by at most half a pixel diagonal
//! and rounding introduces no new crossings.

mod faces;
mod rings;
pub mod snap;
pub mod sweep;
pub mod validity;

use std::fmt;

pub use validity::{is_valid_polygonal, ValidityReport};

/// A finite planar point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn as_array(self) -> [f64; 2] {
        [self.x, self.y]
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

pub type Ring = Vec<Point>;

#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    pub exterior: Ring,
    pub holes: Vec<Ring>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MultiPolygon(pub Vec<Polygon>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}

impl BooleanOp {
    fn evaluate(self, left: bool, right: bool) -> bool {
        match self {
            Self::Union => left || right,
            Self::Intersection => left && right,
            Self::Difference => left && !right,
            Self::SymmetricDifference => left ^ right,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverlayResult {
    pub geometry: MultiPolygon,
    pub max_snap_motion: f64,
    pub snap_motion_bound: f64,
    pub snap_step: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayError(pub String);

impl fmt::Display for OverlayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for OverlayError {}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Segment {
    pub start: Point,
    pub end: Point,
}

fn polygon_segments(polygonal: &MultiPolygon) -> Vec<Segment> {
    let mut segments = Vec::new();
    for polygon in &polygonal.0 {
        for ring in std::iter::once(&polygon.exterior).chain(&polygon.holes) {
            for pair in ring.windows(2) {
                if pair[0] != pair[1] {
                    segments.push(Segment {
                        start: pair[0],
                        end: pair[1],
                    });
                }
            }
        }
    }
    segments
}

fn snap_polygonal(value: &MultiPolygon, grid: snap::SnapGrid) -> MultiPolygon {
    MultiPolygon(
        value
            .0
            .iter()
            .map(|polygon| Polygon {
                exterior: polygon
                    .exterior
                    .iter()
                    .copied()
                    .map(|point| grid.snap(point))
                    .collect(),
                holes: polygon
                    .holes
                    .iter()
                    .map(|ring| ring.iter().copied().map(|point| grid.snap(point)).collect())
                    .collect(),
            })
            .collect(),
    )
}

/// Execute one deterministic snap-rounded polygonal boolean operation.
pub fn overlay(
    left: &MultiPolygon,
    right: &MultiPolygon,
    operation: BooleanOp,
) -> Result<OverlayResult, OverlayError> {
    let left_validity = is_valid_polygonal(left);
    if !left_validity.valid {
        return Err(OverlayError(format!(
            "invalid left polygonal input: {}",
            left_validity.reasons.join("; ")
        )));
    }
    let right_validity = is_valid_polygonal(right);
    if !right_validity.valid {
        return Err(OverlayError(format!(
            "invalid right polygonal input: {}",
            right_validity.reasons.join("; ")
        )));
    }

    let grid = snap::SnapGrid::for_polygonals(left, right)?;
    let snapped_left = snap_polygonal(left, grid);
    let snapped_right = snap_polygonal(right, grid);
    let mut segments = polygon_segments(&snapped_left);
    segments.extend(polygon_segments(&snapped_right));
    let atomic = sweep::snap_rounded_atomic_segments(&segments, grid)?;
    let geometry = faces::assemble(&atomic, &snapped_left, &snapped_right, operation, grid)?;
    let report = is_valid_polygonal(&geometry);
    if !report.valid {
        return Err(OverlayError(format!(
            "constructive overlay produced invalid topology: {}; geometry={geometry:?}",
            report.reasons.join("; "),
        )));
    }
    let max_snap_motion = left
        .0
        .iter()
        .chain(&right.0)
        .flat_map(|polygon| std::iter::once(&polygon.exterior).chain(&polygon.holes))
        .flat_map(|ring| ring.iter().copied())
        .map(|point| grid.motion(point))
        .fold(0.0, f64::max);
    Ok(OverlayResult {
        geometry,
        max_snap_motion,
        snap_motion_bound: grid.motion_bound(),
        snap_step: grid.step(),
    })
}

#[cfg(test)]
mod tests;
