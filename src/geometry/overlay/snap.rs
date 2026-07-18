//! Power-of-two hot-pixel grid for deterministic snap rounding.

use super::{MultiPolygon, OverlayError, Point, Segment};

#[derive(Debug, Clone, Copy)]
pub struct SnapGrid {
    step: f64,
}

impl SnapGrid {
    pub fn for_polygonals(left: &MultiPolygon, right: &MultiPolygon) -> Result<Self, OverlayError> {
        let mut maximum = 0.0f64;
        let mut any = false;
        for point in left
            .0
            .iter()
            .chain(&right.0)
            .flat_map(|polygon| std::iter::once(&polygon.exterior).chain(&polygon.holes))
            .flat_map(|ring| ring.iter().copied())
        {
            if !point.is_finite() {
                return Err(OverlayError(
                    "non-finite coordinate rejected before polygon sweep".to_string(),
                ));
            }
            any = true;
            maximum = maximum.max(point.x.abs()).max(point.y.abs());
        }
        if !any {
            return Ok(Self {
                step: 2f64.powi(-50),
            });
        }
        let exponent = if maximum == 0.0 {
            -1
        } else {
            ((maximum.to_bits() >> 52) & 0x7ff) as i32 - 1023
        };
        // Four ulps at the largest coordinate.  A power-of-two cell makes the
        // round-to-even index and its reconstruction byte-identical.
        let step = 2f64.powi((exponent - 50).clamp(-1070, 970));
        Ok(Self { step })
    }

    pub fn step(self) -> f64 {
        self.step
    }

    pub fn snap(self, point: Point) -> Point {
        Point::new(
            (point.x / self.step).round_ties_even() * self.step,
            (point.y / self.step).round_ties_even() * self.step,
        )
    }

    pub fn motion(self, point: Point) -> f64 {
        let snapped = self.snap(point);
        (snapped.x - point.x).hypot(snapped.y - point.y)
    }

    pub fn motion_bound(self) -> f64 {
        self.step * std::f64::consts::FRAC_1_SQRT_2
    }

    pub(crate) fn hot_pixel_intersects(self, segment: Segment, center: Point) -> bool {
        let half = self.step * 0.5;
        if center.x + half < segment.start.x.min(segment.end.x)
            || center.x - half > segment.start.x.max(segment.end.x)
            || center.y + half < segment.start.y.min(segment.end.y)
            || center.y - half > segment.start.y.max(segment.end.y)
        {
            return false;
        }
        let dx = segment.end.x - segment.start.x;
        let dy = segment.end.y - segment.start.y;
        let length2 = dx * dx + dy * dy;
        if length2 == 0.0 {
            return (center.x - segment.start.x).abs() <= half
                && (center.y - segment.start.y).abs() <= half;
        }
        let projection = (((center.x - segment.start.x) * dx + (center.y - segment.start.y) * dy)
            / length2)
            .clamp(0.0, 1.0);
        let closest = Point::new(
            segment.start.x + projection * dx,
            segment.start.y + projection * dy,
        );
        (closest.x - center.x).abs() <= half && (closest.y - center.y).abs() <= half
    }
}
