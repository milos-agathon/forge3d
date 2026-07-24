use super::types::{CoverageGeometry, CoverageLayer, FillRule, PrimitiveRecord};
use crate::core::error::RenderError;
use crate::vector::api::{PolygonDef, PolylineDef};
use glam::Vec2;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

const GEOMETRY_EPSILON: f32 = 1.0e-6;

/// Converts pre-tessellation vector inputs into analytic boundary records.
pub struct CoverageGeometryBuilder {
    width: u32,
    height: u32,
    primitives: Vec<PrimitiveRecord>,
    layers: Vec<CoverageLayer>,
    next_stable_id: u32,
}

impl CoverageGeometryBuilder {
    pub fn new(width: u32, height: u32) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::Upload(
                "vector_coverage_invalid_extent: width and height must be non-zero".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            primitives: Vec::new(),
            layers: Vec::new(),
            next_stable_id: 0,
        })
    }

    pub fn add_layer(
        &mut self,
        name: impl Into<String>,
        fill_rule: FillRule,
        color: [f32; 4],
    ) -> Result<u32, RenderError> {
        if !color.into_iter().all(f32::is_finite)
            || color
                .into_iter()
                .any(|channel| !(0.0..=1.0).contains(&channel))
        {
            return Err(RenderError::Upload(
                "vector_coverage_invalid_color: RGBA channels must be finite and in [0,1]".into(),
            ));
        }
        let layer = u32::try_from(self.layers.len()).map_err(|_| {
            RenderError::Upload("vector_coverage_layer_overflow: more than u32::MAX layers".into())
        })?;
        self.layers.push(CoverageLayer {
            fill_rule,
            color,
            name: name.into(),
        });
        Ok(layer)
    }

    pub fn push_polygon(&mut self, layer: u32, polygon: &PolygonDef) -> Result<(), RenderError> {
        self.validate_layer(layer)?;
        self.push_ring(layer, &polygon.exterior, true)?;
        for hole in &polygon.holes {
            self.push_ring(layer, hole, false)?;
        }
        Ok(())
    }

    /// Expand a round-cap/round-join stroke as the union of exact capsules.
    ///
    /// A round stroke is the Minkowski sum `polyline ⊕ disk(radius)`.  The
    /// distributive identity over a segment chain makes this exactly the union
    /// of `segment ⊕ disk(radius)` capsules.  Keeping every capsule in the same
    /// nonzero layer makes overlaps (collinear joints and 180-degree folds
    /// included) a single filled region instead of double coverage.
    pub fn push_round_polyline(
        &mut self,
        layer: u32,
        polyline: &PolylineDef,
    ) -> Result<(), RenderError> {
        self.validate_layer(layer)?;
        if self.layers[layer as usize].fill_rule != FillRule::NonZero {
            return Err(RenderError::Upload(
                "vector_coverage_stroke_fill_rule: analytic strokes require nonzero fill".into(),
            ));
        }
        let radius = 0.5 * polyline.style.stroke_width;
        if !radius.is_finite() || radius <= 0.0 {
            return Err(RenderError::Upload(
                "vector_coverage_invalid_stroke_width: stroke width must be finite and positive"
                    .into(),
            ));
        }
        if polyline.path.len() < 2 {
            return Err(RenderError::Upload(
                "vector_coverage_invalid_polyline: at least two vertices are required".into(),
            ));
        }

        let mut emitted = 0_u32;
        for pair in polyline.path.windows(2) {
            let p0 = pair[0];
            let p1 = pair[1];
            self.validate_point(p0)?;
            self.validate_point(p1)?;
            let delta = p1 - p0;
            let length = delta.length();
            if length <= GEOMETRY_EPSILON {
                continue;
            }
            let tangent = delta / length;
            let normal = Vec2::new(-tangent.y, tangent.x) * radius;

            // Clockwise capsule boundary.  Nonzero fill is orientation-agnostic
            // as long as every capsule uses the same orientation.
            self.push_line(layer, p0 + normal, p1 + normal)?;
            self.push_arc(layer, p1, radius, f64::from(normal.y.atan2(normal.x)), -PI)?;
            self.push_line(layer, p1 - normal, p0 - normal)?;
            self.push_arc(
                layer,
                p0,
                radius,
                f64::from((-normal.y).atan2(-normal.x)),
                -PI,
            )?;
            emitted += 1;
        }
        if emitted == 0 {
            return Err(RenderError::Upload(
                "vector_coverage_degenerate_polyline: all segments have zero length".into(),
            ));
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<CoverageGeometry, RenderError> {
        if self.layers.is_empty() {
            return Err(RenderError::Upload(
                "vector_coverage_no_layers: at least one layer is required".into(),
            ));
        }
        // Atomic insertion order in the GPU bin pass is intentionally ignored.
        // Stable IDs are the canonical order the raster kernel restores.
        self.primitives
            .sort_by_key(|primitive| primitive.stable_id());
        Ok(CoverageGeometry {
            width: self.width,
            height: self.height,
            primitives: self.primitives,
            layers: self.layers,
        })
    }

    fn validate_layer(&self, layer: u32) -> Result<(), RenderError> {
        if layer as usize >= self.layers.len() {
            return Err(RenderError::Upload(format!(
                "vector_coverage_unknown_layer: {layer}"
            )));
        }
        Ok(())
    }

    fn validate_point(&self, point: Vec2) -> Result<(), RenderError> {
        if !point.is_finite() {
            return Err(RenderError::Upload(
                "vector_coverage_non_finite_geometry: coordinates must be finite".into(),
            ));
        }
        Ok(())
    }

    fn push_ring(&mut self, layer: u32, input: &[Vec2], exterior: bool) -> Result<(), RenderError> {
        let mut ring = input.to_vec();
        if ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 {
            return Err(RenderError::Upload(
                "vector_coverage_invalid_ring: at least three distinct vertices are required"
                    .into(),
            ));
        }
        for &point in &ring {
            self.validate_point(point)?;
        }
        ring.dedup_by(|a, b| (*a - *b).length_squared() <= GEOMETRY_EPSILON.powi(2));
        if ring.len() < 3 {
            return Err(RenderError::Upload(
                "vector_coverage_degenerate_ring: fewer than three vertices remain".into(),
            ));
        }

        let twice_area = signed_twice_area(&ring);
        if twice_area.abs() <= f64::EPSILON {
            return Err(RenderError::Upload(
                "vector_coverage_zero_area_ring: ring has zero signed area".into(),
            ));
        }
        let should_be_positive = exterior;
        if (twice_area > 0.0) != should_be_positive {
            ring.reverse();
        }
        for index in 0..ring.len() {
            self.push_line(layer, ring[index], ring[(index + 1) % ring.len()])?;
        }
        Ok(())
    }

    fn push_line(&mut self, layer: u32, p0: Vec2, p1: Vec2) -> Result<(), RenderError> {
        let stable_id = self.take_id()?;
        if let Some(record) = PrimitiveRecord::line(p0.to_array(), p1.to_array(), layer, stable_id)
        {
            self.primitives.push(record);
        }
        Ok(())
    }

    fn push_arc(
        &mut self,
        layer: u32,
        center: Vec2,
        radius: f32,
        start: f64,
        sweep: f64,
    ) -> Result<(), RenderError> {
        // Split at every quadrant boundary.  Each emitted record is y-monotone
        // and stays on one x branch, which makes its scanline integral the
        // exact circular-segment antiderivative.
        let mut cuts = vec![0.0, 1.0];
        let end = start + sweep;
        let low = start.min(end);
        let high = start.max(end);
        let first = (low / FRAC_PI_2).floor() as i32 - 1;
        let last = (high / FRAC_PI_2).ceil() as i32 + 1;
        for quadrant in first..=last {
            let angle = f64::from(quadrant) * FRAC_PI_2;
            let t = (angle - start) / sweep;
            // Angles originate in f32 vector data.  Treat quadrant cuts within
            // one f32-scale epsilon of an endpoint as that endpoint; otherwise
            // an axis-aligned semicircle acquires a microscopic extra record.
            if t > 1.0e-7 && t < 1.0 - 1.0e-7 {
                cuts.push(t);
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);

        for pair in cuts.windows(2) {
            let a0 = start + sweep * pair[0];
            let a1 = start + sweep * pair[1];
            let mid = 0.5 * (a0 + a1);
            let direction0 = crate::camera::Anchor::direction_to_render(glam::DVec3::new(
                (a0 % TAU).cos(),
                (a0 % TAU).sin(),
                0.0,
            ));
            let direction1 = crate::camera::Anchor::direction_to_render(glam::DVec3::new(
                (a1 % TAU).cos(),
                (a1 % TAU).sin(),
                0.0,
            ));
            let e0 = center + direction0.truncate() * radius;
            let e1 = center + direction1.truncate() * radius;
            let winding = if e1.y > e0.y { 1 } else { -1 };
            let branch = if mid.cos() >= 0.0 { 1.0 } else { -1.0 };
            let bounds = [
                e0.x.min(e1.x),
                e0.y.min(e1.y),
                e0.x.max(e1.x),
                e0.y.max(e1.y),
            ];
            let stable_id = self.take_id()?;
            if let Some(record) = PrimitiveRecord::arc(
                center.to_array(),
                radius,
                branch,
                bounds,
                winding,
                layer,
                stable_id,
            ) {
                self.primitives.push(record);
            }
        }
        Ok(())
    }

    fn take_id(&mut self) -> Result<u32, RenderError> {
        let id = self.next_stable_id;
        self.next_stable_id = self.next_stable_id.checked_add(1).ok_or_else(|| {
            RenderError::Upload(
                "vector_coverage_primitive_overflow: more than u32::MAX records".into(),
            )
        })?;
        Ok(id)
    }
}

fn signed_twice_area(ring: &[Vec2]) -> f64 {
    let mut area = 0.0_f64;
    for index in 0..ring.len() {
        let a = ring[index].as_dvec2();
        let b = ring[(index + 1) % ring.len()].as_dvec2();
        area += a.x * b.y - b.x * a.y;
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::VectorStyle;

    fn layer(builder: &mut CoverageGeometryBuilder, fill_rule: FillRule) -> u32 {
        builder
            .add_layer("test", fill_rule, [0.2, 0.4, 0.8, 1.0])
            .unwrap()
    }

    #[test]
    fn polygon_ingest_normalizes_exterior_and_hole_winding() {
        let mut builder = CoverageGeometryBuilder::new(64, 64).unwrap();
        let layer = layer(&mut builder, FillRule::NonZero);
        let polygon = PolygonDef {
            // Deliberately clockwise exterior and counter-clockwise hole.
            exterior: vec![
                Vec2::new(4.0, 4.0),
                Vec2::new(4.0, 60.0),
                Vec2::new(60.0, 60.0),
                Vec2::new(60.0, 4.0),
            ],
            holes: vec![vec![
                Vec2::new(20.0, 20.0),
                Vec2::new(44.0, 20.0),
                Vec2::new(44.0, 44.0),
                Vec2::new(20.0, 44.0),
            ]],
            style: VectorStyle::default(),
        };
        builder.push_polygon(layer, &polygon).unwrap();
        let geometry = builder.finish().unwrap();
        assert_eq!(geometry.primitives.len(), 8);
        let exterior: Vec<_> = geometry.primitives[..4]
            .iter()
            .map(|record| Vec2::new(record.geometry[0], record.geometry[1]))
            .collect();
        let hole: Vec<_> = geometry.primitives[4..]
            .iter()
            .map(|record| Vec2::new(record.geometry[0], record.geometry[1]))
            .collect();
        assert!(signed_twice_area(&exterior) > 0.0);
        assert!(signed_twice_area(&hole) < 0.0);
    }

    #[test]
    fn round_segment_is_two_lines_and_four_monotone_arcs() {
        let mut builder = CoverageGeometryBuilder::new(64, 64).unwrap();
        let layer = layer(&mut builder, FillRule::NonZero);
        let line = PolylineDef {
            path: vec![Vec2::new(8.0, 24.0), Vec2::new(56.0, 24.0)],
            style: VectorStyle {
                stroke_width: 4.0,
                ..VectorStyle::default()
            },
        };
        builder.push_round_polyline(layer, &line).unwrap();
        let geometry = builder.finish().unwrap();
        let line_count = geometry
            .primitives
            .iter()
            .filter(|primitive| primitive.kind() == super::super::PrimitiveKind::Line)
            .count();
        let arc_count = geometry.primitives.len() - line_count;
        assert_eq!(line_count, 2);
        assert_eq!(arc_count, 4);
        for arc in geometry
            .primitives
            .iter()
            .filter(|primitive| primitive.kind() == super::super::PrimitiveKind::Arc)
        {
            assert!(arc.bounds[1] <= arc.bounds[3]);
            assert!(matches!(arc.winding(), -1 | 1));
        }
    }

    #[test]
    fn collinear_and_180_degree_segments_remain_capsule_union() {
        let mut builder = CoverageGeometryBuilder::new(64, 64).unwrap();
        let layer = layer(&mut builder, FillRule::NonZero);
        let line = PolylineDef {
            path: vec![
                Vec2::new(8.0, 24.0),
                Vec2::new(32.0, 24.0),
                Vec2::new(56.0, 24.0),
                Vec2::new(32.0, 24.0),
            ],
            style: VectorStyle {
                stroke_width: 2.0,
                ..VectorStyle::default()
            },
        };
        builder.push_round_polyline(layer, &line).unwrap();
        let geometry = builder.finish().unwrap();
        assert_eq!(geometry.primitives.len(), 18);
        assert_eq!(geometry.layers[0].fill_rule, FillRule::NonZero);
    }
}
