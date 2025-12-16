// src/vector/line_helpers.rs
// Helper functions for line rendering calculations
// RELEVANT FILES: shaders/line_aa.wgsl

use glam::Vec2;

/// Calculate line join points for smooth connections
pub fn calculate_line_joins(path: &[Vec2], width: f32) -> Vec<Vec2> {
    if path.len() < 2 {
        return Vec::new();
    }

    let mut joins = Vec::with_capacity(path.len());
    let half_width = width * 0.5;

    for i in 0..path.len() {
        if i == 0 || i == path.len() - 1 {
            // Endpoints don't need join calculation
            joins.push(path[i]);
            continue;
        }

        let prev = path[i - 1];
        let curr = path[i];
        let next = path[i + 1];

        // Calculate join normal
        let seg1 = (curr - prev).normalize_or_zero();
        let seg2 = (next - curr).normalize_or_zero();
        let join_normal = (seg1 + seg2).normalize_or_zero();

        // Calculate miter offset
        let miter_dot = seg1.dot(join_normal);
        let miter_length = if miter_dot.abs() > 0.01 {
            half_width / miter_dot
        } else {
            half_width
        };

        // Apply miter limit
        let limited_length = miter_length.min(half_width * 4.0);
        joins.push(curr + join_normal.perp() * limited_length);
    }

    joins
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::PolylineDef;
    use crate::vector::api::VectorStyle;
    use crate::vector::line::{LineCap, LineJoin, LineRenderer};

    #[test]
    fn test_pack_simple_polyline() {
        let device = crate::core::gpu::create_device_for_test();
        let renderer = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let polyline = PolylineDef {
            path: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(2.0, 0.0),
            ],
            style: VectorStyle {
                stroke_width: 2.0,
                stroke_color: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
        };

        let instances = renderer.pack_polylines(&[polyline]).unwrap();

        assert_eq!(instances.len(), 2); // 3 points = 2 segments
        assert_eq!(instances[0].width, 2.0);
        assert_eq!(instances[0].color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_skip_degenerate_segments() {
        let device = crate::core::gpu::create_device_for_test();
        let renderer = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let polyline = PolylineDef {
            path: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(0.0, 0.0), // Duplicate point
                Vec2::new(1.0, 1.0),
            ],
            style: VectorStyle::default(),
        };

        let instances = renderer.pack_polylines(&[polyline]).unwrap();

        // Should skip the degenerate segment
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn test_line_joins() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
        ];

        let joins = calculate_line_joins(&path, 1.0);
        assert_eq!(joins.len(), 3);

        // First and last points should remain unchanged
        assert_eq!(joins[0], path[0]);
        assert_eq!(joins[2], path[2]);

        // Middle point should be offset for smooth join
        assert_ne!(joins[1], path[1]);
    }

    #[test]
    fn test_reject_short_polyline() {
        let device = crate::core::gpu::create_device_for_test();
        let renderer = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let short_line = PolylineDef {
            path: vec![Vec2::new(0.0, 0.0)], // Only 1 point
            style: VectorStyle::default(),
        };

        let result = renderer.pack_polylines(&[short_line]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 2 points"));
    }
}
