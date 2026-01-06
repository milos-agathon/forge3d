//! Line label placement along polylines.
//!
//! Computes glyph positions along a polyline path for curved/angled text.

use glam::{Mat4, Vec3, Vec4};

use crate::labels::types::{GlyphPlacement, LineLabelPlacement};

/// Compute line label glyph placements along a polyline.
///
/// # Arguments
/// * `polyline` - World space polyline vertices
/// * `text` - Label text
/// * `glyph_advances` - Advance width for each glyph in pixels
/// * `view_proj` - Combined view-projection matrix
/// * `screen_width` - Screen width in pixels
/// * `screen_height` - Screen height in pixels
/// * `placement` - Placement mode (center or along)
/// * `font_size` - Font size in pixels
///
/// # Returns
/// Vector of glyph placements, or empty if line is too short
pub fn compute_line_label_placement(
    polyline: &[Vec3],
    text: &str,
    glyph_advances: &[f32],
    view_proj: Mat4,
    screen_width: f32,
    screen_height: f32,
    placement: LineLabelPlacement,
    font_size: f32,
) -> Vec<GlyphPlacement> {
    if polyline.len() < 2 || text.is_empty() {
        return Vec::new();
    }

    // Project polyline to screen space
    let screen_points: Vec<Option<[f32; 2]>> = polyline
        .iter()
        .map(|p| project_to_screen(*p, view_proj, screen_width, screen_height))
        .collect();

    // Filter to visible segments
    let visible_points: Vec<[f32; 2]> = screen_points.iter().filter_map(|p| *p).collect();

    if visible_points.len() < 2 {
        return Vec::new();
    }

    // Compute total screen-space path length
    let path_length = compute_path_length(&visible_points);

    // Compute total text width
    let text_width: f32 = glyph_advances.iter().sum();

    if text_width > path_length * 0.9 {
        // Text too long for path
        return Vec::new();
    }

    match placement {
        LineLabelPlacement::Center => {
            compute_center_placement(&visible_points, glyph_advances, font_size, path_length, text_width)
        }
        LineLabelPlacement::Along => {
            compute_along_placement(&visible_points, glyph_advances, font_size, path_length, text_width)
        }
    }
}

/// Compute placement at center of line.
fn compute_center_placement(
    points: &[[f32; 2]],
    glyph_advances: &[f32],
    font_size: f32,
    path_length: f32,
    text_width: f32,
) -> Vec<GlyphPlacement> {
    let start_offset = (path_length - text_width) * 0.5;
    place_glyphs_along_path(points, glyph_advances, font_size, start_offset)
}

/// Compute placement along the entire line.
fn compute_along_placement(
    points: &[[f32; 2]],
    glyph_advances: &[f32],
    font_size: f32,
    path_length: f32,
    text_width: f32,
) -> Vec<GlyphPlacement> {
    let start_offset = (path_length - text_width) * 0.5;
    place_glyphs_along_path(points, glyph_advances, font_size, start_offset)
}

/// Place glyphs along a path starting at a given offset.
fn place_glyphs_along_path(
    points: &[[f32; 2]],
    glyph_advances: &[f32],
    font_size: f32,
    start_offset: f32,
) -> Vec<GlyphPlacement> {
    let mut placements = Vec::with_capacity(glyph_advances.len());
    let mut current_offset = start_offset;

    for advance in glyph_advances {
        // Find position and tangent at current offset
        if let Some((pos, tangent)) = sample_path_at_offset(points, current_offset + advance * 0.5) {
            let rotation = tangent.atan2(1.0);

            // Flip text if it would be upside down
            let rotation = if rotation.abs() > std::f32::consts::FRAC_PI_2 {
                rotation + std::f32::consts::PI
            } else {
                rotation
            };

            placements.push(GlyphPlacement {
                screen_pos: pos,
                rotation,
                scale: font_size,
            });
        }

        current_offset += advance;
    }

    placements
}

/// Sample position and tangent direction at a given offset along the path.
fn sample_path_at_offset(points: &[[f32; 2]], offset: f32) -> Option<([f32; 2], f32)> {
    if points.len() < 2 || offset < 0.0 {
        return None;
    }

    let mut accumulated = 0.0;

    for i in 0..points.len() - 1 {
        let p0 = points[i];
        let p1 = points[i + 1];
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let segment_length = (dx * dx + dy * dy).sqrt();

        if accumulated + segment_length >= offset {
            // Interpolate within this segment
            let t = (offset - accumulated) / segment_length.max(0.001);
            let x = p0[0] + dx * t;
            let y = p0[1] + dy * t;
            let tangent = dy.atan2(dx);
            return Some(([x, y], tangent));
        }

        accumulated += segment_length;
    }

    // Past end of path - return last point
    let last = points.len() - 1;
    let p0 = points[last - 1];
    let p1 = points[last];
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let tangent = dy.atan2(dx);
    Some((p1, tangent))
}

/// Compute total path length.
fn compute_path_length(points: &[[f32; 2]]) -> f32 {
    let mut length = 0.0;
    for i in 0..points.len().saturating_sub(1) {
        let dx = points[i + 1][0] - points[i][0];
        let dy = points[i + 1][1] - points[i][1];
        length += (dx * dx + dy * dy).sqrt();
    }
    length
}

/// Project world position to screen coordinates.
fn project_to_screen(
    world_pos: Vec3,
    view_proj: Mat4,
    screen_width: f32,
    screen_height: f32,
) -> Option<[f32; 2]> {
    let clip = view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

    if clip.w <= 0.0001 {
        return None;
    }

    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let ndc_z = clip.z / clip.w;

    // Check if within frustum
    if ndc_x < -1.2 || ndc_x > 1.2 || ndc_y < -1.2 || ndc_y > 1.2 || ndc_z < 0.0 || ndc_z > 1.0 {
        return None;
    }

    let screen_x = (ndc_x + 1.0) * 0.5 * screen_width;
    let screen_y = (1.0 - ndc_y) * 0.5 * screen_height;

    Some([screen_x, screen_y])
}

/// Compute glyph advances from text using approximate widths.
pub fn compute_glyph_advances(text: &str, font_size: f32) -> Vec<f32> {
    text.chars()
        .map(|c| {
            if c.is_ascii_uppercase() || c == 'W' || c == 'M' {
                font_size * 0.7
            } else if c == 'i' || c == 'l' || c == '!' || c == '.' {
                font_size * 0.3
            } else if c == ' ' {
                font_size * 0.3
            } else {
                font_size * 0.5
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_length() {
        let points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];
        let length = compute_path_length(&points);
        assert!((length - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_path() {
        let points = [[0.0, 0.0], [10.0, 0.0]];
        let (pos, tangent) = sample_path_at_offset(&points, 5.0).unwrap();
        assert!((pos[0] - 5.0).abs() < 0.001);
        assert!((pos[1] - 0.0).abs() < 0.001);
        assert!(tangent.abs() < 0.001); // Horizontal line
    }
}
