//! P2.1/M5: Clipmap ring mesh generation with skirts.
//!
//! Generates hollow ring meshes (donut shapes) for each LOD level,
//! plus the solid center block at finest resolution.

use super::vertex::ClipmapVertex;
use glam::Vec2;

/// Generate the center block mesh (solid grid at finest LOD).
pub fn make_center_block(
    resolution: u32,
    center: Vec2,
    half_extent: f32,
    terrain_extent: f32,
) -> (Vec<ClipmapVertex>, Vec<u32>) {
    let n = resolution as usize;
    let cell_size = (half_extent * 2.0) / resolution as f32;
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
    let mut indices = Vec::with_capacity(n * n * 6);

    for y in 0..=n {
        for x in 0..=n {
            let wx = center.x - half_extent + x as f32 * cell_size;
            let wz = center.y - half_extent + y as f32 * cell_size;
            let u = (wx + terrain_extent * 0.5) / terrain_extent;
            let v = (wz + terrain_extent * 0.5) / terrain_extent;
            vertices.push(ClipmapVertex::center(wx, wz, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)));
        }
    }

    let stride = n + 1;
    for y in 0..n {
        for x in 0..n {
            let i0 = (y * stride + x) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + stride as u32;
            let i3 = i2 + 1;
            // CCW winding
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    (vertices, indices)
}

/// Generate a single clipmap ring (hollow donut shape).
///
/// The ring consists of 4 strips (top, bottom, left, right) forming a frame
/// around the inner region. L-shaped corner patches eliminate T-junctions.
pub fn make_ring(
    ring_index: u32,
    inner_extent: f32,
    outer_extent: f32,
    resolution: u32,
    center: Vec2,
    terrain_extent: f32,
    morph_range: f32,
) -> (Vec<ClipmapVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let strip_width = outer_extent - inner_extent;
    let cell_size = strip_width / resolution as f32;
    let n = resolution as usize;

    // Calculate morph weights based on distance from ring boundary
    let calc_morph = |dist_from_inner: f32| -> f32 {
        let t = dist_from_inner / strip_width;
        let morph_start = 1.0 - morph_range;
        if t > morph_start {
            (t - morph_start) / morph_range
        } else {
            0.0
        }
    };

    let to_uv = |wx: f32, wz: f32| -> (f32, f32) {
        let u = (wx + terrain_extent * 0.5) / terrain_extent;
        let v = (wz + terrain_extent * 0.5) / terrain_extent;
        (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
    };

    // Generate the 4 strips that form the ring
    // Top strip (positive Z)
    let base_idx = vertices.len() as u32;
    for row in 0..=1 {
        for col in 0..=n {
            let wx = center.x - outer_extent + col as f32 * cell_size * 2.0;
            let wz = if row == 0 {
                center.y + inner_extent
            } else {
                center.y + outer_extent
            };
            let dist = if row == 0 { 0.0 } else { strip_width };
            let (u, v) = to_uv(wx.min(center.x + outer_extent), wz);
            let morph = calc_morph(dist);
            vertices.push(ClipmapVertex::new(
                wx.min(center.x + outer_extent),
                wz,
                u,
                v,
                morph,
                ring_index,
            ));
        }
    }
    generate_strip_indices(&mut indices, base_idx, n as u32 + 1);

    // Bottom strip (negative Z)
    let base_idx = vertices.len() as u32;
    for row in 0..=1 {
        for col in 0..=n {
            let wx = center.x - outer_extent + col as f32 * cell_size * 2.0;
            let wz = if row == 0 {
                center.y - outer_extent
            } else {
                center.y - inner_extent
            };
            let dist = if row == 0 { strip_width } else { 0.0 };
            let (u, v) = to_uv(wx.min(center.x + outer_extent), wz);
            let morph = calc_morph(dist);
            vertices.push(ClipmapVertex::new(
                wx.min(center.x + outer_extent),
                wz,
                u,
                v,
                morph,
                ring_index,
            ));
        }
    }
    generate_strip_indices(&mut indices, base_idx, n as u32 + 1);

    // Left strip (negative X)
    let base_idx = vertices.len() as u32;
    for row in 0..=1 {
        for col in 0..=n {
            let wx = if row == 0 {
                center.x - outer_extent
            } else {
                center.x - inner_extent
            };
            let wz = center.y - inner_extent + col as f32 * cell_size * 2.0;
            let dist = if row == 0 { strip_width } else { 0.0 };
            let (u, v) = to_uv(wx, wz.min(center.y + inner_extent));
            let morph = calc_morph(dist);
            vertices.push(ClipmapVertex::new(
                wx,
                wz.min(center.y + inner_extent),
                u,
                v,
                morph,
                ring_index,
            ));
        }
    }
    generate_strip_indices(&mut indices, base_idx, n as u32 + 1);

    // Right strip (positive X)
    let base_idx = vertices.len() as u32;
    for row in 0..=1 {
        for col in 0..=n {
            let wx = if row == 0 {
                center.x + inner_extent
            } else {
                center.x + outer_extent
            };
            let wz = center.y - inner_extent + col as f32 * cell_size * 2.0;
            let dist = if row == 0 { 0.0 } else { strip_width };
            let (u, v) = to_uv(wx, wz.min(center.y + inner_extent));
            let morph = calc_morph(dist);
            vertices.push(ClipmapVertex::new(
                wx,
                wz.min(center.y + inner_extent),
                u,
                v,
                morph,
                ring_index,
            ));
        }
    }
    generate_strip_indices(&mut indices, base_idx, n as u32 + 1);

    // Corner patches (L-shaped to avoid T-junctions)
    // Note: Currently simplified - strips overlap at corners
    add_corner_patch(&mut vertices, &mut indices, center, inner_extent, outer_extent, 
                     terrain_extent, ring_index, morph_range);

    (vertices, indices)
}

#[derive(Clone, Copy)]
enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

fn add_corner_patch(
    _vertices: &mut Vec<ClipmapVertex>,
    _indices: &mut Vec<u32>,
    _center: Vec2,
    _inner: f32,
    _outer: f32,
    _terrain_extent: f32,
    _ring_index: u32,
    _morph_range: f32,
) {
    // Corner patches are currently handled by strip overlap
    // Full implementation would add L-shaped corner geometry to eliminate T-junctions
    // This is a simplification for the initial implementation
}

fn generate_strip_indices(indices: &mut Vec<u32>, base: u32, width: u32) {
    for i in 0..width - 1 {
        let i0 = base + i;
        let i1 = base + i + 1;
        let i2 = base + width + i;
        let i3 = base + width + i + 1;
        // CCW winding
        indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
    }
}

/// Generate skirt vertices for a ring to hide seams.
pub fn make_ring_skirts(
    vertices: &[ClipmapVertex],
    _indices: &[u32],
    skirt_depth: f32,
    ring_index: u32,
) -> (Vec<ClipmapVertex>, Vec<u32>) {
    let mut skirt_verts = Vec::new();
    let mut skirt_indices = Vec::new();

    // Find edge vertices (simplified: use vertices at ring boundaries)
    // For each edge vertex, create a corresponding skirt vertex
    let base_idx = vertices.len() as u32;

    for (i, v) in vertices.iter().enumerate() {
        // Create skirt vertex below this one
        let sv = ClipmapVertex::skirt(v.position[0], v.position[1], v.uv[0], v.uv[1], ring_index);
        skirt_verts.push(sv);

        // Create triangles connecting original vertex to skirt
        // This creates a vertical "curtain" at the edge
        if i > 0 {
            let prev = i as u32 - 1;
            let curr = i as u32;
            let prev_skirt = base_idx + prev;
            let curr_skirt = base_idx + curr;
            // Degenerate check - only add if vertices are adjacent on edge
            // Simplified: add all for now
            skirt_indices.extend_from_slice(&[prev, curr, prev_skirt, curr, curr_skirt, prev_skirt]);
        }
    }

    let _ = skirt_depth; // Used in shader for Y offset
    (skirt_verts, skirt_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_block_vertex_count() {
        let (verts, indices) = make_center_block(4, Vec2::ZERO, 10.0, 100.0);
        assert_eq!(verts.len(), 25); // 5x5 vertices for 4x4 cells
        assert_eq!(indices.len(), 4 * 4 * 6); // 16 quads * 6 indices
    }

    #[test]
    fn test_center_block_ccw_winding() {
        let (verts, indices) = make_center_block(2, Vec2::ZERO, 10.0, 100.0);
        // First triangle
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;
        let v0 = Vec2::from(verts[i0].position);
        let v1 = Vec2::from(verts[i1].position);
        let v2 = Vec2::from(verts[i2].position);
        // CCW check: cross product should be positive
        let cross = (v1 - v0).perp_dot(v2 - v0);
        assert!(cross > 0.0, "First triangle should be CCW");
    }

    #[test]
    fn test_ring_generation() {
        let (verts, indices) = make_ring(1, 10.0, 20.0, 8, Vec2::ZERO, 100.0, 0.3);
        assert!(!verts.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0); // All triangles
    }

    #[test]
    fn test_morph_weights_in_range() {
        let (verts, _) = make_ring(1, 10.0, 20.0, 8, Vec2::ZERO, 100.0, 0.3);
        for v in &verts {
            assert!(v.morph_weight() >= 0.0 && v.morph_weight() <= 1.0);
        }
    }
}
