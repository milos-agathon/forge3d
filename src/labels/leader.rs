//! Leader line rendering for offset labels.
//!
//! Renders connector lines from label anchor points to offset label positions.

use crate::labels::types::LeaderLine;

/// Generate leader line geometry from anchor to label position.
///
/// # Arguments
/// * `anchor_screen` - Anchor position in screen space (original world position projected)
/// * `label_screen` - Label position in screen space (after offset applied)
/// * `color` - Line color RGBA
/// * `width` - Line width in pixels
///
/// # Returns
/// LeaderLine data for rendering
pub fn create_leader_line(
    anchor_screen: [f32; 2],
    label_screen: [f32; 2],
    color: [f32; 4],
    width: f32,
) -> LeaderLine {
    LeaderLine {
        start: anchor_screen,
        end: label_screen,
        color,
        width,
    }
}

/// Generate vertex data for rendering leader lines as triangles.
///
/// # Arguments
/// * `leaders` - Slice of leader lines to render
///
/// # Returns
/// Vertex data as [x, y, r, g, b, a] per vertex, 6 vertices per line (2 triangles)
pub fn generate_leader_vertices(leaders: &[LeaderLine]) -> Vec<f32> {
    let mut vertices = Vec::with_capacity(leaders.len() * 6 * 6);

    for leader in leaders {
        let dx = leader.end[0] - leader.start[0];
        let dy = leader.end[1] - leader.start[1];
        let len = (dx * dx + dy * dy).sqrt();

        if len < 0.001 {
            continue;
        }

        // Perpendicular direction for line width
        let px = -dy / len * leader.width * 0.5;
        let py = dx / len * leader.width * 0.5;

        // Four corners of the line quad
        let v0 = [leader.start[0] + px, leader.start[1] + py];
        let v1 = [leader.start[0] - px, leader.start[1] - py];
        let v2 = [leader.end[0] + px, leader.end[1] + py];
        let v3 = [leader.end[0] - px, leader.end[1] - py];

        let c = leader.color;

        // Triangle 1: v0, v1, v2
        vertices.extend_from_slice(&[v0[0], v0[1], c[0], c[1], c[2], c[3]]);
        vertices.extend_from_slice(&[v1[0], v1[1], c[0], c[1], c[2], c[3]]);
        vertices.extend_from_slice(&[v2[0], v2[1], c[0], c[1], c[2], c[3]]);

        // Triangle 2: v1, v3, v2
        vertices.extend_from_slice(&[v1[0], v1[1], c[0], c[1], c[2], c[3]]);
        vertices.extend_from_slice(&[v3[0], v3[1], c[0], c[1], c[2], c[3]]);
        vertices.extend_from_slice(&[v2[0], v2[1], c[0], c[1], c[2], c[3]]);
    }

    vertices
}

/// Compute optimal leader line path with elbow (L-shaped).
///
/// Creates a cleaner connection that avoids crossing the label text.
///
/// # Arguments
/// * `anchor` - Anchor point in screen space
/// * `label_pos` - Label position in screen space
/// * `label_width` - Width of the label in pixels
/// * `label_height` - Height of the label in pixels
///
/// # Returns
/// Vector of points forming the leader path
pub fn compute_elbow_leader(
    anchor: [f32; 2],
    label_pos: [f32; 2],
    label_width: f32,
    label_height: f32,
) -> Vec<[f32; 2]> {
    let dx = label_pos[0] - anchor[0];
    let dy = label_pos[1] - anchor[1];

    // Determine which side of the label to connect to
    let connect_x = if dx >= 0.0 {
        label_pos[0] - label_width * 0.5 - 4.0
    } else {
        label_pos[0] + label_width * 0.5 + 4.0
    };

    let connect_y = label_pos[1];

    // Simple L-shaped path
    if dy.abs() > label_height * 0.5 {
        // Vertical then horizontal
        vec![anchor, [anchor[0], connect_y], [connect_x, connect_y]]
    } else {
        // Direct line
        vec![anchor, [connect_x, connect_y]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_leader() {
        let leader = create_leader_line([0.0, 0.0], [100.0, 50.0], [0.0, 0.0, 0.0, 1.0], 2.0);
        assert_eq!(leader.start, [0.0, 0.0]);
        assert_eq!(leader.end, [100.0, 50.0]);
    }

    #[test]
    fn test_generate_vertices() {
        let leaders = vec![create_leader_line(
            [0.0, 0.0],
            [100.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            2.0,
        )];
        let verts = generate_leader_vertices(&leaders);
        assert_eq!(verts.len(), 6 * 6); // 6 vertices, 6 floats each
    }

    #[test]
    fn test_elbow_leader() {
        let path = compute_elbow_leader([0.0, 0.0], [100.0, 50.0], 60.0, 20.0);
        assert!(!path.is_empty());
    }
}
