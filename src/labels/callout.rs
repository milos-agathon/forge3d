//! Callout box rendering for labels.
//!
//! Provides callout boxes with rounded corners, background fill,
//! and pointer/arrow geometry connecting to anchor points.

/// Callout box style configuration.
#[derive(Debug, Clone)]
pub struct CalloutStyle {
    /// Background color as RGBA.
    pub background_color: [f32; 4],
    /// Border color as RGBA.
    pub border_color: [f32; 4],
    /// Border width in pixels.
    pub border_width: f32,
    /// Corner radius in pixels.
    pub corner_radius: f32,
    /// Padding inside the box in pixels.
    pub padding: f32,
    /// Pointer/arrow size in pixels.
    pub pointer_size: f32,
    /// Shadow offset (x, y) in pixels.
    pub shadow_offset: [f32; 2],
    /// Shadow blur radius.
    pub shadow_blur: f32,
    /// Shadow color as RGBA.
    pub shadow_color: [f32; 4],
}

impl Default for CalloutStyle {
    fn default() -> Self {
        Self {
            background_color: [1.0, 1.0, 1.0, 0.95],
            border_color: [0.2, 0.2, 0.2, 1.0],
            border_width: 1.0,
            corner_radius: 4.0,
            padding: 8.0,
            pointer_size: 8.0,
            shadow_offset: [2.0, 2.0],
            shadow_blur: 4.0,
            shadow_color: [0.0, 0.0, 0.0, 0.3],
        }
    }
}

impl CalloutStyle {
    /// Create a style with custom background color.
    pub fn with_background(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.background_color = [r, g, b, a];
        self
    }

    /// Create a style with custom border.
    pub fn with_border(mut self, color: [f32; 4], width: f32) -> Self {
        self.border_color = color;
        self.border_width = width;
        self
    }

    /// Create a style with no border.
    pub fn no_border(mut self) -> Self {
        self.border_width = 0.0;
        self
    }

    /// Create a style with custom corner radius.
    pub fn with_corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }

    /// Create a style with custom padding.
    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// Create a style with no shadow.
    pub fn no_shadow(mut self) -> Self {
        self.shadow_color[3] = 0.0;
        self
    }
}

/// Direction of the callout pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointerDirection {
    /// Pointer points down (callout above anchor).
    #[default]
    Down,
    /// Pointer points up (callout below anchor).
    Up,
    /// Pointer points left (callout to right of anchor).
    Left,
    /// Pointer points right (callout to left of anchor).
    Right,
    /// No pointer.
    None,
}

/// A callout box with text and pointer.
#[derive(Debug, Clone)]
pub struct Callout {
    /// Unique identifier.
    pub id: u64,
    /// Text content (may be multi-line).
    pub text: String,
    /// Anchor position in screen space.
    pub anchor: [f32; 2],
    /// Box position (top-left corner) in screen space.
    pub box_pos: [f32; 2],
    /// Box size (width, height).
    pub box_size: [f32; 2],
    /// Style configuration.
    pub style: CalloutStyle,
    /// Pointer direction.
    pub pointer_direction: PointerDirection,
    /// Whether the callout is visible.
    pub visible: bool,
}

/// Vertex for callout geometry rendering.
#[derive(Debug, Clone, Copy)]
pub struct CalloutVertex {
    /// Position in screen space.
    pub position: [f32; 2],
    /// Color.
    pub color: [f32; 4],
    /// UV for rounded corner SDF (0,0 at corner center).
    pub uv: [f32; 2],
    /// Corner radius for this vertex.
    pub corner_radius: f32,
}

/// Generate vertices for a rounded rectangle.
pub fn generate_rounded_rect_vertices(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    radius: f32,
    color: [f32; 4],
) -> (Vec<CalloutVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Clamp radius to half the smaller dimension
    let r = radius.min(width / 2.0).min(height / 2.0);

    // For simplicity, generate a simple rectangle
    // A proper implementation would generate corner arcs
    let corners = [
        [x, y],                     // Top-left
        [x + width, y],             // Top-right
        [x + width, y + height],    // Bottom-right
        [x, y + height],            // Bottom-left
    ];

    for (i, corner) in corners.iter().enumerate() {
        vertices.push(CalloutVertex {
            position: *corner,
            color,
            uv: match i {
                0 => [-1.0, -1.0],
                1 => [1.0, -1.0],
                2 => [1.0, 1.0],
                3 => [-1.0, 1.0],
                _ => [0.0, 0.0],
            },
            corner_radius: r,
        });
    }

    // Two triangles for the quad
    indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);

    (vertices, indices)
}

/// Generate vertices for a pointer/arrow.
pub fn generate_pointer_vertices(
    tip: [f32; 2],
    base_center: [f32; 2],
    size: f32,
    direction: PointerDirection,
    color: [f32; 4],
) -> (Vec<CalloutVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    if direction == PointerDirection::None {
        return (vertices, indices);
    }

    let half_base = size * 0.6;

    let (base_left, base_right) = match direction {
        PointerDirection::Down | PointerDirection::Up => (
            [base_center[0] - half_base, base_center[1]],
            [base_center[0] + half_base, base_center[1]],
        ),
        PointerDirection::Left | PointerDirection::Right => (
            [base_center[0], base_center[1] - half_base],
            [base_center[0], base_center[1] + half_base],
        ),
        PointerDirection::None => return (vertices, indices),
    };

    let base_idx = vertices.len() as u32;

    vertices.push(CalloutVertex {
        position: tip,
        color,
        uv: [0.0, 0.0],
        corner_radius: 0.0,
    });
    vertices.push(CalloutVertex {
        position: base_left,
        color,
        uv: [0.0, 0.0],
        corner_radius: 0.0,
    });
    vertices.push(CalloutVertex {
        position: base_right,
        color,
        uv: [0.0, 0.0],
        corner_radius: 0.0,
    });

    indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2]);

    (vertices, indices)
}

/// Generate all vertices for a callout box.
pub fn generate_callout_geometry(callout: &Callout) -> (Vec<CalloutVertex>, Vec<u32>) {
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    // Generate shadow if enabled
    if callout.style.shadow_color[3] > 0.0 {
        let shadow_x = callout.box_pos[0] + callout.style.shadow_offset[0];
        let shadow_y = callout.box_pos[1] + callout.style.shadow_offset[1];
        let (shadow_verts, shadow_idxs) = generate_rounded_rect_vertices(
            shadow_x,
            shadow_y,
            callout.box_size[0],
            callout.box_size[1],
            callout.style.corner_radius,
            callout.style.shadow_color,
        );
        let base = all_vertices.len() as u32;
        all_vertices.extend(shadow_verts);
        all_indices.extend(shadow_idxs.iter().map(|i| i + base));
    }

    // Generate main box
    let (box_verts, box_idxs) = generate_rounded_rect_vertices(
        callout.box_pos[0],
        callout.box_pos[1],
        callout.box_size[0],
        callout.box_size[1],
        callout.style.corner_radius,
        callout.style.background_color,
    );
    let base = all_vertices.len() as u32;
    all_vertices.extend(box_verts);
    all_indices.extend(box_idxs.iter().map(|i| i + base));

    // Generate pointer
    if callout.pointer_direction != PointerDirection::None {
        let (tip, base_center) = compute_pointer_positions(callout);
        let (ptr_verts, ptr_idxs) = generate_pointer_vertices(
            tip,
            base_center,
            callout.style.pointer_size,
            callout.pointer_direction,
            callout.style.background_color,
        );
        let base = all_vertices.len() as u32;
        all_vertices.extend(ptr_verts);
        all_indices.extend(ptr_idxs.iter().map(|i| i + base));
    }

    // Generate border if enabled
    if callout.style.border_width > 0.0 {
        // Border would be generated as line segments around the box
        // For simplicity, we'll skip detailed border generation here
    }

    (all_vertices, all_indices)
}

/// Compute pointer tip and base center positions.
fn compute_pointer_positions(callout: &Callout) -> ([f32; 2], [f32; 2]) {
    let box_center_x = callout.box_pos[0] + callout.box_size[0] / 2.0;
    let box_center_y = callout.box_pos[1] + callout.box_size[1] / 2.0;

    match callout.pointer_direction {
        PointerDirection::Down => (
            callout.anchor,
            [box_center_x, callout.box_pos[1] + callout.box_size[1]],
        ),
        PointerDirection::Up => (
            callout.anchor,
            [box_center_x, callout.box_pos[1]],
        ),
        PointerDirection::Left => (
            callout.anchor,
            [callout.box_pos[0], box_center_y],
        ),
        PointerDirection::Right => (
            callout.anchor,
            [callout.box_pos[0] + callout.box_size[0], box_center_y],
        ),
        PointerDirection::None => (callout.anchor, callout.anchor),
    }
}

/// Determine the best pointer direction based on anchor and box positions.
pub fn auto_pointer_direction(anchor: [f32; 2], box_pos: [f32; 2], box_size: [f32; 2]) -> PointerDirection {
    let box_center_x = box_pos[0] + box_size[0] / 2.0;
    let box_center_y = box_pos[1] + box_size[1] / 2.0;

    let dx = anchor[0] - box_center_x;
    let dy = anchor[1] - box_center_y;

    if dx.abs() > dy.abs() {
        if dx > 0.0 {
            PointerDirection::Right
        } else {
            PointerDirection::Left
        }
    } else if dy > 0.0 {
        PointerDirection::Down
    } else {
        PointerDirection::Up
    }
}

/// Calculate box position from anchor with offset.
pub fn calculate_box_position(
    anchor: [f32; 2],
    box_size: [f32; 2],
    offset: [f32; 2],
    pointer_size: f32,
) -> ([f32; 2], PointerDirection) {
    let dir = if offset[1] < 0.0 {
        PointerDirection::Down
    } else if offset[1] > 0.0 {
        PointerDirection::Up
    } else if offset[0] < 0.0 {
        PointerDirection::Right
    } else if offset[0] > 0.0 {
        PointerDirection::Left
    } else {
        PointerDirection::None
    };

    let box_x = anchor[0] + offset[0] - box_size[0] / 2.0;
    let box_y = match dir {
        PointerDirection::Down => anchor[1] + offset[1] - box_size[1] - pointer_size,
        PointerDirection::Up => anchor[1] + offset[1] + pointer_size,
        _ => anchor[1] + offset[1] - box_size[1] / 2.0,
    };

    ([box_x, box_y], dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callout_style_default() {
        let style = CalloutStyle::default();
        assert!(style.background_color[3] > 0.0);
        assert!(style.corner_radius > 0.0);
    }

    #[test]
    fn test_auto_pointer_direction() {
        let box_pos = [100.0, 100.0];
        let box_size = [50.0, 30.0];

        // Anchor below box
        let dir = auto_pointer_direction([125.0, 150.0], box_pos, box_size);
        assert_eq!(dir, PointerDirection::Down);

        // Anchor above box
        let dir = auto_pointer_direction([125.0, 80.0], box_pos, box_size);
        assert_eq!(dir, PointerDirection::Up);
    }

    #[test]
    fn test_generate_rounded_rect() {
        let (verts, indices) = generate_rounded_rect_vertices(
            0.0, 0.0, 100.0, 50.0, 5.0,
            [1.0, 1.0, 1.0, 1.0],
        );
        assert_eq!(verts.len(), 4);
        assert_eq!(indices.len(), 6);
    }
}
