// UV transformation utilities for texture coordinate mapping
// Handles Y-flip, scale, and offset transformations

struct UVTransform {
    scale: vec2<f32>,
    offset: vec2<f32>,
    y_flip: u32,
    _pad: u32,
}

/// Apply UV transformation with optional Y-flip
fn apply_uv_transform(uv: vec2<f32>, xform: UVTransform) -> vec2<f32> {
    // Handle Y-flip: flip around center (0.5) then scale and offset
    let y_mult = select(1.0, -1.0, xform.y_flip != 0u);
    let uv_y_adjusted = select(uv.y, 1.0 - uv.y, xform.y_flip != 0u);
    let uv_flipped = vec2<f32>(uv.x, uv_y_adjusted);
    
    // Apply scale and offset, then clamp to valid range
    return clamp(uv_flipped * xform.scale + xform.offset, vec2<f32>(0.0), vec2<f32>(1.0));
}

/// Simple UV transform without Y-flip (for common case)
fn apply_uv_scale_offset(uv: vec2<f32>, scale: vec2<f32>, offset: vec2<f32>) -> vec2<f32> {
    return clamp(uv * scale + offset, vec2<f32>(0.0), vec2<f32>(1.0));
}
