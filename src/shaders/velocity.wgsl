// P1.1: Motion vector / velocity buffer computation
// Screen-space velocity for TAA reprojection, motion blur, and temporal stability
//
// Usage: Include this file and call compute_velocity() in fragment shaders
// that output to the velocity render target.

/// Compute screen-space velocity from current and previous frame positions
/// Returns velocity in UV space (multiply by resolution for pixel velocity)
///
/// Arguments:
///   - clip_pos: Current frame clip-space position (after projection)
///   - world_pos: World-space position for reprojection to previous frame
///   - prev_view_proj: Previous frame's view-projection matrix
///
/// Returns: vec2<f32> velocity in NDC space scaled to [−0.5, 0.5] range
fn compute_velocity(
    clip_pos: vec4<f32>,
    world_pos: vec3<f32>,
    prev_view_proj: mat4x4<f32>
) -> vec2<f32> {
    // Current NDC position (clip.xy / clip.w gives [-1, 1] range)
    let ndc = clip_pos.xy / clip_pos.w;
    
    // Reproject world position to previous frame's clip space
    let prev_clip = prev_view_proj * vec4<f32>(world_pos, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    
    // Velocity in NDC space: current - previous
    // Scaled by 0.5 to map from [-2, 2] to [-1, 1] range
    // This represents how much a pixel moved in screen space
    let velocity = (ndc - prev_ndc) * 0.5;
    
    return velocity;
}

/// Compute velocity for static geometry (camera motion only)
/// Simpler version when world position is already known to be static
fn compute_velocity_static(
    ndc_pos: vec2<f32>,
    world_pos: vec3<f32>,
    prev_view_proj: mat4x4<f32>
) -> vec2<f32> {
    let prev_clip = prev_view_proj * vec4<f32>(world_pos, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    return (ndc_pos - prev_ndc) * 0.5;
}

/// Encode velocity for storage in Rg16Float texture
/// Input: velocity in [-1, 1] range
/// Output: velocity ready for texture storage (no encoding needed for float formats)
fn encode_velocity(velocity: vec2<f32>) -> vec2<f32> {
    // For Rg16Float, no encoding needed - store directly
    // Clamp to reasonable range to avoid inf/nan issues
    return clamp(velocity, vec2<f32>(-1.0), vec2<f32>(1.0));
}

/// Decode velocity from Rg16Float texture
/// Input: raw texture value
/// Output: velocity in [-1, 1] range
fn decode_velocity(encoded: vec2<f32>) -> vec2<f32> {
    // For Rg16Float, no decoding needed
    return encoded;
}

/// Compute velocity magnitude for visualization/debugging
fn velocity_magnitude(velocity: vec2<f32>) -> f32 {
    return length(velocity);
}

/// Convert velocity to RGB for debug visualization
/// Maps velocity direction to hue, magnitude to saturation
fn velocity_to_debug_color(velocity: vec2<f32>) -> vec3<f32> {
    let mag = length(velocity);
    if (mag < 0.001) {
        return vec3<f32>(0.5, 0.5, 0.5); // Gray for no motion
    }
    
    // Map velocity to color: X=Red channel, Y=Green channel
    // Add 0.5 to shift from [-0.5, 0.5] to [0, 1]
    let r = velocity.x + 0.5;
    let g = velocity.y + 0.5;
    let b = 0.5; // Fixed blue for contrast
    
    return vec3<f32>(r, g, b);
}
