
// Depth of Field shader with separable blur and tilt-shift support

struct Uniforms {
    screen_dims: vec4<f32>,    // width, height, 1/width, 1/height
    dof_params: vec4<f32>,     // focus_distance, f_stop, focal_length, max_blur_radius
    dof_params2: vec4<f32>,    // near_plane, far_plane, blur_direction (0=h, 1=v), quality
    camera_params: vec4<f32>,  // sensor_height, blur_strength, tilt_pitch, tilt_yaw
};

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Linearize depth from depth buffer
fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    // Standard-Z (glam::perspective_rh): depth=0 at near, depth=1 at far
    return near * far / (far - depth * (far - near));
}

// Calculate screen-space blur for tilt-shift miniature effect
// Returns a blur multiplier based on distance from the tilted focus band
fn calculate_tilt_blur_factor(uv: vec2<f32>, tilt_pitch: f32, tilt_yaw: f32) -> f32 {
    // Convert UV to centered coordinates [-1, 1]
    let centered = (uv - 0.5) * 2.0;
    
    // Calculate signed distance from the tilted focus plane in screen space
    // tilt_pitch tilts the focus band (positive = top sharp, bottom blurry)
    // tilt_yaw rotates the band (positive = left sharp, right blurry)
    // For classic miniature: pitch tilts vertically, creating horizontal sharp band
    let pitch_contrib = centered.y * sin(tilt_pitch);
    let yaw_contrib = centered.x * sin(tilt_yaw);
    
    // Distance from the focus band (0 = on band, 1 = far from band)
    let distance_from_band = abs(pitch_contrib + yaw_contrib);
    
    // Smooth falloff from focus band
    return smoothstep(0.0, 1.0, distance_from_band);
}

// Calculate effective focus distance with tilt-shift (Scheimpflug principle)
fn calculate_tilted_focus_distance(uv: vec2<f32>, base_focus: f32, tilt_pitch: f32, tilt_yaw: f32) -> f32 {
    // Convert UV to centered coordinates [-1, 1]
    let centered = (uv - 0.5) * 2.0;
    
    // Calculate tilt offset: pitch affects Y, yaw affects X
    let tilt_offset = centered.y * tan(tilt_pitch) + centered.x * tan(tilt_yaw);
    
    // Scale by focus distance for realistic plane tilt
    let focus_variation = base_focus * tilt_offset * 0.5;
    
    return max(base_focus + focus_variation, 1.0);
}

// Calculate Circle of Confusion (CoC) in pixels using thin lens model
fn calculate_coc(linear_depth: f32, focus_dist: f32, focal_length_mm: f32, f_stop: f32, sensor_height_mm: f32) -> f32 {
    // Convert to meters
    let focal_length = focal_length_mm / 1000.0;
    let sensor_height = sensor_height_mm / 1000.0;
    
    // Aperture diameter
    let aperture = focal_length / max(f_stop, 1.4);
    
    // Signed distance from focus plane (positive = behind focus, negative = in front)
    let signed_depth_diff = linear_depth - focus_dist;
    
    // Thin lens CoC formula:
    // CoC = |aperture * focal_length * (depth - focus) / (depth * (focus - focal_length))|
    let denominator = linear_depth * max(focus_dist - focal_length, 0.001);
    let coc_sensor = abs(aperture * focal_length * signed_depth_diff / denominator);
    
    // Convert sensor CoC to screen pixels
    let screen_height = u.screen_dims.y;
    let coc_pixels = coc_sensor * screen_height / sensor_height;
    
    return coc_pixels;
}

// Gaussian weight
fn gaussian_weight(offset: f32, sigma: f32) -> f32 {
    return exp(-0.5 * offset * offset / (sigma * sigma));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_focus_dist = u.dof_params.x;
    let f_stop = u.dof_params.y;
    let focal_length = u.dof_params.z;
    let max_blur = u.dof_params.w;
    let near = u.dof_params2.x;
    let far = u.dof_params2.y;
    let direction = u.dof_params2.z;
    let quality = u.dof_params2.w;
    let sensor_height = u.camera_params.x;
    let blur_strength = u.camera_params.y;
    let tilt_pitch = u.camera_params.z;
    let tilt_yaw = u.camera_params.w;
    
    // Sample depth at current pixel
    let w = max(i32(u.screen_dims.x) - 1, 0);
    let h = max(i32(u.screen_dims.y) - 1, 0);
    let xy = vec2<i32>(
        clamp(i32(in.uv.x * u.screen_dims.x), 0, w),
        clamp(i32(in.uv.y * u.screen_dims.y), 0, h),
    );
    let depth_sample = textureLoad(depth_tex, xy, 0);
    let linear_depth = linearize_depth(depth_sample, near, far);
    
    // Calculate effective focus distance (with tilt-shift if enabled)
    var focus_dist = base_focus_dist;
    let has_tilt = abs(tilt_pitch) > 0.001 || abs(tilt_yaw) > 0.001;
    if has_tilt {
        focus_dist = calculate_tilted_focus_distance(in.uv, base_focus_dist, tilt_pitch, tilt_yaw);
    }
    
    // Calculate CoC using thin lens model
    var coc = calculate_coc(linear_depth, focus_dist, focal_length, f_stop, sensor_height);
    
    // Apply blur strength multiplier for landscape scale
    coc = coc * blur_strength;
    
    // For tilt-shift: use screen-space blur for miniature effect
    // This creates the classic "fake miniature" look with a sharp horizontal band
    if has_tilt {
        let tilt_blur = calculate_tilt_blur_factor(in.uv, tilt_pitch, tilt_yaw);
        // Tilt blur OVERRIDES physical CoC to create proper miniature effect
        // tilt_blur = 0 at focus band (sharp), 1 at edges (full blur)
        // Scale to max_blur for full effect
        coc = tilt_blur * max_blur;
    }
    
    // Clamp to max blur radius
    coc = clamp(coc, 0.0, max_blur);
    
    // Get original color
    let original_color = textureSample(color_tex, samp, in.uv).rgb;
    
    // If CoC is very small, return original (no blur needed)
    if coc < 0.5 {
        return vec4<f32>(original_color, 1.0);
    }
    
    // Blur direction vector
    var blur_dir: vec2<f32>;
    if direction < 0.5 {
        blur_dir = vec2<f32>(u.screen_dims.z, 0.0);  // horizontal (1/width, 0)
    } else {
        blur_dir = vec2<f32>(0.0, u.screen_dims.w);  // vertical (0, 1/height)
    }
    
    // Number of samples based on quality
    let num_samples = i32(quality);
    let sigma = max(coc / 2.5, 1.0);
    
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    // Gaussian blur along direction
    for (var i = -num_samples; i <= num_samples; i++) {
        let offset = f32(i);
        let sample_uv = in.uv + blur_dir * offset * (coc / f32(num_samples));
        
        // Clamp to valid UV range
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        
        // Sample color
        let sample_color = textureSample(color_tex, samp, clamped_uv).rgb;
        
        // Gaussian weight only (no bilateral - it causes issues with separable blur)
        let weight = gaussian_weight(offset, sigma);
        
        color_sum += sample_color * weight;
        weight_sum += weight;
    }
    
    // Normalize
    let blurred_color = color_sum / max(weight_sum, 0.001);
    
    // Blend based on CoC strength
    let blend = smoothstep(0.5, 3.0, coc);
    let final_color = mix(original_color, blurred_color, blend);
    
    return vec4<f32>(final_color, 1.0);
}
