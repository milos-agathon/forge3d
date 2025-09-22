// B11: Water Surface Color Toggle - Configurable water surface rendering
// Provides controllable water albedo/hue with simple surface effects
// Supports transparency, color toggling, and basic wave animation

// ---------- Water Surface Uniforms ----------
struct WaterSurfaceUniforms {
    view_proj: mat4x4<f32>,                    // View-projection matrix
    world_transform: mat4x4<f32>,              // World transformation matrix
    surface_params: vec4<f32>,                 // size (x), height (y), enabled (z), alpha (w)
    color_params: vec4<f32>,                   // base_color (rgb) + hue_shift (w)
    wave_params: vec4<f32>,                    // wave_amplitude (x), wave_frequency (y), wave_speed (z), time (w)
    tint_params: vec4<f32>,                    // tint_color (rgb) + tint_strength (w)
    lighting_params: vec4<f32>,                // reflection_strength (x), refraction_strength (y), fresnel_power (z), roughness (w)
    animation_params: vec4<f32>,               // ripple_scale (x), ripple_speed (y), flow_direction (xy)
};

@group(0) @binding(0) var<uniform> water_uniforms : WaterSurfaceUniforms;

// ---------- Vertex Input/Output ----------
struct VsIn {
    @location(0) position: vec3<f32>,          // Local vertex position
    @location(1) uv: vec2<f32>,                // UV coordinates
    @location(2) normal: vec3<f32>,            // Vertex normal
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) view_distance: f32,           // Distance to camera
    @location(4) wave_offset: vec2<f32>,       // Animated wave offset
};

// ---------- Utility Functions ----------
fn hue_shift(color: vec3<f32>, shift: f32) -> vec3<f32> {
    // Simple hue shift using color rotation
    let cos_shift = cos(shift);
    let sin_shift = sin(shift);

    // Convert to approximate HSV-like rotation
    let shifted = mat3x3<f32>(
        cos_shift + (1.0 - cos_shift) * 0.213, (1.0 - cos_shift) * 0.715 - sin_shift * 0.072, (1.0 - cos_shift) * 0.072 + sin_shift * 0.213,
        (1.0 - cos_shift) * 0.213 + sin_shift * 0.143, cos_shift + (1.0 - cos_shift) * 0.715, (1.0 - cos_shift) * 0.072 - sin_shift * 0.928,
        (1.0 - cos_shift) * 0.213 - sin_shift * 0.787, (1.0 - cos_shift) * 0.715 + sin_shift * 0.072, cos_shift + (1.0 - cos_shift) * 0.072
    ) * color;

    return clamp(shifted, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn simple_wave(uv: vec2<f32>, time: f32, amplitude: f32, frequency: f32, speed: f32) -> f32 {
    let wave1 = sin(uv.x * frequency + time * speed) * amplitude;
    let wave2 = sin(uv.y * frequency * 1.3 + time * speed * 0.8) * amplitude * 0.7;
    let wave3 = sin((uv.x + uv.y) * frequency * 0.6 + time * speed * 1.2) * amplitude * 0.5;
    return wave1 + wave2 + wave3;
}

fn water_normal(uv: vec2<f32>, time: f32, amplitude: f32, frequency: f32, speed: f32) -> vec3<f32> {
    let epsilon = 0.01;
    let h_center = simple_wave(uv, time, amplitude, frequency, speed);
    let h_right = simple_wave(uv + vec2<f32>(epsilon, 0.0), time, amplitude, frequency, speed);
    let h_up = simple_wave(uv + vec2<f32>(0.0, epsilon), time, amplitude, frequency, speed);

    let tangent_x = vec3<f32>(epsilon, h_right - h_center, 0.0);
    let tangent_z = vec3<f32>(0.0, h_up - h_center, epsilon);

    return normalize(cross(tangent_x, tangent_z));
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ---------- Vertex Shader ----------
@vertex
fn vs_main(in: VsIn) -> VsOut {
    let time = water_uniforms.wave_params.w;
    let amplitude = water_uniforms.wave_params.x;
    let frequency = water_uniforms.wave_params.y;
    let speed = water_uniforms.wave_params.z;

    // Calculate wave displacement
    let wave_height = simple_wave(in.uv, time, amplitude, frequency, speed);
    let displaced_pos = in.position + vec3<f32>(0.0, wave_height, 0.0);

    // Transform vertex to world space
    let world_pos = (water_uniforms.world_transform * vec4<f32>(displaced_pos, 1.0)).xyz;

    // Calculate clip space position
    let clip_pos = water_uniforms.view_proj * vec4<f32>(world_pos, 1.0);

    // Calculate view distance for effects
    let view_distance = length(world_pos);

    // Calculate animated wave offset for texture sampling
    let flow_dir = water_uniforms.animation_params.zw;
    let ripple_speed = water_uniforms.animation_params.y;
    let wave_offset = flow_dir * time * ripple_speed;

    var out: VsOut;
    out.clip_pos = clip_pos;
    out.world_pos = world_pos;
    out.uv = in.uv;
    out.normal = water_normal(in.uv, time, amplitude, frequency, speed);
    out.view_distance = view_distance;
    out.wave_offset = wave_offset;

    return out;
}

// ---------- Fragment Shader ----------
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Check if water surface is enabled
    if water_uniforms.surface_params.z < 0.5 {
        discard;
    }

    // Base water color
    let base_color = water_uniforms.color_params.rgb;
    let hue_shift_amount = water_uniforms.color_params.w;

    // Apply hue shift to base color
    var water_color = hue_shift(base_color, hue_shift_amount);

    // Apply tint color blending
    let tint_color = water_uniforms.tint_params.rgb;
    let tint_strength = water_uniforms.tint_params.w;
    water_color = mix(water_color, tint_color, tint_strength);

    // Simple lighting calculation
    let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.2)); // Default light direction
    let view_dir = normalize(-in.world_pos); // Assuming camera at origin

    // Use calculated water normal for lighting
    let normal = normalize(in.normal);
    let ndotl = max(dot(normal, light_dir), 0.0);
    let ndotv = max(dot(normal, view_dir), 0.0);

    // Fresnel effect for water reflectivity
    let f0 = vec3<f32>(0.02); // Water's base reflectance
    let fresnel = fresnel_schlick(ndotv, f0);
    let reflection_strength = water_uniforms.lighting_params.x;

    // Calculate final color with lighting
    let ambient = 0.3;
    let diffuse = ndotl * 0.7;
    let lighting_factor = ambient + diffuse;

    // Apply fresnel reflection effect
    let reflection_factor = 1.0 + reflection_strength * length(fresnel);
    let final_color = water_color * lighting_factor * reflection_factor;

    // Distance-based alpha fading (optional)
    let fade_distance = 1000.0;
    let distance_alpha = clamp(1.0 - (in.view_distance / fade_distance), 0.0, 1.0);
    let final_alpha = water_uniforms.surface_params.w * distance_alpha;

    return vec4<f32>(final_color, final_alpha);
}

// ---------- Utility Functions for Animation ----------
fn calculate_flow_offset(uv: vec2<f32>, time: f32, flow_speed: f32, flow_dir: vec2<f32>) -> vec2<f32> {
    return uv + flow_dir * time * flow_speed;
}

fn water_depth_color(depth: f32, shallow_color: vec3<f32>, deep_color: vec3<f32>) -> vec3<f32> {
    let depth_factor = clamp(depth / 10.0, 0.0, 1.0); // Normalize depth
    return mix(shallow_color, deep_color, depth_factor);
}

// ---------- Alternative Water Effects ----------
fn caustics_pattern(uv: vec2<f32>, time: f32) -> f32 {
    let scale = 4.0;
    let speed = 2.0;
    let uv_scaled = uv * scale;

    let caustic1 = sin(uv_scaled.x + time * speed) * sin(uv_scaled.y + time * speed * 0.7);
    let caustic2 = sin(uv_scaled.x * 1.3 + time * speed * 1.1) * sin(uv_scaled.y * 0.8 + time * speed * 0.9);

    return (caustic1 + caustic2) * 0.5 + 0.5;
}

fn foam_pattern(uv: vec2<f32>, wave_height: f32, threshold: f32) -> f32 {
    // Simple foam generation based on wave height
    let foam_intensity = smoothstep(threshold, threshold + 0.1, wave_height);

    // Add noise-like foam texture
    let foam_noise = fract(sin(dot(uv * 20.0, vec2<f32>(12.9898, 78.233))) * 43758.5453);

    return foam_intensity * foam_noise;
}