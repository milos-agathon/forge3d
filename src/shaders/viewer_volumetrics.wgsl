// src/shaders/viewer_volumetrics.wgsl
// P5: Volumetric fog and light shafts for terrain viewer

struct VolumetricsUniforms {
    // Camera (64 bytes)
    inv_view_proj: mat4x4<f32>,
    // camera_pos.xyz, pad (16 bytes)
    camera_pos: vec4<f32>,
    // near, far, density, height_falloff (16 bytes)
    near_far: vec4<f32>,
    // scattering, absorption, pad, pad (16 bytes)
    scatter_absorb: vec4<f32>,
    // sun_direction.xyz, sun_intensity (16 bytes)
    sun_direction: vec4<f32>,
    // shaft_intensity, light_shafts_enabled, steps, mode (16 bytes)
    shaft_params: vec4<f32>,
    // screen_width, screen_height, pad, pad (16 bytes)
    screen_dims: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: VolumetricsUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

// Reconstruct world position from depth
fn world_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world_h = u.inv_view_proj * ndc;
    return world_h.xyz / world_h.w;
}

// Extract uniform fields for clarity
fn get_density() -> f32 { return u.near_far.z; }
fn get_height_falloff() -> f32 { return u.near_far.w; }
fn get_scattering() -> f32 { return u.scatter_absorb.x; }
fn get_absorption() -> f32 { return u.scatter_absorb.y; }
fn get_sun_intensity() -> f32 { return u.sun_direction.w; }
fn get_shaft_intensity() -> f32 { return u.shaft_params.x; }
fn get_light_shafts_enabled() -> bool { return u.shaft_params.y > 0.5; }
fn get_steps() -> u32 { return u32(u.shaft_params.z); }
fn get_mode() -> u32 { return u32(u.shaft_params.w); }
fn get_near() -> f32 { return u.near_far.x; }
fn get_far() -> f32 { return u.near_far.y; }

// Height-based fog density
fn fog_density_at(world_pos: vec3<f32>) -> f32 {
    let density = get_density();
    if get_mode() == 0u {
        // Uniform fog
        return density;
    } else {
        // Height-based exponential falloff
        let height = world_pos.y;
        return density * exp(-height * get_height_falloff());
    }
}

// Henyey-Greenstein phase function for anisotropic scattering
fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// Linearize depth from NDC for wgpu reverse-Z (depth=1 at near, depth=0 at far)
fn linearize_depth(ndc_depth: f32, near: f32, far: f32) -> f32 {
    // wgpu uses reverse-Z: ndc_depth=1.0 at near plane, ndc_depth=0.0 at far plane
    // For reverse-Z perspective: linear_z = near * far / (far - ndc_depth * (far - near))
    // But we need the VIEW-SPACE distance, not clip z
    // Correct formula for reverse-Z to get view distance:
    let z = clamp(ndc_depth, 0.0001, 0.9999);  // Avoid division issues at extremes
    return near * far / (near + z * (far - near));
}

// Light shaft shadow estimation using sun ray march through volume
// Returns visibility factor [0,1] for light shafts
fn estimate_light_shaft_shadow(world_pos: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    // March toward sun to estimate shadow from terrain/volume
    // Use fewer steps for performance
    let shaft_steps = 8u;
    let shaft_distance = 500.0;  // How far to march toward sun
    let step_size = shaft_distance / f32(shaft_steps);
    
    var occlusion = 0.0;
    for (var i = 0u; i < shaft_steps; i++) {
        let t = (f32(i) + 0.5) * step_size;
        let sample_pos = world_pos + sun_dir * t;
        
        // Height-based shadow: higher positions are more likely lit
        // Lower positions accumulate more occlusion
        let height_factor = clamp(sample_pos.y * 0.002, 0.0, 1.0);
        let local_density = fog_density_at(sample_pos);
        
        // Accumulate occlusion based on density along sun ray
        occlusion += local_density * (1.0 - height_factor) * step_size * 0.01;
    }
    
    // Convert accumulated occlusion to visibility
    return exp(-occlusion * get_absorption() * 2.0);
}

// Ray march through fog volume with depth-aware termination
// Returns fog color (rgb) and fog amount (a) for proper compositing
fn raymarch_fog_with_depth(ray_origin: vec3<f32>, ray_dir: vec3<f32>, scene_dist: f32) -> vec4<f32> {
    let steps = get_steps();
    let sun_dir = normalize(u.sun_direction.xyz);
    let scattering = get_scattering();
    let absorption = get_absorption();
    let shaft_intensity = get_shaft_intensity();
    let sun_intensity = get_sun_intensity();
    let base_density = get_density();
    
    // Scene scale for normalization
    let scene_scale = max(get_far() * 0.1, 1000.0);
    let density_multiplier = 50.0 / scene_scale;  // Stronger fog effect
    
    // Limit fog distance to avoid excessive accumulation
    let fog_max_dist = min(scene_dist, scene_scale * 2.0);
    let step_size = fog_max_dist / f32(steps);
    
    // Base fog color (atmospheric blue-gray)
    let fog_base_color = vec3<f32>(0.7, 0.8, 0.9);
    // Sun-tinted fog color (warm golden)
    let sun_fog_color = vec3<f32>(1.0, 0.95, 0.85);
    
    var accumulated_fog = 0.0;
    var light_accumulation = 0.0;
    
    // Jitter start position to reduce banding
    let jitter = fract(sin(dot(ray_dir.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 0.5;
    
    for (var i = 0u; i < steps; i++) {
        let t = (f32(i) + 0.5 + jitter) * step_size;
        if t >= scene_dist {
            break;
        }
        
        let pos = ray_origin + ray_dir * t;
        let local_density = fog_density_at(pos) * density_multiplier;
        
        if local_density < 0.00001 {
            continue;
        }
        
        // Phase function for directional scattering
        let cos_theta = dot(ray_dir, sun_dir);
        let phase = phase_hg(cos_theta, scattering);
        
        // Light shaft shadow estimation
        let shadow = estimate_light_shaft_shadow(pos, sun_dir);
        
        // Height-based light attenuation for god ray effect
        let height_norm = clamp(pos.y / scene_scale, 0.0, 1.0);
        let height_light = 1.0 - height_norm * 0.5;  // More light at lower heights
        
        // Accumulated light contribution (creates visible shafts)
        let light_contrib = shadow * phase * sun_intensity * shaft_intensity * height_light;
        light_accumulation += light_contrib * local_density * step_size * (1.0 - accumulated_fog);
        
        // Accumulated fog density
        let fog_step = local_density * step_size * (1.0 + absorption);
        accumulated_fog += fog_step * (1.0 - accumulated_fog);
        
        // Early exit if fog is saturated
        if accumulated_fog > 0.85 {
            accumulated_fog = 0.85;
            break;
        }
    }
    
    // Blend fog color based on light accumulation (sun influence)
    let sun_influence = clamp(light_accumulation * 2.0, 0.0, 0.7);
    let fog_color = mix(fog_base_color, sun_fog_color, sun_influence);
    
    return vec4<f32>(fog_color, accumulated_fog);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    
    // Sample scene color
    let scene_color = textureSample(color_tex, color_sampler, uv).rgb;
    
    // Get density - if zero, just pass through (baseline preservation)
    let density = get_density();
    if density < 0.0001 {
        return vec4<f32>(scene_color, 1.0);
    }
    
    // Sample depth buffer to get scene distance
    let ndc_depth = textureSample(depth_tex, depth_sampler, uv);
    let near = get_near();
    let far = get_far();
    
    // Detect sky pixels (at or very close to far plane in reverse-Z)
    // In reverse-Z: ndc_depth near 0 means far plane (sky)
    let is_sky = ndc_depth < 0.001;
    
    // For sky pixels, skip volumetric processing to avoid artifacts
    if is_sky {
        return vec4<f32>(scene_color, 1.0);
    }
    
    // Reconstruct world position and calculate ray parameters
    let camera_pos = u.camera_pos.xyz;
    let world_pos = world_pos_from_depth(uv, ndc_depth);
    let ray_dir = normalize(world_pos - camera_pos);
    let scene_dist = length(world_pos - camera_pos);
    
    // Scene scale for density normalization
    let scene_scale = max(far * 0.1, 1000.0);
    let density_multiplier = 1000.0 / scene_scale;
    
    // Use raymarching for proper volumetric effects (especially light shafts)
    let light_shafts = get_light_shafts_enabled();
    
    if light_shafts {
        // Full volumetric raymarching with light shafts
        let vol_result = raymarch_fog_with_depth(camera_pos, ray_dir, scene_dist);
        let fog_color = vol_result.rgb;
        let fog_amount = vol_result.a;
        
        // Composite with scene
        let final_color = mix(scene_color, fog_color, fog_amount);
        return vec4<f32>(final_color, 1.0);
    }
    
    // Simplified fog for non-light-shaft modes (faster)
    let mode = get_mode();
    var fog_amount: f32;
    
    // Distance-normalized fog factor
    let dist_factor = scene_dist / scene_scale;
    
    if mode == 0u {
        // Uniform fog: exponential distance fog
        // density 0.01-0.05 should produce visible fog at typical viewing distances
        let fog_strength = density * 50.0;  // Amplify density for visible effect
        fog_amount = 1.0 - exp(-fog_strength * dist_factor);
    } else {
        // Height-based fog: denser at lower elevations
        let height_falloff = get_height_falloff();
        
        // Height factor: fog is denser at lower heights
        // Normalize height relative to scene (0 = lowest, 1 = highest visible)
        let height_norm = clamp(world_pos.y / scene_scale, 0.0, 2.0);
        let height_factor = exp(-height_norm * height_falloff * 20.0);
        
        // Combined distance and height fog
        let fog_strength = density * 60.0;  // Slightly stronger for height mode
        let base_fog = 1.0 - exp(-fog_strength * dist_factor);
        fog_amount = base_fog * (0.2 + 0.8 * height_factor);
    }
    
    // Clamp fog to preserve scene visibility
    fog_amount = clamp(fog_amount, 0.0, 0.75);
    
    // Calculate fog color with scattering
    let sun_dir = normalize(u.sun_direction.xyz);
    let cos_angle = dot(ray_dir, sun_dir);
    let scattering = get_scattering();
    let phase = phase_hg(cos_angle, scattering);
    
    // Base atmospheric fog color (blue-gray)
    let fog_base_color = vec3<f32>(0.7, 0.8, 0.9);
    
    // Sun-influenced color (warm golden for forward scatter)
    let sun_color = vec3<f32>(1.0, 0.95, 0.85);
    
    // Scattering effect: stronger when looking toward sun
    let sun_intensity = get_sun_intensity();
    let scatter_strength = phase * sun_intensity * scattering;
    
    // Blend fog colors based on scattering
    var fog_color = mix(fog_base_color, sun_color, clamp(scatter_strength, 0.0, 0.6));
    
    // Apply absorption (slightly darkens fog)
    let absorption = get_absorption();
    fog_color *= (1.0 - absorption * 0.2);
    
    // Composite: blend scene with fog
    let final_color = mix(scene_color, fog_color, fog_amount);
    
    return vec4<f32>(final_color, 1.0);
}
