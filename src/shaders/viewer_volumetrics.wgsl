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

// Linearize depth from NDC (0=near, 1=far in reversed-Z or standard)
fn linearize_depth(ndc_depth: f32, near: f32, far: f32) -> f32 {
    // For standard perspective projection with reversed-Z (common in wgpu)
    // depth = 0 at far, depth = 1 at near
    // Linear depth = near * far / (far - depth * (far - near))
    let z = ndc_depth;
    return near * far / (far - z * (far - near));
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
fn raymarch_fog_with_depth(ray_origin: vec3<f32>, ray_dir: vec3<f32>, scene_dist: f32) -> vec4<f32> {
    let steps = get_steps();
    let max_dist = min(scene_dist, get_far());
    let step_size = max_dist / f32(steps);
    let sun_dir = normalize(u.sun_direction.xyz);
    let scattering = get_scattering();
    let absorption = get_absorption();
    let light_shafts = get_light_shafts_enabled();
    let shaft_intensity = get_shaft_intensity();
    let sun_intensity = get_sun_intensity();
    
    var inscatter = vec3<f32>(0.0);
    var transmittance = 1.0;
    
    for (var i = 0u; i < steps; i++) {
        let t = (f32(i) + 0.5) * step_size;
        
        // Stop if we've reached the scene geometry
        if t >= scene_dist {
            break;
        }
        
        let pos = ray_origin + ray_dir * t;
        let density = fog_density_at(pos);
        
        // Skip if density is negligible
        if density < 0.0001 {
            continue;
        }
        
        // Phase function for anisotropic scattering toward sun
        let cos_theta = dot(ray_dir, sun_dir);
        let phase = phase_hg(cos_theta, scattering);
        
        // Calculate sun visibility for light shafts
        var sun_vis = 1.0;
        if light_shafts {
            // Full light shaft shadow estimation
            sun_vis = estimate_light_shaft_shadow(pos, sun_dir);
            
            // Additional height-based attenuation for god ray effect
            let height_atten = clamp(pos.y * 0.001 + 0.5, 0.0, 1.0);
            sun_vis *= height_atten;
        }
        
        // In-scattering contribution
        let light = sun_vis * sun_intensity * phase * shaft_intensity;
        
        // Accumulate inscattering and transmittance
        inscatter += transmittance * light * density * step_size;
        transmittance *= exp(-density * absorption * step_size);
        
        // Early exit if fully opaque
        if transmittance < 0.01 {
            break;
        }
    }
    
    return vec4<f32>(inscatter, transmittance);
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
    
    // Linearize depth to get world-space distance
    let linear_depth = linearize_depth(ndc_depth, near, far);
    
    // If depth is at far plane (sky), use full far distance
    let scene_dist = select(linear_depth, far, ndc_depth > 0.9999);
    
    // Reconstruct ray direction from camera through pixel
    let camera_pos = u.camera_pos.xyz;
    let world_pos = world_pos_from_depth(uv, ndc_depth);
    let ray_dir = normalize(world_pos - camera_pos);
    
    // Raymarch fog with depth-aware termination
    let fog_result = raymarch_fog_with_depth(camera_pos, ray_dir, scene_dist);
    let fog_inscatter = fog_result.rgb;
    let fog_transmittance = fog_result.w;
    
    // Atmospheric fog color (blue-ish haze, tinted by sun)
    let sun_color = vec3<f32>(1.0, 0.95, 0.85);  // Warm sun tint
    let sky_color = vec3<f32>(0.5, 0.6, 0.75);   // Cool sky tint
    let sun_dir = normalize(u.sun_direction.xyz);
    let view_sun_angle = max(0.0, dot(ray_dir, sun_dir));
    let fog_tint = mix(sky_color, sun_color, view_sun_angle * view_sun_angle);
    
    let final_inscatter = fog_inscatter * fog_tint;
    
    // Composite: scene * transmittance + inscatter
    let final_color = scene_color * fog_transmittance + final_inscatter;
    
    return vec4<f32>(final_color, 1.0);
}
