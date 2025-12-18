// src/shaders/heightfield_sun_vis.wgsl
// Heightfield ray-traced sun visibility compute shader
// Computes visibility factor by ray-marching along sun direction
// Output: R32Float texture where 1.0 = fully lit, 0.0 = fully shadowed

struct SunVisUniforms {
    // x=samples, y=steps, z=max_distance (world units), w=softness
    params0: vec4<f32>,
    // x=spacing_x, y=spacing_y, z=height_scale, w=height_min
    params1: vec4<f32>,
    // x=output_width, y=output_height, z=height_tex_width, w=height_tex_height
    params2: vec4<f32>,
    // x=sun_dir_x, y=sun_dir_y, z=sun_dir_z (normalized), w=bias
    params3: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u_sun: SunVisUniforms;

@group(0) @binding(1)
var height_tex: texture_2d<f32>;

@group(0) @binding(2)
var height_samp: sampler;

@group(0) @binding(3)
var vis_output: texture_storage_2d<r32float, write>;

fn sample_height(uv: vec2<f32>) -> f32 {
    // Use textureLoad since R32Float doesn't support filtering
    let tex_size = vec2<f32>(u_sun.params2.z, u_sun.params2.w);
    let pixel = vec2<i32>(uv * tex_size);
    let clamped_pixel = clamp(pixel, vec2<i32>(0), vec2<i32>(tex_size) - vec2<i32>(1));
    let h = textureLoad(height_tex, clamped_pixel, 0).r;
    return h * u_sun.params1.z + u_sun.params1.w;
}

fn compute_sun_visibility(center_uv: vec2<f32>, center_height: f32) -> f32 {
    let samples = u32(u_sun.params0.x);
    let steps = u32(u_sun.params0.y);
    let max_distance = u_sun.params0.z;
    let softness = u_sun.params0.w;
    let spacing = vec2<f32>(u_sun.params1.x, u_sun.params1.y);
    let bias = u_sun.params3.w;
    
    // Sun direction (pointing toward sun, normalized)
    let sun_dir = vec3<f32>(u_sun.params3.x, u_sun.params3.y, u_sun.params3.z);
    
    // Project sun direction onto XZ plane (UV plane) for ray marching
    // sun_dir.y is the vertical component (elevation)
    let sun_horizontal = vec2<f32>(sun_dir.x, sun_dir.z);
    let sun_horizontal_len = length(sun_horizontal);
    
    // If sun is directly overhead, no shadows from heightfield
    if (sun_horizontal_len < 0.001) {
        return 1.0;
    }
    
    // Normalize horizontal direction
    let march_dir = sun_horizontal / sun_horizontal_len;
    
    // Compute sun elevation tangent (vertical / horizontal)
    let sun_tan_elevation = sun_dir.y / sun_horizontal_len;
    
    // Convert max_distance from world units to UV space
    let height_tex_size = vec2<f32>(u_sun.params2.z, u_sun.params2.w);
    let world_extent = spacing * height_tex_size;
    let max_uv_distance = max_distance / max(world_extent.x, world_extent.y);
    let step_uv = max_uv_distance / f32(steps);
    
    var total_visibility: f32 = 0.0;
    
    // For soft shadows, use multiple samples with jitter
    // For hard shadows (samples=1), single ray
    for (var s: u32 = 0u; s < samples; s = s + 1u) {
        // Simple jitter based on sample index and pixel position
        let jitter_offset = (f32(s) - f32(samples - 1u) * 0.5) * 0.1;
        let jittered_dir = normalize(march_dir + vec2<f32>(jitter_offset, -jitter_offset));
        
        var sample_visibility: f32 = 1.0;
        var accumulated_occlusion: f32 = 0.0;
        
        // March along sun direction (toward the sun)
        for (var t: u32 = 1u; t <= steps; t = t + 1u) {
            let march_dist_uv = step_uv * f32(t);
            let sample_uv = center_uv + jittered_dir * march_dist_uv;
            
            // Skip if outside [0,1] bounds
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                break;
            }
            
            // Sample height at this position
            let sample_h = sample_height(sample_uv);
            
            // Compute horizontal distance in world units
            let horizontal_dist = length((sample_uv - center_uv) * world_extent);
            
            // Expected height along sun ray at this distance
            // If sun is at angle θ, expected height = center_height + horizontal_dist * tan(θ)
            let expected_height = center_height + bias + horizontal_dist * sun_tan_elevation;
            
            // If terrain is above the expected sun ray height, we're in shadow
            let height_diff = sample_h - expected_height;
            
            if (height_diff > 0.0) {
                if (softness > 0.0) {
                    // Soft shadows: accumulate partial occlusion
                    let occlusion = smoothstep(0.0, softness, height_diff);
                    accumulated_occlusion = max(accumulated_occlusion, occlusion);
                } else {
                    // Hard shadows: binary occlusion
                    sample_visibility = 0.0;
                    break;
                }
            }
        }
        
        // Apply soft shadow occlusion
        if (softness > 0.0) {
            sample_visibility = 1.0 - accumulated_occlusion;
        }
        
        total_visibility = total_visibility + sample_visibility;
    }
    
    // Average visibility across all samples
    return total_visibility / f32(samples);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = vec2<u32>(u32(u_sun.params2.x), u32(u_sun.params2.y));
    
    // Bounds check
    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {
        return;
    }
    
    // Compute UV for this pixel (center of pixel)
    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(output_size);
    
    // Sample center height
    let center_height = sample_height(uv);
    
    // Compute sun visibility
    let visibility = compute_sun_visibility(uv, center_height);
    
    // Write output
    textureStore(vis_output, vec2<i32>(global_id.xy), vec4<f32>(visibility, 0.0, 0.0, 1.0));
}
