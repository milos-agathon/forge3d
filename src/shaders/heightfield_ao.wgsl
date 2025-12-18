// src/shaders/heightfield_ao.wgsl
// Heightfield ray-traced ambient occlusion compute shader
// Computes horizon-based AO by ray-marching the heightfield in multiple directions
// Output: R8Unorm texture where 1.0 = no occlusion, 0.0 = fully occluded

struct HeightAoUniforms {
    // x=directions, y=steps, z=max_distance (world units), w=strength
    params0: vec4<f32>,
    // x=spacing_x, y=spacing_y, z=height_scale, w=height_min
    params1: vec4<f32>,
    // x=output_width, y=output_height, z=height_tex_width, w=height_tex_height
    params2: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u_ao: HeightAoUniforms;

@group(0) @binding(1)
var height_tex: texture_2d<f32>;

@group(0) @binding(2)
var height_samp: sampler;

@group(0) @binding(3)
var ao_output: texture_storage_2d<r32float, write>;

const PI: f32 = 3.14159265359;

fn sample_height(uv: vec2<f32>) -> f32 {
    // Use textureLoad since R32Float doesn't support filtering
    let tex_size = vec2<f32>(u_ao.params2.z, u_ao.params2.w);
    let pixel = vec2<i32>(uv * tex_size);
    let clamped_pixel = clamp(pixel, vec2<i32>(0), vec2<i32>(tex_size) - vec2<i32>(1));
    let h = textureLoad(height_tex, clamped_pixel, 0).r;
    return h * u_ao.params1.z + u_ao.params1.w;
}

fn compute_horizon_ao(center_uv: vec2<f32>, center_height: f32) -> f32 {
    let directions = u32(u_ao.params0.x);
    let steps = u32(u_ao.params0.y);
    let max_distance = u_ao.params0.z;
    let spacing = vec2<f32>(u_ao.params1.x, u_ao.params1.y);
    
    // Convert max_distance from world units to UV space
    let height_tex_size = vec2<f32>(u_ao.params2.z, u_ao.params2.w);
    let world_extent = spacing * height_tex_size;
    let max_uv_distance = max_distance / max(world_extent.x, world_extent.y);
    let step_uv = max_uv_distance / f32(steps);
    
    var total_occlusion: f32 = 0.0;
    
    // March in multiple directions around the horizon
    for (var d: u32 = 0u; d < directions; d = d + 1u) {
        let angle = (f32(d) / f32(directions)) * 2.0 * PI;
        let dir = vec2<f32>(cos(angle), sin(angle));
        
        var max_tan_angle: f32 = -999.0;
        
        // March along this direction
        for (var s: u32 = 1u; s <= steps; s = s + 1u) {
            let sample_uv = center_uv + dir * step_uv * f32(s);
            
            // Skip if outside [0,1] bounds
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                break;
            }
            
            let sample_height = sample_height(sample_uv);
            let height_diff = sample_height - center_height;
            let horizontal_dist = length((sample_uv - center_uv) * world_extent);
            
            // Compute tangent of elevation angle
            if (horizontal_dist > 0.001) {
                let tan_angle = height_diff / horizontal_dist;
                max_tan_angle = max(max_tan_angle, tan_angle);
            }
        }
        
        // Convert max tangent angle to occlusion factor
        // Positive tan_angle means terrain is above us → occlusion
        // Use atan to get angle, then map to [0,1]
        if (max_tan_angle > -999.0) {
            let horizon_angle = atan(max_tan_angle);
            // Normalize: 0 at horizon (angle=0), 1 at 90° above (angle=PI/2)
            let occlusion = clamp(horizon_angle / (PI * 0.5), 0.0, 1.0);
            total_occlusion = total_occlusion + occlusion;
        }
    }
    
    // Average occlusion across all directions
    let avg_occlusion = total_occlusion / f32(directions);
    
    // Return visibility (1 - occlusion), so 1.0 = fully visible, 0.0 = fully occluded
    return 1.0 - avg_occlusion;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = vec2<u32>(u32(u_ao.params2.x), u32(u_ao.params2.y));
    
    // Bounds check
    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {
        return;
    }
    
    // Compute UV for this pixel (center of pixel)
    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(output_size);
    
    // Sample center height
    let center_height = sample_height(uv);
    
    // Compute horizon-based AO
    let ao = compute_horizon_ao(uv, center_height);
    
    // Apply strength multiplier
    let strength = u_ao.params0.w;
    let final_ao = mix(1.0, ao, strength);
    
    // Write output (R8Unorm: 1.0 = no occlusion)
    textureStore(ao_output, vec2<i32>(global_id.xy), vec4<f32>(final_ao, 0.0, 0.0, 1.0));
}
