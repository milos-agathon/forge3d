// shaders/ssr.wgsl
// P5: Screen-Space Reflections
// Hierarchical Z-buffer ray marching with thickness testing

struct SsrSettings {
    max_steps: u32,
    thickness: f32,
    max_distance: f32,
    intensity: f32,
    inv_resolution: vec2<f32>,
    _pad0: vec2<f32>,
};

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

// SSR compute pass
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var color_tex: texture_2d<f32>;
@group(0) @binding(3) var ssr_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> settings: SsrSettings;
@group(0) @binding(5) var<uniform> camera: CameraParams;
@group(0) @binding(6) var env_texture: texture_cube<f32>;
@group(0) @binding(7) var env_sampler: sampler;

// Reconstruct view-space position from depth
fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view_pos = camera.inv_proj_matrix * ndc;
    return view_pos.xyz / view_pos.w;
}

// Project view-space position to screen UV and depth
fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = camera.proj_matrix * vec4<f32>(view_pos, 1.0);
    let ndc = clip_pos.xyz / clip_pos.w;
    let uv = ndc.xy * 0.5 + 0.5;
    return vec3<f32>(uv, ndc.z);
}

// Hash function for noise
fn hash(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    let p3_dot = dot(p3, vec3<f32>(p3.yzx) + 33.33);
    return fract((p3.x + p3.y) * p3_dot + p3.z);
}

// Binary search refinement for ray intersection
fn binary_search(ray_start: vec3<f32>, ray_end: vec3<f32>, dims: vec2<u32>) -> vec3<f32> {
    var start = ray_start;
    var end = ray_end;
    
    for (var i = 0; i < 4; i++) {
        let mid = (start + end) * 0.5;
        let screen = project_to_screen(mid);
        
        if (screen.x < 0.0 || screen.x > 1.0 || screen.y < 0.0 || screen.y > 1.0) {
            return vec3<f32>(-1.0);
        }
        
        let sample_coord = vec2<i32>(screen.xy * vec2<f32>(dims));
        let sample_depth = textureLoad(depth_tex, sample_coord, 0).r;
        
        if (screen.z > sample_depth) {
            end = mid;
        } else {
            start = mid;
        }
    }
    
    let final_screen = project_to_screen(end);
    return final_screen;
}

// Screen-space ray march with hierarchical Z-buffer
fn screen_space_ray_march(origin: vec3<f32>, direction: vec3<f32>, dims: vec2<u32>) -> vec4<f32> {
    // Calculate ray end point
    let ray_end = origin + direction * settings.max_distance;
    
    // Project start and end to screen space
    let start_screen = project_to_screen(origin);
    let end_screen = project_to_screen(ray_end);
    
    // Check if ray goes off-screen
    if (start_screen.x < 0.0 || start_screen.x > 1.0 || 
        start_screen.y < 0.0 || start_screen.y > 1.0) {
        return vec4<f32>(0.0);
    }
    
    // Calculate step in screen space
    let ray_screen = end_screen - start_screen;
    let step_size = 1.0 / f32(settings.max_steps);
    let ray_step = ray_screen * step_size;
    
    var current_screen = start_screen;
    var prev_pos = origin;
    var current_pos = origin;
    
    // March the ray
    for (var i = 0u; i < settings.max_steps; i = i + 1u) {
        current_screen += ray_step;
        
        // Check bounds
        if (current_screen.x < 0.0 || current_screen.x > 1.0 || 
            current_screen.y < 0.0 || current_screen.y > 1.0) {
            return vec4<f32>(0.0);
        }
        
        // Sample depth buffer
        let sample_coord = vec2<i32>(current_screen.xy * vec2<f32>(dims));
        let sample_depth = textureLoad(depth_tex, sample_coord, 0).r;
        
        // Reconstruct current ray position
        let t = f32(i + 1u) * step_size;
        current_pos = origin + direction * settings.max_distance * t;
        
        // Check for intersection with thickness testing
        let depth_diff = current_screen.z - sample_depth;
        if (depth_diff > 0.0 && depth_diff < settings.thickness) {
            // Refine with binary search
            let refined = binary_search(prev_pos, current_pos, dims);
            
            if (refined.x >= 0.0) {
                // Valid hit - sample color
                let hit_coord = vec2<i32>(refined.xy * vec2<f32>(dims));
                let hit_color = textureLoad(color_tex, hit_coord, 0).rgb;
                
                // Calculate fade based on screen edge distance
                let edge_fade = min(
                    min(refined.x, 1.0 - refined.x),
                    min(refined.y, 1.0 - refined.y)
                ) * 5.0;
                let fade = clamp(edge_fade, 0.0, 1.0);
                
                return vec4<f32>(hit_color, fade);
            }
        }
        
        prev_pos = current_pos;
    }
    
    return vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(depth_tex);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(dims);
    
    // Sample center pixel
    let center_depth = textureLoad(depth_tex, coord, 0).r;
    let center_normal_encoded = textureLoad(normal_tex, coord, 0).xyz;
    let center_normal = normalize(center_normal_encoded * 2.0 - 1.0);
    
    // Early out for sky/far plane
    if (center_depth >= 0.9999) {
        textureStore(ssr_output, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    let center_pos = reconstruct_position(uv, center_depth);
    let view_dir = normalize(-center_pos);
    
    // Calculate reflection direction
    let reflect_dir = reflect(-view_dir, center_normal);
    
    // Ray march
    let reflection = screen_space_ray_march(center_pos, reflect_dir, dims);
    
    // If no hit, sample environment map
    if (reflection.a < 0.01) {
        // Convert view-space reflection direction to world-space using inverse view
        let world_reflect = (camera.inv_view_matrix * vec4<f32>(reflect_dir, 0.0)).xyz;
        let env_color = textureSampleLevel(env_texture, env_sampler, world_reflect, 0.0).rgb;
        textureStore(ssr_output, coord, vec4<f32>(env_color * settings.intensity, 0.5));
    } else {
        textureStore(ssr_output, coord, vec4<f32>(reflection.rgb * settings.intensity, reflection.a));
    }
}

// Temporal filter for SSR
@group(1) @binding(0) var ssr_current: texture_2d<f32>;
@group(1) @binding(1) var ssr_history: texture_2d<f32>;
@group(1) @binding(2) var ssr_filtered: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_ssr_temporal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(ssr_current);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let coord = vec2<i32>(global_id.xy);
    let current = textureLoad(ssr_current, coord, 0);
    let history = textureLoad(ssr_history, coord, 0);
    
    // Temporal blend with higher weight to current for reflections
    let alpha = 0.2;
    let filtered = mix(history, current, alpha);
    
    textureStore(ssr_filtered, coord, filtered);
}
