//! H17,H19: GPU culling compute shader
//! Frustum culling and draw command generation

struct CullingUniforms {
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,  // 6 frustum planes (left, right, top, bottom, near, far)
    camera_position: vec3<f32>,
    _pad0: f32,
    cull_distance: f32,
    enable_frustum_cull: u32,
    enable_distance_cull: u32,
    enable_occlusion_cull: u32,
}

struct CullableInstance {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    transform: mat4x4<f32>,
    primitive_type: u32,
    draw_command_index: u32,
}

struct IndirectDrawCommand {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

struct Counters {
    total_objects: atomic<u32>,
    visible_objects: atomic<u32>,
    frustum_culled: atomic<u32>,
    distance_culled: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> uniforms: CullingUniforms;

@group(0) @binding(1)
var<storage, read> input_instances: array<CullableInstance>;

@group(0) @binding(2)
var<storage, read_write> draw_commands: array<IndirectDrawCommand>;

@group(0) @binding(3)
var<storage, read_write> counters: Counters;

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let instance_index = global_id.x;
    
    // Bounds check
    if (instance_index >= arrayLength(&input_instances)) {
        return;
    }
    
    let instance = input_instances[instance_index];
    atomicAdd(&counters.total_objects, 1u);
    
    // Calculate object world space center and bounds
    let aabb_center = (instance.aabb_min + instance.aabb_max) * 0.5;
    let world_center = (instance.transform * vec4<f32>(aabb_center, 1.0)).xyz;
    let world_extent = (instance.aabb_max - instance.aabb_min) * 0.5;
    
    // Calculate bounding sphere radius (conservative)
    let radius = length(world_extent);
    
    var is_visible = true;
    
    // Distance culling
    if (uniforms.enable_distance_cull != 0u) {
        let distance_to_camera = length(world_center - uniforms.camera_position);
        if (distance_to_camera > uniforms.cull_distance) {
            atomicAdd(&counters.distance_culled, 1u);
            is_visible = false;
        }
    }
    
    // Frustum culling using sphere-plane tests
    if (is_visible && uniforms.enable_frustum_cull != 0u) {
        var inside_frustum = true;
        
        // Test against all 6 frustum planes
        for (var i = 0u; i < 6u; i++) {
            let plane = uniforms.frustum_planes[i];
            let distance_to_plane = dot(plane.xyz, world_center) + plane.w;
            
            // If sphere is completely behind any plane, it's culled
            if (distance_to_plane < -radius) {
                inside_frustum = false;
                break;
            }
        }
        
        if (!inside_frustum) {
            atomicAdd(&counters.frustum_culled, 1u);
            is_visible = false;
        }
    }
    
    // TODO: Occlusion culling
    if (is_visible && uniforms.enable_occlusion_cull != 0u) {
        // Placeholder for hierarchical Z-buffer occlusion culling
        // Would test object against depth pyramid/Hi-Z buffer
        // is_visible = is_visible && !is_occluded(world_center, radius);
    }
    
    // Generate draw command if visible
    if (is_visible) {
        atomicAdd(&counters.visible_objects, 1u);
        
        // Get vertex count based on primitive type
        var vertex_count = 3u; // Default to triangle
        switch (instance.primitive_type) {
            case 0u: { vertex_count = 3u; } // Triangle
            case 1u: { vertex_count = 4u; } // Quad
            case 2u: { vertex_count = 1u; } // Point
            case 3u: { vertex_count = 2u; } // Line
            default: { vertex_count = 3u; }
        }
        
        // Write draw command
        let draw_index = instance.draw_command_index;
        if (draw_index < arrayLength(&draw_commands)) {
            draw_commands[draw_index] = IndirectDrawCommand(
                vertex_count,        // vertex_count
                1u,                  // instance_count
                0u,                  // first_vertex
                instance_index       // first_instance
            );
        }
    }
}

// Helper function for sphere-AABB intersection (for future occlusion culling)
fn sphere_aabb_intersect(sphere_center: vec3<f32>, sphere_radius: f32, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let closest_point = clamp(sphere_center, aabb_min, aabb_max);
    let distance_squared = dot(sphere_center - closest_point, sphere_center - closest_point);
    return distance_squared <= sphere_radius * sphere_radius;
}

// Extract frustum planes from view-projection matrix (for CPU-GPU consistency)
fn extract_frustum_planes(view_proj: mat4x4<f32>) -> array<vec4<f32>, 6> {
    var planes: array<vec4<f32>, 6>;
    
    // Left plane: row4 + row1
    planes[0] = vec4<f32>(
        view_proj[0][3] + view_proj[0][0],
        view_proj[1][3] + view_proj[1][0], 
        view_proj[2][3] + view_proj[2][0],
        view_proj[3][3] + view_proj[3][0]
    );
    
    // Right plane: row4 - row1  
    planes[1] = vec4<f32>(
        view_proj[0][3] - view_proj[0][0],
        view_proj[1][3] - view_proj[1][0],
        view_proj[2][3] - view_proj[2][0], 
        view_proj[3][3] - view_proj[3][0]
    );
    
    // Top plane: row4 - row2
    planes[2] = vec4<f32>(
        view_proj[0][3] - view_proj[0][1],
        view_proj[1][3] - view_proj[1][1],
        view_proj[2][3] - view_proj[2][1],
        view_proj[3][3] - view_proj[3][1]
    );
    
    // Bottom plane: row4 + row2
    planes[3] = vec4<f32>(
        view_proj[0][3] + view_proj[0][1],
        view_proj[1][3] + view_proj[1][1],
        view_proj[2][3] + view_proj[2][1], 
        view_proj[3][3] + view_proj[3][1]
    );
    
    // Near plane: row4 + row3
    planes[4] = vec4<f32>(
        view_proj[0][3] + view_proj[0][2],
        view_proj[1][3] + view_proj[1][2],
        view_proj[2][3] + view_proj[2][2],
        view_proj[3][3] + view_proj[3][2]
    );
    
    // Far plane: row4 - row3
    planes[5] = vec4<f32>(
        view_proj[0][3] - view_proj[0][2],
        view_proj[1][3] - view_proj[1][2], 
        view_proj[2][3] - view_proj[2][2],
        view_proj[3][3] - view_proj[3][2]
    );
    
    // Normalize planes
    for (var i = 0u; i < 6u; i++) {
        let length = sqrt(dot(planes[i].xyz, planes[i].xyz));
        planes[i] = planes[i] / length;
    }
    
    return planes;
}