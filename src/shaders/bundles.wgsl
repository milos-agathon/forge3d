// Render Bundles shaders for optimized GPU command execution
// Provides efficient rendering of multiple objects with shared state

// Common vertex input for bundle rendering
struct BundleVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
}

// Instance data for instanced rendering
struct InstanceData {
    @location(4) transform_0: vec4<f32>,  // Transform matrix row 0
    @location(5) transform_1: vec4<f32>,  // Transform matrix row 1
    @location(6) transform_2: vec4<f32>,  // Transform matrix row 2
    @location(7) transform_3: vec4<f32>,  // Transform matrix row 3
    @location(8) instance_color: vec4<f32>, // Per-instance color modulation
}

// Camera uniforms
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
}

// Bundle rendering uniforms
struct BundleUniforms {
    time: f32,              // Animation time
    alpha: f32,             // Global alpha multiplier
    tint_color: vec4<f32>,  // Global tint color
    flags: u32,             // Rendering flags
    _padding: vec3<f32>,
}

// Bind groups
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> bundle_uniforms: BundleUniforms;
@group(2) @binding(0) var bundle_texture: texture_2d<f32>;
@group(2) @binding(1) var bundle_sampler: sampler;

// Vertex output structure
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) instance_id: u32,
}

// =============================================================================
// INSTANCED RENDERING SHADERS
// =============================================================================

@vertex
fn instanced_vs_main(input: BundleVertexInput, instance: InstanceData, 
                     @builtin(instance_index) instance_id: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Reconstruct instance transform matrix
    let instance_transform = mat4x4<f32>(
        instance.transform_0,
        instance.transform_1,
        instance.transform_2,
        instance.transform_3
    );
    
    // Transform vertex to world space
    let world_pos = instance_transform * vec4<f32>(input.position, 1.0);
    out.world_position = world_pos.xyz;
    
    // Transform normal to world space (assume uniform scaling)
    let world_normal = (instance_transform * vec4<f32>(input.normal, 0.0)).xyz;
    out.world_normal = normalize(world_normal);
    
    // Transform to clip space
    out.clip_position = camera.view_projection * world_pos;
    
    // Pass through other attributes
    out.uv = input.uv;
    out.color = input.color * instance.instance_color;
    out.instance_id = instance_id;
    
    return out;
}

@fragment
fn instanced_fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample texture
    var base_color = textureSample(bundle_texture, bundle_sampler, input.uv);
    
    // Apply vertex color
    base_color *= input.color;
    
    // Apply global tint and alpha
    base_color *= bundle_uniforms.tint_color;
    base_color.a *= bundle_uniforms.alpha;
    
    // Simple lighting calculation
    let light_dir = normalize(vec3<f32>(0.3, -0.7, 0.5));
    let n_dot_l = max(dot(input.world_normal, -light_dir), 0.0);
    let ambient = vec3<f32>(0.2, 0.2, 0.2);
    let diffuse = base_color.rgb * n_dot_l;
    
    let final_color = ambient + diffuse;
    
    return vec4<f32>(final_color, base_color.a);
}

// =============================================================================
// UI RENDERING SHADERS
// =============================================================================

// UI vertex input (simpler than 3D)
struct UiVertexInput {
    @location(0) position: vec2<f32>,  // Screen space position
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct UiVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

// UI uniforms
struct UiUniforms {
    screen_size: vec2<f32>,     // Screen dimensions
    ui_scale: f32,              // UI scaling factor
    _padding: f32,
}

@group(0) @binding(0) var<uniform> ui_uniforms: UiUniforms;
@group(1) @binding(0) var ui_texture: texture_2d<f32>;
@group(1) @binding(1) var ui_sampler: sampler;

@vertex
fn ui_vs_main(input: UiVertexInput) -> UiVertexOutput {
    var out: UiVertexOutput;
    
    // Convert screen space to clip space
    let screen_pos = input.position * ui_uniforms.ui_scale;
    let clip_pos = vec2<f32>(
        (screen_pos.x / ui_uniforms.screen_size.x) * 2.0 - 1.0,
        1.0 - (screen_pos.y / ui_uniforms.screen_size.y) * 2.0
    );
    
    out.clip_position = vec4<f32>(clip_pos, 0.0, 1.0);
    out.uv = input.uv;
    out.color = input.color;
    
    return out;
}

@fragment
fn ui_fs_main(input: UiVertexOutput) -> @location(0) vec4<f32> {
    // Sample UI texture (often a texture atlas)
    var texture_color = textureSample(ui_texture, ui_sampler, input.uv);
    
    // Apply vertex color modulation
    var final_color = texture_color * input.color;
    
    // Apply global UI effects
    final_color.a *= bundle_uniforms.alpha;
    
    return final_color;
}

// =============================================================================
// PARTICLE RENDERING SHADERS
// =============================================================================

// Particle vertex input
struct ParticleVertexInput {
    @location(0) position: vec3<f32>,      // World position
    @location(1) velocity: vec3<f32>,      // Velocity vector
    @location(2) size: f32,                // Particle size
    @location(3) life: f32,                // Particle life (0.0 = dead, 1.0 = born)
    @location(4) color: vec4<f32>,         // Particle color
}

struct ParticleVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) life: f32,
}

// Particle uniforms
struct ParticleUniforms {
    gravity: vec3<f32>,         // Gravity acceleration
    drag: f32,                  // Air resistance
    time_delta: f32,            // Time step for animation
    fade_start: f32,            // Life value to start fading
    fade_end: f32,              // Life value to end fading
    _padding: f32,
}

@group(1) @binding(1) var<uniform> particle_uniforms: ParticleUniforms;

@vertex
fn particle_vs_main(input: ParticleVertexInput, @builtin(vertex_index) vertex_id: u32) -> ParticleVertexOutput {
    var out: ParticleVertexOutput;
    
    // Skip dead particles
    if (input.life <= 0.0) {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.life = 0.0;
        return out;
    }
    
    // Generate quad vertices for billboard
    let quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),  // Bottom-left
        vec2<f32>( 1.0, -1.0),  // Bottom-right
        vec2<f32>(-1.0,  1.0),  // Top-left
        vec2<f32>( 1.0,  1.0)   // Top-right
    );
    
    let quad_uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),  // Bottom-left
        vec2<f32>(1.0, 1.0),  // Bottom-right
        vec2<f32>(0.0, 0.0),  // Top-left
        vec2<f32>(1.0, 0.0)   // Top-right
    );
    
    // Get current vertex position and UV
    let vertex_offset = quad_vertices[vertex_id % 4u];
    out.uv = quad_uvs[vertex_id % 4u];
    
    // Calculate camera right and up vectors for billboarding
    let view_matrix = camera.view;
    let camera_right = normalize(vec3<f32>(view_matrix[0].x, view_matrix[1].x, view_matrix[2].x));
    let camera_up = normalize(vec3<f32>(view_matrix[0].y, view_matrix[1].y, view_matrix[2].y));
    
    // Create billboard quad in world space
    let world_position = input.position + 
        (camera_right * vertex_offset.x + camera_up * vertex_offset.y) * input.size;
    
    // Transform to clip space
    out.clip_position = camera.view_projection * vec4<f32>(world_position, 1.0);
    
    // Calculate alpha based on life
    var alpha = input.color.a;
    if (input.life < particle_uniforms.fade_end) {
        let fade_range = particle_uniforms.fade_start - particle_uniforms.fade_end;
        let fade_factor = (input.life - particle_uniforms.fade_end) / fade_range;
        alpha *= clamp(fade_factor, 0.0, 1.0);
    }
    
    out.color = vec4<f32>(input.color.rgb, alpha);
    out.life = input.life;
    
    return out;
}

@fragment
fn particle_fs_main(input: ParticleVertexOutput) -> @location(0) vec4<f32> {
    // Skip dead particles
    if (input.life <= 0.0) {
        discard;
    }
    
    // Sample particle texture (often a soft circular texture)
    var texture_color = textureSample(bundle_texture, bundle_sampler, input.uv);
    
    // Apply particle color
    var final_color = texture_color * input.color;
    
    // Apply global effects
    final_color.a *= bundle_uniforms.alpha;
    
    // Soft particles: reduce alpha near geometry edges (would need depth buffer)
    // This is a simplified version without depth testing
    
    return final_color;
}

// =============================================================================
// BATCH RENDERING SHADERS (Multiple different objects in one draw call)
// =============================================================================

// Batch rendering supports different geometry types in one bundle
struct BatchVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material_id: u32,        // Which material/texture to use
    @location(5) transform_id: u32,       // Which transform to apply
}

struct BatchVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material_id: u32,
}

// Batch uniforms with transform array
struct BatchUniforms {
    transform_count: u32,
    material_count: u32,
    _padding: vec2<f32>,
    // Followed by array of transforms and material data
}

// Storage buffers for batch data
@group(1) @binding(2) var<storage, read> transforms: array<mat4x4<f32>>;
@group(1) @binding(3) var<storage, read> materials: array<vec4<f32>>; // Material properties
@group(2) @binding(2) var material_textures: texture_2d_array<f32>;   // Texture array

@vertex
fn batch_vs_main(input: BatchVertexInput) -> BatchVertexOutput {
    var out: BatchVertexOutput;
    
    // Get transform for this vertex
    let transform = transforms[input.transform_id];
    
    // Transform vertex to world space
    let world_pos = transform * vec4<f32>(input.position, 1.0);
    out.world_position = world_pos.xyz;
    
    // Transform normal
    let world_normal = (transform * vec4<f32>(input.normal, 0.0)).xyz;
    out.world_normal = normalize(world_normal);
    
    // Transform to clip space
    out.clip_position = camera.view_projection * world_pos;
    
    // Pass through other attributes
    out.uv = input.uv;
    out.color = input.color;
    out.material_id = input.material_id;
    
    return out;
}

@fragment
fn batch_fs_main(input: BatchVertexOutput) -> @location(0) vec4<f32> {
    // Sample from texture array using material ID as layer
    var base_color = textureSample(material_textures, bundle_sampler, input.uv, input.material_id);
    
    // Apply vertex color
    base_color *= input.color;
    
    // Get material properties
    let material_props = materials[input.material_id];
    let metallic = material_props.x;
    let roughness = material_props.y;
    let emission = material_props.z;
    
    // Simple lighting (could be enhanced with PBR)
    let light_dir = normalize(vec3<f32>(0.3, -0.7, 0.5));
    let view_dir = normalize(camera.position - input.world_position);
    let half_dir = normalize(light_dir + view_dir);
    
    let n_dot_l = max(dot(input.world_normal, -light_dir), 0.0);
    let n_dot_h = max(dot(input.world_normal, half_dir), 0.0);
    
    // Simple Blinn-Phong approximation
    let specular_power = mix(1.0, 128.0, 1.0 - roughness);
    let specular = pow(n_dot_h, specular_power) * (1.0 - roughness);
    
    let ambient = vec3<f32>(0.1, 0.1, 0.1);
    let diffuse = base_color.rgb * n_dot_l * (1.0 - metallic);
    let specular_color = mix(vec3<f32>(0.04, 0.04, 0.04), base_color.rgb, metallic);
    let final_specular = specular_color * specular;
    
    var final_color = ambient + diffuse + final_specular;
    
    // Add emission
    final_color += base_color.rgb * emission;
    
    // Apply global effects
    final_color *= bundle_uniforms.tint_color.rgb;
    
    return vec4<f32>(final_color, base_color.a * bundle_uniforms.alpha);
}

// =============================================================================
// DEBUG/WIREFRAME RENDERING SHADERS
// =============================================================================

struct WireframeVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) barycentric: vec3<f32>,  // Barycentric coordinates for wireframe
}

struct WireframeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) barycentric: vec3<f32>,
}

@vertex
fn wireframe_vs_main(input: WireframeVertexInput) -> WireframeVertexOutput {
    var out: WireframeVertexOutput;
    
    // Simple transform to clip space
    out.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    out.barycentric = input.barycentric;
    
    return out;
}

@fragment
fn wireframe_fs_main(input: WireframeVertexOutput) -> @location(0) vec4<f32> {
    // Calculate wireframe using barycentric coordinates
    let line_width = 0.02;
    let edge_distance = min(input.barycentric.x, min(input.barycentric.y, input.barycentric.z));
    
    if (edge_distance < line_width) {
        return bundle_uniforms.tint_color; // Wire color
    } else {
        discard; // Transparent interior
    }
}