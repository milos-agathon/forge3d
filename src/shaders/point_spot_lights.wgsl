// B13: Point & Spot Lights (Realtime) - WGSL shader implementation
// Supports multiple point and spot lights with shadows and penumbra shaping

// Light type constants
const LIGHT_TYPE_POINT: u32 = 0u;
const LIGHT_TYPE_SPOT: u32 = 1u;

// Shadow map constants
const SHADOW_MAP_SIZE: f32 = 1024.0;
const SHADOW_BIAS: f32 = 0.001;

// Individual light structure (64 bytes aligned)
struct Light {
    // Position and type (16 bytes)
    position: vec3<f32>,
    light_type: u32,            // 0 = point, 1 = spot

    // Direction and range (16 bytes)
    direction: vec3<f32>,       // For spot lights
    range: f32,                 // Maximum light distance

    // Color and intensity (16 bytes)
    color: vec3<f32>,
    intensity: f32,

    // Spot light parameters (16 bytes)
    inner_cone_angle: f32,      // Inner cone angle (radians)
    outer_cone_angle: f32,      // Outer cone angle (radians)
    penumbra_softness: f32,     // Penumbra transition softness
    shadow_enabled: f32,        // 0.0 = disabled, 1.0 = enabled
}

// Per-frame uniforms
struct PointSpotLightUniforms {
    // Camera and view parameters (64 bytes)
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,

    // Global lighting (16 bytes)
    ambient_color: vec3<f32>,
    ambient_intensity: f32,

    // Light count and control (16 bytes)
    active_light_count: u32,
    max_lights: u32,
    shadow_quality: u32,        // 0=off, 1=low, 2=medium, 3=high
    debug_mode: u32,            // 0=normal, 1=show_light_bounds, 2=show_shadows

    // Global shadow parameters (16 bytes)
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_softness: f32,
    _pad0: f32,
}

// Light array (storage buffer)
struct LightArray {
    lights: array<Light>,
}

// Vertex output structure
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_pos: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

// Vertex shader
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Full-screen triangle
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );

    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    out.world_pos = vec3<f32>(uv * 2.0 - 1.0, 0.0);
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.view_pos = out.world_pos;

    return out;
}

// Point light attenuation calculation
fn calculate_point_light_attenuation(distance: f32, range: f32) -> f32 {
    if distance >= range {
        return 0.0;
    }

    // Physically based attenuation with falloff
    let attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    // Smooth cutoff near range limit
    let range_factor = max(0.0, 1.0 - pow(distance / range, 4.0));

    return attenuation * range_factor;
}

// Spot light cone calculation with penumbra
fn calculate_spot_light_cone(
    light_dir: vec3<f32>,
    to_light: vec3<f32>,
    inner_cone: f32,
    outer_cone: f32,
    penumbra_softness: f32
) -> f32 {
    let spot_dir = normalize(-light_dir);
    let light_direction = normalize(to_light);
    let dot_product = dot(spot_dir, light_direction);

    let inner_cos = cos(inner_cone * 0.5);
    let outer_cos = cos(outer_cone * 0.5);

    if dot_product < outer_cos {
        return 0.0; // Outside outer cone
    }

    if dot_product > inner_cos {
        return 1.0; // Inside inner cone (full intensity)
    }

    // Penumbra region - smooth transition
    let cone_factor = (dot_product - outer_cos) / (inner_cos - outer_cos);

    // Apply penumbra softness for smooth falloff
    return smoothstep(0.0, 1.0, pow(cone_factor, 1.0 / (penumbra_softness + 0.1)));
}

// Shadow mapping functions
fn sample_shadow_map(shadow_coord: vec3<f32>, shadow_map: texture_2d<f32>, shadow_sampler: sampler) -> f32 {
    if shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
       shadow_coord.y < 0.0 || shadow_coord.y > 1.0 {
        return 1.0; // No shadow outside shadow map
    }

    let depth = textureSample(shadow_map, shadow_sampler, shadow_coord.xy).r;
    return select(0.0, 1.0, shadow_coord.z <= depth + SHADOW_BIAS);
}

// PCF (Percentage Closer Filtering) for soft shadows
fn sample_shadow_pcf(
    shadow_coord: vec3<f32>,
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    softness: f32
) -> f32 {
    let texel_size = 1.0 / SHADOW_MAP_SIZE;
    let offset = texel_size * softness;

    var shadow_factor = 0.0;
    let samples = 4; // 4x4 PCF kernel

    for (var x: i32 = -samples; x <= samples; x = x + 1) {
        for (var y: i32 = -samples; y <= samples; y = y + 1) {
            let sample_coord = shadow_coord.xy + vec2<f32>(f32(x), f32(y)) * offset;
            let sample_shadow_coord = vec3<f32>(sample_coord, shadow_coord.z);
            shadow_factor += sample_shadow_map(sample_shadow_coord, shadow_map, shadow_sampler);
        }
    }

    let total_samples = f32((samples * 2 + 1) * (samples * 2 + 1));
    return shadow_factor / total_samples;
}

// Calculate light contribution from a single light
fn calculate_light_contribution(
    light: Light,
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    view_dir: vec3<f32>,
    shadow_coord: vec3<f32>,
    shadow_map: texture_2d<f32>,
    shadow_sampler: sampler,
    shadow_quality: u32
) -> vec3<f32> {
    let to_light = light.position - world_pos;
    let distance = length(to_light);
    let light_dir = normalize(to_light);

    // Calculate attenuation
    var attenuation = calculate_point_light_attenuation(distance, light.range);

    if attenuation <= 0.0 {
        return vec3<f32>(0.0); // Outside light range
    }

    // For spot lights, apply cone calculation
    if light.light_type == LIGHT_TYPE_SPOT {
        let cone_factor = calculate_spot_light_cone(
            light.direction,
            to_light,
            light.inner_cone_angle,
            light.outer_cone_angle,
            light.penumbra_softness
        );
        attenuation *= cone_factor;

        if attenuation <= 0.0 {
            return vec3<f32>(0.0); // Outside spot cone
        }
    }

    // Calculate diffuse lighting (Lambertian)
    let n_dot_l = max(0.0, dot(world_normal, light_dir));
    var diffuse = n_dot_l;

    // Calculate specular lighting (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(0.0, dot(world_normal, half_dir));
    let specular = pow(n_dot_h, 32.0) * 0.3; // Moderate specular

    // Apply shadow mapping if enabled
    var shadow_factor = 1.0;
    if light.shadow_enabled > 0.5 && shadow_quality > 0u {
        if shadow_quality >= 2u {
            shadow_factor = sample_shadow_pcf(shadow_coord, shadow_map, shadow_sampler, 0.5);
        } else {
            shadow_factor = sample_shadow_map(shadow_coord, shadow_map, shadow_sampler);
        }
    }

    // Combine lighting components
    let lighting = (diffuse + specular) * attenuation * shadow_factor;
    return light.color * light.intensity * lighting;
}

// Bind groups and resources
@group(0) @binding(0) var<uniform> uniforms: PointSpotLightUniforms;
@group(0) @binding(1) var<storage, read> lights: LightArray;
@group(0) @binding(2) var g_buffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var g_buffer_normal: texture_2d<f32>;
@group(0) @binding(4) var g_buffer_depth: texture_2d<f32>;
@group(0) @binding(5) var g_buffer_sampler: sampler;
@group(1) @binding(0) var shadow_map_array: texture_2d_array<f32>;
@group(1) @binding(1) var shadow_sampler: sampler;

// Fragment shader - deferred lighting pass
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample G-buffer
    let albedo = textureSample(g_buffer_albedo, g_buffer_sampler, in.uv);
    let normal_sample = textureSample(g_buffer_normal, g_buffer_sampler, in.uv);
    let depth = textureSample(g_buffer_depth, g_buffer_sampler, in.uv).r;

    // Reconstruct world normal
    let world_normal = normalize(normal_sample.xyz * 2.0 - 1.0);

    // Reconstruct world position from depth
    let ndc = vec3<f32>(in.uv * 2.0 - 1.0, depth);
    let view_pos = uniforms.proj_matrix * vec4<f32>(ndc, 1.0);
    let world_pos = (uniforms.view_matrix * vec4<f32>(view_pos.xyz / view_pos.w, 1.0)).xyz;

    // Calculate view direction
    let camera_pos = (uniforms.view_matrix * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let view_dir = normalize(camera_pos - world_pos);

    // Start with ambient lighting
    var final_color = uniforms.ambient_color * uniforms.ambient_intensity * albedo.rgb;

    // Process each active light
    for (var i: u32 = 0u; i < uniforms.active_light_count; i = i + 1u) {
        let light = lights.lights[i];

        // Calculate shadow coordinates (simplified - would need proper light matrices)
        var shadow_coord = vec3<f32>(0.5, 0.5, 0.5); // Placeholder

        // Calculate light contribution
        let light_contribution = calculate_light_contribution(
            light,
            world_pos,
            world_normal,
            view_dir,
            shadow_coord,
            shadow_map_array, // Would need to select correct layer
            shadow_sampler,
            uniforms.shadow_quality
        );

        final_color += light_contribution * albedo.rgb;
    }

    // Debug visualization modes
    if uniforms.debug_mode == 1u {
        // Show light bounds - visualize which lights affect this pixel
        var light_count = 0u;
        for (var i: u32 = 0u; i < uniforms.active_light_count; i = i + 1u) {
            let light = lights.lights[i];
            let distance = length(light.position - world_pos);
            if distance < light.range {
                light_count = light_count + 1u;
            }
        }

        let heat_color = vec3<f32>(
            f32(light_count) / 8.0,
            max(0.0, 1.0 - f32(light_count) / 4.0),
            max(0.0, 1.0 - f32(light_count) / 2.0)
        );
        final_color = mix(final_color, heat_color, 0.5);
    }

    return vec4<f32>(final_color, albedo.a);
}

// Forward rendering version for immediate mode
struct ForwardVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct ForwardVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_forward(input: ForwardVertexInput) -> ForwardVertexOutput {
    var out: ForwardVertexOutput;

    let world_pos = vec4<f32>(input.position, 1.0);
    out.world_pos = world_pos.xyz;
    out.world_normal = input.normal;
    out.uv = input.uv;

    let view_pos = uniforms.view_matrix * world_pos;
    out.position = uniforms.proj_matrix * view_pos;

    return out;
}

@fragment
fn fs_forward(in: ForwardVertexOutput) -> @location(0) vec4<f32> {
    let world_normal = normalize(in.world_normal);
    let camera_pos = (uniforms.view_matrix * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let view_dir = normalize(camera_pos - in.world_pos);

    // Base material color (could be textured)
    let albedo = vec3<f32>(0.8, 0.8, 0.8);

    // Start with ambient
    var final_color = uniforms.ambient_color * uniforms.ambient_intensity * albedo;

    // Process each light
    for (var i: u32 = 0u; i < uniforms.active_light_count; i = i + 1u) {
        let light = lights.lights[i];

        // Simplified shadow coord (would need proper calculation)
        var shadow_coord = vec3<f32>(0.5, 0.5, 0.5);

        let light_contribution = calculate_light_contribution(
            light,
            in.world_pos,
            world_normal,
            view_dir,
            shadow_coord,
            shadow_map_array,
            shadow_sampler,
            uniforms.shadow_quality
        );

        final_color += light_contribution * albedo;
    }

    return vec4<f32>(final_color, 1.0);
}

// Shadow map generation vertex shader
struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
}

struct ShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
}

@group(2) @binding(0) var<uniform> light_view_proj: mat4x4<f32>;

@vertex
fn vs_shadow(input: ShadowVertexInput) -> ShadowVertexOutput {
    var out: ShadowVertexOutput;
    out.position = light_view_proj * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_shadow() -> @location(0) f32 {
    // Depth is automatically written to depth buffer
    return 0.0;
}