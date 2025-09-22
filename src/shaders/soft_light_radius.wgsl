// B12: Soft Light Radius (Raster) - Light falloff with configurable radius control
// Implements soft vs hard light boundaries for raster lighting systems

struct SoftLightRadiusUniforms {
    // Light position and intensity
    light_position: vec3<f32>,
    light_intensity: f32,

    // Radius parameters
    inner_radius: f32,          // Distance where light starts to falloff
    outer_radius: f32,          // Distance where light reaches zero
    falloff_exponent: f32,      // Controls falloff curve steepness (1.0=linear, 2.0=quadratic)
    edge_softness: f32,         // Additional softening factor for edges

    // Color and control
    light_color: vec3<f32>,
    enabled: f32,               // 0.0=disabled, 1.0=enabled

    // Quality and modes
    falloff_mode: u32,          // 0=linear, 1=quadratic, 2=cubic, 3=exponential
    shadow_softness: f32,       // Softness for shadow edges
    _pad0: f32,
    _pad1: f32,
}

// Vertex shader for full-screen quad
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
}

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

    return out;
}

// Distance-based light falloff functions
fn linear_falloff(distance: f32, inner_radius: f32, outer_radius: f32) -> f32 {
    if distance <= inner_radius {
        return 1.0;
    }
    if distance >= outer_radius {
        return 0.0;
    }
    return 1.0 - (distance - inner_radius) / (outer_radius - inner_radius);
}

fn quadratic_falloff(distance: f32, inner_radius: f32, outer_radius: f32, exponent: f32) -> f32 {
    if distance <= inner_radius {
        return 1.0;
    }
    if distance >= outer_radius {
        return 0.0;
    }
    let t = (distance - inner_radius) / (outer_radius - inner_radius);
    return pow(1.0 - t, exponent);
}

fn cubic_falloff(distance: f32, inner_radius: f32, outer_radius: f32) -> f32 {
    if distance <= inner_radius {
        return 1.0;
    }
    if distance >= outer_radius {
        return 0.0;
    }
    let t = (distance - inner_radius) / (outer_radius - inner_radius);
    return 1.0 - t * t * t;
}

fn exponential_falloff(distance: f32, inner_radius: f32, outer_radius: f32, falloff_rate: f32) -> f32 {
    if distance <= inner_radius {
        return 1.0;
    }
    if distance >= outer_radius {
        return 0.0;
    }
    let t = (distance - inner_radius) / (outer_radius - inner_radius);
    return exp(-falloff_rate * t);
}

// Smooth step function for edge softening
fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Calculate soft light contribution at a world position
fn calculate_soft_light(
    world_pos: vec3<f32>,
    uniforms: SoftLightRadiusUniforms
) -> vec3<f32> {
    if uniforms.enabled < 0.5 {
        return vec3<f32>(0.0);
    }

    // Calculate distance from light
    let light_vector = world_pos - uniforms.light_position;
    let distance = length(light_vector);

    // Early exit if beyond outer radius
    if distance >= uniforms.outer_radius {
        return vec3<f32>(0.0);
    }

    // Calculate base falloff
    var falloff: f32;
    switch uniforms.falloff_mode {
        case 0u: {
            falloff = linear_falloff(distance, uniforms.inner_radius, uniforms.outer_radius);
        }
        case 1u: {
            falloff = quadratic_falloff(distance, uniforms.inner_radius, uniforms.outer_radius, uniforms.falloff_exponent);
        }
        case 2u: {
            falloff = cubic_falloff(distance, uniforms.inner_radius, uniforms.outer_radius);
        }
        case 3u: {
            falloff = exponential_falloff(distance, uniforms.inner_radius, uniforms.outer_radius, uniforms.falloff_exponent);
        }
        default: {
            falloff = linear_falloff(distance, uniforms.inner_radius, uniforms.outer_radius);
        }
    }

    // Apply edge softening
    if uniforms.edge_softness > 0.0 {
        let soft_inner = uniforms.inner_radius - uniforms.edge_softness;
        let soft_outer = uniforms.outer_radius + uniforms.edge_softness;
        let soft_factor = smooth_step(soft_outer, soft_inner, distance);
        falloff *= soft_factor;
    }

    // Apply intensity and color
    let final_intensity = falloff * uniforms.light_intensity;
    return uniforms.light_color * final_intensity;
}

// Fragment shader
@group(0) @binding(0) var<uniform> uniforms: SoftLightRadiusUniforms;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var depth_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample depth to reconstruct world position
    let depth = textureSample(depth_texture, depth_sampler, in.uv).r;

    // Convert screen space to world space (simplified)
    // In a real implementation, this would use inverse view-projection matrices
    let world_pos = vec3<f32>(
        (in.uv.x - 0.5) * 2.0 * 100.0,  // Scale for world space
        (in.uv.y - 0.5) * 2.0 * 100.0,
        depth * 100.0 - 50.0
    );

    // Calculate soft light contribution
    let light_contribution = calculate_soft_light(world_pos, uniforms);

    // Output as additive blend
    return vec4<f32>(light_contribution, 1.0);
}

// Multiple light sources support
@group(1) @binding(0) var<storage, read> light_array: array<SoftLightRadiusUniforms>;

@fragment
fn fs_multiple_lights(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample depth to reconstruct world position
    let depth = textureSample(depth_texture, depth_sampler, in.uv).r;

    let world_pos = vec3<f32>(
        (in.uv.x - 0.5) * 2.0 * 100.0,
        (in.uv.y - 0.5) * 2.0 * 100.0,
        depth * 100.0 - 50.0
    );

    var total_light = vec3<f32>(0.0);

    // Accumulate contributions from all lights
    for (var i: u32 = 0u; i < arrayLength(&light_array); i = i + 1u) {
        total_light += calculate_soft_light(world_pos, light_array[i]);
    }

    return vec4<f32>(total_light, 1.0);
}

// Shadow-aware version with soft shadows
@group(2) @binding(0) var shadow_map: texture_2d<f32>;
@group(2) @binding(1) var shadow_sampler: sampler;

fn sample_soft_shadow(world_pos: vec3<f32>, light_pos: vec3<f32>, softness: f32) -> f32 {
    // Simplified soft shadow sampling
    // In practice, this would use proper shadow mapping with PCF
    let shadow_coord = (world_pos - light_pos) * 0.01 + 0.5;

    if shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
       shadow_coord.y < 0.0 || shadow_coord.y > 1.0 {
        return 1.0;  // No shadow outside shadow map
    }

    // Sample shadow map with soft filtering
    var shadow_factor = 0.0;
    let samples = 4u;
    let offset = softness / f32(samples);

    for (var x: u32 = 0u; x < samples; x = x + 1u) {
        for (var y: u32 = 0u; y < samples; y = y + 1u) {
            let sample_coord = shadow_coord.xy + vec2<f32>(
                (f32(x) - f32(samples) * 0.5) * offset,
                (f32(y) - f32(samples) * 0.5) * offset
            );
            let shadow_depth = textureSample(shadow_map, shadow_sampler, sample_coord).r;
            if shadow_depth > shadow_coord.z - 0.001 {
                shadow_factor += 1.0;
            }
        }
    }

    return shadow_factor / f32(samples * samples);
}

@fragment
fn fs_soft_shadows(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_texture, depth_sampler, in.uv).r;

    let world_pos = vec3<f32>(
        (in.uv.x - 0.5) * 2.0 * 100.0,
        (in.uv.y - 0.5) * 2.0 * 100.0,
        depth * 100.0 - 50.0
    );

    // Calculate base light contribution
    let light_contribution = calculate_soft_light(world_pos, uniforms);

    // Apply soft shadows
    let shadow_factor = sample_soft_shadow(world_pos, uniforms.light_position, uniforms.shadow_softness);

    return vec4<f32>(light_contribution * shadow_factor, 1.0);
}