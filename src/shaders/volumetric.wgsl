// Volumetric fog and god-rays (P6)
// Implements exponential height fog, Henyey-Greenstein scattering, and volumetric shadows

struct VolumetricParams {
    density: f32,                  // Base fog density [0.001-1.0]
    height_falloff: f32,           // Exponential height falloff [0-1]
    phase_g: f32,                  // Henyey-Greenstein asymmetry [-1 to 1, 0=isotropic, 0.7=forward]
    max_steps: u32,                // Ray marching steps [16-128]

    start_distance: f32,           // Near plane for fog [0.1-10]
    max_distance: f32,             // Far plane for fog [10-1000]
    scattering_color: vec3<f32>,   // Fog tint color
    absorption: f32,               // Absorption coefficient [0-1]

    sun_direction: vec3<f32>,      // Sun direction for in-scattering
    sun_intensity: f32,            // Sun light intensity
    ambient_color: vec3<f32>,      // Ambient sky contribution
    temporal_alpha: f32,           // Temporal reprojection blend [0-0.9]

    use_shadows: u32,              // 1=sample shadow map for god-rays
    jitter_strength: f32,          // Ray march jitter [0-1]
    frame_index: u32,              // Frame index for temporal jitter sequencing
    _pad0: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
    near: f32,
    far: f32,
    _pad: vec3<f32>,
}

const PI: f32 = 3.14159265359;

@group(0) @binding(0) var<uniform> params: VolumetricParams;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_sampler: sampler;

// Shadow map for god-rays
@group(1) @binding(0) var shadow_map: texture_depth_2d;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;
@group(1) @binding(2) var<uniform> shadow_matrix: mat4x4<f32>;

// Output and history
@group(2) @binding(0) var output_fog: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var history_fog: texture_2d<f32>;
@group(2) @binding(2) var history_sampler: sampler;

// ============================================================================
// Utility functions
// ============================================================================

// Blue noise / interleaved gradient noise for jittering
fn interleaved_gradient_noise(pixel: vec2<u32>, frame: u32) -> f32 {
    let px = vec2<f32>(f32(pixel.x), f32(pixel.y)) + vec2<f32>(f32(frame % 64u));
    return fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
}

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
    let clip_pos = vec4<f32>(ndc, 1.0);
    let world_pos = camera.inv_view * camera.inv_proj * clip_pos;
    return world_pos.xyz / world_pos.w;
}

// ============================================================================
// Henyey-Greenstein phase function
// ============================================================================

fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

// ============================================================================
// Exponential height fog density
// ============================================================================

fn fog_density_at_height(world_pos: vec3<f32>) -> f32 {
    let height = world_pos.y;

    // Exponential height falloff (denser near ground)
    let height_factor = exp(-max(height, 0.0) * params.height_falloff);

    return params.density * height_factor;
}

// ============================================================================
// Shadow sampling for god-rays
// ============================================================================

fn sample_shadow(world_pos: vec3<f32>) -> f32 {
    if (params.use_shadows == 0u) {
        return 1.0;
    }

    // Transform to shadow map space
    let shadow_pos = shadow_matrix * vec4<f32>(world_pos, 1.0);
    let shadow_ndc = shadow_pos.xyz / shadow_pos.w;

    // Convert to shadow map UV
    let shadow_uv = vec2<f32>(shadow_ndc.x * 0.5 + 0.5, 1.0 - (shadow_ndc.y * 0.5 + 0.5));

    // Out of shadow map bounds = fully lit
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;
    }

    // PCF shadow sampling for smoother god-rays
    let shadow_depth = shadow_ndc.z;
    var shadow_sum = 0.0;
    let pcf_radius = 2;
    var sample_count = 0.0;

    let texel_size = 1.0 / vec2<f32>(textureDimensions(shadow_map));

    for (var y = -pcf_radius; y <= pcf_radius; y = y + 1) {
        for (var x = -pcf_radius; x <= pcf_radius; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow_sum = shadow_sum + textureSampleCompare(
                shadow_map,
                shadow_sampler,
                shadow_uv + offset,
                shadow_depth - 0.001  // Small bias
            );
            sample_count = sample_count + 1.0;
        }
    }

    return shadow_sum / sample_count;
}

// ============================================================================
// In-scattering calculation
// ============================================================================

fn calculate_inscattering(
    world_pos: vec3<f32>,
    view_dir: vec3<f32>,
    shadow_factor: f32
) -> vec3<f32> {
    // Phase function for directional scattering
    let cos_theta = dot(view_dir, params.sun_direction);
    let phase = henyey_greenstein_phase(cos_theta, params.phase_g);

    // Sun in-scattering
    let sun_scatter = params.scattering_color * phase * params.sun_intensity * shadow_factor;

    // Ambient sky in-scattering (omnidirectional)
    let ambient_scatter = params.ambient_color * 0.5;

    return sun_scatter + ambient_scatter;
}

// ============================================================================
// Ray marching volumetric fog
// ============================================================================

fn ray_march_fog(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    max_dist: f32,
    jitter: f32
) -> vec4<f32> {
    let step_count = f32(params.max_steps);
    let step_size = max_dist / step_count;

    var accumulated_fog = vec3<f32>(0.0);
    var accumulated_transmittance = 1.0;

    // Jittered starting point for temporal stability
    let jitter_offset = jitter * step_size * params.jitter_strength;
    var current_dist = params.start_distance + jitter_offset;

    for (var i = 0u; i < params.max_steps; i = i + 1u) {
        if (current_dist >= max_dist) {
            break;
        }

        let world_pos = ray_origin + ray_dir * current_dist;

        // Sample fog density at this point
        let density = fog_density_at_height(world_pos);

        if (density > 0.0001) {
            // Sample shadow for god-rays
            let shadow = sample_shadow(world_pos);

            // Calculate in-scattered light
            let inscatter = calculate_inscattering(world_pos, ray_dir, shadow);

            // Beer-Lambert law for extinction
            let extinction = exp(-density * params.absorption * step_size);
            let scatter_integral = (1.0 - extinction) / max(density * params.absorption, 0.0001);

            // Accumulate scattering
            accumulated_fog = accumulated_fog +
                inscatter * scatter_integral * accumulated_transmittance * density;

            // Update transmittance
            accumulated_transmittance = accumulated_transmittance * extinction;

            // Early out if fully occluded
            if (accumulated_transmittance < 0.001) {
                break;
            }
        }

        current_dist = current_dist + step_size;
    }

    return vec4<f32>(accumulated_fog, 1.0 - accumulated_transmittance);
}

// ============================================================================
// Main compute shader
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_volumetric(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(output_fog);

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);

    // Sample scene depth
    let scene_depth = textureSample(depth_texture, depth_sampler, uv).r;

    // Reconstruct view ray
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let clip_pos = vec4<f32>(ndc, 1.0, 1.0);
    let view_pos = camera.inv_proj * clip_pos;
    let view_dir_vs = normalize(view_pos.xyz / view_pos.w);
    let view_dir_ws = normalize((camera.inv_view * vec4<f32>(view_dir_vs, 0.0)).xyz);

    // Calculate ray march distance
    // Stop at scene geometry or max fog distance
    let world_depth_pos = reconstruct_world_pos(uv, scene_depth);
    let scene_distance = length(world_depth_pos - camera.eye_position);
    let max_march_dist = min(scene_distance, params.max_distance);

    // Skip if too close
    if (max_march_dist <= params.start_distance) {
        textureStore(output_fog, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Jitter for temporal stability
    let jitter = interleaved_gradient_noise(pixel, params.frame_index);

    // Ray march
    let fog_result = ray_march_fog(camera.eye_position, view_dir_ws, max_march_dist, jitter);

    // Temporal reprojection for stability
    var final_fog = fog_result;
    if (params.temporal_alpha > 0.0) {
        let history = textureSample(history_fog, history_sampler, uv);
        final_fog = mix(fog_result, history, params.temporal_alpha);
    }

    textureStore(output_fog, pixel, final_fog);
}

// ============================================================================
// Froxel-based volumetric fog (alternative, more efficient approach)
// ============================================================================

struct Froxel {
    scattering: vec3<f32>,
    extinction: f32,
}

// Froxel grid dimensions
const FROXEL_GRID_X: u32 = 16u;
const FROXEL_GRID_Y: u32 = 8u;
const FROXEL_GRID_Z: u32 = 64u;

@group(3) @binding(0) var froxel_buffer: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn cs_build_froxels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= FROXEL_GRID_X ||
        global_id.y >= FROXEL_GRID_Y ||
        global_id.z >= FROXEL_GRID_Z) {
        return;
    }

    // Compute froxel world position
    let froxel_size = vec3<f32>(
        2.0 / f32(FROXEL_GRID_X),
        2.0 / f32(FROXEL_GRID_Y),
        (params.max_distance - params.start_distance) / f32(FROXEL_GRID_Z)
    );

    let froxel_ndc = vec3<f32>(
        (f32(global_id.x) + 0.5) / f32(FROXEL_GRID_X) * 2.0 - 1.0,
        (f32(global_id.y) + 0.5) / f32(FROXEL_GRID_Y) * 2.0 - 1.0,
        0.0  // Will compute depth separately
    );

    // Compute froxel depth (exponential distribution for better near-field detail)
    let z_linear = params.start_distance +
        (f32(global_id.z) + 0.5) / f32(FROXEL_GRID_Z) *
        (params.max_distance - params.start_distance);

    // Reconstruct world position for this froxel center
    let view_ray = normalize(vec3<f32>(froxel_ndc.xy, 1.0));
    let world_pos = camera.eye_position + view_ray * z_linear;

    // Calculate density
    let density = fog_density_at_height(world_pos);

    // Sample shadow
    let shadow = sample_shadow(world_pos);

    // Calculate in-scattering
    let inscatter = calculate_inscattering(world_pos, view_ray, shadow);

    // Store froxel data
    let extinction = density * params.absorption;
    textureStore(froxel_buffer, global_id, vec4<f32>(inscatter * density, extinction));
}

// ============================================================================
// Froxel sampling for rendering
// ============================================================================

@group(3) @binding(1) var froxel_texture: texture_3d<f32>;
@group(3) @binding(2) var froxel_sampler: sampler;

@compute @workgroup_size(8, 8, 1)
fn cs_apply_froxels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(output_fog);

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);

    // Sample scene depth
    let scene_depth = textureSample(depth_texture, depth_sampler, uv).r;
    let world_depth_pos = reconstruct_world_pos(uv, scene_depth);
    let scene_distance = length(world_depth_pos - camera.eye_position);

    // March through froxels along view ray
    var accumulated_fog = vec3<f32>(0.0);
    var accumulated_transmittance = 1.0;

    let froxel_march_steps = params.max_steps;
    let step_size = scene_distance / f32(froxel_march_steps);

    for (var i = 0u; i < froxel_march_steps; i = i + 1u) {
        let dist = f32(i) * step_size;

        // Map to froxel UVW coordinates
        let froxel_z = (dist - params.start_distance) / (params.max_distance - params.start_distance);
        if (froxel_z < 0.0 || froxel_z > 1.0) {
            continue;
        }

        let froxel_uvw = vec3<f32>(uv.x, uv.y, froxel_z);

        // Sample froxel
        let froxel_data = textureSample(froxel_texture, froxel_sampler, froxel_uvw);
        let inscatter = froxel_data.rgb;
        let extinction = froxel_data.a;

        // Integrate
        let transmittance_step = exp(-extinction * step_size);
        accumulated_fog = accumulated_fog + inscatter * accumulated_transmittance * step_size;
        accumulated_transmittance = accumulated_transmittance * transmittance_step;
    }

    textureStore(output_fog, pixel, vec4<f32>(accumulated_fog, 1.0 - accumulated_transmittance));
}
