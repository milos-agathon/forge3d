// Missing HDR composition bindings; AO/GI/SSR themselves remain authoritative
// in their existing modules. No material-visualization output is used here.
struct ConcordiaComposeUniforms {
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    terrain_scale_bias: vec4<f32>,
    weights: vec4<f32>,
}
@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(1) var ao: texture_2d<f32>;
@group(0) @binding(2) var height_ao: texture_2d<f32>;
@group(0) @binding(3) var gi: texture_2d<f32>;
@group(0) @binding(4) var reflection: texture_2d<f32>;
@group(0) @binding(5) var linear_depth: texture_2d<f32>;
@group(0) @binding(6) var<uniform> frame: ConcordiaComposeUniforms;
@group(0) @binding(7) var output_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var surface_kind: texture_2d<u32>;

fn compose_world_position(pixel: vec2<u32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(textureDimensions(linear_depth));
    let ndc = uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let near_h = frame.inv_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;
    let view_point = mix(near, far, (depth + near.z) / (near.z - far.z));
    return (frame.inv_view * vec4<f32>(view_point, 1.0)).xyz;
}

@compute @workgroup_size(8, 8, 1)
fn combine_ao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(input_color);
    if (any(pixel >= dims)) { return; }
    let color = textureLoad(input_color, pixel, 0);
    let depth = textureLoad(linear_depth, pixel, 0).r;
    var visibility = 1.0;
    if (depth > 0.0) {
        let gtao = textureLoad(ao, min(pixel, textureDimensions(ao) - 1u), 0).r;
        let world = compose_world_position(pixel, depth);
        let uv = clamp(world.xy * frame.terrain_scale_bias.xy + frame.terrain_scale_bias.zw,
            vec2<f32>(0.0), vec2<f32>(1.0));
        let height_size = textureDimensions(height_ao);
        let height_pixel = min(vec2<u32>(uv * vec2<f32>(height_size)), height_size - 1u);
        let horizon = select(1.0, textureLoad(height_ao, height_pixel, 0).r,
            textureLoad(surface_kind,pixel,0).r == 1u);
        // Bounded visibility approximation: the kernels own different geometry,
        // and independent visibility intervals compose by a product. Correlated
        // solid angles can overlap; the mixed-scene PT ablation gate measures
        // that approximation. The terrain term is never evaluated twice.
        visibility = clamp(gtao, 0.0, 1.0) * clamp(horizon, 0.0, 1.0);
    }
    textureStore(output_color, pixel, vec4<f32>(color.rgb * visibility, color.a));
}

@compute @workgroup_size(8, 8, 1)
fn add_gi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    if (any(pixel >= textureDimensions(input_color))) { return; }
    let color = textureLoad(input_color, pixel, 0);
    let bounce = textureLoad(gi, pixel, 0).rgb;
    textureStore(output_color, pixel, vec4<f32>(color.rgb + bounce * frame.weights.y, color.a));
}

@compute @workgroup_size(8, 8, 1)
fn add_reflections(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    if (any(pixel >= textureDimensions(input_color))) { return; }
    let color = textureLoad(input_color, pixel, 0);
    let specular = textureLoad(reflection, pixel, 0);
    // An SSR miss already retains the terrain/building IBL/probe contribution.
    // Only real hits add screen-space radiance; misses never blacken the frame.
    let hit_weight = select(clamp(specular.a, 0.0, 1.0), 1.0,
        specular.a <= 0.0 && textureLoad(surface_kind,pixel,0).r == 2u) * frame.weights.z;
    textureStore(output_color, pixel, vec4<f32>(color.rgb + specular.rgb * hit_weight, color.a));
}

@compute @workgroup_size(8, 8, 1)
fn copy_color(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= textureDimensions(input_color))) { return; }
    textureStore(output_color, gid.xy, textureLoad(input_color, gid.xy, 0));
}

@compute @workgroup_size(8,8,1)
fn composite_fog(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= textureDimensions(input_color))) { return; }
    let color = textureLoad(input_color,gid.xy,0);
    let fog = textureLoad(gi,gid.xy,0);
    // The existing froxel apply pass publishes RGB inscatter and alpha opacity.
    textureStore(output_color,gid.xy,vec4<f32>(color.rgb*(1.0-clamp(fog.a,0.0,1.0))+fog.rgb,color.a));
}
