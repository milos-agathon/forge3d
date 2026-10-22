// Appended to the existing terrain or scatter module. Group 7 in terrain,
// group 3 in scatter. Both consume the same frame uniform and cloud field.
struct ConcordiaFrameUniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    cloud_scale_bias: vec4<f32>,
    effects: vec4<u32>,
    geometry: vec4<f32>,
}
@group(7) @binding(0) var<uniform> concordia_frame: ConcordiaFrameUniforms;
@group(7) @binding(1) var concordia_cloud_field: texture_2d<f32>;
@group(7) @binding(2) var concordia_cloud_sampler: sampler;

struct ConcordiaSurface {
    @location(0) color: vec4<f32>,
    @location(1) material: vec4<f32>,
    @location(2) normal_roughness: vec4<f32>,
    @location(3) depth: f32,
    @location(4) velocity: vec2<f32>,
    @location(5) surface_kind: u32,
}

fn concordia_cloud_visibility(world: vec3<f32>) -> f32 {
    if (concordia_frame.effects.x == 0u) { return 1.0; }
    let uv = world.xy * concordia_frame.cloud_scale_bias.xy
        + concordia_frame.cloud_scale_bias.zw;
    return clamp(textureSampleLevel(
        concordia_cloud_field, concordia_cloud_sampler, uv, 0.0
    ).r, 0.0, 1.0);
}

fn concordia_surface(
    world: vec3<f32>, normal: vec3<f32>, roughness: f32,
    albedo: vec3<f32>, metallic: f32, color: vec4<f32>, raster: vec4<f32>, kind: u32,
) -> ConcordiaSurface {
    // Terrain's public world_position retains source elevation; rasterized
    // geometry is height-centered. Recover the actual raster position once so
    // depth, AO, velocity and overlays agree for terrain and scatter alike.
    let uv = raster.xy / vec2<f32>(concordia_frame.effects.yz);
    let current_ndc3 = vec3<f32>(uv * vec2<f32>(2.0,-2.0) + vec2<f32>(-1.0,1.0), raster.z);
    let homogeneous = concordia_frame.inv_proj * vec4<f32>(current_ndc3, 1.0);
    let view_position = homogeneous / homogeneous.w;
    let raster_world = concordia_frame.inv_view * view_position;
    let previous_clip = concordia_frame.previous_view_proj * raster_world;
    let view_normal = normalize((concordia_frame.view * vec4<f32>(normal, 0.0)).xyz);
    let current_ndc = current_ndc3.xy - concordia_frame.geometry.zw
        / vec2<f32>(concordia_frame.effects.yz) * vec2<f32>(2.0,-2.0);
    // A receiver behind the previous camera has no valid history. NDC 3
    // denotes an off-screen previous sample, avoiding division by zero.
    var previous_ndc = vec2<f32>(3.0);
    if (previous_clip.w > 0.0) {
        previous_ndc = previous_clip.xy / previous_clip.w;
    }
    var out: ConcordiaSurface;
    out.color = color;
    out.surface_kind = kind;
    out.material = vec4<f32>(clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(metallic, 0.0, 1.0));
    out.normal_roughness = vec4<f32>(view_normal * 0.5 + 0.5, clamp(roughness, 0.0, 1.0));
    out.depth = max(-view_position.z, 0.0);
    // GBuffer/TAA contract: current minus previous UV; positive Y points down.
    // Motion exceeding a whole viewport already rejects history. Bounding
    // it to that UV domain keeps RG16F finite across camera-plane crossings.
    out.velocity = clamp((current_ndc - previous_ndc) * vec2<f32>(0.5, -0.5),
        vec2<f32>(-1.0), vec2<f32>(1.0));
    return out;
}
