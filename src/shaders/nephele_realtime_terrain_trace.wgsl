// Batched realtime terrain queries for NEPHELE. The production PROMETHEUS
// terrain_trace implementation is assembled before this adapter.
struct NepheleTraceUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera: vec4<f32>,
    sun: vec4<f32>,
    sun_radiance: vec4<f32>,
    sigma_s: vec4<f32>,
    sigma_t: vec4<f32>,
    depth: vec4<f32>,
    grid: vec4<u32>,
    viewport: vec4<u32>,
    scattering: vec4<f32>,
    terrain_albedo: vec4<f32>,
    diffuse_ibl: vec4<f32>,
    world_min: vec4<f32>,
    world_max: vec4<f32>,
}

struct NepheleTerrainHit {
    point_t: vec4<f32>,
    normal_hit: vec4<f32>,
}

@group(3) @binding(8) var<uniform> nephele: NepheleTraceUniforms;
@group(3) @binding(9) var nephele_blue_noise: texture_2d<u32>;
@group(3) @binding(10) var nephele_extinction: texture_3d<f32>;
@group(3) @binding(11) var<storage, read_write> nephele_sun_hits: array<NepheleTerrainHit>;
@group(3) @binding(12) var<storage, read_write> nephele_phase_hits: array<NepheleTerrainHit>;
@group(3) @binding(13) var nephele_world_medium: texture_3d<f32>;
@group(3) @binding(14) var nephele_world_sun_out: texture_storage_3d<rgba16float, write>;
@group(3) @binding(15) var nephele_world_sun: texture_3d<f32>;
@group(3) @binding(16) var nephele_surface_cloudlight: texture_storage_2d<rgba16float, write>;

fn nephele_froxel_distance(u: f32) -> f32 {
    return nephele.depth.x * exp(nephele.depth.z * clamp(u, 0.0, 1.0));
}

fn nephele_froxel_world(id: vec3<u32>) -> vec3<f32> {
    let visible = nephele.grid.xy - vec2<u32>(2u);
    let uv = (vec2<f32>(id.xy) + 0.5 - vec2<f32>(1.0)) / vec2<f32>(visible);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far_h = nephele.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let direction = normalize(far_h.xyz / far_h.w - nephele.camera.xyz);
    return nephele.camera.xyz + direction * nephele_froxel_distance(
        (f32(id.z) + 0.5) / f32(nephele.grid.z)
    );
}

fn nephele_rank(id: vec3<u32>, salt: u32) -> f32 {
    let dimensions = vec2<u32>(textureDimensions(nephele_blue_noise));
    let p = (id.xy + vec2<u32>(id.z * 3u + salt, id.z * 5u + salt * 7u)) % dimensions;
    let rank = textureLoad(nephele_blue_noise, vec2<i32>(p), 0).r;
    return (f32(rank) + 0.5) / 64.0;
}

fn nephele_phase_direction(incident: vec3<f32>, id: vec3<u32>) -> vec3<f32> {
    let u1 = nephele_rank(id, 0u);
    let u2 = nephele_rank(id, 1u);
    var cos_theta = 1.0 - 2.0 * u1;
    if (nephele.scattering.w >= 0.5) {
        let g = nephele.scattering.z;
        if (abs(g) > 1e-4) {
            let ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
            cos_theta = clamp((1.0 + g * g - ratio * ratio) / (2.0 * g), -1.0, 1.0);
        }
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 6.283185307179586 * u2;
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(incident.y) < 0.999);
    let tangent = normalize(cross(helper, incident));
    let bitangent = cross(incident, tangent);
    return normalize(tangent * (cos(phi) * sin_theta) + bitangent * (sin(phi) * sin_theta) + incident * cos_theta);
}

fn nephele_store_hit(index: u32, sun: bool, hit: HybridHitResult) {
    let encoded = NepheleTerrainHit(vec4<f32>(hit.point, hit.t), vec4<f32>(hit.normal, f32(hit.hit)));
    if (sun) {
        nephele_sun_hits[index] = encoded;
    } else {
        nephele_phase_hits[index] = encoded;
    }
}

@compute @workgroup_size(4, 4, 4)
fn main_nephele_realtime_terrain_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= nephele.grid.xyz)) {
        return;
    }
    let index = id.x + nephele.grid.x * (id.y + nephele.grid.y * id.z);
    let canonical = textureLoad(nephele_extinction, vec3<i32>(id), 0);
    if (!(canonical.a > 0.0)) {
        var miss: HybridHitResult;
        miss.t = nephele.depth.y;
        miss.point = vec3<f32>(0.0);
        miss.normal = vec3<f32>(0.0);
        miss.material_id = 0u;
        miss.hit_type = 3u;
        miss.hit = 0u;
        miss._pad = vec2<u32>(0u);
        nephele_store_hit(index, true, miss);
        nephele_store_hit(index, false, miss);
        return;
    }
    let world = nephele_froxel_world(id);
    let incident = normalize(world - nephele.camera.xyz);
    // These are two distinct physical terms, not two quality samples.
    let sun_hit = terrain_trace(Ray(world, 1e-4, normalize(nephele.sun.xyz), nephele.depth.y), false, true);
    let continuation = nephele_phase_direction(incident, id);
    let phase_hit = terrain_trace(Ray(world, 1e-4, continuation, nephele.depth.y), false, true);
    nephele_store_hit(index, true, sun_hit);
    nephele_store_hit(index, false, phase_hit);
}

// Direction-independent sunlight source over the full canonical medium.
// Terrain visibility uses the same production bilinear traversal as froxels.
@compute @workgroup_size(4, 4, 4)
fn main_nephele_world_sun(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(nephele_world_medium);
    if (any(id >= dimensions)) { return; }
    let canonical = textureLoad(nephele_world_medium, vec3<i32>(id), 0);
    if (!(canonical.a > 0.0) || nephele.world_min.w < 0.5) {
        textureStore(nephele_world_sun_out, id, vec4<f32>(0.0));
        return;
    }
    let world = nephele.world_min.xyz + (nephele.world_max.xyz - nephele.world_min.xyz)
        * (vec3<f32>(id) + 0.5) / vec3<f32>(dimensions);
    let direction = normalize(nephele.sun.xyz);
    // Terrain traversal clips this finite-f32 ray against its own bounds;
    // a blocker beyond the medium box can still shadow the medium.
    let hit = terrain_trace(Ray(world, 1e-4, direction, 3.402823466e38), false, true);
    let visible = nephele.scattering.x < 0.5 || hit.hit == 0u;
    let source = nephele.sigma_s.rgb * canonical.a * canonical.rgb
        * nephele.sun_radiance.rgb * nephele.sun_radiance.a * f32(visible);
    textureStore(nephele_world_sun_out, id, vec4<f32>(source, canonical.a));
}

fn nephele_cloudlight_sample(world: vec3<f32>) -> vec4<f32> {
    if (any(world < nephele.world_min.xyz) || any(world > nephele.world_max.xyz)) {
        return vec4<f32>(0.0);
    }
    let dimensions = vec3<i32>(textureDimensions(nephele_world_sun));
    let unit = (world - nephele.world_min.xyz) / (nephele.world_max.xyz - nephele.world_min.xyz);
    let coordinate = unit * vec3<f32>(dimensions) - 0.5;
    let base = vec3<i32>(floor(coordinate));
    let f = fract(coordinate);
    var result = vec4<f32>(0.0);
    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let offset = vec3<i32>(x, y, z);
                let weight = mix(vec3<f32>(1.0) - f, f, vec3<f32>(offset));
                let c = clamp(base + offset, vec3<i32>(0), dimensions - 1);
                result += textureLoad(nephele_world_sun, c, 0) * weight.x * weight.y * weight.z;
            }
        }
    }
    return result;
}

fn nephele_medium_interval(origin: vec3<f32>, direction: vec3<f32>) -> vec2<f32> {
    var begin = 0.0;
    var end = length(nephele.world_max.xyz - nephele.world_min.xyz)
        + length(origin - nephele.world_min.xyz);
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        if (direction[axis] == 0.0) {
            if (origin[axis] < nephele.world_min[axis] || origin[axis] > nephele.world_max[axis]) {
                return vec2<f32>(0.0);
            }
        } else {
            let a = (nephele.world_min[axis] - origin[axis]) / direction[axis];
            let b = (nephele.world_max[axis] - origin[axis]) / direction[axis];
            begin = max(begin, min(a, b));
            end = min(end, max(a, b));
        }
    }
    return vec2<f32>(begin, max(begin, end));
}

fn nephele_cloudlight_phase(cosine: f32) -> f32 {
    if (nephele.scattering.w < 0.5) { return 0.07957747154594767; }
    let g = nephele.scattering.z;
    let d = max(1.0 + g*g - 2.0*g*clamp(cosine, -1.0, 1.0), 1e-12);
    return (1.0 - g*g) / (12.566370614359172 * d * sqrt(d));
}

// The 192-direction Fibonacci quadrature is the owner-selected round-2
// surface_cloudlight.py term. No gain is fitted to the frozen beauty image.
@compute @workgroup_size(8, 8, 1)
fn main_nephele_surface_cloudlight(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= nephele.viewport.xy)) { return; }
    if (nephele.world_min.w < 0.5) {
        textureStore(nephele_surface_cloudlight, id.xy, vec4<f32>(0.0));
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(nephele.viewport.xy);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far_h = nephele.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ray = normalize(far_h.xyz / far_h.w - nephele.camera.xyz);
    let surface_hit = terrain_trace(Ray(nephele.camera.xyz, 1e-4, ray, nephele.depth.y), false, true);
    if (surface_hit.hit == 0u) {
        textureStore(nephele_surface_cloudlight, id.xy, vec4<f32>(0.0));
        return;
    }
    let normal = normalize(surface_hit.normal);
    let origin = surface_hit.point + normal * 1e-3;
    let cell = (nephele.world_max.xyz - nephele.world_min.xyz)
        / vec3<f32>(textureDimensions(nephele_world_sun));
    // Nested midpoint spacing follows the canonical sun integrator's
    // half-cell rule, without imposing a ray-distance or march-count cap.
    let spacing = min(cell.x, min(cell.y, cell.z)) * 0.5;
    var irradiance = vec3<f32>(0.0);
    for (var sample = 0u; sample < 192u; sample = sample + 1u) {
        let i = f32(sample) + 0.5;
        let ct = 1.0 - 2.0 * i / 192.0;
        let st = sqrt(max(0.0, 1.0 - ct*ct));
        let theta = 3.141592653589793 * (1.0 + sqrt(5.0)) * i;
        let direction = vec3<f32>(cos(theta)*st, ct, sin(theta)*st);
        let cosine = dot(normal, direction);
        if (!(cosine > 0.0)) { continue; }
        let interval = nephele_medium_interval(origin, direction);
        var end = interval.y;
        if (!(end > interval.x)) { continue; }
        if (nephele.scattering.x > 0.5) {
            let blocker = terrain_trace(Ray(origin, 1e-4, direction, end), false, true);
            if (blocker.hit != 0u) { end = min(end, blocker.t); }
        }
        if (!(end > interval.x)) { continue; }
        let steps = u32(ceil((end - interval.x) / spacing));
        let step = (end - interval.x) / f32(steps);
        let phase = nephele_cloudlight_phase(dot(direction, normalize(nephele.sun.xyz)));
        var transmittance = vec3<f32>(1.0);
        var radiance = vec3<f32>(0.0);
        for (var s = 0u; s < steps; s = s + 1u) {
            let p = origin + direction * (interval.x + (f32(s) + 0.5) * step);
            let source = nephele_cloudlight_sample(p);
            let extinction = nephele.sigma_t.rgb * source.a;
            let attenuation = exp(-extinction * step);
            radiance += transmittance * exp(-extinction * step * 0.5) * source.rgb * phase * step;
            transmittance *= attenuation;
        }
        irradiance += radiance * cosine * (12.566370614359172 / 192.0);
    }
    textureStore(nephele_surface_cloudlight, id.xy, vec4<f32>(irradiance, 1.0));
}
