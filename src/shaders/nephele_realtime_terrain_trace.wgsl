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
    // Match the froxel consumer's 2D CDF stratification. A neighbouring
    // blue-noise rank is strongly anti-correlated with the first coordinate.
    let rank = u32(u1 * 64.0);
    let u2 = f32(reverseBits(rank)) / 4294967296.0 + 0.5 / 64.0;
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

// Discrete-ordinate transport qualified by the preserved 2026-10-22 CPU study.
// Assembled after the production terrain traversal and NEPHELE trace adapter.
// No terrain-interior density rejection. All new visibility uses scattering.x.
const NEPHELE_DIRECTIONS: u32 = 192u;
struct NepheleDensityNodes { dimensions: vec4<u32>, values: array<f32>, }
@group(3) @binding(17) var<storage, read> nd_density: NepheleDensityNodes;
@group(3) @binding(18) var nd_source: texture_2d<u32>;
@group(3) @binding(19) var nd_source_next: texture_storage_2d<rg32uint,write>;
@group(3) @binding(20) var nd_incoming: texture_2d<u32>;
@group(3) @binding(21) var<storage, read> nd_surface: array<vec2<u32>>;
@group(3) @binding(22) var<storage, read_write> nd_surface_next: array<vec2<u32>>;
@group(3) @binding(23) var nd_sky: texture_2d<f32>;
@group(3) @binding(24) var nd_scatter_out: texture_storage_3d<rgba16float, write>;
@group(3) @binding(25) var nd_single_out: texture_storage_3d<rgba16float, write>;
@group(3) @binding(26) var<storage, read> nd_first_surface: array<vec2<u32>>;
@group(3) @binding(27) var nd_incoming_out: texture_storage_2d<rgba32uint,write>;
@group(3) @binding(28) var nd_active: texture_3d<u32>;
@group(3) @binding(29) var nd_source_zero: texture_storage_2d<rg32uint,write>;
struct NepheleConvergence {
    changed: atomic<u32>,
    source_delta: atomic<u32>,
    surface_delta: atomic<u32>,
    invalid: atomic<u32>,
}
@group(3) @binding(30) var<storage, read_write> nd_progress: NepheleConvergence;

fn nd_pack(v: vec3<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(v.xy), pack2x16float(vec2<f32>(v.z, 0.0)));
}
fn nd_unpack(v: vec2<u32>) -> vec3<f32> {
    return vec3<f32>(unpack2x16float(v.x), unpack2x16float(v.y).x);
}
fn nd_direction(j: u32) -> vec3<f32> {
    let i = f32(j) + 0.5;
    let y = 1.0 - 2.0 * i / f32(NEPHELE_DIRECTIONS);
    let theta = 3.141592653589793 * (1.0 + sqrt(5.0)) * i;
    let r = sqrt(max(0.0, 1.0 - y*y));
    return vec3<f32>(cos(theta)*r, y, sin(theta)*r);
}
fn nd_density_at(p: vec3<f32>) -> f32 {
    if (any(p < nephele.world_min.xyz) || any(p > nephele.world_max.xyz)) { return 0.0; }
    let dims = vec3<i32>(nd_density.dimensions.xyz);
    let q = (p - nephele.world_min.xyz) / (nephele.world_max.xyz - nephele.world_min.xyz) * vec3<f32>(dims) - 0.5;
    let base = vec3<i32>(floor(q));
    let f = fract(q);
    var result = 0.0;
    for (var z=0; z<2; z=z+1) { for (var y=0; y<2; y=y+1) { for (var x=0; x<2; x=x+1) {
        let offset = vec3<i32>(x,y,z);
        let c = vec3<u32>(clamp(base+offset, vec3<i32>(0), dims-1));
        let w = mix(vec3<f32>(1.0)-f, f, vec3<f32>(offset));
        result += nd_density.values[c.x + u32(dims.x)*(c.y + u32(dims.y)*c.z)] * w.x*w.y*w.z;
    }}}
    return result;
}
fn nd_density_integral(p: vec3<f32>, d: vec3<f32>, reach: f32) -> f32 {
    // Each restriction of trilinear density is cubic. Split on texel-centre
    // planes and use two Gauss nodes, exactly as the qualified CPU operator.
    let interval = nephele_medium_interval(p,d);
    let end = min(interval.y, reach);
    var left = interval.x;
    var value = 0.0;
    let dims = vec3<f32>(nd_density.dimensions.xyz);
    let cell = (nephele.world_max.xyz-nephele.world_min.xyz)/dims;
    loop {
        if (!(end > left)) { break; }
        var right = end;
        for (var axis=0u; axis<3u; axis=axis+1u) {
            if (d[axis] != 0.0) {
                // All density planes, bounded by the actual input dimensions.
                // No floating-point epsilon or selected crossing-count cap.
                for (var j=0u; j<nd_density.dimensions[axis]; j=j+1u) {
                    let t = (nephele.world_min[axis]+(f32(j)+0.5)*cell[axis]-p[axis])/d[axis];
                    if (t > left && t < right) { right=t; }
                }
            }
        }
        let half_step = (right-left)*0.5;
        let middle = (left+right)*0.5;
        let offset = half_step/sqrt(3.0);
        value += half_step*(nd_density_at(p+d*(middle-offset))+nd_density_at(p+d*(middle+offset)));
        left = right;
    }
    return value;
}
fn nd_froxel_index(p: vec3<f32>) -> u32 {
    let clip = nephele.view_proj*vec4<f32>(p,1.0);
    let ndc = clip.xy/clip.w;
    let xy = vec2<f32>(ndc.x*0.5+0.5,0.5-ndc.y*0.5)*vec2<f32>(nephele.grid.xy-2u)+1.0;
    let z = clamp(log(max(length(p-nephele.camera.xyz),nephele.depth.x)/nephele.depth.x)/nephele.depth.z,0.0,1.0)*f32(nephele.grid.z);
    let c = vec3<u32>(clamp(vec3<f32>(xy,z),vec3<f32>(0.0),vec3<f32>(nephele.grid.xyz)-1.0));
    return c.x+nephele.grid.x*(c.y+nephele.grid.y*c.z);
}
fn nd_froxel_id(flat: u32) -> vec3<u32> {
    return vec3<u32>(flat%nephele.grid.x,(flat/nephele.grid.x)%nephele.grid.y,flat/(nephele.grid.x*nephele.grid.y));
}
fn nd_cache_coordinate(flat: u32, j: u32, dimensions: vec2<u32>) -> vec2<i32> {
    let active_index=textureLoad(nd_active,vec3<i32>(nd_froxel_id(flat)),0).r;
    let index=(active_index-1u)*NEPHELE_DIRECTIONS+j;
    return vec2<i32>(i32(index%dimensions.x),i32(index/dimensions.x));
}
fn nd_sky_radiance(d: vec3<f32>) -> vec3<f32> {
    let uv = vec2<f32>(atan2(d.z,d.x)*0.15915494309189535+0.5,acos(clamp(d.y,-1.0,1.0))*0.3183098861837907);
    let dims = textureDimensions(nd_sky);
    return max(textureLoad(nd_sky,vec2<i32>(min(vec2<u32>(uv*vec2<f32>(dims)),dims-1u)),0).rgb,vec3<f32>(0.0));
}
fn nd_surface_uv(p: vec3<f32>) -> vec2<f32> {
    let extent = terrain.origin_spacing.zw*vec2<f32>(terrain.dims.xy-1u);
    return clamp((p.xz-terrain.origin_spacing.xy)/extent,vec2<f32>(0.0),vec2<f32>(1.0))*vec2<f32>(nephele.grid.xy-1u);
}
fn nd_ground_radiance(p: vec3<f32>) -> vec3<f32> {
    let uv = nd_surface_uv(p);
    let base = min(vec2<u32>(floor(uv)),nephele.grid.xy-2u);
    let f = uv-vec2<f32>(base);
    let i = base.x+nephele.grid.x*base.y;
    return mix(mix(nd_unpack(nd_surface[i]),nd_unpack(nd_surface[i+1u]),f.x),mix(nd_unpack(nd_surface[i+nephele.grid.x]),nd_unpack(nd_surface[i+nephele.grid.x+1u]),f.x),f.y);
}
fn nd_sun_transmittance(p: vec3<f32>) -> vec3<f32> {
    let sun = normalize(nephele.sun.xyz);
    if (nephele.scattering.x>0.5) {
        let hit = terrain_trace(Ray(p,1e-3,sun,3.402823466e38),false,true);
        if (hit.hit != 0u) { return vec3<f32>(0.0); }
    }
    return exp(-nephele.sigma_t.rgb*nd_density_integral(p,sun,3.402823466e38));
}
fn nd_sun(p: vec3<f32>) -> vec3<f32> {
    return nephele.sun_radiance.rgb*nephele.sun_radiance.a*nd_sun_transmittance(p);
}
fn nd_froxel_sun(p: vec3<f32>) -> vec3<f32> {
    return nephele.sun_radiance.rgb*nephele.sun_radiance.a*nd_unpack(nd_pack(nd_sun_transmittance(p)));
}
fn nd_first_terminal(p: vec3<f32>, j: u32) -> vec3<f32> {
    let d = nd_direction(j);
    var terminal = nd_sky_radiance(d);
    var reach = 3.402823466e38;
    if (nephele.scattering.x>0.5) {
        let hit = terrain_trace(Ray(p,1e-3,d,reach),false,true);
        if (hit.hit!=0u) {
            reach=hit.t;
            let uv=nd_surface_uv(hit.point);
            let base=min(vec2<u32>(floor(uv)),nephele.grid.xy-2u);
            let f=uv-vec2<f32>(base);
            let i=base.x+nephele.grid.x*base.y;
            terminal=mix(mix(nd_unpack(nd_first_surface[i]),nd_unpack(nd_first_surface[i+1u]),f.x),mix(nd_unpack(nd_first_surface[i+nephele.grid.x]),nd_unpack(nd_first_surface[i+nephele.grid.x+1u]),f.x),f.y);
        }
    }
    return terminal*exp(-nephele.sigma_t.rgb*nd_density_integral(p,d,reach));
}
fn nd_characteristic(p: vec3<f32>, j: u32) -> vec3<f32> {
    let d = nd_direction(j);
    var terminal = nd_sky_radiance(d);
    var reach = 3.402823466e38;
    if (nephele.scattering.x>0.5) {
        let hit = terrain_trace(Ray(p,1e-3,d,reach),false,true);
        if (hit.hit != 0u) { reach=hit.t; terminal=nd_ground_radiance(hit.point); }
    }
    let interval = nephele_medium_interval(p,d);
    let end = min(interval.y,reach);
    var trans = vec3<f32>(1.0);
    var value = vec3<f32>(0.0);
    if (end>interval.x) {
        let step = (end-interval.x)/f32(nephele.grid.z);
        for (var k=0u; k<nephele.grid.z; k=k+1u) {
            let density_integral=nd_density_integral(p+d*(interval.x+f32(k)*step),d,step);
            let segment = nephele.sigma_t.rgb*density_integral;
            let sample = p+d*(interval.x+(f32(k)+0.5)*step);
            let at = nd_froxel_index(sample);
            let rho = textureLoad(nephele_extinction,vec3<i32>(nd_froxel_id(at)),0).a;
            if (rho>0.0) {
                var weight=vec3<f32>(density_integral);
                for (var c=0u;c<3u;c=c+1u) {
                    if (nephele.sigma_t[c]>0.0) { weight[c]=(1.0-exp(-segment[c]))/nephele.sigma_t[c]; }
                }
                value += trans*weight*nd_unpack(textureLoad(nd_source,nd_cache_coordinate(at,j,textureDimensions(nd_source)),0).xy)/rho;
            }
            trans *= exp(-segment);
        }
    }
    return value+trans*terminal;
}
fn nd_vertex_point(id: vec2<u32>) -> vec3<f32> {
    let uv = vec2<f32>(id)/vec2<f32>(nephele.grid.xy-1u)*vec2<f32>(terrain.dims.xy-1u);
    let cell = min(vec2<u32>(floor(uv)),terrain.dims.xy-2u);
    let f = uv-vec2<f32>(cell);
    let h = terrain_cell_heights(cell.x,cell.y);
    let xz = terrain.origin_spacing.xy+uv*terrain.origin_spacing.zw;
    return vec3<f32>(xz.x,mix(mix(h.x,h.y,f.x),mix(h.z,h.w,f.x),f.y),xz.y);
}
fn nd_measure_change(previous_packed: vec2<u32>, next_packed: vec2<u32>) -> u32 {
    let previous = nd_unpack(previous_packed);
    let next = nd_unpack(next_packed);
    // 65504 is the largest finite f16 value. Reject overflow and loss of the
    // positive operator's monotonicity instead of clipping or looping forever.
    if (any(!(next>=vec3<f32>(0.0))) || any(next>vec3<f32>(65504.0)) || any(next<previous)) {
        atomicStore(&nd_progress.invalid,1u);
    }
    if (any(next_packed!=previous_packed)) { atomicStore(&nd_progress.changed,1u); }
    let delta=abs(next-previous);
    return bitcast<u32>(max(delta.x,max(delta.y,delta.z)));
}
@compute @workgroup_size(4,4,4)
fn main_nephele_directional_incoming(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id>=nephele.grid.xyz)) { return; }
    let at = id.x+nephele.grid.x*(id.y+nephele.grid.y*id.z);
    if (!(textureLoad(nephele_extinction,vec3<i32>(id),0).a>0.0)) { return; }
    let p = nephele_froxel_world(id);
    for (var j=0u;j<NEPHELE_DIRECTIONS;j=j+1u) {
        textureStore(nd_incoming_out,nd_cache_coordinate(at,j,textureDimensions(nd_incoming_out)),bitcast<vec4<u32>>(vec4<f32>(nd_characteristic(p,j),0.0)));
    }
}
@compute @workgroup_size(4,4,4)
fn main_nephele_directional_source(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id>=nephele.grid.xyz)) { return; }
    let at = id.x+nephele.grid.x*(id.y+nephele.grid.y*id.z);
    let rho = textureLoad(nephele_extinction,vec3<i32>(id),0).a;
    if (!(rho>0.0)) { return; }
    let solar = nd_froxel_sun(nephele_froxel_world(id));
    for (var j=0u;j<NEPHELE_DIRECTIONS;j=j+1u) {
        let d = nd_direction(j);
        var sum = solar*nephele_cloudlight_phase(dot(d,normalize(nephele.sun.xyz)));
        for (var k=0u;k<NEPHELE_DIRECTIONS;k=k+1u) {
            sum += (12.566370614359172/f32(NEPHELE_DIRECTIONS))*nephele_cloudlight_phase(dot(d,nd_direction(k)))*bitcast<vec4<f32>>(textureLoad(nd_incoming,nd_cache_coordinate(at,k,textureDimensions(nd_incoming)),0)).rgb;
        }
        let packed=nd_pack(nephele.sigma_s.rgb*rho*sum);
        let previous=textureLoad(nd_source,nd_cache_coordinate(at,j,textureDimensions(nd_source)),0).xy;
        atomicMax(&nd_progress.source_delta,nd_measure_change(previous,packed));
        textureStore(nd_source_next,nd_cache_coordinate(at,j,textureDimensions(nd_source_next)),vec4<u32>(packed,0u,0u));
    }
}
@compute @workgroup_size(8,8,1)
fn main_nephele_directional_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy>=nephele.grid.xy)) { return; }
    let p = nd_vertex_point(id.xy);
    let cell = min(vec2<u32>(max((p.xz-terrain.origin_spacing.xy)/terrain.origin_spacing.zw,vec2<f32>(0.0))),terrain.dims.xy-2u);
    let n = terrain_normal_at(p,cell.x,cell.y);
    var sum = nd_sun(p)*max(dot(n,normalize(nephele.sun.xyz)),0.0);
    for (var j=0u;j<NEPHELE_DIRECTIONS;j=j+1u) {
        sum += nd_characteristic(p,j)*max(dot(n,nd_direction(j)),0.0)*(12.566370614359172/f32(NEPHELE_DIRECTIONS));
    }
    let at=id.x+nephele.grid.x*id.y;
    let packed=nd_pack(nephele.terrain_albedo.rgb*sum*0.3183098861837907);
    atomicMax(&nd_progress.surface_delta,nd_measure_change(nd_surface[at],packed));
    nd_surface_next[at]=packed;
}
@compute @workgroup_size(4,4,4)
fn main_nephele_directional_camera(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id>=nephele.grid.xyz)) { return; }
    let at = id.x+nephele.grid.x*(id.y+nephele.grid.y*id.z);
    let rho = textureLoad(nephele_extinction,vec3<i32>(id),0).a;
    var sum = vec3<f32>(0.0);
    var single = vec3<f32>(0.0);
    if (rho>0.0) {
        let p = nephele_froxel_world(id);
        let incident = normalize(p-nephele.camera.xyz);
        let solar = nd_froxel_sun(p)*nephele_cloudlight_phase(dot(incident,normalize(nephele.sun.xyz)));
        sum = solar;
        single = solar;
        for (var j=0u;j<NEPHELE_DIRECTIONS;j=j+1u) {
            sum += bitcast<vec4<f32>>(textureLoad(nd_incoming,nd_cache_coordinate(at,j,textureDimensions(nd_incoming)),0)).rgb*nephele_cloudlight_phase(dot(incident,nd_direction(j)))*(12.566370614359172/f32(NEPHELE_DIRECTIONS));
            single += nd_first_terminal(p,j)*nephele_cloudlight_phase(dot(incident,nd_direction(j)))*(12.566370614359172/f32(NEPHELE_DIRECTIONS));
        }
        sum *= nephele.sigma_s.rgb*rho;
        single *= nephele.sigma_s.rgb*rho;
    }
    textureStore(nd_scatter_out,id,vec4<f32>(sum,dot(max(sum-single,vec3<f32>(0.0)),vec3<f32>(0.2126,0.7152,0.0722))));
    textureStore(nd_single_out,id,vec4<f32>(single,0.0));
}
@compute @workgroup_size(8,8,1)
fn main_nephele_directional_primary(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy>=nephele.viewport.xy)) { return; }
    let uv = (vec2<f32>(id.xy)+0.5)/vec2<f32>(nephele.viewport.xy);
    let h = nephele.inv_view_proj*vec4<f32>(uv.x*2.0-1.0,1.0-uv.y*2.0,1.0,1.0);
    let ray = normalize(h.xyz/h.w-nephele.camera.xyz);
    let hit = terrain_trace(Ray(nephele.camera.xyz,1e-3,ray,nephele.depth.y),false,true);
    var value = vec3<f32>(0.0);
    if (hit.hit!=0u) {
        var irradiance = vec3<f32>(0.0);
        for (var j=0u;j<NEPHELE_DIRECTIONS;j=j+1u) {
            irradiance += nd_characteristic(hit.point,j)*max(dot(hit.normal,nd_direction(j)),0.0)*(12.566370614359172/f32(NEPHELE_DIRECTIONS));
        }
        value = irradiance-nephele.diffuse_ibl.rgb;
    }
    // Signed environment-irradiance correction plus all medium return. Direct
    // surface sunlight and the material BRDF remain owned by terrain shading.
    textureStore(nephele_surface_cloudlight,id.xy,vec4<f32>(value,select(2.0,3.0,hit.hit!=0u)));
}

@compute @workgroup_size(8,8,1)
fn main_nephele_directional_clear(@builtin(global_invocation_id) id:vec3<u32>) {
    if (any(id.xy>=textureDimensions(nd_source_next))) { return; }
    textureStore(nd_source_next,id.xy,vec4<u32>(0u));
    textureStore(nd_source_zero,id.xy,vec4<u32>(0u));
}
