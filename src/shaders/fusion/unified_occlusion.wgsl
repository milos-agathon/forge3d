// src/shaders/fusion/unified_occlusion.wgsl
// SPLAT-FUSED: one occlusion model across three radically different
// primitive densities, traversed through ONE top-level acceleration
// structure and consumed by the hybrid ReSTIR kernel.
//
//   T_total = T_splat * T_lidar * T_terrain
//
//   * Splats  (sparse, anisotropic): soft analytic transmittance
//       prod exp(-kappa * rho_i)        (gaussian_intersect.wgsl)
//   * LiDAR / COPC points (dense): fixed-radius sphelets whose coverage
//       accumulates as 1 - prod(1 - c_i); T_lidar = prod(1 - c_i), so a
//       dense swath saturates to opaque while a thin one stays translucent
//   * Terrain (continuous): exact ray / heightfield any-hit through the
//       min-max descent of hybrid_terrain_traversal.wgsl — binary per ray,
//       softened by the solar-disc integral the reservoir light sample takes
//
// Every factor is a physically meaningful transmittance in [0, 1], so a
// splat shadows terrain and terrain shadows a LiDAR swath through the same
// code path. Fog (VolumetricParams) is a separate Beer-Lambert factor applied
// once by the caller: optical depths add, nothing is double counted.
//
// The top-level BVH has three leaf kinds — splat pages, COPC octree-node
// pages and terrain tiles. Splat/point leaves reference a page; a page that
// is not in the residency pool raises a request in the page table (traversal
// priority = number of requesting rays) and is skipped for this frame.
//
// This file is assembled into the hybrid kernel by
// shader_sources::fused_kernel(); it uses Ray / HybridHitResult /
// terrain_trace / RestirReservoir / uniforms / lighting from that module.
// CPU mirror of the arithmetic: src/splat/kernel.rs.
// RELEVANT FILES: src/shaders/splat/gaussian_intersect.wgsl,
//                 src/splat/fusion.rs, src/shader_sources.rs

const FUSION_KIND_SPLAT: u32 = 1u;
const FUSION_KIND_LIDAR: u32 = 2u;
const FUSION_KIND_TERRAIN: u32 = 3u;
const FUSION_LEAF_FLAG: u32 = 0x80000000u;
const FUSION_COUNT_MASK: u32 = 0x0fffffffu;
const FUSION_NOT_RESIDENT: u32 = 0xffffffffu;
const FUSION_TLAS_STACK: u32 = 48u;
const FUSION_BLAS_STACK: u32 = 32u;
// HybridHitResult.hit_type values for fused surfaces (3 = terrain).
const FUSION_HIT_SPLAT: u32 = 4u;
const FUSION_HIT_LIDAR: u32 = 5u;
// Page table: 4 header words, then (slot, requests, touched, reserved).
const FUSION_PAGE_HEADER: u32 = 4u;
const FUSION_PAGE_STRIDE: u32 = 4u;
// Hit colours travel in HybridHitResult._pad as 16-bit fixed point.
const FUSION_COLOR_SCALE: f32 = 8.0;
// Grazing secondary rays skip at most 1 / this many bias lengths.
const FUSION_BIAS_MIN_COSINE: f32 = 0.25;

struct FusionNode {
    aabb_min: vec3<f32>,
    a: u32,   // interior: left child   | leaf: page id / first primitive
    aabb_max: vec3<f32>,
    b: u32,   // interior: right child  | leaf: flag | kind << 28 | count
}

struct FusionSplat {
    pos_opacity: vec4<f32>,
    sh0_degree: vec4<f32>,
    sh1_0: vec4<f32>,
    sh1_1: vec4<f32>,
    sh1_2: vec4<f32>,
}

struct FusionInvCov {
    m0: vec4<f32>,   // xx, xy, xz, yy
    m1: vec4<f32>,   // yz, zz, -, -
}

struct FusionPoint {
    pos_radius: vec4<f32>,
    color: vec4<f32>,
}

struct FusionUniforms {
    counts: vec4<u32>,    // tlas_node_count, page_count, blas_base, nodes_per_slot
    pools: vec4<u32>,     // splat_slots, point_slots, prims_per_slot, flags
    optics: vec4<f32>,    // kappa, lidar_opacity, transmittance_epsilon, ibl_occlusion_distance
    sampling: vec4<f32>,  // cos(sun radius), restir defensive floor, accept threshold, ray epsilon
    surface: vec4<f32>,   // self-shadow bias: splat sigmas, LiDAR radii, -, -
    shading: ShadingParamsGPU,
}

@group(1) @binding(5) var<uniform> fusion: FusionUniforms;
// The fused top-level BVH, followed by one per-page tree per residency slot.
@group(1) @binding(6) var<storage, read> fusion_bvh: array<FusionNode>;
// Page table + traversal feedback.
@group(1) @binding(7) var<storage, read_write> fusion_pages: array<atomic<u32>>;
// Splat SoA records and their packed inverse covariance.
@group(1) @binding(8) var<storage, read> fusion_splats: array<FusionSplat>;
@group(1) @binding(9) var<storage, read> fusion_splat_icov: array<FusionInvCov>;
// COPC / LiDAR point pages.
@group(1) @binding(10) var<storage, read> fusion_points: array<FusionPoint>;
// Participating media: the VolumetricParams layout of volumetric.wgsl.
@group(1) @binding(11) var<uniform> fusion_media: VolumetricParams;

// Per-invocation state. A ray that reached a non-resident page bumps
// fusion_miss_events; the last closest hit is remembered so the shadow and
// environment rays leaving it do not occlude themselves.
var<private> fusion_miss_events: u32;
var<private> fusion_self_kind: u32;
var<private> fusion_self_id: u32;
var<private> fusion_deterministic: bool;
// A splat/point surface is a layer of overlapping soft primitives. Secondary
// rays leaving it skip the first stretch of the ray — until it has risen
// `fusion_self_bias` along the surface normal, clear of the hit primitive's
// own extent — so the layer does not shadow itself. The ray itself is not
// moved, so shadows cast by everything beyond that stretch stay exactly
// where they are. `fusion_hit_bias` tracks the bias of the closest hit
// accepted so far during a traversal.
var<private> fusion_hit_bias: f32;
var<private> fusion_self_bias: f32;
var<private> fusion_self_normal: vec3<f32>;

fn fusion_enabled() -> bool {
    return fusion.counts.x != 0u;
}

fn fusion_hash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

// Per-ray seed from the ray itself: jittered camera rays differ per sample
// and frame, so no RNG state has to be threaded through the traversal.
fn fusion_ray_seed(ray: Ray) -> u32 {
    let o = bitcast<vec3<u32>>(ray.origin);
    let d = bitcast<vec3<u32>>(ray.direction);
    var h = fusion_hash(d.z ^ uniforms.frame_index ^ uniforms.seed_hi);
    h = fusion_hash(d.y ^ h);
    h = fusion_hash(d.x ^ h);
    h = fusion_hash(o.z ^ h);
    h = fusion_hash(o.y ^ h);
    return fusion_hash(o.x ^ h);
}

fn fusion_unit(seed: u32) -> f32 {
    return f32(fusion_hash(seed) >> 8u) * (1.0 / 16777216.0);
}

fn fusion_inv_dir(ray: Ray) -> vec3<f32> {
    return vec3<f32>(
        terrain_safe_inv(ray.direction.x),
        terrain_safe_inv(ray.direction.y),
        terrain_safe_inv(ray.direction.z),
    );
}

// Slab test: (t_enter, t_exit) clipped to [ray.tmin, tmax]; empty if x > y.
fn fusion_aabb_span(
    ray: Ray,
    inv_dir: vec3<f32>,
    lo: vec3<f32>,
    hi: vec3<f32>,
    tmax: f32,
) -> vec2<f32> {
    let t0 = (lo - ray.origin) * inv_dir;
    let t1 = (hi - ray.origin) * inv_dir;
    let near = min(t0, t1);
    let far = max(t0, t1);
    return vec2<f32>(
        max(max(near.x, near.y), max(near.z, ray.tmin)),
        min(min(far.x, far.y), min(far.z, tmax)),
    );
}

// Resolve a page to its residency slot. A miss raises an asynchronous page
// request (request count = traversal priority); a hit marks the page used.
fn fusion_page_slot(page: u32) -> u32 {
    let base = FUSION_PAGE_HEADER + page * FUSION_PAGE_STRIDE;
    let slot = atomicLoad(&fusion_pages[base]);
    if (slot == FUSION_NOT_RESIDENT) {
        let requests = atomicAdd(&fusion_pages[base + 1u], 1u);
        let misses = atomicAdd(&fusion_pages[0], 1u);
        fusion_miss_events = fusion_miss_events + 1u;
    } else {
        atomicStore(&fusion_pages[base + 2u], 1u);
    }
    return slot;
}

fn fusion_pack_color(color: vec3<f32>) -> vec2<u32> {
    let unit = clamp(color / FUSION_COLOR_SCALE, vec3<f32>(0.0), vec3<f32>(1.0));
    let q = vec3<u32>(unit * 65535.0 + vec3<f32>(0.5));
    return vec2<u32>(q.x | (q.y << 16u), q.z);
}

fn fusion_unpack_color(packed: vec2<u32>) -> vec3<f32> {
    return vec3<f32>(
        f32(packed.x & 0xffffu),
        f32(packed.x >> 16u),
        f32(packed.y & 0xffffu),
    ) * (FUSION_COLOR_SCALE / 65535.0);
}

fn fusion_is_self(kind: u32, id: u32) -> bool {
    return fusion_self_kind == kind && fusion_self_id == id;
}

// A soft primitive becomes the visible surface with probability 1 - T (so
// camera visibility and shadow transmittance agree in expectation), or
// against the fixed threshold on deterministic G-buffer / AOV rays.
fn fusion_accept(probability: f32, seed: u32, id: u32) -> bool {
    if (fusion_deterministic) {
        return probability >= fusion.sampling.z;
    }
    return fusion_unit(seed ^ fusion_hash(id + 0x9e3779b9u)) < probability;
}

// ---------------------------------------------------------------------------
// Closest hit inside one resident page
// ---------------------------------------------------------------------------

fn fusion_splat_page_closest(
    ray: Ray,
    inv_dir: vec3<f32>,
    slot: u32,
    page: u32,
    seed: u32,
    best: ptr<function, HybridHitResult>,
) {
    let node_base = fusion.counts.z + slot * fusion.counts.w;
    let prim_base = slot * fusion.pools.z;
    var stack: array<u32, FUSION_BLAS_STACK>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = fusion_bvh[node_base + stack[sp]];
        let span = fusion_aabb_span(ray, inv_dir, node.aabb_min, node.aabb_max, (*best).t);
        if (span.x > span.y) { continue; }
        if ((node.b & FUSION_LEAF_FLAG) == 0u) {
            if (sp + 2u <= FUSION_BLAS_STACK) {
                stack[sp] = node.a;
                stack[sp + 1u] = node.b;
                sp = sp + 2u;
            }
            continue;
        }
        let count = node.b & FUSION_COUNT_MASK;
        for (var k = 0u; k < count; k = k + 1u) {
            let local = node.a + k;
            let id = page * fusion.pools.z + local;
            if (fusion_is_self(FUSION_HIT_SPLAT, id)) { continue; }
            let splat = fusion_splats[prim_base + local];
            let icov = fusion_splat_icov[prim_base + local];
            let h = gaussian_closest_hit(
                ray.origin, ray.direction, ray.tmin, ray.tmax,
                splat.pos_opacity.xyz, icov.m0, icov.m1, splat.pos_opacity.w);
            if (h.response <= 0.0 || h.t_star <= ray.tmin) { continue; }
            let t = max(h.t_hit, ray.tmin + 1e-4);
            if (t >= (*best).t) { continue; }
            let probability = 1.0 - gaussian_transmittance(h.response, fusion.optics.x);
            if (!fusion_accept(probability, seed, id)) { continue; }
            (*best).hit = 1u;
            (*best).t = t;
            (*best).point = ray.origin + ray.direction * t;
            (*best).normal = h.shading_normal;
            (*best).material_id = id;
            (*best).hit_type = FUSION_HIT_SPLAT;
            // Standard deviation of the splat along its shading normal.
            let along = dot(h.shading_normal,
                gaussian_sym_mul(icov.m0, icov.m1, h.shading_normal));
            fusion_hit_bias = fusion.surface.x / sqrt(max(along, 1e-20));
            (*best)._pad = fusion_pack_color(gaussian_sh_color(
                splat.sh0_degree.xyz, splat.sh0_degree.w,
                splat.sh1_0.xyz, splat.sh1_1.xyz, splat.sh1_2.xyz, ray.direction));
        }
    }
}

struct FusionSphelet {
    coverage: f32,   // opacity * max(1 - b^2 / r^2, 0)
    t_hit: f32,      // entry into the sphere
}

// Fixed-radius disc/sphelet coverage of one LiDAR return.
fn fusion_sphelet(ray: Ray, point: FusionPoint) -> FusionSphelet {
    var out: FusionSphelet;
    let oc = point.pos_radius.xyz - ray.origin;
    let t_closest = dot(oc, ray.direction);
    let perp = oc - t_closest * ray.direction;
    let b2 = dot(perp, perp);
    let r2 = point.pos_radius.w * point.pos_radius.w;
    out.coverage = 0.0;
    if (t_closest > ray.tmin && t_closest < ray.tmax && b2 < r2) {
        out.coverage = fusion.optics.y * (1.0 - b2 / r2);
    }
    out.t_hit = t_closest - sqrt(max(r2 - b2, 0.0));
    return out;
}

fn fusion_point_page_closest(
    ray: Ray,
    inv_dir: vec3<f32>,
    slot: u32,
    page: u32,
    seed: u32,
    best: ptr<function, HybridHitResult>,
) {
    let node_base = fusion.counts.z + (fusion.pools.x + slot) * fusion.counts.w;
    let prim_base = slot * fusion.pools.z;
    var stack: array<u32, FUSION_BLAS_STACK>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = fusion_bvh[node_base + stack[sp]];
        let span = fusion_aabb_span(ray, inv_dir, node.aabb_min, node.aabb_max, (*best).t);
        if (span.x > span.y) { continue; }
        if ((node.b & FUSION_LEAF_FLAG) == 0u) {
            if (sp + 2u <= FUSION_BLAS_STACK) {
                stack[sp] = node.a;
                stack[sp + 1u] = node.b;
                sp = sp + 2u;
            }
            continue;
        }
        let count = node.b & FUSION_COUNT_MASK;
        for (var k = 0u; k < count; k = k + 1u) {
            let local = node.a + k;
            let id = page * fusion.pools.z + local;
            if (fusion_is_self(FUSION_HIT_LIDAR, id)) { continue; }
            let point = fusion_points[prim_base + local];
            let s = fusion_sphelet(ray, point);
            if (s.coverage <= 0.0) { continue; }
            let t = max(s.t_hit, ray.tmin + 1e-4);
            if (t >= (*best).t) { continue; }
            if (!fusion_accept(s.coverage, seed, id)) { continue; }
            let p = ray.origin + ray.direction * t;
            var n = p - point.pos_radius.xyz;
            let len = length(n);
            if (len > 1e-20) { n = n / len; } else { n = -ray.direction; }
            if (dot(n, ray.direction) > 0.0) { n = -n; }
            (*best).hit = 1u;
            (*best).t = t;
            (*best).point = p;
            (*best).normal = n;
            (*best).material_id = id;
            (*best).hit_type = FUSION_HIT_LIDAR;
            fusion_hit_bias = fusion.surface.y * point.pos_radius.w;
            (*best)._pad = fusion_pack_color(point.color.rgb);
        }
    }
}

// Clip a ray to a terrain tile's span, backing off slightly so a hit that
// lies exactly on a shared tile boundary is found by one of the two tiles.
fn fusion_tile_ray(ray: Ray, span: vec2<f32>, tmax: f32) -> Ray {
    var tile = ray;
    let slack = 1e-3 * (1.0 + abs(span.x));
    tile.tmin = max(ray.tmin, span.x - slack);
    tile.tmax = min(tmax, span.y + slack);
    return tile;
}

// Closest fused surface along the ray: one traversal of the single top-level
// structure, dispatching on the leaf kind. Children are visited near-first.
fn fusion_closest_hit(ray: Ray) -> HybridHitResult {
    var best: HybridHitResult;
    best.hit = 0u;
    best.t = ray.tmax;
    if (!fusion_enabled()) { return best; }
    let inv_dir = fusion_inv_dir(ray);
    let seed = fusion_ray_seed(ray);
    var stack: array<u32, FUSION_TLAS_STACK>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = fusion_bvh[stack[sp]];
        let span = fusion_aabb_span(ray, inv_dir, node.aabb_min, node.aabb_max, best.t);
        if (span.x > span.y) { continue; }
        if ((node.b & FUSION_LEAF_FLAG) == 0u) {
            let left = fusion_bvh[node.a];
            let right = fusion_bvh[node.b];
            let sl = fusion_aabb_span(ray, inv_dir, left.aabb_min, left.aabb_max, best.t);
            let sr = fusion_aabb_span(ray, inv_dir, right.aabb_min, right.aabb_max, best.t);
            let hit_left = sl.x <= sl.y;
            let hit_right = sr.x <= sr.y;
            if (hit_left && hit_right && sp + 2u <= FUSION_TLAS_STACK) {
                // Far child first so the near child is popped first.
                let left_near = sl.x <= sr.x;
                stack[sp] = select(node.a, node.b, left_near);
                stack[sp + 1u] = select(node.b, node.a, left_near);
                sp = sp + 2u;
            } else if (hit_left && sp < FUSION_TLAS_STACK) {
                stack[sp] = node.a;
                sp = sp + 1u;
            } else if (hit_right && sp < FUSION_TLAS_STACK) {
                stack[sp] = node.b;
                sp = sp + 1u;
            }
            continue;
        }
        let kind = (node.b >> 28u) & 7u;
        if (kind == FUSION_KIND_TERRAIN) {
            let hit = terrain_trace(fusion_tile_ray(ray, span, best.t), false, false);
            if (hit.hit != 0u && hit.t < best.t) { best = hit; }
            continue;
        }
        let slot = fusion_page_slot(node.a);
        if (slot == FUSION_NOT_RESIDENT) { continue; }
        if (kind == FUSION_KIND_SPLAT) {
            fusion_splat_page_closest(ray, inv_dir, slot, node.a, seed, &best);
        } else {
            fusion_point_page_closest(ray, inv_dir, slot, node.a, seed, &best);
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Shadow / any-hit transmittance inside one resident page
// ---------------------------------------------------------------------------

fn fusion_splat_page_shadow(
    ray: Ray,
    inv_dir: vec3<f32>,
    slot: u32,
    page: u32,
    incoming: f32,
) -> f32 {
    var transmittance = incoming;
    let node_base = fusion.counts.z + slot * fusion.counts.w;
    let prim_base = slot * fusion.pools.z;
    var stack: array<u32, FUSION_BLAS_STACK>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = fusion_bvh[node_base + stack[sp]];
        let span = fusion_aabb_span(ray, inv_dir, node.aabb_min, node.aabb_max, ray.tmax);
        if (span.x > span.y) { continue; }
        if ((node.b & FUSION_LEAF_FLAG) == 0u) {
            if (sp + 2u <= FUSION_BLAS_STACK) {
                stack[sp] = node.a;
                stack[sp + 1u] = node.b;
                sp = sp + 2u;
            }
            continue;
        }
        let count = node.b & FUSION_COUNT_MASK;
        for (var k = 0u; k < count; k = k + 1u) {
            let local = node.a + k;
            if (fusion_is_self(FUSION_HIT_SPLAT, page * fusion.pools.z + local)) { continue; }
            let splat = fusion_splats[prim_base + local];
            let icov = fusion_splat_icov[prim_base + local];
            transmittance = gaussian_any_hit(
                transmittance, ray.origin, ray.direction, ray.tmin, ray.tmax,
                splat.pos_opacity.xyz, icov.m0, icov.m1, splat.pos_opacity.w,
                fusion.optics.x);
        }
        // Early out once the splat product is below epsilon.
        if (transmittance < fusion.optics.z) { return transmittance; }
    }
    return transmittance;
}

fn fusion_point_page_shadow(
    ray: Ray,
    inv_dir: vec3<f32>,
    slot: u32,
    page: u32,
    incoming: f32,
) -> f32 {
    var transmittance = incoming;
    let node_base = fusion.counts.z + (fusion.pools.x + slot) * fusion.counts.w;
    let prim_base = slot * fusion.pools.z;
    var stack: array<u32, FUSION_BLAS_STACK>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = fusion_bvh[node_base + stack[sp]];
        let span = fusion_aabb_span(ray, inv_dir, node.aabb_min, node.aabb_max, ray.tmax);
        if (span.x > span.y) { continue; }
        if ((node.b & FUSION_LEAF_FLAG) == 0u) {
            if (sp + 2u <= FUSION_BLAS_STACK) {
                stack[sp] = node.a;
                stack[sp + 1u] = node.b;
                sp = sp + 2u;
            }
            continue;
        }
        let count = node.b & FUSION_COUNT_MASK;
        for (var k = 0u; k < count; k = k + 1u) {
            let local = node.a + k;
            if (fusion_is_self(FUSION_HIT_LIDAR, page * fusion.pools.z + local)) { continue; }
            // Coverage accumulates as 1 - prod(1 - c_i); in transmittance
            // space that is the running product of (1 - c_i).
            let s = fusion_sphelet(ray, fusion_points[prim_base + local]);
            transmittance = transmittance * (1.0 - clamp(s.coverage, 0.0, 1.0));
        }
        if (transmittance < fusion.optics.z) { return transmittance; }
    }
    return transmittance;
}

// The three sub-occluders along a ray as (T_splat, T_lidar, T_terrain).
// Splat/point pages only occlude up to `prim_tmax` (environment rays bound
// it; sun rays pass ray.tmax). `sun_curvature` selects the sun-ray earth
// curvature policy of the terrain descent.
fn fusion_shadow_parts(leaving: Ray, prim_tmax: f32, sun_curvature: bool) -> vec3<f32> {
    var parts = vec3<f32>(1.0);
    if (!fusion_enabled()) { return parts; }
    // Leave the soft surface the ray starts on (see fusion_self_bias).
    var ray = leaving;
    if (fusion_self_kind == FUSION_HIT_SPLAT || fusion_self_kind == FUSION_HIT_LIDAR) {
        let rise = max(dot(fusion_self_normal, leaving.direction), FUSION_BIAS_MIN_COSINE);
        ray.tmin = leaving.tmin + fusion_self_bias / rise;
    }
    let epsilon = fusion.optics.z;
    let inv_dir = fusion_inv_dir(ray);
    var page_ray = ray;
    page_ray.tmax = min(ray.tmax, prim_tmax);
    var stack: array<u32, FUSION_TLAS_STACK>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = fusion_bvh[stack[sp]];
        let span = fusion_aabb_span(ray, inv_dir, node.aabb_min, node.aabb_max, ray.tmax);
        if (span.x > span.y) { continue; }
        if ((node.b & FUSION_LEAF_FLAG) == 0u) {
            if (sp + 2u <= FUSION_TLAS_STACK) {
                stack[sp] = node.a;
                stack[sp + 1u] = node.b;
                sp = sp + 2u;
            }
            continue;
        }
        let kind = (node.b >> 28u) & 7u;
        if (kind == FUSION_KIND_TERRAIN) {
            terrain_is_shadow = true;
            let hit = terrain_trace(fusion_tile_ray(ray, span, ray.tmax), true, sun_curvature);
            terrain_is_shadow = false;
            if (hit.hit != 0u) {
                // Exact heightfield occlusion: binary. Nothing behind it matters.
                parts.z = 0.0;
                return parts;
            }
            continue;
        }
        if (span.x > page_ray.tmax) { continue; }
        let slot = fusion_page_slot(node.a);
        if (slot == FUSION_NOT_RESIDENT) { continue; }
        if (kind == FUSION_KIND_SPLAT) {
            parts.x = fusion_splat_page_shadow(page_ray, inv_dir, slot, node.a, parts.x);
        } else {
            parts.y = fusion_point_page_shadow(page_ray, inv_dir, slot, node.a, parts.y);
        }
        if (parts.x * parts.y < epsilon) { return parts; }
    }
    return parts;
}

// THE unified occlusion query consumed by ReSTIR light sampling:
// T_total = T_splat * T_lidar * T_terrain along the ray.
fn shadow_transmittance(ray: Ray) -> f32 {
    let parts = fusion_shadow_parts(ray, ray.tmax, true);
    return parts.x * parts.y * parts.z;
}

// Exponential height fog in the VolumetricParams convention: optical depth
// tau = density * integral exp(-height_falloff * y) ds, closed form along a
// straight ray. Applied once, as its own factor next to shadow_transmittance.
fn fusion_media_transmittance(ray: Ray) -> f32 {
    if (fusion_media.density <= 0.0) { return 1.0; }
    let distance = min(ray.tmax, fusion_media.max_distance) - ray.tmin;
    if (distance <= 0.0) { return 1.0; }
    let base = fusion_media.density * exp(-fusion_media.height_falloff * ray.origin.y);
    let k = fusion_media.height_falloff * ray.direction.y;
    var tau = base * distance;
    if (abs(k) >= 1e-6) {
        tau = base * (1.0 - exp(-k * distance)) / k;
    }
    return exp(-max(tau, 0.0));
}

// ---------------------------------------------------------------------------
// Hybrid-kernel integration: the base traversal entry points are renamed
// `*_base` at assembly time and these take their place, so every caller in
// the existing integrator (beauty rays, G-buffer, AOVs, shadow, IBL) sees
// the fused scene without a second integrator.
// ---------------------------------------------------------------------------

fn intersect_hybrid(ray: Ray) -> HybridHitResult {
    fusion_self_kind = 0u;
    var best = intersect_hybrid_base(ray);
    if (best.hit == 0u) { best.t = ray.tmax; }
    var fused_ray = ray;
    fused_ray.tmax = best.t;
    let fused = fusion_closest_hit(fused_ray);
    if (fused.hit != 0u && fused.t < best.t) { best = fused; }
    fusion_self_kind = select(0u, best.hit_type, best.hit != 0u);
    fusion_self_id = best.material_id;
    fusion_self_normal = best.normal;
    fusion_self_bias = 0.0;
    if (fusion_self_kind == FUSION_HIT_SPLAT || fusion_self_kind == FUSION_HIT_LIDAR) {
        fusion_self_bias = fusion_hit_bias;
    }
    return best;
}

fn get_surface_properties(hit: HybridHitResult) -> vec3f {
    if (hit.hit_type == FUSION_HIT_SPLAT || hit.hit_type == FUSION_HIT_LIDAR) {
        return fusion_unpack_color(hit._pad);
    }
    return get_surface_properties_base(hit);
}

fn intersect_shadow_ray(ray: Ray, max_distance: f32) -> bool {
    if (intersect_shadow_ray_base(ray, max_distance)) { return true; }
    var clipped = ray;
    clipped.tmax = min(ray.tmax, max_distance);
    return shadow_transmittance(clipped) < fusion.sampling.z;
}

// Environment visibility is a stochastic estimate of the transmittance (one
// uniform per ray), which the accumulation converges. Splats and points only
// occlude within the configured distance; terrain occludes at any distance.
fn intersect_ibl_occlusion_ray(ray: Ray, max_distance: f32) -> bool {
    if (intersect_ibl_occlusion_ray_base(ray, max_distance)) { return true; }
    var clipped = ray;
    clipped.tmax = min(ray.tmax, max_distance);
    let parts = fusion_shadow_parts(clipped, fusion.optics.w, false);
    let transmittance = parts.x * parts.y * parts.z;
    return fusion_unit(fusion_ray_seed(ray) ^ 0x5bd1e995u) >= transmittance;
}

// Surface response through the shared BRDF dispatcher. The hybrid kernel's
// light colour is pi-normalised (a white Lambert surface returns the light
// colour), so the BRDF is scaled by pi: Lambert reproduces the kernel's
// albedo * light * cos shading exactly.
fn fusion_surface_response(
    normal: vec3<f32>,
    view: vec3<f32>,
    light: vec3<f32>,
    albedo: vec3<f32>,
) -> vec3<f32> {
    return eval_brdf(normal, view, light, albedo, fusion.shading) * PI;
}

// Uniform direction inside the solar disc around `wi`.
fn fusion_sample_sun(wi: vec3<f32>, st: ptr<function, u32>) -> vec3<f32> {
    let cos_max = fusion.sampling.x;
    if (cos_max >= 0.9999999) { return wi; }
    let u1 = xorshift32(st);
    let u2 = xorshift32(st);
    let cos_t = 1.0 - u1 * (1.0 - cos_max);
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let phi = 6.28318530718 * u2;
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(wi.y) > 0.99);
    let tx = normalize(cross(up, wi));
    let ty = cross(wi, tx);
    return normalize(tx * (sin_t * cos(phi)) + ty * (sin_t * sin(phi)) + wi * cos_t);
}

// ReSTIR candidate generation with a visibility-aware target function.
//
// The candidate is a direction y inside the solar disc. Its target is
//     p_hat(y) = floor + (1 - floor) * shadow_transmittance(x -> y)
// — the light-visibility term is the unified occlusion query, so resampling
// favours unoccluded parts of the disc across splats, LiDAR and terrain
// alike; the defensive floor keeps shadowed samples alive (their visibility
// of zero must stay reusable) and bounds the weight at 1 / floor. Candidates
// are streamed through the reservoir (weighted reservoir sampling). The
// selected sample's visibility is stored in sample.intensity as the
// last-known visibility that shading falls back to on a page miss.
//
// A candidate whose shadow ray reached a non-resident page is not counted:
// the temporal pass then keeps the history reservoir for this pixel.
fn fusion_restir_candidate(
    cand: ptr<function, RestirReservoir>,
    hit_point: vec3<f32>,
    n: vec3<f32>,
    albedo: vec3<f32>,
    wi: vec3<f32>,
    st: ptr<function, u32>,
) {
    let y = fusion_sample_sun(wi, st);
    let ndotl = max(dot(n, y), 0.0);
    if (terrain_luminance(albedo * lighting.light_color * ndotl) <= 0.0) { return; }
    var visibility = 1.0;
    if (lighting.shadows_enabled != 0u) {
        let before = fusion_miss_events;
        let eps = fusion.sampling.w;
        let sray = Ray(hit_point + n * eps, eps, y, 1e30);
        visibility = shadow_transmittance(sray) * fusion_media_transmittance(sray);
        if (fusion_miss_events != before) { return; }
    }
    let floor_pdf = fusion.sampling.y;
    let target_pdf = floor_pdf + (1.0 - floor_pdf) * visibility;
    (*cand).w_sum = (*cand).w_sum + target_pdf;
    (*cand).m = (*cand).m + 1u;
    if (xorshift32(st) * (*cand).w_sum <= target_pdf) {
        (*cand).sample.position = hit_point;
        (*cand).sample.light_index = 0u;
        (*cand).sample.direction = y;
        (*cand).sample.intensity = visibility;
        (*cand).sample.light_type = 1u;
        (*cand).target_pdf = target_pdf;
    }
}

// Re-express the finalized candidate under the canonical target (1) the
// temporal and spatial reuse passes merge with: W is unchanged, and
// w_sum = W * M * 1 keeps every reservoir in the chain on one target.
fn fusion_restir_publish(cand: ptr<function, RestirReservoir>) {
    if ((*cand).m > 0u && (*cand).weight > 0.0 && (*cand).target_pdf > 0.0) {
        (*cand).w_sum = (*cand).weight * f32((*cand).m);
        (*cand).target_pdf = 1.0;
    }
}

// Sun visibility for shading: the unified transmittance along the reservoir's
// direction. If the ray reached a page that is still being streamed in, the
// reservoir's last-known visibility stands in for this frame.
fn fusion_shading_visibility(sray: Ray, history_valid: bool, history_visibility: f32) -> f32 {
    let before = fusion_miss_events;
    var visibility = shadow_transmittance(sray) * fusion_media_transmittance(sray);
    if (fusion_miss_events != before && history_valid) {
        visibility = clamp(history_visibility, 0.0, 1.0);
    }
    return visibility;
}

// Fused diagnostic AOVs from the unjittered centre ray:
//   aov_direct   rgb = sun radiance reaching the surface
//   aov_indirect (T_total, T_splat, T_lidar, T_terrain) toward the sun centre
//   aov_emission (hit kind, N.L, page-miss flag, self-shadow bias)
fn fusion_write_aovs(coord: vec2<i32>, hit: HybridHitResult, albedo: vec3<f32>, is_hit: bool) {
    var parts = vec3<f32>(1.0);
    var ndotl = 0.0;
    var direct = vec3<f32>(0.0);
    var kind = 0.0;
    if (is_hit) {
        let wi = normalize(lighting.light_dir);
        ndotl = dot(hit.normal, wi);
        kind = f32(hit.hit_type);
        let eps = fusion.sampling.w;
        let sray = Ray(hit.point + hit.normal * eps, eps, wi, 1e30);
        parts = fusion_shadow_parts(sray, 1e30, true);
        let view = normalize(uniforms.cam_origin - hit.point);
        direct = fusion_surface_response(hit.normal, view, wi, albedo)
            * lighting.light_color * terrain_sun_transmit(wi.y) * max(ndotl, 0.0)
            * parts.x * parts.y * parts.z * fusion_media_transmittance(sray);
    }
    let missed = select(0.0, 1.0, fusion_miss_events != 0u);
    if (aov_enabled(AOV_DIRECT_BIT)) {
        textureStore(aov_direct, coord, vec4<f32>(direct, 1.0));
    }
    if (aov_enabled(AOV_INDIRECT_BIT)) {
        textureStore(aov_indirect, coord,
            vec4<f32>(parts.x * parts.y * parts.z, parts.x, parts.y, parts.z));
    }
    if (aov_enabled(AOV_EMISSION_BIT)) {
        textureStore(aov_emission, coord, vec4<f32>(kind, ndotl, missed, fusion_self_bias));
    }
}
