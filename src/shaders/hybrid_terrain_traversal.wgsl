// src/shaders/hybrid_terrain_traversal.wgsl
// PROMETHEUS: heightfield-native ray traversal for the hybrid path tracer.
// Implements a min-max quadtree DDA over the RG32Float pyramid built by
// terrain_heightfield.rs: descend from the coarsest mip, skip any node whose
// ray segment lies entirely above max_height or below min_height, refine into
// children only where the ray brackets the height band, and solve the exact
// ray/bilinear-patch intersection at leaf cells (the vertical deviation along
// the ray is exactly quadratic in t). Primary and sun-shadow rays reuse the
// identical descent. Also defines `main_terrain`, the accumulation-aware
// kernel entry used by HybridPathTracer::render_terrain_reference, and
// `main_terrain_gbuffer`, which feeds the pt_restir_spatial reuse pass.
// Concatenated into the hybrid kernel module by hybrid_compute/setup.rs; uses
// structs/bindings declared in hybrid_traversal.wgsl and hybrid_kernel.wgsl.
// RELEVANT FILES: src/path_tracing/hybrid_compute/terrain_heightfield.rs,
//                 src/path_tracing/hybrid_compute/render_terrain.rs

// Layout documented in terrain_heightfield.rs (six vec4 rows, 96 bytes).
struct TerrainPtUniforms {
    origin_spacing: vec4<f32>, // origin_x, origin_z, spacing_x, spacing_z
    h_params: vec4<f32>,       // h_min, h_max, exaggeration, env_intensity
    // terrain albedo rgb + turbidity excess (albedo_pad.w = tau - 1, so a
    // zeroed/default block means the clean reference atmosphere tau = 1).
    albedo_pad: vec4<f32>,
    dims: vec4<u32>,           // width_texels, height_texels, cell_w, cell_h
    // mip_count, flags (bit0 terrain enabled, bit1 per-texel albedo map at
    // group 2 binding 16, bit2 spectral ReSTIR), env_w, env_h.
    mips: vec4<u32>,
    extra: vec4<u32>,          // spp, stats readback cadence, unused, unused
}

struct EarthCurvatureUniforms {
    inv_two_r_prime: f32,
    _pad0: f32,
    ray_origin_geodetic: vec2<f32>,
    enabled: u32,
    _pad1: u32,
}

// Canonical ReSTIR DI structs — byte-compatible with the Rust `Reservoir` /
// `LightSample` in src/path_tracing/restir/types.rs and with the standalone
// pt_restir_temporal.wgsl / pt_restir_spatial.wgsl reuse passes that this
// path dispatches between accumulation frames (storage stride 80 bytes).
struct RestirLightSample {
    position: vec3<f32>,
    light_index: u32,
    direction: vec3<f32>,
    intensity: f32,
    light_type: u32,   // 0=point, 1=directional, 2=area
    params: vec3<f32>,
}

struct RestirReservoir {
    sample: RestirLightSample,
    w_sum: f32,
    m: u32,
    weight: f32,       // W = w_sum / (M * target_pdf)
    target_pdf: f32,
}

@group(2) @binding(1) var terrain_height_tex: texture_2d<f32>;
@group(2) @binding(2) var terrain_minmax_tex: texture_2d<f32>;
@group(2) @binding(3) var<uniform> terrain: TerrainPtUniforms;
// Per-pixel statistics record (48 bytes, matches `TerrainStatistics` in
// render_terrain.rs): x/y are the all-frames Welford mean and M2 of the
// per-frame mean luminance — the host converts M2/(N*(N-1)) into the
// estimated variance of the mean. `overflow` flags a u32 counter wrap;
// primary/shadow carry the LAST beauty frame's measured traversal counts in
// lanes (node_visits, minmax_loads, height_loads, rays), split by whether
// the trace ran under a shadow-ray query (`intersect_shadow_ray`).
struct TerrainStatistics {
    x: f32,
    y: f32,
    overflow: u32,
    trace_error: u32,
    primary: vec4<u32>,
    shadow: vec4<u32>,
}
@group(2) @binding(4) var<storage, read_write> terrain_welford: array<TerrainStatistics>;
// Fresh per-frame light candidates (input to the temporal reuse pass).
@group(2) @binding(5) var<storage, read_write> terrain_reservoirs_curr: array<RestirReservoir>;
@group(2) @binding(6) var terrain_env_tex: texture_2d<f32>;
// Reservoirs merged by last frame's temporal+spatial passes; shading reads
// them and M-clamps in place (standard ReSTIR history clamp).
@group(2) @binding(7) var<storage, read_write> terrain_reservoirs_prev: array<RestirReservoir>;
// ReSTIR G-buffer consumed by pt_restir_spatial.wgsl for target-pdf
// re-evaluation at the receiving pixel. Written once (static scene) by the
// main_terrain_gbuffer entry, which has its own pipeline layout.
@group(2) @binding(8) var<storage, read_write> terrain_gbuffer_nr: array<vec4<f32>>;
@group(2) @binding(9) var<storage, read_write> terrain_gbuffer_pos: array<vec4<f32>>;
@group(2) @binding(10) var<uniform> earth_curvature: EarthCurvatureUniforms;
// Optional per-texel linear-RGB terrain albedo at DEM texel resolution.
// Read by `terrain_albedo_at` only when `terrain.mips.y` flag bit 1 is set;
// otherwise the uniform `terrain.albedo_pad.rgb` is the albedo. Pipelines
// that never set the flag bind a 1x1 white dummy here.
@group(2) @binding(16) var terrain_albedo_tex: texture_2d<f32>;

const TERRAIN_STACK_SIZE: u32 = 64u;
const TERRAIN_PI: f32 = 3.14159265358979323846;
// ReSTIR history cap: prev reservoirs are rescaled to at most this M before
// each temporal merge so w_sum/M cannot blow up across hundreds of frames.
const TERRAIN_RESTIR_M_CAP: u32 = 512u;
// Upper bound on the finalized reservoir weight W.
//
// W = w_sum / (M * target_pdf), and target_pdf is proportional to N.L, so a
// GRAZING surface drives target_pdf to 0+ while w_sum still carries the value
// accumulated when the sample was bright. A fresh candidate always finalizes to
// exactly W = 1 (one directional light => stream weight w = target_pdf), so the
// blow-up only ever arrives down the TEMPORAL REUSE path.
//
// Measured: a Paris plate containing the Eiffel Tower - a near-vertical lattice
// under a 60 deg sun - produced W = 4.4e6, far outside the
// terrain_reservoirs_prev.weights buffer contract of [0, 65536], and the render
// aborted. Reliefs 3.0/4.0/6.0 tripped it while 5.0 happened not to, i.e. it
// presents as erratic rather than as a clean threshold.
//
// Clamping here is SHADING-NEUTRAL: the only two consumers of .weight are a
// `> 0.0` validity test and `clamp(prev_r.weight, 0.0, TERRAIN_RESTIR_W_CAP)`
// at the shading site, so no value above the cap could ever have reached the
// image. It only stops an unbounded number being written to a buffer whose
// declared range forbids it.
const TERRAIN_RESTIR_W_CAP: f32 = 4.0;

// --- Measured traversal counters -----------------------------------------
// Private (per-invocation, zero-initialized per WGSL) counters accumulated
// by terrain_trace / terrain_cell_heights. `main_terrain` snapshots them
// into terrain_welford[pix] BEFORE the AOV center ray, so the published
// counts cover only this invocation's beauty-frame rays — the unjittered
// G-buffer/center rays are traced by other dispatches or after the snapshot
// and are never published. Other entry points that call into the terrain
// descent (e.g. `main`, `main_terrain_gbuffer`) also tick these counters
// but never publish them.
var<private> terrain_primary_counts: vec4<u32>;
var<private> terrain_shadow_counts: vec4<u32>;
var<private> terrain_is_shadow: bool;
var<private> terrain_count_overflow: u32;

fn terrain_count(delta: vec4<u32>) {
    var before = terrain_primary_counts;
    if (terrain_is_shadow) { before = terrain_shadow_counts; }
    let after = before + delta;
    if (any(after < before)) { terrain_count_overflow = 1u; }
    if (terrain_is_shadow) { terrain_shadow_counts = after; }
    else { terrain_primary_counts = after; }
}

fn terrain_reservoir_weight(w_sum: f32, m: u32, target_pdf: f32) -> f32 {
    return clamp(w_sum / (f32(m) * target_pdf), 0.0, TERRAIN_RESTIR_W_CAP);
}

@compute @workgroup_size(8, 8, 1)
fn main_terrain_publish(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) { return; }
    let pix = gid.y * uniforms.width + gid.x;
    var r = terrain_reservoirs_prev[pix];
    if (r.m > TERRAIN_RESTIR_M_CAP) {
        let scale = f32(TERRAIN_RESTIR_M_CAP) / f32(r.m);
        r.w_sum = r.w_sum * scale;
        r.m = TERRAIN_RESTIR_M_CAP;
        if (r.target_pdf > 0.0) {
            r.weight = terrain_reservoir_weight(r.w_sum, r.m, r.target_pdf);
        }
        terrain_reservoirs_prev[pix] = r;
    }
}

fn terrain_enabled() -> bool {
    return (terrain.mips.y & 1u) != 0u;
}

// Safe reciprocal that avoids inf propagation for axis-parallel rays.
fn terrain_safe_inv(d: f32) -> f32 {
    let ad = max(abs(d), 1e-12);
    return select(1.0 / ad, -1.0 / ad, d < 0.0);
}

// Camera-relative formulation: t * direction.xz is the accumulated horizontal
// ray displacement, so d² never subtracts large world coordinates in f32.
fn terrain_curved_height(ray: Ray, t: f32, apply_curvature: bool) -> f32 {
    let horizontal_d2 = t * t * dot(ray.direction.xz, ray.direction.xz);
    let correction = select(
        0.0,
        horizontal_d2 * earth_curvature.inv_two_r_prime,
        apply_curvature && earth_curvature.enabled != 0u,
    );
    return ray.origin.y + t * ray.direction.y + correction;
}

// The corrected ray height is a convex quadratic. Endpoints bound its maximum;
// including the in-span vertex gives the exact minimum, hence node rejection
// is conservative without widening or duplicating the min-max pyramid.
fn terrain_curved_height_range(
    ray: Ray,
    t0: f32,
    t1: f32,
    apply_curvature: bool,
) -> vec2<f32> {
    let y0 = terrain_curved_height(ray, t0, apply_curvature);
    let y1 = terrain_curved_height(ray, t1, apply_curvature);
    var minimum = min(y0, y1);
    if (apply_curvature && earth_curvature.enabled != 0u) {
        let a = dot(ray.direction.xz, ray.direction.xz) * earth_curvature.inv_two_r_prime;
        if (a > 0.0) {
            let vertex = -ray.direction.y / (2.0 * a);
            if (vertex >= t0 && vertex <= t1) {
                minimum = min(minimum, terrain_curved_height(ray, vertex, true));
            }
        }
    }
    return vec2<f32>(minimum, max(y0, y1));
}

// Ray parameter span over the world-space xz rectangle of a node.
// Returns (t_enter, t_exit); empty if t_enter > t_exit.
fn terrain_slab_xz(ray: Ray, x0: f32, x1: f32, z0: f32, z1: f32) -> vec2<f32> {
    let inv_x = terrain_safe_inv(ray.direction.x);
    let inv_z = terrain_safe_inv(ray.direction.z);
    var tx0 = (x0 - ray.origin.x) * inv_x;
    var tx1 = (x1 - ray.origin.x) * inv_x;
    if (tx0 > tx1) { let tmp = tx0; tx0 = tx1; tx1 = tmp; }
    var tz0 = (z0 - ray.origin.z) * inv_z;
    var tz1 = (z1 - ray.origin.z) * inv_z;
    if (tz0 > tz1) { let tmp = tz0; tz0 = tz1; tz1 = tmp; }
    return vec2<f32>(max(tx0, tz0), min(tx1, tz1));
}

// Node (level, x, y) -> packed stack word. Supports DEMs up to 8192 cells.
fn terrain_pack_node(level: u32, x: u32, y: u32) -> u32 {
    return (level << 26u) | (y << 13u) | x;
}

// Exaggerated corner heights of DEM cell (cx, cz): h00, h10, h01, h11.
fn terrain_cell_heights(cx: u32, cz: u32) -> vec4<f32> {
    let ex = terrain.h_params.z;
    terrain_count(vec4<u32>(0u, 0u, 4u, 0u)); // the four height-texel loads below
    let h00 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx), i32(cz)), 0).r;
    let h10 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx + 1u), i32(cz)), 0).r;
    let h01 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx), i32(cz + 1u)), 0).r;
    let h11 = textureLoad(terrain_height_tex, vec2<i32>(i32(cx + 1u), i32(cz + 1u)), 0).r;
    return vec4<f32>(h00, h10, h01, h11) * ex;
}

struct TerrainLeafHit {
    t: f32,
    hit: bool,
}

// Exact ray vs bilinear-patch test inside cell (cx, cz) over ray span
// [t0, t1]. The vertical deviation d(t) = ray_y(t) - H(t) is exactly
// quadratic in t (H is bilinear, the ray footprint is linear), so fit the
// quadratic through d(t0), d(mid), d(t1) and take its smallest root in range.
fn terrain_leaf_intersect(
    ray: Ray,
    cx: u32,
    cz: u32,
    t0: f32,
    t1: f32,
    apply_curvature: bool,
    any_hit: bool,
) -> TerrainLeafHit {
    var out: TerrainLeafHit;
    out.hit = false;
    out.t = 1e30;
    let h = terrain_cell_heights(cx, cz);
    let sx = terrain.origin_spacing.z;
    let sz = terrain.origin_spacing.w;
    let ox = terrain.origin_spacing.x;
    let oz = terrain.origin_spacing.y;

    let tm = 0.5 * (t0 + t1);
    var d3: vec3<f32>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let t = select(select(t1, tm, i == 1u), t0, i == 0u);
        let px = ray.origin.x + t * ray.direction.x;
        let pz = ray.origin.z + t * ray.direction.z;
        let u = clamp((px - ox) / sx - f32(cx), 0.0, 1.0);
        let v = clamp((pz - oz) / sz - f32(cz), 0.0, 1.0);
        let hh = mix(mix(h.x, h.y, u), mix(h.z, h.w, u), v);
        d3[i] = terrain_curved_height(ray, t, apply_curvature) - hh;
    }

    // d(s) = a s^2 + b s + c on s in [0,1] with s = (t - t0)/(t1 - t0).
    let c = d3.x;
    let a = 2.0 * d3.z + 2.0 * d3.x - 4.0 * d3.y;
    let b = d3.z - d3.x - a;

    // A rounded shared-cell boundary can place this leaf's entry infinitesimally
    // below the continuous surface. That is already an any-hit intersection;
    // requiring another crossing inside the leaf would create a numeric crack.
    var s_hit = 1e30;
    if (any_hit && c <= 0.0) {
        s_hit = 0.0;
    } else if (abs(a) < 1e-12) {
        if (abs(b) > 1e-12) {
            let s = -c / b;
            if (s >= 0.0 && s <= 1.0) { s_hit = s; }
        }
    } else {
        let disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            // Numerically stable quadratic roots (Citardauq form for the
            // second root avoids cancellation).
            let sq = sqrt(disc);
            let q = -0.5 * (b + select(-sq, sq, b >= 0.0));
            var r0 = q / a;
            var r1 = select(c / q, 1e30, abs(q) < 1e-30);
            if (r0 > r1) { let tmp = r0; r0 = r1; r1 = tmp; }
            if (r0 >= 0.0 && r0 <= 1.0) { s_hit = r0; }
            else if (r1 >= 0.0 && r1 <= 1.0) { s_hit = r1; }
        }
    }
    if (s_hit <= 1.0) {
        let t = t0 + s_hit * (t1 - t0);
        if (t > ray.tmin && t < ray.tmax) {
            out.hit = true;
            out.t = t;
        }
    }
    return out;
}

// Geometric normal from the analytic bilinear gradient at world point p in
// cell (cx, cz) — the same surface the leaf test intersected.
fn terrain_normal_at(p: vec3<f32>, cx: u32, cz: u32) -> vec3<f32> {
    let h = terrain_cell_heights(cx, cz);
    let sx = terrain.origin_spacing.z;
    let sz = terrain.origin_spacing.w;
    let u = clamp((p.x - terrain.origin_spacing.x) / sx - f32(cx), 0.0, 1.0);
    let v = clamp((p.z - terrain.origin_spacing.y) / sz - f32(cz), 0.0, 1.0);
    let dh_du = mix(h.y - h.x, h.w - h.z, v);
    let dh_dv = mix(h.z - h.x, h.w - h.y, u);
    return normalize(vec3<f32>(-dh_du / sx, 1.0, -dh_dv / sz));
}

// Min-max quadtree DDA. `any_hit` controls early exit and conservatively closes
// rounded shared-leaf entry cracks. Curvature is an explicit sun-ray policy
// because arbitrary IBL directions must not reuse the azimuth-specific
// effective radius uploaded for the Sun.
fn terrain_trace(ray: Ray, any_hit: bool, apply_curvature: bool) -> HybridHitResult {
    var res: HybridHitResult;
    res.hit = 0u;
    res.t = ray.tmax;
    res.hit_type = 3u; // terrain
    if (!terrain_enabled()) { return res; }
    terrain_count(vec4<u32>(0u, 0u, 0u, 1u));

    let cell_w = terrain.dims.z;
    let cell_h = terrain.dims.w;
    let ox = terrain.origin_spacing.x;
    let oz = terrain.origin_spacing.y;
    let sx = terrain.origin_spacing.z;
    let sz = terrain.origin_spacing.w;

    var stack: array<u32, TERRAIN_STACK_SIZE>;
    var sp = 0u;
    stack[sp] = terrain_pack_node(terrain.mips.x - 1u, 0u, 0u);
    sp = sp + 1u;

    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = stack[sp];
        terrain_count(vec4<u32>(1u, 0u, 0u, 0u)); // one visit per popped node
        let level = node >> 26u;
        let ny = (node >> 13u) & 0x1FFFu;
        let nx = node & 0x1FFFu;

        // Cell range covered by this node (clamped at ragged edges).
        let cx0 = nx << level;
        let cz0 = ny << level;
        if (cx0 >= cell_w || cz0 >= cell_h) { continue; }
        let cx1 = min((nx + 1u) << level, cell_w);
        let cz1 = min((ny + 1u) << level, cell_h);

        let span = terrain_slab_xz(
            ray,
            ox + f32(cx0) * sx,
            ox + f32(cx1) * sx,
            oz + f32(cz0) * sz,
            oz + f32(cz1) * sz,
        );
        let t_lo = max(span.x, ray.tmin);
        let t_hi = min(span.y, min(ray.tmax, res.t));
        if (t_lo > t_hi) { continue; }

        // Height band test: skip when the ray segment stays entirely above
        // max or below min over this node's footprint.
        terrain_count(vec4<u32>(0u, 1u, 0u, 0u));
        let mm = textureLoad(terrain_minmax_tex, vec2<i32>(i32(nx), i32(ny)), i32(level)).rg
            * terrain.h_params.z;
        let ray_height = terrain_curved_height_range(ray, t_lo, t_hi, apply_curvature);
        if (ray_height.x > mm.y || ray_height.y < mm.x) { continue; }

        if (level == 0u) {
            let leaf = terrain_leaf_intersect(ray, cx0, cz0, t_lo, t_hi, apply_curvature, any_hit);
            if (leaf.hit && leaf.t < res.t) {
                res.hit = 1u;
                res.t = leaf.t;
                res.point = ray.origin + ray.direction * leaf.t;
                res.normal = terrain_normal_at(res.point, cx0, cz0);
                res.material_id = 0u;
                res.hit_type = 3u;
                if (any_hit) { return res; }
            }
            continue;
        }

        // Push the (up to) four children ordered far-to-near by t_enter so
        // the nearest is popped first; empty children are skipped.
        let child_level = level - 1u;
        var child_t: array<f32, 4u>;
        var child_id: array<u32, 4u>;
        var child_count = 0u;
        for (var cy = 0u; cy < 2u; cy = cy + 1u) {
            for (var cxi = 0u; cxi < 2u; cxi = cxi + 1u) {
                let ccx = nx * 2u + cxi;
                let ccy = ny * 2u + cy;
                let gx0 = ccx << child_level;
                let gz0 = ccy << child_level;
                if (gx0 >= cell_w || gz0 >= cell_h) { continue; }
                let gx1 = min((ccx + 1u) << child_level, cell_w);
                let gz1 = min((ccy + 1u) << child_level, cell_h);
                let cs = terrain_slab_xz(
                    ray,
                    ox + f32(gx0) * sx,
                    ox + f32(gx1) * sx,
                    oz + f32(gz0) * sz,
                    oz + f32(gz1) * sz,
                );
                let ct_lo = max(cs.x, t_lo);
                let ct_hi = min(cs.y, t_hi);
                if (ct_lo > ct_hi) { continue; }
                child_t[child_count] = ct_lo;
                child_id[child_count] = terrain_pack_node(child_level, ccx, ccy);
                child_count = child_count + 1u;
            }
        }
        // Insertion sort (descending t_enter) then push: nearest ends on top.
        for (var i = 1u; i < child_count; i = i + 1u) {
            let kt = child_t[i];
            let kid = child_id[i];
            var j = i;
            loop {
                if (j == 0u || child_t[j - 1u] >= kt) { break; }
                child_t[j] = child_t[j - 1u];
                child_id[j] = child_id[j - 1u];
                j = j - 1u;
            }
            child_t[j] = kt;
            child_id[j] = kid;
        }
        for (var i = 0u; i < child_count; i = i + 1u) {
            if (sp < TERRAIN_STACK_SIZE) {
                stack[sp] = child_id[i];
                sp = sp + 1u;
            }
        }
    }
    return res;
}

fn terrain_intersect(ray: Ray) -> HybridHitResult {
    return terrain_trace(ray, false, false);
}

fn terrain_occluded(ray: Ray, max_distance: f32) -> bool {
    var r = ray;
    r.tmax = min(ray.tmax, max_distance);
    let hit = terrain_trace(r, true, true);
    return hit.hit != 0u;
}

// ---------------------------------------------------------------------------
// Environment (IBL) term
// ---------------------------------------------------------------------------
// Equirect env map lookup by direction (nearest texel — deterministic). When
// no map is bound (env_w == 0) the fallback is a constant white environment
// scaled by env_intensity: still routed through this same function so there
// is no silent divergence between the two configurations.
fn terrain_env_radiance(dir: vec3<f32>) -> vec3<f32> {
    let intensity = terrain.h_params.w;
    let ew = terrain.mips.z;
    let eh = terrain.mips.w;
    if (ew == 0u || eh == 0u) {
        return vec3<f32>(intensity);
    }
    let d = normalize(dir);
    let uu = (atan2(d.z, d.x) / (2.0 * TERRAIN_PI)) + 0.5;
    let vv = acos(clamp(d.y, -1.0, 1.0)) / TERRAIN_PI;
    let px = min(u32(uu * f32(ew)), ew - 1u);
    let py = min(u32(vv * f32(eh)), eh - 1u);
    return textureLoad(terrain_env_tex, vec2<i32>(i32(px), i32(py)), 0).rgb * intensity;
}

// ---------------------------------------------------------------------------
// Per-texel albedo + shared atmosphere model
// ---------------------------------------------------------------------------
// Bilinear albedo at DEM texel resolution. Same (origin, spacing, dims)
// texel grid as the height pyramid: world xz -> texel coordinate. Bit 1 of
// terrain.mips.y selects the map; bit 0 only gates terrain visibility.
// `terrain_albedo_taps_at` additionally returns the four texel ids and their
// bilinear weights — the reverse pass scatters dL/d(albedo) into exactly
// those texels with exactly these weights, so the gradient is the adjoint of
// the same bilinear operator the primal evaluates.
struct TerrainAlbedoTaps {
    alb: vec3<f32>,
    texels: vec4<u32>,   // row-major texel ids; 0xffffffff when the map is off
    weights: vec4<f32>,  // w00, w10, w01, w11 matching `texels`
}
fn terrain_albedo_taps_at(p: vec3<f32>) -> TerrainAlbedoTaps {
    var t: TerrainAlbedoTaps;
    t.texels = vec4<u32>(0xffffffffu);
    t.weights = vec4<f32>(0.0);
    if ((terrain.mips.y & 2u) == 0u) {
        t.alb = terrain.albedo_pad.rgb;
        return t;
    }
    let aw = terrain.dims.x;
    let ah = terrain.dims.y;
    let tx = clamp((p.x - terrain.origin_spacing.x) / terrain.origin_spacing.z,
        0.0, f32(aw) - 1.0);
    let tz = clamp((p.z - terrain.origin_spacing.y) / terrain.origin_spacing.w,
        0.0, f32(ah) - 1.0);
    let x0 = u32(floor(tx));
    let z0 = u32(floor(tz));
    let x1 = min(x0 + 1u, aw - 1u);
    let z1 = min(z0 + 1u, ah - 1u);
    let fx = tx - f32(x0);
    let fz = tz - f32(z0);
    let c00 = textureLoad(terrain_albedo_tex, vec2<i32>(i32(x0), i32(z0)), 0).rgb;
    let c10 = textureLoad(terrain_albedo_tex, vec2<i32>(i32(x1), i32(z0)), 0).rgb;
    let c01 = textureLoad(terrain_albedo_tex, vec2<i32>(i32(x0), i32(z1)), 0).rgb;
    let c11 = textureLoad(terrain_albedo_tex, vec2<i32>(i32(x1), i32(z1)), 0).rgb;
    t.alb = mix(mix(c00, c10, fx), mix(c01, c11, fx), fz);
    t.texels = vec4<u32>(z0 * aw + x0, z0 * aw + x1, z1 * aw + x0, z1 * aw + x1);
    t.weights = vec4<f32>(
        (1.0 - fx) * (1.0 - fz), fx * (1.0 - fz),
        (1.0 - fx) * fz, fx * fz);
    return t;
}
fn terrain_albedo_at(p: vec3<f32>) -> vec3<f32> {
    return terrain_albedo_taps_at(p).alb;
}

// Minimal aerosol-turbidity atmosphere, shared by every terrain path that
// consumes these uniforms (the accumulation kernel, its G-buffer twin, and
// any downstream pass reusing this module):
//   * direct sun is attenuated by a spectral Beer extinction
//         T_c = exp(-beta_c * (tau - 1) * air_mass),
//     with air_mass = 1/sin(elevation) clamped at the horizon. Blue light is
//     extincted more strongly than red, so rising turbidity both dims and
//     warms the direct beam — the colour separation is what lets a renderer
//     tell "dimmer sun" apart from "darker ground".
//   * the environment is scaled by a diffuse in-scatter gain
//         h = 1 + K_HAZE * (tau - 1),
//     representing extincted direct light returning as skylight.
// tau lives in albedo_pad.w as (tau - 1); a zeroed block is the clean-air
// reference tau = 1 under which both terms are exactly identity.
const TERRAIN_ATMOS_BETA: vec3<f32> = vec3<f32>(0.10, 0.16, 0.26);
const TERRAIN_ATMOS_HAZE: f32 = 0.12;

fn terrain_turbidity() -> f32 {
    return max(terrain.albedo_pad.w, 0.0) + 1.0;
}

fn terrain_sun_transmit(sun_wy: f32) -> vec3<f32> {
    let air_mass = 1.0 / max(sun_wy, 0.05);
    return exp(-TERRAIN_ATMOS_BETA * (terrain_turbidity() - 1.0) * air_mass);
}

fn terrain_env_haze() -> f32 {
    return 1.0 + TERRAIN_ATMOS_HAZE * (terrain_turbidity() - 1.0);
}

// Environment radiance after in-scatter — the value the shading equations
// actually see (miss-ray background and cosine-lobe IBL alike).
fn terrain_env_effective(dir: vec3<f32>) -> vec3<f32> {
    return terrain_env_radiance(dir) * terrain_env_haze();
}

fn terrain_spectral_restir_enabled() -> bool {
    return (terrain.mips.y & 4u) != 0u;
}

fn terrain_spectral_channel_mask(channel: u32) -> vec3<f32> {
    if (channel == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
    if (channel == 1u) { return vec3<f32>(0.0, 1.0, 0.0); }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn terrain_spectral_probabilities() -> vec3<f32> {
    let spectrum = max(
        lighting.light_color * terrain_sun_transmit(normalize(lighting.light_dir).y),
        vec3<f32>(0.0));
    let baseline = min(spectrum.x, min(spectrum.y, spectrum.z));
    let spectral_target = vec3<f32>(0.2126, 0.7152, 0.0722)
        * max(spectrum - vec3<f32>(baseline), vec3<f32>(0.0));
    let target_sum = spectral_target.x + spectral_target.y + spectral_target.z;
    var normalized = vec3<f32>(1.0 / 3.0);
    if (target_sum > 0.0) {
        normalized = spectral_target / target_sum;
    }
    let floor_probability = 1.0 / TERRAIN_RESTIR_W_CAP;
    let focused_mass = 1.0 - 3.0 * floor_probability;
    return vec3<f32>(floor_probability) + focused_mass * normalized;
}

fn terrain_sample_spectral_channel(probabilities: vec3<f32>, u: f32) -> u32 {
    if (u < probabilities.x) { return 0u; }
    if (u < probabilities.x + probabilities.y) { return 1u; }
    return 2u;
}

// Zero-mean tent sample in [-1, 1] (inverse-CDF; the kernel-local
// tent_filter in hybrid_kernel.wgsl returns the PDF value, not a sample).
fn terrain_tent_offset(u: f32) -> f32 {
    if (u < 0.5) {
        return sqrt(2.0 * u) - 1.0;
    }
    return 1.0 - sqrt(2.0 * (1.0 - u));
}

fn terrain_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Cosine-weighted hemisphere direction about n.
fn terrain_cosine_dir(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bt = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    let r = sqrt(u1);
    let phi = 2.0 * TERRAIN_PI * u2;
    let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    return normalize(local.x * t + local.y * bt + local.z * n);
}

// Strong avalanche mixer for the per-(seed, pixel, frame) stream seed: the
// raw xor mix let weak xorshift top bits correlate across seeds/pixels.
fn terrain_seed_hash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

// ---------------------------------------------------------------------------
// Accumulating terrain-reference kernel entry
// ---------------------------------------------------------------------------
// Per frame: `spp` jittered camera samples averaged into accum_hdr, canonical
// ReSTIR candidate generation into terrain_reservoirs_curr (merged afterwards
// by the pt_restir_temporal + pt_restir_spatial passes the driver dispatches),
// sun shading gated through the merged reservoir from the previous frame's
// reuse chain, an all-frames Welford update over the per-frame mean
// luminance (the host converts M2/(N*(N-1)) into the estimated variance of
// the mean — the convergence metric — and reads this buffer every
// `terrain.extra.y` frames as its cadence), a snapshot of this frame's
// measured traversal counters, and the tonemapped running mean written to
// out_tex. AOVs are written from an UNJITTERED center ray when aov_flags is
// set (the driver sets it on frame 0 only) so geometric AOVs match
// rasterizer pixel-center sampling.
@compute @workgroup_size(8, 8, 1)
fn main_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = uniforms.width;
    let H = uniforms.height;
    if (gid.x >= W || gid.y >= H) { return; }
    let pix = gid.y * W + gid.x;

    // --- ReSTIR history M-clamp + fetch the merged reservoir for shading ---
    var prev_r = terrain_reservoirs_prev[pix];
    if (prev_r.m > TERRAIN_RESTIR_M_CAP) {
        let scale = f32(TERRAIN_RESTIR_M_CAP) / f32(prev_r.m);
        prev_r.w_sum = prev_r.w_sum * scale;
        prev_r.m = TERRAIN_RESTIR_M_CAP;
        if (prev_r.target_pdf > 0.0) {
            prev_r.weight = terrain_reservoir_weight(prev_r.w_sum, prev_r.m, prev_r.target_pdf);
        }
        terrain_reservoirs_prev[pix] = prev_r;
    }
    let prev_valid = uniforms.frame_index > 0u && prev_r.m > 0u
        && prev_r.weight > 0.0 && prev_r.target_pdf > 0.0
        && prev_r.sample.light_type == 1u;

    // Independently seeded pseudorandom camera/env estimates: avalanche-hash
    // the (seed, pixel, frame) tuple so distinct seeds/pixels decorrelate.
    var st = terrain_seed_hash(uniforms.seed_hi ^ uniforms.seed_lo
        ^ terrain_seed_hash(pix) ^ terrain_seed_hash(uniforms.frame_index + 1u));
    st = select(st, 0x6d2b79f5u, st == 0u);
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let spp = max(terrain.extra.x, 1u);
    let wi = normalize(lighting.light_dir);
    let spectral_restir = terrain_spectral_restir_enabled();

    var frame_radiance = vec3<f32>(0.0);
    var cand: RestirReservoir; // zero-initialized: m=0 marks "no candidate"
    if (spectral_restir) {
        let probabilities = terrain_spectral_probabilities();
        var reservoir_st = st;
        let channel = terrain_sample_spectral_channel(probabilities, xorshift32(&reservoir_st));
        cand.sample.position = vec3<f32>(0.0);
        cand.sample.light_index = channel;
        cand.sample.direction = wi;
        cand.sample.intensity = 1.0;
        cand.sample.light_type = 1u;
        cand.sample.params = probabilities;
        cand.w_sum = 1.0;
        cand.m = 1u;
        cand.target_pdf = probabilities[channel];
        cand.weight = 1.0 / probabilities[channel];
    }

    for (var s = 0u; s < spp; s = s + 1u) {
        let jx = terrain_tent_offset(xorshift32(&st)) * 0.5;
        let jy = terrain_tent_offset(xorshift32(&st)) * 0.5;

        // Jittered beauty ray.
        let ndc_x = ((f32(gid.x) + 0.5 + jx) / f32(W)) * 2.0 - 1.0;
        let ndc_y = (1.0 - (f32(gid.y) + 0.5 + jy) / f32(H)) * 2.0 - 1.0;
        var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
        rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
        let ray = Ray(uniforms.cam_origin, 1e-3, rd, 1e30);

        let hit = intersect_hybrid(ray);
        if (hit.hit == 0u) {
            frame_radiance = frame_radiance + terrain_env_effective(rd);
            continue;
        }
        let n = hit.normal;
        let albedo = get_surface_properties(hit);

        // --- Sun candidate generation. Mapped terrain uses the spectral
        // residual reservoir initialized above; the legacy uniform-albedo
        // path keeps its single delta-light candidate unchanged. ---
        if (!spectral_restir) {
            let ndotl = max(dot(n, wi), 0.0);
            let target_pdf = select(0.0, 1.0, terrain_luminance(albedo * lighting.light_color * ndotl) > 0.0);
            if (target_pdf > 0.0) {
                cand.sample.position = hit.point;
                cand.sample.light_index = 0u;
                cand.sample.direction = wi;
                cand.sample.intensity = terrain_luminance(lighting.light_color);
                cand.sample.light_type = 1u;
                cand.w_sum = cand.w_sum + target_pdf;
                cand.m = cand.m + 1u;
                cand.target_pdf = target_pdf;
            }
        }

        // --- Sun shading through the merged reservoir (temporal + spatial
        // reuse) from the previous frame; frame 0 falls back to the fresh
        // candidate, which is the identical delta sample with W = 1. ---
        var sun_dir = wi;
        var reuse_w = 1.0;
        var selected_channel = 3u;
        if (prev_valid) {
            sun_dir = normalize(prev_r.sample.direction);
            reuse_w = clamp(prev_r.weight, 0.0, TERRAIN_RESTIR_W_CAP);
            selected_channel = prev_r.sample.light_index;
        }
        var sun = vec3<f32>(0.0);
        let nd = max(dot(n, sun_dir), 0.0);
        if (nd > 0.0) {
            let sray = Ray(hit.point + n * 1e-3, 1e-3, sun_dir, 1e30);
            var vis = 1.0;
            if (lighting.shadows_enabled != 0u && intersect_shadow_ray(sray, 1e30)) {
                vis = 0.0;
            }
            // Spectral aerosol extinction on the direct beam (tau-1 in
            // albedo_pad.w; identity at the clean-air reference tau = 1).
            let spectrum = lighting.light_color * terrain_sun_transmit(sun_dir.y);
            var sampled_spectrum = spectrum * reuse_w;
            if (spectral_restir) {
                sampled_spectrum = spectrum;
                if (prev_valid && selected_channel < 3u) {
                    let baseline = min(spectrum.x, min(spectrum.y, spectrum.z));
                    let residual = max(spectrum - vec3<f32>(baseline), vec3<f32>(0.0));
                    sampled_spectrum = vec3<f32>(baseline)
                        + terrain_spectral_channel_mask(selected_channel) * residual * reuse_w;
                }
            }
            sun = albedo * sampled_spectrum * nd * vis;
        }

        // --- IBL: one cosine-weighted env sample per camera sample
        // (converges by accumulation); single-sample estimator of the
        // Lambert env integral (pdf = cos/pi) is albedo * env(wi) * V. ---
        let u1 = xorshift32(&st);
        let u2 = xorshift32(&st);
        let ei = terrain_cosine_dir(n, u1, u2);
        let eray = Ray(hit.point + n * 1e-3, 1e-3, ei, 1e30);
        var env_vis = 1.0;
        if (intersect_ibl_occlusion_ray(eray, 1e30)) {
            env_vis = 0.0;
        }
        let ibl = albedo * terrain_env_effective(ei) * env_vis;

        frame_radiance = frame_radiance + sun + ibl;
    }
    frame_radiance = frame_radiance / f32(spp);

    // Finalize + publish this frame's candidate reservoir for the reuse chain.
    if (cand.m > 0u && cand.w_sum > 0.0 && cand.target_pdf > 0.0) {
        cand.weight = terrain_reservoir_weight(cand.w_sum, cand.m, cand.target_pdf);
    }
    terrain_reservoirs_curr[pix] = cand;

    // --- Accumulate the per-frame mean radiance ---
    let prev = accum_hdr[pix];
    let acc = vec4<f32>(prev.rgb + frame_radiance, prev.a + 1.0);
    accum_hdr[pix] = acc;

    // --- All-frames Welford over the PER-FRAME mean luminance: wf.x/wf.y
    // are the running mean and M2 of `frame_radiance` luminance across every
    // accumulated frame, so the host's M2/(N*(N-1)) is the estimated
    // variance of the mean estimator (scalar convergence metric, not a
    // confidence bound on transport accuracy). Mapped terrain adds a seeded
    // spectral-residual reservoir draw to the pseudorandom camera/env
    // estimate; this does NOT promise provable independence under future
    // multi-light reuse. The counter snapshot
    // is taken BEFORE the AOV center-ray block below, so published counts
    // cover only this invocation's beauty-frame rays. ---
    var wf = terrain_welford[pix];
    let sample_lum = terrain_luminance(frame_radiance);
    let k = f32(uniforms.frame_index) + 1.0;
    let delta = sample_lum - wf.x;
    let mean = wf.x + delta / k;
    wf.y = wf.y + delta * (sample_lum - mean);
    wf.x = mean;
    wf.overflow = terrain_count_overflow;
    wf.trace_error = wf.trace_error | hybrid_sdf_error;
    wf.primary = terrain_primary_counts;
    wf.shadow = terrain_shadow_counts;
    terrain_welford[pix] = wf;

    // --- Resolve running mean to the output image ---
    let mean_rgb = acc.rgb / acc.a;
    let ldr = reinhard_tonemap(mean_rgb, uniforms.cam_exposure);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(ldr, 1.0));

    // --- Geometric AOVs from the unjittered center ray (frame 0 only via
    // aov_flags) so they align with rasterizer pixel-center sampling ---
    if (uniforms.aov_flags != 0u) {
        let cx = ((f32(gid.x) + 0.5) / f32(W)) * 2.0 - 1.0;
        let cy = (1.0 - (f32(gid.y) + 0.5) / f32(H)) * 2.0 - 1.0;
        var crd = normalize(vec3<f32>(cx * half_w, cy * half_h, -1.0));
        crd = normalize(crd.x * uniforms.cam_right + crd.y * uniforms.cam_up + crd.z * (-uniforms.cam_forward));
        let cray = Ray(uniforms.cam_origin, 1e-3, crd, 1e30);
        let chit = intersect_hybrid(cray);
        let is_hit = chit.hit != 0u;
        let calbedo = get_surface_properties(chit);
        let coord = vec2<i32>(i32(gid.x), i32(gid.y));
        if (aov_enabled(AOV_ALBEDO_BIT)) {
            textureStore(aov_albedo, coord,
                select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(calbedo, 1.0), is_hit));
        }
        if (aov_enabled(AOV_NORMAL_BIT)) {
            textureStore(aov_normal, coord,
                select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(chit.normal, 1.0), is_hit));
        }
        if (aov_enabled(AOV_DEPTH_BIT)) {
            let depth_val: f32 = select(bitcast<f32>(0x7fc00000u), chit.t, is_hit);
            textureStore(aov_depth, coord, vec4<f32>(depth_val, 0.0, 0.0, 0.0));
        }
        if (aov_enabled(AOV_VISIBILITY_BIT)) {
            textureStore(aov_visibility, coord, vec4<f32>(select(0.0, 1.0, is_hit), 0.0, 0.0, 1.0));
        }
    }
    // The AOV center ray also marches the compiled SDF; accumulate so a
    // failure on any earlier frame survives until the host polls.
    terrain_welford[pix].trace_error = terrain_welford[pix].trace_error | hybrid_sdf_error;
}

// ---------------------------------------------------------------------------
// ReSTIR G-buffer entry (own pipeline layout — see hybrid_compute/setup.rs)
// ---------------------------------------------------------------------------
// Writes the per-pixel surface record (world normal + roughness, world
// position) that pt_restir_spatial.wgsl re-evaluates target pdfs against.
// Camera and scene are static across the accumulation, so the driver runs
// this once before the frame loop, from the unjittered center ray.
@compute @workgroup_size(8, 8, 1)
fn main_terrain_gbuffer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = uniforms.width;
    let H = uniforms.height;
    if (gid.x >= W || gid.y >= H) { return; }
    let pix = gid.y * W + gid.x;

    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let ndc_x = ((f32(gid.x) + 0.5) / f32(W)) * 2.0 - 1.0;
    let ndc_y = (1.0 - (f32(gid.y) + 0.5) / f32(H)) * 2.0 - 1.0;
    var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
    rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
    let ray = Ray(uniforms.cam_origin, 1e-3, rd, 1e30);

    let hit = intersect_hybrid(ray);
    if (hit.hit != 0u) {
        terrain_gbuffer_nr[pix] = vec4<f32>(hit.normal, 1.0);
        terrain_gbuffer_pos[pix] = vec4<f32>(hit.point, 1.0);
    } else {
        // Sky pixels: shading never consults their reservoirs; keep the
        // record finite for the spatial pass's normalize().
        terrain_gbuffer_nr[pix] = vec4<f32>(0.0, 0.0, 1.0, 1.0);
        terrain_gbuffer_pos[pix] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    // This pipeline's bind group cannot reach terrain_welford, so an SDF
    // march failure on the center ray is signalled through gbuffer_pos.w and
    // rejected by the host readback.
    if (hybrid_sdf_error != 0u) {
        terrain_gbuffer_pos[pix].w = -1.0;
    }
}
