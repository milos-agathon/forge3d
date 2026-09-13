// src/shaders/hybrid_traversal.wgsl
// Hybrid traversal combining mesh BVH traversal with legacy hooks for SDF.

// GPU hybrid traversal covers mesh BVH, terrain heightfield, and SDF scenes.
// An SDF scene is compiled to constant WGSL by hybrid_compute/sdf_scene.rs and
// specialized into the declarations below before pipeline creation; the
// assembled default is an empty, disabled SDF.

// Hybrid scene data structures
struct HybridUniforms {
    sdf_primitive_count: u32,
    sdf_node_count: u32,
    mesh_vertex_count: u32,
    mesh_index_count: u32,
    mesh_bvh_node_count: u32,
    traversal_mode: u32, // 0 = hybrid, 1 = SDF only, 2 = mesh only, 3 = terrain only
    _pad: vec2u,
}

struct HybridHitResult {
    t: f32,
    point: vec3f,
    normal: vec3f,
    material_id: u32,
    hit_type: u32, // 0 = mesh, 1 = SDF, 3 = terrain
    hit: u32, // 0 = false, 1 = true
    _pad: vec2u,
}

struct Ray {
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
}

// --- Compiled-constant SDF scene --------------------------------------------
// The default assembled kernel ships an empty SDF: HYBRID_SDF_ENABLED=false
// and hybrid_sdf_evaluate always misses. hybrid_compute/sdf_scene.rs replaces
// these exact declarations (and only these) with validated scene constants
// before pipeline creation.
const HYBRID_SDF_ENABLED: bool = false;
const HYBRID_SDF_MIN: vec3<f32> = vec3<f32>(0.0);
const HYBRID_SDF_MAX: vec3<f32> = vec3<f32>(0.0);
fn hybrid_sdf_evaluate(point: vec3<f32>) -> CsgResult {
    return CsgResult(1e30, 0u);
}
// Set when the SDF march produces a non-finite value, stagnates, or exhausts
// its step budget inside the slab. Surfaced to the host through
// TerrainStatistics.trace_error (and terrain_gbuffer_pos.w in the G-buffer
// pipeline, which cannot reach binding 4); never a silent miss.
var<private> hybrid_sdf_error: u32;
fn hybrid_sdf_finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
// March epsilon inherited from raymarch_sdf in src/sdf/hybrid.rs. The step
// budget is geometric per ray, not a fixed count: a non-hit advances by
// |d| > EPS, and for finite round-to-nearest f32 addition that makes
// progress the actual increment is at least half the requested increment,
// so 2 * clipped_length / EPS + 2 bounds the iterations.
const SDF_MARCH_EPS: f32 = 0.001;

// BVH structures (matching existing pt_kernel.wgsl)
struct BvhNode {
    aabb_min: vec3f,
    left: u32,
    aabb_max: vec3f,
    right: u32,
    flags: u32,
    _pad: u32,
}

struct MeshVertex {
    position: vec3f,
    _pad: f32,
}

// Bind groups for hybrid traversal
// NOTE: Consolidated into group(1) to stay within max_bind_groups=4
@group(1) @binding(1) var<uniform> hybrid_uniforms: HybridUniforms;
@group(1) @binding(2) var<storage, read> mesh_vertices: array<MeshVertex>;
@group(1) @binding(3) var<storage, read> mesh_indices: array<u32>;
@group(1) @binding(4) var<storage, read> mesh_bvh_nodes: array<BvhNode>;

// Ray-AABB intersection for BVH traversal
fn ray_aabb_intersect(ray: Ray, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    var tmin = ray.tmin;
    var tmax = ray.tmax;

    for (var i = 0u; i < 3u; i = i + 1u) {
        let inv_dir = 1.0 / ray.direction[i];
        var t0 = (aabb_min[i] - ray.origin[i]) * inv_dir;
        var t1 = (aabb_max[i] - ray.origin[i]) * inv_dir;

        if (inv_dir < 0.0) {
            let temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tmin = max(tmin, t0);
        tmax = min(tmax, t1);

        if (tmin > tmax) {
            return false;
        }
    }

    return true;
}

// Ray-triangle intersection
fn ray_triangle_intersect(
    ray: Ray,
    v0: vec3f,
    v1: vec3f,
    v2: vec3f
) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);

    let epsilon = 1e-7;
    if (abs(a) < epsilon) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(edge2, q);
    if (t > ray.tmin && t < ray.tmax) {
        let normal = normalize(cross(edge1, edge2));
        result.hit = 1u;
        result.t = t;
        result.point = ray.origin + ray.direction * t;
        result.normal = normal;
        result.material_id = 0u; // Default mesh material
        result.hit_type = 0u; // mesh
    }

    return result;
}

// BVH traversal for mesh intersection
const MAX_BVH_STACK_SIZE: u32 = 32u;

fn intersect_mesh(ray: Ray) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh

    // Brute-force triangle sweep. This keeps the shader simple and guarantees
    // we shade meshes even when no GPU BVH data is available.
    let index_count = hybrid_uniforms.mesh_index_count;
    if (index_count < 3u) {
        return result;
    }

    for (var tri = 0u; tri + 2u < index_count; tri = tri + 3u) {
        let i0 = mesh_indices[tri];
        let i1 = mesh_indices[tri + 1u];
        let i2 = mesh_indices[tri + 2u];

        if (i0 >= hybrid_uniforms.mesh_vertex_count ||
            i1 >= hybrid_uniforms.mesh_vertex_count ||
            i2 >= hybrid_uniforms.mesh_vertex_count) {
            continue;
        }

        let v0 = mesh_vertices[i0].position;
        let v1 = mesh_vertices[i1].position;
        let v2 = mesh_vertices[i2].position;

        let tri_hit = ray_triangle_intersect(ray, v0, v1, v2);
        if (tri_hit.hit != 0u && tri_hit.t < result.t) {
            result = tri_hit;
        }
    }

    return result;
}

// SDF raymarch against the compiled-constant scene. Ported from
// raymarch_sdf in src/sdf/hybrid.rs: MAX_STEPS=128, surface epsilon 0.001,
// abs-distance stepping, central-difference normal at the same epsilon.
// Unlike the CPU version the march is clipped to the validated scene bound
// [HYBRID_SDF_MIN, HYBRID_SDF_MAX] instead of a fixed 100-unit cap. Every
// numerical failure sets hybrid_sdf_error so the host can reject the frame;
// a true miss is reported honestly.
fn intersect_sdf(ray: Ray) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 1u;
    if (!HYBRID_SDF_ENABLED) {
        return result;
    }

    // Slab clip of [tmin, tmax] against the scene bound. A zero direction
    // component means the ray is parallel to that slab: the origin must
    // already lie inside the axis extent or there is no interval at all.
    var t_enter = ray.tmin;
    var t_exit = ray.tmax;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let d = ray.direction[axis];
        let o = ray.origin[axis];
        if (d == 0.0) {
            if (o < HYBRID_SDF_MIN[axis] || o > HYBRID_SDF_MAX[axis]) {
                return result;
            }
        } else {
            let t0 = (HYBRID_SDF_MIN[axis] - o) / d;
            let t1 = (HYBRID_SDF_MAX[axis] - o) / d;
            t_enter = max(t_enter, min(t0, t1));
            t_exit = min(t_exit, max(t0, t1));
        }
    }
    if (t_exit < t_enter) {
        return result;
    }

    // Geometric iteration bound for the clipped interval (see SDF_MARCH_EPS
    // comment). The u32 representability check is machine counter capacity,
    // not a chosen workload budget.
    let step_bound = ceil((t_exit - t_enter) / (0.5 * SDF_MARCH_EPS)) + 2.0;
    if (!hybrid_sdf_finite(step_bound) || step_bound < 2.0 || step_bound >= 4294967296.0) {
        hybrid_sdf_error = 1u;
        return result;
    }
    let max_steps = u32(step_bound);

    var t = t_enter;
    for (var step = 0u; step < max_steps; step = step + 1u) {
        if (t > t_exit) {
            return result;
        }
        let point = ray.origin + ray.direction * t;
        if (!(hybrid_sdf_finite(point.x) && hybrid_sdf_finite(point.y)
            && hybrid_sdf_finite(point.z))) {
            hybrid_sdf_error = 1u;
            return result;
        }
        let d = hybrid_sdf_evaluate(point);
        if (!hybrid_sdf_finite(d.distance)) {
            hybrid_sdf_error = 1u;
            return result;
        }
        if (abs(d.distance) <= SDF_MARCH_EPS) {
            // Central-difference gradient at the march epsilon.
            let gx = hybrid_sdf_evaluate(point + vec3f(SDF_MARCH_EPS, 0.0, 0.0)).distance
                - hybrid_sdf_evaluate(point - vec3f(SDF_MARCH_EPS, 0.0, 0.0)).distance;
            let gy = hybrid_sdf_evaluate(point + vec3f(0.0, SDF_MARCH_EPS, 0.0)).distance
                - hybrid_sdf_evaluate(point - vec3f(0.0, SDF_MARCH_EPS, 0.0)).distance;
            let gz = hybrid_sdf_evaluate(point + vec3f(0.0, 0.0, SDF_MARCH_EPS)).distance
                - hybrid_sdf_evaluate(point - vec3f(0.0, 0.0, SDF_MARCH_EPS)).distance;
            if (!(hybrid_sdf_finite(gx) && hybrid_sdf_finite(gy)
                && hybrid_sdf_finite(gz))) {
                hybrid_sdf_error = 1u;
                return result;
            }
            let grad = vec3f(gx, gy, gz);
            let glen = length(grad);
            if (!hybrid_sdf_finite(glen) || glen <= 0.0) {
                hybrid_sdf_error = 1u;
                return result;
            }
            result.hit = 1u;
            result.t = t;
            result.point = point;
            result.normal = grad / glen;
            result.material_id = d.material_id;
            return result;
        }
        let next_t = t + abs(d.distance);
        if (!hybrid_sdf_finite(next_t) || next_t <= t) {
            hybrid_sdf_error = 1u;
            return result;
        }
        if (next_t > t_exit) {
            return result;
        }
        t = next_t;
    }
    // Step budget exhausted while still inside the slab.
    hybrid_sdf_error = 1u;
    return result;
}

// Main hybrid intersection function
fn intersect_hybrid(ray: Ray) -> HybridHitResult {
    var best_hit: HybridHitResult;
    best_hit.hit = 0u;
    best_hit.t = ray.tmax;

    // Test mesh geometry if enabled
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 2u) {
        let mesh_hit = intersect_mesh(ray);
        if (mesh_hit.hit != 0u && mesh_hit.t < best_hit.t) {
            best_hit = mesh_hit;
        }
    }

    // Compiled-constant SDF scene (modes 0 and 1). Shares the one
    // intersect_sdf implementation with the shadow path below.
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 1u) {
        var sray = ray;
        sray.tmax = best_hit.t;
        let sdf_hit = intersect_sdf(sray);
        if (sdf_hit.hit != 0u && sdf_hit.t < best_hit.t) {
            best_hit = sdf_hit;
        }
    }

    // Terrain heightfield (hybrid mix, or selectable as the primary
    // intersectable via mode 3); defined in hybrid_terrain_traversal.wgsl.
    if ((hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 3u)
        && terrain_enabled()) {
        var tray = ray;
        tray.tmax = best_hit.t;
        let terrain_hit = terrain_intersect(tray);
        if (terrain_hit.hit != 0u && terrain_hit.t < best_hit.t) {
            best_hit = terrain_hit;
        }
    }

    return best_hit;
}

// Performance-optimized early termination
fn intersect_hybrid_optimized(
    ray: Ray,
    early_exit_distance: f32,
    apply_terrain_curvature: bool,
) -> HybridHitResult {
    var best_hit: HybridHitResult;
    best_hit.hit = 0u;
    best_hit.t = ray.tmax;

    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 2u) {
        let mesh_hit = intersect_mesh(ray);
        if (mesh_hit.hit != 0u && mesh_hit.t < early_exit_distance) {
            return mesh_hit;
        }
        if (mesh_hit.hit != 0u && mesh_hit.t < best_hit.t) {
            best_hit = mesh_hit;
        }
    }

    // Same compiled SDF scene; early-out like the mesh lane above.
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 1u) {
        var sray = ray;
        sray.tmax = best_hit.t;
        let sdf_hit = intersect_sdf(sray);
        if (sdf_hit.hit != 0u && sdf_hit.t < early_exit_distance) {
            return sdf_hit;
        }
        if (sdf_hit.hit != 0u && sdf_hit.t < best_hit.t) {
            best_hit = sdf_hit;
        }
    }

    // Terrain heightfield shadow/any-hit path shares the min-max descent.
    if ((hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 3u)
        && terrain_enabled()) {
        var tray = ray;
        tray.tmax = best_hit.t;
        let terrain_hit = terrain_trace(tray, true, apply_terrain_curvature);
        if (terrain_hit.hit != 0u && terrain_hit.t < best_hit.t) {
            best_hit = terrain_hit;
        }
    }

    return best_hit;
}

// Utility function to get surface properties at hit point
fn get_surface_properties(hit: HybridHitResult) -> vec3f {
    // Terrain hits use the uniform terrain albedo; everything else keeps the
    // legacy constant.
    if (hit.hit_type == 3u) {
        return terrain.albedo_pad.rgb;
    }
    return vec3f(0.7, 0.7, 0.8);
}

// Shadow ray testing for both SDF and mesh geometry. The terrain descent
// counters (hybrid_terrain_traversal.wgsl) attribute any trace issued from
// here to the shadow lane; private state is per-invocation, so the flag is
// always reset before return.
fn intersect_shadow_ray(ray: Ray, max_distance: f32) -> bool {
    terrain_is_shadow = true;
    let hit = intersect_hybrid_optimized(ray, 0.01, true);
    terrain_is_shadow = false;
    return hit.hit != 0u && hit.t < max_distance;
}

// IBL rays are arbitrary hemisphere samples, so the Sun's azimuth-specific
// effective curvature radius is not physically applicable to their terrain
// occlusion test.
fn intersect_ibl_occlusion_ray(ray: Ray, max_distance: f32) -> bool {
    terrain_is_shadow = true;
    let hit = intersect_hybrid_optimized(ray, 0.01, false);
    terrain_is_shadow = false;
    return hit.hit != 0u && hit.t < max_distance;
}

// Test occlusion for soft shadows (SDF can provide smoother shadows)
fn soft_shadow_factor(ray: Ray, max_distance: f32, softness: f32) -> f32 {
    return select(1.0, 0.0, intersect_shadow_ray(ray, max_distance));
}
