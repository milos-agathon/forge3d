// src/shaders/sdf_primitives.wgsl
// WGSL implementations of analytic SDF primitives for GPU raymarching

// SDF primitive data structures matching Rust layout
struct SdfSphere {
    center: vec3f,
    radius: f32,
}

struct SdfBox {
    center: vec3f,
    _pad1: f32,
    extents: vec3f,
    _pad2: f32,
}

struct SdfCylinder {
    center: vec3f,
    radius: f32,
    height: f32,
    _pad: vec3f,
}

struct SdfPlane {
    normal: vec3f,
    distance: f32,
}

struct SdfTorus {
    center: vec3f,
    major_radius: f32,
    minor_radius: f32,
    _pad: vec3f,
}

struct SdfCapsule {
    point_a: vec3f,
    radius: f32,
    point_b: vec3f,
    _pad: f32,
}

struct SdfPrimitive {
    primitive_type: u32,
    material_id: u32,
    _pad: vec2u,
    params: array<f32, 16>,
}

// Primitive type constants
const SDF_SPHERE: u32 = 0u;
const SDF_BOX: u32 = 1u;
const SDF_CYLINDER: u32 = 2u;
const SDF_PLANE: u32 = 3u;
const SDF_TORUS: u32 = 4u;
const SDF_CAPSULE: u32 = 5u;

// Helper functions for vector operations
fn length2(v: vec2f) -> f32 {
    return dot(v, v);
}

fn length3(v: vec3f) -> f32 {
    return sqrt(dot(v, v));
}

// SDF evaluation functions

// Sphere SDF: |p - center| - radius
fn sdf_sphere(point: vec3f, sphere: SdfSphere) -> f32 {
    return length3(point - sphere.center) - sphere.radius;
}

// Box SDF with proper handling of interior and exterior
fn sdf_box(point: vec3f, sdf_box: SdfBox) -> f32 {
    let local_point = abs(point - sdf_box.center);
    let q = local_point - sdf_box.extents;
    return length3(max(q, vec3f(0.0))) + min(max(max(q.x, q.y), q.z), 0.0);
}

// Cylinder SDF oriented along Y-axis
fn sdf_cylinder(point: vec3f, cylinder: SdfCylinder) -> f32 {
    let local_point = point - cylinder.center;
    let half_height = cylinder.height * 0.5;

    // Distance in XZ plane
    let xz_dist = length(vec2f(local_point.x, local_point.z));
    let radial_dist = xz_dist - cylinder.radius;

    // Distance along Y axis
    let vertical_dist = abs(local_point.y) - half_height;

    // Combine distances
    return length(vec2f(max(radial_dist, 0.0), max(vertical_dist, 0.0))) +
           min(max(radial_dist, vertical_dist), 0.0);
}

// Infinite plane SDF
fn sdf_plane(point: vec3f, plane: SdfPlane) -> f32 {
    return dot(point, plane.normal) + plane.distance;
}

// Torus SDF
fn sdf_torus(point: vec3f, torus: SdfTorus) -> f32 {
    let local_point = point - torus.center;
    let xz_dist = length(vec2f(local_point.x, local_point.z));
    let q = vec2f(xz_dist - torus.major_radius, local_point.y);
    return length(q) - torus.minor_radius;
}

// Capsule SDF (line segment with radius)
fn sdf_capsule(point: vec3f, capsule: SdfCapsule) -> f32 {
    let segment = capsule.point_b - capsule.point_a;
    let pa = point - capsule.point_a;
    let h = clamp(dot(pa, segment) / dot(segment, segment), 0.0, 1.0);
    let closest = capsule.point_a + segment * h;
    return length3(point - closest) - capsule.radius;
}

// Generic SDF primitive evaluation
fn evaluate_sdf_primitive(point: vec3f, primitive: SdfPrimitive) -> f32 {
    switch primitive.primitive_type {
        case SDF_SPHERE: {
            let sphere = SdfSphere(
                vec3f(primitive.params[0], primitive.params[1], primitive.params[2]),
                primitive.params[3]
            );
            return sdf_sphere(point, sphere);
        }
        case SDF_BOX: {
            let sdf_box = SdfBox(
                vec3f(primitive.params[0], primitive.params[1], primitive.params[2]),
                primitive.params[3],
                vec3f(primitive.params[4], primitive.params[5], primitive.params[6]),
                primitive.params[7]
            );
            return sdf_box(point, sdf_box);
        }
        case SDF_CYLINDER: {
            let cylinder = SdfCylinder(
                vec3f(primitive.params[0], primitive.params[1], primitive.params[2]),
                primitive.params[3],
                primitive.params[4],
                vec3f(primitive.params[5], primitive.params[6], primitive.params[7])
            );
            return sdf_cylinder(point, cylinder);
        }
        case SDF_PLANE: {
            let plane = SdfPlane(
                vec3f(primitive.params[0], primitive.params[1], primitive.params[2]),
                primitive.params[3]
            );
            return sdf_plane(point, plane);
        }
        case SDF_TORUS: {
            let torus = SdfTorus(
                vec3f(primitive.params[0], primitive.params[1], primitive.params[2]),
                primitive.params[3],
                primitive.params[4],
                vec3f(primitive.params[5], primitive.params[6], primitive.params[7])
            );
            return sdf_torus(point, torus);
        }
        case SDF_CAPSULE: {
            let capsule = SdfCapsule(
                vec3f(primitive.params[0], primitive.params[1], primitive.params[2]),
                primitive.params[3],
                vec3f(primitive.params[4], primitive.params[5], primitive.params[6]),
                primitive.params[7]
            );
            return sdf_capsule(point, capsule);
        }
        default: {
            return 1e20; // Very large distance for unknown primitive
        }
    }
}

// Calculate normal using finite differences (gradient)
fn sdf_normal(point: vec3f, primitive: SdfPrimitive) -> vec3f {
    let eps = 0.001;
    let normal = vec3f(
        evaluate_sdf_primitive(point + vec3f(eps, 0.0, 0.0), primitive) -
        evaluate_sdf_primitive(point - vec3f(eps, 0.0, 0.0), primitive),

        evaluate_sdf_primitive(point + vec3f(0.0, eps, 0.0), primitive) -
        evaluate_sdf_primitive(point - vec3f(0.0, eps, 0.0), primitive),

        evaluate_sdf_primitive(point + vec3f(0.0, 0.0, eps), primitive) -
        evaluate_sdf_primitive(point - vec3f(0.0, 0.0, eps), primitive)
    );
    return normalize(normal);
}

// Optimized normal calculation using tetrahedron technique
fn sdf_normal_optimized(point: vec3f, primitive: SdfPrimitive) -> vec3f {
    let eps = 0.001;
    let k1 = vec3f(1.0, -1.0, -1.0);
    let k2 = vec3f(-1.0, -1.0, 1.0);
    let k3 = vec3f(-1.0, 1.0, -1.0);
    let k4 = vec3f(1.0, 1.0, 1.0);

    let normal = k1 * evaluate_sdf_primitive(point + k1 * eps, primitive) +
                 k2 * evaluate_sdf_primitive(point + k2 * eps, primitive) +
                 k3 * evaluate_sdf_primitive(point + k3 * eps, primitive) +
                 k4 * evaluate_sdf_primitive(point + k4 * eps, primitive);

    return normalize(normal);
}