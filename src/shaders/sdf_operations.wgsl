// src/shaders/sdf_operations.wgsl
// WGSL implementations of CSG (Constructive Solid Geometry) operations for SDF

#include "sdf_primitives.wgsl"

// CSG operation types (matching Rust enum)
const CSG_UNION: u32 = 0u;
const CSG_INTERSECTION: u32 = 1u;
const CSG_SUBTRACTION: u32 = 2u;
const CSG_SMOOTH_UNION: u32 = 3u;
const CSG_SMOOTH_INTERSECTION: u32 = 4u;
const CSG_SMOOTH_SUBTRACTION: u32 = 5u;

// CSG node structure matching Rust layout
struct CsgNode {
    operation: u32,
    left_child: u32,
    right_child: u32,
    smoothing: f32,
    material_id: u32,
    is_leaf: u32,
    _pad: vec2u,
}

// CSG evaluation result
struct CsgResult {
    distance: f32,
    material_id: u32,
}

// Smooth minimum function for smooth CSG operations
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) {
        return min(a, b);
    }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Smooth maximum function for smooth CSG operations
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    return -smooth_min(-a, -b, k);
}

// CSG union: minimum distance
fn csg_union(a: CsgResult, b: CsgResult) -> CsgResult {
    if (a.distance <= b.distance) {
        return a;
    } else {
        return b;
    }
}

// CSG intersection: maximum distance
fn csg_intersection(a: CsgResult, b: CsgResult) -> CsgResult {
    if (a.distance >= b.distance) {
        return a;
    } else {
        return b;
    }
}

// CSG subtraction: subtract b from a
fn csg_subtraction(a: CsgResult, b: CsgResult) -> CsgResult {
    let neg_b = CsgResult(-b.distance, b.material_id);
    return csg_intersection(a, neg_b);
}

// Smooth CSG union
fn csg_smooth_union(a: CsgResult, b: CsgResult, k: f32) -> CsgResult {
    let distance = smooth_min(a.distance, b.distance, k);

    // Blend materials based on contribution
    let t = select(0.5, b.distance / (a.distance + b.distance), abs(a.distance + b.distance) > 1e-6);
    let material_id = select(a.material_id, b.material_id, t >= 0.5);

    return CsgResult(distance, material_id);
}

// Smooth CSG intersection
fn csg_smooth_intersection(a: CsgResult, b: CsgResult, k: f32) -> CsgResult {
    let distance = smooth_max(a.distance, b.distance, k);
    let material_id = select(b.material_id, a.material_id, a.distance >= b.distance);

    return CsgResult(distance, material_id);
}

// Smooth CSG subtraction
fn csg_smooth_subtraction(a: CsgResult, b: CsgResult, k: f32) -> CsgResult {
    let neg_b = CsgResult(-b.distance, b.material_id);
    return csg_smooth_intersection(a, neg_b, k);
}

// Apply CSG operation to two results
fn apply_csg_operation(
    operation: u32,
    a: CsgResult,
    b: CsgResult,
    smoothing: f32
) -> CsgResult {
    switch operation {
        case CSG_UNION: {
            return csg_union(a, b);
        }
        case CSG_INTERSECTION: {
            return csg_intersection(a, b);
        }
        case CSG_SUBTRACTION: {
            return csg_subtraction(a, b);
        }
        case CSG_SMOOTH_UNION: {
            return csg_smooth_union(a, b, smoothing);
        }
        case CSG_SMOOTH_INTERSECTION: {
            return csg_smooth_intersection(a, b, smoothing);
        }
        case CSG_SMOOTH_SUBTRACTION: {
            return csg_smooth_subtraction(a, b, smoothing);
        }
        default: {
            return csg_union(a, b); // Fallback to union
        }
    }
}

// Storage for CSG evaluation (bind groups to be defined by caller)
// @group(0) @binding(0) var<storage, read> csg_nodes: array<CsgNode>;
// @group(0) @binding(1) var<storage, read> sdf_primitives: array<SdfPrimitive>;

// Recursive CSG tree evaluation
// Note: WGSL doesn't support true recursion, so this would need to be implemented
// using an iterative approach with a stack for deep trees
fn evaluate_csg_tree_iterative(
    point: vec3f,
    root_node: u32,
    nodes: ptr<function, array<CsgNode, 64>>, // Limited stack size
    primitives: ptr<function, array<SdfPrimitive, 64>>,
    node_count: u32,
    primitive_count: u32
) -> CsgResult {
    // Simple implementation for demonstration
    // In practice, this would use an iterative stack-based approach

    if (root_node >= node_count) {
        return CsgResult(1e20, 0u);
    }

    let node = (*nodes)[root_node];

    if (node.is_leaf != 0u) {
        // Leaf node: evaluate primitive
        if (node.left_child >= primitive_count) {
            return CsgResult(1e20, node.material_id);
        }

        let primitive = (*primitives)[node.left_child];
        let distance = evaluate_sdf_primitive(point, primitive);

        return CsgResult(distance, node.material_id);
    } else {
        // Operation node: would evaluate children and apply operation
        // For simplicity, returning a placeholder here
        // In a real implementation, this would use an iterative stack
        return CsgResult(1e20, node.material_id);
    }
}

// Simpler evaluation for single primitive or simple operations
fn evaluate_simple_csg(
    point: vec3f,
    primitive_a: SdfPrimitive,
    primitive_b: SdfPrimitive,
    operation: u32,
    smoothing: f32
) -> CsgResult {
    let result_a = CsgResult(
        evaluate_sdf_primitive(point, primitive_a),
        primitive_a.material_id
    );

    let result_b = CsgResult(
        evaluate_sdf_primitive(point, primitive_b),
        primitive_b.material_id
    );

    return apply_csg_operation(operation, result_a, result_b, smoothing);
}

// Calculate normal for CSG result using finite differences
fn csg_normal(
    point: vec3f,
    primitive_a: SdfPrimitive,
    primitive_b: SdfPrimitive,
    operation: u32,
    smoothing: f32
) -> vec3f {
    let eps = 0.001;

    let normal = vec3f(
        evaluate_simple_csg(point + vec3f(eps, 0.0, 0.0), primitive_a, primitive_b, operation, smoothing).distance -
        evaluate_simple_csg(point - vec3f(eps, 0.0, 0.0), primitive_a, primitive_b, operation, smoothing).distance,

        evaluate_simple_csg(point + vec3f(0.0, eps, 0.0), primitive_a, primitive_b, operation, smoothing).distance -
        evaluate_simple_csg(point - vec3f(0.0, eps, 0.0), primitive_a, primitive_b, operation, smoothing).distance,

        evaluate_simple_csg(point + vec3f(0.0, 0.0, eps), primitive_a, primitive_b, operation, smoothing).distance -
        evaluate_simple_csg(point - vec3f(0.0, 0.0, eps), primitive_a, primitive_b, operation, smoothing).distance
    );

    return normalize(normal);
}

// Domain repetition operations for procedural patterns
fn domain_repeat_infinite(point: vec3f, spacing: vec3f) -> vec3f {
    return point - spacing * round(point / spacing);
}

fn domain_repeat_limited(point: vec3f, spacing: vec3f, limit: vec3f) -> vec3f {
    let q = point / spacing;
    let id = clamp(round(q), -limit, limit);
    return point - spacing * id;
}

// Domain transformation operations
fn domain_twist(point: vec3f, twist_amount: f32) -> vec3f {
    let angle = point.y * twist_amount;
    let c = cos(angle);
    let s = sin(angle);

    return vec3f(
        c * point.x - s * point.z,
        point.y,
        s * point.x + c * point.z
    );
}

fn domain_bend(point: vec3f, bend_amount: f32) -> vec3f {
    let angle = point.x * bend_amount;
    let c = cos(angle);
    let s = sin(angle);

    return vec3f(
        point.x,
        c * point.y - s * point.z,
        s * point.y + c * point.z
    );
}