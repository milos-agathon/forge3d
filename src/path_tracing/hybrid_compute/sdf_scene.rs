// src/path_tracing/hybrid_compute/sdf_scene.rs
// PROMETHEUS GPU SDF: validates a native SdfScene and compiles it into the
// hybrid kernel's constant declarations. The CSG tree becomes straight-line
// WGSL evaluated by the existing sdf_primitives/sdf_operations helpers — no
// storage binding, no fixed scene capacity, no runtime array. All validation
// and bound computation happen here on the CPU; the shader only executes the
// already-validated expression tree. Nothing in this module evaluates pixels
// or approximates the march; a scene that cannot be validated is rejected
// before any pipeline is created.

use glam::Vec3;

use crate::core::error::RenderError;
use crate::sdf::primitives::SdfPrimitiveType;
use crate::sdf::{CsgOperation, SdfPrimitive, SdfScene};

// Exact declarations in src/shaders/hybrid_traversal.wgsl. specialize()
// replaces each once; a drift in either file is a hard error, not a partial
// specialization.
const ENABLED_PLACEHOLDER: &str = "const HYBRID_SDF_ENABLED: bool = false;";
const MIN_PLACEHOLDER: &str = "const HYBRID_SDF_MIN: vec3<f32> = vec3<f32>(0.0);";
const MAX_PLACEHOLDER: &str = "const HYBRID_SDF_MAX: vec3<f32> = vec3<f32>(0.0);";
const EVALUATE_BODY_PLACEHOLDER: &str = "    return CsgResult(1e30, 0u);";

fn err(msg: impl Into<String>) -> RenderError {
    RenderError::render(msg.into())
}

/// Axis-aligned bound of a subtree. `Empty` marks a provably empty
/// intersection; `Unbounded` marks a subtree containing a plane (or any
/// future primitive without a finite extent) that was never intersected
/// with bounded geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
enum SdfBound {
    Empty,
    Box(Vec3, Vec3),
    Unbounded,
}

fn bound_union(a: SdfBound, b: SdfBound) -> SdfBound {
    match (a, b) {
        (SdfBound::Empty, x) | (x, SdfBound::Empty) => x,
        (SdfBound::Unbounded, _) | (_, SdfBound::Unbounded) => SdfBound::Unbounded,
        (SdfBound::Box(amin, amax), SdfBound::Box(bmin, bmax)) => {
            SdfBound::Box(amin.min(bmin), amax.max(bmax))
        }
    }
}

fn bound_intersection(a: SdfBound, b: SdfBound) -> SdfBound {
    match (a, b) {
        (SdfBound::Empty, _) | (_, SdfBound::Empty) => SdfBound::Empty,
        // An unbounded child contributes no extent; the bounded child
        // bounds the intersection.
        (SdfBound::Unbounded, x) | (x, SdfBound::Unbounded) => x,
        (SdfBound::Box(amin, amax), SdfBound::Box(bmin, bmax)) => {
            let lo = amin.max(bmin);
            let hi = amax.min(bmax);
            if lo.x < hi.x && lo.y < hi.y && lo.z < hi.z {
                SdfBound::Box(lo, hi)
            } else {
                SdfBound::Empty
            }
        }
    }
}

fn bound_expand(b: SdfBound, k: f32) -> SdfBound {
    match b {
        SdfBound::Box(lo, hi) => SdfBound::Box(lo - Vec3::splat(k), hi + Vec3::splat(k)),
        other => other,
    }
}

/// Conservative bound for one CSG operation over child bounds that were
/// already expanded to their required level set. Union and smooth union
/// take the union (the smooth blend's k/4 reach was propagated into the
/// child leaf bounds by the level pass, so it is not expanded again);
/// intersection and smooth intersection take the intersection; both
/// subtractions keep the left child's bound.
fn operation_bound(operation: u32, left: SdfBound, right: SdfBound) -> SdfBound {
    match operation {
        x if x == CsgOperation::Union as u32 => bound_union(left, right),
        x if x == CsgOperation::Intersection as u32 => bound_intersection(left, right),
        x if x == CsgOperation::Subtraction as u32 => left,
        x if x == CsgOperation::SmoothUnion as u32 => bound_union(left, right),
        x if x == CsgOperation::SmoothIntersection as u32 => bound_intersection(left, right),
        x if x == CsgOperation::SmoothSubtraction as u32 => left,
        _ => SdfBound::Empty,
    }
}

/// Per-primitive bound. Returns Err for any computed coordinate that is
/// non-finite or degenerate; Plane has no finite bound by construction.
fn primitive_bound(prim: &SdfPrimitive) -> Result<SdfBound, RenderError> {
    let p = &prim.params;
    let center = Vec3::new(p[0], p[1], p[2]);
    let make = |lo: Vec3, hi: Vec3| -> Result<SdfBound, RenderError> {
        if !lo.is_finite() || !hi.is_finite() || !lo.cmplt(hi).all() {
            return Err(err(
                "SDF primitive produced a degenerate or non-finite bound",
            ));
        }
        Ok(SdfBound::Box(lo, hi))
    };
    match prim.primitive_type {
        t if t == SdfPrimitiveType::Sphere as u32 => {
            let r = p[3];
            make(center - Vec3::splat(r), center + Vec3::splat(r))
        }
        t if t == SdfPrimitiveType::Box as u32 => {
            let e = Vec3::new(p[4], p[5], p[6]);
            make(center - e, center + e)
        }
        t if t == SdfPrimitiveType::Cylinder as u32 => {
            let (r, h) = (p[3], p[4]);
            make(
                center - Vec3::new(r, h / 2.0, r),
                center + Vec3::new(r, h / 2.0, r),
            )
        }
        t if t == SdfPrimitiveType::Plane as u32 => Ok(SdfBound::Unbounded),
        t if t == SdfPrimitiveType::Torus as u32 => {
            let (major, minor) = (p[3], p[4]);
            let e = Vec3::new(major + minor, minor, major + minor);
            make(center - e, center + e)
        }
        t if t == SdfPrimitiveType::Capsule as u32 => {
            let (a, b, r) = (center, Vec3::new(p[4], p[5], p[6]), p[3]);
            make(a.min(b) - Vec3::splat(r), a.max(b) + Vec3::splat(r))
        }
        _ => Err(err("SDF primitive type must be 0..=5")),
    }
}

/// Validate one primitive's parameter block against the layout shared with
/// evaluate_sdf_primitive in sdf_primitives.wgsl.
fn validate_primitive(prim: &SdfPrimitive, index: usize) -> Result<(), RenderError> {
    if prim.params.iter().any(|v| !v.is_finite()) {
        return Err(err(format!(
            "SDF primitive {index} has non-finite parameters; all params must be finite"
        )));
    }
    let p = &prim.params;
    match prim.primitive_type {
        t if t == SdfPrimitiveType::Sphere as u32 => {
            if p[3] <= 0.0 {
                return Err(err(format!(
                    "SDF sphere primitive {index} radius must be > 0"
                )));
            }
        }
        t if t == SdfPrimitiveType::Box as u32 => {
            if p[4] <= 0.0 || p[5] <= 0.0 || p[6] <= 0.0 {
                return Err(err(format!(
                    "SDF box primitive {index} extents must be > 0"
                )));
            }
        }
        t if t == SdfPrimitiveType::Cylinder as u32 => {
            if p[3] <= 0.0 || p[4] <= 0.0 {
                return Err(err(format!(
                    "SDF cylinder primitive {index} radius and height must be > 0"
                )));
            }
        }
        t if t == SdfPrimitiveType::Plane as u32 => {
            // Huge finite components overflow length() to inf; a subnormal
            // norm turns distance/norm into inf. Reject both before
            // emit_primitive normalizes.
            let norm = Vec3::new(p[0], p[1], p[2]).length();
            if !norm.is_finite() || norm <= 0.0 || !(p[3] / norm).is_finite() {
                return Err(err(format!(
                    "SDF plane primitive {index} normal must be finite and non-zero"
                )));
            }
        }
        t if t == SdfPrimitiveType::Torus as u32 => {
            if p[3] <= 0.0 || p[4] <= 0.0 {
                return Err(err(format!(
                    "SDF torus primitive {index} major and minor radii must be > 0"
                )));
            }
        }
        t if t == SdfPrimitiveType::Capsule as u32 => {
            let a = Vec3::new(p[0], p[1], p[2]);
            let b = Vec3::new(p[4], p[5], p[6]);
            if p[3] <= 0.0 {
                return Err(err(format!(
                    "SDF capsule primitive {index} radius must be > 0"
                )));
            }
            // sdf_capsule divides by dot(segment, segment): distinct
            // endpoints are not enough — a tiny segment underflows the
            // squared length to zero and a huge one overflows it.
            let segment_squared = (b - a).length_squared();
            if !segment_squared.is_finite() || segment_squared <= 0.0 {
                return Err(err(format!(
                    "SDF capsule primitive {index} endpoints must be distinct"
                )));
            }
        }
        _ => {
            return Err(err(format!(
                "SDF primitive {index} type {} is invalid; expected 0..=5",
                prim.primitive_type
            )));
        }
    }
    Ok(())
}

/// Emit one SdfPrimitive constructor expression. Floats are formatted from
/// validated finite f32 values in `{:e}` form (e.g. `1.5e0`, `-2e-1`), which
/// WGSL accepts as decimal literals — no user-supplied text reaches the
/// source. Plane normals are normalized and the distance divided by the
/// same norm, preserving the zero set while keeping the field a true SDF.
fn emit_primitive(prim: &SdfPrimitive) -> String {
    let mut params = prim.params;
    if prim.primitive_type == SdfPrimitiveType::Plane as u32 {
        let norm = Vec3::new(params[0], params[1], params[2]).length();
        params[0] /= norm;
        params[1] /= norm;
        params[2] /= norm;
        params[3] /= norm;
    }
    let fields = params
        .iter()
        .map(|v| format!("{v:e}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "SdfPrimitive({}u, {}u, vec2u(0u), array<f32, 16>({}))",
        prim.primitive_type, prim.material_id, fields
    )
}

fn replace_once(source: &str, placeholder: &str, replacement: &str) -> Result<String, RenderError> {
    if source.matches(placeholder).count() != 1 {
        return Err(err(format!(
            "SDF specialization placeholder is not unique in the assembled hybrid kernel: {placeholder:?}"
        )));
    }
    Ok(source.replacen(placeholder, replacement, 1))
}

/// Validate `scene` and return the assembled hybrid kernel with the SDF
/// declarations specialized to its constants. The returned source is the
/// same module as `shader_sources::hybrid_kernel()` with exactly four
/// substitutions: the enabled flag, the bound corners, and the body of
/// `hybrid_sdf_evaluate`.
pub(crate) fn specialize(scene: &SdfScene) -> Result<String, RenderError> {
    let tree = &scene.csg_tree;
    if tree.nodes.is_empty() {
        return Err(err("SDF scene is empty: it has no CSG nodes"));
    }

    // Validate every primitive up front: all params finite, known type,
    // positive dimensions, distinct capsule endpoints.
    for (index, prim) in tree.primitives.iter().enumerate() {
        validate_primitive(prim, index)?;
    }

    // Nodes are stored topologically: every operation's children must point
    // at strictly earlier nodes, which rejects cycles and forward
    // references without a separate graph walk.
    for (i, node) in tree.nodes.iter().enumerate() {
        if node.is_leaf > 1 {
            return Err(err(format!(
                "SDF CSG node {i} has invalid is_leaf {}; expected 0 or 1",
                node.is_leaf
            )));
        }
        if node.is_leaf == 1 {
            let prim_index = node.left_child as usize;
            if prim_index >= tree.primitives.len() {
                return Err(err(format!(
                    "SDF leaf node {i} references primitive {prim_index} but only {} exist",
                    tree.primitives.len()
                )));
            }
        } else {
            if node.operation > CsgOperation::SmoothSubtraction as u32 {
                return Err(err(format!(
                    "SDF CSG node {i} has invalid operation {}; expected 0..=5",
                    node.operation
                )));
            }
            if !node.smoothing.is_finite() || node.smoothing < 0.0 {
                return Err(err(format!(
                    "SDF CSG node {i} smoothing must be finite and >= 0"
                )));
            }
            let (l, r) = (node.left_child as usize, node.right_child as usize);
            if l >= i || r >= i {
                return Err(err(format!(
                    "SDF CSG node {i} references child ({l}, {r}) that is not an earlier node"
                )));
            }
        }
    }

    // Level propagation: a subtree feeding a smooth union must keep its
    // positive level set up to the parent's reach, not just its zero set.
    // The smooth-union blend lowers the field by at most k/4 (polynomial
    // smooth-min), so geometry up to k/4 outside the child zero set can
    // still contribute; an empty hard intersection can become non-empty
    // once smoothed. levels[i] is the required half-width: the root needs
    // its zero set (0), and each smooth union adds smoothing/4 to both
    // children's requirements. Traversing in reverse order visits every
    // parent before the children it marks, so DAG-shared children keep the
    // maximum requested level. Both subtraction children inherit the level:
    // conservative, since the right child only constrains.
    let mut levels: Vec<Option<f32>> = vec![None; tree.nodes.len()];
    *levels.last_mut().expect("non-empty node list") = Some(0.0);
    for i in (0..tree.nodes.len()).rev() {
        let Some(level) = levels[i] else { continue };
        let node = &tree.nodes[i];
        if node.is_leaf == 1 {
            continue;
        }
        let child_level = if node.operation == CsgOperation::SmoothUnion as u32 {
            level + node.smoothing / 4.0
        } else {
            level
        };
        if !child_level.is_finite() {
            return Err(err(format!(
                "SDF CSG node {i} smooth-union level is not finite"
            )));
        }
        for child in [node.left_child as usize, node.right_child as usize] {
            levels[child] = Some(match levels[child] {
                Some(existing) => existing.max(child_level),
                None => child_level,
            });
        }
    }

    // Forward bound pass over reachable nodes; leaves are expanded by their
    // required level so the bound covers the level set the parent needs.
    let mut bounds: Vec<SdfBound> = Vec::with_capacity(tree.nodes.len());
    for (i, node) in tree.nodes.iter().enumerate() {
        let bound = match levels[i] {
            None => SdfBound::Empty,
            Some(level) => {
                if node.is_leaf == 1 {
                    let prim = &tree.primitives[node.left_child as usize];
                    bound_expand(primitive_bound(prim)?, level)
                } else {
                    let (l, r) = (node.left_child as usize, node.right_child as usize);
                    operation_bound(node.operation, bounds[l], bounds[r])
                }
            }
        };
        bounds.push(bound);
    }

    // The last node is the native root (CsgTree::root_node).
    let (bound_min, bound_max) =
        match bounds[tree.nodes.len() - 1] {
            SdfBound::Box(lo, hi) => {
                // Expansion by a level can overflow an otherwise valid bound.
                if !lo.is_finite() || !hi.is_finite() || !lo.cmplt(hi).all() {
                    return Err(err("SDF root bound is non-finite or degenerate"));
                }
                (lo, hi)
            }
            SdfBound::Empty => {
                return Err(err(
                    "SDF scene bounds are empty: the CSG tree intersects to no geometry",
                ))
            }
            SdfBound::Unbounded => return Err(err(
                "SDF root must be bounded; intersect unbounded primitives with bounded geometry",
            )),
        };

    // A supplied scene bound is a promise, not a crop: it must be finite,
    // well-formed, and contain the computed root bound, which stays
    // authoritative so geometry is never silently clipped.
    if let Some((smin, smax)) = scene.bounds {
        if !smin.is_finite() || !smax.is_finite() || !smin.cmplt(smax).all() {
            return Err(err(
                "supplied SDF scene bounds are non-finite or degenerate",
            ));
        }
        if !(smin.cmple(bound_min).all() && smax.cmpge(bound_max).all()) {
            return Err(err(
                "supplied SDF scene bounds do not contain the computed root bound",
            ));
        }
    }

    // Straight-line WGSL for the whole node list, in node order.
    let mut body = String::new();
    for (i, node) in tree.nodes.iter().enumerate() {
        if node.is_leaf == 1 {
            let prim = &tree.primitives[node.left_child as usize];
            body.push_str(&format!(
                "    let sdf_node_{i} = CsgResult(evaluate_sdf_primitive(point, {}), {}u);\n",
                emit_primitive(prim),
                node.material_id
            ));
        } else {
            body.push_str(&format!(
                "    let sdf_node_{i} = apply_csg_operation({}u, sdf_node_{}, sdf_node_{}, {:e});\n",
                node.operation, node.left_child, node.right_child, node.smoothing
            ));
        }
    }
    body.push_str(&format!("    return sdf_node_{};", tree.nodes.len() - 1));

    let source = crate::shader_sources::hybrid_kernel();
    let source = replace_once(
        &source,
        ENABLED_PLACEHOLDER,
        "const HYBRID_SDF_ENABLED: bool = true;",
    )?;
    let source = replace_once(
        &source,
        MIN_PLACEHOLDER,
        &format!(
            "const HYBRID_SDF_MIN: vec3<f32> = vec3<f32>({:e}, {:e}, {:e});",
            bound_min.x, bound_min.y, bound_min.z
        ),
    )?;
    let source = replace_once(
        &source,
        MAX_PLACEHOLDER,
        &format!(
            "const HYBRID_SDF_MAX: vec3<f32> = vec3<f32>({:e}, {:e}, {:e});",
            bound_max.x, bound_max.y, bound_max.z
        ),
    )?;
    replace_once(&source, EVALUATE_BODY_PLACEHOLDER, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::SdfSceneBuilder;
    use crate::shader_sources::hybrid_kernel;

    fn validate_wgsl(source: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    fn build_sphere() -> SdfScene {
        let mut b = SdfSceneBuilder::new();
        b.add_sphere_mut(Vec3::new(0.0, 8.0, 0.0), 3.0, 1);
        b.build()
    }

    #[test]
    fn default_kernel_has_finite_empty_sdf() {
        let source = hybrid_kernel();
        validate_wgsl(&source);
        assert!(source.contains(ENABLED_PLACEHOLDER));
        assert!(source.contains(MIN_PLACEHOLDER));
        assert!(source.contains(MAX_PLACEHOLDER));
        assert!(source.contains(EVALUATE_BODY_PLACEHOLDER));
        // Disabled path: the march returns immediately.
        assert!(source.contains("if (!HYBRID_SDF_ENABLED)"));
    }

    #[test]
    fn all_six_primitives_generate_valid_wgsl() {
        let mut b = SdfSceneBuilder::new();
        let s = b.add_sphere_mut(Vec3::ZERO, 1.0, 1);
        let bx = b.add_box_mut(Vec3::new(5.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0), 2);
        let cy = b.add_cylinder_mut(Vec3::new(-5.0, 0.0, 0.0), 0.5, 2.0, 3);
        let to = b.add_torus_mut(Vec3::new(0.0, 5.0, 0.0), 2.0, 0.5, 4);
        let ca = b.add_capsule_mut(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, -5.0, 2.0), 0.5, 5);
        // Plane is unbounded alone; bound it by intersecting with a box.
        let pl = b.add_plane_mut(Vec3::Y, 1.0, 6);
        let bound_box = b.add_box_mut(Vec3::ZERO, Vec3::splat(20.0), 7);
        let bounded_plane = b.intersect_mut(pl, bound_box, 6);
        let u1 = b.union_mut(s, bx, 8);
        let u2 = b.union_mut(u1, cy, 8);
        let u3 = b.union_mut(u2, to, 8);
        let u4 = b.union_mut(u3, ca, 8);
        b.union_mut(u4, bounded_plane, 8);
        let scene = b.build();
        let source = specialize(&scene).expect("six-primitive scene must specialize");
        validate_wgsl(&source);
        assert!(source.contains("const HYBRID_SDF_ENABLED: bool = true;"));
    }

    #[test]
    fn all_six_csg_operations_generate_valid_wgsl() {
        for op in [
            CsgOperation::Union,
            CsgOperation::Intersection,
            CsgOperation::Subtraction,
            CsgOperation::SmoothUnion,
            CsgOperation::SmoothIntersection,
            CsgOperation::SmoothSubtraction,
        ] {
            let mut b = SdfSceneBuilder::new();
            let a = b.add_sphere_mut(Vec3::ZERO, 2.0, 1);
            let c = b.add_sphere_mut(Vec3::new(1.0, 0.0, 0.0), 1.5, 2);
            match op {
                CsgOperation::Union => {
                    b.union_mut(a, c, 1);
                }
                CsgOperation::Intersection => {
                    b.intersect_mut(a, c, 1);
                }
                CsgOperation::Subtraction => {
                    b.subtract_mut(a, c, 1);
                }
                CsgOperation::SmoothUnion => {
                    b.smooth_union_mut(a, c, 0.5, 1);
                }
                CsgOperation::SmoothIntersection => {
                    b.smooth_intersection_mut(a, c, 0.5, 1);
                }
                CsgOperation::SmoothSubtraction => {
                    b.smooth_subtraction_mut(a, c, 0.5, 1);
                }
            }
            let scene = b.build();
            let source = specialize(&scene).unwrap_or_else(|e| panic!("{op:?} scene failed: {e}"));
            validate_wgsl(&source);
            assert!(source.contains(&format!("apply_csg_operation({}u,", op as u32)));
        }
    }

    #[test]
    fn bounded_plane_intersection_is_valid() {
        let mut b = SdfSceneBuilder::new();
        let pl = b.add_plane_mut(Vec3::Y, 4.0, 1);
        let bx = b.add_box_mut(Vec3::ZERO, Vec3::splat(10.0), 1);
        b.intersect_mut(pl, bx, 1);
        let mut scene = b.build();
        // Bypass the constructor's normalization so emit_primitive's own
        // normalize-and-rescale-distance path is what gets exercised:
        // normal (0,2,0) with distance 8 must emit (0,1,0) and distance 4.
        scene.csg_tree.primitives[0].params[1] = 2.0;
        scene.csg_tree.primitives[0].params[3] = 8.0;
        let source = specialize(&scene).expect("plane intersected with box must specialize");
        validate_wgsl(&source);
        assert!(source.contains("0e0, 1e0, 0e0, 4e0"));
    }

    #[test]
    fn empty_scene_rejected() {
        let scene = SdfScene::new();
        let msg = specialize(&scene).unwrap_err().to_string();
        assert!(msg.contains("empty"), "got: {msg}");
    }

    #[test]
    fn invalid_graph_rejected() {
        let mut scene = build_sphere();
        // Forward reference: node 1's child is not an earlier node.
        scene.csg_tree.nodes.push(crate::sdf::CsgNode::operation(
            CsgOperation::Union,
            0,
            5,
            0.0,
            1,
        ));
        let msg = specialize(&scene).unwrap_err().to_string();
        assert!(msg.contains("earlier node"), "got: {msg}");

        let mut scene = build_sphere();
        scene.csg_tree.nodes[0].is_leaf = 7;
        let msg = specialize(&scene).unwrap_err().to_string();
        assert!(msg.contains("is_leaf"), "got: {msg}");

        let mut scene = build_sphere();
        scene.csg_tree.nodes.push(crate::sdf::CsgNode::leaf(9, 1));
        let msg = specialize(&scene).unwrap_err().to_string();
        assert!(msg.contains("primitive"), "got: {msg}");
    }

    #[test]
    fn nonfinite_and_bad_dimensions_rejected() {
        let mut b = SdfSceneBuilder::new();
        b.add_sphere_mut(Vec3::ZERO, f32::NAN, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("finite"), "got: {msg}");

        let mut b = SdfSceneBuilder::new();
        b.add_sphere_mut(Vec3::ZERO, -1.0, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("radius"), "got: {msg}");

        let mut b = SdfSceneBuilder::new();
        b.add_box_mut(Vec3::ZERO, Vec3::new(1.0, 0.0, 1.0), 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("extents"), "got: {msg}");
    }

    #[test]
    fn zero_length_capsule_rejected() {
        let mut b = SdfSceneBuilder::new();
        b.add_capsule_mut(Vec3::ZERO, Vec3::ZERO, 1.0, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("capsule"), "got: {msg}");
    }

    #[test]
    fn unbounded_root_rejected() {
        let mut b = SdfSceneBuilder::new();
        b.add_plane_mut(Vec3::Y, 0.0, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("bounded"), "got: {msg}");

        // Union with a bounded child does not bound an unbounded one.
        let mut b = SdfSceneBuilder::new();
        let pl = b.add_plane_mut(Vec3::Y, 0.0, 1);
        let s = b.add_sphere_mut(Vec3::ZERO, 1.0, 1);
        b.union_mut(pl, s, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("bounded"), "got: {msg}");
    }

    #[test]
    fn smooth_union_bound_expands_by_k_over_4() {
        let mut b = SdfSceneBuilder::new();
        let a = b.add_sphere_mut(Vec3::ZERO, 1.0, 1);
        let c = b.add_sphere_mut(Vec3::new(4.0, 0.0, 0.0), 1.0, 1);
        b.smooth_union_mut(a, c, 2.0, 1);
        let scene = b.build();
        let source = specialize(&scene).expect("smooth union must specialize");
        // Union bound is x in [-1, 5]; k/4 = 0.5 expands it to [-1.5, 5.5].
        assert!(source.contains("const HYBRID_SDF_MIN: vec3<f32> = vec3<f32>(-1.5e0"));
        assert!(source.contains("const HYBRID_SDF_MAX: vec3<f32> = vec3<f32>(5.5e0"));
        validate_wgsl(&source);
    }

    #[test]
    fn more_than_64_nodes_validate() {
        // 80 spheres in a union chain: 80 leaf nodes + 79 operations. The
        // generated function is straight-line code, so no inherited
        // fixed-capacity array limits the scene size.
        let mut b = SdfSceneBuilder::new();
        let mut acc = b.add_sphere_mut(Vec3::ZERO, 1.0, 1);
        for i in 1..80 {
            let s = b.add_sphere_mut(Vec3::new(i as f32 * 3.0, 0.0, 0.0), 1.0, 1);
            acc = b.union_mut(acc, s, 1);
        }
        let scene = b.build();
        assert!(scene.node_count() > 64);
        let source = specialize(&scene).expect(">64-node scene must specialize");
        validate_wgsl(&source);
        assert!(source.contains("return sdf_node_158;"));
    }

    #[test]
    fn smooth_union_keeps_positive_level_sets() {
        // Intersection of two disjoint boxes has an empty zero set but a
        // non-empty positive level set: smoothing it can surface geometry
        // the hard bound would drop. CPU evaluate is the independent
        // witness that the zero set is non-empty at the origin.
        let mut b = SdfSceneBuilder::new();
        let a = b.add_box_mut(Vec3::new(-1.1, 0.0, 0.0), Vec3::ONE, 1);
        let c = b.add_box_mut(Vec3::new(1.1, 0.0, 0.0), Vec3::ONE, 1);
        let inter = b.intersect_mut(a, c, 1);
        b.smooth_union_mut(inter, inter, 1.0, 1);
        let scene = b.build();
        assert!(
            scene.evaluate(Vec3::ZERO).distance < 0.0,
            "smoothed empty intersection must contain the origin"
        );
        let source =
            specialize(&scene).expect("level-expanded bound must keep the smoothed geometry");
        validate_wgsl(&source);
        // The boxes are disjoint only in X: bounds [-2.1,-0.1] and
        // [0.1,2.1] expand by k/4 = 0.25 and intersect on about
        // [-0.15, 0.15] there (f32 arithmetic emits -1.4999998e-1).
        assert!(source.contains("vec3<f32>(-1.4999998e-1"));
        assert!(source.contains("vec3<f32>(1.4999998e-1"));
    }

    #[test]
    fn root_bound_overflow_rejected() {
        // Level expansion can push a finite leaf bound past f32 range.
        let mut b = SdfSceneBuilder::new();
        let a = b.add_sphere_mut(Vec3::new(2.5e38, 0.0, 0.0), 5e37, 1);
        let c = b.add_sphere_mut(Vec3::ZERO, 1.0, 1);
        b.smooth_union_mut(a, c, 2e38, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("bound"), "got: {msg}");
    }

    #[test]
    fn plane_with_overflowing_norm_rejected() {
        let mut b = SdfSceneBuilder::new();
        let pl = b.add_plane_mut(Vec3::Y, 4.0, 1);
        let bx = b.add_box_mut(Vec3::ZERO, Vec3::splat(10.0), 1);
        b.intersect_mut(pl, bx, 1);
        let mut scene = b.build();
        // (1e30)^2 * 3 overflows f32, so length() is inf and the emitted
        // normalized normal would collapse to zero.
        scene.csg_tree.primitives[0].params[0] = 1e30;
        scene.csg_tree.primitives[0].params[1] = 1e30;
        scene.csg_tree.primitives[0].params[2] = 1e30;
        let msg = specialize(&scene).unwrap_err().to_string();
        assert!(msg.contains("normal"), "got: {msg}");
    }

    #[test]
    fn degenerate_capsule_segment_rejected() {
        // Distinct endpoints whose squared length underflows to zero.
        let mut b = SdfSceneBuilder::new();
        b.add_capsule_mut(Vec3::ZERO, Vec3::new(1e-30, 0.0, 0.0), 1.0, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("capsule"), "got: {msg}");

        // Endpoints far enough apart that the squared length overflows.
        let mut b = SdfSceneBuilder::new();
        b.add_capsule_mut(Vec3::ZERO, Vec3::splat(1e30), 1.0, 1);
        let msg = specialize(&b.build()).unwrap_err().to_string();
        assert!(msg.contains("capsule"), "got: {msg}");
    }

    #[test]
    fn supplied_bounds_must_contain_root() {
        let scene = build_sphere().with_bounds(Vec3::ZERO, Vec3::splat(1.0));
        let msg = specialize(&scene).unwrap_err().to_string();
        assert!(msg.contains("bounds"), "got: {msg}");

        // The sphere at (0,8,0) r=3 computes the root bound [-3,3]x[5,11]x[-3,3].
        let scene = build_sphere().with_bounds(Vec3::splat(-20.0), Vec3::splat(20.0));
        specialize(&scene).expect("containing supplied bounds must be accepted");
    }
}
