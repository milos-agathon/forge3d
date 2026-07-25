// src/camera/anchor.rs
// MENSURA: camera-relative anchoring — the f64→f32 cliff, moved on purpose.
//
// World coordinates stay f64 until the last possible instant. `Anchor` holds
// an f64 world origin, rebased whenever the camera moves more than
// `anchor_epsilon` (default 1 km) away from it; the view matrix is built from
// anchor-relative positions and per-object model matrices carry the
// anchor-relative object origin. Narrowing happens in exactly ONE place —
// `Anchor::narrow`, used only by `to_render_*` — and that single `as f32`
// site is grep-gated by tests/test_world_coord_f32_gate.py.
// RELEVANT FILES: src/geo/units.rs, src/scene/py_api/base.rs, src/camera/mod.rs

use glam::{DVec3, Mat4, Vec3};

use crate::core::dd::DDVec3;
use crate::geo::units::{Coord, CrsTag, EpochTag};

/// An f64 world-space origin that render-space f32 values are measured from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    origin: DVec3,
    anchor_epsilon: f64,
}

impl Default for Anchor {
    fn default() -> Self {
        Self::new()
    }
}

impl Anchor {
    /// Default rebase threshold: 1 km. At that offset an f32 relative
    /// coordinate still resolves ~0.06 mm.
    pub const DEFAULT_EPSILON_M: f64 = 1_000.0;

    pub fn new() -> Self {
        Self {
            origin: DVec3::ZERO,
            anchor_epsilon: Self::DEFAULT_EPSILON_M,
        }
    }

    /// Checked public threshold constructor. Invalid values are rejected in
    /// every build profile; there is no release-only unchecked path.
    pub fn with_epsilon(anchor_epsilon: f64) -> Option<Self> {
        (anchor_epsilon.is_finite() && anchor_epsilon > 0.0).then_some(Self {
            origin: DVec3::ZERO,
            anchor_epsilon,
        })
    }

    /// Checked epsilon constructor for trust boundaries: rejects a non-finite
    /// or non-positive rebase threshold rather than silently accepting a
    /// degenerate anchor that would never (or always) rebase.
    pub fn try_with_epsilon(anchor_epsilon: f64) -> Option<Self> {
        Self::with_epsilon(anchor_epsilon)
    }

    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    pub fn anchor_epsilon(&self) -> f64 {
        self.anchor_epsilon
    }

    /// Rebase the anchor onto the camera eye when it has drifted more than
    /// `anchor_epsilon` from the current origin. Returns true on rebase (the
    /// caller must then refresh every model offset derived from this anchor).
    pub fn rebase_if_needed(&mut self, eye: DVec3) -> bool {
        // A non-finite camera position is an upstream bug; never poison the
        // f64 anchor origin with it (that would silently NaN every offset).
        if !eye.is_finite() {
            return false;
        }
        if (eye - self.origin).length() > self.anchor_epsilon {
            self.origin = eye;
            true
        } else {
            false
        }
    }

    /// The ONLY sanctioned f64→f32 narrowing of a world coordinate in the
    /// codebase (single textual `as f32`, enforced at source level).
    #[inline]
    fn narrow(value: f64) -> f32 {
        value as f32
    }

    /// Narrow an anchor-relative world position to render-space f32.
    pub fn to_render_vec3(&self, p: DVec3) -> Vec3 {
        let rel = p - self.origin;
        Vec3::new(
            Self::narrow(rel.x),
            Self::narrow(rel.y),
            Self::narrow(rel.z),
        )
    }

    /// Narrow a unitless direction at the render boundary. Directions are not
    /// world positions, so they are not rebased; they still use the one
    /// sanctioned narrowing implementation.
    pub fn to_render_direction(&self, direction: DVec3) -> Vec3 {
        Self::direction_to_render(direction)
    }

    /// Narrow a non-position direction or dimension without constructing a
    /// throwaway anchor. Viewer code must use the active frame anchor for
    /// positions; this associated helper exists only for translation-invariant
    /// quantities such as colors, UVs, normals, and physical spans.
    pub fn direction_to_render(direction: DVec3) -> Vec3 {
        Vec3::new(
            Self::narrow(direction.x),
            Self::narrow(direction.y),
            Self::narrow(direction.z),
        )
    }

    /// Typed entry point: the single named function where a `Coord` leaves
    /// the f64 world and becomes a render-space `Vec3`.
    pub fn to_render_f32<C: CrsTag, E: EpochTag>(&self, p: Coord<C, E>) -> Vec3 {
        self.to_render_vec3(p.raw())
    }

    /// Right-handed look-at view matrix built from anchor-relative eye and
    /// target. The subtraction happens in f64; only the small relative
    /// vectors are narrowed.
    pub fn view_look_at(&self, eye: DVec3, target: DVec3, up: Vec3) -> Mat4 {
        Mat4::look_at_rh(self.to_render_vec3(eye), self.to_render_vec3(target), up)
    }

    /// Anchor-relative translation an object's model matrix must carry for
    /// geometry authored relative to `object_origin`.
    pub fn model_offset(&self, object_origin: DVec3) -> Vec3 {
        self.to_render_vec3(object_origin)
    }

    /// Split an absolute f64 world position into a normalized double-float
    /// pair per component without discarding the f64 residual.
    pub fn to_dd(&self, p: DVec3) -> DDVec3 {
        DDVec3::from_dvec3(p)
    }

    /// Restore a render-space coordinate already represented as f64 (for
    /// example, a CPU pick widened from the f32 render BVH) to world space.
    pub fn to_world_from_render_f64(&self, render: DVec3) -> DVec3 {
        self.origin + render
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::units::{Coord, Ecef, Itrf2014};

    #[test]
    fn anchor_defaults_to_world_origin_and_identity_behaviour() {
        let anchor = Anchor::new();
        let eye = DVec3::new(3.0, 2.0, 3.0);
        let view = anchor.view_look_at(eye, DVec3::ZERO, Vec3::Y);
        let legacy = Mat4::look_at_rh(Vec3::new(3.0, 2.0, 3.0), Vec3::ZERO, Vec3::Y);
        assert!(
            (view
                .to_cols_array()
                .iter()
                .zip(legacy.to_cols_array().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max))
                < 1e-7
        );
    }

    #[test]
    fn rebase_triggers_only_beyond_epsilon() {
        let mut anchor = Anchor::new();
        assert!(!anchor.rebase_if_needed(DVec3::new(999.0, 0.0, 0.0)));
        assert_eq!(anchor.origin(), DVec3::ZERO);
        assert!(anchor.rebase_if_needed(DVec3::new(1_500.0, 0.0, 0.0)));
        assert_eq!(anchor.origin(), DVec3::new(1_500.0, 0.0, 0.0));
    }

    #[test]
    fn try_with_epsilon_rejects_degenerate_thresholds() {
        assert!(Anchor::try_with_epsilon(1_000.0).is_some());
        assert!(Anchor::try_with_epsilon(0.0).is_none());
        assert!(Anchor::try_with_epsilon(-5.0).is_none());
        assert!(Anchor::try_with_epsilon(f64::NAN).is_none());
        assert!(Anchor::try_with_epsilon(f64::INFINITY).is_none());
    }

    #[test]
    fn exact_threshold_does_not_rebase_but_next_f64_does() {
        let mut anchor = Anchor::with_epsilon(1_000.0).unwrap();
        let below = f64::from_bits(1_000.0_f64.to_bits() - 1);
        assert!(!anchor.rebase_if_needed(DVec3::new(below, 0.0, 0.0)));
        assert!(!anchor.rebase_if_needed(DVec3::new(1_000.0, 0.0, 0.0)));
        let next = f64::from_bits(1_000.0_f64.to_bits() + 1);
        assert!(anchor.rebase_if_needed(DVec3::new(next, 0.0, 0.0)));
        assert!(!anchor.rebase_if_needed(DVec3::new(next, 0.0, 0.0)));
    }

    #[test]
    fn repeated_threshold_crossings_publish_at_most_one_rebase_per_focus() {
        let mut anchor = Anchor::with_epsilon(1_000.0).unwrap();
        let mut count = 0;
        for focus in [1_001.0, 1_001.0, 2_002.0, 2_002.0, 3_003.0, 3_003.0] {
            count += usize::from(anchor.rebase_if_needed(DVec3::new(focus, 0.0, 0.0)));
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn rebase_ignores_non_finite_eye_and_keeps_origin_valid() {
        let mut anchor = Anchor::new();
        anchor.rebase_if_needed(DVec3::new(2_000.0, 0.0, 0.0));
        let before = anchor.origin();
        assert!(!anchor.rebase_if_needed(DVec3::new(f64::NAN, 0.0, 0.0)));
        assert_eq!(anchor.origin(), before);
        assert!(anchor.origin().is_finite());
    }

    #[test]
    fn stationary_object_stays_accurate_across_a_rebase_at_ecef_scale() {
        // A fixed ECEF object renders to its true camera-relative offset both
        // before and after the camera crosses a 1 km rebase boundary, because
        // the offset is recomputed against the (new) f64 origin each time.
        let object = DVec3::new(6_378_137.0 + 3.0, 100.0, -50.0);
        let mut anchor = Anchor::new();
        anchor.rebase_if_needed(DVec3::new(6_378_137.0, 0.0, 0.0));
        let before = anchor.to_render_vec3(object);
        assert!((before.as_dvec3() - (object - anchor.origin())).length() < 1e-3);

        assert!(anchor.rebase_if_needed(DVec3::new(6_378_137.0 + 1_500.0, 0.0, 0.0)));
        let after = anchor.to_render_vec3(object);
        assert!((after.as_dvec3() - (object - anchor.origin())).length() < 1e-3);
    }

    #[test]
    fn repeated_kilometre_rebases_keep_a_nearby_point_submillimetre() {
        // Walk the camera 10 km in ~1 km steps at ECEF scale; a point 2 m from
        // the camera stays sub-mm accurate at every step (UTM/ECEF magnitudes).
        let mut anchor = Anchor::new();
        let mut cam = DVec3::new(6_378_137.0, 500_000.0, 0.0);
        anchor.rebase_if_needed(cam);
        for _ in 0..10 {
            cam += DVec3::new(1_100.0, 0.0, 0.0);
            anchor.rebase_if_needed(cam);
            let near = cam + DVec3::new(2.0, -1.0, 0.5);
            let rel = anchor.to_render_vec3(near);
            let truth = near - anchor.origin();
            assert!((rel.as_dvec3() - truth).length() < 1e-3, "drift too large");
        }
    }

    #[test]
    fn anchored_narrowing_preserves_submillimetre_offsets_at_earth_radius() {
        // The whole point: 6.38e6 + 0.25 mm survives narrowing when measured
        // relative to a nearby anchor, and is destroyed without one.
        let base = DVec3::new(6_378_137.0, 0.0, 0.0);
        let mut anchor = Anchor::new();
        anchor.rebase_if_needed(base);
        let p = Coord::<Ecef, Itrf2014>::ecef(6_378_137.000_25, 0.0, 0.0);
        let rel = anchor.to_render_f32(p);
        assert!((rel.x - 0.000_25).abs() < 1e-6, "rel.x = {}", rel.x);
        // Unanchored narrowing of the same coordinate loses the offset
        // entirely (~0.5 m quantization at this magnitude).
        let unanchored = Anchor::new().to_render_f32(p);
        assert_eq!(unanchored.x, 6_378_137.0f32);
    }
}
