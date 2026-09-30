// src/camera/anchor.rs
// MENSURA: camera-relative anchoring — the f64→f32 cliff, moved on purpose.
//
// World coordinates stay f64 until the last possible instant. `Anchor` holds
// an f64 world origin, rebased whenever the camera moves more than
// `anchor_epsilon` (default 1 km) away from it; the view matrix is built from
// anchor-relative positions and per-object model matrices carry the
// anchor-relative object origin. Narrowing happens in exactly ONE place —
// `Anchor::to_render_f32`, which accepts only a typed `Coord` — and that
// single `as f32` site is grep-gated by tests/test_world_coord_f32_gate.py.
// RELEVANT FILES: src/geo/units.rs, src/scene/py_api/base.rs, src/camera/mod.rs
//
//! An untyped `DVec3` cannot be narrowed: it must first be tagged as a
//! position (`SceneCoord::scene`) or a displacement (`SceneOffset::scene`).
//!
//! ```
//! use forge3d::camera::Anchor;
//! use forge3d::geo::units::{SceneCoord, SceneOffset};
//! use glam::DVec3;
//! let anchor = Anchor::new();
//! let p = anchor.to_render_f32(SceneCoord::scene(DVec3::new(1.0, 2.0, 3.0)));
//! let d = Anchor::offset_to_render(SceneOffset::scene(DVec3::X));
//! assert_eq!((p.x, d.x), (1.0, 1.0));
//! ```
//!
//! ```compile_fail
//! use forge3d::camera::Anchor;
//! use glam::DVec3;
//! // ERROR: expected `Coord<_, _>`, found `DVec3`.
//! let _ = Anchor::new().to_render_f32(DVec3::new(1.0, 2.0, 3.0));
//! ```
//!
//! ```compile_fail
//! use forge3d::camera::Anchor;
//! use forge3d::geo::units::SceneOffset;
//! use glam::DVec3;
//! // ERROR: a displacement is not a position; it cannot be anchored.
//! let _ = Anchor::new().to_render_f32(SceneOffset::scene(DVec3::X));
//! ```

use glam::{DVec3, Mat4, Vec3};

use crate::core::dd::DDVec3;
use crate::geo::units::{Coord, CoordOffset, CrsTag, EpochTag, Unreferenced};

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

    /// The single f64→f32 crossing for world coordinates in the codebase: a
    /// typed `Coord` leaves the f64 world as an anchor-relative render-space
    /// `Vec3`. The subtraction runs in f64; only the small relative vector is
    /// narrowed. This body holds the only textual `as f32` in this module,
    /// enforced by tests/test_world_coord_f32_gate.py.
    pub fn to_render_f32<C: CrsTag, E: EpochTag>(&self, p: Coord<C, E>) -> Vec3 {
        let rel = p.raw() - self.origin;
        Vec3::from_array(rel.to_array().map(|component| component as f32))
    }

    /// Narrow a translation-invariant displacement or direction (spans,
    /// normals, view directions). It reuses the one crossing above relative to
    /// an anchor at the frame origin, so `offset - 0.0` is exact and no second
    /// narrowing implementation exists.
    pub fn offset_to_render<C: CrsTag>(offset: CoordOffset<C>) -> Vec3 {
        Self::new().to_render_f32(Coord::<C, Unreferenced>::from_raw(offset.raw()))
    }

    /// Right-handed look-at view matrix built from anchor-relative eye and
    /// target positions.
    pub fn view_look_at<C: CrsTag, E: EpochTag>(
        &self,
        eye: Coord<C, E>,
        target: Coord<C, E>,
        up: Vec3,
    ) -> Mat4 {
        Mat4::look_at_rh(self.to_render_f32(eye), self.to_render_f32(target), up)
    }

    /// Anchor-relative translation an object's model matrix must carry for
    /// geometry authored relative to `object_origin`.
    pub fn model_offset<C: CrsTag, E: EpochTag>(&self, object_origin: Coord<C, E>) -> Vec3 {
        self.to_render_f32(object_origin)
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
    use crate::geo::units::{Coord, Ecef, Itrf2014, SceneCoord, SceneOffset};

    #[test]
    fn anchor_defaults_to_world_origin_and_identity_behaviour() {
        let anchor = Anchor::new();
        let eye = DVec3::new(3.0, 2.0, 3.0);
        let view = anchor.view_look_at(
            SceneCoord::scene(eye),
            SceneCoord::scene(DVec3::ZERO),
            Vec3::Y,
        );
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
        let before = anchor.to_render_f32(SceneCoord::scene(object));
        assert!((before.as_dvec3() - (object - anchor.origin())).length() < 1e-3);

        assert!(anchor.rebase_if_needed(DVec3::new(6_378_137.0 + 1_500.0, 0.0, 0.0)));
        let after = anchor.to_render_f32(SceneCoord::scene(object));
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
            let rel = anchor.to_render_f32(SceneCoord::scene(near));
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

    #[test]
    fn offset_narrowing_is_translation_invariant_and_exact_at_zero_origin() {
        let mut anchor = Anchor::new();
        anchor.rebase_if_needed(DVec3::new(6_378_137.0, 0.0, 0.0));
        let span = DVec3::new(1_234.5, -0.0, 7.25);
        let narrowed = Anchor::offset_to_render(SceneOffset::scene(span));
        assert_eq!(narrowed.to_array(), [1_234.5f32, -0.0, 7.25]);
        assert!(narrowed.y.is_sign_negative());
    }

    #[test]
    fn typed_crossing_matches_componentwise_f64_subtraction() {
        let mut anchor = Anchor::new();
        anchor.rebase_if_needed(DVec3::new(500_000.25, 4_649_776.5, 1_000.0));
        let p = DVec3::new(500_123.125, 4_649_700.75, 987.5);
        let rel = p - anchor.origin();
        let render = anchor.to_render_f32(SceneCoord::scene(p));
        assert_eq!(
            render.to_array(),
            [rel.x as f32, rel.y as f32, rel.z as f32]
        );
    }
}
