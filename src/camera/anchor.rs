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

    pub fn with_epsilon(anchor_epsilon: f64) -> Self {
        Self {
            origin: DVec3::ZERO,
            anchor_epsilon,
        }
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
