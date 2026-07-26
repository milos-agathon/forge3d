use crate::camera::Anchor;
use glam::{DMat4, DVec3, DVec4, Vec3};

/// Coordinate mode used by a clipmap frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobeMode {
    Flat,
    Globe,
}

/// A camera-relative position and its local up direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraRelative {
    pub position: Vec3,
    pub up: Vec3,
}

/// Planetary f64 world frame with an f32 render-space boundary at the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobeFrame {
    radius: f64,
    anchor: Anchor,
    ecef_to_local: DMat4,
    mode: GlobeMode,
}

impl GlobeFrame {
    pub const WGS84_MEAN_RADIUS_M: f64 = 6_371_000.0;

    /// Create a spherical Earth frame using the WGS84 mean radius.
    pub fn globe(camera_anchor: DVec3) -> Option<Self> {
        Self::with_radius(Self::WGS84_MEAN_RADIUS_M, camera_anchor)
    }

    /// Create a spherical planetary frame with a caller-supplied radius.
    pub fn with_radius(radius: f64, camera_anchor: DVec3) -> Option<Self> {
        let camera_distance = camera_anchor.length();
        if !radius.is_finite()
            || radius <= 0.0
            || !camera_anchor.is_finite()
            || !camera_distance.is_finite()
            || camera_distance == 0.0
        {
            return None;
        }
        Some(Self {
            radius,
            anchor: anchor_at(camera_anchor)?,
            ecef_to_local: tangent_transform(camera_anchor),
            mode: GlobeMode::Globe,
        })
    }

    /// Create an identity local frame for the existing flat clipmap path.
    pub fn flat(camera_anchor: DVec3) -> Option<Self> {
        camera_anchor.is_finite().then(|| Self {
            radius: Self::WGS84_MEAN_RADIUS_M,
            anchor: anchor_at(camera_anchor).expect("finite anchor is valid"),
            ecef_to_local: DMat4::IDENTITY,
            mode: GlobeMode::Flat,
        })
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn mode(&self) -> GlobeMode {
        self.mode
    }

    pub fn camera_anchor(&self) -> DVec3 {
        self.anchor.origin()
    }

    /// Return the same frame mode and radius anchored at a new camera.
    pub fn reanchored(&self, camera_anchor: DVec3) -> Option<Self> {
        match self.mode {
            GlobeMode::Flat => Self::flat(camera_anchor),
            GlobeMode::Globe => Self::with_radius(self.radius, camera_anchor),
        }
    }

    /// Convert longitude/latitude in degrees and altitude in metres to ECEF.
    pub fn lonlat_alt_to_ecef(&self, lon_deg: f64, lat_deg: f64, altitude_m: f64) -> Option<DVec3> {
        if !lon_deg.is_finite()
            || !lat_deg.is_finite()
            || !altitude_m.is_finite()
            || !(-90.0..=90.0).contains(&lat_deg)
            || altitude_m <= -self.radius
        {
            return None;
        }
        let lon = lon_deg.to_radians();
        let lat = lat_deg.to_radians();
        let radius = self.radius + altitude_m;
        let cos_lat = lat.cos();
        Some(DVec3::new(
            radius * cos_lat * lon.cos(),
            radius * cos_lat * lon.sin(),
            radius * lat.sin(),
        ))
    }

    /// Convert ECEF to `(longitude degrees, latitude degrees, altitude metres)`.
    pub fn ecef_to_lonlat_alt(&self, ecef: DVec3) -> Option<DVec3> {
        if !ecef.is_finite() {
            return None;
        }
        let distance = ecef.length();
        if distance == 0.0 {
            return None;
        }
        Some(DVec3::new(
            ecef.y.atan2(ecef.x).to_degrees(),
            (ecef.z / distance).clamp(-1.0, 1.0).asin().to_degrees(),
            distance - self.radius,
        ))
    }

    /// Rotate an ECEF vector into the camera's local east/north/up frame.
    pub fn ecef_to_local_vector(&self, vector: DVec3) -> DVec3 {
        self.ecef_to_local.transform_vector3(vector)
    }

    /// Build a stable local-east/north/up to ECEF transform at a world point.
    pub(crate) fn tangent_to_ecef(origin: DVec3) -> Option<DMat4> {
        let distance = origin.length();
        (origin.is_finite() && distance.is_finite() && distance > 0.0)
            .then(|| tangent_transform(origin).transpose())
    }

    /// Subtract the f64 camera anchor before the sole f32 render conversion.
    pub fn camera_relative(&self, ecef: DVec3) -> Option<CameraRelative> {
        if !ecef.is_finite() {
            return None;
        }
        let globe_distance = (self.mode == GlobeMode::Globe).then(|| ecef.length());
        if globe_distance.is_some_and(|distance| !distance.is_finite() || distance == 0.0) {
            return None;
        }
        let local = self.ecef_to_local_vector(ecef - self.anchor.origin());
        let position = self.anchor.to_render_vec3(self.anchor.origin() + local);
        let up = match self.mode {
            GlobeMode::Flat => Vec3::Z,
            GlobeMode::Globe => {
                let local_up = self.ecef_to_local_vector(ecef.normalize());
                Anchor::direction_to_render(local_up).normalize()
            }
        };
        Some(CameraRelative { position, up })
    }
}

fn anchor_at(origin: DVec3) -> Option<Anchor> {
    if !origin.is_finite() {
        return None;
    }
    let mut anchor = Anchor::try_with_epsilon(f64::MIN_POSITIVE)?;
    let _ = anchor.rebase_if_needed(origin);
    Some(anchor)
}

fn tangent_transform(camera_anchor: DVec3) -> DMat4 {
    let up = camera_anchor.normalize();
    let lon = up.y.atan2(up.x);
    let lat = up.z.clamp(-1.0, 1.0).asin();
    let east = DVec3::new(-lon.sin(), lon.cos(), 0.0);
    let north = DVec3::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos());
    DMat4::from_cols(
        east.extend(0.0),
        north.extend(0.0),
        up.extend(0.0),
        DVec4::W,
    )
    .transpose()
}

#[cfg(test)]
mod tests {
    use super::{GlobeFrame, GlobeMode};
    use glam::DVec3;

    const EPSILON: f64 = 1.0e-9;

    #[test]
    fn default_radius_and_cardinal_ecef_points_are_exact() {
        let frame = GlobeFrame::globe(DVec3::new(6_371_000.0, 0.0, 0.0)).unwrap();
        assert_eq!(frame.radius(), GlobeFrame::WGS84_MEAN_RADIUS_M);
        assert_eq!(frame.mode(), GlobeMode::Globe);
        assert!(
            (frame.lonlat_alt_to_ecef(0.0, 0.0, 0.0).unwrap() - DVec3::new(6_371_000.0, 0.0, 0.0))
                .length()
                < EPSILON
        );
        assert!(
            (frame.lonlat_alt_to_ecef(90.0, 0.0, 0.0).unwrap() - DVec3::new(0.0, 6_371_000.0, 0.0))
                .length()
                < EPSILON
        );
        assert!(
            (frame.lonlat_alt_to_ecef(0.0, 90.0, 0.0).unwrap() - DVec3::new(0.0, 0.0, 6_371_000.0))
                .length()
                < EPSILON
        );
    }

    #[test]
    fn geodetic_round_trip_is_better_than_one_part_per_million() {
        let frame = GlobeFrame::globe(DVec3::new(6_371_000.0, 0.0, 0.0)).unwrap();
        for geodetic in [
            DVec3::new(-121.7603, 46.8523, 4_392.0),
            DVec3::new(179.999, -80.0, 408_000.0),
            DVec3::new(-179.999, 0.0, -400.0),
        ] {
            let ecef = frame
                .lonlat_alt_to_ecef(geodetic.x, geodetic.y, geodetic.z)
                .unwrap();
            let actual = frame.ecef_to_lonlat_alt(ecef).unwrap();
            assert!((actual.x - geodetic.x).abs() < 1.0e-9);
            assert!((actual.y - geodetic.y).abs() < 1.0e-9);
            assert!((actual.z - geodetic.z).abs() < 1.0e-6);
        }
    }

    #[test]
    fn camera_relative_subtraction_preserves_small_offsets_at_planet_scale() {
        let camera = DVec3::new(6_371_000.0 + 408_000.0, 25.0, -10.0);
        let frame = GlobeFrame::globe(camera).unwrap();
        let point = camera + DVec3::new(0.000_25, 2.0, -1.0);
        let relative = frame.camera_relative(point).unwrap();
        let truth = frame.ecef_to_local_vector(point - camera);
        assert!((relative.position.as_dvec3() - truth).length() < 1.0e-6);
        assert!((relative.up.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn flat_mode_is_identity_camera_relative_space() {
        let camera = DVec3::new(125.0, -75.0, 12.0);
        let frame = GlobeFrame::flat(camera).unwrap();
        let point = camera + DVec3::new(3.5, -2.0, 9.0);
        let relative = frame.camera_relative(point).unwrap();
        assert_eq!(frame.mode(), GlobeMode::Flat);
        assert!((relative.position.as_dvec3() - (point - camera)).length() < 1.0e-6);
        assert_eq!(relative.up, glam::Vec3::Z);
    }

    #[test]
    fn invalid_world_inputs_are_rejected() {
        assert!(GlobeFrame::with_radius(0.0, DVec3::X).is_none());
        assert!(GlobeFrame::with_radius(f64::NAN, DVec3::X).is_none());
        assert!(GlobeFrame::globe(DVec3::ZERO).is_none());
        assert!(GlobeFrame::globe(DVec3::new(f64::NAN, 0.0, 0.0)).is_none());
        assert!(GlobeFrame::globe(DVec3::splat(f64::MAX)).is_none());

        let frame = GlobeFrame::globe(DVec3::new(6_371_000.0, 0.0, 0.0)).unwrap();
        assert!(frame.lonlat_alt_to_ecef(f64::NAN, 0.0, 0.0).is_none());
        assert!(frame.lonlat_alt_to_ecef(0.0, 91.0, 0.0).is_none());
        assert!(frame.ecef_to_lonlat_alt(DVec3::ZERO).is_none());
        assert!(frame.camera_relative(DVec3::ZERO).is_none());
        assert!(frame.camera_relative(DVec3::splat(f64::INFINITY)).is_none());
    }
}
