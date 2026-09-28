use crate::camera::Anchor;
use glam::{DMat4, DVec3, DVec4, Vec3};
use std::error::Error;
use std::fmt;

/// Checked failures at the geodetic/ECEF trust boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobeFrameError {
    InvalidRadius,
    NonFiniteCameraAnchor,
    ZeroCameraAnchor,
    CameraAnchorOutOfRange,
    AnchorUnrepresentable,
    NonFiniteLongitude,
    NonFiniteLatitude,
    LatitudeOutOfRange,
    NonFiniteAltitude,
    AltitudeAtOrBelowCenter,
    NonFiniteEcef,
    ZeroEcef,
    EcefOutOfRange,
}

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

    /// Create a spherical planetary frame with a caller-supplied radius.
    pub fn globe(radius: f64, camera_anchor: DVec3) -> Result<Self, GlobeFrameError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(GlobeFrameError::InvalidRadius);
        }
        if !camera_anchor.is_finite() {
            return Err(GlobeFrameError::NonFiniteCameraAnchor);
        }
        let camera_distance = camera_anchor.length();
        if camera_distance == 0.0 {
            return Err(GlobeFrameError::ZeroCameraAnchor);
        }
        if !camera_distance.is_finite() {
            return Err(GlobeFrameError::CameraAnchorOutOfRange);
        }
        Ok(Self {
            radius,
            anchor: anchor_at(camera_anchor)?,
            ecef_to_local: tangent_transform(camera_anchor),
            mode: GlobeMode::Globe,
        })
    }

    /// Create an identity local frame for the existing flat clipmap path.
    pub fn flat(camera_anchor: DVec3) -> Result<Self, GlobeFrameError> {
        if !camera_anchor.is_finite() {
            return Err(GlobeFrameError::NonFiniteCameraAnchor);
        }
        if !camera_anchor.length().is_finite() {
            return Err(GlobeFrameError::CameraAnchorOutOfRange);
        }
        Ok(Self {
            radius: Self::WGS84_MEAN_RADIUS_M,
            anchor: anchor_at(camera_anchor)?,
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
    pub fn reanchored(&self, camera_anchor: DVec3) -> Result<Self, GlobeFrameError> {
        if !camera_anchor.is_finite() {
            return Err(GlobeFrameError::NonFiniteCameraAnchor);
        }
        match self.mode {
            GlobeMode::Flat => Self::flat(camera_anchor),
            GlobeMode::Globe => Self::globe(self.radius, camera_anchor),
        }
    }

    /// Convert longitude/latitude in degrees and altitude in metres to ECEF.
    pub fn lonlat_alt_to_ecef(
        &self,
        lon_deg: f64,
        lat_deg: f64,
        altitude_m: f64,
    ) -> Result<DVec3, GlobeFrameError> {
        if !lon_deg.is_finite() {
            return Err(GlobeFrameError::NonFiniteLongitude);
        }
        if !lat_deg.is_finite() {
            return Err(GlobeFrameError::NonFiniteLatitude);
        }
        if !(-90.0..=90.0).contains(&lat_deg) {
            return Err(GlobeFrameError::LatitudeOutOfRange);
        }
        if !altitude_m.is_finite() {
            return Err(GlobeFrameError::NonFiniteAltitude);
        }
        if altitude_m <= -self.radius {
            return Err(GlobeFrameError::AltitudeAtOrBelowCenter);
        }
        let lon = lon_deg.to_radians();
        let lat = lat_deg.to_radians();
        let radius = self.radius + altitude_m;
        let cos_lat = lat.cos();
        let ecef = DVec3::new(
            radius * cos_lat * lon.cos(),
            radius * cos_lat * lon.sin(),
            radius * lat.sin(),
        );
        ecef.is_finite()
            .then_some(ecef)
            .ok_or(GlobeFrameError::EcefOutOfRange)
    }

    /// Convert ECEF to `(longitude degrees, latitude degrees, altitude metres)`.
    pub fn ecef_to_lonlat_alt(&self, ecef: DVec3) -> Result<DVec3, GlobeFrameError> {
        if !ecef.is_finite() {
            return Err(GlobeFrameError::NonFiniteEcef);
        }
        let distance = ecef.length();
        if distance == 0.0 {
            return Err(GlobeFrameError::ZeroEcef);
        }
        if !distance.is_finite() {
            return Err(GlobeFrameError::EcefOutOfRange);
        }
        Ok(DVec3::new(
            ecef.y.atan2(ecef.x).to_degrees(),
            (ecef.z / distance).clamp(-1.0, 1.0).asin().to_degrees(),
            distance - self.radius,
        ))
    }

    /// Rotate an ECEF vector into the camera's local east/north/up frame.
    pub fn ecef_to_local_vector(&self, vector: DVec3) -> Result<DVec3, GlobeFrameError> {
        if !vector.is_finite() {
            return Err(GlobeFrameError::NonFiniteEcef);
        }
        if !vector.length().is_finite() {
            return Err(GlobeFrameError::EcefOutOfRange);
        }
        let local = self.ecef_to_local.transform_vector3(vector);
        local
            .is_finite()
            .then_some(local)
            .ok_or(GlobeFrameError::EcefOutOfRange)
    }

    /// Build a stable local-east/north/up to ECEF transform at a world point.
    pub(crate) fn tangent_to_ecef(origin: DVec3) -> Option<DMat4> {
        let distance = origin.length();
        (origin.is_finite() && distance.is_finite() && distance > 0.0)
            .then(|| tangent_transform(origin).transpose())
    }

    /// Subtract the f64 camera anchor before the sole f32 render conversion.
    pub fn camera_relative(&self, ecef: DVec3) -> Result<CameraRelative, GlobeFrameError> {
        if !ecef.is_finite() {
            return Err(GlobeFrameError::NonFiniteEcef);
        }
        if self.mode == GlobeMode::Globe {
            let distance = ecef.length();
            if distance == 0.0 {
                return Err(GlobeFrameError::ZeroEcef);
            }
            if !distance.is_finite() {
                return Err(GlobeFrameError::EcefOutOfRange);
            }
        }
        let local = self.ecef_to_local_vector(ecef - self.anchor.origin())?;
        let position = self
            .anchor
            .to_render_f32(crate::geo::units::SceneCoord::scene(
                self.anchor.origin() + local,
            ));
        if !local.is_finite() || !position.is_finite() {
            return Err(GlobeFrameError::EcefOutOfRange);
        }
        let up = match self.mode {
            GlobeMode::Flat => Vec3::Z,
            GlobeMode::Globe => {
                let local_up = self.ecef_to_local_vector(ecef.normalize())?;
                Anchor::offset_to_render(crate::geo::units::SceneOffset::scene(local_up))
                    .normalize()
            }
        };
        if !up.is_finite() {
            return Err(GlobeFrameError::EcefOutOfRange);
        }
        Ok(CameraRelative { position, up })
    }
}

impl fmt::Display for GlobeFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidRadius => "globe radius must be finite and positive",
            Self::NonFiniteCameraAnchor => "camera anchor must contain only finite ECEF values",
            Self::ZeroCameraAnchor => "globe camera anchor must be non-zero",
            Self::CameraAnchorOutOfRange => "camera anchor magnitude is out of range",
            Self::AnchorUnrepresentable => "camera anchor could not be represented",
            Self::NonFiniteLongitude => "longitude must be finite",
            Self::NonFiniteLatitude => "latitude must be finite",
            Self::LatitudeOutOfRange => "latitude must be in -90..=90 degrees",
            Self::NonFiniteAltitude => "altitude must be finite",
            Self::AltitudeAtOrBelowCenter => "altitude must be greater than negative globe radius",
            Self::NonFiniteEcef => "ECEF position must contain only finite values",
            Self::ZeroEcef => "globe ECEF position must be non-zero",
            Self::EcefOutOfRange => "ECEF position magnitude is out of range",
        };
        formatter.write_str(message)
    }
}

impl Error for GlobeFrameError {}

fn anchor_at(origin: DVec3) -> Result<Anchor, GlobeFrameError> {
    let mut anchor = Anchor::try_with_epsilon(f64::MIN_POSITIVE)
        .ok_or(GlobeFrameError::AnchorUnrepresentable)?;
    let _ = anchor.rebase_if_needed(origin);
    (anchor.origin() == origin)
        .then_some(anchor)
        .ok_or(GlobeFrameError::AnchorUnrepresentable)
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
    use super::{GlobeFrame, GlobeFrameError, GlobeMode};
    use glam::DVec3;
    use std::fmt::Debug;
    use std::panic::{catch_unwind, UnwindSafe};

    const EPSILON: f64 = 1.0e-9;

    fn assert_non_panicking_error<T: Debug>(
        expected: GlobeFrameError,
        operation: impl FnOnce() -> Result<T, GlobeFrameError> + UnwindSafe,
    ) {
        match catch_unwind(operation) {
            Ok(Err(actual)) => assert_eq!(actual, expected),
            Ok(Ok(value)) => panic!("invalid globe input unexpectedly succeeded: {value:?}"),
            Err(_) => panic!("invalid globe input panicked instead of returning {expected}"),
        }
    }

    #[test]
    fn every_public_invalid_input_returns_a_specific_error_without_panicking() {
        assert_non_panicking_error(GlobeFrameError::InvalidRadius, || {
            GlobeFrame::globe(f64::NAN, DVec3::X)
        });
        assert_non_panicking_error(GlobeFrameError::InvalidRadius, || {
            GlobeFrame::globe(f64::INFINITY, DVec3::X)
        });
        assert_non_panicking_error(GlobeFrameError::InvalidRadius, || {
            GlobeFrame::globe(0.0, DVec3::X)
        });
        assert_non_panicking_error(GlobeFrameError::InvalidRadius, || {
            GlobeFrame::globe(-1.0, DVec3::X)
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteCameraAnchor, || {
            GlobeFrame::flat(DVec3::new(f64::NAN, 0.0, 0.0))
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteCameraAnchor, || {
            GlobeFrame::flat(DVec3::new(f64::NEG_INFINITY, 0.0, 0.0))
        });
        assert_non_panicking_error(GlobeFrameError::CameraAnchorOutOfRange, || {
            GlobeFrame::flat(DVec3::splat(f64::MAX))
        });
        assert_non_panicking_error(GlobeFrameError::ZeroCameraAnchor, || {
            GlobeFrame::globe(1.0, DVec3::ZERO)
        });
        assert_non_panicking_error(GlobeFrameError::ZeroCameraAnchor, || {
            GlobeFrame::globe(1.0, DVec3::splat(f64::MIN_POSITIVE))
        });
        assert_non_panicking_error(GlobeFrameError::CameraAnchorOutOfRange, || {
            GlobeFrame::globe(1.0, DVec3::splat(f64::MAX))
        });

        let frame = GlobeFrame::globe(GlobeFrame::WGS84_MEAN_RADIUS_M, DVec3::X).unwrap();
        assert_non_panicking_error(GlobeFrameError::NonFiniteCameraAnchor, || {
            frame.reanchored(DVec3::splat(f64::INFINITY))
        });
        assert_non_panicking_error(GlobeFrameError::ZeroCameraAnchor, || {
            frame.reanchored(DVec3::ZERO)
        });
        assert_non_panicking_error(GlobeFrameError::CameraAnchorOutOfRange, || {
            frame.reanchored(DVec3::splat(f64::MAX))
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteLongitude, || {
            frame.lonlat_alt_to_ecef(f64::NAN, 0.0, 0.0)
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteLatitude, || {
            frame.lonlat_alt_to_ecef(0.0, f64::INFINITY, 0.0)
        });
        assert_non_panicking_error(GlobeFrameError::LatitudeOutOfRange, || {
            frame.lonlat_alt_to_ecef(0.0, 90.000_001, 0.0)
        });
        assert_non_panicking_error(GlobeFrameError::LatitudeOutOfRange, || {
            frame.lonlat_alt_to_ecef(0.0, -90.000_001, 0.0)
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteAltitude, || {
            frame.lonlat_alt_to_ecef(0.0, 0.0, f64::NEG_INFINITY)
        });
        assert_non_panicking_error(GlobeFrameError::AltitudeAtOrBelowCenter, || {
            frame.lonlat_alt_to_ecef(0.0, 0.0, -GlobeFrame::WGS84_MEAN_RADIUS_M)
        });
        assert_non_panicking_error(GlobeFrameError::AltitudeAtOrBelowCenter, || {
            frame.lonlat_alt_to_ecef(0.0, 0.0, -GlobeFrame::WGS84_MEAN_RADIUS_M - 1.0)
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteEcef, || {
            frame.ecef_to_lonlat_alt(DVec3::splat(f64::NAN))
        });
        assert_non_panicking_error(GlobeFrameError::ZeroEcef, || {
            frame.ecef_to_lonlat_alt(DVec3::ZERO)
        });
        assert_non_panicking_error(GlobeFrameError::ZeroEcef, || {
            frame.ecef_to_lonlat_alt(DVec3::splat(f64::MIN_POSITIVE))
        });
        assert_non_panicking_error(GlobeFrameError::EcefOutOfRange, || {
            frame.ecef_to_lonlat_alt(DVec3::splat(f64::MAX))
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteEcef, || {
            frame.ecef_to_local_vector(DVec3::splat(f64::NAN))
        });
        assert_non_panicking_error(GlobeFrameError::EcefOutOfRange, || {
            frame.ecef_to_local_vector(DVec3::splat(f64::MAX))
        });
        assert_non_panicking_error(GlobeFrameError::NonFiniteEcef, || {
            frame.camera_relative(DVec3::splat(f64::INFINITY))
        });
        assert_non_panicking_error(GlobeFrameError::ZeroEcef, || {
            frame.camera_relative(DVec3::ZERO)
        });
        assert_non_panicking_error(GlobeFrameError::ZeroEcef, || {
            frame.camera_relative(DVec3::splat(f64::MIN_POSITIVE))
        });
        assert_non_panicking_error(GlobeFrameError::EcefOutOfRange, || {
            frame.camera_relative(DVec3::splat(f64::MAX))
        });

        let flat = GlobeFrame::flat(DVec3::ZERO).unwrap();
        assert_non_panicking_error(GlobeFrameError::NonFiniteCameraAnchor, || {
            flat.reanchored(DVec3::splat(f64::NAN))
        });
        assert_non_panicking_error(GlobeFrameError::CameraAnchorOutOfRange, || {
            flat.reanchored(DVec3::splat(f64::MAX))
        });
        assert_non_panicking_error(GlobeFrameError::EcefOutOfRange, || {
            flat.camera_relative(DVec3::splat(f64::MAX))
        });
    }

    #[test]
    fn default_radius_and_cardinal_ecef_points_are_exact() {
        let frame = GlobeFrame::globe(
            GlobeFrame::WGS84_MEAN_RADIUS_M,
            DVec3::new(6_371_000.0, 0.0, 0.0),
        )
        .unwrap();
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
        let frame = GlobeFrame::globe(
            GlobeFrame::WGS84_MEAN_RADIUS_M,
            DVec3::new(6_371_000.0, 0.0, 0.0),
        )
        .unwrap();
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
        let frame = GlobeFrame::globe(GlobeFrame::WGS84_MEAN_RADIUS_M, camera).unwrap();
        let point = camera + DVec3::new(0.000_25, 2.0, -1.0);
        let relative = frame.camera_relative(point).unwrap();
        let truth = frame.ecef_to_local_vector(point - camera).unwrap();
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
        assert!(GlobeFrame::globe(0.0, DVec3::X).is_err());
        assert!(GlobeFrame::globe(f64::NAN, DVec3::X).is_err());
        assert!(GlobeFrame::globe(1.0, DVec3::ZERO).is_err());
        assert!(GlobeFrame::globe(1.0, DVec3::new(f64::NAN, 0.0, 0.0)).is_err());
        assert!(GlobeFrame::globe(1.0, DVec3::splat(f64::MAX)).is_err());

        let frame = GlobeFrame::globe(
            GlobeFrame::WGS84_MEAN_RADIUS_M,
            DVec3::new(6_371_000.0, 0.0, 0.0),
        )
        .unwrap();
        assert!(frame.lonlat_alt_to_ecef(f64::NAN, 0.0, 0.0).is_err());
        assert!(frame.lonlat_alt_to_ecef(0.0, 91.0, 0.0).is_err());
        assert!(frame.ecef_to_lonlat_alt(DVec3::ZERO).is_err());
    }
}
