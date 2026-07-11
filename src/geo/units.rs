// src/geo/units.rs
// MENSURA: phantom-typed geodetic scalar algebra.
// Unit, height-system, CRS, and epoch mismatches are compile errors, not runtime bugs.
// RELEVANT FILES: src/geo/projections/mod.rs, src/geo/geoid.rs, src/camera/anchor.rs
//
//! The only sanctioned way to turn a world coordinate into render-space `f32`
//! is `crate::camera::anchor::Anchor::to_render_f32` — a grep-gated invariant
//! (tests/test_world_coord_f32_gate.py).
//!
//! Same-tag arithmetic compiles (this example also proves the imports used
//! by the `compile_fail` blocks below are valid, so those blocks fail for
//! the right reason):
//!
//! ```
//! use forge3d::geo::units::{Angle, Coord, Degree, Ecef, Height, Itrf2014, Length, Metre, Wgs84};
//! let d = Length::<Metre>::new(2.0) + Length::<Metre>::new(0.5);
//! assert_eq!(d.value(), 2.5);
//! let a = Angle::<Degree>::new(1.5);
//! let h = Height::<forge3d::geo::units::Ellipsoidal>::new(10.0);
//! let c = Coord::<Wgs84, Itrf2014>::geographic(a, Angle::new(52.5), h);
//! let e = Coord::<Ecef, Itrf2014>::ecef(4.0e6, 1.0e6, 4.8e6);
//! let off = e - e;
//! assert_eq!((off.dx, off.dy, off.dz), (0.0, 0.0, 0.0));
//! let _ = (c.lon(), c.lat(), c.height());
//! ```
//!
//! # Uncompilable errors (rustdoc `compile_fail` proofs)
//!
//! Adding a length to an angle does not compile:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Degree, Length, Metre};
//! let l = Length::<Metre>::new(1.0);
//! let a = Angle::<Degree>::new(1.0);
//! let _ = l + a; // ERROR: no `Add<Angle<Degree>>` for `Length<Metre>`
//! ```
//!
//! Mixing height systems does not compile — go through
//! `forge3d::geo::geoid::orthometric_to_ellipsoidal`:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Egm96, Ellipsoidal, Height, Orthometric};
//! let e = Height::<Ellipsoidal>::new(100.0);
//! let o = Height::<Orthometric<Egm96>>::new(100.0);
//! let _ = e - o; // ERROR: height systems differ
//! ```
//!
//! Differencing coordinates from different epochs does not compile — go
//! through `forge3d::geo::units::epoch_transform`:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Coord, Height, Itrf2000, Itrf2014, Wgs84};
//! let a = Coord::<Wgs84, Itrf2014>::geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0));
//! let b = Coord::<Wgs84, Itrf2000>::geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0));
//! let _ = a - b; // ERROR: reference epochs differ
//! ```
//!
//! Differencing coordinates from different CRSs does not compile:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Coord, Ecef, Height, Itrf2014, Wgs84};
//! let a = Coord::<Wgs84, Itrf2014>::geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0));
//! let b = Coord::<Ecef, Itrf2014>::ecef(4.0e6, 1.0e6, 4.8e6);
//! let _ = a - b; // ERROR: CRS tags differ
//! ```
//!
//! Casting a world coordinate to `f32` without an `Anchor` does not compile —
//! accessors return typed scalars, never a bare `f64`:
//!
//! ```compile_fail
//! use forge3d::geo::units::{Angle, Coord, Height, Itrf2014, Wgs84};
//! let c = Coord::<Wgs84, Itrf2014>::geographic(
//!     Angle::new(13.4), Angle::new(52.5), Height::new(0.0));
//! let _ = c.lon() as f32; // ERROR: `Angle<Degree>` is not a primitive
//! ```
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use glam::DVec3;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

/// Marker trait for linear units.
pub trait LengthUnit: Sealed + Copy + core::fmt::Debug + 'static {
    const METRES_PER_UNIT: f64;
    const SYMBOL: &'static str;
}

/// SI metre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metre {}
/// International foot (exactly 0.3048 m).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Foot {}
/// US survey foot (exactly 1200/3937 m).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum USSurveyFoot {}

impl Sealed for Metre {}
impl Sealed for Foot {}
impl Sealed for USSurveyFoot {}
impl LengthUnit for Metre {
    const METRES_PER_UNIT: f64 = 1.0;
    const SYMBOL: &'static str = "m";
}
impl LengthUnit for Foot {
    const METRES_PER_UNIT: f64 = 0.3048;
    const SYMBOL: &'static str = "ft";
}
impl LengthUnit for USSurveyFoot {
    const METRES_PER_UNIT: f64 = 1200.0 / 3937.0;
    const SYMBOL: &'static str = "ftUS";
}

/// A length in unit `U`. Arithmetic is only defined between identical units.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Length<U: LengthUnit> {
    value: f64,
    _unit: PhantomData<U>,
}

impl<U: LengthUnit> Length<U> {
    pub const fn new(value: f64) -> Self {
        Self {
            value,
            _unit: PhantomData,
        }
    }
    /// Numeric value in this unit.
    pub const fn value(self) -> f64 {
        self.value
    }
    /// Explicit unit conversion.
    pub fn to<V: LengthUnit>(self) -> Length<V> {
        Length::new(self.value * U::METRES_PER_UNIT / V::METRES_PER_UNIT)
    }
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }
}

impl<U: LengthUnit> Add for Length<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}
impl<U: LengthUnit> Sub for Length<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}
impl<U: LengthUnit> Neg for Length<U> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}
impl<U: LengthUnit> Mul<f64> for Length<U> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs)
    }
}
impl<U: LengthUnit> Div<f64> for Length<U> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.value / rhs)
    }
}

// ---------------------------------------------------------------------------
// Angle
// ---------------------------------------------------------------------------

/// Marker trait for angular units.
pub trait AngleUnit: Sealed + Copy + core::fmt::Debug + 'static {
    const RADIANS_PER_UNIT: f64;
    const SYMBOL: &'static str;
}

/// Sexagesimal degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Degree {}
/// SI radian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Radian {}
/// Gradian (400 per turn).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grad {}

impl Sealed for Degree {}
impl Sealed for Radian {}
impl Sealed for Grad {}
impl AngleUnit for Degree {
    const RADIANS_PER_UNIT: f64 = core::f64::consts::PI / 180.0;
    const SYMBOL: &'static str = "deg";
}
impl AngleUnit for Radian {
    const RADIANS_PER_UNIT: f64 = 1.0;
    const SYMBOL: &'static str = "rad";
}
impl AngleUnit for Grad {
    const RADIANS_PER_UNIT: f64 = core::f64::consts::PI / 200.0;
    const SYMBOL: &'static str = "gon";
}

/// An angle in unit `U`. Arithmetic is only defined between identical units.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Angle<U: AngleUnit> {
    value: f64,
    _unit: PhantomData<U>,
}

impl<U: AngleUnit> Angle<U> {
    pub const fn new(value: f64) -> Self {
        Self {
            value,
            _unit: PhantomData,
        }
    }
    pub const fn value(self) -> f64 {
        self.value
    }
    pub fn to<V: AngleUnit>(self) -> Angle<V> {
        Angle::new(self.value * U::RADIANS_PER_UNIT / V::RADIANS_PER_UNIT)
    }
    pub fn radians(self) -> f64 {
        self.value * U::RADIANS_PER_UNIT
    }
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }
}

impl<U: AngleUnit> Add for Angle<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}
impl<U: AngleUnit> Sub for Angle<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}
impl<U: AngleUnit> Neg for Angle<U> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}
impl<U: AngleUnit> Mul<f64> for Angle<U> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs)
    }
}
impl<U: AngleUnit> Div<f64> for Angle<U> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.value / rhs)
    }
}

// ---------------------------------------------------------------------------
// Height systems
// ---------------------------------------------------------------------------

/// Marker trait for geoid models a height can be referenced to.
pub trait GeoidModel: Sealed + Copy + core::fmt::Debug + 'static {
    const NAME: &'static str;
}

/// The EGM96 geoid (see `crate::geo::geoid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Egm96 {}
impl Sealed for Egm96 {}
impl GeoidModel for Egm96 {
    const NAME: &'static str = "EGM96";
}

/// Marker trait for vertical reference systems.
pub trait HeightSystem: Sealed + Copy + core::fmt::Debug + 'static {
    const NAME: &'static str;
}

/// Height above the reference ellipsoid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ellipsoidal {}
impl Sealed for Ellipsoidal {}
impl HeightSystem for Ellipsoidal {
    const NAME: &'static str = "ellipsoidal";
}

/// Height above the geoid `G` (what a DEM usually encodes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orthometric<G: GeoidModel> {
    _geoid: PhantomData<G>,
}
impl<G: GeoidModel> Sealed for Orthometric<G> {}
impl<G: GeoidModel> HeightSystem for Orthometric<G> {
    const NAME: &'static str = "orthometric";
}

/// Height above a nautical chart datum (type slot only; no conversions shipped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChartDatum {}
impl Sealed for ChartDatum {}
impl HeightSystem for ChartDatum {
    const NAME: &'static str = "chart_datum";
}

/// A height in metres referenced to system `S`. Heights in different systems
/// cannot be mixed; convert via `crate::geo::geoid`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Height<S: HeightSystem> {
    metres: f64,
    _system: PhantomData<S>,
}

impl<S: HeightSystem> Height<S> {
    pub const fn new(metres: f64) -> Self {
        Self {
            metres,
            _system: PhantomData,
        }
    }
    pub const fn metres(self) -> f64 {
        self.metres
    }
}

/// Raising/lowering a height by a metric length stays in the same system.
impl<S: HeightSystem> Add<Length<Metre>> for Height<S> {
    type Output = Self;
    fn add(self, rhs: Length<Metre>) -> Self {
        Self::new(self.metres + rhs.value())
    }
}
impl<S: HeightSystem> Sub<Length<Metre>> for Height<S> {
    type Output = Self;
    fn sub(self, rhs: Length<Metre>) -> Self {
        Self::new(self.metres - rhs.value())
    }
}
/// The difference of two heights in the SAME system is a length.
impl<S: HeightSystem> Sub for Height<S> {
    type Output = Length<Metre>;
    fn sub(self, rhs: Self) -> Length<Metre> {
        Length::new(self.metres - rhs.metres)
    }
}

// ---------------------------------------------------------------------------
// CRS and epoch tags
// ---------------------------------------------------------------------------

/// Marker trait for coordinate reference systems a `Coord` can be tagged with.
pub trait CrsTag: Sealed + Copy + core::fmt::Debug + 'static {
    const EPSG: u32;
    const NAME: &'static str;
}

/// Geographic WGS84 (EPSG:4326), lon/lat degrees + ellipsoidal height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wgs84 {}
/// Spherical Web Mercator (EPSG:3857), metres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebMercator {}
/// Earth-centred earth-fixed geocentric (EPSG:4978), metres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ecef {}

impl Sealed for Wgs84 {}
impl Sealed for WebMercator {}
impl Sealed for Ecef {}
impl CrsTag for Wgs84 {
    const EPSG: u32 = 4326;
    const NAME: &'static str = "WGS 84";
}
impl CrsTag for WebMercator {
    const EPSG: u32 = 3857;
    const NAME: &'static str = "WGS 84 / Pseudo-Mercator";
}
impl CrsTag for Ecef {
    const EPSG: u32 = 4978;
    const NAME: &'static str = "WGS 84 (geocentric)";
}

/// Marker trait for reference-frame realization epochs (ITRF sense).
pub trait EpochTag: Sealed + Copy + core::fmt::Debug + 'static {
    const NAME: &'static str;
}

/// ITRF2014 realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Itrf2014 {}
/// ITRF2008 realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Itrf2008 {}
/// ITRF2000 realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Itrf2000 {}

impl Sealed for Itrf2014 {}
impl Sealed for Itrf2008 {}
impl Sealed for Itrf2000 {}
impl EpochTag for Itrf2014 {
    const NAME: &'static str = "ITRF2014";
}
impl EpochTag for Itrf2008 {
    const NAME: &'static str = "ITRF2008";
}
impl EpochTag for Itrf2000 {
    const NAME: &'static str = "ITRF2000";
}

// ---------------------------------------------------------------------------
// Coord
// ---------------------------------------------------------------------------

/// Marker for CRSs whose axes are geographic lon/lat degrees.
pub trait GeographicCrs: CrsTag {}
impl GeographicCrs for Wgs84 {}

/// Marker for CRSs whose axes are projected/geocentric metres.
pub trait MetricCrs: CrsTag {}
impl MetricCrs for WebMercator {}
impl MetricCrs for Ecef {}

/// A position tagged with its CRS and its reference epoch. Coordinates from
/// different CRSs or epochs cannot be differenced; convert first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coord<C: CrsTag, E: EpochTag> {
    x: f64,
    y: f64,
    z: f64,
    _crs: PhantomData<C>,
    _epoch: PhantomData<E>,
}

impl<C: GeographicCrs, E: EpochTag> Coord<C, E> {
    /// Build a geographic coordinate from typed lon/lat/height.
    pub fn geographic(lon: Angle<Degree>, lat: Angle<Degree>, h: Height<Ellipsoidal>) -> Self {
        Self {
            x: lon.value(),
            y: lat.value(),
            z: h.metres(),
            _crs: PhantomData,
            _epoch: PhantomData,
        }
    }
    pub fn lon(&self) -> Angle<Degree> {
        Angle::new(self.x)
    }
    pub fn lat(&self) -> Angle<Degree> {
        Angle::new(self.y)
    }
    pub fn height(&self) -> Height<Ellipsoidal> {
        Height::new(self.z)
    }
}

impl<E: EpochTag> Coord<Ecef, E> {
    /// Build a geocentric coordinate from raw ECEF metres.
    pub fn ecef(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            _crs: PhantomData,
            _epoch: PhantomData,
        }
    }
    pub fn x(&self) -> Length<Metre> {
        Length::new(self.x)
    }
    pub fn y(&self) -> Length<Metre> {
        Length::new(self.y)
    }
    pub fn z(&self) -> Length<Metre> {
        Length::new(self.z)
    }
}

impl<C: CrsTag, E: EpochTag> Coord<C, E> {
    /// Raw f64 triple, crate-internal. The public surface only exposes typed
    /// scalars; the render-space `f32` cliff lives solely in
    /// `crate::camera::anchor::Anchor::to_render_f32`.
    pub(crate) fn raw(&self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }

    pub(crate) fn from_raw(v: DVec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            _crs: PhantomData,
            _epoch: PhantomData,
        }
    }
}

/// Metric offset between two coordinates of identical CRS and epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordOffset<C: CrsTag> {
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    _crs: PhantomData<C>,
}

impl<C: CrsTag, E: EpochTag> Sub for Coord<C, E> {
    type Output = CoordOffset<C>;
    fn sub(self, rhs: Self) -> CoordOffset<C> {
        CoordOffset {
            dx: self.x - rhs.x,
            dy: self.y - rhs.y,
            dz: self.z - rhs.z,
            _crs: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Datum / epoch transforms (WGS84 ↔ ITRF-family Helmert path only)
// ---------------------------------------------------------------------------

/// A 7-parameter Helmert transform (position-vector convention, small angles).
/// Translations in metres, rotations in radians, `scale_ppb` in parts-per-billion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Helmert {
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    pub scale_ppb: f64,
}

impl Helmert {
    pub const IDENTITY: Helmert = Helmert {
        tx: 0.0,
        ty: 0.0,
        tz: 0.0,
        rx: 0.0,
        ry: 0.0,
        rz: 0.0,
        scale_ppb: 0.0,
    };

    /// Apply to a geocentric position (position-vector rotation convention).
    pub fn apply(&self, p: DVec3) -> DVec3 {
        let s = 1.0 + self.scale_ppb * 1e-9;
        DVec3::new(
            self.tx + s * (p.x - self.rz * p.y + self.ry * p.z),
            self.ty + s * (self.rz * p.x + p.y - self.rx * p.z),
            self.tz + s * (-self.ry * p.x + self.rx * p.y + p.z),
        )
    }

    /// Exact-enough inverse for small-angle Helmert (negated parameters).
    pub fn inverse(&self) -> Helmert {
        Helmert {
            tx: -self.tx,
            ty: -self.ty,
            tz: -self.tz,
            rx: -self.rx,
            ry: -self.ry,
            rz: -self.rz,
            scale_ppb: -self.scale_ppb,
        }
    }
}

/// A typed datum/epoch transform: the only sanctioned way to move a `Coord`
/// between epoch tags. Only the WGS84/ITRF-family Helmert path ships; grid
/// shifts (NTv2/NADCON) are deliberately out of scope.
pub trait DatumTransform<From: EpochTag, To: EpochTag> {
    fn helmert(&self) -> Helmert;
}

/// ITRF2000 → ITRF2014 Helmert (IERS/IGN published values at epoch 2010.0;
/// ITRF2014 Transformation Parameters, Altamimi et al. 2016, Table 1,
/// sign-inverted from the published ITRF2014→ITRF2000 direction):
/// T = (-0.7, -1.2, 26.1) mm, D = -2.12 ppb, R = (0, 0, 0) mas.
#[derive(Clone, Copy, Debug, Default)]
pub struct Itrf2000ToItrf2014;

impl DatumTransform<Itrf2000, Itrf2014> for Itrf2000ToItrf2014 {
    fn helmert(&self) -> Helmert {
        Helmert {
            tx: -0.7e-3,
            ty: -1.2e-3,
            tz: 26.1e-3,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            scale_ppb: -2.12,
        }
    }
}

/// Move a geocentric coordinate between epochs through a typed transform.
pub fn epoch_transform<From, To, T>(coord: Coord<Ecef, From>, transform: &T) -> Coord<Ecef, To>
where
    From: EpochTag,
    To: EpochTag,
    T: DatumTransform<From, To>,
{
    Coord::from_raw(transform.helmert().apply(coord.raw()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_unit_conversions_are_exact() {
        let ft = Length::<USSurveyFoot>::new(3937.0);
        assert!((ft.to::<Metre>().value() - 1200.0).abs() < 1e-9);
        let m = Length::<Metre>::new(0.3048);
        assert!((m.to::<Foot>().value() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn angle_unit_conversions_are_exact() {
        let d = Angle::<Degree>::new(180.0);
        assert!((d.to::<Radian>().value() - core::f64::consts::PI).abs() < 1e-15);
        assert!((d.to::<Grad>().value() - 200.0).abs() < 1e-12);
    }

    #[test]
    fn same_system_height_difference_is_a_length() {
        let a = Height::<Ellipsoidal>::new(120.5);
        let b = Height::<Ellipsoidal>::new(100.0);
        assert!(((a - b).value() - 20.5).abs() < 1e-12);
    }

    #[test]
    fn same_tag_coord_difference_compiles_and_is_metric() {
        let a = Coord::<Ecef, Itrf2014>::ecef(1.0, 2.0, 3.0);
        let b = Coord::<Ecef, Itrf2014>::ecef(0.5, 0.5, 0.5);
        let d = a - b;
        assert_eq!((d.dx, d.dy, d.dz), (0.5, 1.5, 2.5));
    }

    #[test]
    fn helmert_roundtrip_closes_below_a_tenth_of_a_millimetre() {
        let t = Itrf2000ToItrf2014.helmert();
        let p = DVec3::new(4_027_894.006, 307_045.600, 4_919_474.910);
        let back = t.inverse().apply(t.apply(p));
        assert!((back - p).length() < 1e-4);
    }

    #[test]
    fn epoch_transform_changes_the_tag() {
        let c2000 = Coord::<Ecef, Itrf2000>::ecef(4_027_894.006, 307_045.600, 4_919_474.910);
        let c2014: Coord<Ecef, Itrf2014> = epoch_transform(c2000, &Itrf2000ToItrf2014);
        // ΔZ = tz + (scale − 1)·Z = 26.1 mm − 2.12 ppb × 4.919e6 m.
        let expected = 0.0261 - 2.12e-9 * 4_919_474.910;
        assert!((c2014.z().value() - c2000.raw().z - expected).abs() < 1e-9);
    }
}
