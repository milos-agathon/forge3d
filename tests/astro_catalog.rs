use forge3d::astro::{
    catalog::{
        bright_star_catalog, bv_to_linear_rgb, magnitude_to_irradiance, star_instances,
        twilight_visibility,
    },
    frames,
    time::{julian_day_tt, julian_day_ut1, UtcDateTime},
    vsop, Observer,
};
use forge3d::geo::units::Angle;
use glam::{DMat3, DVec3};

#[test]
fn committed_bright_star_catalog_is_complete_and_bounded() {
    let stars = bright_star_catalog().expect("catalog");
    assert!((9_000..=9_110).contains(&stars.len()), "{}", stars.len());
    assert!(stars.iter().all(|star| {
        star.ra_j2000().value().is_finite()
            && (0.0..360.0).contains(&star.ra_j2000().value())
            && (-90.0..=90.0).contains(&star.dec_j2000().value())
            && star.v_magnitude().is_finite()
    }));
}

#[test]
fn catalog_photometry_and_per_frame_transform_are_finite() {
    let five_magnitudes = magnitude_to_irradiance(0.0) / magnitude_to_irradiance(5.0);
    assert!((five_magnitudes - 100.0).abs() < 1.0e-12);
    for bv in [-0.4, 0.0, 0.65, 1.5, 2.0] {
        assert!(bv_to_linear_rgb(bv)
            .iter()
            .all(|component| component.is_finite() && (0.0..=1.0).contains(component)));
    }
    assert_eq!(twilight_visibility(Angle::new(-4.0)), 0.0);
    assert_eq!(twilight_visibility(Angle::new(-18.0)), 1.0);

    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0).unwrap();
    let observer = Observer::new(Angle::new(52.3676), Angle::new(4.9041), 0.0).unwrap();
    let instances = star_instances(utc, observer, Angle::new(-18.0)).expect("instances");
    assert_eq!(instances.len(), bright_star_catalog().unwrap().len());
    assert!(instances.iter().all(|star| {
        star.azimuth.value().is_finite()
            && star.altitude.value().is_finite()
            && star.irradiance_w_m2.is_finite()
            && star
                .linear_rgb
                .iter()
                .all(|component| component.is_finite())
    }));
}

#[test]
fn catalog_instances_are_apparent_not_mean_place() {
    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0).unwrap();
    let observer = Observer::new(Angle::new(52.3676), Angle::new(4.9041), 0.0).unwrap();
    let star = bright_star_catalog().unwrap()[0];
    let ra = star.ra_j2000().radians();
    let dec = star.dec_j2000().radians();
    let j2000 = DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
    let jd_tt = julian_day_tt(utc).unwrap();
    let mean = frames::precess_j2000_to_date(j2000, jd_tt);
    let true_place = frames::nutate_mean_to_true(mean, jd_tt);
    let velocity = DMat3::from_rotation_x(frames::mean_obliquity(jd_tt))
        * vsop::earth_velocity(jd_tt).unwrap();
    let apparent = frames::annual_aberration(true_place, velocity);
    let expected = frames::equatorial_to_horizontal(
        apparent,
        observer,
        frames::gast(julian_day_ut1(utc).unwrap(), jd_tt),
    );
    let actual = star_instances(utc, observer, Angle::new(-18.0)).unwrap()[0];
    assert!((actual.azimuth.value() - expected.0.value()).abs() < 1.0e-10);
    assert!((actual.altitude.value() - expected.1.value()).abs() < 1.0e-10);
}
