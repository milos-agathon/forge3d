// Unit tests for the geometry-operation module (moved out of geometry.rs
// to keep the operation surface readable; `use super::*` sees the same
// parent items as the previous inline `mod tests`). Dateline/antimeridian
// behavior is covered in the `dateline` submodule.

use super::*;

mod dateline;

fn crs_spec(code: &str) -> Option<CrsSpec> {
    Some(CrsSpec::from_string(format!("EPSG:{code}")).expect("crs parses"))
}

fn wgs84() -> Option<CrsSpec> {
    crs_spec("4326")
}

fn planar() -> Option<CrsSpec> {
    // A declared projected CRS forces planar handling even when the numeric
    // coordinates fall in lon/lat ranges.
    crs_spec("3857")
}

#[test]
fn polygon_area_length_and_centroid_are_planar() {
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    });

    let measure = geometry_measure(
        &geometry,
        &["area".to_string(), "length".to_string()],
        MeasureMode::Planar,
    )
    .expect("measurement succeeds");
    let centroid = geometry_centroid(&geometry, planar()).expect("centroid succeeds");

    assert_eq!(measure["area"], json!(1.0));
    assert_eq!(measure["length"], json!(4.0));
    assert_eq!(centroid["geometry"]["coordinates"], json!([0.5, 0.5]));
}

#[test]
fn geographic_from_spec_uses_crs_not_ranges() {
    // EPSG:4326 => geographic; any projected CRS => planar even for in-range
    // coordinates; an unsupported geographic CRS or missing CRS is an error.
    assert!(geographic_from_spec(&wgs84().unwrap()).unwrap());
    assert!(!geographic_from_spec(&planar().unwrap()).unwrap());
    assert!(!geographic_from_spec(&crs_spec("32633").unwrap()).unwrap());
    assert!(geographic_from_spec(&crs_spec("4269").unwrap()).is_err());
}

#[test]
fn epsg_4000_block_is_classified_not_range_guessed() {
    // The 4000-4999 block is NOT homogeneous. EPSG:4087 (World Equidistant
    // Cylindrical) is PROJECTED => planar; EPSG:4978 (WGS84 geocentric) is
    // 3D Cartesian => a geocentric-specific error, never "geographic"; an
    // uncurated block code is a stable classification error.
    assert!(!geographic_from_spec(&crs_spec("4087").unwrap()).unwrap());
    let geocentric = geographic_from_spec(&crs_spec("4978").unwrap());
    assert!(
        matches!(&geocentric, Err(GisError::InvalidCrs(msg)) if msg.contains("geocentric")),
        "got {geocentric:?}"
    );
    let unclassified = geographic_from_spec(&crs_spec("4999").unwrap());
    assert!(
        matches!(&unclassified, Err(GisError::InvalidCrs(msg)) if msg.contains("classification")),
        "got {unclassified:?}"
    );
}

#[test]
fn geometry_op_requires_a_crs() {
    // No explicit crs= and no embedded metadata => stable missing_crs error,
    // never a coordinate-range guess.
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    });
    let err = geometry_centroid(&geometry, None).expect_err("missing crs is an error");
    assert!(matches!(err, GisError::MissingCrs(_)), "got {err:?}");
}

fn feature_with_crs(code: &str, coordinates: serde_json::Value) -> serde_json::Value {
    json!({
        "type": "Feature",
        "properties": {},
        "info": {"crs_authority": {"name": "EPSG", "code": code}},
        "geometry": {"type": "Polygon", "coordinates": coordinates},
    })
}

#[test]
fn mixed_crs_feature_collection_raises_crs_mismatch() {
    // Regression: the FeatureCollection path read only collection-level CRS
    // metadata, so a collection mixing EPSG:4326 and EPSG:3857 Features
    // bypassed the pairwise check the array input path already enforced.
    let square = json!([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]);
    let collection = json!({
        "type": "FeatureCollection",
        "features": [
            feature_with_crs("4326", square.clone()),
            feature_with_crs("3857", square.clone()),
        ],
    });
    // The check runs at input normalization, before any topology backend.
    let union = union_geometries(&collection, None).expect_err("mixed CRS is an error");
    assert!(matches!(union, GisError::CrsMismatch(_)), "got {union:?}");
    let centroid = geometry_centroid(&collection, None).expect_err("mixed CRS is an error");
    assert!(
        matches!(centroid, GisError::CrsMismatch(_)),
        "got {centroid:?}"
    );
}

#[test]
fn feature_level_crs_conflicts_with_collection_level_crs() {
    let square = json!([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]);
    let collection = json!({
        "type": "FeatureCollection",
        "info": {"crs_authority": {"name": "EPSG", "code": "4326"}},
        "features": [feature_with_crs("3857", square)],
    });
    let err = geometry_centroid(&collection, None).expect_err("mixed CRS is an error");
    assert!(matches!(err, GisError::CrsMismatch(_)), "got {err:?}");
}

#[test]
fn agreeing_feature_level_crs_resolves_the_operation() {
    // Feature-level metadata alone (no collection-level `info`, no crs= arg)
    // both passes the compatibility check and resolves geographic handling.
    let square = json!([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]);
    let collection = json!({
        "type": "FeatureCollection",
        "features": [
            feature_with_crs("4326", square.clone()),
            feature_with_crs("4326", square.clone()),
        ],
    });
    let centroid = geometry_centroid(&collection, None).expect("agreeing CRS succeeds");
    assert_eq!(centroid["geometry"]["coordinates"], json!([0.5, 0.5]));
}

#[test]
fn geodesic_measure_returns_metres_not_degrees() {
    // 1°×1° quad at the equator: ~1.2308e10 m² and ~443 km perimeter.
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    });
    let measure = geometry_measure(
        &geometry,
        &["area".to_string(), "length".to_string()],
        MeasureMode::GeodesicWgs84,
    )
    .expect("measurement succeeds");
    let area = measure["area"].as_f64().unwrap();
    let length = measure["length"].as_f64().unwrap();
    assert!((area - 1.2308e10).abs() < 2e8, "area = {area}");
    assert!((length - 443_770.0).abs() < 2_000.0, "length = {length}");
    assert_eq!(measure["units"], json!("metres_geodesic_wgs84"));
}

#[test]
fn measure_mode_rejects_non_wgs84_geographic_crs() {
    let wgs84 = crate::gis::raster_write::CrsSpec::from_string("EPSG:4326".to_string())
        .expect("crs parses");
    assert_eq!(
        measure_mode_for_crs(&wgs84).unwrap(),
        MeasureMode::GeodesicWgs84
    );
    let utm = crate::gis::raster_write::CrsSpec::from_string("EPSG:32633".to_string())
        .expect("crs parses");
    assert_eq!(measure_mode_for_crs(&utm).unwrap(), MeasureMode::Planar);
    let nad83 = crate::gis::raster_write::CrsSpec::from_string("EPSG:4269".to_string())
        .expect("crs parses");
    assert!(measure_mode_for_crs(&nad83).is_err());
    // Projected/geocentric members of the 4000-4999 block classify by kind,
    // not by numeric range.
    let world_eqc = crate::gis::raster_write::CrsSpec::from_string("EPSG:4087".to_string())
        .expect("crs parses");
    assert_eq!(
        measure_mode_for_crs(&world_eqc).unwrap(),
        MeasureMode::Planar
    );
    let geocentric = crate::gis::raster_write::CrsSpec::from_string("EPSG:4978".to_string())
        .expect("crs parses");
    assert!(measure_mode_for_crs(&geocentric).is_err());
}

#[test]
fn bowtie_polygon_is_invalid() {
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]
    });

    let result = validate_geometry(&geometry).expect("validation returns a result");

    assert_eq!(result["valid"], json!(false));
    assert!(result["reason"]
        .as_str()
        .unwrap()
        .contains(model::INVALID_GEOMETRY));
}

#[test]
fn multiline_interpolation_is_cumulative() {
    let geometry = json!({
        "type": "MultiLineString",
        "coordinates": [
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 2.0]]
        ]
    });

    let result = interpolate_line(&geometry, 3.0, false, planar()).expect("interpolation succeeds");

    assert_eq!(result["geometry"]["coordinates"], json!([2.0, 1.0]));
}

#[test]
fn geographic_interpolation_measures_metres_not_degrees() {
    // MENSURA M-04: under EPSG:4326 the interpolation distance is geodesic
    // metres. A 1° equatorial line is ~111,319.49 m long: its metric
    // midpoint sits at lon 0.5, and 0.5 (which would be the midpoint if
    // degrees were the unit) is a point exactly 0.5 m from the start.
    let geometry = json!({
        "type": "LineString",
        "coordinates": [[0.0, 0.0], [1.0, 0.0]]
    });

    let midpoint =
        interpolate_line(&geometry, 55_659.746, false, wgs84()).expect("interpolation succeeds");
    let lon = midpoint["geometry"]["coordinates"][0].as_f64().unwrap();
    assert!((lon - 0.5).abs() < 1e-6, "metric midpoint lon = {lon}");

    // 0.5 m along the WGS84 equator is 0.5 / (a * π/180) degrees. Assert the
    // geodesic value itself (~4.4916e-6°, i.e. 0.5 m), not merely < 1e-4°
    // (which would tolerate an ~11 m error).
    let expected = 0.5 / 111_319.490_793_273_58;
    let near_start =
        interpolate_line(&geometry, 0.5, false, wgs84()).expect("interpolation succeeds");
    let lon = near_start["geometry"]["coordinates"][0].as_f64().unwrap();
    assert!(
        (lon - expected).abs() < 1e-9,
        "0.5 m along the line must sit {expected}° from the start, got {lon}"
    );

    // The normalized form stays a unitless fraction.
    let normalized =
        interpolate_line(&geometry, 0.5, true, wgs84()).expect("interpolation succeeds");
    let lon = normalized["geometry"]["coordinates"][0].as_f64().unwrap();
    assert!((lon - 0.5).abs() < 1e-6, "normalized midpoint lon = {lon}");
}
