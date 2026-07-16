// Dateline/antimeridian behavior of the geometry operations: CRS-driven
// unwrapping, the canonical ±180 output contract (every geographic output
// has longitudes in [-180, 180] and is split at the antimeridian), and the
// sheet-alignment regressions for clip/intersect.

use super::*;

#[test]
fn projected_crs_in_lonlat_range_stays_planar() {
    // A polygon whose coordinates fall in lon/lat ranges but crosses +/-180
    // numerically must NOT be dateline-unwrapped when a projected CRS is
    // declared: the centroid stays the planar (complement-side) value.
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[[179.0, 0.0], [-179.0, 0.0], [-179.0, 1.0], [179.0, 1.0], [179.0, 0.0]]]
    });
    let centroid = geometry_centroid(&geometry, planar()).expect("centroid succeeds");
    let lon = centroid["geometry"]["coordinates"][0].as_f64().unwrap();
    // Planar centroid of the numeric ring sits near lon 0 (the complement),
    // NOT at the dateline — proving no range-based geographic unwrap ran.
    assert!(
        lon.abs() < 1.0,
        "planar centroid must sit near 0, got {lon}"
    );
}

#[test]
fn dateline_polygon_measures_and_centroids_locally() {
    // Regression (MENSURA item 4): a polygon spanning 179° → -179° must
    // behave as a 2°-wide patch, not the 358°-wide complement.
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[[179.0, 0.0], [-179.0, 0.0], [-179.0, 1.0], [179.0, 1.0], [179.0, 0.0]]]
    });
    let measure = geometry_measure(&geometry, &["area".to_string()], MeasureMode::GeodesicWgs84)
        .expect("measurement succeeds");
    let area = measure["area"].as_f64().unwrap();
    assert!((area - 2.0 * 1.2308e10).abs() < 4e8, "area = {area}");

    let centroid = geometry_centroid(&geometry, wgs84()).expect("centroid succeeds");
    let lon = centroid["geometry"]["coordinates"][0].as_f64().unwrap();
    let lat = centroid["geometry"]["coordinates"][1].as_f64().unwrap();
    assert!(
        lon.abs() > 179.0,
        "centroid lon must sit at the dateline, got {lon}"
    );
    assert!((lat - 0.5).abs() < 1e-9, "centroid lat = {lat}");
}

#[test]
fn same_sheet_point_outputs_are_wrapped_canonically() {
    // Operands authored on one continuous sheet (175..185) never trigger the
    // unwrap path, but geographic point outputs must still land in
    // [-180, 180]: 182°E is reported as -178.
    let geometry = json!({
        "type": "Polygon",
        "coordinates": [[[175.0, -5.0], [185.0, -5.0], [185.0, 5.0], [175.0, 5.0], [175.0, -5.0]]]
    });
    let centroid = geometry_centroid(&geometry, wgs84()).expect("centroid succeeds");
    let lon = centroid["geometry"]["coordinates"][0].as_f64().unwrap();
    assert!(lon.abs() <= 180.0, "canonical centroid lon, got {lon}");

    // ~7/8 along a 175..185 equatorial line sits at 183.75°E => -176.25.
    let line = json!({
        "type": "LineString",
        "coordinates": [[175.0, 0.0], [185.0, 0.0]]
    });
    let point = interpolate_line(&line, 0.875, true, wgs84()).expect("interpolation succeeds");
    let lon = point["geometry"]["coordinates"][0].as_f64().unwrap();
    assert!(
        (-180.0..=-175.0).contains(&lon),
        "interpolated lon must be wrapped west of the dateline, got {lon}"
    );
}

#[cfg(feature = "geos-topology")]
#[test]
fn same_sheet_operands_intersect_to_canonical_split_output() {
    // Reviewer repro: two EPSG:4326 operands both authored on 175..185 need
    // neither unwrapping nor sheet alignment, yet their intersection crosses
    // ±180 and must come back split with longitudes in [-180, 180] — not an
    // unsplit Polygon spanning 176..184.
    let left = json!({
        "type": "Polygon",
        "coordinates": [[[175.0, 5.0], [185.0, 5.0], [185.0, -5.0], [175.0, -5.0], [175.0, 5.0]]]
    });
    let right = json!({
        "type": "Polygon",
        "coordinates": [[[176.0, 3.0], [184.0, 3.0], [184.0, -3.0], [176.0, -3.0], [176.0, 3.0]]]
    });
    let result = intersect_polygonal_geometry_values(&left, &right, true)
        .expect("intersection succeeds")
        .expect("non-empty result");
    assert_eq!(result["type"], json!("MultiPolygon"), "got {result}");
    let pieces = result["coordinates"].as_array().unwrap();
    assert_eq!(pieces.len(), 2, "band splits at the antimeridian: {result}");
    for piece in pieces {
        for p in piece[0].as_array().unwrap() {
            let x = p[0].as_f64().unwrap();
            assert!(x.abs() <= 180.0 + 1e-9, "lon in range, got {x}");
        }
    }
}

#[cfg(feature = "geos-topology")]
#[test]
fn same_sheet_union_splits_at_antimeridian() {
    let geometries = json!([{
        "type": "Polygon",
        "coordinates": [[[175.0, 5.0], [185.0, 5.0], [185.0, -5.0], [175.0, -5.0], [175.0, 5.0]]]
    }]);
    let result = union_geometries(&geometries, wgs84()).expect("union succeeds");
    assert_eq!(
        result["geometry"]["type"],
        json!("MultiPolygon"),
        "got {result}"
    );
    let pieces = result["geometry"]["coordinates"].as_array().unwrap();
    assert_eq!(pieces.len(), 2, "union splits at the antimeridian");
    for piece in pieces {
        for p in piece[0].as_array().unwrap() {
            let x = p[0].as_f64().unwrap();
            assert!(x.abs() <= 180.0 + 1e-9, "lon in range, got {x}");
        }
    }
}

#[cfg(feature = "geos-topology")]
#[test]
fn union_of_polygon_touching_minus_180_is_identity() {
    // Unconditional canonicalization must not damage an already-canonical
    // west-side polygon: [-180, -175] touches the antimeridian without
    // crossing it and comes back as the same single Polygon, not as a ring
    // with a world-spanning edge.
    let coords = json!([[
        [-180.0, 5.0],
        [-175.0, 5.0],
        [-175.0, -5.0],
        [-180.0, -5.0],
        [-180.0, 5.0]
    ]]);
    let geometries = json!([{ "type": "Polygon", "coordinates": coords }]);
    let result = union_geometries(&geometries, wgs84()).expect("union succeeds");
    assert_eq!(result["geometry"]["type"], json!("Polygon"), "got {result}");
    let ring = result["geometry"]["coordinates"][0].as_array().unwrap();
    let xs: Vec<f64> = ring.iter().map(|p| p[0].as_f64().unwrap()).collect();
    assert!(
        xs.iter().all(|x| (-180.0..=-175.0).contains(x)),
        "identity west-side ring, got {xs:?}"
    );
    let max_edge = xs
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f64::max);
    assert!(max_edge <= 5.0 + 1e-9, "no world-spanning edge: {max_edge}");
}

#[cfg(feature = "geos-topology")]
#[test]
fn clip_across_dateline_keeps_both_pieces() {
    // MENSURA M-04 root-cause regression at the Rust boundary: a source
    // polygon 175 -> -175 clipped by a clip polygon 178 -> -178 (both crossing
    // the antimeridian) must keep BOTH dateline pieces, not just the eastern
    // half. `geographic = true` selects WGS84 dateline handling.
    let source = json!({
        "type": "Polygon",
        "coordinates": [[[175.0, 5.0], [-175.0, 5.0], [-175.0, -5.0], [175.0, -5.0], [175.0, 5.0]]]
    });
    let clip = json!({
        "type": "Polygon",
        "coordinates": [[[178.0, 3.0], [-178.0, 3.0], [-178.0, -3.0], [178.0, -3.0], [178.0, 3.0]]]
    });
    let mask = prepare_polygonal_clip_mask(&clip, true).expect("mask builds");
    let clipped = clip_polygonal_geometry_value(&source, &mask, true)
        .expect("clip succeeds")
        .expect("non-empty result");
    assert_eq!(clipped["type"], json!("MultiPolygon"));
    let pieces = clipped["coordinates"].as_array().unwrap();
    assert_eq!(pieces.len(), 2, "both dateline pieces preserved: {clipped}");
    let mut has_east = false;
    let mut has_west = false;
    for piece in pieces {
        let outer = piece[0].as_array().unwrap();
        let xs: Vec<f64> = outer.iter().map(|p| p[0].as_f64().unwrap()).collect();
        assert!(
            xs.iter().all(|x| x.abs() <= 180.0 + 1e-9),
            "in range: {xs:?}"
        );
        if xs.iter().all(|&x| x >= 0.0) {
            has_east = true;
        }
        if xs.iter().all(|&x| x <= 0.0) {
            has_west = true;
        }
    }
    assert!(has_east && has_west, "expected an east and a west piece");
}

#[cfg(feature = "geos-topology")]
#[test]
fn canonical_split_multipolygon_clips_without_losing_a_sheet() {
    // A MultiPolygon already split at the antimeridian (east piece 175..180,
    // west piece -180..-175 — the canonical output of a previous dateline
    // op) intersected with a crossing clip must keep BOTH pieces. Whole-
    // operand alignment by the first vertex stranded the west piece 360°
    // away and silently dropped it.
    let source = json!({
        "type": "MultiPolygon",
        "coordinates": [
            [[[175.0, 5.0], [180.0, 5.0], [180.0, -5.0], [175.0, -5.0], [175.0, 5.0]]],
            [[[-180.0, 5.0], [-175.0, 5.0], [-175.0, -5.0], [-180.0, -5.0], [-180.0, 5.0]]]
        ]
    });
    let clip = json!({
        "type": "Polygon",
        "coordinates": [[[178.0, 3.0], [-178.0, 3.0], [-178.0, -3.0], [178.0, -3.0], [178.0, 3.0]]]
    });
    let mask = prepare_polygonal_clip_mask(&clip, true).expect("mask builds");
    let clipped = clip_polygonal_geometry_value(&source, &mask, true)
        .expect("clip succeeds")
        .expect("non-empty result");
    assert_eq!(clipped["type"], json!("MultiPolygon"), "got {clipped}");
    let pieces = clipped["coordinates"].as_array().unwrap();
    assert_eq!(pieces.len(), 2, "both dateline pieces preserved: {clipped}");
    let mut has_east = false;
    let mut has_west = false;
    for piece in pieces {
        let xs: Vec<f64> = piece[0]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p[0].as_f64().unwrap())
            .collect();
        assert!(
            xs.iter().all(|x| x.abs() <= 180.0 + 1e-9),
            "in range: {xs:?}"
        );
        has_east |= xs.iter().all(|&x| x >= 0.0);
        has_west |= xs.iter().all(|&x| x <= 0.0);
    }
    assert!(has_east && has_west, "expected an east and a west piece");
}

#[cfg(feature = "geos-topology")]
#[test]
fn opposite_sheet_operands_intersect_nonempty() {
    // Operands authored on different 360° sheets — 175..185 and
    // -185..-175 describe the SAME band around the antimeridian — must
    // intersect as that band, not as empty.
    let left = json!({
        "type": "Polygon",
        "coordinates": [[[175.0, 5.0], [185.0, 5.0], [185.0, -5.0], [175.0, -5.0], [175.0, 5.0]]]
    });
    let right = json!({
        "type": "Polygon",
        "coordinates": [[[-185.0, 3.0], [-175.0, 3.0], [-175.0, -3.0], [-185.0, -3.0], [-185.0, 3.0]]]
    });
    let result = intersect_polygonal_geometry_values(&left, &right, true)
        .expect("intersection succeeds")
        .expect("non-empty result");
    let pieces = result["coordinates"].as_array().unwrap();
    assert_eq!(result["type"], json!("MultiPolygon"), "got {result}");
    assert_eq!(pieces.len(), 2, "band splits at the antimeridian: {result}");
    for piece in pieces {
        for p in piece[0].as_array().unwrap() {
            let x = p[0].as_f64().unwrap();
            assert!(x.abs() <= 180.0 + 1e-9, "lon in range, got {x}");
        }
    }
}
