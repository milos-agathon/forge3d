"""G-002c C3 vector geometry validity and measurement tests."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS vector geometry tests require the compiled _forge3d extension",
)


def _codes(items) -> set[str]:
    return {item["code"] for item in items}


def _unit_square():
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        ],
    }


def _bowtie():
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }


def _line():
    return {"type": "LineString", "coordinates": [[0.0, 0.0], [3.0, 4.0]]}


def _feature(geometry):
    return {"type": "Feature", "properties": {"name": "test"}, "geometry": geometry}


def _feature_collection(geometries):
    return {"type": "FeatureCollection", "features": [_feature(geom) for geom in geometries]}


def _read_vector_result(geometries):
    payload = _feature_collection(geometries)
    payload["info"] = {
        "path": "",
        "driver": "GeoJSON",
        "layer_name": None,
        "layer_count": 1,
        "geometry_type": "Mixed",
        "feature_count": len(geometries),
        "schema": [],
        "crs_wkt": None,
        "crs_authority": {"name": "EPSG", "code": "4326"},
        "bounds": None,
        "is_georeferenced": True,
        "warnings": [],
    }
    return payload


def _polygon_with_hole():
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
            [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 1.0]],
        ],
    }


def test_validate_geometry_valid_unit_square():
    result = gis.validate_geometry(_unit_square())

    assert result == {
        "valid": True,
        "reason": None,
        "geometry_type": "Polygon",
        "warnings": [],
    }


def test_validate_geometry_detects_bowtie_self_intersection():
    result = gis.validate_geometry(_bowtie())

    assert result["valid"] is False
    assert "invalid_geometry" in result["reason"]
    assert "invalid_geometry" in _codes(result["warnings"])


def test_validate_geometry_empty_geometry_reports_empty_geometry():
    result = gis.validate_geometry({"type": "LineString", "coordinates": []})

    assert result["valid"] is False
    assert "empty_geometry" in result["reason"]
    assert "empty_geometry" in _codes(result["warnings"])


def test_validate_geometry_malformed_and_unsupported_geometry_errors():
    malformed = gis.validate_geometry({"type": "Point", "coordinates": ["x", 1.0]})
    unsupported = gis.validate_geometry(
        {"type": "CircularString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}
    )

    assert malformed["valid"] is False
    assert "invalid_geometry" in malformed["reason"]
    assert unsupported["valid"] is False
    assert "unsupported_geometry_type" in unsupported["reason"]


def test_feature_inputs_are_normalized_for_c3_operations():
    line_feature = _feature(_line())

    assert gis.validate_geometry(line_feature)["geometry_type"] == "LineString"
    assert gis.geometry_measure(line_feature, crs=32631)["length"] == pytest.approx(5.0)
    assert gis.geometry_centroid(line_feature, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([1.5, 2.0])
    assert gis.representative_point(line_feature, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([1.5, 2.0])
    assert gis.interpolate_line(line_feature, 2.5, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([1.5, 2.0])


def test_feature_collection_validation_measurement_and_centroid():
    payload = _feature_collection(
        [
            {"type": "Point", "coordinates": [100.0, 100.0]},
            _line(),
            _unit_square(),
        ]
    )

    validation = gis.validate_geometry(payload)
    measure = gis.geometry_measure(payload, crs=32631)
    centroid = gis.geometry_centroid(payload, crs=32631)

    assert validation["valid"] is True
    assert validation["geometry_type"] == "FeatureCollection"
    assert measure["area"] == pytest.approx(1.0)
    assert measure["length"] == pytest.approx(9.0)
    assert measure["operation"]["input_count"] == 3
    assert centroid["geometry"]["coordinates"] == pytest.approx([0.5, 0.5])
    assert centroid["operation"]["input_count"] == 3


def test_read_vector_result_style_crs_propagates_to_operations():
    payload = _read_vector_result([_unit_square()])

    measure = gis.geometry_measure(payload, crs=32631)
    centroid = gis.geometry_centroid(payload)

    assert measure["operation"]["crs"]["source_kind"] == "vector"
    assert measure["operation"]["crs"]["authority"] == {"name": "EPSG", "code": "4326"}
    assert centroid["operation"]["crs"]["source_kind"] == "vector"
    assert centroid["operation"]["crs"]["authority"] == {"name": "EPSG", "code": "4326"}


def test_declared_projected_crs_does_not_trigger_dateline_unwrap_from_numeric_ranges():
    payload = _read_vector_result(
        [{"type": "LineString", "coordinates": [[179.0, 0.0], [-179.0, 0.0]]}]
    )
    payload["info"]["crs_authority"] = {"name": "EPSG", "code": "3857"}
    centroid = gis.geometry_centroid(payload)
    assert centroid["geometry"]["coordinates"] == pytest.approx([0.0, 0.0])


def test_dateline_representative_and_interpolated_points_stay_local():
    line = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "LineString", "coordinates": [[179.0, 0.0], [-179.0, 0.0]]},
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }
    representative = gis.representative_point(line, crs="EPSG:4326")
    rep_lon = representative["geometry"]["coordinates"][0]
    assert abs(rep_lon) > 179.0

    interpolated = gis.interpolate_line(line, 0.5, normalized=True, crs="EPSG:4326")
    interp_lon = interpolated["geometry"]["coordinates"][0]
    assert abs(interp_lon) > 179.0


def test_geometry_measure_polygon_area_and_boundary_length():
    result = gis.geometry_measure(_unit_square(), crs=32631)

    assert result["area"] == pytest.approx(1.0)
    assert result["length"] == pytest.approx(4.0)
    assert result["units"] == "source_crs_planar"
    assert result["operation"]["name"] == "geometry_measure"


def test_geometry_measure_polygon_with_hole_area_boundary_and_centroid():
    measure = gis.geometry_measure(_polygon_with_hole(), crs=32631)
    centroid = gis.geometry_centroid(_polygon_with_hole(), crs=32631)

    assert measure["area"] == pytest.approx(12.0)
    assert measure["length"] == pytest.approx(24.0)
    assert centroid["geometry"]["coordinates"] == pytest.approx([2.0, 2.0])


def test_geometry_measure_metric_subsets():
    area_only = gis.geometry_measure(_unit_square(), crs=32631, metrics=("area",))
    length_only = gis.geometry_measure(_unit_square(), crs=32631, metrics=("length",))

    assert area_only["area"] == pytest.approx(1.0)
    assert area_only["length"] is None
    assert length_only["area"] is None
    assert length_only["length"] == pytest.approx(4.0)


def test_geometry_measure_line_length():
    result = gis.geometry_measure(_line(), crs=32631)

    assert result["area"] is None
    assert result["length"] == pytest.approx(5.0)


def test_geometry_measure_points_return_none_for_area_and_length():
    result = gis.geometry_measure({"type": "Point", "coordinates": [1.0, 2.0]}, crs=32631)

    assert result["area"] is None
    assert result["length"] is None


def test_geometry_measure_geometry_collection_sums_applicable_metrics():
    result = gis.geometry_measure(
        {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [10.0, 10.0]},
                _line(),
                _unit_square(),
            ],
        },
        crs=32631,
    )

    assert result["area"] == pytest.approx(1.0)
    assert result["length"] == pytest.approx(9.0)
    assert result["operation"]["input_count"] == 3


def test_geometry_measure_rejects_unsupported_metric():
    with pytest.raises(ValueError, match="unsupported_option"):
        gis.geometry_measure(_unit_square(), crs=32631, metrics=("area", "volume"))


def test_geometry_centroid_point_line_polygon_and_collection():
    point = gis.geometry_centroid(
        {"type": "Point", "coordinates": [2.0, 3.0]}, crs=32631
    )
    line = gis.geometry_centroid(
        {"type": "LineString", "coordinates": [[0.0, 0.0], [4.0, 0.0]]}, crs=32631
    )
    polygon = gis.geometry_centroid(_unit_square(), crs=32631)
    collection = gis.geometry_centroid(
        {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [100.0, 100.0]},
                _line(),
                _unit_square(),
            ],
        },
        crs=32631,
    )

    assert point["geometry"]["coordinates"] == pytest.approx([2.0, 3.0])
    assert line["geometry"]["coordinates"] == pytest.approx([2.0, 0.0])
    assert polygon["geometry"]["coordinates"] == pytest.approx([0.5, 0.5])
    assert collection["geometry"]["coordinates"] == pytest.approx([0.5, 0.5])


def test_representative_point_point_and_line_without_topology_backend():
    point = gis.representative_point(
        {"type": "Point", "coordinates": [2.0, 3.0]}, crs=32631
    )
    line = gis.representative_point(_line(), crs=32631)

    assert point["geometry"]["coordinates"] == pytest.approx([2.0, 3.0])
    assert line["geometry"]["coordinates"] == pytest.approx([1.5, 2.0])


def test_representative_point_geometry_collection_uses_input_order_for_points_and_lines():
    point_first = {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Point", "coordinates": [9.0, 9.0]},
            {"type": "LineString", "coordinates": [[0.0, 0.0], [4.0, 0.0]]},
        ],
    }
    line_first = {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "LineString", "coordinates": [[0.0, 0.0], [4.0, 0.0]]},
            {"type": "Point", "coordinates": [9.0, 9.0]},
        ],
    }

    assert gis.representative_point(point_first, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([9.0, 9.0])
    assert gis.representative_point(line_first, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([2.0, 0.0])


def test_representative_point_polygon_requires_or_uses_geos_topology():
    with pytest.raises(RuntimeError, match="backend_unavailable|geos-topology"):
        gis.representative_point(_unit_square(), crs=32631)


def test_interpolate_line_linestring_distance():
    result = gis.interpolate_line(_line(), 2.5, crs=32631)
    normalized = gis.interpolate_line(_line(), 0.5, normalized=True, crs=32631)

    assert result["geometry"]["coordinates"] == pytest.approx([1.5, 2.0])
    assert result["distance"] == pytest.approx(2.5)
    assert result["normalized"] is False
    assert normalized["geometry"]["coordinates"] == pytest.approx([1.5, 2.0])
    assert normalized["normalized"] is True


def test_interpolate_line_boundary_distances():
    line = _line()

    assert gis.interpolate_line(line, 0.0, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([0.0, 0.0])
    assert gis.interpolate_line(line, 5.0, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([3.0, 4.0])
    assert gis.interpolate_line(line, 0.0, normalized=True, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([0.0, 0.0])
    assert gis.interpolate_line(line, 1.0, normalized=True, crs=32631)["geometry"][
        "coordinates"
    ] == pytest.approx([3.0, 4.0])


def test_interpolate_line_multilinestring_cumulative_order():
    geometry = {
        "type": "MultiLineString",
        "coordinates": [
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 2.0]],
        ],
    }

    result = gis.interpolate_line(geometry, 3.0, crs=32631)

    assert result["geometry"]["coordinates"] == pytest.approx([2.0, 1.0])


def test_interpolate_line_rejects_negative_out_of_range_and_unsupported_geometry():
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.interpolate_line(_line(), -0.1, crs=32631)
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.interpolate_line(_line(), 6.0, crs=32631)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.interpolate_line(_unit_square(), 0.5, crs=32631)


def test_repair_geometry_make_valid_requires_or_uses_geos_topology():
    with pytest.raises(RuntimeError, match="backend_unavailable|geos-topology"):
        gis.repair_geometry(_bowtie())


def test_repair_geometry_unsupported_method_returns_unsupported_option():
    with pytest.raises(ValueError, match="unsupported_option"):
        gis.repair_geometry(_unit_square(), method="buffer_zero")


def test_feature_collection_rejected_for_repair_representative_and_interpolate():
    payload = _feature_collection([_line()])

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.repair_geometry(payload)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.representative_point(payload)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.interpolate_line(payload, 0.0)


def test_no_python_gis_backend_imports_in_forge3d_gis():
    source = inspect.getsource(gis)
    tree = ast.parse(source)
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    for banned in ("rasterio", "geopandas", "shapely", "rioxarray", "xarray", "terra"):
        assert banned not in imports




def test_union_splits_dateline_crossing_polygon_into_coherent_pieces():
    # MENSURA M-04: a polygon crossing +/-180 comes back as a MultiPolygon split
    # at the antimeridian, each piece coherent in [-180, 180] rather than the
    # world-spanning complement.
    poly = {
        "type": "Polygon",
        "coordinates": [[[170, 10], [-170, 10], [-170, -10], [170, -10], [170, 10]]],
        "info": {"crs_authority": {"name": "EPSG", "code": "4326"}},
    }
    geom = gis.union_geometries([poly])["geometry"]
    assert geom["type"] == "MultiPolygon"
    assert len(geom["coordinates"]) == 2
    for piece in geom["coordinates"]:
        xs = [pt[0] for ring in piece for pt in ring]
        assert all(-180.0 <= x <= 180.0 for x in xs), xs
        east = all(x >= 170.0 - 1e-6 for x in xs)
        west = all(x <= -170.0 + 1e-6 for x in xs)
        assert east or west, f"piece straddles dateline: {xs}"


def test_union_of_mixed_crs_geometries_raises_crs_mismatch():
    # Mixed-CRS union input is ill-defined: two items declaring different
    # embedded CRSs raise crs_mismatch rather than silently unioning under the
    # first declaration.
    poly_4326 = dict(_unit_square(), info={"crs_authority": {"name": "EPSG", "code": "4326"}})
    poly_3857 = dict(_unit_square(), info={"crs_authority": {"name": "EPSG", "code": "3857"}})
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.union_geometries([poly_4326, poly_3857])


def test_mixed_crs_feature_collection_raises_crs_mismatch():
    # Regression: per-feature CRS metadata is checked too — a FeatureCollection
    # mixing EPSG:4326 and EPSG:3857 Features raises crs_mismatch instead of
    # silently operating under the collection-level (or first) declaration.
    def feature(code):
        return {
            "type": "Feature",
            "properties": {},
            "info": {"crs_authority": {"name": "EPSG", "code": code}},
            "geometry": _unit_square(),
        }

    collection = {
        "type": "FeatureCollection",
        "features": [feature("4326"), feature("3857")],
    }
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.union_geometries(collection)
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.geometry_centroid(collection)


def _mixed_crs_source_collection():
    def feature(code):
        return {
            "type": "Feature",
            "properties": {},
            "info": {"crs_authority": {"name": "EPSG", "code": code}},
            "geometry": _unit_square(),
        }

    return {
        "type": "FeatureCollection",
        "info": {"crs_authority": {"name": "EPSG", "code": "4326"}},
        "features": [feature("4326"), feature("3857")],
    }


def _fc_with_crs(code, geometry):
    return {
        "type": "FeatureCollection",
        "info": {"crs_authority": {"name": "EPSG", "code": code}},
        "features": [{"type": "Feature", "properties": {}, "geometry": geometry}],
    }


def test_clip_vector_mixed_crs_source_raises_crs_mismatch():
    # Regression: the clip source resolver folds per-feature `info` CRS
    # metadata into the source CRS, so a collection-level EPSG:4326 source
    # containing an EPSG:3857 feature raises crs_mismatch instead of reaching
    # the topology backend under the collection-level declaration.
    aoi = dict(
        _unit_square(), info={"crs_authority": {"name": "EPSG", "code": "4326"}}
    )
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.clip_vector(_mixed_crs_source_collection(), aoi)


def test_intersect_vectors_mixed_crs_source_raises_crs_mismatch():
    clean = _fc_with_crs("4326", _unit_square())
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.intersect_vectors(_mixed_crs_source_collection(), clean)
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.intersect_vectors(clean, _mixed_crs_source_collection())


def test_dissolve_vector_mixed_crs_source_raises_crs_mismatch():
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.dissolve_vector(_mixed_crs_source_collection())


def test_union_identity_for_polygon_touching_minus_180():
    # Canonical-output regression: a west-side polygon TOUCHING -180 (not
    # crossing it) must survive canonicalization unchanged — the naive
    # per-vertex wrap flipped the exact -180 vertices to +180 and produced a
    # world-spanning 355-degree edge.
    poly = {
        "type": "Polygon",
        "coordinates": [[[-180, 5], [-175, 5], [-175, -5], [-180, -5], [-180, 5]]],
        "info": {"crs_authority": {"name": "EPSG", "code": "4326"}},
    }
    geom = gis.union_geometries([poly])["geometry"]
    assert geom["type"] == "Polygon"
    xs = [pt[0] for ring in geom["coordinates"] for pt in ring]
    assert all(-180.0 <= x <= -175.0 for x in xs), xs


def test_union_splits_same_sheet_dateline_polygon():
    # Canonical +/-180 contract: a polygon authored continuously on 175..185
    # (no wrap jump for the unwrapper to detect) still comes back split at the
    # antimeridian with every longitude in [-180, 180], not as an unsplit
    # Polygon spanning 175..185.
    poly = {
        "type": "Polygon",
        "coordinates": [[[175, 5], [185, 5], [185, -5], [175, -5], [175, 5]]],
        "info": {"crs_authority": {"name": "EPSG", "code": "4326"}},
    }
    geom = gis.union_geometries([poly])["geometry"]
    assert geom["type"] == "MultiPolygon"
    assert len(geom["coordinates"]) == 2
    for piece in geom["coordinates"]:
        xs = [pt[0] for ring in piece for pt in ring]
        assert all(-180.0 <= x <= 180.0 for x in xs), xs


def test_declared_projected_crs_polygon_is_not_dateline_split():
    # MENSURA M-04: a DECLARED projected CRS forces planar handling even when the
    # numeric coordinates happen to fall in lon/lat ranges — no false split.
    poly = {
        "type": "Polygon",
        "coordinates": [[[170, 10], [-170, 10], [-170, -10], [170, -10], [170, 10]]],
        "info": {"crs_authority": {"name": "EPSG", "code": "32633"}},
    }
    geom = gis.union_geometries([poly])["geometry"]
    # Planar: no antimeridian semantics, so it stays a single Polygon.
    assert geom["type"] == "Polygon"


@pytest.mark.parametrize(
    "call",
    [
        lambda: gis.geometry_centroid(_unit_square()),
        lambda: gis.representative_point(_unit_square()),
        lambda: gis.union_geometries([_unit_square()]),
        lambda: gis.buffer_geometry(_unit_square(), 0.1),
        lambda: gis.buffer_geometry(_unit_square(), 0.0),
        lambda: gis.simplify_geometry(_unit_square(), 0.01),
        lambda: gis.interpolate_line(_line(), 0.5),
    ],
    ids=[
        "centroid",
        "representative",
        "union",
        "buffer",
        "buffer_zero",
        "simplify",
        "interpolate",
    ],
)
def test_geometry_ops_require_explicit_or_embedded_crs(call):
    # MENSURA M-04: with no crs= and no embedded CRS metadata, geometry ops raise a
    # stable missing_crs error rather than guessing geographic from coordinate
    # ranges. The fixtures here have numerically valid lon/lat coordinates.
    with pytest.raises(ValueError, match="missing_crs"):
        call()


def test_custom_projected_method_dict_is_accepted_as_planar():
    # A supported custom projected-method CRS dict (CrsSpec::from_projection)
    # carries no EPSG code, but it is planar by construction and must satisfy
    # the geometry-op CRS contract rather than being rejected for lacking one.
    albers = {
        "method": "aea",
        "a": 6378137.0,
        "inv_f": 298.257223563,
        "lat0": 23.0,
        "lon0": -96.0,
        "lat1": 29.5,
        "lat2": 45.5,
        "false_easting": 0.0,
        "false_northing": 0.0,
    }
    square_m = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [100000.0, 0.0], [100000.0, 100000.0],
             [0.0, 100000.0], [0.0, 0.0]]
        ],
    }
    centroid = gis.geometry_centroid(square_m, crs=albers)
    assert centroid["geometry"]["coordinates"] == pytest.approx([50000.0, 50000.0])
    # Planar handling: a topology op under the custom projection succeeds and
    # never dateline-splits metre coordinates.
    buffered = gis.buffer_geometry(square_m, 1000.0, crs=albers)
    assert buffered["geometry"]["type"] == "Polygon"
    # Measurement is planar in CRS units (m / m^2), not geodesic degrees.
    measured = gis.geometry_measure(square_m, crs=albers, metrics=("area",))
    assert measured["area"] == pytest.approx(1.0e10)
    assert measured["units"] == "source_crs_planar"


def _numeric_dateline_polygon():
    # Crosses +/-180 numerically; a projected CRS must NOT dateline-unwrap it.
    return {
        "type": "Polygon",
        "coordinates": [
            [[179.0, 0.0], [-179.0, 0.0], [-179.0, 1.0], [179.0, 1.0], [179.0, 0.0]]
        ],
    }


def test_projected_crs_in_lonlat_range_stays_planar():
    # MENSURA M-04: numeric coordinates inside [-180, 180] declared under a
    # PROJECTED CRS stay planar. The centroid of the numeric ring sits near lon 0
    # (the planar complement), proving no range-based geographic unwrap ran.
    centroid = gis.geometry_centroid(_numeric_dateline_polygon(), crs="EPSG:3857")
    lon = centroid["geometry"]["coordinates"][0]
    assert abs(lon) < 1.0, f"planar centroid must sit near 0, got {lon}"


def test_geographic_4326_enables_dateline_handling():
    # The SAME polygon under EPSG:4326 is handled geographically: the centroid
    # sits at the dateline (|lon| > 179), proving the choice is CRS-driven only.
    centroid = gis.geometry_centroid(_numeric_dateline_polygon(), crs="EPSG:4326")
    lon = centroid["geometry"]["coordinates"][0]
    assert abs(lon) > 179.0, f"geographic centroid must sit at the dateline, got {lon}"


def test_unsupported_geographic_crs_raises_stable_error():
    # A non-WGS84 geographic CRS (NAD83) is rejected, never measured/handled in
    # degrees.
    with pytest.raises(ValueError, match="invalid_crs"):
        gis.geometry_centroid(_unit_square(), crs="EPSG:4269")


def test_epsg_4000_block_is_classified_not_range_guessed():
    # The EPSG 4000-4999 block is not homogeneous: EPSG:4087 (World Equidistant
    # Cylindrical) is PROJECTED and handled planar; EPSG:4978 (WGS84 geocentric)
    # is 3D Cartesian and rejected with a geocentric-specific diagnostic.
    centroid = gis.geometry_centroid(_unit_square(), crs="EPSG:4087")
    assert centroid["geometry"]["coordinates"] == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="geocentric"):
        gis.geometry_centroid(_unit_square(), crs="EPSG:4978")


def test_geographic_interpolation_distance_is_metres():
    # MENSURA M-04: under EPSG:4326 the interpolation distance is geodesic
    # metres — the metric midpoint of a 1-degree equatorial line is ~55,659.7 m
    # along, and a degree-style 0.5 stays ~0.5 m from the start.
    line = {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0]]}
    midpoint = gis.interpolate_line(line, 55659.746, crs="EPSG:4326")
    assert midpoint["geometry"]["coordinates"][0] == pytest.approx(0.5, abs=1e-6)
    # 0.5 m along the WGS84 equator is 0.5 / (a * pi/180) degrees. Assert the
    # geodesic value itself (~4.4916e-6 deg == 0.5 m), not merely < 1e-4 deg,
    # which would tolerate an ~11 m error.
    near_start = gis.interpolate_line(line, 0.5, crs="EPSG:4326")
    assert near_start["geometry"]["coordinates"][0] == pytest.approx(
        0.5 / 111319.49079327358, abs=1e-9
    )


def _fc_4326(poly):
    return {
        "type": "FeatureCollection",
        "info": {"crs_authority": {"name": "EPSG", "code": "4326"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": poly}],
    }


def _box(w, e, s, n):
    return {"type": "Polygon", "coordinates": [[[w, n], [e, n], [e, s], [w, s], [w, n]]]}


def _unwrap_lons(xs):
    """Unwrap a longitude sequence so a piece touching +/-180 reports its true
    (small) span rather than a 360-wide jump."""
    out = [xs[0]]
    for x in xs[1:]:
        prev = out[-1]
        while x - prev > 180.0:
            x -= 360.0
        while x - prev < -180.0:
            x += 360.0
        out.append(x)
    return out


def _outer_bounds(ring):
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _lon_span_unwrapped(ring):
    un = _unwrap_lons([pt[0] for pt in ring])
    return max(un) - min(un)


def _ring_planar_area_unwrapped(ring):
    un = _unwrap_lons([pt[0] for pt in ring])
    ys = [pt[1] for pt in ring]
    s = 0.0
    for i in range(len(un) - 1):
        s += un[i] * ys[i + 1] - un[i + 1] * ys[i]
    return abs(s) * 0.5


def _polygon_pieces(geom):
    """Return the list of piece ring-lists for a Polygon or MultiPolygon."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return list(geom["coordinates"])
    raise AssertionError(f"expected polygonal geometry, got {geom['type']}")


def _assert_coherent_dateline_split(geom, *, side_threshold=170.0, max_span=30.0):
    """A dateline-crossing geographic op must return the small crossing geometry
    (a MultiPolygon split at +/-180), NOT the world-spanning complement: >= 2
    pieces, each closed, each entirely on ONE side of the antimeridian, and each
    with a small unwrapped longitude span rather than the ~358-wide complement."""
    assert geom["type"] == "MultiPolygon", geom["type"]
    pieces = _polygon_pieces(geom)
    assert len(pieces) >= 2, f"expected a split, got {len(pieces)} piece(s)"
    for rings in pieces:
        for ring in rings:
            assert len(ring) >= 4 and ring[0] == ring[-1], f"ring not closed: {ring}"
        outer = rings[0]
        xs = [pt[0] for pt in outer]
        assert all(-180.0 - 1e-6 <= x <= 180.0 + 1e-6 for x in xs), xs
        east = all(x >= side_threshold - 1e-6 for x in xs)
        west = all(x <= -side_threshold + 1e-6 for x in xs)
        assert east or west, f"piece straddles the dateline (complement): {xs}"
        assert _lon_span_unwrapped(outer) <= max_span, xs


@pytest.mark.parametrize(
    "op",
    [
        lambda p: gis.union_geometries([p]),
        lambda p: gis.buffer_geometry(p, 0.5),
        lambda p: gis.simplify_geometry(p, 0.01),
    ],
    ids=["union", "buffer", "simplify"],
)
def test_topology_ops_split_dateline_crossing_polygon(op):
    # MENSURA M-04: EVERY geographic topology op emits the split (small) geometry
    # for a dateline-crossing polygon — coherent pieces, not just in-range coords.
    poly = {
        "type": "Polygon",
        "coordinates": [[[175, 5], [-175, 5], [-175, -5], [175, -5], [175, 5]]],
        "info": {"crs_authority": {"name": "EPSG", "code": "4326"}},
    }
    _assert_coherent_dateline_split(op(poly)["geometry"])


def test_clip_dateline_preserves_both_pieces():
    # MENSURA M-04 root-cause regression: a source polygon 175 -> -175 clipped by
    # a clip polygon 178 -> -178 (both crossing the antimeridian) must return BOTH
    # dateline pieces. The pre-fix path pre-split the clip mask onto opposite 360
    # sheets, so the intersection dropped the western half and returned only the
    # eastern [178, 180] polygon.
    src = _fc_4326(_box(175, -175, -5, 5))
    src["features"][0]["properties"] = {"name": "island", "id": 7}
    clip = _box(178, -178, -3, 3)

    res = gis.clip_vector(src, clip, clip_crs="EPSG:4326")
    feats = res["features"]
    assert len(feats) == 1
    feat = feats[0]
    # 7. source feature properties survive
    assert feat["properties"] == {"name": "island", "id": 7}
    geom = feat["geometry"]
    # 1. both dateline pieces present
    assert geom["type"] == "MultiPolygon", geom["type"]
    pieces = _polygon_pieces(geom)
    assert len(pieces) == 2, f"expected 2 pieces, got {len(pieces)}"

    east = next(rings for rings in pieces if all(pt[0] >= 0.0 for pt in rings[0]))
    west = next(rings for rings in pieces if all(pt[0] <= 0.0 for pt in rings[0]))
    # 2/3. per-piece bounds
    assert _outer_bounds(east[0]) == pytest.approx((178.0, -3.0, 180.0, 3.0), abs=1e-6)
    assert _outer_bounds(west[0]) == pytest.approx((-180.0, -3.0, -178.0, 3.0), abs=1e-6)
    # 4. every ring closed
    for rings in pieces:
        for ring in rings:
            assert len(ring) >= 4 and ring[0] == ring[-1]
    # 5. each individual piece spans <= 2 deg (+ tolerance)
    for rings in pieces:
        assert _lon_span_unwrapped(rings[0]) <= 2.0 + 1e-6
    # 6. combined planar unwrapped area ~ 24 deg^2 (2 deg lon x 6 deg lat x 2)
    total_area = sum(_ring_planar_area_unwrapped(rings[0]) for rings in pieces)
    assert total_area == pytest.approx(24.0, abs=1e-3), total_area
    # 8. deterministic across two identical calls
    assert gis.clip_vector(src, clip, clip_crs="EPSG:4326") == res


def _split_dateline_multipolygon(reach, s, n):
    """An ALREADY-SPLIT dateline MultiPolygon: east piece [180-reach, 180] and
    west piece [-180, -180+reach] — the exact shape forge3d's own splitter
    emits, fed back in as an input."""
    return {
        "type": "MultiPolygon",
        "coordinates": [
            _box(180.0 - reach, 180.0, s, n)["coordinates"],
            _box(-180.0, -180.0 + reach, s, n)["coordinates"],
        ],
    }


def test_clip_with_already_split_multipolygon_mask_preserves_both_pieces():
    # MENSURA M-04 regression: a clip mask that is ALREADY split east/west at
    # the antimeridian (e.g. a previous forge3d output re-used as a mask) must
    # keep BOTH pieces. Aligning the mask by a single whole-geometry longitude
    # stranded its west piece 360 deg away and silently returned only the
    # eastern polygon; per-part sheet alignment keeps both.
    src = _fc_4326(_box(175, -175, -5, 5))  # unsplit crossing source
    mask = _split_dateline_multipolygon(2.0, -3.0, 3.0)

    res = gis.clip_vector(src, mask, clip_crs="EPSG:4326")
    geom = res["features"][0]["geometry"]
    _assert_coherent_dateline_split(geom)
    pieces = _polygon_pieces(geom)
    assert len(pieces) == 2
    east = next(rings for rings in pieces if all(pt[0] >= 0.0 for pt in rings[0]))
    west = next(rings for rings in pieces if all(pt[0] <= 0.0 for pt in rings[0]))
    assert _outer_bounds(east[0]) == pytest.approx((178.0, -3.0, 180.0, 3.0), abs=1e-6)
    assert _outer_bounds(west[0]) == pytest.approx((-180.0, -3.0, -178.0, 3.0), abs=1e-6)


def test_clip_already_split_source_with_crossing_mask_preserves_both_pieces():
    # The mirror case: the SOURCE is the already-split MultiPolygon and the
    # mask is an unsplit crossing polygon. Both dateline pieces must survive.
    src = _fc_4326(_split_dateline_multipolygon(5.0, -5.0, 5.0))
    mask = _box(178, -178, -3, 3)

    res = gis.clip_vector(src, mask, clip_crs="EPSG:4326")
    geom = res["features"][0]["geometry"]
    _assert_coherent_dateline_split(geom)
    pieces = _polygon_pieces(geom)
    assert len(pieces) == 2
    total_area = sum(_ring_planar_area_unwrapped(rings[0]) for rings in pieces)
    assert total_area == pytest.approx(24.0, abs=1e-3), total_area


def test_intersect_with_already_split_multipolygon_operand():
    # intersect_vectors with an already-split right operand mirrors the clip
    # mask case; both pieces of the overlap must survive.
    left = _fc_4326(_box(175, -175, -5, 5))
    right = _fc_4326(_split_dateline_multipolygon(2.0, -4.0, 4.0))
    res = gis.intersect_vectors(left, right)
    _assert_coherent_dateline_split(res["features"][0]["geometry"])


def test_clip_dissolve_intersect_split_dateline_crossing_geometry():
    # MENSURA M-04: the remaining feature-gated topology ops (clip, dissolve,
    # intersect) also emit coherent split geometry across the antimeridian.
    cross = _box(175, -175, -5, 5)  # crosses +/-180

    dissolved = gis.dissolve_vector(_fc_4326(cross))
    _assert_coherent_dateline_split(dissolved["features"][0]["geometry"])

    clipped = gis.clip_vector(_fc_4326(cross), _box(178, -178, -3, 3), clip_crs="EPSG:4326")
    _assert_coherent_dateline_split(clipped["features"][0]["geometry"])

    intersected = gis.intersect_vectors(_fc_4326(cross), _fc_4326(_box(178, -170, -4, 4)))
    _assert_coherent_dateline_split(intersected["features"][0]["geometry"])
