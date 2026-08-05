# tools/europe_smoke/tests/test_boundary.py
import json
import stat
import zipfile

import pytest

from tools.europe_smoke import boundary, config, provenance


def test_boundary_path_matches_the_engine_naming_convention(tmp_path):
    p = boundary.osm_boundary_path(tmp_path)
    # the engine hardcodes a california_boundary_osm_r{ID}.geojson pattern
    assert p.name == f"california_boundary_osm_r{config.SENTINEL_RELATION_ID}.geojson"
    assert p.parent.name == "osm"


def test_sentinel_id_is_not_a_real_osm_relation():
    # real relation ids are far below this; 165475 is California
    assert config.SENTINEL_RELATION_ID == 900000003
    assert config.SENTINEL_RELATION_ID != 165475


@pytest.mark.slow
def test_land_union_covers_the_full_request_area(tmp_path, osm_land_artifact):
    from shapely.geometry import shape
    p = boundary.build_land_union(tmp_path)
    geom = shape(json.loads(p.read_text())["features"][0]["geometry"])
    north, west, south, east = config.AREA
    minx, miny, maxx, maxy = geom.bounds
    # the union must reach every edge of the request, or the DEM crop will be
    # smaller than the CAMS grid and the shader will sample outside the basemap
    assert minx <= west + 0.5 and maxx >= east - 0.5
    assert miny <= south + 0.5 and maxy >= north - 0.5
    assert geom.area > 100.0  # deg^2; Europe+N.Africa land is far larger


@pytest.mark.slow
def test_land_union_is_valid_and_multipart(tmp_path, osm_land_artifact):
    from shapely.geometry import shape
    geom = shape(json.loads(boundary.build_land_union(tmp_path).read_text())
                 ["features"][0]["geometry"])
    assert geom.is_valid
    assert geom.geom_type in {"Polygon", "MultiPolygon"}


def test_prepare_osm_land_uses_the_verified_zip_and_all_sidecars(tmp_path, monkeypatch):
    import hashlib
    import zipfile

    archive = tmp_path / "land.zip"
    members = {
        "land/simplified_land_polygons.shp": b"shp",
        "land/simplified_land_polygons.shx": b"shx",
        "land/simplified_land_polygons.dbf": b"dbf",
        "land/simplified_land_polygons.prj": b"prj",
    }
    with zipfile.ZipFile(archive, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    entry = {"class": "required", "path": str(archive), "sha256": digest,
             "bytes": archive.stat().st_size}
    monkeypatch.setattr(boundary.config, "OSM_LAND_ZIP", archive, raising=False)
    monkeypatch.setattr(boundary.provenance, "load_manifest", lambda: {"osm_land": entry})

    shp = boundary.prepare_osm_land(tmp_path / "cache")

    assert shp.name == "simplified_land_polygons.shp"
    assert {shp.with_suffix(s).name for s in (".shp", ".shx", ".dbf", ".prj")} <= {
        p.name for p in shp.parent.iterdir()
    }
    shp.with_suffix(".dbf").unlink()
    with pytest.raises(RuntimeError, match="partial OSM extraction"):
        boundary.prepare_osm_land(tmp_path / "cache")


def test_load_countries_rejects_a_malicious_natural_earth_zip(tmp_path):
    """The NE zip arrives over plain HTTPS from an S3 bucket; treat it as input."""
    zip_path = tmp_path / "ne" / "ne_10m_admin_0_countries.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in ("ne_10m_admin_0_countries.shp", "ne_10m_admin_0_countries.dbf"):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, "not really a shapefile")
        evil = zipfile.ZipInfo("placeholder")
        evil.filename = "../../../pwned.dll"
        evil.create_system = 3
        evil.external_attr = (stat.S_IFREG | 0o644) << 16
        zf.writestr(evil, "payload")

    with pytest.raises(provenance.UnsafeArchiveError, match=r"\.\."):
        boundary.load_countries(tmp_path)

    assert not (tmp_path / "pwned.dll").exists()
    assert not (tmp_path.parent / "pwned.dll").exists()
    # all-or-nothing: not even the benign members landed
    assert sorted(p.name for p in zip_path.parent.iterdir()) == [
        "ne_10m_admin_0_countries.zip"]


def test_load_countries_rejects_a_zip_that_would_replace_itself(tmp_path):
    """load_countries extracts into zip_path.parent and caches on 'does the zip
    exist', so a member named after the archive would substitute the download
    permanently (measured: it did, with only the traversal rules in place)."""
    zip_path = tmp_path / "ne" / "ne_10m_admin_0_countries.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, body in (("ne_10m_admin_0_countries.shp", "shp"),
                           ("ne_10m_admin_0_countries.zip", "CLOBBER")):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, body)
    before = zip_path.read_bytes()
    with pytest.raises(provenance.UnsafeArchiveError, match="archive being read"):
        boundary.load_countries(tmp_path)
    assert zip_path.read_bytes() == before


def test_load_countries_reports_a_zip_without_the_shapefile(tmp_path):
    zip_path = tmp_path / "ne" / "ne_10m_admin_0_countries.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "wrong archive")
    with pytest.raises(RuntimeError, match="carried no ne_10m_admin_0_countries.shp"):
        boundary.load_countries(tmp_path)
