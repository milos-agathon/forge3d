import importlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import geopandas as gpd
    import pyogrio
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import LineString, Point

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    source = importlib.import_module("c07_swiss_sources")
    bounds = (2639000, 1156000, 2640000, 1157000)
    monkeypatch.setattr(source, "BOUNDS", bounds)
    dem = tmp_path / "tile.tif"
    with rasterio.open(dem, "w", driver="GTiff", count=1, width=500, height=500,
                       dtype="float32", crs="EPSG:2056", transform=from_origin(bounds[0], bounds[3], 2, 2)) as tile:
        tile.write(np.ones((500, 500), dtype=np.float32), 1)
    monkeypatch.setattr(source, "terrain_assets", lambda: ["https://example.invalid/tile.tif"])

    def copy_tile(url, path):
        if not path.exists():
            path.write_bytes(dem.read_bytes())
        return path

    monkeypatch.setattr(source, "download", copy_tile)

    shapes = tmp_path / "shapes"
    shapes.mkdir()
    archive = tmp_path / "tlm.zip"
    point = Point(2639500, 1156500, 1000)
    for layer in source.LAYERS:
        geometry = LineString([point, Point(2639600, 1156600, 900)]) if layer == "TLM_EISENBAHN" else point
        frame = gpd.GeoDataFrame({"UUID": ["summit"], "NAME": ["Original"], "OBJEKTART": [1]},
                                 geometry=[geometry], crs="EPSG:2056+5728")
        pyogrio.write_dataframe(frame, shapes / f"swissTLM3D_{layer}.shp", encoding="UTF-8")
    with ZipFile(archive, "w") as bundle:
        for member in shapes.iterdir():
            bundle.write(member, arcname=f"members/{member.name}")
    monkeypatch.setattr(source, "RemoteZip", lambda url: archive.open("rb"))
    out = tmp_path / "prepared"
    source.prepare(out)

    def no_network(*args, **kwargs):
        pytest.fail("pinned inputs must be checked locally")

    monkeypatch.setattr(source, "RemoteZip", no_network)
    monkeypatch.setattr(source, "terrain_assets", no_network)
    return source, out


def _evidence(out):
    return {name: (out / name).read_bytes() for name in ("sources.json", "features.json", "terrain.npy")}


def test_verified_cache_reuse_and_valid_dbf_mutation(prepared):
    import pyogrio

    source, out = prepared
    before = _evidence(out)
    source.prepare(out)
    assert _evidence(out) == before
    dbf = out / "sources/swissTLM3D_TLM_NAME_PKT.dbf"
    original = dbf.read_bytes()
    assert b"Original" in original
    dbf.write_bytes(original.replace(b"Original", b"Mutation"))
    assert pyogrio.read_dataframe(dbf.with_suffix(".shp")).NAME.tolist() == ["Mutation"]
    with pytest.raises(ValueError, match=r"source receipt: swissTLM3D_TLM_NAME_PKT\.dbf"):
        source.prepare(out)
    assert _evidence(out) == before


@pytest.mark.parametrize(
    ("change", "value", "message"),
    [("tlm_edition", "swisstlm3d_other", "TLM edition"),
     ("tlm", "https://example.invalid/other.zip", "TLM edition"),
     ("tlm_members_sha256", None, "source receipt lacks")],
)
def test_invalid_prior_receipt_preserves_evidence(prepared, change, value, message):
    source, out = prepared
    path = out / "sources.json"
    receipt = json.loads(path.read_bytes())
    if value is None:
        receipt.pop(change, None)
    else:
        receipt[change] = value
    path.write_text(json.dumps(receipt), encoding="utf-8")
    before = _evidence(out)
    with pytest.raises(ValueError, match=message):
        source.prepare(out)
    assert _evidence(out) == before


def test_changed_derivation_preserves_the_pinned_receipt(prepared, monkeypatch):
    source, out = prepared
    before = _evidence(out)
    read_dataframe = source.pyogrio.read_dataframe

    def changed_derivation(*args, **kwargs):
        frame = read_dataframe(*args, **kwargs)
        frame["NAME"] = "Changed derivation"
        return frame

    monkeypatch.setattr(source.pyogrio, "read_dataframe", changed_derivation)
    with pytest.raises(ValueError, match="derived TLM features differ"):
        source.prepare(out)
    assert _evidence(out) == before


def test_cache_without_receipt_is_not_pinned_on_reuse(prepared):
    source, out = prepared
    before = _evidence(out)
    receipt = out / "sources.json"
    receipt.rename(out / "retained.json")
    with pytest.raises(ValueError, match="without a source receipt"):
        source.prepare(out)
    assert not receipt.exists()
    assert (out / "retained.json").read_bytes() == before["sources.json"]
    assert (out / "features.json").read_bytes() == before["features.json"]
