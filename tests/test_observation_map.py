import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from forge3d.chronos import FlythroughManifest
from forge3d.map_scene import (
    LightingPreset, MapFurnitureLayer, MapSceneNativeUnavailable, OrbitCamera,
    OutputSpec, SceneRecipe, TerrainSource,
)
from forge3d.observation_map import ObservationEvents, ObservationRasterStyle, compile_observation_map
from forge3d.observation_time import ObservationRecord, ObservationTrack
from forge3d.publication import PublicationSource


@pytest.fixture
def inputs(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    grid = dict(driver="GTiff", width=64, height=64, transform=from_origin(500000, 4400000, 30, 30),
                crs="EPSG:32610")
    terrain = tmp_path / "terrain.tif"
    with rasterio.open(terrain, "w", **grid, count=1, dtype="float32") as dst:
        dst.write(np.indices((64, 64), dtype=np.float32)[0], 1)
    path = tmp_path / "source.tif"
    valid = np.ones((64, 64), bool)
    valid[20:44, 20:44] = False

    def write(value):
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
            path, "w", **grid, count=3, dtype="uint16", photometric="RGB"
        ) as dst:
            dst.write(np.full((3, 64, 64), value, np.uint16))
            dst.write_mask(valid)
            dst.scales = (0.001,) * 3
            dst.offsets = (-0.1,) * 3

    write(200)
    base = SceneRecipe(TerrainSource(path=terrain, crs=grid["crs"]), OrbitCamera(),
                       LightingPreset(settings={"unlit_albedo": True}), target_crs=grid["crs"],
                       output=OutputSpec(720, 320), map_furniture=MapFurnitureLayer(
                           legend={"entries": ["Fixed RGB stretch"]}))
    source = PublicationSource(path, "Test sensor", "2021", "test-v1", "fixture://test",
                               "reflectance", "QA excluded")
    return base, source, valid, write


def _time(hour):
    return f"2021-07-11T{hour:02}:00:00Z"


def _raster(source, name, obs, start, end):
    return ObservationRecord(name, _time(obs), _time(start), _time(end), {"source": source.description()})


def _event(name, observed_at, valid_until):
    return ObservationRecord(name, observed_at, observed_at, valid_until, {
        "source": {"id": "detections"},
        "feature": {"type": "Feature", "id": name, "properties": {},
                    "geometry": {"type": "Point", "coordinates": [500200.0, 4399800.0]}},
    })


def test_compile_boundaries_windows_and_c06_binding(inputs, tmp_path):
    import rasterio

    base, source, valid, _ = inputs
    early = _raster(source, "earlier", 9, 8, 11)
    latest = _raster(source, "latest", 10, 10, 12)
    times = [_time(8), _time(9), "2021-07-11T03:00:00-07:00", _time(11), _time(12)]
    events = ObservationEvents((_event("start-exclusive", _time(9), _time(12)),
                                _event("simultaneous-a", _time(10), _time(12)),
                                _event("simultaneous-b", _time(10), _time(11))),
                               timedelta(hours=1), base.target_crs, "Sensor reports", {})
    kwargs = dict(raster_style=ObservationRasterStyle((0, 0.4)), snapshot_dir=tmp_path / "snapshots",
                  events=events)
    track = ObservationTrack([early, latest], "hold")
    compiled = compile_observation_map(base, track, times, **kwargs)
    table = compiled.selection_table()["frames"]
    assert [row["raster_selection"] for row in table] == [s.to_dict() for s in track.select(times)]
    assert [row["raster_selection"]["state"] for row in table] == [
        "missing", "observed", "observed", "observed", "held"]
    assert [r["record_id"] for r in table[2]["vector_window"]["records"]] == [
        "simultaneous-a", "simultaneous-b"]
    assert table[3]["vector_window"]["records"] == []
    assert "coverage unknown" in table[3]["captions"][2]
    assert table[2]["raster_selection"]["presentation_time"] == _time(10)
    assert "latest" in table[4]["captions"][1] and "Held" in table[4]["captions"][1]
    assert table[4]["raster_selection"]["age_seconds"] == 7200
    assert "Missing raster" in table[0]["captions"][1]
    assert table[2]["raster"]["excluded_pixels"] == int((~valid).sum())
    with rasterio.open(table[2]["raster"]["snapshot_path"]) as snapshot:
        np.testing.assert_array_equal(snapshot.dataset_mask() > 0, valid)
        assert set(snapshot.read(1)[valid]) == {64}
    frame = compiled.frame(2)
    assert frame.compiled_plan is not None
    assert frame.recipe.map_furniture.metadata["observation_map"] == table[2]
    assert frame.recipe.map_furniture.legend == base.map_furniture.legend
    assert len(frame.recipe.layers[-1].features) == 2
    assert compiled.frame(0).recipe.layers == ()
    with pytest.raises(IndexError):
        compiled.frame(-1)
    tied = _raster(source, "same-acquisition", 10, 9, 13)
    with pytest.raises(ValueError, match="Ambiguous observations"):
        compile_observation_map(base, ObservationTrack([latest, tied], "hold"), times, **kwargs)
    missing = compile_observation_map(base, ObservationTrack([latest], "no-data"), [_time(12)], **kwargs)
    assert missing.selection_table()["frames"][0]["raster_selection"]["state"] == "missing"


def test_native_sequence_random_replay_and_same_path_replacement(inputs, tmp_path):
    base, source, _, replace_pixels = inputs
    track = ObservationTrack([_raster(source, "first", 9, 9, 11)], "hold")
    options = dict(raster_style=ObservationRasterStyle((0, 0.4)), snapshot_dir=tmp_path / "snapshots")
    original = compile_observation_map(base, track, [_time(8), _time(9), _time(11)], **options)
    cache = tmp_path / "cache"
    sequential = original.render(tmp_path / "original", base_seed=0, cache=cache)
    random = original.render(tmp_path / "random", frames=[2, 0], base_seed=0)
    for index in (0, 2):
        assert random.frame_record(index) == sequential.frame_record(index)
    path = tmp_path / "original" / "flythrough_manifest.json"
    loaded = FlythroughManifest.load(path)
    saved = tmp_path / "round-trip.json"
    loaded.save(saved)
    assert saved.read_bytes() == path.read_bytes()
    assert loaded.replay_observation_frame(2, tmp_path / "replay.png")[
        "pixel_hash"] == sequential.frame_record(2)["pixel_hash"]

    replace_pixels(440)
    revised = compile_observation_map(base, track, [_time(8), _time(9), _time(12)], **options,
                                      interventions=["Replaced the acquisition at the same path and retimed frame 2."])
    result = revised.render(tmp_path / "revised", base_seed=0, cache=cache)
    assert result.frame_record(0) == sequential.frame_record(0)
    assert result.frame_record(1)["scene_hash"] != sequential.frame_record(1)["scene_hash"]
    assert result.frame_record(1)["pixel_hash"] != sequential.frame_record(1)["pixel_hash"]
    with pytest.raises(ValueError, match="replay inputs do not match"):
        loaded.replay_frame(revised.frame(1), 1, tmp_path / "bad-replay.png")
    assert loaded.replay_observation_frame(1, tmp_path / "old-source.png")["pixel_hash"] == (
        sequential.frame_record(1)["pixel_hash"])
    frame = original.frame(1)
    selection = json.loads(sequential.frame_record(1)["compiled_frame"])["scene"]["scene"][
        "map_furniture"]["metadata"]["observation_map"]
    assert selection == original.selection_table()["frames"][1]
    from pathlib import Path
    Path(selection["raster"]["snapshot_path"]).write_bytes(b"modified")
    with pytest.raises(MapSceneNativeUnavailable, match="snapshot was modified"):
        frame.render(tmp_path / "tampered.png", cache=cache)


def test_compile_applies_single_band_scale_and_mask(inputs, tmp_path):
    import rasterio

    base, source, valid, _ = inputs
    path = tmp_path / "single.tif"
    with rasterio.open(source.path) as rgb:
        profile = dict(rgb.profile, count=1)
        profile.pop("photometric", None)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(path, "w", **profile) as dst:
            dst.write(rgb.read(1), 1)
            dst.write_mask(valid)
            dst.scales = (0.001,)
            dst.offsets = (-0.1,)
    record = _raster(replace(source, path=path), "single", 9, 9, 11)
    frame = compile_observation_map(
        base, ObservationTrack([record], "no-data"), [_time(9)],
        raster_style=ObservationRasterStyle((0, 0.4)), snapshot_dir=tmp_path / "snapshots",
    ).selection_table()["frames"][0]
    with rasterio.open(frame["raster"]["snapshot_path"]) as snapshot:
        np.testing.assert_array_equal(snapshot.dataset_mask() > 0, valid)
        assert set(snapshot.read()[:, valid].ravel()) == {64}
