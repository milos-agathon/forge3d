"""C06 dependency/geometry contracts. Fake frames below are unit-test fixtures.

Actual native execution, cache receipts and CBS revisions are exercised by
examples/c06_population_publication.py, never inferred from these fixtures.
"""
from dataclasses import replace
import hashlib
import os

import numpy as np
import pytest

from forge3d import gis
from forge3d._png import save_png
from forge3d.map_plate import BBox
from forge3d.map_scene import MapScene, MapSceneNativeUnavailable
from forge3d.publication import PublicationSource, PublicationExtent, PublicationRecipe

pytestmark = pytest.mark.usefixtures("pro_license")


@pytest.mark.parametrize("fault", [None, "tampered", "native-failure"])
def test_flyover_cache_forwarding_and_failed_delivery(publication, fake_render, tmp_path, monkeypatch, fault):
    import json
    from types import SimpleNamespace
    from forge3d import _publication_flyover
    from forge3d._png import load_png_rgba
    from forge3d.map_scene import OrbitCamera
    from forge3d.publication import PublicationLayout
    recipe, page, _ = publication
    recipe.camera_mode = "flat-perspective"
    frame = replace(page, camera=OrbitCamera(distance=500,elevation_deg=30,azimuth_deg=270,fov_deg=35))
    cache_path = tmp_path / "cache"
    def native(scenes, *, base_seed, samples, out_dir, cache):
        assert cache == cache_path
        assert base_seed == 17 and samples == 1
        if fault == "native-failure":
            raise RuntimeError("Native test interruption")
        out_dir.mkdir()
        records = []
        for i, scene in enumerate(scenes):
            path = out_dir / f"frame-{i}.png"
            scene.render(path, cache=cache)
            record = {"frame_index":i,"image":path.name,"pixel_hash":hashlib.sha256(load_png_rgba(path).tobytes()).hexdigest(),
                      "camera_hash":"fixture","scene_hash":"fixture"}
            if fault == "tampered":
                record["pixel_hash"] = "0"*64
            records.append(record)
        (out_dir / "flythrough_manifest.json").write_text(json.dumps(records))
        return SimpleNamespace(frames=records)
    monkeypatch.setattr(_publication_flyover,"render_flythrough",native)
    output = tmp_path / "delivery"
    if fault:
        with pytest.raises((ValueError, RuntimeError), match="interruption|pixels differ"):
            recipe.render_flyover([frame], output, layout=PublicationLayout(recipe.config), base_seed=17, cache=cache_path)
        assert not list(output.iterdir())
        assert not list(tmp_path.glob(".flyover-*"))
    else:
        result = recipe.render_flyover([frame], output, layout=PublicationLayout(recipe.config), base_seed=17, cache=cache_path)
        assert len(result["frames"]) == 1
        assert (output / "publication-flyover.json").is_file()


def test_flyover_rejects_empty_sequence_and_unsupported_camera(publication, tmp_path):
    from forge3d.publication import PublicationLayout
    recipe,page,_ = publication
    with pytest.raises(ValueError,match="nonempty"):
        recipe.render_flyover([],tmp_path / "empty",layout=PublicationLayout(recipe.config),base_seed=0)
    with pytest.raises(ValueError,match="flat-perspective"):
        recipe.render_flyover([page],tmp_path / "screen",layout=PublicationLayout(recipe.config),base_seed=0)


def test_simple_legend_keeps_all_style_keys_and_legacy_labels_have_no_swatches(monkeypatch):
    from forge3d import _map_scene_render as render
    texts, colours = [], []
    monkeypatch.setattr(render, "_draw_text", lambda image, text, *a, **k: texts.append(text))
    monkeypatch.setattr(render, "_draw_pixel_block", lambda image, x, y, colour, **k: colours.append(colour))
    entries = [{"label": str(i), "rgba": [i, 40, 80, 255]} for i in range(12)]
    image = np.zeros((400, 400, 4), np.uint8)
    render._draw_simple_legend(image, {"entries": entries})
    assert texts == [e["label"] for e in entries]
    assert colours == [tuple(e["rgba"]) for e in entries]
    colours.clear()
    render._draw_simple_legend(image, {"labels": ["roads", "labels"]})
    assert not colours
    with pytest.raises(ValueError, match="cannot fit all entries"):
        render._draw_simple_legend(image[:50], {"entries": entries})
    render._draw_simple_legend(image, {"items": [{"label": "Forest", "color": "#2f855a"},
                                                  {"label": "Snow", "color": "#f8fafc"}]})
    assert colours == [(47, 133, 90, 255), (248, 250, 252, 255)]
    with pytest.raises(ValueError, match="explicit RGBA8"):
        render._draw_simple_legend(image, {"items": [{"label": "invalid", "color": "bad-colour"}]})


def test_perspective_scale_needs_explicit_value_and_visible_qualification(publication, monkeypatch):
    from forge3d import _map_scene_render as render
    from forge3d.map_scene import LightingPreset, OrbitCamera, OutputSpec, TerrainSource
    recipe, _, _ = publication
    scene = MapScene(terrain=TerrainSource(data=np.zeros((2, 2)), crs="EPSG:32631"),
                     camera=OrbitCamera(), output=OutputSpec(1000, 600),
                     lighting=LightingPreset(settings={"cli_params": {"camera_mode": "mesh:zup"}}))
    image = np.zeros((600, 1000, 4), np.uint8)
    with pytest.raises(ValueError, match="Automatic scale requires"):
        render._compose_scale_bar(image, scene.recipe, {})
    with pytest.raises(ValueError, match="visible local-scale qualification"):
        render._compose_scale_bar(image, scene.recipe, {"meters_per_pixel": 10})
    labels = []
    monkeypatch.setattr(render, "_draw_text", lambda image, text, *a, **k: labels.append(text))
    render._compose_scale_bar(image, scene.recipe, {"meters_per_pixel": 10, "width_px": 600,
                              "qualification": "at centre, horizontal"})
    assert "at centre, horizontal" in labels[-1]


def test_reusable_layout_insets_regenerate_with_source_and_style(publication, fake_render, tmp_path):
    from forge3d.publication import PublicationLayout, PublicationInset
    recipe, page, _ = publication
    layout = PublicationLayout(replace(recipe.config, width=2000, margin=(100, 460, 170, 450)),
                               [PublicationInset("detail", "Linked detail", page.bounds, (20, 120, 420, 550))])
    recipe.layout = layout
    first = recipe.render_series([page], tmp_path / "first")[0]
    recipe.breaks = [50, 250]
    changed = recipe.render_series([page], tmp_path / "changed")[0]
    assert first["insets"][0]["id"] != changed["insets"][0]["id"]
    assert first["insets"][0]["classification_id"] == first["dependencies"]["classification"]["id"]
    assert len(fake_render) == 6
    assert layout.insets[0].bounds == page.bounds
    assert recipe.config.width == 1600  # Applying a layout never mutates its recipe config.
    recipe.layout = replace(layout, insets=[replace(layout.insets[0], rect=(500, 120, 900, 550))])
    with pytest.raises(ValueError, match="overlaps"):
        recipe.render_series([page], tmp_path / "overlap")


def test_subject_series_uses_declared_source_encoding_and_shared_layout(publication, fake_render, tmp_path):
    from forge3d.publication import PublicationSubject, publication_output_support
    recipe, page, values = publication
    other_path = tmp_path / "households.tif"
    gis.write_raster(other_path, np.where(values < 0, values, values / 2), crs="EPSG:32631",
                     nodata=-9999, transform=(100, 0, 600000, 0, -100, 5820000))
    other = replace(recipe.source, path=other_path, name="Households", units="households/source cell")
    subjects = [PublicationSubject("population", "Population", recipe.source, recipe.breaks, recipe.palette),
                PublicationSubject("comparison", "Households", other, [50, 250], recipe.palette)]
    rows = recipe.render_subjects(subjects, [page], tmp_path)
    assert len(rows) == 2
    assert rows[0]["dependencies"]["classification"] != rows[1]["dependencies"]["classification"]
    assert rows[0]["main_viewport"] == rows[1]["main_viewport"]
    assert rows[0]["source"]["sha256"] != rows[1]["source"]["sha256"]
    assert rows[1]["legend"]["units"] == "households/source cell"
    assert "Households" in rows[1]["caption"]
    assert "/comparison/" in rows[1]["output"]["path"].replace("\\", "/")
    for support in publication_output_support().values():
        assert support["pdf"] is support["svg"] is support["editable_text"] is support["print_colour"] is False
    with pytest.raises(ValueError, match="distinct safe IDs"):
        recipe.render_subjects([subjects[0], replace(subjects[0], subject_id="POPULATION")], [page], tmp_path)


def test_locator_drawn_pixels_independent_geographic_registration(publication, fake_render, tmp_path):
    from PIL import Image
    from rasterio.transform import from_bounds, rowcol
    recipe, page, _ = publication
    # Interior asymmetric rectangle avoids conflating viewport border and footprint.
    bounds = BBox(600500, 5818500, 601500, 5819500, "EPSG:32631")
    result = recipe.render_series([replace(page, bounds=bounds)], tmp_path)[0]
    image = np.asarray(Image.open(result["output"]["path"]).convert("RGBA"))
    x0, y0, x1, y1 = result["locator_rect"]
    locator = image[y0:y1, x0:x1]
    # Rasterio's independent inverse affine registers WORLD coordinates to the
    # saved locator pixels; neither receipt footprint nor recipe helper is used.
    transform = from_bounds(600000, 5818000, 603000, 5820000, x1-x0, y1-y0)
    yy, xx = rowcol(transform, [600500, 601500, 601000, 601000],
                             [5819000, 5819000, 5818500, 5819500])
    for x, y in zip(xx, yy):
        assert np.any(np.all(locator[y-1:y+2, x-1:x+2] == [18, 33, 43, 255], axis=-1))
    # No spurious extent rectangle at the source/map bounds.
    assert not np.any(np.all(locator[0] == [18, 33, 43, 255], axis=-1))


def test_bookends_share_revision_and_final_frame_geometry(publication, fake_render, tmp_path):
    from forge3d.publication import PublicationLayout
    recipe, page, _ = publication
    layout = PublicationLayout(replace(recipe.config, width=1920, height=1080))
    base = recipe.render_bookends(page, replace(page, title="End"), tmp_path / "base", layout=layout)
    recipe.breaks = [50, 250]
    revised = recipe.render_bookends(page, replace(page, title="End"), tmp_path / "revision", layout=layout)
    assert [r["page_id"] for r in revised] == ["title", "end"]
    assert all(r["output"]["size"] == [1920, 1080] for r in revised)
    assert revised[0]["dependencies"]["classification"] == revised[1]["dependencies"]["classification"]
    for old, new in zip(base, revised):
        assert old["dependencies"]["legend"] != new["dependencies"]["legend"]
        assert old["output"]["pixel_sha256"] != new["output"]["pixel_sha256"]


def test_time_series_consumes_c04_source_caption_and_missing_state(publication, fake_render, tmp_path):
    import json
    from forge3d.observation_time import ObservationRecord, ObservationTrack
    recipe, page, _ = publication
    record = ObservationRecord("2024", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z",
                               "2025-01-01T00:00:00Z", {"source": recipe.source.description()})
    times = ["2023-12-31T23:59:59Z", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
    track = ObservationTrack([record], "hold")
    receipts = recipe.render_times(track, times, [page], tmp_path)
    assert [r["selection"]["state"] for r in receipts] == ["missing", "observed", "held"]
    assert receipts[0]["map_rendered"] is False and receipts[0]["source"] is None
    assert "No observation" in receipts[0]["caption"]
    assert len(fake_render) == 4  # A missing time must not manufacture a map.
    assert "Held observation" in receipts[-1]["caption"] and "2024" in receipts[-1]["caption"]
    table = json.loads((tmp_path / "selection-table.json").read_text())
    assert [r["selection"] for r in receipts] == table["selections"]
    assert receipts[1]["dependencies"]["map"] == receipts[2]["dependencies"]["map"]
    assert receipts[1]["dependencies"]["caption"] != receipts[2]["dependencies"]["caption"]
    selected = track.select([times[1]])[0]
    with pytest.raises(ValueError, match="selected record"):
        replace(recipe, source=replace(recipe.source, year="2023")).render_series(
            [page], tmp_path / "wrong", selection=selected)


def test_time_series_rejects_stale_outputs_before_mutation(publication, fake_render, tmp_path):
    from forge3d.observation_time import ObservationTrack
    recipe, page, _ = publication
    output = tmp_path / "series"
    track = ObservationTrack([], "no-data")
    times = ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
    recipe.render_times(track, times, [page], output)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Stale time-series outputs"):
        recipe.render_times(track, times[:1], [page], output)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Stale time-series outputs"):
        recipe.render_times(track, times, [replace(page, page_id="other")], output)


def test_source_year_matches_declared_observation_year(publication, fake_render, tmp_path):
    from forge3d.observation_time import ObservationRecord, ObservationTrack
    recipe, page, _ = publication
    def track(source):
        return ObservationTrack([ObservationRecord("a", "2024-01-01T00:00:00+01:00",
            "2024-01-01T00:00:00+01:00", "2025-01-01T00:00:00+01:00",
            {"source": source.description()})], "hold")
    time = ["2025-01-01T00:00:00+01:00"]
    good = recipe.render_times(track(recipe.source), time, [page], tmp_path / "good")[0]
    assert "Time 2025-01-01T00:00:00+01:00" in good["caption"]
    assert "Age 366 days" in good["caption"]
    wrong = track(replace(recipe.source, year="2031"))
    with pytest.raises(ValueError, match="observation year"):
        recipe.render_times(wrong, time, [page], tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_series_delegates_forward_cache_to_mapscene(publication, fake_render, tmp_path):
    from forge3d.observation_time import ObservationRecord, ObservationTrack
    from forge3d.publication import PublicationLayout, PublicationSubject
    recipe, page, _ = publication
    cache = tmp_path / "cache"
    record = ObservationRecord("a", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z",
                               "2025-01-01T00:00:00Z", {"source": recipe.source.description()})
    recipe.render_times(ObservationTrack([record], "hold"), ["2024-01-01T00:00:00Z"],
                        [page], tmp_path / "time", cache=cache)
    recipe.render_bookends(page, page, tmp_path / "book", layout=PublicationLayout(recipe.config), cache=cache)
    recipe.render_subjects([PublicationSubject("subject", "Subject", recipe.source, recipe.breaks, recipe.palette)],
                           [page], tmp_path / "subject", cache=cache)
    assert fake_render and all(s.last_render_metadata["cache_argument"] == str(cache) for s in fake_render)


def test_missing_page_write_failure_preserves_existing_files(publication, monkeypatch, tmp_path):
    from forge3d.observation_time import ObservationTrack
    from forge3d import publication as module
    recipe, page, _ = publication
    track = ObservationTrack([], "no-data")
    output = tmp_path / "series"
    recipe.render_times(track, ["2024-01-01T00:00:00Z"], [page], output)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    def fail(path, image):
        path.write_bytes(b"incomplete PNG")
        raise OSError("Interrupted write")
    monkeypatch.setattr(module, "save_png", fail)
    with pytest.raises(OSError, match="Interrupted write"):
        recipe.render_times(track, ["2025-01-01T00:00:00Z"], [page], output)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_selection_table_write_is_atomic(publication, monkeypatch, tmp_path):
    from pathlib import Path
    from forge3d.observation_time import ObservationTrack
    recipe, page, _ = publication
    output = tmp_path / "series"
    track = ObservationTrack([], "no-data")
    times = ["2024-01-01T00:00:00Z"]
    recipe.render_times(track, times, [page], output)
    table = output / "selection-table.json"
    original = table.read_bytes()
    write = Path.write_text
    def interrupted(path, text, *args, **kwargs):
        if path.name == "selection-table.json":
            write(path, '{"incomplete":', *args, **kwargs)
            raise OSError("Interrupted table")
        return write(path, text, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", interrupted)
    with pytest.raises(OSError, match="Interrupted table"):
        recipe.render_times(track, times, [page], output)
    assert table.read_bytes() == original
    assert not list(output.glob(".selection-*"))


@pytest.fixture
def publication(tmp_path):
    path = tmp_path / "2024.tif"
    values = np.arange(600, dtype=np.float32).reshape(20, 30)
    values[0, 0] = -9999
    gis.write_raster(path, values, crs="EPSG:32631", nodata=-9999,
                     transform=(100, 0, 600000, 0, -100, 5820000))
    source = PublicationSource(path, "Fixture", "2024", "v2", "https://example.test/2024",
                               "people/source cell", "No published count")
    bounds = BBox(600000, 5818000, 603000, 5820000, "EPSG:32631")
    recipe = PublicationRecipe(source, [100, 300], [(240, 230, 200, 255),
                                (220, 130, 70, 255), (110, 30, 50, 255)], bounds)
    page = PublicationExtent("city", "Fixture city", BBox(600000, 5819000, 601000, 5820000, bounds.crs))
    return recipe, page, values


@pytest.fixture
def fake_render(monkeypatch):
    observed = type("Observed", (list,), {"gain": 1.0})()  # gain < 1 mimics a lit frame.
    def render(scene, path, *, cache=None):
        assert scene.recipe.lighting.settings["unlit_albedo"] is True and not scene.recipe.output.aovs
        rgba = scene.recipe.layers[0].style.apply(scene.recipe.layers[0].data).rgba
        yy = np.arange(scene.recipe.output.height) * rgba.shape[0] // scene.recipe.output.height
        xx = np.arange(scene.recipe.output.width) * rgba.shape[1] // scene.recipe.output.width
        frame = rgba[yy[:, None], xx].copy()
        frame[..., :3] = np.rint(frame[..., :3] * observed.gain)
        save_png(path, frame)
        scene.last_render_backend = "unit-test-fixture"
        scene.last_render_metadata = {"cache_argument": str(cache)}
        observed.append(scene)
    monkeypatch.setattr(MapScene, "render", render)
    return observed


def ids(receipt):
    return {key: value["id"] for key, value in receipt["dependencies"].items()}


def test_revisions_use_current_inputs_and_preserve_unaffected_dependencies(publication, fake_render, tmp_path):
    recipe, page, values = publication
    def run(name):
        return recipe.render_series([page], tmp_path / name, cache=tmp_path / "cache")[0]
    first = run("first")
    same = run("same")
    assert ids(first) == ids(same)
    assert first["output"]["pixel_sha256"] == same["output"]["pixel_sha256"]
    assert len(fake_render) == 4  # Every call still enters MapScene's cache authority.
    recipe.breaks = [50, 250]
    changed = run("breaks")
    for key in ("classification", "legend", "map", "locator"):
        assert ids(first)[key] != ids(changed)[key]
    for key in ("source", "caption", "scale"):
        assert ids(first)[key] == ids(changed)[key]
    assert first["output"]["pixel_sha256"] != changed["output"]["pixel_sha256"]
    recipe.breaks = [100, 300]
    page = replace(page, bounds=BBox(600100, 5819000, 601100, 5820000, "EPSG:32631"))
    moved = run("extent")
    assert ids(first)["map"] != ids(moved)["map"]
    assert ids(first)["locator"] != ids(moved)["locator"]
    assert ids(first)["legend"] == ids(moved)["legend"]
    assert first["footprint"]["coordinates"] != moved["footprint"]["coordinates"]
    newpath = tmp_path / "2023.tif"
    gis.write_raster(newpath, values + 20, crs="EPSG:32631", nodata=-9979,
                     transform=(100, 0, 600000, 0, -100, 5820000))
    recipe.source = replace(recipe.source, path=newpath, year="2023", edition="v1",
                            uri="https://example.test/2023")
    previous = run("year")
    assert "2023" in previous["caption"] and "v1" in previous["caption"]
    for key in ("source", "classification", "caption", "map", "locator"):
        assert ids(moved)[key] != ids(previous)[key]
    # Same path/size/mtime edits must also invalidate the actual map.
    data = values + 50
    previous_stat = newpath.stat()
    gis.write_raster(newpath, data, crs="EPSG:32631", nodata=-9949,
                     transform=(100, 0, 600000, 0, -100, 5820000), overwrite=True)
    os.utime(newpath, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    assert newpath.stat().st_size == previous_stat.st_size
    assert newpath.stat().st_mtime_ns == previous_stat.st_mtime_ns
    edited = run("edited")
    assert ids(previous)["map"] != ids(edited)["map"]
    assert previous["source"]["sha256"] != edited["source"]["sha256"]


def test_native_classifier_closure_palette_units_missing(publication):
    recipe, _, _ = publication
    values = np.array([[0, 99, 100, 300, -9999]], np.float32)
    valid = values != -9999
    for right, expected in ((False, [[1, 1, 2, 3, 0]]), (True, [[1, 1, 1, 2, 0]])):
        recipe.right = right
        classes, style, result, table = recipe._classify(values, valid)
        np.testing.assert_array_equal(classes, expected)
        assert result.legend == style.legend()
        assert result.legend["units"] == recipe.source.units
        assert result.legend["entries"][0]["label"] == recipe.source.missing_label
        assert len(result.legend["entries"]) == 4
        for entry in result.legend["entries"]:
            np.testing.assert_array_equal(result.rgba[classes == entry["class_id"]],
                np.tile(entry["rgba"], (np.count_nonzero(classes == entry["class_id"]), 1)))
        assert sum(row["count"] for row in table) == values.size


def test_registration_and_scale_independent_of_recipe(publication, fake_render, tmp_path):
    from pyproj import Transformer, Geod
    recipe, page, _ = publication
    result = recipe.render_series([page], tmp_path)[0]
    # Independent source affine -> locator pixels, without recipe projection helpers.
    xy = np.array(result["footprint"]["coordinates"])
    expected = (xy - [600000, 5820000]) / [3000, -2000] * result["locator_viewport"]
    np.testing.assert_array_equal(expected, result["footprint"]["locator_pixels"])
    transform = Transformer.from_crs(32631, 4326, always_xy=True)
    lon, lat = transform.transform([600000, 601000], [5819500, 5819500])
    distance = Geod(ellps="WGS84").inv(*[lon[0], lat[0], lon[1], lat[1]])[2]
    # Existing MENSURA/PROJ agreement is measured; round to the publication
    # pixel's ground footprint, rather than inventing a geographic tolerance.
    actual = result["scale"]["meters_per_pixel"] * result["main_viewport"][0]
    assert round(actual / result["scale"]["meters_per_pixel"]) == round(distance / result["scale"]["meters_per_pixel"])
    label, unit = result["scale"]["bar_label"].split()
    labelled_metres = float(label) * {"m": 1, "km": 1000}[unit]
    independent_mpp = distance / result["main_viewport"][0]
    assert abs(labelled_metres - result["scale"]["bar_pixel_span"] * independent_mpp) < independent_mpp


@pytest.mark.parametrize("case", ["camera", "unaligned", "outside", "crs", "locator", "palette", "bins", "empty", "duplicate"])
def test_unsupported_cases_are_explicit(publication, fake_render, tmp_path, case):
    recipe, page, _ = publication
    pages = [page]
    if case == "camera": recipe.camera_mode = "mesh:zup"
    if case == "unaligned": pages = [replace(page, bounds=replace(page.bounds, west=600001))]
    if case == "outside": pages = [replace(page, bounds=replace(page.bounds, west=599900))]
    if case == "crs": pages = [replace(page, bounds=replace(page.bounds, crs="EPSG:4326"))]
    if case == "locator": recipe.locator_bounds = page.bounds; pages = [replace(page, bounds=replace(page.bounds, east=601100))]
    if case == "palette": recipe.palette = [(0, 0, 0, 255)]
    if case == "bins": recipe.breaks = [300, 100]
    if case == "empty": pages = []
    if case == "duplicate": pages = [page, page]
    with pytest.raises((ValueError, TypeError)):
        recipe.render_series(pages, tmp_path / case)
    assert not fake_render


def test_unavailable_native_does_not_return_old_output(publication, monkeypatch, tmp_path):
    recipe, page, _ = publication
    old = tmp_path / "city.png"
    old.write_bytes(b"old output is not a valid result")
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    def unavailable(*args, **kwargs):
        raise MapSceneNativeUnavailable({"layer": "publication", "reason": "test native absence",
                                         "required_native": "physical renderer"})
    monkeypatch.setattr(MapScene, "render", unavailable)
    with pytest.raises(MapSceneNativeUnavailable, match="test native absence"):
        recipe.render_series([page], tmp_path)
    assert hashlib.sha256(old.read_bytes()).hexdigest() == digest


def test_palette_units_and_missing_key_are_linked(publication, fake_render, tmp_path):
    recipe, page, _ = publication
    base = recipe.render_series([page], tmp_path / "base")[0]
    recipe.palette = [(10, 20, 30, 255), *recipe.palette[1:]]
    palette = recipe.render_series([page], tmp_path / "palette")[0]
    assert ids(base)["map"] != ids(palette)["map"]
    assert palette["legend"]["entries"][1]["rgba"] == [10, 20, 30, 255]
    recipe.source = replace(recipe.source, units="revised units", missing_label="Absent or suppressed")
    semantic = recipe.render_series([page], tmp_path / "semantic")[0]
    assert semantic["legend"]["units"] == "revised units"
    assert semantic["legend"]["entries"][0]["label"] == "Absent or suppressed"
    assert ids(palette)["legend"] != ids(semantic)["legend"]
    assert ids(palette)["map"] != ids(semantic)["map"]


@pytest.mark.parametrize("case", ["geographic", "feet", "unsupported_crs", "rotated", "multiband", "empty_data"])
def test_unsupported_source_registration(publication, fake_render, tmp_path, monkeypatch, case):
    recipe, page, values = publication
    crs, transform = "EPSG:32631", (100, 0, 600000, 0, -100, 5820000)
    if case == "geographic": crs = "EPSG:4326"
    if case == "feet": crs = "EPSG:2263"
    if case == "unsupported_crs": crs = "EPSG:28992"
    if case == "rotated": transform = (100, 1, 600000, 0, -100, 5820000)
    if case == "multiband": values = np.stack([values, values])
    if case == "empty_data": values = np.full_like(values, -9999)
    # The native writer itself rejects these CRS codes. Exercise the reader
    # boundary with equivalent metadata from an externally authored GeoTIFF.
    written_crs = "EPSG:32631" if case in ("feet", "unsupported_crs") else crs
    gis.write_raster(recipe.source.path, values, crs=written_crs, transform=transform,
                     nodata=-9999, overwrite=True)
    if case in ("feet", "unsupported_crs"):
        original = gis.read_raster
        def read(*args, **kwargs):
            result = original(*args, **kwargs)
            result["info"]["crs_authority"] = {"name": "EPSG", "code": crs.split(":")[1]}
            return result
        monkeypatch.setattr(gis, "read_raster", read)
    with pytest.raises(ValueError):
        recipe.render_series([page], tmp_path / case)
    assert not fake_render


def test_immutable_categorical_style_can_be_snapshotted(publication):
    import copy
    recipe, _, values = publication
    _, style, _, _ = recipe._classify(values, values >= 0)
    assert copy.deepcopy(style) is style
    with pytest.raises(TypeError):
        style.palette[0] = (0, 0, 0, 255)
    with pytest.raises(TypeError):
        style.labels[0] = "changed"


def test_legend_retains_precise_breaks_and_text_never_silently_clips(publication, fake_render, tmp_path):
    recipe, page, _ = publication
    recipe.breaks = [1.0000001, 1.0000002]
    _, _, result, _ = recipe._classify(np.array([[0, 2]], np.float64), np.ones((1, 2), bool))
    assert "1.0000001" in result.legend["entries"][2]["label"]
    assert "1.0000002" in result.legend["entries"][2]["label"]
    with pytest.raises(ValueError, match="title cannot fit"):
        recipe.render_series([replace(page, title="Long title " * 200)], tmp_path / "long")
    assert not fake_render


def _shown_and_expected(result, values, breaks):
    """Saved page pixels at cell centres vs an independent NumPy classification."""
    from PIL import Image
    image = np.asarray(Image.open(result["output"]["path"]).convert("RGBA")).astype(int)
    x0, y0, x1, y1 = result["map_rect"]
    rows, cols = values.shape
    ys = y0 + ((np.arange(rows) + .5) * (y1 - y0) / rows).astype(int)
    xs = x0 + ((np.arange(cols) + .5) * (x1 - x0) / cols).astype(int)
    classes = np.where(values != -9999, np.searchsorted(breaks, values, side="right") + 1, 0)
    palette = {e["class_id"]: e["rgba"][:3] for e in result["legend"]["entries"]}
    expected = np.array([palette[k] for k in range(len(palette))])[classes]
    bx0, by0, bx1, by1 = result["scale"]["bar_rect"]
    keep = ~((ys[:, None] >= by0) & (ys[:, None] < by1) & (xs[None] >= bx0) & (xs[None] < bx1))
    return image[np.ix_(ys, xs)][..., :3][keep], expected[keep], set(np.unique(classes).tolist())


def test_displayed_map_colours_are_the_legend_colours(publication, fake_render, tmp_path):
    recipe, page, values = publication
    whole = replace(page, bounds=recipe.locator_bounds)  # Values 0-599 plus the missing cell.
    result = recipe.render_series([whole], tmp_path / "ok")[0]
    assert result["main_render"]["max_legend_deviation_levels"] <= 1
    shown, expected, classes = _shown_and_expected(result, values, recipe.breaks)
    assert classes == {0, 1, 2, 3}  # Missing and every interval are exercised.
    assert np.abs(shown - expected).max() <= 1


def test_shaded_frame_or_absent_albedo_is_rejected(publication, fake_render, tmp_path):
    from forge3d.map_scene import OutputSpec, _unlit_albedo_rgba
    recipe, page, _ = publication
    fake_render.gain = 0.46  # The measured lit/tone-mapped darkening of the beauty frame.
    with pytest.raises(ValueError, match="differ from the legend"):
        recipe.render_series([page], tmp_path / "shaded")
    assert not (tmp_path / "shaded" / "city.png").exists()
    with pytest.raises(MapSceneNativeUnavailable, match="albedo"):
        _unlit_albedo_rgba(None, OutputSpec(width=4, height=4))


@pytest.mark.parametrize("field", ["missing_color", "palette"])
def test_non_opaque_keys_are_rejected_not_flattened(publication, fake_render, tmp_path, field):
    recipe, page, _ = publication
    if field == "missing_color":
        recipe.missing_color = (0, 0, 0, 0)
    else:
        recipe.palette = [(240, 230, 200, 128), *recipe.palette[1:]]
    with pytest.raises(ValueError, match="fully opaque"):
        recipe.render_series([page], tmp_path / field)
    assert not fake_render


def test_cell_edges_tolerate_only_representational_rounding(tmp_path):
    path = tmp_path / "fine.tif"
    gis.write_raster(path, np.arange(100, dtype=np.float32).reshape(10, 10), crs="EPSG:32631", nodata=-9999,
                     transform=(0.1, 0, 600000.1, 0, -0.1, 5820000.1))
    source = PublicationSource(path, "Fine", "2024", "v", "https://example.test", "people", "missing")
    bounds = BBox(600000.1, 5819999.1, 600001.1, 5820000.1, "EPSG:32631")
    recipe = PublicationRecipe(source, [10], [(1, 2, 3, 255), (4, 5, 6, 255)], bounds)
    values, _, grid, resolution, _ = recipe._read()
    for k in range(1, 10):  # Several of these edges divide to 0.9999999997..., 3.0000000004...
        west = grid.west + k * resolution[0]
        rows, cols = recipe._window(replace(bounds, west=west), grid, resolution, values.shape)
        assert (rows.start, rows.stop, cols.start, cols.stop) == (0, 10, k, 10)
    for offset in (0.5, 0.01, 1e-6):  # In cells: small, but far beyond rounding.
        with pytest.raises(ValueError, match="cell edges"):
            recipe._window(replace(bounds, west=grid.west + offset * resolution[0]), grid, resolution, values.shape)
    with pytest.raises(ValueError, match="too fine"):
        recipe._window(bounds, grid, (1e-10, 1e-10), values.shape)


def test_scale_bar_is_drawn_on_the_map_not_the_margin(publication, fake_render, tmp_path):
    from PIL import Image
    recipe, page, _ = publication
    narrow = replace(page, bounds=replace(page.bounds, east=600600))  # 6 x 10 cells, centred narrow map.
    result = recipe.render_series([narrow], tmp_path / "narrow")[0]
    x0, y0, x1, y1 = result["map_rect"]
    assert result["main_viewport"] == [x1 - x0, y1 - y0] == [558, 930]
    bx0, by0, bx1, by1 = result["scale"]["bar_rect"]
    assert x0 <= bx0 < bx1 <= x1 and y0 <= by0 < by1 <= y1
    image = np.asarray(Image.open(result["output"]["path"]).convert("RGBA"))
    # The bar's black frame is inside the map; the margin left of it stays background.
    assert np.any(np.all(image[by0:by1, bx0:bx1, :3] == 0, axis=2))
    assert np.all(image[y0:y1, x0 - 30:x0, :3] == 255)
    with pytest.raises(ValueError, match="scale bar cannot fit"):
        recipe.render_series([replace(page, bounds=replace(page.bounds, east=600100))], tmp_path / "sliver")


@pytest.mark.parametrize("width", [60, 300, 479, 480, 1000])
def test_short_scale_label_matches_independently_measured_bar(publication, fake_render, tmp_path, width):
    from PIL import Image
    from pyproj import Geod, Transformer
    recipe, page, values = publication
    cell = width / 10
    bounds = BBox(600000, 5820000 - width, 600000 + width, 5820000, "EPSG:32631")
    gis.write_raster(recipe.source.path, values[:10, :10], crs=bounds.crs, nodata=-9999,
                     transform=(cell, 0, bounds.west, 0, -cell, bounds.north), overwrite=True)
    recipe.locator_bounds = bounds
    page = replace(page, bounds=bounds)
    receipt = recipe.render_series([page], tmp_path / "short")[0]
    transform = Transformer.from_crs(bounds.crs, 4326, always_xy=True)
    lon, lat = transform.transform([bounds.west, bounds.east], [bounds.center_lat] * 2)
    ground_width = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])[2]
    mpp = ground_width / receipt["main_viewport"][0]
    image = np.asarray(Image.open(receipt["output"]["path"]).convert("RGBA"))
    x0, y0, x1, y1 = receipt["scale"]["bar_rect"]
    # Detect the actual border in the exported image, not a second recipe span.
    row = np.flatnonzero(np.all(image[y0 + 12, x0:x1, :3] == 0, axis=1))
    drawn_metres = (row[-1] - row[0] + 1) * mpp
    number, units = receipt["scale"]["bar_label"].split()
    assert units == "m"
    assert abs(float(number) - drawn_metres) <= mpp


def test_wrong_scale_label_cannot_replace_a_page(publication, fake_render, tmp_path, monkeypatch):
    from forge3d.scale_bar import ScaleBar
    recipe, page, _ = publication
    old = tmp_path / "city.png"
    old.write_bytes(b"prior page")
    original = ScaleBar.render_geometry
    def wrong_label(self, **kwargs):
        image, _, anchor = original(self, **kwargs)
        return image, "0.0 km", anchor
    monkeypatch.setattr(ScaleBar, "render_geometry", wrong_label)
    with pytest.raises(ValueError, match="scale bar label"):
        recipe.render_series([page], tmp_path)
    assert old.read_bytes() == b"prior page"


@pytest.mark.parametrize("which", ["missing", "classes"])
@pytest.mark.parametrize("difference", [0, 1, 2])
def test_legend_keys_must_exceed_frame_colour_tolerance(publication, which, difference):
    recipe, _, values = publication
    first = recipe.palette[0]
    close = (first[0] + difference, *first[1:])
    if which == "missing":
        recipe.missing_color = close
    else:
        recipe.palette = [first, close, recipe.palette[2]]
    if difference <= 1:
        with pytest.raises(ValueError, match="distinguishable"):
            recipe._classify(values, values >= 0)
    else:
        recipe._classify(values, values >= 0)


def test_casefolded_page_ids_fail_before_render_or_replacement(publication, fake_render, tmp_path):
    recipe, page, _ = publication
    old = tmp_path / "City.png"
    old.write_bytes(b"prior page")
    with pytest.raises(ValueError, match="distinct page IDs"):
        recipe.render_series([replace(page, page_id="City"), replace(page, page_id="city")], tmp_path)
    assert not fake_render
    assert old.read_bytes() == b"prior page"


def _set_gdal_metadata(path, items):
    """Write actual embedded GeoTIFF metadata, preserving geographic TIFF tags."""
    from PIL import Image
    with Image.open(path) as source:
        pixels = source.copy()
        tags = dict(source.tag_v2)
    tags[42112] = "<GDALMetadata>" + items + "</GDALMetadata>"
    pixels.save(path, tiffinfo=tags)


@pytest.mark.parametrize(("scale", "offset"), [("0.1", "0"), ("1", "5"), ("nan", "0")])
def test_encoded_values_rejected_before_classification(publication, fake_render, tmp_path, scale, offset):
    recipe, page, _ = publication
    _set_gdal_metadata(recipe.source.path,
        f'<Item name="SCALE" sample="0" role="scale">{scale}</Item>'
        f'<Item name="OFFSET" sample="0" role="offset">{offset}</Item>')
    with pytest.raises(ValueError, match="scale/offset"):
        recipe.render_series([page], tmp_path / "encoded")
    assert not fake_render


@pytest.mark.parametrize("year", [None, "2024", "1999", ""])
def test_observation_year_tag_must_match_declaration(publication, fake_render, tmp_path, year):
    recipe, page, _ = publication
    items = '<Item name="SCALE" sample="0" role="scale">1</Item><Item name="OFFSET" sample="0" role="offset">0</Item>'
    if year is not None:
        items += f'<Item name="observation_year">{year}</Item>'
    _set_gdal_metadata(recipe.source.path, items)
    if year in (None, "2024"):
        values, _, _, _, description = recipe._read()
        assert values.shape == (20, 30)
    else:
        with pytest.raises(ValueError, match="observation_year"):
            recipe.render_series([page], tmp_path / "year")
    assert not fake_render


def test_external_metadata_sidecar_is_explicitly_unsupported(publication):
    recipe, _, _ = publication
    from pathlib import Path
    Path(str(recipe.source.path) + ".aux.xml").write_text('<PAMDataset/>', encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        recipe._read()
