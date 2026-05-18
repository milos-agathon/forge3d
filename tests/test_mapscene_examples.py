import importlib.util
from pathlib import Path


EXAMPLES = {
    "terrain_raster": Path("examples/mapscene_terrain_raster.py"),
    "vector_labels": Path("examples/mapscene_vector_labels.py"),
    "buildings_labels": Path("examples/mapscene_buildings_labels.py"),
    "bundled_datasets_showcase": Path("examples/mapscene_bundled_datasets_showcase.py"),
}


def _load_example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_mapscene_examples_exist_and_do_not_use_raw_ipc():
    for path in EXAMPLES.values():
        assert path.exists(), f"missing canonical example: {path}"
        text = path.read_text(encoding="utf-8")
        assert "MapScene" in text
        assert "viewer_ipc" not in text
        assert "send_ipc" not in text


def test_terrain_raster_example_validates_renders_and_bundles(tmp_path):
    module = _load_example(EXAMPLES["terrain_raster"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "ok"
    assert payload["render_status"] == "ok"
    assert payload["render_backend"] == "native/offscreen"
    assert payload["bundle_status"] == "ok"
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()
    assert "placeholder_fallback" not in payload["diagnostic_codes"]


def test_vector_labels_example_compiles_label_plan_and_bundles(tmp_path):
    module = _load_example(EXAMPLES["vector_labels"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "warning"
    assert payload["render_status"] == "warning"
    assert payload["render_backend"] == "native/offscreen"
    assert payload["bundle_status"] == "warning"
    assert payload["accepted_label_ids"] == ["city", "park"]
    assert payload["rejected_label_reasons"] == {"blocked-title": "keepout_region"}
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()


def test_buildings_labels_example_uses_honest_diagnostics_when_blocked(tmp_path):
    module = _load_example(EXAMPLES["buildings_labels"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "error"
    assert payload["render_status"] == "blocked_by_diagnostics"
    assert payload["bundle_status"] == "error"
    assert "pro_gated_path" in payload["diagnostic_codes"]
    assert not Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()


def test_bundled_datasets_showcase_covers_specs_001_to_004(tmp_path):
    module = _load_example(EXAMPLES["bundled_datasets_showcase"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "warning"
    assert payload["render_status"] == "warning"
    assert payload["bundle_status"] == "warning"
    assert payload["dataset_names"] == ["mini_dem", "sample_boundaries"]
    assert payload["accepted_label_ids"] == [
        "label-central-basin",
        "label-north-reserve",
        "label-south-ridge",
    ]
    assert payload["diagnostic_codes"] == ["label_rejection_summary"]
    assert payload["support_levels"]["mapscene.validation"] == "supported"
    assert payload["support_levels"]["mapscene.render_png"] == "supported"
    assert payload["support_levels"]["mapscene.save_bundle"] == "supported"
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()
