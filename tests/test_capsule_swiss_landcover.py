import json
from pathlib import Path

CAPSULE = Path(__file__).resolve().parents[1] / "examples" / "capsules" / "swiss-landcover"


def _registry():
    from forge3d import datasets as ds

    return ds._REMOTE_DATASETS


def test_manifest_matches_registry_hashes():
    manifest = json.loads((CAPSULE / "data-manifest.json").read_text(encoding="utf-8"))
    reg = _registry()
    by_name = {d["name"]: d for d in manifest["datasets"]}
    assert set(by_name) == {"swiss", "swiss-land-cover"}
    for name, entry in by_name.items():
        assert entry["sha256"] == reg[name].known_hash.removeprefix("sha256:")
        assert entry["file"] == reg[name].filename


def test_provenance_complete_no_tbd():
    prov = json.loads((CAPSULE / "provenance.json").read_text(encoding="utf-8"))
    assert {r["dataset"] for r in prov["records"]} == {"swiss", "swiss-land-cover"}
    required = {
        "dataset",
        "distributed_file",
        "distributed_sha256",
        "upstream_provider",
        "upstream_product",
        "upstream_url",
        "license_basis",
        "required_attribution",
        "acquired_date",
        "transformations",
        "notes",
    }
    for record in prov["records"]:
        assert required <= set(record), f"missing keys in {record['dataset']}"
        for key in required:
            assert "TBD" not in json.dumps(record[key]), f"TBD left in {record['dataset']}.{key}"


def test_landcover_attribution_line_exact():
    prov = json.loads((CAPSULE / "provenance.json").read_text(encoding="utf-8"))
    lc = next(r for r in prov["records"] if r["dataset"] == "swiss-land-cover")
    assert lc["required_attribution"] == (
        "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft"
    )


def _load_recipe_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("swiss_capsule_recipe", CAPSULE / "recipe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_recipe_imports_without_gpu_and_declares_contract():
    mod = _load_recipe_module()
    # Proven Swiss PT register, verbatim (spec: do not re-derive).
    assert (mod.GRID_MAX, mod.FRAME, mod.TILES, mod.MAX_FRAMES, mod.SEED) == (1536, 1120, 2, 2048, 7)
    assert mod.SNAPSHOT.name == "swiss_landcover.png"
    assert mod.CERT_PATH.name == "local.certificate.json"
    assert mod.EXIT_NO_GPU == 2 and mod.EXIT_NO_HITS == 3 and mod.EXIT_NO_CERT == 4


def test_recipe_has_no_pro_imports_and_no_telemetry():
    text = (CAPSULE / "recipe.py").read_text(encoding="utf-8")
    for banned in ("map_plate", "export_svg", "export_pdf", "add_buildings", "set_license_key",
                   "requests.", "urllib.request", "httpx"):
        assert banned not in text, f"banned reference in recipe.py: {banned}"
