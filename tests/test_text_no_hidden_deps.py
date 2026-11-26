from pathlib import Path
import hashlib
import json
import tomllib
import ast

import forge3d

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATHS = ["text.py", "text_atlas.py", "label_plan.py", "_map_scene_render.py", "scale_bar.py", "north_arrow.py", "legend.py", "map_plate.py"]


def test_text_routes_use_native_outlines_without_hidden_dependencies():
    forbidden = ["_draw_text_fallback", "ImageFont", "from PIL", "import PIL", "scipy"]
    counts = {name: 0 for name in forbidden}
    for name in TEXT_PATHS:
        source = (ROOT / "python/forge3d" / name).read_text(encoding="utf-8")
        if name == "map_plate.py":
            # Its optional JPEG encoder does not render text. Audit the title
            # text route, matching the existing furniture dependency contract.
            tree = ast.parse(source)
            method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_render_title")
            source = ast.get_source_segment(source, method)
        for token in forbidden:
            counts[token] += source.count(token)
    print("text source counts:", counts)
    assert all(count == 0 for count in counts.values())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["dependencies"] == ["numpy>=1.21"]


def test_shaped_corpus_hash_is_locally_byte_identical():
    cases = []
    for path in sorted((ROOT / "tests/data/shaping").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.extend(payload["cases"] if isinstance(payload, dict) else payload)
    def shaped_bytes():
        records = []
        for case in cases:
            shaped = forge3d.text.shape(case["text"], [str(ROOT / font) for font in case["fonts"]], case["size"], **case["options"])
            payload = shaped.to_dict()
            # Source filenames are platform-dependent. Hash the immutable font
            # content and complete shaping/positioning payload instead.
            payload.pop("font_sources")
            records.append(payload)
        return json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    first, second = shaped_bytes(), shaped_bytes()
    assert first == second
    print(f"shaped strings={len(cases)} local SHA256={hashlib.sha256(first).hexdigest()}")


def test_atlas_gpu_allocations_remain_tracked():
    for name in ("src/labels/atlas.rs", "src/core/text_overlay.rs"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "tracked_create_texture(" in source
        assert "device.create_texture(" not in source
