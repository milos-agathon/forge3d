from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import forge3d as f3d


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/examples/index.md"
GUIDES = (
    CATALOG,
    ROOT / "docs/guides/data_and_scene_workflows.md",
    ROOT / "docs/guides/feature_map.md",
)
EXAMPLE_REF = re.compile(r"examples/[A-Za-z0-9_./-]+\.(?:py|ipynb|json)")


def _tracked_examples() -> set[str]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "examples"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(output.splitlines())


def _has_main_entrypoint(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
        strings = {
            child.value
            for child in ast.walk(node.test)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        if "__name__" in names and "__main__" in strings:
            return True
    return False


def _refs(path: Path) -> set[str]:
    return set(EXAMPLE_REF.findall(path.read_text(encoding="utf-8")))


def test_catalog_exactly_covers_tracked_runnable_examples_and_support_files():
    tracked = _tracked_examples()
    runnable = {
        path
        for path in tracked
        if path.endswith(".py") and _has_main_entrypoint(ROOT / path)
    }
    notebooks = {path for path in tracked if path.endswith(".ipynb")}
    support = tracked - runnable - notebooks
    catalog_refs = _refs(CATALOG)

    assert catalog_refs & runnable == runnable
    assert catalog_refs & notebooks == notebooks
    assert catalog_refs & support == support
    assert catalog_refs == tracked


def test_catalog_and_guides_use_exact_tracked_example_paths():
    tracked = _tracked_examples()
    for doc in GUIDES:
        text = doc.read_text(encoding="utf-8")
        code_paths = re.findall(r"`([^`]+\.(?:py|ipynb|json))`", text)
        assert code_paths, f"no example paths found in {doc}"
        assert all(path.startswith("examples/") for path in code_paths), doc
        assert set(code_paths) <= tracked, f"stale example reference in {doc}"


def test_fuji_docs_and_source_use_the_public_mapscene_label_interface():
    source = (ROOT / "examples/fuji_labels_demo.py").read_text(encoding="utf-8")
    attributes = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }
    assert {"MapScene", "LabelLayer"} <= attributes
    assert "viewer_ipc" not in source
    assert "ViewerHandle" not in source
    assert f3d.MapScene is not None
    assert f3d.LabelLayer is not None

    for doc in GUIDES:
        text = doc.read_text(encoding="utf-8")
        fuji_lines = [line for line in text.splitlines() if "fuji_labels_demo.py" in line]
        assert all("viewer_ipc" not in line and "ViewerHandle" not in line for line in fuji_lines)
        sections = re.split(r"\n\s*\n", text)
        contexts = [section for section in sections if "fuji_labels_demo.py" in section]
        assert contexts, doc
        context = " ".join(contexts)
        assert "MapScene" in context
        assert "LabelLayer" in context
