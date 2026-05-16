import json
from pathlib import Path


SUPPORT_TERMS = {
    "supported",
    "underdeveloped",
    "missing",
    "Pro-gated",
    "placeholder/fallback",
    "experimental",
    "unsupported",
    "non-goal",
}


REQUIRED_LABEL_CAPABILITIES = {
    "point labels": "supported",
    "line labels": "supported",
    "curved labels": "experimental",
    "callouts": "supported",
    "typography controls": "experimental",
    "decluttering controls": "experimental",
    "atlas loading": "supported",
    "missing glyph diagnostics": "supported",
    "upside-down line handling": "supported",
    "terrain-elevated line labels": "experimental",
    "deterministic labelplan": "supported",
}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _support_rows(path: str):
    rows = {}
    for line_number, line in enumerate(_read(path).splitlines(), start=1):
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] == "Capability":
            continue
        rows[cells[0].strip("`").lower()] = {
            "line": line_number,
            "support": cells[1].strip("`"),
            "scope": cells[2],
            "diagnostics": cells[3],
        }
    return rows


def test_label_support_matrix_covers_public_api_support_levels():
    rows = _support_rows("docs/guides/label_support_matrix.md")

    for capability, expected_support in REQUIRED_LABEL_CAPABILITIES.items():
        assert capability in rows, f"missing support row for {capability}"
        row = rows[capability]
        assert row["support"] in SUPPORT_TERMS, f"invalid support term at row {row['line']}"
        assert row["support"] == expected_support, capability

    assert "stable id" in _read("docs/guides/label_support_matrix.md").lower()
    assert "raw ipc" in _read("docs/guides/label_support_matrix.md").lower()


def test_label_docs_do_not_overclaim_experimental_paths():
    docs = "\n".join(
        [
            _read("docs/guides/label_support_matrix.md"),
            _read("docs/api/api_reference.rst"),
            _read("docs/guides/feature_map.md"),
        ]
    ).lower()

    forbidden = [
        "curved labels are supported",
        "curved labels are production-ready",
        "typography controls are supported",
        "decluttering controls are supported",
        "terrain-elevated line labels are supported",
    ]
    for phrase in forbidden:
        assert phrase not in docs

    assert "experimental_feature" in docs
    assert "placeholder_fallback" in docs
    assert "missing_glyphs" in docs


def test_public_docs_prefer_viewerhandle_label_workflow_over_raw_ipc():
    api_reference = _read("docs/api/api_reference.rst")
    feature_map = _read("docs/guides/feature_map.md")

    assert "ViewerHandle.add_label" in api_reference
    assert "ViewerHandle.add_labels" in api_reference
    assert "ViewerHandle.add_line_label" in api_reference
    assert "ViewerHandle.add_callout" in api_reference
    assert "ViewerHandle.load_label_atlas" in api_reference
    assert "ViewerHandle.set_label_typography" in api_reference
    assert "ViewerHandle.set_declutter_algorithm" in api_reference

    vector_label_row = next(
        line for line in feature_map.splitlines() if line.startswith("| Vector overlays and labels |")
    )
    assert "ViewerHandle.add_label" in vector_label_row
    assert "ViewerHandle.add_vector_overlay" in vector_label_row
    assert "viewer_ipc.add_label" not in vector_label_row


def test_label_contract_matches_current_viewerhandle_signatures():
    contract = _read("specs/002-label-api-truth/contracts/label-api-contract.md")

    expected_fragments = [
        "viewer.add_label(text: str, world_pos: tuple[float, float, float]",
        "viewer.add_line_label(text: str, polyline: Sequence[Point3]",
        "viewer.add_curved_label(text: str, path: Sequence[Point3]",
        "viewer.load_label_atlas(atlas_png_path: str | Path, metrics_json_path: str | Path)",
        "viewer.add_labels(labels: Sequence[Mapping[str, object]]) -> LabelBatchResult",
        "viewer.set_label_typography(*, tracking: float | None",
        "viewer.set_declutter_algorithm(algorithm: str, *, seed: int | None",
    ]
    for fragment in expected_fragments:
        assert fragment in contract

    stale_fragments = [
        "position: tuple",
        "path: Sequence[Point3], **style) -> str | DiagnosticResult",
        "load_label_atlas(path:",
        "settings: Mapping[str, object]",
    ]
    for fragment in stale_fragments:
        assert fragment not in contract


def test_label_api_truth_artifacts_exist_independently_of_active_feature_pointer():
    payload = json.loads(_read(".specify/feature.json"))

    active_feature = Path(payload["feature_directory"])
    assert payload["feature_directory"].startswith("specs/")
    assert active_feature.exists()

    label_api_truth_dir = Path("specs/002-label-api-truth")
    for relative_path in [
        "spec.md",
        "plan.md",
        "tasks.md",
        "contracts/label-api-contract.md",
        "quickstart.md",
    ]:
        assert (label_api_truth_dir / relative_path).exists()
