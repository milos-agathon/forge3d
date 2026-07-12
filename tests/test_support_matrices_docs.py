from pathlib import Path
import subprocess


DOCS = {
    "docs/guides/offline_3d_map_rendering.md",
    "docs/guides/diagnostics_reference.md",
    "docs/guides/style_support_matrix.md",
    "docs/guides/label_support_matrix.md",
    "docs/guides/building_support_matrix.md",
    "docs/guides/tiles3d_support_matrix.md",
    "docs/guides/virtual_texturing_support_matrix.md",
    "docs/guides/competitive_positioning.md",
}

GIT_VISIBLE_EVIDENCE = {
    "specs/001-diagnostics-support-matrices/tasks.md",
    "docs/guides/diagnostics_reference.md",
    "docs/guides/style_support_matrix.md",
    "docs/guides/label_support_matrix.md",
    "docs/guides/building_support_matrix.md",
    "docs/guides/tiles3d_support_matrix.md",
    "docs/guides/virtual_texturing_support_matrix.md",
    "docs/superpowers/state/requirements-verification-matrix.md",
    "docs/superpowers/state/implementation-ledger.md",
    "docs/superpowers/state/current-context-pack.md",
}

SUPPORT_MATRIX_DOCS = {
    "docs/guides/style_support_matrix.md",
    "docs/guides/label_support_matrix.md",
    "docs/guides/building_support_matrix.md",
    "docs/guides/tiles3d_support_matrix.md",
    "docs/guides/virtual_texturing_support_matrix.md",
}

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

DIAGNOSTIC_CODES = {
    "crs_mismatch",
    "missing_glyphs",
    "unsupported_style_field",
    "unsupported_style_layer_type",
    "pro_gated_path",
    "placeholder_fallback",
    "experimental_feature",
    "vt_unsupported_family",
    "python_public_3dtiles_incomplete",
    "estimated_gpu_memory",
    "label_rejection_summary",
}


def _markdown_table_rows(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == "Capability":
            continue
        if len(cells) >= 4:
            yield index + 1, cells


def test_required_evidence_files_are_not_git_ignored():
    ignored = []
    for path in sorted(GIT_VISIBLE_EVIDENCE):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", path],
            check=False,
        )
        if result.returncode == 0:
            ignored.append(path)
    assert ignored == []


def test_required_support_matrix_docs_exist_and_use_prd_terms():
    for doc in DOCS:
        path = Path(doc)
        assert path.exists(), f"missing required doc: {doc}"
        text = path.read_text(encoding="utf-8")
        assert any(term in text for term in SUPPORT_TERMS), doc
        assert "partial support" not in text.lower()


def test_support_matrix_rows_use_prd_terms_and_remediation_columns():
    for doc in sorted(SUPPORT_MATRIX_DOCS):
        rows = list(_markdown_table_rows(Path(doc)))
        assert rows, f"missing support matrix rows: {doc}"
        for line_number, cells in rows:
            capability, support_level, _scope, diagnostics = cells[:4]
            assert support_level.strip("`") in SUPPORT_TERMS, f"{doc}:{line_number} {capability}"
            if support_level.strip("`") != "supported":
                diagnostic_text = diagnostics.lower()
                assert diagnostics and (
                    any(code in diagnostics for code in DIAGNOSTIC_CODES)
                    or "diagnostic" in diagnostic_text
                    or "documentation boundary" in diagnostic_text
                    or "future" in diagnostic_text
                ), f"{doc}:{line_number} {capability}"


def test_diagnostics_reference_lists_required_codes_and_fields():
    text = Path("docs/guides/diagnostics_reference.md").read_text(encoding="utf-8")

    for code in sorted(DIAGNOSTIC_CODES):
        assert code in text
    for field in ["code", "severity", "message", "remediation", "support_level", "layer_id", "object_id"]:
        assert field in text


def test_style_support_matrix_states_local_scope_and_layers_without_overclaim():
    text = Path("docs/guides/style_support_matrix.md").read_text(encoding="utf-8")

    assert "local/provided features" in text
    assert "streamed MVT" in text
    assert "`fill`" in text
    assert "`line`" in text
    assert "`circle`" in text
    assert "full Mapbox Style Specification support" not in text


def test_public_style_api_docstrings_do_not_overclaim_mapbox_parity():
    text = Path("python/forge3d/style.py").read_text(encoding="utf-8")

    forbidden_claims = [
        "Complete Mapbox GL Style specification",
        "complete Mapbox GL Style specification",
        "full Mapbox Style Specification support",
    ]

    for claim in forbidden_claims:
        assert claim not in text
