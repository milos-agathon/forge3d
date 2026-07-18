"""DUPLA double-float source and generation contracts."""

from __future__ import annotations

import re
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WGSL = ROOT / "src" / "shaders" / "includes" / "dd.wgsl"
RUST = ROOT / "src" / "core" / "dd.rs"
GENERATOR = ROOT / "scripts" / "generate_dd.py"


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}\b[^{{]*\{{", source)
    assert match, f"missing function {name}"
    depth = 1
    cursor = match.end()
    while cursor < len(source) and depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    assert depth == 0, f"unterminated function {name}"
    return source[match.end() : cursor - 1]


def _without_comments(source: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)


def test_generated_mirrors_are_current() -> None:
    assert GENERATOR.is_file()
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lockstep operation order verified" in result.stdout


def test_generator_rejects_cross_language_statement_reordering() -> None:
    spec = importlib.util.spec_from_file_location("generate_dd", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rust = RUST.read_text(encoding="utf-8")
    wgsl = WGSL.read_text(encoding="utf-8")
    first = "let e1 = e0 + det_barrier(sa.hi * sb.lo);"
    second = "let e2 = e1 + det_barrier(sa.lo * sb.hi);"
    mutated = wgsl.replace(first + "\n    " + second, second + "\n    " + first)
    assert mutated != wgsl
    try:
        module.validate_lockstep(rust, mutated)
    except SystemExit as error:
        assert "operation-order mismatch" in str(error)
    else:
        raise AssertionError("generator accepted divergent Rust/WGSL statement order")

    newton = "y = det_barrier(y * (1.5 - det_barrier(half_x * det_barrier(y * y))));"
    reordered = "y = det_barrier((1.5 - det_barrier(half_x * det_barrier(y * y))) * y);"
    mutated = wgsl.replace(newton, reordered, 1)
    assert mutated != wgsl
    try:
        module.validate_lockstep(rust, mutated)
    except SystemExit as error:
        assert "operation-order mismatch" in str(error)
    else:
        raise AssertionError("generator accepted divergent Newton operand order")


def test_wgsl_exports_complete_dd_surface_and_sources() -> None:
    source = WGSL.read_text(encoding="utf-8")
    assert "struct DD" in source
    assert "struct DDVec3" in source
    for name in (
        "two_sum",
        "quick_two_sum",
        "two_prod_split",
        "two_prod_fma",
        "two_prod",
        "dd_renorm",
        "dd_add",
        "dd_sub",
        "dd_mul",
        "dd_div",
        "dd_sqrt",
        "dd_dot3",
        "dd_length3",
        "dd_sub_vec3",
    ):
        _function_body(source, name)
    for citation in ("Knuth", "Dekker", "Veltkamp", "Joldes", "Muller", "Popescu"):
        assert citation in source
    assert "DD_ADD_BOUND_U2: f32 = 3.0" in source
    assert "DD_MUL_BOUND_U2: f32 = 7.0" in source
    assert "DD_DIV_BOUND_U2: f32 = 15.0" in source
    assert "DD_SQRT_BOUND_U2: f32 = 15.0" in source
    assert "fn dd_barrier" not in source
    assert "det_barrier" in source
    assert "Requires determinism.wgsl" in source


def test_codegen_files_respect_repository_size_guideline() -> None:
    for path in (GENERATOR, ROOT / "scripts" / "dd.rs.in", ROOT / "scripts" / "dd.wgsl.in", RUST):
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 300, path


def test_error_free_and_refinement_bodies_avoid_forbidden_ops() -> None:
    source = _without_comments(WGSL.read_text(encoding="utf-8"))
    for name in (
        "two_sum",
        "quick_two_sum",
        "two_prod_split",
        "dd_renorm",
        "dd_add",
        "dd_sub",
        "dd_mul",
        "dd_reciprocal_refine",
        "dd_div",
        "dd_inverse_sqrt_refine",
        "dd_sqrt",
    ):
        body = _function_body(source, name)
        assert "/" not in body, f"{name} contains hardware division"
        assert not re.search(r"\bsqrt\s*\(", body), f"{name} contains hardware sqrt"
    seed = _function_body(source, "dd_inverse_sqrt_seed")
    assert seed.count("inverseSqrt(") == 1


def test_product_selection_is_shader_compile_time_not_runtime() -> None:
    source = WGSL.read_text(encoding="utf-8")
    body = _function_body(source, "two_prod")
    assert "__DD_TWO_PROD_CALL__" in body
    assert "select(" not in body
    assert "if" not in body
