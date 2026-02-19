#!/usr/bin/env python3
"""
Generate a reproducible audit snapshot for forge3d.

This script generates three sections intended to be pasted into an audit doc:

1. Codebase Scale Overview
2. Python Wrapper Classification
3. Duplicate Implementations (Rust vs Python)

It is intentionally evidence-first: each section is derived from repository state
instead of manual estimates.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ORCHESTRATION_MODULES = {"__init__.py", "render.py", "terrain_demo.py", "viewer.py"}


@dataclass
class WrapperClassification:
    module: str
    category: str
    reason: str
    line_count: int
    native_refs: int
    native_calls: int


@dataclass
class DuplicateFeature:
    name: str
    rust_paths: Sequence[str]
    python_paths: Sequence[str]
    rust_binding_patterns: Sequence[str]
    python_api_patterns: Sequence[str]
    python_native_patterns: Sequence[str]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def run_git(root: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def list_files(base: Path, suffix: str) -> List[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.rglob(f"*{suffix}") if p.is_file())


def list_python_package_files(root: Path) -> List[Path]:
    pkg = root / "python" / "forge3d"
    return list_files(pkg, ".py")


def list_python_top_level_modules(root: Path) -> List[Path]:
    pkg = root / "python" / "forge3d"
    if not pkg.exists():
        return []
    return sorted(p for p in pkg.glob("*.py") if p.is_file())


def list_rust_source_files(root: Path) -> List[Path]:
    return list_files(root / "src", ".rs")


def parse_top_level_modules(root: Path) -> List[str]:
    lib_rs = root / "src" / "lib.rs"
    text = read_text(lib_rs)
    modules = set(re.findall(r"^\s*pub\s+mod\s+([A-Za-z_][A-Za-z0-9_]*)\b", text, re.MULTILINE))
    return sorted(modules)


def count_pattern_in_files(files: Iterable[Path], pattern: str, flags: int = 0) -> int:
    rx = re.compile(pattern, flags)
    count = 0
    for file in files:
        count += len(rx.findall(read_text(file)))
    return count


def files_matching_pattern(files: Iterable[Path], pattern: str, flags: int = 0) -> List[Path]:
    rx = re.compile(pattern, flags)
    matches = []
    for file in files:
        if rx.search(read_text(file)):
            matches.append(file)
    return matches


def is_test_file(path: Path) -> bool:
    name = path.name.lower()
    parts = {p.lower() for p in path.parts}
    if "tests" in parts:
        return True
    if name.startswith("test_"):
        return True
    if name.endswith("_test.py") or name.endswith("_tests.py"):
        return True
    if name.endswith("_test.rs") or name.endswith("_tests.rs"):
        return True
    return False


def gather_test_files(root: Path) -> List[Path]:
    candidates = []
    for base in (root / "src", root / "tests", root / "python"):
        candidates.extend(list_files(base, ".rs"))
        candidates.extend(list_files(base, ".py"))
    unique = sorted(set(candidates))
    return [p for p in unique if is_test_file(p)]


def classify_module(path: Path, override: Optional[str] = None) -> WrapperClassification:
    text = read_text(path)
    line_count = text.count("\n") + (1 if text else 0)

    native_ref_rx = re.compile(r"\b(_forge3d|get_native_module|NATIVE_AVAILABLE|forge3d_native)\b")
    native_call_rx = re.compile(r"\b(_forge3d|forge3d_native|_NATIVE)\.[A-Za-z_][A-Za-z0-9_]*\s*\(")

    native_refs = len(native_ref_rx.findall(text))
    native_calls = len(native_call_rx.findall(text))

    if override:
        return WrapperClassification(
            module=path.name,
            category=override,
            reason="manual override",
            line_count=line_count,
            native_refs=native_refs,
            native_calls=native_calls,
        )

    if path.name in ORCHESTRATION_MODULES:
        return WrapperClassification(
            module=path.name,
            category="Orchestration",
            reason="known workflow/entry module",
            line_count=line_count,
            native_refs=native_refs,
            native_calls=native_calls,
        )

    if native_refs == 0 and native_calls == 0:
        return WrapperClassification(
            module=path.name,
            category="Pure Python",
            reason="no native binding references detected",
            line_count=line_count,
            native_refs=native_refs,
            native_calls=native_calls,
        )

    if native_calls <= 3 and line_count < 260:
        return WrapperClassification(
            module=path.name,
            category="Thin wrapper",
            reason="small module with limited native call surface",
            line_count=line_count,
            native_refs=native_refs,
            native_calls=native_calls,
        )

    return WrapperClassification(
        module=path.name,
        category="Thick wrapper",
        reason="non-trivial module with native integration",
        line_count=line_count,
        native_refs=native_refs,
        native_calls=native_calls,
    )


def infer_duplicate_conclusion(
    rust_present: bool,
    python_present: bool,
    rust_bound: bool,
    python_public: bool,
    native_signal: bool,
) -> Tuple[str, str]:
    if rust_present and python_present:
        if rust_bound and native_signal:
            return "Both (inferred)", "Medium"
        if python_public and not rust_bound:
            return "Python path (inferred)", "Medium"
        if rust_bound and not python_public:
            return "Rust-bound but not public API (inferred)", "Low"
        return "Unknown (trace call sites)", "Low"
    if rust_present and not python_present:
        return "Rust only", "High"
    if python_present and not rust_present:
        return "Python only", "High"
    return "Not present", "Low"


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def format_list(values: Sequence[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(values)


def load_overrides(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(read_text(path))
    if not isinstance(data, dict):
        raise ValueError("Overrides file must be a JSON object of 'module.py' -> category")
    out: Dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Overrides keys and values must be strings")
        out[key] = value
    return out


def build_section_1(root: Path, rust_files: Sequence[Path], py_files: Sequence[Path]) -> Tuple[str, Dict[str, object]]:
    top_level_modules = parse_top_level_modules(root)

    pyclass_count = count_pattern_in_files(rust_files, r"\bpyclass\s*\(")
    pyfunction_count = count_pattern_in_files(rust_files, r"#\[\s*pyfunction\s*\]")
    pyo3_annotated_files = files_matching_pattern(
        rust_files,
        r"\bpyclass\s*\(|#\[\s*pyfunction\s*\]|#\[\s*pymethods\s*\]|#\[\s*pymodule\s*\]",
    )

    test_files = gather_test_files(root)
    rust_test_fn_count = count_pattern_in_files(rust_files, r"#\[\s*test\s*\]")
    py_test_fn_count = count_pattern_in_files(
        list_files(root / "python", ".py") + list_files(root / "tests", ".py"),
        r"^\s*def\s+test_[A-Za-z0-9_]*\s*\(",
        flags=re.MULTILINE,
    )

    viewer_ipc_present = (root / "src" / "viewer" / "ipc").exists() and (
        root / "python" / "forge3d" / "viewer_ipc.py"
    ).exists()

    metrics = {
        "rust_source_files": len(rust_files),
        "rust_top_level_modules": len(top_level_modules),
        "python_package_files": len(py_files),
        "pyo3_classes": pyclass_count,
        "pyo3_functions": pyfunction_count,
        "rust_files_with_pyo3_annotations": len(pyo3_annotated_files),
        "test_files": len(test_files),
        "test_functions_total": rust_test_fn_count + py_test_fn_count,
        "test_functions_rust": rust_test_fn_count,
        "test_functions_python": py_test_fn_count,
        "viewer_ipc_present": viewer_ipc_present,
        "top_level_module_names": top_level_modules,
    }

    lines = []
    lines.append("## Section 1: Codebase Scale Overview")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Rust source files (`src/**/*.rs`) | {metrics['rust_source_files']} |")
    lines.append(f"| Rust top-level modules (`pub mod` in `src/lib.rs`) | {metrics['rust_top_level_modules']} |")
    lines.append(f"| Python package files (`python/forge3d/**/*.py`) | {metrics['python_package_files']} |")
    lines.append(f"| PyO3 classes (`pyclass`) | {metrics['pyo3_classes']} |")
    lines.append(f"| PyO3 functions (`#[pyfunction]`) | {metrics['pyo3_functions']} |")
    lines.append(
        "| Rust files with PyO3 annotations | "
        f"{metrics['rust_files_with_pyo3_annotations']} "
        f"({(100.0 * metrics['rust_files_with_pyo3_annotations'] / max(1, metrics['rust_source_files'])):.1f}% of Rust files) |"
    )
    lines.append(f"| Test files (Rust + Python, naming/path heuristic) | {metrics['test_files']} |")
    lines.append(
        f"| Test functions (Rust `#[test]` + Python `def test_*`) | {metrics['test_functions_total']} |"
    )
    lines.append("")
    lines.append("Architecture signals:")
    lines.append("- Rust core + PyO3 bindings: detected")
    lines.append(f"- Viewer IPC (TCP/NDJSON path): {'detected' if viewer_ipc_present else 'not detected'}")
    lines.append(
        "- Rust top-level modules: "
        + format_list(metrics["top_level_module_names"][:20])
        + (" ..." if len(metrics["top_level_module_names"]) > 20 else "")
    )
    lines.append("")

    return "\n".join(lines), metrics


def build_section_3(root: Path, overrides: Dict[str, str]) -> Tuple[str, Dict[str, object]]:
    modules = list_python_top_level_modules(root)
    classified: List[WrapperClassification] = []
    by_category: Dict[str, List[str]] = defaultdict(list)

    for module in modules:
        override = overrides.get(module.name)
        result = classify_module(module, override=override)
        classified.append(result)
        by_category[result.category].append(module.stem)

    ordered_categories = ["Pure Python", "Thick wrapper", "Thin wrapper", "Orchestration"]

    lines = []
    lines.append("## Section 3: Python Wrapper Classification")
    lines.append("")
    lines.append("Classification rules:")
    lines.append("- `Orchestration`: known workflow/entry modules (`__init__`, `render`, `terrain_demo`, `viewer`) unless overridden.")
    lines.append("- `Pure Python`: no native-binding references/calls detected.")
    lines.append("- `Thin wrapper`: native references present but limited call surface in a small module.")
    lines.append("- `Thick wrapper`: larger module with non-trivial native integration.")
    lines.append("")
    lines.append("| Category | Count | Files |")
    lines.append("|---|---:|---|")
    for category in ordered_categories:
        files = sorted(by_category.get(category, []))
        lines.append(f"| {category} | {len(files)} | {format_list(files)} |")
    lines.append("")
    lines.append("Per-module details:")
    lines.append("")
    lines.append("| File | Category | LOC | Native refs | Native calls | Reason |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in sorted(classified, key=lambda x: x.module):
        lines.append(
            f"| `python/forge3d/{row.module}` | {row.category} | {row.line_count} | "
            f"{row.native_refs} | {row.native_calls} | {row.reason} |"
        )
    lines.append("")

    raw = {
        "module_count": len(modules),
        "categories": {k: sorted(v) for k, v in by_category.items()},
        "rows": [
            {
                "module": r.module,
                "category": r.category,
                "reason": r.reason,
                "line_count": r.line_count,
                "native_refs": r.native_refs,
                "native_calls": r.native_calls,
            }
            for r in classified
        ],
    }
    return "\n".join(lines), raw


def build_section_5(root: Path) -> Tuple[str, Dict[str, object]]:
    binding_text = read_text(root / "src" / "lib.rs") + "\n" + read_text(root / "src" / "scene" / "mod.rs")
    init_py_text = read_text(root / "python" / "forge3d" / "__init__.py")
    python_tree_text = "\n".join(read_text(p) for p in list_files(root / "python" / "forge3d", ".py"))

    features = [
        DuplicateFeature(
            name="SVG export",
            rust_paths=("src/export/svg.rs", "src/export/svg_labels.rs"),
            python_paths=("python/forge3d/export.py",),
            rust_binding_patterns=(r"vectors_to_svg", r"labels_to_svg_text"),
            python_api_patterns=(r"from \.export import", r"\bexport_svg\b", r"\bexport_pdf\b"),
            python_native_patterns=(r"_forge3d\.[A-Za-z0-9_]*svg",),
        ),
        DuplicateFeature(
            name="Mapbox style parser",
            rust_paths=("src/style/mod.rs", "src/style/parser.rs"),
            python_paths=("python/forge3d/style.py", "python/forge3d/style_expressions.py"),
            rust_binding_patterns=(r"\bparse_style\b",),
            python_api_patterns=(r"from \.style import", r"\bparse_style\b", r"\bapply_style\b"),
            python_native_patterns=(r"_forge3d\.[A-Za-z0-9_]*style",),
        ),
        DuplicateFeature(
            name="3D Tiles parser",
            rust_paths=("src/tiles3d/mod.rs",),
            python_paths=("python/forge3d/tiles3d.py",),
            rust_binding_patterns=(r"wrap_pyfunction!\([^)]*tiles3d", r"m\.add_class::<[^>]*Tiles"),
            python_api_patterns=(r"\btiles3d\b",),
            python_native_patterns=(r"_forge3d\.[A-Za-z0-9_]*tiles",),
        ),
        DuplicateFeature(
            name="Bundle format",
            rust_paths=("src/bundle/mod.rs",),
            python_paths=("python/forge3d/bundle.py",),
            rust_binding_patterns=(r"\bsave_bundle\b", r"\bload_bundle\b", r"\bis_bundle\b"),
            python_api_patterns=(r"\bsave_bundle\b", r"\bload_bundle\b", r"\bis_bundle\b"),
            python_native_patterns=(r"_forge3d\.[A-Za-z0-9_]*bundle",),
        ),
        DuplicateFeature(
            name="SDF raymarcher / hybrid render",
            rust_paths=("src/sdf/mod.rs", "src/sdf/hybrid.rs"),
            python_paths=("python/forge3d/sdf.py",),
            rust_binding_patterns=(r"\bhybrid_render\b",),
            python_api_patterns=(r"\bsdf\b",),
            python_native_patterns=(r"\bhybrid_render\s*\(",),
        ),
        DuplicateFeature(
            name="Path tracer",
            rust_paths=("src/path_tracing/mod.rs", "src/path_tracing/compute.rs"),
            python_paths=("python/forge3d/path_tracing.py", "python/forge3d/render.py"),
            rust_binding_patterns=(r"_pt_render_gpu_mesh", r"_pt_render_gpu"),
            python_api_patterns=(r"\bPathTracer\b", r"\brender_raytrace_mesh\b"),
            python_native_patterns=(r"_pt_render_gpu_mesh\s*\(",),
        ),
        DuplicateFeature(
            name="Point cloud LOD/rendering",
            rust_paths=("src/pointcloud/mod.rs", "src/pointcloud/renderer.rs"),
            python_paths=("python/forge3d/pointcloud.py",),
            rust_binding_patterns=(r"\bset_point_lod_threshold\b", r"\bset_point_shape_mode\b"),
            python_api_patterns=(r"\bpointcloud\b",),
            python_native_patterns=(r"_forge3d\.[A-Za-z0-9_]*point",),
        ),
        DuplicateFeature(
            name="Denoise",
            rust_paths=("src/core/postfx/chain.rs",),
            python_paths=("python/forge3d/denoise.py",),
            rust_binding_patterns=(r"\bdenoise\b",),
            python_api_patterns=(r"\bdenoise\b",),
            python_native_patterns=(r"_forge3d\.[A-Za-z0-9_]*denoise",),
        ),
    ]

    rows = []
    lines = []
    lines.append("## Section 5: Duplicate Implementations (Rust vs Python)")
    lines.append("")
    lines.append("Note: `Likely route` is an inference from binding/export/call signals, not full runtime tracing.")
    lines.append("")
    lines.append("| Feature | Rust present | Python present | Rust bound to Python | Public Python API | Native-call signal in Python | Likely route | Confidence |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for feature in features:
        rust_present = any((root / p).exists() for p in feature.rust_paths)
        python_present = any((root / p).exists() for p in feature.python_paths)

        rust_bound = any(re.search(pattern, binding_text) for pattern in feature.rust_binding_patterns)
        python_public = any(re.search(pattern, init_py_text) for pattern in feature.python_api_patterns)
        native_signal = any(re.search(pattern, python_tree_text) for pattern in feature.python_native_patterns)

        likely_route, confidence = infer_duplicate_conclusion(
            rust_present=rust_present,
            python_present=python_present,
            rust_bound=rust_bound,
            python_public=python_public,
            native_signal=native_signal,
        )

        lines.append(
            f"| {feature.name} | {yes_no(rust_present)} | {yes_no(python_present)} | "
            f"{yes_no(rust_bound)} | {yes_no(python_public)} | {yes_no(native_signal)} | "
            f"{likely_route} | {confidence} |"
        )

        rows.append(
            {
                "name": feature.name,
                "rust_present": rust_present,
                "python_present": python_present,
                "rust_bound_to_python": rust_bound,
                "python_public_api": python_public,
                "python_native_call_signal": native_signal,
                "likely_route": likely_route,
                "confidence": confidence,
            }
        )

    lines.append("")
    return "\n".join(lines), {"rows": rows}


def build_snapshot(root: Path, overrides: Dict[str, str]) -> Tuple[str, Dict[str, object]]:
    rust_files = list_rust_source_files(root)
    py_files = list_python_package_files(root)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    commit = run_git(root, ["rev-parse", "HEAD"])
    branch = run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])

    section1_md, section1_raw = build_section_1(root, rust_files, py_files)
    section3_md, section3_raw = build_section_3(root, overrides)
    section5_md, section5_raw = build_section_5(root)

    lines = []
    lines.append("# Audit Snapshot (Generated)")
    lines.append("")
    lines.append(f"- Generated (UTC): {timestamp}")
    lines.append(f"- Commit: `{commit}`")
    lines.append(f"- Branch: `{branch}`")
    lines.append(f"- Script: `scripts/generate_audit_snapshot.py`")
    lines.append("")
    lines.append(section1_md)
    lines.append(section3_md)
    lines.append(section5_md)
    lines.append("")

    raw = {
        "generated_utc": timestamp,
        "commit": commit,
        "branch": branch,
        "section1": section1_raw,
        "section3": section3_raw,
        "section5": section5_raw,
    }
    return "\n".join(lines), raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate forge3d audit snapshot sections.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from script location).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/notes/audit_snapshot.md"),
        help="Markdown output path relative to repo root.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs/notes/audit_snapshot.json"),
        help="JSON output path relative to repo root.",
    )
    parser.add_argument(
        "--classification-overrides",
        type=Path,
        default=None,
        help="Optional JSON file: {'module.py': 'Category'} for Section 3 overrides.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    overrides_path = None
    if args.classification_overrides is not None:
        overrides_path = (
            args.classification_overrides
            if args.classification_overrides.is_absolute()
            else root / args.classification_overrides
        )
    overrides = load_overrides(overrides_path)

    markdown, raw = build_snapshot(root, overrides)

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    json_path = args.json_output if args.json_output.is_absolute() else root / args.json_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print(f"Wrote {rel(output_path, root)}")
    print(f"Wrote {rel(json_path, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
