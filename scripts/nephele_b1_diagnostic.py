"""Run or prepare the non-acceptance NEPHELE T4/B1 parity diagnostic.

This tool captures one no-medium candidate frame and compares it with the
committed 40-spp vacuum control or a separately registered diagnostic control.
Its separately invoked control generator can create the authorized 320-spp
vacuum control from a pre-registered plan. It never replaces an artifact and
keeps the B1 pass rule unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
import zipfile

import numpy as np


AUDIT_REFERENCE = Path("docs/audits/nephele-b1-2026-09-30/vacuum-reference-spp40.npy")
CONTROL_GENERATOR = Path("scripts/nephele_b1_diagnostic.py")
REFERENCE_WHEEL_DIRECTORY = Path("tests/nephele/acceptance-runtime/reference-wheel")
CONTROL_SAMPLES_PER_PIXEL = 320
PASS_FRACTION = 0.95
CONTROL_CRITERION = {"minimum": PASS_FRACTION, "delta_e_strict_upper_bound": 2.0}
VACUUM_COEFFICIENTS = {"sigma_a": [0.0] * 3, "sigma_s": [0.0] * 3}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _save(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _save_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _repo_relative(repo: Path, path: Path, label: str) -> Path:
    try:
        return path.resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be inside the Forge3D checkout") from error


def _artifact_record(repo: Path, path: Path) -> dict[str, str]:
    relative = _repo_relative(repo, path, "artifact").as_posix()
    return {"artifact": path.name, "path": relative, "sha256": _sha256(path)}


def _require_committed_file(repo: Path, relative: Path, label: str) -> Path:
    path = (repo / relative).resolve()
    normalized = relative.as_posix()
    _git(repo, "ls-files", "--error-unmatch", "--", normalized)
    if _git(repo, "hash-object", normalized) != _git(repo, "rev-parse", f"HEAD:{normalized}"):
        raise ValueError(f"{label} differs from its committed HEAD blob")
    return path


def _frozen_fixture(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = _require_committed_file(
        repo, Path("tests/nephele/fixture-manifest.json"), "fixture manifest"
    )
    provenance_path = _require_committed_file(
        repo,
        Path("tests/nephele/fixture/reference-provenance.json"),
        "reference provenance",
    )
    manifest = _json(manifest_path)
    provenance = _json(provenance_path)
    if manifest.get("status") != "APPROVED":
        raise RuntimeError("T4/B1 runs only with an APPROVED frozen fixture")
    if (
        provenance.get("schema") != "forge3d.nephele.reference_provenance/2"
        or provenance.get("acceptance_eligible") is not True
    ):
        raise RuntimeError("T4/B1 requires the approved acceptance reference provenance")
    for role, item in manifest["scene_inputs"].items():
        _manifest_path(repo, item, f"scene input {role}")
    _manifest_path(repo, manifest["files"]["terrain_mask"], "terrain mask")
    return manifest, provenance


def _reference_wheel(
    repo: Path, native_runtime: dict[str, Any]
) -> tuple[Path, dict[str, str]]:
    if set(native_runtime) != {
        "source_revision",
        "wheel_filename",
        "wheel_sha256",
        "wheel_native_member",
        "native_sha256",
    }:
        raise ValueError("reference provenance native runtime is incomplete")
    filename = native_runtime.get("wheel_filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("reference provenance has an invalid wheel filename")
    wheel_relative = REFERENCE_WHEEL_DIRECTORY / filename
    wheel = _require_committed_file(repo, wheel_relative, "reference wheel")
    if _sha256(wheel) != native_runtime.get("wheel_sha256"):
        raise ValueError("committed reference wheel differs from reference provenance")
    member = native_runtime.get("wheel_native_member")
    if not isinstance(member, str):
        raise ValueError("reference provenance has no native wheel member")
    with zipfile.ZipFile(wheel) as archive:
        native_members = [
            name
            for name in archive.namelist()
            if name.startswith("forge3d/_forge3d")
            and Path(name).suffix.lower() in {".so", ".dylib", ".pyd"}
        ]
        if native_members != [member]:
            raise ValueError("committed reference wheel native member is missing or ambiguous")
        member_bytes = archive.read(member)
    if (
        hashlib.sha256(member_bytes).hexdigest() != native_runtime.get("native_sha256")
        or str(native_runtime["source_revision"]).encode("ascii") not in member_bytes
    ):
        raise ValueError("committed reference wheel native member differs from provenance")
    return wheel, _artifact_record(repo, wheel)


def _planned_output_path(repo: Path, record: dict[str, Any], label: str) -> Path:
    if set(record) != {"artifact", "path"}:
        raise ValueError(f"{label}: planned output record has unexpected fields")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: invalid planned output path")
    path = (repo / relative).resolve()
    _repo_relative(repo, path, label)
    if path.name != record["artifact"]:
        raise ValueError(f"{label}: artifact name differs from planned output path")
    return path


def _control_plan(
    repo: Path, control_output: Path, manifest: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    wheel, wheel_record = _reference_wheel(repo, provenance["native_runtime"])
    del wheel
    generator = (repo / CONTROL_GENERATOR).resolve()
    reference = control_output / f"vacuum-reference-spp{CONTROL_SAMPLES_PER_PIXEL}.npy"
    registration = control_output / f"vacuum-control-spp{CONTROL_SAMPLES_PER_PIXEL}.json"
    return {
        "schema": "forge3d.nephele.vacuum_control_plan/1",
        "samples_per_pixel": CONTROL_SAMPLES_PER_PIXEL,
        "source_revision": provenance["source_revision"],
        "fixture_manifest_sha256": _sha256(repo / "tests/nephele/fixture-manifest.json"),
        "terrain_mask": manifest["files"]["terrain_mask"],
        "reference_provenance": _artifact_record(
            repo, repo / "tests/nephele/fixture/reference-provenance.json"
        ),
        "native_runtime": provenance["native_runtime"],
        "native_wheel": wheel_record,
        "generator": _artifact_record(repo, generator),
        "seed": provenance["seed"],
        "medium_coefficients": VACUUM_COEFFICIENTS,
        "criterion": CONTROL_CRITERION,
        "expected_diagnostics": {
            key: provenance["diagnostics"][key]
            for key in ("source_revision", "adapter", "backend", "driver")
        },
        "camera_contract": provenance["camera_contract"],
        "reference_path": _repo_relative(repo, reference, "control reference").as_posix(),
        "outputs": {
            "reference": {
                "artifact": reference.name,
                "path": _repo_relative(repo, reference, "control reference").as_posix(),
            },
            "registration": {
                "artifact": registration.name,
                "path": _repo_relative(repo, registration, "control registration").as_posix(),
            },
        },
    }


def _validate_control_plan(
    repo: Path, plan: dict[str, Any], plan_path: Path | None = None
) -> tuple[Path, Path, Path]:
    manifest, provenance = _frozen_fixture(repo)
    if plan.get("schema") != "forge3d.nephele.vacuum_control_plan/1":
        raise ValueError("unsupported diagnostic control plan")
    if set(plan) != {
        "schema",
        "samples_per_pixel",
        "source_revision",
        "fixture_manifest_sha256",
        "terrain_mask",
        "reference_provenance",
        "native_runtime",
        "native_wheel",
        "generator",
        "seed",
        "medium_coefficients",
        "criterion",
        "expected_diagnostics",
        "camera_contract",
        "reference_path",
        "outputs",
    }:
        raise ValueError("control plan fields differ from the registered schema")
    if plan.get("samples_per_pixel") != CONTROL_SAMPLES_PER_PIXEL:
        raise ValueError("control plan must specify the authorized 320 samples per pixel")
    if plan.get("source_revision") != provenance.get("source_revision"):
        raise ValueError("control plan does not use the approved reference source")
    if plan.get("fixture_manifest_sha256") != _sha256(
        repo / "tests/nephele/fixture-manifest.json"
    ):
        raise ValueError("control plan fixture manifest differs")
    if plan.get("terrain_mask") != manifest["files"]["terrain_mask"]:
        raise ValueError("control plan terrain mask differs from the frozen fixture")
    expected_provenance = _artifact_record(
        repo, repo / "tests/nephele/fixture/reference-provenance.json"
    )
    if plan.get("reference_provenance") != expected_provenance:
        raise ValueError("control plan reference provenance differs")
    if plan.get("native_runtime") != provenance.get("native_runtime"):
        raise ValueError("control plan does not use the committed reference wheel runtime")
    wheel, wheel_record = _reference_wheel(repo, provenance["native_runtime"])
    if plan.get("native_wheel") != wheel_record:
        raise ValueError("control plan reference wheel path or hash differs")
    generator = _artifact_record(repo, (repo / CONTROL_GENERATOR).resolve())
    if plan.get("generator") != generator:
        raise ValueError("control plan generator path or hash differs")
    if plan.get("seed") != provenance.get("seed"):
        raise ValueError("control plan seed differs from the approved reference")
    if plan.get("medium_coefficients") != VACUUM_COEFFICIENTS:
        raise ValueError("control plan must specify a vacuum")
    if plan.get("criterion") != CONTROL_CRITERION:
        raise ValueError("control plan must preserve the B1 pass rule")
    expected_diagnostics = {
        key: provenance["diagnostics"][key]
        for key in ("source_revision", "adapter", "backend", "driver")
    }
    if plan.get("expected_diagnostics") != expected_diagnostics:
        raise ValueError("control plan runtime identities differ from provenance")
    if plan.get("camera_contract") != provenance.get("camera_contract"):
        raise ValueError("control plan camera differs from the approved reference")
    outputs = plan.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"reference", "registration"}:
        raise ValueError("control plan outputs are incomplete")
    reference = _planned_output_path(repo, outputs["reference"], "control reference")
    registration = _planned_output_path(
        repo, outputs["registration"], "control registration"
    )
    if reference.parent != registration.parent or reference == registration:
        raise ValueError("control outputs must be distinct files in one directory")
    if plan.get("reference_path") != outputs["reference"]["path"]:
        raise ValueError("control reference path differs within the plan")
    if reference.name != f"vacuum-reference-spp{CONTROL_SAMPLES_PER_PIXEL}.npy":
        raise ValueError("control reference filename does not identify its sample count")
    if registration.name != f"vacuum-control-spp{CONTROL_SAMPLES_PER_PIXEL}.json":
        raise ValueError("control registration filename does not identify its sample count")
    if plan_path is not None and (
        plan_path.resolve() in {reference, registration}
        or plan_path.resolve().is_relative_to(reference.parent)
    ):
        raise ValueError("control plan must be serialized separately from generated outputs")
    return wheel, reference, registration


def _require_committed_reference(repo: Path, supplied: Path) -> Path:
    expected = (repo / AUDIT_REFERENCE).resolve()
    reference = supplied.resolve()
    if reference != expected:
        raise ValueError(f"vacuum reference must be the committed audit artifact: {expected}")
    relative = AUDIT_REFERENCE.as_posix()
    _git(repo, "ls-files", "--error-unmatch", "--", relative)
    working_hash = _git(repo, "hash-object", relative)
    committed_hash = _git(repo, "rev-parse", f"HEAD:{relative}")
    if working_hash != committed_hash:
        raise ValueError("vacuum reference differs from its committed HEAD blob")
    return reference


def _require_registered_reference(
    repo: Path, supplied: Path, registration_path: Path
) -> tuple[Path, dict[str, Any]]:
    """Bind a completed control to its separately saved, pre-render capture plan."""
    registration = _json(registration_path)
    if registration.get("schema") != "forge3d.nephele.vacuum_control/1":
        raise ValueError("unsupported diagnostic control registration")
    plan_path = _manifest_path(repo, registration["plan"], "control plan")
    if registration.get("published_plan_sha256") != registration["plan"]["sha256"]:
        raise ValueError("control plan differs from the published pre-render digest")
    plan = _json(plan_path)
    _, planned_reference, planned_registration = _validate_control_plan(
        repo, plan, plan_path
    )
    if planned_registration != registration_path.resolve():
        raise ValueError("control registration path differs from its pre-render plan")
    provenance = _json(repo / "tests/nephele/fixture/reference-provenance.json")
    count = plan.get("samples_per_pixel")
    if registration.get("samples_per_pixel") != count:
        raise ValueError("control sample count differs from its pre-render plan")
    if registration.get("seed") != plan["seed"]:
        raise ValueError("executed control seed differs from its pre-render plan")
    if registration.get("medium_coefficients") != plan["medium_coefficients"]:
        raise ValueError("executed control is not the planned vacuum")
    if registration.get("criterion") != plan["criterion"]:
        raise ValueError("executed control criterion differs from its pre-render plan")
    if registration.get("native_runtime") != plan["native_runtime"]:
        raise ValueError("executed control runtime differs from its pre-render plan")
    if registration.get("native_wheel") != plan["native_wheel"]:
        raise ValueError("executed control wheel differs from its pre-render plan")
    imported_native = registration.get("imported_native")
    if (
        not isinstance(imported_native, dict)
        or not Path(str(imported_native.get("path", ""))).is_absolute()
        or imported_native.get("sha256") != plan["native_runtime"]["native_sha256"]
    ):
        raise ValueError("executed control native path or hash differs from its plan")
    if registration.get("generator") != plan["generator"]:
        raise ValueError("executed control generator differs from its pre-render plan")
    for key in ("source_revision", "adapter", "backend", "driver"):
        if registration["diagnostics"].get(key) != provenance["diagnostics"][key]:
            raise ValueError(f"executed control {key} differs from the approved reference")
    crop = _json(repo / "tests/nephele/fixture/crop.json")
    if registration["diagnostics"].get("sample_count") != count * crop["width"] * crop["height"]:
        raise ValueError("executed control sample count differs from its pre-render plan")
    if registration.get("camera_contract") != provenance["camera_contract"]:
        raise ValueError("executed control camera differs from the approved reference")
    reference = _manifest_path(repo, registration["reference"], "control reference").resolve()
    if (
        reference != planned_reference
        or reference != supplied.resolve()
        or registration["reference"]["path"] != plan.get("reference_path")
        or registration.get("outputs") != {"reference": registration["reference"]}
    ):
        raise ValueError("control reference differs from its registration or pre-render plan")
    return reference, {
        **registration["reference"],
        "samples_per_pixel": count,
        "reused_committed_blob": False,
        "registration_sha256": _sha256(registration_path),
        "plan": registration["plan"],
    }


def _metrics(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    from tests._deltae import delta_e_2000, srgb_to_lab

    if candidate.shape != reference.shape or candidate.shape[:2] != mask.shape:
        raise ValueError("candidate, vacuum reference, and terrain mask shapes differ")
    values = delta_e_2000(srgb_to_lab(candidate), srgb_to_lab(reference))[mask]
    if not values.size:
        raise ValueError("terrain mask is empty")
    fraction = float(np.mean(values < 2.0))
    return {
        "pixels": int(values.size),
        "mean_delta_e": float(values.mean()),
        "p95_delta_e": float(np.quantile(values, 0.95)),
        "maximum_delta_e": float(values.max()),
        "fraction_delta_e_below_2": fraction,
        "pass": fraction >= PASS_FRACTION,
    }


def _manifest_path(repo: Path, record: dict[str, Any], label: str) -> Path:
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: invalid manifest path")
    path = repo / relative
    if path.name != record["artifact"] or _sha256(path) != record["sha256"]:
        raise ValueError(f"{label}: artifact differs from its recorded hash or name")
    return path


def _require_published_plan(plan_path: Path, published_sha256: str | None) -> dict[str, Any]:
    if published_sha256 is None or _sha256(plan_path) != published_sha256:
        raise ValueError("control plan differs from the digest published before generation")
    return _json(plan_path)


def _verify_imported_native(native: Any, runtime: dict[str, Any], extracted: Path) -> dict[str, str]:
    path = Path(native.__file__).resolve()
    if not path.is_relative_to(extracted.resolve()) or _sha256(path) != runtime["native_sha256"]:
        raise RuntimeError("imported reference native differs from the committed wheel")
    return {"path": str(path), "sha256": _sha256(path)}


def _generate_control(repo: Path, plan_path: Path, published_sha256: str | None) -> int:
    # This check precedes import and GPU work. The caller must publish the plan
    # or digest separately; a command-line digest alone is not publication.
    plan = _require_published_plan(plan_path, published_sha256)
    wheel, reference, registration = _validate_control_plan(repo, plan, plan_path)
    if reference.parent.exists():
        raise FileExistsError(f"refusing to replace control output: {reference.parent}")
    if any(name == "forge3d" or name.startswith("forge3d.") for name in sys.modules):
        raise RuntimeError("control generation requires a fresh process without forge3d imports")
    os.environ.update(FORGE3D_NO_BOOTSTRAP="1", FORGE3D_TEST_INSTALLED_WHEEL="1",
                      WGPU_BACKEND="vulkan", WGPU_BACKENDS="vulkan")
    fixture = repo / "tests/nephele/fixture"
    read = lambda name: _json(fixture / name)
    camera, terrain_data, medium = read("camera.json"), read("terrain.json"), read("medium.json")
    material, sun = read("material.json"), read("sun.json")
    atmosphere, exposure, crop = read("atmosphere.json"), read("exposure.json"), read("crop.json")
    terrain = np.ascontiguousarray(np.load(fixture / terrain_data["dem"], allow_pickle=False), dtype=np.float32)
    shape = medium["domain"]["grid_shape"]
    density = np.asarray(medium["density_r16"], dtype=np.float32).reshape(shape[2], shape[1], shape[0]) / 65535.0
    with tempfile.TemporaryDirectory(prefix="nephele-reference-wheel-", ignore_cleanup_errors=True) as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(extracted)
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(extracted))
        from forge3d.media import Medium, _native_module
        native = _native_module()
        imported_native = _verify_imported_native(native, plan["native_runtime"], extracted)
        from scripts.generate_media_fixture import _aces_srgb8
        from scripts.run_media_physical_capture import _camera_contract
        vacuum = Medium.grid3d([0.0] * 3, [0.0] * 3, density,
            (medium["domain"]["bounds_min"], medium["domain"]["bounds_max"]),
            phase="henyey_greenstein", g=medium["phase"]["g"],
            density_scale=medium["density_scale"], version=1)
        native_camera = {**camera, "terrain_camera": {**camera["terrain_camera"], "target": tuple(camera["terrain_camera"]["target"])}}
        started = time.monotonic()
        result = native._render_volumetric_reference(vacuum._native, terrain, crop["width"], crop["height"], native_camera,
            spacing=tuple(terrain_data["spacing"]), exaggeration=terrain_data["exaggeration"],
            albedo=tuple(material["albedo"]), sun_azimuth_deg=sun["azimuth_deg"],
            sun_elevation_deg=sun["elevation_deg"], sun_intensity=sun["intensity"], sun_color=tuple(sun["color"]),
            environment_intensity=atmosphere["environment_intensity"], exposure=exposure["value"],
            samples_per_pixel=plan["samples_per_pixel"], homogeneous_medium_reach=120.0, seed=plan["seed"],
            full_viewport=tuple(crop["full_viewport"]), crop=(crop["x"], crop["y"], crop["width"], crop["height"]))
        elapsed = time.monotonic() - started
        diagnostics = dict(result["diagnostics"])
        for key, expected in plan["expected_diagnostics"].items():
            if diagnostics.get(key) != expected:
                raise RuntimeError(f"rendered control {key} differs from the approved reference")
        if diagnostics.get("sample_count") != plan["samples_per_pixel"] * crop["width"] * crop["height"]:
            raise RuntimeError("rendered control sample count differs from its plan")
        actual_camera = _camera_contract(result["camera_contract"])
        if actual_camera != plan["camera_contract"]:
            raise RuntimeError("rendered control camera differs from the approved reference")
        rgb = _aces_srgb8(np.asarray(result["beauty"], dtype=np.float32))
        if rgb.shape != (crop["height"], crop["width"], 3):
            raise RuntimeError("rendered control dimensions differ from the fixture")
        _require_published_plan(plan_path, published_sha256)
        _validate_control_plan(repo, plan, plan_path)
        _verify_imported_native(native, plan["native_runtime"], extracted)
        reference.parent.mkdir(parents=True, exist_ok=False)
        with reference.open("xb") as stream:
            np.save(stream, rgb, allow_pickle=False)
        reference_record = _artifact_record(repo, reference)
        record = {
            "schema": "forge3d.nephele.vacuum_control/1", "physical_acceptance": False,
            "plan": _artifact_record(repo, plan_path), "published_plan_sha256": published_sha256,
            "reference": reference_record, "outputs": {"reference": reference_record},
            "samples_per_pixel": diagnostics["sample_count"] // (crop["width"] * crop["height"]),
            "diagnostics": diagnostics, "camera_contract": actual_camera,
            "imported_native": imported_native, "native_runtime": plan["native_runtime"],
            "native_wheel": plan["native_wheel"], "generator": plan["generator"],
            "seed": plan["seed"], "medium_coefficients": plan["medium_coefficients"],
            "criterion": plan["criterion"], "render_wall_seconds": elapsed,
        }
        _save_new(registration, record)
        _require_registered_reference(repo, reference, registration)
        print(json.dumps({"registration": _artifact_record(repo, registration), "render_wall_seconds": elapsed}, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Forge3D checkout to capture")
    parser.add_argument("--output", type=Path, help="new diagnostic or control output directory")
    parser.add_argument("--prepare-control-plan", type=Path, help="save the 320-spp plan before rendering")
    parser.add_argument("--generate-control", type=Path, help="render the separately saved 320-spp plan")
    parser.add_argument("--plan-sha256", help="plan digest published before invoking generation")
    parser.add_argument(
        "--vacuum-reference",
        type=Path,
        help="committed 40-spp control or the control named in --control-registration",
    )
    parser.add_argument("--control-registration", type=Path, help="completed control record bound to a pre-render plan")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    repo = arguments.repo.resolve()
    if arguments.prepare_control_plan is not None:
        if arguments.generate_control is not None or arguments.output is None:
            raise ValueError("plan preparation requires --output and excludes generation")
        plan_path = arguments.prepare_control_plan.resolve()
        _repo_relative(repo, plan_path, "control plan")
        manifest, provenance = _frozen_fixture(repo)
        plan = _control_plan(repo, arguments.output.resolve(), manifest, provenance)
        _validate_control_plan(repo, plan, plan_path)
        if arguments.output.exists():
            raise FileExistsError("control output already exists")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        _save_new(plan_path, plan)
        print(json.dumps(_artifact_record(repo, plan_path), sort_keys=True))
        return 0
    if arguments.generate_control is not None:
        return _generate_control(repo, arguments.generate_control.resolve(), arguments.plan_sha256)
    if arguments.output is None or arguments.vacuum_reference is None:
        raise ValueError("diagnostic requires --output and --vacuum-reference")
    output = arguments.output.resolve()
    fixture = repo / "tests/nephele/fixture"
    if not (repo / ".git").exists():
        raise ValueError(f"not a Forge3D checkout: {repo}")
    if arguments.control_registration is None:
        vacuum_path = _require_committed_reference(repo, arguments.vacuum_reference)
        reference_record = {
            "path": AUDIT_REFERENCE.as_posix(),
            "sha256": _sha256(vacuum_path),
            "samples_per_pixel": 40,
            "reused_committed_blob": True,
        }
    else:
        vacuum_path, reference_record = _require_registered_reference(
            repo, arguments.vacuum_reference, arguments.control_registration.resolve()
        )
    if output.exists():
        raise FileExistsError(f"refusing to replace diagnostic output: {output}")

    manifest_path = repo / "tests/nephele/fixture-manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "APPROVED":
        raise RuntimeError("T4/B1 runs only with an APPROVED frozen fixture")
    for role, item in manifest["scene_inputs"].items():
        _manifest_path(repo, item, f"scene input {role}")
    terrain_mask_path = _manifest_path(
        repo, manifest["files"]["terrain_mask"], "terrain mask"
    )

    os.environ.update(
        FORGE3D_NO_BOOTSTRAP="1",
        FORGE3D_TEST_INSTALLED_WHEEL="1",
        WGPU_BACKEND="vulkan",
        WGPU_BACKENDS="vulkan",
    )
    sys.path.insert(0, str(repo))
    import forge3d as f3d
    from forge3d.media import Medium
    from forge3d.terrain_params import (
        AovSettings,
        TonemapSettings,
        make_terrain_params_config,
    )
    from scripts.run_media_physical_capture import _camera_contract, _crop, _write_hdr

    def read_fixture(name: str) -> dict[str, Any]:
        return _json(fixture / name)

    camera_input = read_fixture("camera.json")
    terrain_data = read_fixture("terrain.json")
    medium_data = read_fixture("medium.json")
    material_input = read_fixture("material.json")
    sun = read_fixture("sun.json")
    atmosphere = read_fixture("atmosphere.json")
    exposure = read_fixture("exposure.json")
    crop = read_fixture("crop.json")
    provenance = read_fixture("reference-provenance.json")
    camera = camera_input["terrain_camera"]
    terrain = np.load(fixture / terrain_data["dem"], allow_pickle=False)
    shape = medium_data["domain"]["grid_shape"]
    density = np.asarray(medium_data["density_r16"], dtype=np.float32).reshape(
        shape[2], shape[1], shape[0]
    ) / 65535.0
    bounds = (
        medium_data["domain"]["bounds_min"],
        medium_data["domain"]["bounds_max"],
    )
    medium = Medium.grid3d(
        medium_data["sigma_a"],
        medium_data["sigma_s"],
        density,
        bounds,
        phase="henyey_greenstein",
        g=medium_data["phase"]["g"],
        density_scale=medium_data["density_scale"],
        version=1,
    )
    spans = [
        terrain_data["spacing"][index] * (terrain_data["dimensions"][index] - 1)
        for index in range(2)
    ]
    if spans[0] != spans[1]:
        raise ValueError("expected the tracked square terrain span")

    # This explicit physical Lambert model is the only intended B1 variable.
    config = make_terrain_params_config(
        size_px=tuple(crop["full_viewport"]),
        render_scale=1.0,
        terrain_span=spans[0],
        msaa_samples=1,
        z_scale=float(terrain_data["exaggeration"]),
        exposure=float(exposure["value"]),
        domain=(float(terrain.min()), float(terrain.max())),
        light_azimuth_deg=sun["azimuth_deg"],
        light_elevation_deg=sun["elevation_deg"],
        sun_intensity=sun["intensity"],
        sun_color=sun["color"],
        albedo_mode=material_input["albedo_mode"],
        colormap_strength=float(material_input["colormap_strength"]),
        cam_radius=camera["radius"],
        cam_phi_deg=camera["phi_deg"],
        cam_theta_deg=camera["theta_deg"],
        cam_target=camera["target"],
        fov_y_deg=float(camera_input["fov_y"]),
        camera_mode=camera["mode"],
        aa_samples=1,
        aa_seed=0x4E455048,
        tonemap=TonemapSettings(operator="aces"),
        aov=AovSettings(
            enabled=True,
            transmittance=True,
            in_scatter=True,
            cloud_shadow=True,
            optical_depth=True,
        ),
        media=medium,
        terrain_shading_model="lambert_physical",
    )
    params = f3d.TerrainRenderParams(config)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.custom(
        tuple(material_input["albedo"]),
        float(material_input["metallic"]),
        float(material_input["roughness"]),
        triplanar_scale=float(material_input["triplanar_scale"]),
        normal_strength=float(material_input["normal_strength"]),
        blend_sharpness=float(material_input["blend_sharpness"]),
    )
    with tempfile.TemporaryDirectory(prefix="nephele-b1-ibl-") as temporary:
        hdr = Path(temporary) / "fixture.hdr"
        _write_hdr(hdr)
        ibl = f3d.IBL.from_hdr(
            str(hdr), intensity=float(atmosphere["environment_intensity"])
        )
        capture = renderer._capture_nephele_acceptance(
            material,
            ibl,
            params,
            terrain,
            terrain_occlusion_in_media=True,
            include_no_medium=True,
        )

    diagnostics = dict(capture["diagnostics"])
    if diagnostics.get("terrain_shading_model") != "lambert_physical":
        raise RuntimeError("B1 capture did not execute the physical terrain shading model")
    reference_diagnostics = provenance["diagnostics"]
    for key in ("adapter", "backend", "driver"):
        if diagnostics.get(key) != reference_diagnostics.get(key):
            raise RuntimeError(f"candidate/reference {key} identities differ")
    actual_camera = _camera_contract(capture["camera_contract"])
    expected_camera = _camera_contract(provenance["camera_contract"])
    if any(
        not np.array_equal(
            np.asarray(actual_camera[key], np.float32),
            np.asarray(expected_camera[key], np.float32),
        )
        for key in expected_camera
    ):
        raise RuntimeError("candidate/reference camera contracts differ")

    candidate = _crop(np.asarray(capture["no_medium_beauty"], dtype=np.uint8), crop)
    vacuum = np.load(vacuum_path, allow_pickle=False)
    terrain_mask = np.load(terrain_mask_path, allow_pickle=False)
    comparison = _metrics(candidate, vacuum, terrain_mask)
    dirty_lines = _git(repo, "status", "--porcelain=v1", "--untracked-files=no").splitlines()
    head = _git(repo, "rev-parse", "HEAD")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "medium-disabled-rgb.npy", candidate, allow_pickle=False)
    # Retain the same executed primary frame and AOVs for the subsequent
    # B3-B9 diagnostics; this avoids a second capture with different inputs.
    for key, filename in (
        ("beauty", "realtime-rgb.npy"),
        ("in_scatter", "in-scatter.npy"),
        ("cloud_shadow", "cloud-shadow-aov.npy"),
        ("transmittance", "transmittance.npy"),
        ("optical_depth", "optical-depth.npy"),
    ):
        np.save(output / filename, _crop(np.asarray(capture[key]), crop), allow_pickle=False)
    _save(output / "candidate-diagnostics.json", diagnostics)
    _save(output / "camera-contract.json", actual_camera)
    report = {
        "schema": "forge3d.nephele.b1_diagnostic/2",
        "physical_acceptance": False,
        "repository": {
            "path": str(repo),
            "head": head,
            "tracked_worktree_clean": not dirty_lines,
            "tracked_dirty_entries": dirty_lines,
        },
        "candidate_native": {
            "source_revision": diagnostics.get("source_revision"),
            "source_matches_repository_head": (
                not dirty_lines and diagnostics.get("source_revision") == head
            ),
            "adapter": diagnostics.get("adapter"),
            "backend": diagnostics.get("backend"),
            "driver": diagnostics.get("driver"),
        },
        "diagnostic_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__)),
        },
        "fixture_manifest_sha256": _sha256(manifest_path),
        "scene_inputs": manifest["scene_inputs"],
        "terrain_mask": manifest["files"]["terrain_mask"],
        "medium_control": (
            "candidate no-medium branch versus registered sigma_a=sigma_s=0 "
            "vacuum reference"
        ),
        "terrain_shading_model": "lambert_physical",
        "vacuum_reference": reference_record,
        "terrain_comparison": comparison,
        "criterion": {
            "metric": "fraction_delta_e_below_2",
            "minimum": PASS_FRACTION,
            "pass": comparison["pass"],
        },
        "gate4_comparison1": "PRECHECK_PASS" if comparison["pass"] else "NOT_PROVEN",
    }
    _save(output / "b1-report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if comparison["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
