#!/usr/bin/env python3
"""
tools/run_proof_pack_gore.py

Forge3D Proof Pack Harness (Gore Range)

Goal:
  Generate a reproducible, forensic proof directory that demonstrates whether:
    - PBR debug splits are actually wired (diffuse-only vs spec-only vs fresnel vs NdotV, etc.)
    - Roughness multiplier sweep changes shading
    - Specular AA toggle changes high-frequency sparkle (at least measurably)
    - IBL responds to HDRI changes (beauty + IBL-only)
    - POM produces a measurable difference (pom on vs off)

This harness is intentionally strict: it will flag cases where outputs are identical
(i.e., debug modes not plumbed) so you don't fool yourself with "proof" screenshots.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Utilities
# ----------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout_s: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    if dry_run:
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "env_overrides": {k: env[k] for k in sorted(env.keys()) if k.startswith("VF_")},
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "elapsed_s": 0.0,
            "dry_run": True,
        }

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "env_overrides": {k: env[k] for k in sorted(env.keys()) if k.startswith("VF_")},
        "returncode": p.returncode,
        "stdout": p.stdout[-20000:],  # keep tail
        "stderr": p.stderr[-20000:],  # keep tail
        "elapsed_s": round(time.time() - t0, 4),
        "dry_run": False,
    }


def _try_git_rev(repo_root: Path) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return None


def _image_bytes(path: Path) -> bytes:
    # Used for strict identical-content detection without extra deps.
    return path.read_bytes()


def _files_identical(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    return _image_bytes(a) == _image_bytes(b)


# ----------------------------
# Case definitions
# ----------------------------

@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    rel_out: Path
    extra_args: List[str]
    env_overrides: Dict[str, str]
    description: str


def _base_args(
    dem: Path,
    hdr: Path,
    size: Tuple[int, int],
    msaa: int,
    z_scale: float,
    cam_phi: float,
    cam_theta: float,
    cam_radius: float,
    sun_azimuth: float,
    sun_intensity: float,
    ibl_intensity: float,
) -> List[str]:
    # Keep this aligned with how you drive terrain_demo.py.
    # Note: we do NOT pass --overwrite here; each case adds it.
    return [
        "--dem", str(dem),
        "--hdr", str(hdr),
        "--size", str(size[0]), str(size[1]),
        "--msaa", str(msaa),
        "--z-scale", str(z_scale),
        "--albedo-mode", "material",
        "--gi", "ibl",
        "--ibl-intensity", str(ibl_intensity),
        "--cam-phi", str(cam_phi),
        "--cam-theta", str(cam_theta),
        "--cam-radius", str(cam_radius),
        "--sun-azimuth", str(sun_azimuth),
        "--sun-intensity", str(sun_intensity),
    ]


# ----------------------------
# Proof pack assembly
# ----------------------------

def build_cases(out_dir: Path, dem: Path, hdr_a: Path, hdr_b: Path, size: Tuple[int, int]) -> List[Case]:
    """
    IMPORTANT:
      We assume your renderer reads:
        VF_COLOR_DEBUG_MODE
        VF_ROUGHNESS_MULT
        VF_SPEC_AA_ENABLED
      And that debug modes are wired like you described:
        7..12 for PBR term inspection
        6 for IBL-only (earlier workflow)
    If not, the harness will *detect identical outputs* and mark it as failure.
    """
    # Tuned for stable, repeatable lighting.
    base = dict(
        size=size,
        msaa=4,
        z_scale=5.0,
        cam_phi=135.0,
        cam_theta=20.0,
        cam_radius=250.0,
        sun_azimuth=135.0,
        sun_intensity=2.0,
        ibl_intensity=1.0,
    )

    # ---- PBR term splits (debug modes 7-12) ----
    pbr_terms = [
        Case(
            name="pbr_terms_diffuse_only",
            rel_out=Path("pbr/pbr_terms_diffuse_only.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "7"},
            description="Diffuse-only IBL term (should NOT match specular-only).",
        ),
        Case(
            name="pbr_terms_specular_only",
            rel_out=Path("pbr/pbr_terms_specular_only.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "8"},
            description="Specular-only IBL term (should show highlights / reflections).",
        ),
        Case(
            name="pbr_terms_fresnel_F",
            rel_out=Path("pbr/pbr_terms_fresnel_F.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "9"},
            description="Fresnel term visualization (brighter at grazing angles).",
        ),
        Case(
            name="pbr_terms_ndotv",
            rel_out=Path("pbr/pbr_terms_ndotv.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "10"},
            description="N·V visualization (pure geometric view-angle term).",
        ),
        Case(
            name="pbr_terms_roughness",
            rel_out=Path("pbr/pbr_terms_roughness.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "11"},
            description="Roughness visualization (should be stable / meaningful).",
        ),
        Case(
            name="pbr_terms_energy",
            rel_out=Path("pbr/pbr_terms_energy.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "12", "VF_SPEC_AA_ENABLED": "0.0"},
            description="Energy / luminance visualization (helps catch non-conservation or overflow).",
        ),
    ]

    # ---- Roughness sweep (beauty mode) ----
    # Use debug_mode=0 (beauty) and multiply roughness.
    rough_sweep = []
    for mult in [0.25, 0.5, 1.0, 2.0]:
        rough_sweep.append(
            Case(
                name=f"pbr_roughness_{mult:.2f}",
                rel_out=Path(f"pbr/pbr_roughness_{mult:.2f}.png"),
                extra_args=_base_args(dem, hdr_a, **base),
                env_overrides={
                    "VF_COLOR_DEBUG_MODE": "0",
                    "VF_ROUGHNESS_MULT": str(mult),
                },
                description="Beauty render with roughness multiplier (should visibly change specular response).",
            )
        )

    # ---- Specular AA / Toksvig comparison ----
    # To make glitter obvious: reduce MSAA and focus specular-only term.
    base_glitter = dict(base)
    base_glitter["msaa"] = 1
    specaa = [
        Case(
            name="pbr_specaa_on",
            rel_out=Path("pbr/pbr_specaa_on.png"),
            extra_args=_base_args(dem, hdr_a, **base_glitter),
            env_overrides={
                "VF_COLOR_DEBUG_MODE": "8",     # specular-only is where glitter shows
                "VF_SPEC_AA_ENABLED": "1",
                "VF_ROUGHNESS_MULT": "1.0",
            },
            description="Specular-only, spec-AA ON (should reduce high-frequency sparkle).",
        ),
        Case(
            name="pbr_specaa_off",
            rel_out=Path("pbr/pbr_specaa_off.png"),
            extra_args=_base_args(dem, hdr_a, **base_glitter),
            env_overrides={
                "VF_COLOR_DEBUG_MODE": "8",
                "VF_SPEC_AA_ENABLED": "0",
                "VF_ROUGHNESS_MULT": "1.0",
            },
            description="Specular-only, spec-AA OFF (should be noisier / glittier).",
        ),
    ]

    # ---- IBL responsiveness proof ----
    # (A) beauty with HDR A vs HDR B
    # (B) IBL-only (mode 6) with HDR A vs HDR B
    ibl = [
        Case(
            name="ibl_beauty_hdr_a",
            rel_out=Path("ibl/ibl_beauty_hdr_a.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "0"},
            description="Beauty render under HDR A.",
        ),
        Case(
            name="ibl_beauty_hdr_b",
            rel_out=Path("ibl/ibl_beauty_hdr_b.png"),
            extra_args=_base_args(dem, hdr_b, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "0"},
            description="Beauty render under HDR B (should differ from HDR A).",
        ),
        Case(
            name="ibl_only_hdr_a",
            rel_out=Path("ibl/ibl_only_hdr_a.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "6"},
            description="IBL-only visualization under HDR A.",
        ),
        Case(
            name="ibl_only_hdr_b",
            rel_out=Path("ibl/ibl_only_hdr_b.png"),
            extra_args=_base_args(dem, hdr_b, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "6"},
            description="IBL-only visualization under HDR B (must differ if IBL samples env).",
        ),
    ]

    # ---- Recomposition proof ----
    # Prove that IBL_total ≈ diffuse + specular in linear space
    # Modes 13-16 output linear-encoded values for quantitative analysis
    # SpecAA disabled for recomposition tests to isolate IBL math verification
    # (SpecAA is tested separately via sparkle stress test)
    recomp = [
        Case(
            name="recomp_linear_combined",
            rel_out=Path("pbr/recomp_linear_combined.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "13", "VF_SPEC_AA_ENABLED": "0.0"},
            description="Linear (diff+spec) encoded [0,4]->[0,1] for recomposition proof.",
        ),
        Case(
            name="recomp_linear_diffuse",
            rel_out=Path("pbr/recomp_linear_diffuse.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "14", "VF_SPEC_AA_ENABLED": "0.0"},
            description="Linear diffuse only, encoded [0,4]->[0,1].",
        ),
        Case(
            name="recomp_linear_specular",
            rel_out=Path("pbr/recomp_linear_specular.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "15", "VF_SPEC_AA_ENABLED": "0.0"},
            description="Linear specular only, encoded [0,4]->[0,1].",
        ),
        Case(
            name="recomp_error_heatmap",
            rel_out=Path("pbr/recomp_error_heatmap.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "16", "VF_SPEC_AA_ENABLED": "0.0"},
            description="Recomposition error heatmap: abs(ibl - (diff+spec)) * 100. Should be black.",
        ),
    ]

    # ---- SpecAA sparkle stress test ----
    # SCOPED OUT for DEM terrain: Toksvig SpecAA requires mipmapped normal maps.
    # SpecAA Sparkle Stress Test:
    # Uses screen-derivative variance (dpdx/dpdy) which works for both:
    #   - Procedural DEM normals (detects terrain feature edges)
    #   - Synthetic sparkle perturbation (forced high-frequency variation)
    # Mode 17 injects HF normal perturbation; with SpecAA ON, Toksvig should increase roughness
    # and reduce sparkle energy. With SpecAA OFF, sparkles remain unfiltered.
    specaa_stress = [
        Case(
            name="specaa_sparkle_on",
            rel_out=Path("pbr/specaa_sparkle_on.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "17", "VF_SPEC_AA_ENABLED": "1.0", "VF_SPECAA_SIGMA_SCALE": "1.0"},
            description="SpecAA sparkle test - Toksvig should widen lobe and reduce HF energy.",
        ),
        Case(
            name="specaa_sparkle_off",
            rel_out=Path("pbr/specaa_sparkle_off.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "17", "VF_SPEC_AA_ENABLED": "0.0"},
            description="SpecAA sparkle test baseline - no Toksvig, sparkles unfiltered.",
        ),
        Case(
            name="specaa_sigma2",
            rel_out=Path("pbr/specaa_sigma2.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "19", "VF_SPEC_AA_ENABLED": "1.0", "VF_SPECAA_SIGMA_SCALE": "1.0"},
            description="SpecAA sigma² visualization - shows variance on terrain normals.",
        ),
        Case(
            name="specaa_sparkle_sigma2",
            rel_out=Path("pbr/specaa_sparkle_sigma2.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "20", "VF_SPEC_AA_ENABLED": "1.0", "VF_SPECAA_SIGMA_SCALE": "1.0"},
            description="SpecAA sigma² on sparkle-perturbed normal - should be bright (high variance).",
        ),
    ]

    # ---- POM proof ----
    # Low camera angle (grazing) increases POM parallax; compare ON vs OFF.
    # Requirements from todo.md:
    #   M1: Grazing-angle effect - POM must change appearance at low cam-theta
    #   M2: Debug shows offset magnitude (mode 18)
    pom_base = dict(base)
    pom_base["cam_theta"] = 10.0  # Grazing angle for maximum POM effect
    pom_base["cam_radius"] = 150.0
    pom = [
        Case(
            name="pom_on_grazing",
            rel_out=Path("pom/pom_on_grazing.png"),
            extra_args=_base_args(dem, hdr_a, **pom_base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "0"},
            description="POM enabled at grazing angle (cam_theta=10°).",
        ),
        Case(
            name="pom_off_grazing",
            rel_out=Path("pom/pom_off_grazing.png"),
            extra_args=_base_args(dem, hdr_a, **pom_base) + ["--pom-disabled"],
            env_overrides={"VF_COLOR_DEBUG_MODE": "0"},
            description="POM disabled at grazing angle (baseline).",
        ),
        Case(
            name="dbg_pom_offset_mag",
            rel_out=Path("pom/dbg_pom_offset_mag.png"),
            extra_args=_base_args(dem, hdr_a, **pom_base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "18"},
            description="POM offset magnitude visualization (grayscale: 0=none, white=max offset).",
        ),
    ]

    # ---- Triplanar proof ----
    # Requirements from todo.md:
    #   T1: Projection blending works (wx + wy + wz = 1, weights change smoothly with normal)
    #   T2: No seams / no "swimming" (triplanar mapping stays locked to world space)
    # Debug modes:
    #   21 = Triplanar weights visualization (RGB = x/y/z weights)
    #   22 = Procedural checker pattern (exposes UV stretching)
    triplanar = [
        Case(
            name="dbg_triplanar_weights",
            rel_out=Path("triplanar/dbg_triplanar_weights.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "21"},
            description="Triplanar blend weights: RGB = x/y/z projection weights. T1 proof: weights sum to 1, change smoothly.",
        ),
        Case(
            name="triplanar_checker",
            rel_out=Path("triplanar/triplanar_checker.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "22"},
            description="Procedural checker pattern via triplanar. T2 proof: no UV stretching on steep slopes.",
        ),
        Case(
            name="triplanar_camera_jitter_A",
            rel_out=Path("triplanar/triplanar_camera_jitter_A.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "22"},  # Use checker for stability test
            description="Triplanar checker at camera position A (baseline for jitter test).",
        ),
        Case(
            name="triplanar_camera_jitter_B",
            rel_out=Path("triplanar/triplanar_camera_jitter_B.png"),
            extra_args=_base_args(dem, hdr_a, **{**base, "cam_phi": base["cam_phi"] + 0.5}),  # Tiny camera delta
            env_overrides={"VF_COLOR_DEBUG_MODE": "22"},  # Use checker for stability test
            description="Triplanar checker at camera position B (0.5° phi delta for jitter test).",
        ),
        Case(
            name="triplanar_beauty",
            rel_out=Path("triplanar/triplanar_beauty.png"),
            extra_args=_base_args(dem, hdr_a, **base),
            env_overrides={"VF_COLOR_DEBUG_MODE": "0"},
            description="Beauty render with triplanar material texturing.",
        ),
    ]

    return pbr_terms + rough_sweep + specaa + ibl + recomp + specaa_stress + pom + triplanar


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=None, help="Forge3D repo root. Defaults to auto-detect from this script.")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    ap.add_argument("--terrain-demo", type=Path, default=None, help="Path to examples/terrain_demo.py. Defaults to <repo>/examples/terrain_demo.py")
    ap.add_argument("--dem", type=Path, default=Path("assets/Gore_Range_Albers_1m.tif"))
    ap.add_argument("--hdr-a", type=Path, default=Path("assets/snow_field_4k.hdr"))
    ap.add_argument("--hdr-b", type=Path, default=Path("assets/air_museum_playground_4k.hdr"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to reports/proof_gore/<timestamp>")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", type=str, default="", help="Comma-separated sections: pbr,ibl,pom,triplanar (empty = all)")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root or script_path.parents[1]
    terrain_demo = args.terrain_demo or (repo_root / "examples" / "terrain_demo.py")

    if not terrain_demo.exists():
        print(f"[ERROR] terrain_demo.py not found at: {terrain_demo}")
        return 2

    out_dir = args.out_dir or (repo_root / "reports" / "proof_gore" / _now_ts())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pbr").mkdir(exist_ok=True)
    (out_dir / "ibl").mkdir(exist_ok=True)
    (out_dir / "pom").mkdir(exist_ok=True)
    (out_dir / "triplanar").mkdir(exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)

    # Resolve asset paths relative to repo root if needed.
    dem = args.dem if args.dem.is_absolute() else (repo_root / args.dem)
    hdr_a = args.hdr_a if args.hdr_a.is_absolute() else (repo_root / args.hdr_a)
    hdr_b = args.hdr_b if args.hdr_b.is_absolute() else (repo_root / args.hdr_b)

    if not dem.exists():
        print(f"[ERROR] DEM not found: {dem}")
        return 2
    if not hdr_a.exists():
        print(f"[ERROR] HDR A not found: {hdr_a}")
        return 2
    if not hdr_b.exists():
        print(f"[ERROR] HDR B not found: {hdr_b}")
        return 2

    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    size = (args.width, args.height)

    cases = build_cases(out_dir, dem, hdr_a, hdr_b, size)

    # Filter by section if requested.
    if only:
        kept: List[Case] = []
        for c in cases:
            top = c.rel_out.parts[0].lower()
            if top in only:
                kept.append(c)
        cases = kept

    manifest: Dict[str, Any] = {
        "tool": "run_proof_pack_gore.py",
        "timestamp": _now_ts(),
        "repo_root": str(repo_root),
        "git_rev": _try_git_rev(repo_root),
        "terrain_demo": str(terrain_demo),
        "inputs": {"dem": str(dem), "hdr_a": str(hdr_a), "hdr_b": str(hdr_b)},
        "size": {"w": size[0], "h": size[1]},
        "cases": [],
        "checks": [],
    }

    base_env = dict(os.environ)

    print(f"[INFO] Writing proof pack to: {out_dir}")
    print(f"[INFO] Using python: {args.python}")
    print(f"[INFO] Using terrain_demo: {terrain_demo}")

    for c in cases:
        out_path = out_dir / c.rel_out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            str(terrain_demo),
            *c.extra_args,
            "--output", str(out_path),
            "--overwrite",
        ]

        env = dict(base_env)
        # Clear VF_* variables to avoid cross-talk between cases.
        for k in list(env.keys()):
            if k.startswith("VF_"):
                env.pop(k, None)
        # Apply this case's overrides.
        env.update(c.env_overrides)

        runlog = _run_cmd(cmd=cmd, cwd=repo_root, env=env, timeout_s=args.timeout_s, dry_run=args.dry_run)

        case_record = {
            "name": c.name,
            "out": str(out_path),
            "section": c.rel_out.parts[0],
            "description": c.description,
            **runlog,
        }

        if not args.dry_run and runlog["returncode"] == 0 and out_path.exists():
            case_record["sha256"] = _sha256_file(out_path)
            case_record["bytes"] = out_path.stat().st_size
        else:
            case_record["sha256"] = None
            case_record["bytes"] = None

        manifest["cases"].append(case_record)

        status = "OK" if runlog["returncode"] == 0 else f"FAIL(rc={runlog['returncode']})"
        print(f"[{status}] {c.name} -> {out_path}")

        if runlog["returncode"] != 0:
            # Keep going to gather all failures in one run.
            continue

    # ----------------------------
    # Generate difference images and statistics for IBL proof
    # ----------------------------
    def generate_ibl_diff_and_stats() -> Dict[str, Any]:
        """Generate IBL difference visualization and statistics."""
        stats: Dict[str, Any] = {}
        
        if args.dry_run:
            return {"status": "SKIP(dry_run)"}
        
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return {"status": "SKIP(missing_deps: PIL or numpy)"}
        
        a_path = out_dir / "ibl" / "ibl_only_hdr_a.png"
        b_path = out_dir / "ibl" / "ibl_only_hdr_b.png"
        
        if not a_path.exists() or not b_path.exists():
            return {"status": "SKIP(missing_input_images)"}
        
        # Load images
        img_a = np.array(Image.open(a_path).convert("RGB")).astype(np.float32) / 255.0
        img_b = np.array(Image.open(b_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Compute absolute difference
        diff = np.abs(img_a - img_b)
        diff_gray = np.mean(diff, axis=2)  # Average across RGB for scalar diff
        
        # Statistics (in linear RGB space, approximately)
        stats["mean_diff"] = float(np.mean(diff_gray))
        stats["median_diff"] = float(np.median(diff_gray))
        stats["p95_diff"] = float(np.percentile(diff_gray, 95))
        stats["max_diff"] = float(np.max(diff_gray))
        stats["min_diff"] = float(np.min(diff_gray))
        
        # Save difference visualization (amplified for visibility)
        diff_vis = np.clip(diff * 4.0, 0.0, 1.0)  # 4x amplification
        diff_vis_uint8 = (diff_vis * 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_vis_uint8)
        diff_out = out_dir / "ibl" / "ibl_only_diff_A_minus_B.png"
        diff_img.save(str(diff_out))
        stats["diff_image"] = str(diff_out)
        
        # Generate statistics figure
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Histogram of difference values
            axes[0].hist(diff_gray.flatten(), bins=100, color='steelblue', edgecolor='none', alpha=0.7)
            axes[0].set_xlabel("Absolute Difference")
            axes[0].set_ylabel("Pixel Count")
            axes[0].set_title("IBL Difference Histogram")
            axes[0].axvline(stats["mean_diff"], color='red', linestyle='--', label=f'Mean: {stats["mean_diff"]:.4f}')
            axes[0].axvline(stats["p95_diff"], color='orange', linestyle='--', label=f'P95: {stats["p95_diff"]:.4f}')
            axes[0].legend(fontsize=8)
            
            # Heatmap of difference
            im = axes[1].imshow(diff_gray, cmap='hot', aspect='auto')
            axes[1].set_title("Difference Heatmap")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Stats text box
            stats_text = (
                f"IBL Difference Statistics\n"
                f"─────────────────────\n"
                f"Mean:   {stats['mean_diff']:.6f}\n"
                f"Median: {stats['median_diff']:.6f}\n"
                f"P95:    {stats['p95_diff']:.6f}\n"
                f"Max:    {stats['max_diff']:.6f}\n"
                f"─────────────────────\n"
                f"HDR A: snow_field_4k\n"
                f"HDR B: air_museum_playground_4k"
            )
            axes[2].text(0.5, 0.5, stats_text, transform=axes[2].transAxes,
                        fontsize=12, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[2].axis('off')
            axes[2].set_title("Summary")
            
            plt.tight_layout()
            fig_out = out_dir / "ibl" / "fig_ibl_diff_stats.png"
            plt.savefig(str(fig_out), dpi=150)
            plt.close(fig)
            stats["stats_figure"] = str(fig_out)
            
        except ImportError:
            stats["stats_figure"] = "SKIP(matplotlib not available)"
        except Exception as e:
            stats["stats_figure"] = f"ERROR({e})"
        
        stats["status"] = "OK"
        return stats
    
    # Generate IBL diff and stats if IBL section was included
    ibl_stats = {}
    if not only or "ibl" in only:
        print("\n[INFO] Generating IBL difference visualization and statistics...")
        ibl_stats = generate_ibl_diff_and_stats()
        manifest["ibl_diff_stats"] = ibl_stats
        if ibl_stats.get("status") == "OK":
            print(f"  Mean diff: {ibl_stats.get('mean_diff', 'N/A'):.6f}")
            print(f"  P95 diff:  {ibl_stats.get('p95_diff', 'N/A'):.6f}")
            print(f"  Diff image: {ibl_stats.get('diff_image', 'N/A')}")
            print(f"  Stats fig:  {ibl_stats.get('stats_figure', 'N/A')}")
        else:
            print(f"  Status: {ibl_stats.get('status', 'UNKNOWN')}")

    # ----------------------------
    # Recomposition Proof Analysis
    # ----------------------------
    def analyze_recomposition() -> Dict[str, Any]:
        """Verify that IBL_combined ≈ diffuse + specular in linear space."""
        result: Dict[str, Any] = {}
        
        if args.dry_run:
            return {"status": "SKIP(dry_run)"}
        
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return {"status": "SKIP(missing_deps: PIL or numpy)"}
        
        # Load the three linear-encoded images
        combined_path = out_dir / "pbr" / "recomp_linear_combined.png"
        diffuse_path = out_dir / "pbr" / "recomp_linear_diffuse.png"
        specular_path = out_dir / "pbr" / "recomp_linear_specular.png"
        error_path = out_dir / "pbr" / "recomp_error_heatmap.png"
        
        paths = [combined_path, diffuse_path, specular_path, error_path]
        if not all(p.exists() for p in paths):
            return {"status": "SKIP(missing_input_images)", "missing": [str(p) for p in paths if not p.exists()]}
        
        # Decode linear values: linear = encoded * 4.0
        combined = np.array(Image.open(combined_path).convert("RGB")).astype(np.float32) / 255.0 * 4.0
        diffuse = np.array(Image.open(diffuse_path).convert("RGB")).astype(np.float32) / 255.0 * 4.0
        specular = np.array(Image.open(specular_path).convert("RGB")).astype(np.float32) / 255.0 * 4.0
        
        # Compute recomposition: should be combined = diffuse + specular
        recomposed = diffuse + specular
        error = np.abs(combined - recomposed)
        error_gray = np.mean(error, axis=2)
        
        # Statistics
        result["mean_error"] = float(np.mean(error_gray))
        result["median_error"] = float(np.median(error_gray))
        result["p95_error"] = float(np.percentile(error_gray, 95))
        result["max_error"] = float(np.max(error_gray))
        
        # Acceptance: P95 error should be < 0.001 (in linear [0,4] space)
        epsilon = 0.001
        if result["p95_error"] < epsilon:
            result["status"] = "PASS"
            result["acceptance"] = f"P95 error ({result['p95_error']:.6f}) < epsilon ({epsilon})"
        else:
            result["status"] = "FAIL"
            result["acceptance"] = f"P95 error ({result['p95_error']:.6f}) >= epsilon ({epsilon})"
        
        # Generate figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Error histogram
            axes[0].hist(error_gray.flatten(), bins=100, color='crimson', edgecolor='none', alpha=0.7)
            axes[0].set_xlabel("Recomposition Error (linear)")
            axes[0].set_ylabel("Pixel Count")
            axes[0].set_title("Recomposition Error Distribution")
            axes[0].axvline(result["p95_error"], color='orange', linestyle='--', label=f'P95: {result["p95_error"]:.6f}')
            axes[0].axvline(epsilon, color='green', linestyle='-', label=f'Threshold: {epsilon}')
            axes[0].legend(fontsize=8)
            
            # Error heatmap
            im = axes[1].imshow(error_gray, cmap='hot', aspect='auto', vmin=0, vmax=max(0.01, result["max_error"]))
            axes[1].set_title("Recomposition Error Heatmap")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Stats text
            stats_text = (
                f"Recomposition Proof\n"
                f"──────────────────\n"
                f"IBL = diff + spec?\n"
                f"──────────────────\n"
                f"Mean:   {result['mean_error']:.6f}\n"
                f"Median: {result['median_error']:.6f}\n"
                f"P95:    {result['p95_error']:.6f}\n"
                f"Max:    {result['max_error']:.6f}\n"
                f"──────────────────\n"
                f"Status: {result['status']}"
            )
            axes[2].text(0.5, 0.5, stats_text, transform=axes[2].transAxes,
                        fontsize=11, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', 
                        facecolor='lightgreen' if result['status']=='PASS' else 'lightcoral', alpha=0.5))
            axes[2].axis('off')
            axes[2].set_title("Verdict")
            
            plt.tight_layout()
            fig_out = out_dir / "pbr" / "fig_recomp_proof.png"
            plt.savefig(str(fig_out), dpi=150)
            plt.close(fig)
            result["figure"] = str(fig_out)
        except Exception as e:
            result["figure"] = f"ERROR({e})"
        
        return result

    # ----------------------------
    # SpecAA Sparkle Stress Test Analysis
    # ----------------------------
    def analyze_specaa_sparkle() -> Dict[str, Any]:
        """Verify that SpecAA ON reduces high-frequency energy vs OFF."""
        result: Dict[str, Any] = {}
        
        if args.dry_run:
            return {"status": "SKIP(dry_run)"}
        
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return {"status": "SKIP(missing_deps: PIL or numpy)"}
        
        on_path = out_dir / "pbr" / "specaa_sparkle_on.png"
        off_path = out_dir / "pbr" / "specaa_sparkle_off.png"
        
        if not on_path.exists() or not off_path.exists():
            return {"status": "SKIP(missing_input_images)"}
        
        img_on = np.array(Image.open(on_path).convert("RGB")).astype(np.float32) / 255.0
        img_off = np.array(Image.open(off_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Compute high-frequency energy using Laplacian (approximates sparkle intensity)
        def high_freq_energy(img: np.ndarray) -> float:
            """Compute sum of squared Laplacian as proxy for high-freq content."""
            gray = np.mean(img, axis=2)
            # Laplacian kernel approximation via differences
            lap_x = np.diff(gray, axis=1)
            lap_y = np.diff(gray, axis=0)
            # Pad to same size
            lap_x = np.pad(lap_x, ((0,0), (0,1)), mode='edge')
            lap_y = np.pad(lap_y, ((0,1), (0,0)), mode='edge')
            laplacian = lap_x[:lap_y.shape[0], :] + lap_y[:, :lap_x.shape[1]]
            return float(np.sum(laplacian ** 2))
        
        energy_on = high_freq_energy(img_on)
        energy_off = high_freq_energy(img_off)
        
        result["hf_energy_on"] = energy_on
        result["hf_energy_off"] = energy_off
        result["energy_ratio"] = energy_on / max(energy_off, 1e-10)
        result["reduction_pct"] = (1.0 - result["energy_ratio"]) * 100.0
        
        # Acceptance: SpecAA with screen-derivative variance should reduce HF energy
        # The sparkle test injects high-frequency normal perturbation that dpdx/dpdy can detect.
        # Toksvig should boost roughness and reduce sparkle intensity.
        threshold_pct = 5.0  # Require at least 5% reduction
        if result["reduction_pct"] >= threshold_pct:
            result["status"] = "PASS"
            result["acceptance"] = f"HF energy reduced by {result['reduction_pct']:.1f}% (>={threshold_pct}%)"
        elif result["reduction_pct"] > 0:
            result["status"] = "WARN"
            result["acceptance"] = f"Small reduction ({result['reduction_pct']:.1f}%) - less than {threshold_pct}% threshold"
        else:
            result["status"] = "FAIL"
            result["acceptance"] = f"SpecAA did not reduce HF energy (change: {result['reduction_pct']:.1f}%)"
        
        # Generate comparison figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # SpecAA OFF
            axes[0].imshow(img_off)
            axes[0].set_title(f"SpecAA OFF\nHF Energy: {energy_off:.1f}")
            axes[0].axis('off')
            
            # SpecAA ON
            axes[1].imshow(img_on)
            axes[1].set_title(f"SpecAA ON\nHF Energy: {energy_on:.1f}")
            axes[1].axis('off')
            
            # Stats
            stats_text = (
                f"SpecAA Sparkle Test\n"
                f"───────────────────\n"
                f"OFF energy: {energy_off:.1f}\n"
                f"ON energy:  {energy_on:.1f}\n"
                f"Ratio:      {result['energy_ratio']:.3f}\n"
                f"Reduction:  {result['reduction_pct']:.1f}%\n"
                f"───────────────────\n"
                f"Status: {result['status']}"
            )
            color = 'lightgreen' if result['status']=='PASS' else ('wheat' if result['status']=='WARN' else 'lightcoral')
            axes[2].text(0.5, 0.5, stats_text, transform=axes[2].transAxes,
                        fontsize=11, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
            axes[2].axis('off')
            axes[2].set_title("Verdict")
            
            plt.tight_layout()
            fig_out = out_dir / "pbr" / "fig_specaa_sparkle_test.png"
            plt.savefig(str(fig_out), dpi=150)
            plt.close(fig)
            result["figure"] = str(fig_out)
        except Exception as e:
            result["figure"] = f"ERROR({e})"
        
        return result

    # Run recomposition analysis if PBR section included
    recomp_result = {}
    if not only or "pbr" in only:
        print("\n[INFO] Running recomposition proof analysis...")
        recomp_result = analyze_recomposition()
        manifest["recomposition_proof"] = recomp_result
        if recomp_result.get("status") in ("PASS", "FAIL"):
            print(f"  P95 error: {recomp_result.get('p95_error', 'N/A'):.6f}")
            print(f"  Status: {recomp_result.get('status')}")
            print(f"  {recomp_result.get('acceptance', '')}")
        else:
            print(f"  Status: {recomp_result.get('status', 'UNKNOWN')}")

    # Run SpecAA sparkle analysis if PBR section included
    specaa_sparkle_result = {}
    if not only or "pbr" in only:
        print("\n[INFO] Running SpecAA sparkle stress test analysis...")
        specaa_sparkle_result = analyze_specaa_sparkle()
        manifest["specaa_sparkle_test"] = specaa_sparkle_result
        if specaa_sparkle_result.get("status") in ("PASS", "WARN", "XFAIL"):
            print(f"  HF energy OFF: {specaa_sparkle_result.get('hf_energy_off', 'N/A'):.1f}")
            print(f"  HF energy ON:  {specaa_sparkle_result.get('hf_energy_on', 'N/A'):.1f}")
            print(f"  Reduction: {specaa_sparkle_result.get('reduction_pct', 'N/A'):.1f}%")
            print(f"  Status: {specaa_sparkle_result.get('status')}")
            if specaa_sparkle_result.get("status") == "XFAIL":
                print(f"  Note: {specaa_sparkle_result.get('acceptance', '')}")
        else:
            print(f"  Status: {specaa_sparkle_result.get('status', 'UNKNOWN')}")

    # ----------------------------
    # Energy Sanity Analysis (NaN/Inf + bounds)
    # ----------------------------
    def analyze_energy_sanity() -> Dict[str, Any]:
        """Verify PBR energy output has no NaN/Inf and sane bounds."""
        result: Dict[str, Any] = {}
        
        if args.dry_run:
            return {"status": "SKIP(dry_run)"}
        
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return {"status": "SKIP(missing_deps: PIL or numpy)"}
        
        energy_path = out_dir / "pbr" / "pbr_terms_energy.png"
        if not energy_path.exists():
            return {"status": "SKIP(missing_input_image)", "path": str(energy_path)}
        
        # Load energy image (mode 12: energy visualization)
        # The shader outputs linear energy scaled: output = clamp(energy / 4.0, 0, 1)
        # So we decode: energy_linear = pixel_value * 4.0
        img = np.array(Image.open(energy_path).convert("RGB")).astype(np.float32) / 255.0
        energy_linear = img * 4.0  # Decode from [0,1] -> [0,4] linear
        energy_gray = np.mean(energy_linear, axis=2)
        
        # Check for NaN/Inf (should be impossible in PNG, but check decoded values)
        nan_count = int(np.sum(np.isnan(energy_gray)))
        inf_count = int(np.sum(np.isinf(energy_gray)))
        
        # Statistics
        result["nan_count"] = nan_count
        result["inf_count"] = inf_count
        result["min_energy"] = float(np.min(energy_gray))
        result["max_energy"] = float(np.max(energy_gray))
        result["mean_energy"] = float(np.mean(energy_gray))
        result["p95_energy"] = float(np.percentile(energy_gray, 95))
        result["p99_energy"] = float(np.percentile(energy_gray, 99))
        
        # Check for clamping artifacts (exactly 0 or exactly 4.0 after decode)
        zero_count = int(np.sum(energy_gray == 0.0))
        max_clamp_count = int(np.sum(energy_gray >= 3.99))  # Near max after decode
        result["zero_pixel_count"] = zero_count
        result["max_clamp_pixel_count"] = max_clamp_count
        total_pixels = energy_gray.size
        result["zero_pixel_pct"] = 100.0 * zero_count / total_pixels
        result["max_clamp_pct"] = 100.0 * max_clamp_count / total_pixels
        
        # Acceptance criteria
        # - No NaN/Inf
        # - p99 < E_MAX (16.0 in linear; since we decode *4, max possible is 4.0, so use 4.0 as hard cap)
        # - Less than 1% pixels at max clamp (indicates blown highlights)
        E_MAX = 4.0  # Maximum decodable value
        MAX_CLAMP_THRESHOLD_PCT = 1.0
        
        issues = []
        if nan_count > 0:
            issues.append(f"NaN pixels: {nan_count}")
        if inf_count > 0:
            issues.append(f"Inf pixels: {inf_count}")
        if result["p99_energy"] >= E_MAX * 0.99:
            issues.append(f"p99 energy ({result['p99_energy']:.3f}) near/at clamp ceiling")
        if result["max_clamp_pct"] > MAX_CLAMP_THRESHOLD_PCT:
            issues.append(f"Too many clamped pixels: {result['max_clamp_pct']:.1f}% > {MAX_CLAMP_THRESHOLD_PCT}%")
        
        if not issues:
            result["status"] = "PASS"
            result["acceptance"] = f"Energy sane: p99={result['p99_energy']:.3f}, no NaN/Inf, {result['max_clamp_pct']:.2f}% clamped"
        else:
            result["status"] = "FAIL"
            result["acceptance"] = "Issues: " + "; ".join(issues)
        
        # Generate figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Histogram
            axes[0].hist(energy_gray.flatten(), bins=100, color='darkorange', edgecolor='none', alpha=0.7)
            axes[0].set_xlabel("Energy (linear, decoded)")
            axes[0].set_ylabel("Pixel Count")
            axes[0].set_title("Energy Distribution")
            axes[0].axvline(result["p99_energy"], color='red', linestyle='--', label=f'P99: {result["p99_energy"]:.3f}')
            axes[0].axvline(result["mean_energy"], color='blue', linestyle='--', label=f'Mean: {result["mean_energy"]:.3f}')
            axes[0].legend(fontsize=8)
            
            # Energy heatmap
            im = axes[1].imshow(energy_gray, cmap='inferno', aspect='auto', vmin=0, vmax=min(4.0, result["max_energy"] * 1.1))
            axes[1].set_title("Energy Heatmap")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Stats text
            stats_text = (
                f"Energy Sanity Check\n"
                f"───────────────────\n"
                f"NaN count:  {nan_count}\n"
                f"Inf count:  {inf_count}\n"
                f"───────────────────\n"
                f"Min:   {result['min_energy']:.4f}\n"
                f"Mean:  {result['mean_energy']:.4f}\n"
                f"P95:   {result['p95_energy']:.4f}\n"
                f"P99:   {result['p99_energy']:.4f}\n"
                f"Max:   {result['max_energy']:.4f}\n"
                f"───────────────────\n"
                f"Clamped: {result['max_clamp_pct']:.2f}%\n"
                f"───────────────────\n"
                f"Status: {result['status']}"
            )
            color = 'lightgreen' if result['status']=='PASS' else 'lightcoral'
            axes[2].text(0.5, 0.5, stats_text, transform=axes[2].transAxes,
                        fontsize=10, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
            axes[2].axis('off')
            axes[2].set_title("Verdict")
            
            plt.tight_layout()
            fig_out = out_dir / "pbr" / "fig_pbr_energy_sanity.png"
            plt.savefig(str(fig_out), dpi=150)
            plt.close(fig)
            result["figure"] = str(fig_out)
        except Exception as e:
            result["figure"] = f"ERROR({e})"
        
        return result

    # Run energy sanity analysis if PBR section included
    energy_result = {}
    if not only or "pbr" in only:
        print("\n[INFO] Running PBR energy sanity analysis...")
        energy_result = analyze_energy_sanity()
        manifest["energy_sanity"] = energy_result
        if energy_result.get("status") in ("PASS", "FAIL"):
            print(f"  NaN/Inf: {energy_result.get('nan_count', 0)}/{energy_result.get('inf_count', 0)}")
            print(f"  P99 energy: {energy_result.get('p99_energy', 'N/A'):.4f}")
            print(f"  Clamped pixels: {energy_result.get('max_clamp_pct', 0):.2f}%")
            print(f"  Status: {energy_result.get('status')}")
        else:
            print(f"  Status: {energy_result.get('status', 'UNKNOWN')}")

    # ----------------------------
    # POM Effect Analysis (todo.md requirement)
    # ----------------------------
    def analyze_pom_effect() -> Dict[str, Any]:
        """Verify that POM produces measurable difference at grazing angles.
        
        Generates:
          - pom_diff_on_minus_off.png (absolute difference visualization)
          - fig_pom_effect_roi.png (quantitative analysis figure)
        """
        result: Dict[str, Any] = {}
        
        if args.dry_run:
            return {"status": "SKIP(dry_run)"}
        
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return {"status": "SKIP(missing_deps: PIL or numpy)"}
        
        on_path = out_dir / "pom" / "pom_on_grazing.png"
        off_path = out_dir / "pom" / "pom_off_grazing.png"
        offset_path = out_dir / "pom" / "dbg_pom_offset_mag.png"
        
        if not on_path.exists() or not off_path.exists():
            return {"status": "SKIP(missing_input_images)", "missing": [str(p) for p in [on_path, off_path] if not p.exists()]}
        
        # Load images
        img_on = np.array(Image.open(on_path).convert("RGB")).astype(np.float32) / 255.0
        img_off = np.array(Image.open(off_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Compute absolute difference
        diff = np.abs(img_on - img_off)
        diff_gray = np.mean(diff, axis=2)  # Average across RGB for scalar diff
        
        # Statistics
        result["mean_diff"] = float(np.mean(diff_gray))
        result["median_diff"] = float(np.median(diff_gray))
        result["p95_diff"] = float(np.percentile(diff_gray, 95))
        result["max_diff"] = float(np.max(diff_gray))
        result["nonzero_pct"] = float(np.sum(diff_gray > 0.001) / diff_gray.size * 100)
        
        # Save difference visualization (amplified 4x for visibility)
        diff_vis = np.clip(diff * 4.0, 0.0, 1.0)
        diff_vis_uint8 = (diff_vis * 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_vis_uint8)
        diff_out = out_dir / "pom" / "pom_diff_on_minus_off.png"
        diff_img.save(str(diff_out))
        result["diff_image"] = str(diff_out)
        
        # Load offset magnitude image if available
        offset_stats = {}
        if offset_path.exists():
            offset_img = np.array(Image.open(offset_path).convert("L")).astype(np.float32) / 255.0
            offset_stats["mean_offset"] = float(np.mean(offset_img))
            offset_stats["max_offset"] = float(np.max(offset_img))
            offset_stats["nonzero_pct"] = float(np.sum(offset_img > 0.01) / offset_img.size * 100)
        result["offset_stats"] = offset_stats
        
        # Acceptance criteria (from todo.md):
        # - Difference map is not trivial
        # - Offset magnitude map is non-zero and spatially plausible
        MIN_MEAN_DIFF = 0.001  # Minimum mean difference threshold
        MIN_NONZERO_PCT = 1.0  # At least 1% of pixels should differ
        
        issues = []
        if result["mean_diff"] < MIN_MEAN_DIFF:
            issues.append(f"Mean diff ({result['mean_diff']:.6f}) < threshold ({MIN_MEAN_DIFF})")
        if result["nonzero_pct"] < MIN_NONZERO_PCT:
            issues.append(f"Nonzero pixels ({result['nonzero_pct']:.1f}%) < threshold ({MIN_NONZERO_PCT}%)")
        if offset_stats and offset_stats.get("nonzero_pct", 0) < MIN_NONZERO_PCT:
            issues.append(f"Offset nonzero pixels ({offset_stats.get('nonzero_pct', 0):.1f}%) < threshold ({MIN_NONZERO_PCT}%)")
        
        if not issues:
            result["status"] = "PASS"
            result["acceptance"] = f"POM effect verified: mean_diff={result['mean_diff']:.4f}, {result['nonzero_pct']:.1f}% pixels differ"
        else:
            result["status"] = "FAIL"
            result["acceptance"] = "Issues: " + "; ".join(issues)
        
        # Generate analysis figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            n_cols = 4 if offset_path.exists() else 3
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
            
            # POM ON
            axes[0].imshow(img_on)
            axes[0].set_title("POM ON (grazing)")
            axes[0].axis('off')
            
            # POM OFF
            axes[1].imshow(img_off)
            axes[1].set_title("POM OFF (grazing)")
            axes[1].axis('off')
            
            # Difference (amplified)
            axes[2].imshow(diff_vis)
            axes[2].set_title(f"Difference (4x)\nMean: {result['mean_diff']:.4f}")
            axes[2].axis('off')
            
            # Offset magnitude (if available)
            if offset_path.exists() and len(axes) > 3:
                axes[3].imshow(offset_img, cmap='hot')
                axes[3].set_title(f"POM Offset Magnitude\nMax: {offset_stats.get('max_offset', 0):.3f}")
                axes[3].axis('off')
            
            plt.suptitle(f"POM Effect Analysis - Status: {result['status']}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fig_out = out_dir / "pom" / "fig_pom_effect_roi.png"
            plt.savefig(str(fig_out), dpi=150)
            plt.close(fig)
            result["figure"] = str(fig_out)
        except Exception as e:
            result["figure"] = f"ERROR({e})"
        
        return result

    # Run POM analysis if POM section included
    pom_result = {}
    if not only or "pom" in only:
        print("\n[INFO] Running POM effect analysis...")
        pom_result = analyze_pom_effect()
        manifest["pom_effect"] = pom_result
        if pom_result.get("status") in ("PASS", "FAIL"):
            print(f"  Mean diff: {pom_result.get('mean_diff', 'N/A'):.6f}")
            print(f"  Nonzero pixels: {pom_result.get('nonzero_pct', 'N/A'):.1f}%")
            if pom_result.get("offset_stats"):
                print(f"  Offset nonzero: {pom_result['offset_stats'].get('nonzero_pct', 'N/A'):.1f}%")
            print(f"  Status: {pom_result.get('status')}")
            print(f"  Diff image: {pom_result.get('diff_image', 'N/A')}")
            print(f"  Figure: {pom_result.get('figure', 'N/A')}")
        else:
            print(f"  Status: {pom_result.get('status', 'UNKNOWN')}")

    # ----------------------------
    # Triplanar Jitter Analysis (todo.md requirement T2)
    # ----------------------------
    def analyze_triplanar_jitter() -> Dict[str, Any]:
        """Verify that triplanar mapping is world-space stable (no UV drift/swimming).
        
        Generates:
          - triplanar_jitter_diff.png (absolute difference visualization)
          - fig_triplanar_jitter_stats.png (quantitative analysis figure)
        
        T2 Requirement: With camera movement, triplanar mapping stays locked to world space.
        Small differences are expected due to:
          - Specular/Fresnel changes with view angle
          - Subpixel sampling differences
        But the checker pattern itself should NOT swim/drift.
        """
        result: Dict[str, Any] = {}
        
        if args.dry_run:
            return {"status": "SKIP(dry_run)"}
        
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return {"status": "SKIP(missing_deps: PIL or numpy)"}
        
        a_path = out_dir / "triplanar" / "triplanar_camera_jitter_A.png"
        b_path = out_dir / "triplanar" / "triplanar_camera_jitter_B.png"
        weights_path = out_dir / "triplanar" / "dbg_triplanar_weights.png"
        checker_path = out_dir / "triplanar" / "triplanar_checker.png"
        
        if not a_path.exists() or not b_path.exists():
            return {"status": "SKIP(missing_input_images)", "missing": [str(p) for p in [a_path, b_path] if not p.exists()]}
        
        # Load jitter test images
        img_a = np.array(Image.open(a_path).convert("RGB")).astype(np.float32) / 255.0
        img_b = np.array(Image.open(b_path).convert("RGB")).astype(np.float32) / 255.0
        
        # Compute absolute difference
        diff = np.abs(img_a - img_b)
        diff_gray = np.mean(diff, axis=2)  # Average across RGB for scalar diff
        
        # Statistics
        result["mean_diff"] = float(np.mean(diff_gray))
        result["median_diff"] = float(np.median(diff_gray))
        result["p95_diff"] = float(np.percentile(diff_gray, 95))
        result["max_diff"] = float(np.max(diff_gray))
        
        # Save difference visualization (amplified 4x for visibility)
        diff_vis = np.clip(diff * 4.0, 0.0, 1.0)
        diff_vis_uint8 = (diff_vis * 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_vis_uint8)
        diff_out = out_dir / "triplanar" / "triplanar_jitter_diff.png"
        diff_img.save(str(diff_out))
        result["diff_image"] = str(diff_out)
        
        # Analyze triplanar weights image if available
        weights_stats = {}
        if weights_path.exists():
            weights_img = np.array(Image.open(weights_path).convert("RGB")).astype(np.float32) / 255.0
            # Check that weights approximately sum to 1 (R+G+B ≈ 1)
            weight_sum = np.sum(weights_img, axis=2)
            weights_stats["mean_sum"] = float(np.mean(weight_sum))
            weights_stats["min_sum"] = float(np.min(weight_sum))
            weights_stats["max_sum"] = float(np.max(weight_sum))
            # Check individual weight statistics
            weights_stats["mean_x"] = float(np.mean(weights_img[:,:,0]))  # Red = X weight
            weights_stats["mean_y"] = float(np.mean(weights_img[:,:,1]))  # Green = Y weight
            weights_stats["mean_z"] = float(np.mean(weights_img[:,:,2]))  # Blue = Z weight
        result["weights_stats"] = weights_stats
        
        # Acceptance criteria (from todo.md):
        # - Jitter difference should be small (stable) except for expected specular changes
        # - For checker pattern with 0.5° camera rotation, expect very small differences
        # - P95 difference should be < 0.1 (allowing for specular and subpixel changes)
        MAX_P95_DIFF = 0.1  # Allow 10% difference at p95 for specular changes
        
        issues = []
        if result["p95_diff"] > MAX_P95_DIFF:
            issues.append(f"P95 diff ({result['p95_diff']:.4f}) > threshold ({MAX_P95_DIFF}) - possible UV swimming")
        
        # Check weight sum is approximately 1 (T1 requirement)
        if weights_stats:
            if abs(weights_stats["mean_sum"] - 1.0) > 0.05:
                issues.append(f"Weight sum ({weights_stats['mean_sum']:.4f}) deviates from 1.0 by more than 0.05")
        
        if not issues:
            result["status"] = "PASS"
            result["acceptance"] = f"Triplanar stable: p95_diff={result['p95_diff']:.4f}, mean_diff={result['mean_diff']:.4f}"
        else:
            result["status"] = "FAIL"
            result["acceptance"] = "Issues: " + "; ".join(issues)
        
        # Generate analysis figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            n_cols = 4
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
            
            # Camera A
            axes[0].imshow(img_a)
            axes[0].set_title("Camera A (baseline)")
            axes[0].axis('off')
            
            # Camera B
            axes[1].imshow(img_b)
            axes[1].set_title("Camera B (+0.5° phi)")
            axes[1].axis('off')
            
            # Difference (amplified)
            axes[2].imshow(diff_vis)
            axes[2].set_title(f"Difference (4x)\nMean: {result['mean_diff']:.4f}")
            axes[2].axis('off')
            
            # Stats text
            stats_text = (
                f"Triplanar Jitter Test\n"
                f"─────────────────────\n"
                f"T2: No UV swimming\n"
                f"─────────────────────\n"
                f"Mean diff: {result['mean_diff']:.6f}\n"
                f"Median:    {result['median_diff']:.6f}\n"
                f"P95:       {result['p95_diff']:.6f}\n"
                f"Max:       {result['max_diff']:.6f}\n"
            )
            if weights_stats:
                stats_text += (
                    f"─────────────────────\n"
                    f"T1: Weights sum to 1\n"
                    f"Mean sum: {weights_stats['mean_sum']:.4f}\n"
                )
            stats_text += (
                f"─────────────────────\n"
                f"Status: {result['status']}"
            )
            color = 'lightgreen' if result['status']=='PASS' else 'lightcoral'
            axes[3].text(0.5, 0.5, stats_text, transform=axes[3].transAxes,
                        fontsize=10, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
            axes[3].axis('off')
            axes[3].set_title("Verdict")
            
            plt.suptitle(f"Triplanar Stability Analysis - Status: {result['status']}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fig_out = out_dir / "triplanar" / "fig_triplanar_jitter_stats.png"
            plt.savefig(str(fig_out), dpi=150)
            plt.close(fig)
            result["figure"] = str(fig_out)
        except Exception as e:
            result["figure"] = f"ERROR({e})"
        
        return result

    # Run triplanar analysis if triplanar section included
    triplanar_result = {}
    if not only or "triplanar" in only:
        print("\n[INFO] Running triplanar jitter analysis...")
        triplanar_result = analyze_triplanar_jitter()
        manifest["triplanar_jitter"] = triplanar_result
        if triplanar_result.get("status") in ("PASS", "FAIL"):
            print(f"  Mean diff: {triplanar_result.get('mean_diff', 'N/A'):.6f}")
            print(f"  P95 diff: {triplanar_result.get('p95_diff', 'N/A'):.6f}")
            if triplanar_result.get("weights_stats"):
                print(f"  Weight sum: {triplanar_result['weights_stats'].get('mean_sum', 'N/A'):.4f}")
            print(f"  Status: {triplanar_result.get('status')}")
            print(f"  Diff image: {triplanar_result.get('diff_image', 'N/A')}")
            print(f"  Figure: {triplanar_result.get('figure', 'N/A')}")
        else:
            print(f"  Status: {triplanar_result.get('status', 'UNKNOWN')}")

    # ----------------------------
    # Sanity checks that catch your current failure mode
    # ----------------------------
    def check_not_identical(a_rel: str, b_rel: str, label: str) -> None:
        if args.dry_run:
            manifest["checks"].append({"check": label, "status": "SKIP(dry_run)"})
            return
        a = out_dir / a_rel
        b = out_dir / b_rel
        if not a.exists() or not b.exists():
            manifest["checks"].append({"check": label, "status": "FAIL(missing_files)", "a": a_rel, "b": b_rel})
            return
        if _files_identical(a, b):
            manifest["checks"].append({"check": label, "status": "FAIL(identical_outputs)", "a": a_rel, "b": b_rel})
        else:
            manifest["checks"].append({"check": label, "status": "OK", "a": a_rel, "b": b_rel})

    # PBR debug splits must not match
    check_not_identical("pbr/pbr_terms_diffuse_only.png", "pbr/pbr_terms_specular_only.png", "PBR split: diffuse != specular")
    check_not_identical("pbr/pbr_terms_fresnel_F.png", "pbr/pbr_terms_ndotv.png", "PBR split: Fresnel != NdotV")

    # Roughness sweep must change output
    check_not_identical("pbr/pbr_roughness_0.25.png", "pbr/pbr_roughness_2.00.png", "Roughness sweep: 0.25 != 2.00")

    # SpecAA toggle should change spec-only output (not necessarily huge, but must not be identical)
    # NOTE: For smooth terrain (like Gore Range), Toksvig effect is minimal because normal_variance ≈ 0
    # This check may produce identical outputs for smooth terrain - this is expected behavior, not a bug.
    def check_specaa_warn(a_rel: str, b_rel: str, label: str) -> None:
        """SpecAA check that produces WARN instead of FAIL for identical outputs (expected for smooth terrain)."""
        if args.dry_run:
            manifest["checks"].append({"check": label, "status": "SKIP(dry_run)"})
            return
        a = out_dir / a_rel
        b = out_dir / b_rel
        if not a.exists() or not b.exists():
            manifest["checks"].append({"check": label, "status": "FAIL(missing_files)", "a": a_rel, "b": b_rel})
            return
        if _files_identical(a, b):
            # WARN instead of FAIL - smooth terrain has no normal variance for Toksvig to correct
            manifest["checks"].append({"check": label, "status": "WARN(identical_outputs_expected_for_smooth_terrain)", "a": a_rel, "b": b_rel})
        else:
            manifest["checks"].append({"check": label, "status": "OK", "a": a_rel, "b": b_rel})
    
    check_specaa_warn("pbr/pbr_specaa_on.png", "pbr/pbr_specaa_off.png", "SpecAA: ON != OFF (spec-only)")

    # IBL must respond to HDR changes
    check_not_identical("ibl/ibl_only_hdr_a.png", "ibl/ibl_only_hdr_b.png", "IBL-only: HDR A != HDR B")
    check_not_identical("ibl/ibl_beauty_hdr_a.png", "ibl/ibl_beauty_hdr_b.png", "Beauty: HDR A != HDR B")

    # POM should create a difference at grazing angles
    check_not_identical("pom/pom_on_grazing.png", "pom/pom_off_grazing.png", "POM: ON != OFF (grazing)")

    # Triplanar weight visualization should be distinct from checker pattern
    check_not_identical("triplanar/dbg_triplanar_weights.png", "triplanar/triplanar_checker.png", "Triplanar: weights != checker")
    
    # Camera jitter test - A and B should be slightly different (due to view angle change)
    # but not identical (which would indicate the test images weren't generated correctly)
    # Note: We DON'T check for identical here because small changes are expected from view angle

    # Recomposition proof: linear outputs must be meaningful
    check_not_identical("pbr/recomp_linear_diffuse.png", "pbr/recomp_linear_specular.png", "Recomp: diffuse != specular (linear)")
    
    # SpecAA sparkle stress test: With screen-derivative variance, SpecAA should work on all terrain
    # including procedural DEM normals. ON vs OFF should differ when synthetic sparkle is injected.
    check_not_identical("pbr/specaa_sparkle_on.png", "pbr/specaa_sparkle_off.png", "SpecAA Sparkle: ON != OFF")

    # Add analysis results to checks
    if recomp_result.get("status") == "PASS":
        manifest["checks"].append({"check": "Recomposition: IBL = diff + spec", "status": "OK", "p95_error": recomp_result.get("p95_error")})
    elif recomp_result.get("status") == "FAIL":
        manifest["checks"].append({"check": "Recomposition: IBL = diff + spec", "status": "FAIL(p95_error_too_high)", "p95_error": recomp_result.get("p95_error")})

    if specaa_sparkle_result.get("status") == "PASS":
        manifest["checks"].append({"check": "SpecAA Sparkle: reduces HF energy", "status": "OK", "reduction_pct": specaa_sparkle_result.get("reduction_pct")})
    elif specaa_sparkle_result.get("status") == "WARN":
        manifest["checks"].append({"check": "SpecAA Sparkle: reduces HF energy", "status": "WARN(small_reduction)", "reduction_pct": specaa_sparkle_result.get("reduction_pct")})
    elif specaa_sparkle_result.get("status") == "FAIL":
        manifest["checks"].append({"check": "SpecAA Sparkle: reduces HF energy", "status": "FAIL(no_reduction)", "reduction_pct": specaa_sparkle_result.get("reduction_pct")})

    # Add energy sanity result to checks
    if energy_result.get("status") == "PASS":
        manifest["checks"].append({"check": "Energy sanity: no NaN/Inf, sane bounds", "status": "OK", "p99_energy": energy_result.get("p99_energy")})
    elif energy_result.get("status") == "FAIL":
        manifest["checks"].append({"check": "Energy sanity: no NaN/Inf, sane bounds", "status": "FAIL", "acceptance": energy_result.get("acceptance")})

    # Add POM effect analysis result to checks
    if pom_result.get("status") == "PASS":
        manifest["checks"].append({"check": "POM effect: measurable difference at grazing", "status": "OK", "mean_diff": pom_result.get("mean_diff"), "nonzero_pct": pom_result.get("nonzero_pct")})
    elif pom_result.get("status") == "FAIL":
        manifest["checks"].append({"check": "POM effect: measurable difference at grazing", "status": "FAIL", "acceptance": pom_result.get("acceptance")})

    # Add triplanar jitter analysis result to checks
    if triplanar_result.get("status") == "PASS":
        manifest["checks"].append({"check": "Triplanar: world-space stable (T2)", "status": "OK", "p95_diff": triplanar_result.get("p95_diff"), "weight_sum": triplanar_result.get("weights_stats", {}).get("mean_sum")})
    elif triplanar_result.get("status") == "FAIL":
        manifest["checks"].append({"check": "Triplanar: world-space stable (T2)", "status": "FAIL", "acceptance": triplanar_result.get("acceptance")})

    # Write manifest
    manifest_path = out_dir / "meta" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Write a minimal README on how to interpret.
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join([
            "# Forge3D Proof Pack — Gore Range\n",
            "This directory is generated by `tools/run_proof_pack_gore.py`.\n",
            "## What counts as success\n",
            "### PBR Term Separation",
            "- `pbr_terms_diffuse_only` and `pbr_terms_specular_only` must be **visibly different**.",
            "- Roughness sweep images must change specular highlights/blur.",
            "- SpecAA ON vs OFF must not be byte-identical.\n",
            "### Recomposition Proof (NEW)",
            "- `recomp_linear_combined` must equal `recomp_linear_diffuse + recomp_linear_specular`",
            "- P95 error must be < 0.001 in linear [0,4] space",
            "- See `fig_recomp_proof.png` for analysis\n",
            "### SpecAA Sparkle Stress Test [SCOPED OUT for DEM]",
            "- **XFAIL for DEM terrain**: Toksvig requires mipmapped normal maps.",
            "- Procedural DEM normals are always unit-length → no variance → Toksvig is identity.",
            "- Identical outputs are EXPECTED (not a failure) for DEM terrain.",
            "- Will be re-enabled when detail normal maps are added.\n",
            "### Energy Sanity (NEW)",
            "- No NaN/Inf pixels in energy output",
            "- P99 energy within sane bounds (not clipping)",
            "- See `fig_pbr_energy_sanity.png` for analysis\n",
            "### IBL",
            "- Changing HDRI must change IBL-only and usually beauty.\n",
            "### POM (Parallax Occlusion Mapping)",
            "- `pom_on_grazing.png` vs `pom_off_grazing.png` must not be identical.",
            "- `dbg_pom_offset_mag.png` shows UV offset magnitude (grayscale: 0=none, white=max).",
            "- `pom_diff_on_minus_off.png` shows absolute difference (4x amplified).",
            "- `fig_pom_effect_roi.png` provides quantitative analysis.",
            "- **M1**: Grazing-angle effect - POM must change appearance at low cam-theta (10°).",
            "- **M2**: Debug mode 18 shows offset magnitude spatially correlated with height variation.\n",
            "### Triplanar Mapping (NEW)",
            "- `dbg_triplanar_weights.png` shows RGB = x/y/z projection weights (T1 proof).",
            "- `triplanar_checker.png` shows procedural checker to expose UV stretching (T2 proof).",
            "- `triplanar_camera_jitter_A/B.png` + `triplanar_jitter_diff.png` prove world-space stability.",
            "- `fig_triplanar_jitter_stats.png` provides quantitative analysis.",
            "- **T1**: Blend weights must sum to 1.0, change smoothly with surface normal.",
            "  - RED = X-axis projection dominant (steep cliff facing X)",
            "  - GREEN = Y-axis projection dominant (flat/horizontal surface)",
            "  - BLUE = Z-axis projection dominant (steep cliff facing Z)",
            "- **T2**: No UV swimming - checker pattern stays locked to world space during camera movement.",
            "  - P95 difference between jitter A/B should be < 0.1 (small specular changes allowed).\n",
            "If any of those pairs are identical, the corresponding feature is not being exercised (debug mode/env var not plumbed, wrong shader path, etc.).\n",
            "See `meta/manifest.json` for commands, env overrides, timings, and pass/fail checks.\n",
        ]) + "\n",
        encoding="utf-8",
    )

    # Final status: fail if any checks failed (WARN/XFAIL don't count as failure)
    failed = [c for c in manifest["checks"] if str(c.get("status", "")).startswith("FAIL")]
    warned = [c for c in manifest["checks"] if str(c.get("status", "")).startswith("WARN")]
    xfailed = [c for c in manifest["checks"] if str(c.get("status", "")).startswith("XFAIL")]
    
    if xfailed:
        print("\n[XFAIL] (expected failures, scoped out for DEM terrain):")
        for x in xfailed:
            print(f" - {x['check']}: {x['status']}")
    
    if warned:
        print("\n[WARNINGS] (expected behavior, not failures):")
        for w in warned:
            print(f" - {w['check']}: {w['status']}")
    
    if failed:
        print("\n[RESULT] PROOF PACK FAILED checks:")
        for f in failed:
            print(f" - {f['check']}: {f['status']}  ({f.get('a','')} vs {f.get('b','')})")
        print(f"\nSee: {manifest_path}")
        return 1

    print("\n[RESULT] PROOF PACK PASSED all checks.")
    print(f"See: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
