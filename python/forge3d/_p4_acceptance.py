#!/usr/bin/env python3
"""
P4 Acceptance Artifacts Generator (CPU-side)

Produces the required gallery files for Milestone 4:
  - reports/p4_env_base.png
  - reports/p4_irradiance_cube.png
  - reports/p4_specular_cube_mips.png
  - reports/p4_brdf_lut.png
  - reports/p4_meta.json

Implementation notes:
- Uses the reference CPU pipeline from examples/m4_generate.py via a robust import shim.
- Includes a lightweight CPU cache (npz + json) keyed by sha256 of env + params to report cache_used and timings.
- Defaults to a synthetic HDR if repo HDR is missing.
- All PNGs are opaque RGB.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Robust importer for examples/m4_generate.py (kept in examples/ to avoid package bloat)
# --------------------------------------------------------------------------------------

def _import_m4_generate():
    import importlib.util, sys
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../forge3d
    m4_path = repo_root / "examples" / "m4_generate.py"
    if not m4_path.exists():
        raise ImportError(f"m4_generate.py not found at {m4_path}")
    spec = importlib.util.spec_from_file_location("m4_generate", str(m4_path))
    if spec is None or spec.loader is None:
        raise ImportError("Failed to import examples/m4_generate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules['m4_generate'] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class P4Params:
    base: int = 512  # base cubemap size (Global constraints)
    irr: int = 64    # irradiance face size (Global constraints)
    brdf: int = 256  # LUT size (square) (Global constraints)


def _sha256_for_env_and_params(env: np.ndarray, params: P4Params) -> str:
    h = hashlib.sha256()
    h.update(str(params.base).encode())
    h.update(str(params.irr).encode())
    h.update(str(params.brdf).encode())
    # Use a stable view of env floats
    env_b = np.ascontiguousarray(env.astype(np.float32, copy=False)).tobytes()
    h.update(env_b)
    return h.hexdigest()


def _make_strip(faces_rgb_u8: np.ndarray, gutter_px: int = 16) -> np.ndarray:
    """Faces: (6, H, W, 3) uint8 -> horizontal strip with gutters."""
    assert faces_rgb_u8.ndim == 4 and faces_rgb_u8.shape[0] == 6
    _, H, W, C = faces_rgb_u8.shape
    gutter = np.zeros((H, gutter_px, C), dtype=np.uint8)
    pieces: List[np.ndarray] = []
    for i in range(6):
        if i:
            pieces.append(gutter)
        pieces.append(faces_rgb_u8[i])
    return np.concatenate(pieces, axis=1)


def _faces_to_rgb_u8(m4, faces_f32: np.ndarray) -> np.ndarray:
    # faces_f32: (6, H, W, 3) float32 linear -> tonemap -> srgb -> u8
    out = np.zeros_like(faces_f32, dtype=np.uint8)
    for i in range(faces_f32.shape[0]):
        out[i] = m4.tonemap_to_u8(faces_f32[i])
    return out


def run_p4_acceptance(
    outdir: str | Path | None = None,
    *,
    base: int = 512,
    irr: int = 64,
    brdf: int = 256,
    use_synthetic_env: bool = False,
) -> Dict[str, Any]:
    """
    Build P4 acceptance artifacts. Returns the meta dict.

    Parameters
    ----------
    outdir : str | Path | None
        Directory to write reports. Defaults to repo_root/reports/
    base, irr, brdf : int
        Sizes for base cube, irradiance, and BRDF LUT respectively.
    use_synthetic_env : bool
        If True, forces synthetic HDR environment. Otherwise tries assets/snow_field_4k.hdr.
    """
    m4 = _import_m4_generate()

    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    out = Path(outdir) if outdir is not None else (repo_root / "reports")
    out.mkdir(parents=True, exist_ok=True)

    # Resolve environment
    hdr_repo = repo_root / "assets" / "snow_field_4k.hdr"
    if use_synthetic_env or not hdr_repo.exists():
        env, mode = m4.load_hdr_environment(hdr_repo, force_synthetic=True)
    else:
        env, mode = m4.load_hdr_environment(hdr_repo, force_synthetic=False)

    params = P4Params(base=base, irr=irr, brdf=brdf)
    key = _sha256_for_env_and_params(env, params)

    # Simple CPU-side cache
    cache_npz = out / f"p4_cache_{base}_{irr}_{brdf}.npz"
    cache_meta = out / "p4_cache_meta.json"
    cache_used = False

    t0 = time.perf_counter()
    prefilter_levels = None
    irradiance_faces = None
    lut = None

    if cache_npz.exists() and cache_meta.exists():
        try:
            with open(cache_meta, "r", encoding="utf-8") as f:
                meta_cache = json.load(f)
            if meta_cache.get("sha256") == key:
                data = np.load(cache_npz)
                # Reconstruct structures
                levels = meta_cache.get("mips", 1)
                prefilter_levels = []
                for i in range(levels):
                    faces = data[f"prefilter_{i}"]
                    rough = float(meta_cache["prefilter"][str(i)]["roughness"])  # type: ignore[index]
                    samples = int(meta_cache["prefilter"][str(i)]["samples"])    # type: ignore[index]
                    size_i = int(meta_cache["prefilter"][str(i)]["size"])        # type: ignore[index]
                    prefilter_levels.append(m4.PrefilterLevel(faces=faces, roughness=rough, samples=samples, size=size_i))
                irradiance_faces = data["irradiance_faces"]
                lut = data["brdf_lut"]
                cache_used = True
        except Exception:
            cache_used = False

    load_ms = int((time.perf_counter() - t0) * 1000.0)

    if prefilter_levels is None or irradiance_faces is None or lut is None:
        # Compute fresh
        t1 = time.perf_counter()
        base_faces, _ = m4.equirect_to_cubemap(env, params.base)
        prefilter_levels, _, _ = m4.compute_prefilter_chain(
            env,
            params.base,
            m4.PREFILTER_SAMPLES_TOP,
            m4.PREFILTER_SAMPLES_BOTTOM,
        )
        irradiance_faces = m4.build_irradiance_cubemap(env, params.irr, m4.IRRADIANCE_SAMPLES)
        lut = m4.compute_dfg_lut(params.brdf, m4.DFG_LUT_SAMPLES)
        compute_ms = int((time.perf_counter() - t1) * 1000.0)

        # Save cache
        try:
            prefilter_dict = {str(i): {"roughness": float(l.roughness), "samples": int(l.samples), "size": int(l.size)} for i, l in enumerate(prefilter_levels)}
            with open(cache_meta, "w", encoding="utf-8") as f:
                json.dump({
                    "sha256": key,
                    "base": params.base,
                    "irr": params.irr,
                    "brdf": params.brdf,
                    "mips": len(prefilter_levels),
                    "prefilter": prefilter_dict,
                }, f)
            # Pack faces and lut
            np.savez_compressed(
                cache_npz,
                **{f"prefilter_{i}": lvl.faces for i, lvl in enumerate(prefilter_levels)},
                irradiance_faces=irradiance_faces,
                brdf_lut=lut,
                base_faces=base_faces,
            )
        except Exception:
            pass
    else:
        # For meta compute_ms we'll set 0 when using cache
        compute_ms = 0
        # Also need base_faces for env strip. Rebuild cheaply from env to keep code simple
        base_faces, _ = m4.equirect_to_cubemap(env, params.base)

    # Write artifacts
    base_rgb = _faces_to_rgb_u8(m4, base_faces[..., :3])
    irr_rgb = _faces_to_rgb_u8(m4, irradiance_faces)
    strip_base = _make_strip(base_rgb, gutter_px=16)
    strip_irr = _make_strip(irr_rgb, gutter_px=16)

    mips_sheet = m4.build_prefilter_contact_sheet(prefilter_levels)
    lut_img = m4.lut_to_image(lut)

    # Ensure opaque RGB PNGs
    m4.write_png(out / "p4_env_base.png", strip_base)
    m4.write_png(out / "p4_irradiance_cube.png", strip_irr)

    # For specular: contact sheet is already RGB uint8
    m4.write_png(out / "p4_specular_cube_mips.png", mips_sheet)
    m4.write_png(out / "p4_brdf_lut.png", lut_img)

    # Meta JSON (strict keys)
    meta = {
        "base": int(params.base),
        "irr": int(params.irr),
        "brdf": int(params.brdf),
        "mips": int(len(prefilter_levels)),
        "cache_used": bool(cache_used),
        "timings": {"compute_ms": int(compute_ms), "load_ms": int(load_ms)},
    }
    with open(out / "p4_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return meta


if __name__ == "__main__":
    meta = run_p4_acceptance()
    print("P4 acceptance artifacts written:")
    for name in ("p4_env_base.png", "p4_irradiance_cube.png", "p4_specular_cube_mips.png", "p4_brdf_lut.png", "p4_meta.json"):
        print("  ", name)
    print("Meta:", json.dumps(meta))
