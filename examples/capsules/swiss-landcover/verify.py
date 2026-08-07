#!/usr/bin/env python3
"""Swiss land cover — Verified Map Capsule verifier.

`python verify.py` is the single acceptance command for this capsule. It:

1. validates `data-manifest.json` and `provenance.json` are well-formed;
2. recomputes the sha256 of both fetched datasets, of `out/swiss_landcover.png`,
   of `recipe.py` and of `out/local.certificate.json`, and cross-checks them
   against `out/recipe_result.json` (written by the recipe) and against the
   committed capsule-root `recipe_result.json` reference binding when present;
3. compares `out/swiss_landcover.png` against `expected/reference.png` — the
   canonical lossless reference, never a lossy web derivative — with RGB SSIM
   plus a mean-absolute-difference cap, using the thresholds pinned in
   `verify_config.json`;
4. gates on the certificate fields that describe what actually ran: engine
   version, per-module WGSL hashes and an empty degradation list. Adapter
   identity and timings are printed but never gate;
5. writes machine-readable `out/verify_result.json` and exits nonzero if any
   check failed.

This tool is standalone by design: stdlib + numpy + PIL, plus `forge3d.datasets`
in `main()` only to resolve the two dataset paths exactly as the recipe does.
It never mutates a certificate, never contacts the network beyond that dataset
resolution, and reports nothing anywhere.

Trust labels are stated plainly and are not upgraded by a passing run: the
reference certificate is development-signed, not production-signed (v1), and
this report is locally produced.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

VERIFY_PATH = Path(__file__).resolve()
CAPSULE_DIR = VERIFY_PATH.parent

CONFIG_NAME = "verify_config.json"
DATA_MANIFEST_NAME = "data-manifest.json"
PROVENANCE_NAME = "provenance.json"
RECIPE_NAME = "recipe.py"
BINDING_NAME = "recipe_result.json"
SNAPSHOT_NAME = "swiss_landcover.png"
LOCAL_CERT_NAME = "local.certificate.json"
REFERENCE_PNG_NAME = "reference.png"
REFERENCE_CERT_NAME = "reference.certificate.json"
VERIFY_RESULT_NAME = "verify_result.json"

EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

# Honesty labels (spec section 6): a passing verification says the local run
# matched the reference, not that anyone signed anything.
TRUST_REFERENCE = "development-signed — not production-signed (v1)"
TRUST_LOCAL = "locally produced"

# Datasets are resolved through the registry with the same calls as the recipe
# (recipe.py: f3d.datasets.fetch_dem("swiss") / f3d.datasets.fetch("swiss-land-cover")),
# so the bytes hashed here are the bytes the recipe consumed.
DATASET_NAMES = ("swiss", "swiss-land-cover")

DATA_MANIFEST_KEYS = ("capsule", "version", "datasets")
DATASET_ENTRY_KEYS = ("name", "file", "sha256", "fetch")
PROVENANCE_RECORD_KEYS = (
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
)
RECIPE_RESULT_KEYS = (
    "png_sha256",
    "inputs",
    "recipe_sha256",
    "certificate_sha256",
    "forge3d_version",
    "created",
)
RECIPE_RESULT_INPUT_KEYS = ("name", "path_name", "sha256")

# Text artifacts hashed newline-normalized so LF/CRLF checkouts agree;
# rasters/PNGs hashed raw. Kept byte-identical to recipe.py::_sha256.
TEXT_SUFFIXES = frozenset({".py", ".json", ".md", ".txt", ".cfg", ".ini", ".toml", ".yaml", ".yml"})


def sha256_file(path) -> str:
    """sha256 of a capsule artifact.

    Text artifacts hashed newline-normalized so LF/CRLF checkouts agree;
    rasters/PNGs hashed raw. Without this a fresh Windows clone (core.autocrlf
    true) would hash `recipe.py` differently from the LF tree that produced the
    binding manifest, and every reproduction would report a false mismatch.
    """
    path = Path(path)
    if path.suffix.lower() in TEXT_SUFFIXES:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# SSIM — copied VERBATIM from tests/_ssim.py (forge3d repo, the golden-image
# policy this capsule adopts, spec section 8). Do not "improve" it here: the
# capsule test suite asserts bit-level agreement with that module, so any edit
# must be made there first.
# ---------------------------------------------------------------------------


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian window for SSIM computation."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    return np.outer(g, g)


def ssim(img1: np.ndarray, img2: np.ndarray,
         data_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03,
         win_size: int = 11,
         sigma: float = 1.5) -> float:
    """
    Compute SSIM between two images.

    Args:
        img1, img2: Images to compare, same shape
        data_range: Maximum possible pixel value difference (255 for uint8)
        k1, k2: SSIM constants (default values from paper)
        win_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian window

    Returns:
        SSIM value between -1 and 1 (1 = identical)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    if img1.ndim != 2 and img1.ndim != 3:
        raise ValueError(f"Images must be 2D or 3D, got {img1.ndim}D")

    # Convert to float64 for computation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # For 3D images (H,W,C), compute SSIM per channel and average
    if img1.ndim == 3:
        ssim_vals = []
        for c in range(img1.shape[2]):
            ssim_vals.append(ssim(img1[:, :, c], img2[:, :, c], data_range, k1, k2, win_size, sigma))
        return np.mean(ssim_vals)

    # SSIM constants
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Create Gaussian window
    window = _gaussian_window(win_size, sigma)

    # Compute local means
    mu1 = _filter2d(img1, window)
    mu2 = _filter2d(img2, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = _filter2d(img1 ** 2, window) - mu1_sq
    sigma2_sq = _filter2d(img2 ** 2, window) - mu2_sq
    sigma12 = _filter2d(img1 * img2, window) - mu1_mu2

    # Compute SSIM map
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    # Return mean SSIM
    return float(np.mean(ssim_map))


def _filter2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution using scipy.ndimage if available, otherwise manual."""
    try:
        from scipy.ndimage import convolve
        return convolve(img, kernel, mode='constant', cval=0.0)
    except ImportError:
        # Fallback to manual convolution (slower but dependency-free)
        return _manual_convolve2d(img, kernel)


def _manual_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2D convolution implementation."""
    kh, kw = kernel.shape
    ih, iw = img.shape

    # Calculate padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad image
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Output array
    output = np.zeros_like(img)

    # Convolution
    for i in range(ih):
        for j in range(iw):
            output[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)

    return output


# --------------------------- end verbatim block ----------------------------


def certificate_verdicts(local_cert, ref_cert) -> dict:
    """The three mandatory certificate gates, in one place.

    `compare()` and the report both read them from here so a local run can
    never be told two different stories about the same certificate.
    """
    le, re_ = local_cert.get("engine", {}) or {}, ref_cert.get("engine", {}) or {}
    return {
        "engine_match": bool(le.get("version") == re_.get("version")),
        "wgsl_match": bool(le.get("wgsl_module_hashes") == re_.get("wgsl_module_hashes")),
        "degradations_empty": bool(not local_cert.get("degradations")),
    }


def compare(local_img, ref_img, local_cert, ref_cert, config) -> dict:
    """Compare one local render + certificate against the capsule reference.

    Gating: SSIM, mean absolute difference, engine version, WGSL module hashes,
    and an empty local degradation list. Adapter identity and timings are
    deliberately absent — they describe the machine, not the render contract.
    """
    if config.get("ssim_min") is None or config.get("mean_abs_max") is None:
        raise ValueError("thresholds not yet calibrated (verify_config.json)")
    local = np.asarray(local_img, dtype=np.float64)[..., :3]
    ref = np.asarray(ref_img, dtype=np.float64)[..., :3]
    if local.shape != ref.shape:
        return {"passed": False, "error": f"shape {local.shape} != reference {ref.shape}"}
    score = ssim(local, ref, data_range=255.0)
    mean_abs = float(np.mean(np.abs(local - ref)))
    result = {
        "ssim": float(score),
        "mean_abs": mean_abs,
        "ssim_ok": bool(score >= config["ssim_min"]),
        "mean_abs_ok": bool(mean_abs <= config["mean_abs_max"]),
        **certificate_verdicts(local_cert, ref_cert),
    }
    result["passed"] = all(result[k] for k in
        ("ssim_ok", "mean_abs_ok", "engine_match", "wgsl_match", "degradations_empty"))
    return result


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value.lower()
    )


def _missing_keys(obj, keys) -> list:
    return [key for key in keys if key not in obj]


def validate_data_manifest(manifest) -> list[str]:
    """Return a list of problems; empty means well-formed."""
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return [f"data-manifest.json must be an object, got {type(manifest).__name__}"]
    for key in _missing_keys(manifest, DATA_MANIFEST_KEYS):
        problems.append(f"missing top-level key: {key}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        problems.append("datasets must be a non-empty list")
        return problems
    for index, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            problems.append(f"datasets[{index}] must be an object")
            continue
        label = entry.get("name", f"datasets[{index}]")
        for key in _missing_keys(entry, DATASET_ENTRY_KEYS):
            problems.append(f"{label}: missing key {key}")
        if "sha256" in entry and not _is_sha256(entry["sha256"]):
            problems.append(f"{label}: sha256 is not a 64-char hex digest")
    return problems


def validate_provenance(provenance) -> list[str]:
    """Return a list of problems; empty means well-formed."""
    problems: list[str] = []
    if not isinstance(provenance, dict):
        return [f"provenance.json must be an object, got {type(provenance).__name__}"]
    records = provenance.get("records")
    if not isinstance(records, list) or not records:
        return ["records must be a non-empty list"]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            problems.append(f"records[{index}] must be an object")
            continue
        label = record.get("dataset", f"records[{index}]")
        for key in _missing_keys(record, PROVENANCE_RECORD_KEYS):
            problems.append(f"{label}: missing key {key}")
        if "distributed_sha256" in record and not _is_sha256(record["distributed_sha256"]):
            problems.append(f"{label}: distributed_sha256 is not a 64-char hex digest")
    return problems


def validate_recipe_result(result) -> list[str]:
    """Return a list of problems; empty means well-formed."""
    problems: list[str] = []
    if not isinstance(result, dict):
        return [f"{BINDING_NAME} must be an object, got {type(result).__name__}"]
    for key in _missing_keys(result, RECIPE_RESULT_KEYS):
        problems.append(f"missing key: {key}")
    for key in ("png_sha256", "recipe_sha256", "certificate_sha256"):
        if key in result and not _is_sha256(result[key]):
            problems.append(f"{key} is not a 64-char hex digest")
    inputs = result.get("inputs")
    if not isinstance(inputs, list):
        problems.append("inputs must be a list")
        return problems
    for index, entry in enumerate(inputs):
        if not isinstance(entry, dict):
            problems.append(f"inputs[{index}] must be an object")
            continue
        label = entry.get("name", f"inputs[{index}]")
        for key in _missing_keys(entry, RECIPE_RESULT_INPUT_KEYS):
            problems.append(f"inputs[{label}]: missing key {key}")
        if "sha256" in entry and not _is_sha256(entry["sha256"]):
            problems.append(f"inputs[{label}]: sha256 is not a 64-char hex digest")
    return problems


def thresholds_calibrated(config) -> bool:
    """True once Task 4's calibration has pinned both thresholds."""
    if not isinstance(config, dict):
        return False
    return isinstance(config.get("ssim_min"), (int, float)) and isinstance(
        config.get("mean_abs_max"), (int, float)
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

PASS, FAIL, ABSENT, NOT_EVALUATED = "pass", "fail", "absent", "not-evaluated"
_STATUS_OK = {PASS: True, FAIL: False, ABSENT: None, NOT_EVALUATED: None}
# `absent` is genuinely not applicable and does not gate. `not-evaluated` means
# we could not measure the check, which is never reported as success.
_GATING_FAILURE = {FAIL, NOT_EVALUATED}


def _read_json(path: Path):
    """Return (payload, error-string). Missing/corrupt files never raise."""
    if not path.is_file():
        return None, f"missing file: {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable {path.name}: {exc}"


def _read_rgb(path: Path):
    """Return (array, error-string) for an image on disk."""
    if not path.is_file():
        return None, f"missing file: {path.name}"
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8), None
    except (OSError, ValueError) as exc:
        return None, f"unreadable {path.name}: {exc}"


def _certificate_facts(certificate) -> dict:
    """Informational-only projection of a certificate. Never gates."""
    if not isinstance(certificate, dict):
        return {}
    passes = certificate.get("passes") or []
    gpu_ms = [float(entry.get("gpu_ms", 0.0)) for entry in passes if isinstance(entry, dict)]
    engine = certificate.get("engine") or {}
    return {
        "adapter": certificate.get("adapter") or {},
        "engine_version": engine.get("version"),
        "engine_git_sha": engine.get("git_sha"),
        "passes": len(passes),
        "total_gpu_ms": round(sum(gpu_ms), 4),
    }


def _wgsl_detail(local_wgsl, ref_wgsl) -> str:
    if local_wgsl == ref_wgsl:
        return f"{len(local_wgsl or {})} modules identical"
    if not isinstance(local_wgsl, dict) or not isinstance(ref_wgsl, dict):
        return f"malformed wgsl_module_hashes: local {type(local_wgsl).__name__}, " \
               f"reference {type(ref_wgsl).__name__}"
    differing = sorted(
        key for key in set(local_wgsl) | set(ref_wgsl) if local_wgsl.get(key) != ref_wgsl.get(key)
    )
    return "differing modules: " + ", ".join(differing)


def _degradation_detail(degradations) -> str:
    if not degradations:
        return "none"
    summaries = []
    for item in degradations:
        text = item.get("consequence", item) if isinstance(item, dict) else item
        summaries.append(str(text).splitlines()[0][:120] if str(text) else "<empty>")
    return f"{len(degradations)} recorded: " + "; ".join(summaries)


def verify(capsule_dir, *, config, dataset_paths, write: bool = True) -> dict:
    """Run every capsule check and return the machine-readable report.

    `dataset_paths` maps registry name -> local file, resolved by the caller
    (`main()` uses forge3d.datasets, tests use fixtures) so this function stays
    offline and GPU-free.
    """
    if not thresholds_calibrated(config):
        raise ValueError(f"thresholds not yet calibrated ({CONFIG_NAME})")
    capsule_dir = Path(capsule_dir)
    out_dir = capsule_dir / "out"
    expected_dir = capsule_dir / "expected"
    checks: dict[str, dict] = {}

    def record(name: str, status: str, detail: str) -> None:
        checks[name] = {"ok": _STATUS_OK[status], "status": status, "detail": detail}

    def record_bool(name: str, ok: bool, detail: str) -> None:
        record(name, PASS if ok else FAIL, detail)

    # --- 1. manifests ------------------------------------------------------
    manifest, manifest_error = _read_json(capsule_dir / DATA_MANIFEST_NAME)
    manifest_problems = [manifest_error] if manifest_error else validate_data_manifest(manifest)
    record_bool(
        "data_manifest_valid",
        not manifest_problems,
        "; ".join(manifest_problems)
        or f"required keys present ({len(manifest['datasets'])} datasets)",
    )

    provenance, provenance_error = _read_json(capsule_dir / PROVENANCE_NAME)
    provenance_problems = (
        [provenance_error] if provenance_error else validate_provenance(provenance)
    )
    record_bool(
        "provenance_valid",
        not provenance_problems,
        "; ".join(provenance_problems)
        or f"required keys present ({len(provenance['records'])} records)",
    )

    result, result_error = _read_json(out_dir / BINDING_NAME)
    result_problems = [result_error] if result_error else validate_recipe_result(result)
    record_bool(
        "recipe_result_valid",
        not result_problems,
        "; ".join(result_problems) or "required keys present",
    )
    if result_problems:
        result = None

    # --- 2. recomputed hashes vs the run's binding manifest ----------------
    dataset_hashes: dict[str, str] = {}
    dataset_problems: list[str] = []
    for name in DATASET_NAMES:
        path = (dataset_paths or {}).get(name)
        if path is None or not Path(path).is_file():
            dataset_problems.append(f"{name}: dataset file not available")
            continue
        dataset_hashes[name] = sha256_file(path)

    if dataset_problems and not dataset_hashes:
        record("input_hashes_match", NOT_EVALUATED, "; ".join(dataset_problems))
    else:
        problems = list(dataset_problems)
        pinned = {}
        if not manifest_problems:
            pinned = {entry["name"]: entry["sha256"] for entry in manifest["datasets"]}
            unchecked = sorted(set(pinned) - set(DATASET_NAMES))
            if unchecked:
                problems.append(
                    f"{DATA_MANIFEST_NAME} lists datasets this verifier does not "
                    f"resolve: {', '.join(unchecked)}"
                )
        recorded = {}
        if result is not None:
            recorded = {entry["name"]: entry["sha256"] for entry in result["inputs"]}
        for name, digest in dataset_hashes.items():
            if pinned and pinned.get(name) != digest:
                problems.append(
                    f"{name}: fetched bytes {digest[:16]}... != pinned "
                    f"{str(pinned.get(name))[:16]}... in {DATA_MANIFEST_NAME}"
                )
            if recorded and recorded.get(name) != digest:
                problems.append(
                    f"{name}: fetched bytes {digest[:16]}... != {BINDING_NAME} "
                    f"{str(recorded.get(name))[:16]}..."
                )
            if not pinned and not recorded:
                problems.append(f"{name}: nothing to compare against (manifests invalid)")
        record_bool(
            "input_hashes_match",
            not problems,
            "; ".join(problems)
            or ", ".join(f"{name} {digest[:16]}..." for name, digest in dataset_hashes.items()),
        )

    def _hash_check(name: str, path: Path, expected_key: str) -> None:
        if not path.is_file():
            record(name, NOT_EVALUATED, f"missing file: {path.name}")
            return
        if result is None:
            record(name, NOT_EVALUATED, f"{BINDING_NAME} invalid or missing")
            return
        digest = sha256_file(path)
        expected = result[expected_key]
        record_bool(
            name,
            digest == expected,
            f"{path.name} {digest[:16]}..."
            + ("" if digest == expected else f" != {BINDING_NAME} {expected[:16]}..."),
        )

    _hash_check("png_hash_matches", out_dir / SNAPSHOT_NAME, "png_sha256")
    _hash_check("recipe_hash_matches", capsule_dir / RECIPE_NAME, "recipe_sha256")
    _hash_check("certificate_hash_matches", out_dir / LOCAL_CERT_NAME, "certificate_sha256")

    # --- 3. committed reference binding (present from Task 4 onward) -------
    binding_path = capsule_dir / BINDING_NAME
    reference_png = expected_dir / REFERENCE_PNG_NAME
    if not binding_path.is_file():
        record(
            "reference_binding_matches",
            ABSENT,
            f"no committed {BINDING_NAME} at the capsule root",
        )
    else:
        binding, binding_error = _read_json(binding_path)
        problems = [binding_error] if binding_error else validate_recipe_result(binding)
        if not problems:
            bound_inputs = {entry["name"]: entry["sha256"] for entry in binding["inputs"]}
            for name, digest in dataset_hashes.items():
                if bound_inputs.get(name) != digest:
                    problems.append(
                        f"{name}: fetched bytes differ from the committed reference binding"
                    )
            for name in DATASET_NAMES:
                if name not in dataset_hashes:
                    problems.append(f"{name}: not hashed, cannot check reference binding")
            recipe_path = capsule_dir / RECIPE_NAME
            if not recipe_path.is_file():
                problems.append(f"missing {RECIPE_NAME}")
            elif sha256_file(recipe_path) != binding["recipe_sha256"]:
                problems.append("recipe.py differs from the committed reference binding")
            if not reference_png.is_file():
                problems.append(f"missing expected/{REFERENCE_PNG_NAME}")
            elif sha256_file(reference_png) != binding["png_sha256"]:
                problems.append(
                    f"expected/{REFERENCE_PNG_NAME} differs from the committed reference binding"
                )
        record_bool(
            "reference_binding_matches",
            not problems,
            "; ".join(problems) or "inputs + recipe + reference PNG bound",
        )

    # --- 4. image + certificate comparison ---------------------------------
    local_img, local_img_error = _read_rgb(out_dir / SNAPSHOT_NAME)
    ref_img, ref_img_error = _read_rgb(reference_png)
    local_cert, local_cert_error = _read_json(out_dir / LOCAL_CERT_NAME)
    ref_cert, ref_cert_error = _read_json(expected_dir / REFERENCE_CERT_NAME)

    image_error = "; ".join(error for error in (local_img_error, ref_img_error) if error)
    cert_error = "; ".join(error for error in (local_cert_error, ref_cert_error) if error)
    image: dict
    if image_error or cert_error:
        image = {"error": "; ".join(part for part in (image_error, cert_error) if part)}
    else:
        image = compare(local_img, ref_img, local_cert, ref_cert, config)

    if "error" in image:
        for name in ("image_ssim", "image_mean_abs"):
            record(name, NOT_EVALUATED, image["error"])
    else:
        record_bool(
            "image_ssim",
            image["ssim_ok"],
            f"ssim {image['ssim']:.6f} vs min {float(config['ssim_min']):.6f}",
        )
        record_bool(
            "image_mean_abs",
            image["mean_abs_ok"],
            f"mean_abs {image['mean_abs']:.4f} vs max {float(config['mean_abs_max']):.4f}",
        )

    if cert_error:
        for name in ("engine_version_match", "wgsl_module_hashes_match", "degradations_empty"):
            record(name, NOT_EVALUATED, cert_error)
    else:
        # One source of truth: whatever compare() decided, or the same helper
        # when compare() could not run because an image was unreadable.
        verdicts = image if "error" not in image else certificate_verdicts(local_cert, ref_cert)
        local_engine = local_cert.get("engine") or {}
        ref_engine = ref_cert.get("engine") or {}
        record_bool(
            "engine_version_match",
            verdicts["engine_match"],
            f"local {local_engine.get('version')} vs reference {ref_engine.get('version')}",
        )
        record_bool(
            "wgsl_module_hashes_match",
            verdicts["wgsl_match"],
            _wgsl_detail(local_engine.get("wgsl_module_hashes"),
                         ref_engine.get("wgsl_module_hashes")),
        )
        record_bool(
            "degradations_empty",
            verdicts["degradations_empty"],
            _degradation_detail(local_cert.get("degradations") or []),
        )

    failed = [name for name, check in checks.items() if check["status"] in _GATING_FAILURE]
    payload = {
        "capsule": capsule_dir.name,
        "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {
            "ssim_min": config.get("ssim_min"),
            "mean_abs_max": config.get("mean_abs_max"),
        },
        "checks": checks,
        "image": image,
        "informational": {
            "local": _certificate_facts(local_cert),
            "reference": _certificate_facts(ref_cert),
            "note": "adapter identity and timings are reported, never gated",
        },
        "trust": {"reference_certificate": TRUST_REFERENCE, "local_report": TRUST_LOCAL},
        "failed_checks": failed,
        "passed": not failed,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / VERIFY_RESULT_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# UNRUN is not a pass: a check we could not measure gates exactly like a
# failure, but the marker says which of the two actually happened.
_MARKERS = {PASS: "PASS", FAIL: "FAIL", ABSENT: "SKIP", NOT_EVALUATED: "UNRUN"}


def _print_report(payload: dict, capsule_dir: Path) -> None:
    print(f"== forge3d verified map capsule: {payload['capsule']} ==")
    print(f"capsule:   {capsule_dir}")
    print(f"reference: expected/{REFERENCE_PNG_NAME} (canonical lossless)")
    print(f"local:     out/{SNAPSHOT_NAME}")
    print()
    width = max(len(name) for name in payload["checks"])
    for name, check in payload["checks"].items():
        print(f"[{_MARKERS[check['status']]}] {name.ljust(width)}  {check['detail']}")
    print()
    print("-- informational (never gating) --")
    for side in ("local", "reference"):
        facts = payload["informational"][side]
        adapter = facts.get("adapter") or {}
        print(
            f"       {side:<10} adapter: {adapter.get('device', 'unknown')} / "
            f"{adapter.get('backend', 'unknown')} (driver {adapter.get('driver_info', '?')}), "
            f"{facts.get('passes', 0)} passes, {facts.get('total_gpu_ms', 0.0)} ms GPU"
        )
    print()
    print("-- trust labels --")
    print(f"       reference certificate: {payload['trust']['reference_certificate']}")
    print(f"       local report:          {payload['trust']['local_report']}")
    print()
    if payload["passed"]:
        print(f"RESULT: PASS — {len(payload['checks'])} checks, all satisfied")
    else:
        print(
            f"RESULT: FAIL — {len(payload['failed_checks'])} of {len(payload['checks'])} "
            f"checks did not pass: {', '.join(payload['failed_checks'])}"
        )
    print(f"wrote {(capsule_dir / 'out' / VERIFY_RESULT_NAME)}")


def main() -> int:
    capsule_dir = CAPSULE_DIR
    out_dir = capsule_dir / "out"
    # A stale report must never survive a refused or failed run.
    (out_dir / VERIFY_RESULT_NAME).unlink(missing_ok=True)

    config, config_error = _read_json(capsule_dir / CONFIG_NAME)
    if config_error:
        print(f"ERROR: {config_error}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    if not thresholds_calibrated(config):
        print(
            f"ERROR: thresholds not yet calibrated — {CONFIG_NAME} still has null "
            "ssim_min/mean_abs_max. This capsule cannot be verified until the "
            "reference render pins them.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN

    required = [
        capsule_dir / RECIPE_NAME,
        capsule_dir / DATA_MANIFEST_NAME,
        capsule_dir / PROVENANCE_NAME,
        capsule_dir / "expected" / REFERENCE_PNG_NAME,
        capsule_dir / "expected" / REFERENCE_CERT_NAME,
        out_dir / SNAPSHOT_NAME,
        out_dir / LOCAL_CERT_NAME,
        out_dir / BINDING_NAME,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print(
            "ERROR: cannot verify — missing "
            + ", ".join(str(path.relative_to(capsule_dir)) for path in missing)
            + ". Run `python recipe.py` first.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN

    try:
        import forge3d as f3d
    except ImportError as exc:  # pragma: no cover - environment problem, not a verdict
        print(f"ERROR: forge3d is not importable ({exc}); cannot resolve datasets.",
              file=sys.stderr)
        return EXIT_CANNOT_RUN
    print("== Resolving pinned datasets (sha256-checked by the registry) ==")
    dataset_paths = {
        "swiss": Path(f3d.datasets.fetch_dem("swiss")),
        "swiss-land-cover": Path(f3d.datasets.fetch("swiss-land-cover")),
    }

    payload = verify(capsule_dir, config=config, dataset_paths=dataset_paths)
    _print_report(payload, capsule_dir)
    return 0 if payload["passed"] else EXIT_FAIL


if __name__ == "__main__":
    try:  # Windows consoles vary; keep the labels readable rather than fatal.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    raise SystemExit(main())
