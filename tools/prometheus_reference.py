import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/tif/Gore_Range_Albers_1m.tif"
SOURCE_SHA256 = "bcacfdfabefc7ef4e7b9a9cefe12375b491fe1769218ad3c3ddd3ae30f9d93bc"
ARCHIVE_URL = "https://zenodo.org/records/3940482/files/Gore_Range_GeoTIFF.zip?download=1"
ARCHIVE_MD5 = "3df0d15994166512cca17893b7af9dc4"


def prepare_dem():
    import rasterio

    source_bytes = SOURCE.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("Gore Range source differs from the reviewed repository asset")
    cache = ROOT / "tmp/prometheus-slice1/Gore_Range_GeoTIFF.zip"
    if cache.exists():
        archive_bytes = cache.read_bytes()
    else:
        with urllib.request.urlopen(ARCHIVE_URL) as response:
            archive_bytes = response.read()
        if hashlib.md5(archive_bytes).hexdigest() != ARCHIVE_MD5:
            raise ValueError("Gore Range release archive does not match the publisher's checksum")
        with cache.open("xb") as handle:
            handle.write(archive_bytes)
    if hashlib.md5(archive_bytes).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Cached Gore Range archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = [name for name in archive.namelist() if Path(name).name == SOURCE.name and not name.startswith("__MACOSX/")]
        if len(members) != 1:
            raise ValueError(f"Expected one source TIFF in the published archive: {members}")
        member = members[0]
        member_sha256 = hashlib.sha256(archive.read(member)).hexdigest()
    if member_sha256 != SOURCE_SHA256:
        raise ValueError("Repository Gore Range TIFF does not match the published release")
    with rasterio.open(SOURCE) as source:
        if source.shape != (1500, 1500) or source.count != 1 or source.crs is None:
            raise ValueError("Unexpected Gore Range raster layout")
        if tuple(source.transform)[:6] != (1.0, 0.0, -28628.0, 0.0, -1.0, 1858391.0):
            raise ValueError("Unexpected Gore Range georeferencing")
        samples = source.read(1)[::12, ::12].astype(np.float32)
        if not np.isfinite(samples).all() or np.any(samples == source.nodata):
            raise ValueError("Measured fixture contains invalid elevation samples")
        provenance = {
            "schema_version": 1,
            "dataset": "Elevation Models for Reproducible Evaluation of Terrain Representation - Multiscale Models - Gore Range GeoTIFF",
            "dataset_doi": "10.5281/zenodo.3940482",
            "dataset_version": "1.0.0",
            "dataset_url": "https://zenodo.org/records/3940482",
            "elevation_source": "USGS National Elevation Dataset",
            "elevation_source_documentation": "https://zenodo.org/records/3938013",
            "authors": ["Patrick J. Kennelly", "Tom Patterson", "Bernhard Jenny", "Daniel P. Huffman", "Brooke E. Marston", "Sarah Bell", "Alexander M. Tait"],
            "article_doi": "10.1080/15230406.2020.1830856",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
            "source_path": SOURCE.relative_to(ROOT).as_posix(),
            "source_sha256": SOURCE_SHA256,
            "source_archive_url": ARCHIVE_URL,
            "source_archive_md5": ARCHIVE_MD5,
            "source_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "source_archive_member": member,
            "source_archive_member_sha256": member_sha256,
            "source_crs_wkt": source.crs.to_wkt(),
            "source_affine": list(source.transform)[:6],
            "source_shape": list(source.shape),
            "source_nodata": source.nodata,
            "sampling": {"method": "point subsampling", "row_step": 12, "column_step": 12},
            "sample_center_affine": [12.0, 0.0, -28627.5, 0.0, -12.0, 1858390.5],
            "shape": list(samples.shape),
            "dtype": "float32",
            "elevation_units": "metres",
            "vertical_datum": "not specified by the release metadata",
            "changes": "Every twelfth source row and column retained, starting at zero; no interpolation or elevation normalization in the stored fixture.",
            "source_access_date": "2026-09-13",
        }
    payload = io.BytesIO()
    np.save(payload, samples, allow_pickle=False)
    provenance["fixture_sha256"] = hashlib.sha256(payload.getvalue()).hexdigest()
    dest = ROOT / "tests/data/prometheus"
    dest.mkdir(exist_ok=True)
    for name in ("gore_range_dem.npy", "gore_range_dem.json"):
        if (dest / name).exists():
            raise FileExistsError(f"Refusing to replace fixture: {dest / name}")
    with (dest / "gore_range_dem.npy").open("xb") as handle:
        handle.write(payload.getvalue())
    with (dest / "gore_range_dem.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"fixture": str(dest), "fixture_sha256": provenance["fixture_sha256"], "archive_member": member, "source_verified": True}))


def capture_reference(directory, resume_checkpoint=None):
    sys.path.insert(0, str(ROOT / "tests"))
    import test_prometheus_dem_reference as t

    if not t.f3d.has_gpu():
        raise RuntimeError("A hardware GPU is required to capture PROMETHEUS references")
    adapter = t.f3d.device_probe(None)
    from _terrain_runtime import _adapter_is_terrain_safe
    if not _adapter_is_terrain_safe(adapter):
        raise RuntimeError(f"Unqualified GPU adapter: {adapter}")
    names = (*t.REFERENCE_NAMES, "gore_range_scores.json")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (directory / name).exists():
            raise FileExistsError(f"Refusing to replace reference: {directory / name}")
    scene = json.loads(json.dumps(t.scene_metadata()))
    checkpoint_path = Path(resume_checkpoint) if resume_checkpoint else ROOT / "tmp/prometheus-slice1/real_dem_capture.npz"
    if resume_checkpoint:
        checkpoint = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
        if checkpoint["scene"] != scene or checkpoint["fixture_sha256"] != t.sha256(t.DEM_PATH) or checkpoint["provenance_sha256"] != t.sha256(t.PROVENANCE_PATH):
            raise ValueError("Checkpoint inputs differ from this reference scene")
        if checkpoint["arrays_sha256"] != t.sha256(checkpoint_path):
            raise ValueError("Checkpoint arrays checksum mismatch")
        with np.load(checkpoint_path, allow_pickle=False) as arrays:
            out = {**checkpoint["out"], **{key: arrays[key] for key in checkpoint["array_names"]}}
        adapter = checkpoint["adapter"]
        independent = checkpoint["independent_reference"]
    else:
        if checkpoint_path.exists() or checkpoint_path.with_suffix(".json").exists():
            raise FileExistsError("Capture checkpoint already exists; use --resume-checkpoint to reuse its measured radiance")
        dem = t.real_dem()
        out = t.hybrid_render_terrain_reference(dem, t.SIZE, t.SIZE, t.CAM, **t._scene_kwargs(dem))
        t.assert_reference(out)
        other = t.hybrid_render_terrain_reference(dem, t.SIZE, t.SIZE, t.CAM, **{**t._scene_kwargs(dem), "seed": 19, "spp": 8})
        t.assert_reference(other)
        independent = t.independent_reference_metrics(out, other)
        arrays = {key: value for key, value in out.items() if isinstance(value, np.ndarray)}
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays, secondary_radiance=other["radiance"], secondary_luminance_variance=other["luminance_variance"])
        from forge3d import _forge3d as native
        checkpoint = {
            "scene": scene, "fixture_sha256": t.sha256(t.DEM_PATH),
            "provenance_sha256": t.sha256(t.PROVENANCE_PATH),
            "adapter": adapter, "native_extension_sha256": t.sha256(native.__file__),
            "arrays_sha256": t.sha256(checkpoint_path), "array_names": list(arrays),
            "out": {key: value for key, value in out.items() if key not in arrays},
            "independent_reference": independent,
        }
        with checkpoint_path.with_suffix(".json").open("x", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2, allow_nan=False)
        print("PROMETHEUS_CHECKPOINT=" + str(checkpoint_path), flush=True)
    t.assert_reference(out)
    if independent["luminance_mse"] >= t.VARIANCE_THRESHOLD:
        raise ValueError(f"Independent radiance reference failed: {independent}")
    raster = t.raster_metrics_in_subprocess(out, ROOT / "tmp/prometheus-slice1/raster_aligned.npz")
    t._assert_raster_parity_metrics(raster)
    t.f3d.numpy_to_png(str(directory / names[0]), out["rgba"])
    with (directory / names[1]).open("xb") as handle:
        np.savez_compressed(handle, **{key: out[key] for key in ("albedo", "normal", "depth", "radiance", "luminance_variance")})
    scores = {
        "schema_version": 1,
        "fixture_sha256": t.sha256(t.DEM_PATH),
        "provenance_sha256": t.sha256(t.PROVENANCE_PATH),
        "scene": t.scene_metadata(),
        "convergence_metric": out["convergence_metric"],
        "frames": out["frames"],
        "variance": out["variance"],
        "gpu_resource_bytes": out["gpu_resource_bytes"],
        "peak_host_visible_bytes": out["peak_host_visible_bytes"],
        "minmax_pyramid_bytes": out["minmax_pyramid_bytes"],
        "traversal": out["traversal"],
        "adapter": adapter,
        "forge3d_version": t.f3d.__version__,
        "native_extension_sha256": checkpoint["native_extension_sha256"],
        "independent_reference": independent,
        "raster_parity": raster,
        "artifact_sha256": {name: t.sha256(directory / name) for name in t.REFERENCE_NAMES},
        "limitations": ["Estimated sampling variance is not a confidence bound or a proof of transport accuracy.", "Direct sun and environment visibility only; not unrestricted multiple-bounce transport.", "Physical render evidence applies to the recorded adapter/backend only."],
    }
    with (directory / names[2]).open("x", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(scores, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="Explicitly prepare the measured DEM or capture a new PROMETHEUS reference set; never replace existing artifacts.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare-dem", action="store_true")
    choice.add_argument("--capture", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.resume_checkpoint and not args.capture:
        parser.error("--resume-checkpoint requires --capture")
    if args.prepare_dem:
        prepare_dem()
    else:
        capture_reference(args.capture, args.resume_checkpoint)


if __name__ == "__main__":
    main()
