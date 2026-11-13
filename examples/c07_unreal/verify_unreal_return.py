"""Check the files returned by the registered-level Unreal exercise."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(job_path, folder):
    job = json.loads(job_path.read_text(encoding="utf-8"))
    returned = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if returned["job_sha256"] != digest(job_path) or returned["run_id"] != folder.name:
        raise ValueError("the result identifies a different job or run directory")
    if returned["status"] not in {"succeeded", "error", "cancelled"}:
        raise ValueError("the result must have a terminal Unreal execution status")
    count = job["frame_count"]
    by_name = {f"frame_{i:08d}.png": i for i in range(count)}
    pngs = {p.name: p for p in folder.glob("*.png") if p.is_file()}
    if any(name not in by_name for name in pngs):
        raise ValueError("unexpected frame names in the run directory")
    if returned["status"] != "succeeded":
        if not returned.get("diagnostic"):
            raise ValueError("an error or cancellation requires the executor diagnostic")
        return {"status": returned["status"], "usable_movie_frames": False, "files": len(pngs)}
    if len(pngs) != count or len(returned["frames"]) != count:
        raise ValueError("successful runs must contain the full frame set")
    seen = set()
    for frame in returned["frames"]:
        name = frame["file"]
        if name in seen or by_name.get(name) != frame["index"]:
            raise ValueError(f"duplicate, missing or misnumbered frame {name}")
        seen.add(name)
        path = pngs[name]
        with Image.open(path) as png:
            png.load()
            if png.format != "PNG" or list(png.size) != job["size"]:
                raise ValueError(f"unexpected image format or size in {name}")
        if digest(path) != frame["sha256"]:
            raise ValueError(f"frame hash mismatch in {name}")
    return {"status": "succeeded", "usable_movie_frames": True, "files": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.job, args.run.resolve()), indent=2))
