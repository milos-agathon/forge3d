"""Encode the existing CHRONOS PNG sequence without modifying source frames."""

import shutil
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Literal


def encode_video(
    frames_dir: Path, output_path: Path, fps: int, *, transfer: Literal["bt709", "iec61966-2-1"],
    frame_count: int | None = None,
) -> None:
    names = {path.name for path in frames_dir.glob("frame_*.png") if path.is_file()}
    count = len(names) if frame_count is None else frame_count
    if count <= 0 or names != {f"frame_{i:08d}.png" for i in range(count)}:
        raise ValueError("encoding requires the complete contiguous zero-based PNG frame set")
    if fps <= 0:
        raise ValueError("encoding requires a positive frame rate")
    if transfer not in ("bt709", "iec61966-2-1") or output_path.suffix.lower() != ".mp4":
        raise ValueError("encoding requires an MP4 target and a supported transfer tag")

    # The PNGs and their CHRONOS provenance stay byte-identical. Filters
    # apply only to the video stream.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the flyover MP4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".encode-", dir=output_path.parent) as staging:
        staged = Path(staging) / output_path.name
        subprocess.run([
            ffmpeg,
            "-y",
            "-xerror",
            "-loglevel",
            "error",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%08d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            # Convert the RGB matrix to BT.709. The caller declares the
            # transfer tag on both the filtered frames and the encoder.
            "-vf",
            "scale=out_color_matrix=bt709,"
            "setparams=range=limited:color_primaries=bt709:"
            f"color_trc={transfer}:colorspace=bt709",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            transfer,
            "-movflags",
            "+faststart",
            str(staged),
        ], check=True)
        staged.replace(output_path)
