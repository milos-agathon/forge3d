# examples/screenshot_demo.py
# Workstream I3: Screenshot/Record Controls demonstration
# - Shows EXIF metadata embedding in PNG
# - Demonstrates frame dumper for animation sequences
# - Validates metadata can be extracted

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import forge3d as f3d
from forge3d.helpers.offscreen import save_png_with_exif, render_offscreen_rgba
from forge3d.helpers.frame_dump import FrameDumper


def demo_exif_metadata(output_path: Path) -> None:
    """Demonstrate screenshot with EXIF metadata."""
    print(f"\n=== EXIF Metadata Demo ===")

    # Render a simple frame
    rgba = render_offscreen_rgba(800, 600, seed=42, frames=1)

    # Metadata describing the camera and exposure settings
    metadata = {
        "camera": {
            "eye": [10.0, 20.0, 30.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 45.0
        },
        "exposure": {
            "mode": "ACES",
            "stops": 0.5,
            "gamma": 2.2
        },
        "description": "forge3d screenshot demo",
        "software": f"forge3d {f3d.__version__}"
    }

    # Save with metadata
    save_png_with_exif(str(output_path), rgba, metadata)
    print(f"Saved: {output_path}")

    # Verify metadata (requires PIL)
    try:
        from PIL import Image
        img = Image.open(output_path)
        print(f"  Image size: {img.size}")
        print(f"  Image mode: {img.mode}")

        # Extract text chunks
        if hasattr(img, 'text') and img.text:
            print(f"  Metadata fields:")
            for key, value in sorted(img.text.items()):
                if key.startswith("forge3d:"):
                    print(f"    {key}: {value}")
        else:
            print("  Warning: No metadata found (PIL version may not support PngInfo extraction)")

    except Exception as exc:
        print(f"  Metadata extraction failed: {exc}")


def demo_frame_dump(output_dir: Path, num_frames: int = 10) -> None:
    """Demonstrate frame sequence dumping."""
    print(f"\n=== Frame Dump Demo ===")

    dumper = FrameDumper(output_dir=output_dir, prefix="demo")
    dumper.start_recording()

    print(f"Recording {num_frames} frames to {output_dir}/...")

    for i in range(num_frames):
        # Render frame with varying seed
        rgba = render_offscreen_rgba(640, 480, seed=i + 1, frames=1)
        path = dumper.capture_frame(rgba)

        if i % 5 == 0:
            print(f"  Captured: {path.name}")

    frame_count = dumper.stop_recording()
    print(f"Recorded {frame_count} frames")

    # Verify files
    frames = sorted(output_dir.glob("demo_*.png"))
    print(f"Verified {len(frames)} frame files:")
    if len(frames) <= 10:
        for frame in frames:
            print(f"  {frame.name}")
    else:
        print(f"  {frames[0].name} ... {frames[-1].name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="forge3d screenshot/record demo")
    parser.add_argument("--exif", action="store_true", help="Demo EXIF metadata")
    parser.add_argument("--record", action="store_true", help="Demo frame recording")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to record")
    parser.add_argument("--out", type=Path, default=Path("reports/screenshot_exif.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/frames"))

    args = parser.parse_args()

    # Default: run both demos if none specified
    run_exif = args.exif or (not args.exif and not args.record)
    run_record = args.record or (not args.exif and not args.record)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if run_exif:
        demo_exif_metadata(args.out)

    if run_record:
        demo_frame_dump(args.out_dir, args.frames)

    print("\n=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
