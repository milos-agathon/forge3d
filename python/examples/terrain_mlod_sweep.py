#!/usr/bin/env python3
"""
Sweep multi-LOD pixel error budgets and profile render time vs. quality.
Generates per-frame CSVs using terrain_e4_stress_test.py, a summary CSV, and charts (if matplotlib available).
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
import csv
import math
from typing import List, Tuple


def parse_budgets(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            raise SystemExit(f"Invalid budget value: {p}")
    if not out:
        raise SystemExit("No budgets specified")
    return out


def run_e4(
    budget: float,
    width: int,
    height: int,
    world: float,
    lod: int,
    tile_px: int,
    mosaic_x: int,
    mosaic_y: int,
    steps: int,
    max_uploads: int,
    overlay_alpha: float,
    mlod_max_uploads: int,
    mlod_freq: int,
    csv_path: str,
) -> int:
    cmd = [
        sys.executable,
        'python/examples/terrain_e4_stress_test.py',
        '--width', str(width),
        '--height', str(height),
        '--world', str(world),
        '--lod', str(lod),
        '--tile-px', str(tile_px),
        '--mosaic-x', str(mosaic_x),
        '--mosaic-y', str(mosaic_y),
        '--steps', str(steps),
        '--max-uploads', str(max_uploads),
        '--colormap', 'viridis',
        '--mlod',
        '--mlod-freq', str(mlod_freq),
        '--mlod-pixel-error', str(budget),
        '--mlod-max-uploads', str(mlod_max_uploads),
        '--overlay',
        '--overlay-alpha', str(overlay_alpha),
        '--csv', csv_path,
    ]
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    return proc.returncode


def read_frame_csv(csv_path: str) -> Tuple[List[float], List[float]]:
    """Return (mlod_ms, stream_ms) lists from per-frame CSV; ignore empty cells."""
    mlod_ms: List[float] = []
    stream_ms: List[float] = []
    with open(csv_path, 'r', newline='') as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row.get('mlod_ms'):
                try:
                    mlod_ms.append(float(row['mlod_ms']))
                except Exception:
                    pass
            if row.get('stream_ms'):
                try:
                    stream_ms.append(float(row['stream_ms']))
                except Exception:
                    pass
    return (mlod_ms, stream_ms)


def percentiles(a: List[float]) -> Tuple[float, float, float, float, float]:
    if not a:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    import numpy as np
    arr = np.asarray(a, dtype=np.float64)
    return (
        float(arr.min()),
        float(np.percentile(arr, 50)),
        float(arr.mean()),
        float(np.percentile(arr, 95)),
        float(arr.max()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description='Sweep multi-LOD pixel error and chart performance')
    ap.add_argument('--budgets', type=str, default='0.5,1.0,1.5,2.0,3.0')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--world', type=float, default=16384.0)
    ap.add_argument('--lod', type=int, default=6)
    ap.add_argument('--tile-px', type=int, default=64)
    ap.add_argument('--mosaic-x', type=int, default=32)
    ap.add_argument('--mosaic-y', type=int, default=32)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--max-uploads', type=int, default=64)
    ap.add_argument('--overlay-alpha', type=float, default=0.85)
    ap.add_argument('--mlod-max-uploads', type=int, default=64)
    ap.add_argument('--mlod-freq', type=int, default=10)
    ap.add_argument('--outdir', type=str, default='sweep_out')
    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    summary_rows: List[List[str]] = []
    summary_header = [
        'pixel_error',
        'mlod_ms_min', 'mlod_ms_p50', 'mlod_ms_mean', 'mlod_ms_p95', 'mlod_ms_max',
        'stream_ms_min', 'stream_ms_p50', 'stream_ms_mean', 'stream_ms_p95', 'stream_ms_max',
        'frames',
    ]

    per_budget_csvs: List[str] = []

    for b in budgets:
        csv_name = f'frames_peb_{str(b).replace(".", "p")}.csv'
        csv_path = os.path.join(outdir, csv_name)
        rc = run_e4(
            budget=b,
            width=args.width,
            height=args.height,
            world=args.world,
            lod=args.lod,
            tile_px=args.tile_px,
            mosaic_x=args.mosaic_x,
            mosaic_y=args.mosaic_y,
            steps=args.steps,
            max_uploads=args.max_uploads,
            overlay_alpha=args.overlay_alpha,
            mlod_max_uploads=args.mlod_max_uploads,
            mlod_freq=args.mlod_freq,
            csv_path=csv_path,
        )
        if rc != 0:
            print(f"Run failed for pixel_error={b} with code {rc}")
            continue
        per_budget_csvs.append(csv_path)
        mlod_ms, stream_ms = read_frame_csv(csv_path)
        m_min, m_p50, m_mean, m_p95, m_max = percentiles(mlod_ms)
        s_min, s_p50, s_mean, s_p95, s_max = percentiles(stream_ms)
        summary_rows.append([
            f"{b}",
            f"{m_min:.3f}", f"{m_p50:.3f}", f"{m_mean:.3f}", f"{m_p95:.3f}", f"{m_max:.3f}",
            f"{s_min:.3f}", f"{s_p50:.3f}", f"{s_mean:.3f}", f"{s_p95:.3f}", f"{s_max:.3f}",
            str(args.steps),
        ])

    summary_csv = os.path.join(outdir, 'sweep_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(summary_header)
        wr.writerows(summary_rows)

    # Try to plot charts (optional)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        if summary_rows:
            x = np.array([float(r[0]) for r in summary_rows], dtype=np.float32)
            y_mean = np.array([float(r[3]) for r in summary_rows], dtype=np.float32)  # mlod mean
            y_p50 = np.array([float(r[2]) for r in summary_rows], dtype=np.float32)
            y_p95 = np.array([float(r[4]) for r in summary_rows], dtype=np.float32)
            order = np.argsort(x)
            x = x[order]
            y_mean = y_mean[order]
            y_p50 = y_p50[order]
            y_p95 = y_p95[order]

            plt.figure(figsize=(7,5))
            plt.plot(x, y_mean, 'o-', label='mlod mean (ms)')
            plt.plot(x, y_p50, 's--', label='mlod p50 (ms)')
            plt.plot(x, y_p95, 'd--', label='mlod p95 (ms)')
            plt.xlabel('Pixel error budget (px)')
            plt.ylabel('Render time (ms)')
            plt.title('Multi-LOD render time vs. pixel error')
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig_path = os.path.join(outdir, 'render_time_vs_pixel_error.png')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            print('Chart saved:', fig_path)
    except Exception as e:
        print('Charting skipped:', e)

    print('Per-frame CSVs:', per_budget_csvs)
    print('Summary CSV:', summary_csv)


if __name__ == '__main__':
    main()
