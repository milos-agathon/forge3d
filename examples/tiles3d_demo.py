#!/usr/bin/env python3
"""P5: 3D Tiles and Point Cloud demo.

Demonstrates:
- Loading and traversing 3D Tiles tilesets
- SSE-based LOD selection
- Loading COPC/EPT point clouds
- Point budget-based LOD traversal

Usage:
    # 3D Tiles
    python examples/tiles3d_demo.py --tileset path/to/tileset.json
    python examples/tiles3d_demo.py --tileset path/to/tileset.json --sse 8.0

    # Point Clouds
    python examples/tiles3d_demo.py --copc path/to/pointcloud.copc.laz
    python examples/tiles3d_demo.py --ept path/to/ept.json
    python examples/tiles3d_demo.py --copc path/to/file.copc.laz --budget 1000000

    # Info only
    python examples/tiles3d_demo.py --tileset path/to/tileset.json --info

Exit Criteria (P5):
    - Tileset loads and traverses with LOD
    - Point cloud respects point budget
    - Visible tile/node count varies with SSE/budget
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P5: 3D Tiles and Point Cloud demo"
    )
    
    # Input sources
    parser.add_argument(
        "--tileset",
        type=Path,
        help="Path to 3D Tiles tileset.json",
    )
    parser.add_argument(
        "--copc",
        type=Path,
        help="Path to COPC (.copc.laz) file",
    )
    parser.add_argument(
        "--ept",
        type=Path,
        help="Path to EPT (ept.json) file",
    )
    
    # 3D Tiles parameters
    parser.add_argument(
        "--sse",
        type=float,
        default=16.0,
        help="Screen-space error threshold in pixels (default: 16.0)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=32,
        help="Maximum traversal depth (default: 32)",
    )
    
    # Point cloud parameters
    parser.add_argument(
        "--budget",
        type=int,
        default=5_000_000,
        help="Point budget for point clouds (default: 5,000,000)",
    )
    
    # Camera
    parser.add_argument(
        "--camera",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 100.0],
        metavar=("X", "Y", "Z"),
        help="Camera position (default: 0 0 100)",
    )
    
    # Output
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset info and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def demo_tileset(args: argparse.Namespace) -> int:
    """Demo 3D Tiles functionality."""
    from forge3d.tiles3d import load_tileset, Tiles3dRenderer
    
    print(f"[3D TILES] Loading: {args.tileset}")
    
    try:
        tileset = load_tileset(args.tileset)
    except Exception as e:
        print(f"Error loading tileset: {e}")
        return 1
    
    print(f"[3D TILES] Version: {tileset.version}")
    print(f"[3D TILES] Geometric error: {tileset.geometric_error:.2f}")
    print(f"[3D TILES] Total tiles: {tileset.tile_count}")
    print(f"[3D TILES] Max depth: {tileset.max_depth}")
    
    if args.info:
        print_tileset_tree(tileset.root, 0, max_depth=3)
        return 0
    
    # Create renderer and traverse
    renderer = Tiles3dRenderer(
        sse_threshold=args.sse,
        max_depth=args.max_depth,
    )
    
    camera_pos = tuple(args.camera)
    print(f"\n[TRAVERSE] Camera: {camera_pos}")
    print(f"[TRAVERSE] SSE threshold: {args.sse}")
    
    visible = renderer.get_visible_tiles(tileset, camera_pos)
    
    print(f"[TRAVERSE] Visible tiles: {len(visible)}")
    
    if args.verbose and visible:
        print("\n[VISIBLE TILES]")
        for i, vt in enumerate(visible[:10]):
            uri = vt.tile.content_uri() or "(no content)"
            print(f"  {i+1}. depth={vt.depth}, sse={vt.sse:.2f}, uri={uri}")
        if len(visible) > 10:
            print(f"  ... and {len(visible) - 10} more")
    
    # Test SSE monotonicity
    print("\n[SSE TEST] Verifying tile count decreases with higher SSE...")
    sse_values = [4.0, 8.0, 16.0, 32.0, 64.0]
    counts = []
    for sse in sse_values:
        renderer.sse_threshold = sse
        v = renderer.get_visible_tiles(tileset, camera_pos)
        counts.append(len(v))
        print(f"  SSE={sse:5.1f} -> {len(v)} tiles")
    
    # Check monotonicity (should generally decrease or stay same)
    monotonic = all(counts[i] >= counts[i+1] for i in range(len(counts)-1))
    print(f"[SSE TEST] Monotonic: {'PASS' if monotonic else 'WARN (not strictly monotonic)'}")
    
    print("\n[DONE] 3D Tiles demo complete")
    return 0


def demo_pointcloud(args: argparse.Namespace, is_copc: bool) -> int:
    """Demo Point Cloud functionality."""
    from forge3d.pointcloud import open_laz, open_ept, open_pointcloud, PointCloudRenderer, LazDataset
    
    path = args.copc if is_copc else args.ept
    
    print(f"[POINTCLOUD] Loading: {path}")
    
    try:
        if is_copc:
            dataset = open_pointcloud(path)  # Auto-detect LAZ vs COPC
        else:
            dataset = open_ept(path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Determine format
    if isinstance(dataset, LazDataset):
        fmt = f"LAZ {dataset.version}"
    elif hasattr(dataset, '_is_copc') and dataset._is_copc:
        fmt = "COPC"
    else:
        fmt = "EPT"
    
    print(f"[{fmt}] Total points: {dataset.total_points:,}")
    print(f"[{fmt}] Octree nodes: {dataset.node_count}")
    
    if dataset.bounds:
        b = dataset.bounds
        print(f"[{fmt}] Bounds: ({b.min[0]:.2f}, {b.min[1]:.2f}, {b.min[2]:.2f}) - ({b.max[0]:.2f}, {b.max[1]:.2f}, {b.max[2]:.2f})")
    
    if args.info:
        return 0
    
    # Create renderer and traverse
    renderer = PointCloudRenderer(
        point_budget=args.budget,
        max_depth=args.max_depth,
    )
    
    camera_pos = tuple(args.camera)
    print(f"\n[TRAVERSE] Camera: {camera_pos}")
    print(f"[TRAVERSE] Point budget: {args.budget:,}")
    
    visible = renderer.get_visible_nodes(dataset, camera_pos)
    
    total_visible_points = sum(n.point_count for n in visible)
    print(f"[TRAVERSE] Visible nodes: {len(visible)}")
    print(f"[TRAVERSE] Visible points: {total_visible_points:,}")
    print(f"[TRAVERSE] Budget usage: {total_visible_points / args.budget * 100:.1f}%")
    
    if args.verbose and visible:
        print("\n[VISIBLE NODES]")
        for i, vn in enumerate(visible[:10]):
            print(f"  {i+1}. key={vn.key}, points={vn.point_count:,}, priority={vn.priority:.4f}")
        if len(visible) > 10:
            print(f"  ... and {len(visible) - 10} more")
    
    # Test budget enforcement
    print("\n[BUDGET TEST] Verifying point count scales with budget...")
    budgets = [100_000, 500_000, 1_000_000, 5_000_000]
    for budget in budgets:
        if budget > dataset.total_points:
            continue
        renderer.set_point_budget(budget)
        v = renderer.get_visible_nodes(dataset, camera_pos)
        total = sum(n.point_count for n in v)
        pct = total / budget * 100
        status = "OK" if total <= budget * 1.05 else "OVER"
        print(f"  Budget={budget:>10,} -> {total:>10,} points ({pct:5.1f}%) [{status}]")
    
    print(f"\n[DONE] {fmt} demo complete")
    return 0


def print_tileset_tree(tile, depth: int, max_depth: int = 3):
    """Print tileset tree structure."""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    uri = tile.content_uri() or "(empty)"
    ge = tile.geometric_error
    print(f"{indent}- GE={ge:.2f}, refine={tile.refine}, content={uri}")
    
    for child in tile.children:
        print_tileset_tree(child, depth + 1, max_depth)


def main() -> int:
    args = parse_args()
    
    if not any([args.tileset, args.copc, args.ept]):
        print("Error: Must specify --tileset, --copc, or --ept")
        print("Example: python tiles3d_demo.py --tileset path/to/tileset.json")
        return 1
    
    if args.tileset:
        if not args.tileset.exists():
            print(f"Error: Tileset not found: {args.tileset}")
            return 1
        return demo_tileset(args)
    
    if args.copc:
        if not args.copc.exists():
            print(f"Error: COPC file not found: {args.copc}")
            return 1
        return demo_pointcloud(args, is_copc=True)
    
    if args.ept:
        if not args.ept.exists():
            print(f"Error: EPT file not found: {args.ept}")
            return 1
        return demo_pointcloud(args, is_copc=False)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
