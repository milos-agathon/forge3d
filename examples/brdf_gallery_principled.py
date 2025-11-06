#!/usr/bin/env python3
"""
M4: Disney Principled BRDF Gallery

Showcases Disney-specific BRDF parameters that extend beyond standard GGX:
- Clearcoat: Additional specular layer for car paint, lacquer effects
- Sheen: Fabric-like rim lighting for cloth materials
- Specular Tint: Colored specular reflections for metals

This gallery demonstrates how these parameters affect material appearance
while keeping the baseline GGX behavior (with all extensions at 0.0) unchanged.
"""

import argparse
import numpy as np
import forge3d as f3d


def render_principled_grid(params_grid, roughness=0.5, tile_size=128):
    """
    Render a grid of Disney Principled BRDF tiles with varying parameters.
    
    Parameters
    ----------
    params_grid : list of dict
        Each dict contains Disney extension parameters:
        {'clearcoat', 'clearcoat_roughness', 'sheen', 'sheen_tint', 'specular_tint', 'label'}
    roughness : float
        Base material roughness
    tile_size : int
        Size of each tile in pixels
    
    Returns
    -------
    np.ndarray
        Stitched mosaic of all tiles
    """
    tiles = []
    
    for params in params_grid:
        tile = f3d.render_brdf_tile(
            "disney",
            roughness,
            tile_size,
            tile_size,
            exposure=1.0,
            light_intensity=0.45,
            clearcoat=params.get('clearcoat', 0.0),
            clearcoat_roughness=params.get('clearcoat_roughness', 0.0),
            sheen=params.get('sheen', 0.0),
            sheen_tint=params.get('sheen_tint', 0.0),
            specular_tint=params.get('specular_tint', 0.0)
        )
        tiles.append(tile)
    
    # Arrange in rows
    n_tiles = len(tiles)
    n_cols = min(n_tiles, 4)  # Max 4 columns
    n_rows = (n_tiles + n_cols - 1) // n_cols
    
    # Stitch tiles with labels
    rows = []
    for r in range(n_rows):
        row_tiles = tiles[r * n_cols:(r + 1) * n_cols]
        if row_tiles:
            row = np.hstack(row_tiles)
            rows.append(row)
    
    if rows:
        mosaic = np.vstack(rows)
    else:
        mosaic = tiles[0]
    
    return mosaic


def main():
    parser = argparse.ArgumentParser(
        description="M4: Disney Principled BRDF Gallery - Showcase clearcoat, sheen, and specular tint"
    )
    parser.add_argument(
        '--roughness', type=float, default=0.5,
        help='Base material roughness (default: 0.5)'
    )
    parser.add_argument(
        '--tile-size', type=int, default=128,
        help='Tile size in pixels (default: 128)'
    )
    parser.add_argument(
        '--out', type=str, default='brdf_gallery_principled.png',
        help='Output PNG filename (default: brdf_gallery_principled.png)'
    )
    
    args = parser.parse_args()
    
    print(f"M4: Disney Principled BRDF Gallery")
    print(f"  Roughness: {args.roughness:.2f}")
    print(f"  Tile size: {args.tile_size}×{args.tile_size}")
    print(f"  Output: {args.out}\n")
    
    # Define parameter variations to showcase
    params_grid = [
        {
            'label': 'Baseline (GGX-equivalent)',
            'clearcoat': 0.0,
            'clearcoat_roughness': 0.0,
            'sheen': 0.0,
            'sheen_tint': 0.0,
            'specular_tint': 0.0,
        },
        {
            'label': 'Clearcoat Layer',
            'clearcoat': 1.0,
            'clearcoat_roughness': 0.0,
            'sheen': 0.0,
            'sheen_tint': 0.0,
            'specular_tint': 0.0,
        },
        {
            'label': 'Rough Clearcoat',
            'clearcoat': 1.0,
            'clearcoat_roughness': 0.5,
            'sheen': 0.0,
            'sheen_tint': 0.0,
            'specular_tint': 0.0,
        },
        {
            'label': 'Sheen (Fabric)',
            'clearcoat': 0.0,
            'clearcoat_roughness': 0.0,
            'sheen': 1.0,
            'sheen_tint': 0.0,
            'specular_tint': 0.0,
        },
        {
            'label': 'Tinted Sheen',
            'clearcoat': 0.0,
            'clearcoat_roughness': 0.0,
            'sheen': 1.0,
            'sheen_tint': 1.0,
            'specular_tint': 0.0,
        },
        {
            'label': 'Specular Tint',
            'clearcoat': 0.0,
            'clearcoat_roughness': 0.0,
            'sheen': 0.0,
            'sheen_tint': 0.0,
            'specular_tint': 1.0,
        },
        {
            'label': 'Clearcoat + Sheen',
            'clearcoat': 0.5,
            'clearcoat_roughness': 0.2,
            'sheen': 0.5,
            'sheen_tint': 0.5,
            'specular_tint': 0.0,
        },
        {
            'label': 'Full Principled',
            'clearcoat': 0.8,
            'clearcoat_roughness': 0.1,
            'sheen': 0.3,
            'sheen_tint': 0.5,
            'specular_tint': 0.5,
        },
    ]
    
    print("Rendering Disney Principled variations...")
    for i, params in enumerate(params_grid, 1):
        print(f"  [{i}/{len(params_grid)}] {params['label']}...")
    
    mosaic = render_principled_grid(params_grid, args.roughness, args.tile_size)
    
    print(f"\nMosaic size: {mosaic.shape[1]}×{mosaic.shape[0]}")
    print(f"Saving to {args.out}...")
    
    f3d.numpy_to_png(args.out, mosaic)
    print(f"✓ Saved: {args.out}")


if __name__ == '__main__':
    main()
