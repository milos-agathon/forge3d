#!/usr/bin/env python3
"""
Script to record P7-09 golden reference images without pytest dependency.
"""
import sys
from pathlib import Path
import numpy as np

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

try:
    import forge3d as f3d
    print("✓ forge3d imported")
except ImportError as e:
    print(f"✗ Failed to import forge3d: {e}")
    print("Build with: maturin develop --release")
    sys.exit(1)

try:
    from PIL import Image
    print("✓ PIL imported")
except ImportError:
    print("✗ PIL required: pip install Pillow")
    sys.exit(1)

GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden" / "p7"


def create_mosaic(tiles: list, gap: int = 4) -> np.ndarray:
    """Stitch tiles into a grid mosaic."""
    if not tiles:
        raise ValueError("Empty tiles list")
    
    # Assume square grid
    grid_size = int(np.sqrt(len(tiles)))
    if grid_size * grid_size != len(tiles):
        raise ValueError(f"Tiles count {len(tiles)} is not a perfect square")
    
    tile_h, tile_w = tiles[0].shape[:2]
    
    # Calculate mosaic dimensions
    mosaic_h = grid_size * tile_h + (grid_size - 1) * gap
    mosaic_w = grid_size * tile_w + (grid_size - 1) * gap
    
    # Create mosaic with dark background
    mosaic = np.full((mosaic_h, mosaic_w, 4), [20, 20, 20, 255], dtype=np.uint8)
    
    # Place tiles
    for idx, tile in enumerate(tiles):
        row = idx // grid_size
        col = idx % grid_size
        y = row * (tile_h + gap)
        x = col * (tile_w + gap)
        mosaic[y:y+tile_h, x:x+tile_w] = tile
    
    return mosaic


def record_golden_3x3():
    """Record primary 3×3 golden: GGX, Disney, Phong @ r=0.3,0.5,0.7"""
    print("\n=== Recording Primary Golden: 3×3 Grid ===")
    
    models = ["ggx", "disney", "phong"]
    roughness_values = [0.3, 0.5, 0.7]
    tile_size = 128
    
    tiles = []
    for model in models:
        for roughness in roughness_values:
            print(f"  Rendering {model} @ r={roughness:.1f}...", end=" ")
            try:
                tile = f3d.render_brdf_tile(model, roughness, tile_size, tile_size, False)
                tiles.append(tile)
                print("✓")
            except Exception as e:
                print(f"✗ {e}")
                return False
    
    # Create mosaic
    print("  Stitching mosaic...")
    mosaic = create_mosaic(tiles, gap=4)
    print(f"  Mosaic size: {mosaic.shape[1]}×{mosaic.shape[0]}")
    
    # Save
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "mosaic_3x3_128.png"
    
    Image.fromarray(mosaic).save(golden_path)
    print(f"✓ Saved: {golden_path}")
    print(f"  File size: {golden_path.stat().st_size / 1024:.1f} KB")
    
    return True


def record_golden_ndf():
    """Record optional NDF-only golden: GGX, Disney @ r=0.2,0.5,0.8 (NDF mode)"""
    print("\n=== Recording Optional NDF Golden: 2×3 Grid ===")
    
    models = ["ggx", "disney"]
    roughness_values = [0.2, 0.5, 0.8]
    tile_size = 128
    
    tiles = []
    for model in models:
        for roughness in roughness_values:
            print(f"  Rendering {model} NDF @ r={roughness:.1f}...", end=" ")
            try:
                tile = f3d.render_brdf_tile(model, roughness, tile_size, tile_size, ndf_only=True)
                tiles.append(tile)
                print("✓")
            except Exception as e:
                print(f"✗ {e}")
                return False
    
    # Pad to 9 tiles for square grid
    while len(tiles) < 9:
        black_tile = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        black_tile[:, :, 3] = 255
        tiles.append(black_tile)
    
    # Create mosaic
    print("  Stitching mosaic...")
    mosaic = create_mosaic(tiles[:9], gap=4)
    print(f"  Mosaic size: {mosaic.shape[1]}×{mosaic.shape[0]}")
    
    # Save
    golden_path = GOLDEN_DIR / "mosaic_2x3_ndf_128.png"
    
    Image.fromarray(mosaic).save(golden_path)
    print(f"✓ Saved: {golden_path}")
    print(f"  File size: {golden_path.stat().st_size / 1024:.1f} KB")
    
    return True


def main():
    print("P7-09 Golden Image Recording")
    print("=" * 60)
    
    # Check if native module is available
    try:
        tile = f3d.render_brdf_tile("ggx", 0.5, 64, 64, False)
        print("✓ Native module available")
    except Exception as e:
        print(f"✗ Native module not available: {e}")
        print("\nBuild with: maturin develop --release")
        return 1
    
    # Record primary golden
    success1 = record_golden_3x3()
    
    # Record optional NDF golden
    success2 = record_golden_ndf()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All goldens recorded successfully")
        print(f"\nGolden images:")
        print(f"  - tests/golden/p7/mosaic_3x3_128.png")
        print(f"  - tests/golden/p7/mosaic_2x3_ndf_128.png")
        print("\nNext steps:")
        print("  1. Visual inspection: open tests/golden/p7/*.png")
        print("  2. Commit: git add tests/golden/p7/*.png")
        print("  3. Run comparison tests (if pytest available)")
        return 0
    else:
        print("✗ Some goldens failed to record")
        return 1


if __name__ == "__main__":
    sys.exit(main())
