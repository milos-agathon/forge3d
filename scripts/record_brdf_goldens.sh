#!/bin/bash
# Script to record BRDF golden reference images
# Usage: ./scripts/record_brdf_goldens.sh

set -e

echo "=== P7 Golden Image Recording ==="
echo ""

# Check if in repo root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from repository root"
    exit 1
fi

# Check if native module is available
if ! python3 -c "import forge3d._forge3d; forge3d._forge3d.render_brdf_tile" 2>/dev/null; then
    echo "Error: Native module not available or not built"
    echo "Build with: maturin develop --release"
    exit 1
fi

# Create golden directory
mkdir -p tests/golden/p7

# Backup existing goldens
if [ -f "tests/golden/p7/mosaic_3x3_128.png" ]; then
    echo "Backing up existing goldens..."
    cp tests/golden/p7/mosaic_3x3_128.png tests/golden/p7/mosaic_3x3_128.png.backup
fi

# Record goldens
echo "Recording golden reference images..."
echo ""
FORGE3D_RECORD_GOLDENS=1 python3 -m pytest tests/test_golden_brdf_mosaic.py -v

# Check results
echo ""
echo "=== Golden Recording Complete ==="
echo ""
ls -lh tests/golden/p7/mosaic_*.png 2>/dev/null || echo "No PNG goldens created"
echo ""
echo "Next steps:"
echo "  1. Review golden images visually"
echo "  2. Run comparison test: pytest tests/test_golden_brdf_mosaic.py -v"
echo "  3. Commit if correct: git add tests/golden/p7/*.png"
echo ""
