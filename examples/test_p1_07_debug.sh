#!/bin/bash
# P1-07/P1-12 Debug Utility Test
# Tests both native GPU path and CPU fallback

set -e

echo "P1-07/P1-12: Testing light debug utility"
echo "=========================================="

# Test 1: Single directional light
echo ""
echo "Test 1: Single directional light"
echo "---------------------------------"
python examples/lights_ssbo_debug.py \
    --light "type=directional,intensity=3.0,color=1,0.9,0.8"

# Test 2: Multiple heterogeneous lights
echo ""
echo "Test 2: Multiple heterogeneous lights"
echo "--------------------------------------"
python examples/lights_ssbo_debug.py \
    --light "type=directional,dir=0.3,0.8,0.5,intensity=5" \
    --light "type=point,pos=0,10,0,intensity=10,color=1,1,0.8" \
    --light "type=spot,pos=5,5,5,dir=0,-1,0,cone=30,intensity=8"

# Test 3: Area lights
echo ""
echo "Test 3: Area lights"
echo "-------------------"
python examples/lights_ssbo_debug.py \
    --light "type=area_rect,pos=0,10,0,dir=0,-1,0,area=2,1.5,intensity=15"

echo ""
echo "=========================================="
echo "✅ All P1-07/P1-12 tests completed"
