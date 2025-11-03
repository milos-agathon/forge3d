#!/bin/bash
# P1-09 CLI Integration Test
# Tests that --light specs are correctly wired to renderer.set_lights()

set -e

echo "P1-09: Testing CLI light integration"
echo "======================================"

# Test 1: Single directional light with debug
echo ""
echo "Test 1: Single directional light with --debug-lights"
echo "------------------------------------------------------"
python examples/terrain_demo.py \
    --dem assets/Gore_Range_Albers_1m.tif \
    --output /tmp/p1_09_test_1.png \
    --size 512 512 \
    --light "type=directional,azimuth=225,elevation=45,intensity=5.0,color=1.0,0.95,0.9" \
    --debug-lights \
    2>&1 | grep -E "(P1-09|Light Buffer|light_type|Wrote)"

# Test 2: Multiple lights
echo ""
echo "Test 2: Multiple heterogeneous lights"
echo "------------------------------------------------------"
python examples/terrain_demo.py \
    --dem assets/Gore_Range_Albers_1m.tif \
    --output /tmp/p1_09_test_2.png \
    --size 512 512 \
    --light "type=directional,azimuth=135,elevation=35,intensity=3.0" \
    --light "type=point,pos=100,50,0,intensity=10,range=200" \
    --debug-lights \
    2>&1 | grep -E "(P1-09|Uploaded|lights)"

# Test 3: Spot light with cone angle
echo ""
echo "Test 3: Spot light with cone angle"
echo "------------------------------------------------------"
python examples/terrain_demo.py \
    --dem assets/Gore_Range_Albers_1m.tif \
    --output /tmp/p1_09_test_3.png \
    --size 512 512 \
    --light "type=spot,pos=50,80,50,dir=0,-1,0,cone_angle=30,intensity=8,range=150" \
    --debug-lights \
    2>&1 | grep -E "(P1-09|Uploaded|cone)"

echo ""
echo "======================================"
echo "✅ All P1-09 CLI tests completed"
echo "Output files:"
echo "  /tmp/p1_09_test_1.png"
echo "  /tmp/p1_09_test_2.png"
echo "  /tmp/p1_09_test_3.png"
