# Flake Proof Pack

This directory contains the deterministically regenerated artifacts for Milestones B, C, and D.

## How to Run

```bash
python scripts/run_flake_proofpack.py
```

The script:
1. Renders the `synthetic_perspective_lod_256` scene with all debug modes
2. Computes metrics for each milestone
3. Generates all required PNG and JSON deliverables
4. Exits with code 0 if all checks pass, non-zero otherwise

## What "PASS" Means

### Milestone B: Diagnostic Modes

- **B1**: Debug grid and individual mode images generated
- **B2**: Modes 26/27 are non-uniform (mean in [0.05, 0.95], range ≥ 0.25, unique bins ≥ 64)
- **B3**: Attribution shows HF energy reduction (ratio ≥ 3.0, max reduction ≤ 0.35)
- **B4**: Sentinel integrity verified

### Milestone C: Ground Truth Normal

- **C1**: Mode 25 is valid (alpha mean ≥ 0.99, luma range ≥ 0.10, unique bins ≥ 32)
- **C2**: Angular error within budget (near: p50 ≤ 3°, p95 ≤ 12°; mid: p50 ≤ 6°, p95 ≤ 18°)
- **C3**: Diff saturation ≤ 10%

### Milestone D: Bandlimit Fade

- **D1**: Blend curve is monotonic with correct boundary values
- **D2**: 36-frame orbit sequence generated
- **D3**: Temporal deltas within thresholds (mean ≤ 1.5, p99 ≤ 12, max ≤ 60)

## Output Structure

```
reports/flake/
├── PROOFPACK.md (this file)
├── proofpack_summary.json
├── milestone_b/
│   ├── perspective/
│   │   ├── debug_grid.png
│   │   ├── mode26_height_lod.png
│   │   ├── mode27_normal_blend.png
│   │   ├── metrics_nonuniform.json
│   │   └── metrics_attribution.json
│   └── sentinels/
│       ├── mode23.png ... mode27.png
│       └── metrics_sentinel.json
├── milestone_c/
│   └── perspective/
│       ├── mode25_ddxddy_normal.png
│       ├── mode25_validity_mask.png
│       ├── mode25_metrics.json
│       ├── normal_compare.png
│       ├── normal_angle_error_heatmap.png
│       ├── normal_angle_error.json
│       ├── normal_diff_raw.json
│       └── normal_diff_amplified.png
└── milestone_d/
    ├── blend_curve.png
    ├── blend_curve_table.json
    ├── temporal_metrics_synth.json
    ├── compare_fade_on_off.json
    └── orbit_synth/
        └── frame_000.png ... frame_350.png
```

## CI Integration

Run as part of CI pipeline:

```yaml
- name: Generate Flake Proof Pack
  run: python scripts/run_flake_proofpack.py
  
- name: Run Flake Tests
  run: |
    pytest tests/test_flake_diagnosis.py -v
    pytest tests/test_blend_curve.py -v
    pytest tests/test_temporal_stability.py -v
```

## Related Documentation

- `docs/flake_debug_contract.md` - Encoding contracts and thresholds
- `docs/plan.md` - Original milestone specifications
