#!/usr/bin/env bash
# scripts/p5_golden.sh
# P5.5: Golden artifact generator for screen-space GI (P5.0–P5.4)
#
# Generates all P5 PNG artifacts and updates reports/p5/p5_meta.json by
# driving the interactive viewer example (plus the headless P5.0 exporter)
# using the shared GI CLI schema and P5 capture helpers.
#
# Artifacts (under reports/p5/):
#   - p5_gbuffer_normals.png
#   - p5_gbuffer_depth_mips.png
#   - p5_gbuffer_material.png
#   - p5_ssao_cornell.png
#   - p5_ssao_buffers_grid.png
#   - p5_ssao_params_grid.png
#   - p5_ssgi_cornell.png
#   - p5_ssgi_temporal_compare.png
#   - p5_ssr_glossy_spheres.png
#   - p5_ssr_thickness_ablation.png
#   - p5_gi_stack_ablation.png
#   - p5_meta.json (merged/updated across runs)
#
# Usage:
#   ./scripts/p5_golden.sh
#
# Run this from the repository root so Cargo and paths resolve correctly.

set -euo pipefail

echo "=== P5 Golden Artifact Generator (P5.0–P5.4) ==="

# Sanity check: ensure we are in the repo root
if [ ! -f "Cargo.toml" ]; then
  echo "Error: Must run from repository root (Cargo.toml not found)" >&2
  exit 1
fi

# P5.0 – G-Buffer & HZB export (headless, offscreen harness)
echo
echo "[P5.0] Generating G-buffer + HZB artifacts (headless)..."
cargo run --release --example p5_dump_gbuffer -- --size 1280 720 --out reports/p5

run_p5_1() {
  echo
  echo "[P5.1] Generating SSAO/GTAO artifacts via interactive_viewer..."
  FORGE3D_AUTO_SNAPSHOT_PATH="reports/p5/p5_p51_auto.png" \
  cargo run --release --example interactive_viewer -- \
    --fov 60 \
    --gi gtao:on \
    --ssao-radius 0.5 \
    --ssao-intensity 1.5 \
    --ssao-mul 1.0 \
    --ao-blur on \
    --ao-temporal-alpha 0.0 \
    --gi-seed 42 <<'EOF'
:p5 cornell
:p5 grid
:p5 sweep
:quit
EOF
}

run_p5_2() {
  echo
  echo "[P5.2] Generating SSGI artifacts via interactive_viewer..."
  FORGE3D_AUTO_SNAPSHOT_PATH="reports/p5/p5_p52_auto.png" \
  cargo run --release --example interactive_viewer -- \
    --size 1920x1080 \
    --fov 60 \
    --viz material \
    --gi ssgi:on \
    --ssgi-steps 24 \
    --ssgi-radius 1.0 \
    --ssgi-temporal-alpha 0.15 \
    --ssgi-temporal-enable on \
    --ssgi-half off \
    --ssgi-edges on \
    --ssgi-upsample-sigma-depth 0.02 \
    --ssgi-upsample-sigma-normal 0.25 \
    --gi-seed 42 <<'EOF'
:p5 ssgi-cornell
:p5 ssgi-temporal
:quit
EOF
}

run_p5_3_glossy() {
  echo
  echo "[P5.3] Generating SSR glossy spheres artifact via interactive_viewer..."
  FORGE3D_AUTO_SNAPSHOT_PATH="reports/p5/p5_p53_glossy_auto.png" \
  cargo run --release --example interactive_viewer -- \
    --size 1920x1080 \
    --viz lit \
    --gi ssr:on \
    --ssr-max-steps 96 \
    --ssr-thickness 0.08 \
    --gi-seed 42 <<'EOF'
:load-ssr-preset
:p5 ssr-glossy
:quit
EOF
}

run_p5_3_thickness() {
  echo
  echo "[P5.3] Generating SSR thickness ablation artifact via interactive_viewer..."
  FORGE3D_AUTO_SNAPSHOT_PATH="reports/p5/p5_p53_thickness_auto.png" \
  cargo run --release --example interactive_viewer -- \
    --size 1920x1080 \
    --viz lit \
    --gi ssr:on \
    --ssr-max-steps 64 \
    --ssr-thickness 0.08 \
    --gi-seed 42 <<'EOF'
:load-ssr-preset
:p5 ssr-thickness
:quit
EOF
}

run_p5_4() {
  echo
  echo "[P5.4] Generating GI stack ablation artifact via interactive_viewer..."
  FORGE3D_AUTO_SNAPSHOT_PATH="reports/p5/p5_p54_auto.png" \
  cargo run --release --example interactive_viewer -- \
    --size 1920x1080 \
    --viz lit \
    --gi-seed 42 <<'EOF'
:load-ssr-preset
:p5 gi-stack
:quit
EOF
}

run_p5_1
run_p5_2
run_p5_3_glossy
run_p5_3_thickness
run_p5_4

echo
echo "=== P5 golden artifacts written under reports/p5/ ==="
echo "Verify p5_meta.json and PNGs before committing."
