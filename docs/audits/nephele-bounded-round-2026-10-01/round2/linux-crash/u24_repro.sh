#!/usr/bin/env bash
# Ubuntu 24.04 + Mesa 25.2.8 (CI-identical) repro of the background-only capture.
# Usage: u24_repro.sh <label> [diagnostic args...]   (env passthrough: MESA_SHADER_CACHE_DISABLE etc.)
R=/mnt/d/forge3d/.worktrees/nephele-172-completion
LABEL=$1; shift
cd "$R" || exit 9
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1 WGPU_BACKEND=vulkan WGPU_BACKENDS=vulkan RUST_BACKTRACE=1
PY=/root/nephele-lvp-venv/bin/python
OUT=/root/cap-$LABEL; rm -rf "$OUT"
timeout 1500 "$PY" -X faulthandler -m scripts.nephele_shadow_inscatter_diagnostic --repo "$R" --output "$OUT" "$@" > "/root/cap-$LABEL.log" 2>&1
echo "exit=$?"; tail -12 "/root/cap-$LABEL.log" | cut -c1-300
