#!/usr/bin/env bash
cd /mnt/d/forge3d/.worktrees/nephele-172-completion || exit 9
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1 WGPU_BACKEND=vulkan WGPU_BACKENDS=vulkan MESA_SHADER_CACHE_DISABLE=true
PY=${PY:-/root/nephele-lvp-venv/bin/python}
HERE=$(dirname "$0")
for v in "$@"; do
  timeout 1500 "$PY" -X faulthandler "$HERE/lvp_bisect.py" "$v" > "/root/bisect-$v.log" 2>&1
  echo "$v -> exit=$? $(grep -m1 -E '^OK|Error' /root/bisect-$v.log | cut -c1-200)"
done
