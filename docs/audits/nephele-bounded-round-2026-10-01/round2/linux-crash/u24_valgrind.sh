#!/usr/bin/env bash
cd /mnt/d/forge3d/.worktrees/nephele-172-completion || exit 9
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1 WGPU_BACKEND=vulkan WGPU_BACKENDS=vulkan MESA_SHADER_CACHE_DISABLE=true
export PYTHONMALLOC=malloc LP_NUM_THREADS=0
HERE=$(dirname "$0")
V=${1:-media_only}
timeout 9000 valgrind --tool=memcheck --smc-check=all --error-limit=no --num-callers=40 \
  --read-var-info=no --track-origins=no --undef-value-errors=no \
  --log-file=/root/vg25-$V.log /root/nephele-lvp-venv/bin/python "$HERE/lvp_bisect.py" $V > /root/vg25-$V.out 2>&1
echo "exit=$?"
grep -cE "Invalid (read|write)" /root/vg25-$V.log
grep -n -m4 -A28 -E "Invalid (read|write)" /root/vg25-$V.log | cut -c1-220
