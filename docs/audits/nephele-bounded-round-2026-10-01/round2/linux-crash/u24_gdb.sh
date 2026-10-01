#!/usr/bin/env bash
cd /mnt/d/forge3d/.worktrees/nephele-172-completion || exit 9
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1 WGPU_BACKEND=vulkan WGPU_BACKENDS=vulkan MESA_SHADER_CACHE_DISABLE=true
export LP_NUM_THREADS=${LP_NUM_THREADS-0}
HERE=$(dirname "$0")
timeout 3000 gdb -q -batch -ex run -ex bt -ex "info registers rip" -ex "x/6i \$pc-12" -ex "info sharedlibrary" --args /root/nephele-lvp-venv/bin/python "$HERE/lvp_bisect.py" "${1:-media_only}" > /root/gdb-$1.log 2>&1
echo exit=$?
grep -n -A140 "SIGSEGV" /root/gdb-$1.log | head -170 | cut -c1-220
