#!/usr/bin/env bash
export DEBIAN_FRONTEND=noninteractive
dpkg -s vulkan-validationlayers >/dev/null 2>&1 || apt-get install -y -q vulkan-validationlayers >/dev/null
cd /mnt/d/forge3d/.worktrees/nephele-172-completion || exit 9
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1 WGPU_BACKEND=vulkan WGPU_BACKENDS=vulkan MESA_SHADER_CACHE_DISABLE=true
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation RUST_LOG=wgpu_hal=warn
HERE=$(dirname "$0")
V=${1:-media_only}
timeout 3000 /root/nephele-lvp-venv/bin/python -X faulthandler "$HERE/lvp_bisect.py" $V > /root/vvl-$V.log 2>&1
echo "exit=$?"
grep -c "VUID" /root/vvl-$V.log
grep -oE "VUID-[A-Za-z0-9_-]+" /root/vvl-$V.log | sort | uniq -c | sort -rn | head -20
