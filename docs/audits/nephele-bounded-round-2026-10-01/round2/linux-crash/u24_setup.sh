#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q mesa-vulkan-drivers libvulkan1 vulkan-tools python3-venv python3-pip valgrind gdb
dpkg -l mesa-vulkan-drivers | tail -1
python3 --version
python3 -m venv /root/nephele-lvp-venv
/root/nephele-lvp-venv/bin/pip install -q numpy pyproj pytest
/root/nephele-lvp-venv/bin/pip install -q --no-deps /mnt/d/forge3d/.worktrees/nephele-172-completion/.tmp/nephele-round2/handoff/forge3d-1.41.0-cp310-abi3-linux_x86_64.whl
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverInfo|apiVersion" || true
echo SETUP_DONE
