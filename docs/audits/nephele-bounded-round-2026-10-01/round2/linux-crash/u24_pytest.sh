#!/usr/bin/env bash
# Run the hosted-Linux failing test file on CI-identical Mesa with a cold shader cache,
# from the exported source tree (WSL cannot resolve the Windows worktree .git file).
cd /root/src || exit 9
[ -d .git ] || { git init -q && git add -A && git -c user.name=x -c user.email=x@x commit -qm snapshot; }
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json MESA_SHADER_CACHE_DISABLE=true
export FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1 WGPU_BACKEND=vulkan WGPU_BACKENDS=vulkan
/root/nephele-lvp-venv/bin/python -m pytest -p no:cacheprovider -q -rA tests/test_nephele_environment_defects.py > /root/pytest-env-defects.log 2>&1
echo "exit=$?"; tail -6 /root/pytest-env-defects.log | cut -c1-300
