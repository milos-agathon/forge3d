#!/usr/bin/env bash
# Build a Linux release wheel of a given worktree revision inside WSL.
set -e
REV=${1:-HEAD}
W=/mnt/d/forge3d/.worktrees/nephele-172-completion
rm -rf /root/src && mkdir -p /root/src /root/wheels
git config --global --add safe.directory '*' || true
# The worktree's .git file points at a Windows path, so export via the Windows-side archive.
tar -xf "$W/.tmp/nephele-round2/src-$REV.tar" -C /root/src
cd /root/src
export PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/root/target
/root/nephele-lvp-venv/bin/maturin build --release -i /root/nephele-lvp-venv/bin/python -o /root/wheels/$REV > /root/build-$REV.log 2>&1
ls /root/wheels/$REV
/root/nephele-lvp-venv/bin/pip install -q --no-deps --force-reinstall /root/wheels/$REV/*.whl
echo BUILD_DONE
