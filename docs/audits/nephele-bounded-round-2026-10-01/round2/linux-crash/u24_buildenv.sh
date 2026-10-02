#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -q build-essential pkg-config libvulkan-dev libproj-dev libsqlite3-dev sqlite3 cmake clang git > /dev/null
if [ ! -x /root/.cargo/bin/cargo ]; then curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.97.1; fi
/root/.cargo/bin/rustc --version
/root/nephele-lvp-venv/bin/pip install -q maturin
echo BUILDENV_DONE
