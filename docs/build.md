Build via CMake (Optional Wrapper)

Overview
- The repository provides an optional CMake wrapper to configure and build the Rust + PyO3 extension across platforms. This wrapper is intended for integrators preferring CMake-style workflows; the primary build remains via `maturin`.

Prerequisites
- Python 3.8+
- Rust toolchain (stable)
- CMake 3.24+

Quick Start
1) Configure the build directory:
   cmake -S . -B build

2) Build the extension/library:
   cmake --build build

Notes
- The CMake build invokes Cargo under the hood to compile the Rust crate. Artifacts are emitted into `build/` and the standard Cargo `target/` folder.
- To rebuild from scratch, remove the `build/` directory.
- For typical Python developer workflows, prefer:
  pip install -U maturin
  maturin develop --release

