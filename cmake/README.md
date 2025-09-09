# CMake Integration for forge3d

This directory contains CMake files for integrating forge3d into CMake-based projects.

## Overview

The CMake wrapper provides:
- **Cross-platform builds** for Windows, macOS, and Linux
- **Python bindings compilation** via Rust/Cargo integration  
- **Development tools** (format, lint, test, docs)
- **Installation support** for system-wide or virtual environment deployment
- **Example integration** for external projects

## Quick Start

### Building from Source

```bash
# Clone and build
git clone <forge3d-repo>
cd forge3d

# Configure
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Install (optional)
cmake --install build
```

### Integration into External Project

Create `CMakeLists.txt` in your project:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project)

# Add forge3d subdirectory
add_subdirectory(extern/forge3d)

# Or find installed forge3d
# find_package(forge3d REQUIRED)

# Your project targets here
add_executable(my_app main.cpp)
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `FORGE3D_BUILD_PYTHON` | `ON` | Build Python bindings |
| `FORGE3D_BUILD_EXAMPLES` | `OFF` | Build example executables |
| `FORGE3D_BUILD_TESTS` | `OFF` | Build and enable tests |
| `FORGE3D_DEVELOPMENT_MODE` | `OFF` | Enable development tools |
| `FORGE3D_USE_SYSTEM_RUST` | `ON` | Use system Rust toolchain |
| `CMAKE_BUILD_TYPE` | `Release` | Build configuration |

### Example Configuration

```bash
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DFORGE3D_BUILD_EXAMPLES=ON \
    -DFORGE3D_BUILD_TESTS=ON \
    -DFORGE3D_DEVELOPMENT_MODE=ON
```

## Build Targets

| Target | Description |
|--------|-------------|
| `forge3d_rust_build` | Build core Rust library |
| `python_dev_install` | Install Python package in development mode |
| `examples` | Build Rust examples |
| `docs` | Generate Rust documentation |
| `format` | Format Rust code (dev mode only) |
| `clippy` | Run Clippy linter (dev mode only) |
| `clean_rust` | Clean Rust build artifacts (dev mode only) |

### Example Usage

```bash
# Build library only
cmake --build build --target forge3d_rust_build

# Install Python bindings for development
cmake --build build --target python_dev_install

# Build examples
cmake --build build --target examples

# Run development tools
cmake --build build --target format
cmake --build build --target clippy
```

## Testing

```bash
# Configure with tests enabled
cmake -B build -S . -DFORGE3D_BUILD_TESTS=ON

# Run tests
cmake --build build
ctest --test-dir build --verbose
```

## Installation

### System-wide Installation

```bash
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build
sudo cmake --install build
```

### Virtual Environment Installation

```bash
# Activate your virtual environment first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install to virtual environment
cmake -B build -S .
cmake --build build
cmake --install build
```

### Custom Installation Path

```bash
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build
cmake --install build
```

## Cross-Platform Considerations

### Windows (MSVC)
```bash
cmake -B build -S . -G "Visual Studio 17 2022"
cmake --build build --config Release
```

### Windows (MinGW)
```bash
cmake -B build -S . -G "MinGW Makefiles"
cmake --build build
```

### macOS
```bash
cmake -B build -S . -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"  # Universal binary
cmake --build build
```

### Linux
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Advanced Usage

### Custom Rust Features

Modify `CMakeLists.txt` to enable specific Rust features:

```cmake
# Enable specific features during Cargo build
set(CARGO_FEATURES "terrain_spike;enable-pbr")
execute_process(
    COMMAND ${CARGO_EXECUTABLE} build --features ${CARGO_FEATURES}
    # ... rest of command
)
```

### Integration with External Projects

For projects that want to embed forge3d:

```cmake
# In your project's CMakeLists.txt
include(FetchContent)

FetchContent_Declare(
    forge3d
    GIT_REPOSITORY https://github.com/example/forge3d.git
    GIT_TAG v0.6.0
)

FetchContent_MakeAvailable(forge3d)

# Use forge3d
add_executable(my_app main.cpp)
# Note: forge3d provides Python bindings, not C++ libraries
```

### Custom Python Installation Path

```cmake
set(Python3_EXECUTABLE "/path/to/specific/python")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
```

## Troubleshooting

### Common Issues

**Rust not found:**
```
CMake Error: Could not find program: cargo
```
**Solution:** Install Rust from https://rustup.rs/ or set `PATH` correctly.

**Python headers missing:**
```
CMake Error: Python3_INCLUDE_DIRS not found
```
**Solution:** Install Python development headers (`python3-dev` on Ubuntu, `python3-devel` on RHEL).

**Build fails on Windows:**
```
error: linking with `link.exe` failed
```
**Solution:** Install Visual Studio Build Tools or use MinGW.

### Debug Information

Enable verbose output:
```bash
cmake -B build -S . --log-level=DEBUG
cmake --build build --verbose
```

### Environment Variables

Useful environment variables:
- `CARGO_TARGET_DIR`: Override Rust build directory
- `RUSTC_WRAPPER`: Use tools like `sccache` for faster builds  
- `PYTHON_EXECUTABLE`: Specify Python interpreter

## Performance Optimization

### Release Builds
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
```

### LTO (Link Time Optimization)
Configure Cargo.toml profile or use:
```bash
export RUSTFLAGS="-C lto=fat"
cmake --build build
```

### Parallel Builds
```bash
cmake --build build --parallel $(nproc)  # Linux/macOS
cmake --build build --parallel %NUMBER_OF_PROCESSORS%  # Windows
```

## Contributing

To contribute to the CMake integration:

1. **Test across platforms** - Windows, macOS, Linux
2. **Validate CMake versions** - Test with minimum required version (3.20+)
3. **Check Rust integration** - Ensure Cargo builds work correctly
4. **Python compatibility** - Test with multiple Python versions
5. **Documentation** - Update this README for any changes

## References

- [CMake Documentation](https://cmake.org/documentation/)
- [Rust Cargo Book](https://doc.rust-lang.org/cargo/)
- [Python Extension Building](https://docs.python.org/3/extending/)
- [Cross-platform CMake](https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/CrossCompiling)