# Packaging Guide

This document describes the improved packaging flow for forge3d, including artifact exclusion and optimized builds.

## Changes Made

### 1. Enhanced .gitignore ✅

Added comprehensive exclusions for:
- **Generated outputs**: `*.png`, `*.jpg`, `/out/`, `/diag_out/`, `/artifacts/`, `/.benchmarks/`
- **Build artifacts**: `build.log`, `test_output.json`, `*.log`, `*.whl`, `*.tar.gz`, `.maturin/`
- **IDE files**: `.vscode/settings.json`, `*.swp`, `*.swo`
- **Temporary files**: `tmp/`, `temp/`, `*.tmp`, `*.temp`
- **Performance data**: `*.prof`, `perf.data*`, `flamegraph*.svg`
- **Documentation builds**: `/docs/_build/`, `/docs/build/`

### 2. Created MANIFEST.in ✅

Explicit control over source distribution contents:
- **Include**: Source code, data files, configuration, documentation source
- **Exclude**: Build artifacts, generated outputs, development files, IDE config

### 3. Improved pyproject.toml ✅

Enhanced packaging metadata:
- **PyPI classifiers**: Proper categorization for discoverability
- **Keywords**: Graphics, rendering, WebGPU, Rust, 3D, terrain, visualization
- **Maturin optimization**: Strip symbols, LTO profile, smaller wheels
- **Package data**: Include necessary data files (`*.rgba`, `*.png`)

### 4. Optimized Cargo.toml ✅

Added release-lto profile for Python wheel builds:
- **Fat LTO**: Maximum optimization
- **Strip symbols**: Smaller binaries
- **Panic abort**: No unwinding overhead
- **Single codegen unit**: Better optimization

## Usage

### Building Python Wheels

```bash
# Development build (faster compilation)
maturin develop

# Optimized wheel for distribution
maturin build --release --profile release-lto --strip

# Build for multiple Python versions
maturin build --release --profile release-lto --strip --interpreter python3.8 python3.9 python3.10 python3.11 python3.12
```

### Source Distribution

```bash
# Create source distribution (uses MANIFEST.in)
maturin sdist

# Build both wheel and sdist
maturin build --release --profile release-lto --strip --sdist
```

### Local Testing

```bash
# Install from local wheel for testing
pip install target/wheels/forge3d-*.whl

# Or install in development mode
maturin develop --release
```

## Package Contents

### Included in Wheels
- ✅ Compiled Rust extension (`_forge3d.pyd`/`_forge3d.so`)
- ✅ Python source code (`python/forge3d/`)
- ✅ Data files (`data/*.rgba`)
- ✅ Asset files (`assets/*.png`)
- ✅ Documentation (README, LICENSE, CHANGELOG)

### Excluded from Wheels
- ❌ Rust source code (`src/`)
- ❌ Build artifacts (`target/`, `build/`)
- ❌ Generated outputs (`out/`, `*.png` outputs)
- ❌ Development files (`.venv/`, `__pycache__/`)
- ❌ IDE configuration (`.vscode/`, `*.swp`)
- ❌ Test artifacts (`.pytest_cache/`, `test_output.json`)

### Included in Source Distribution (sdist)
- ✅ All source code (`src/`, `python/`)
- ✅ Build configuration (`Cargo.toml`, `pyproject.toml`)
- ✅ Documentation source (`docs/`)
- ✅ Tests (`tests/`)
- ✅ Examples (`examples/`)
- ✅ Data and assets (`data/`, `assets/`)

### Excluded from Source Distribution
- ❌ Build artifacts (`target/`, `out/`)
- ❌ Generated files (`*.png`, `*.log`)
- ❌ Development environment (`.venv/`, `.vscode/`)
- ❌ Git metadata (`.git/`, `.gitignore`)

## Optimization Details

### Wheel Size Reduction
- **Symbol stripping**: ~30-50% size reduction
- **Fat LTO**: ~10-20% size reduction + performance boost
- **Panic abort**: Small size reduction + performance
- **Asset exclusion**: Excludes temporary/generated files

### Performance Improvements
- **Fat LTO**: Cross-crate optimization
- **Single codegen unit**: Better inlining
- **Panic abort**: No unwinding overhead
- **Release profile**: Full optimizations enabled

## CI/CD Integration

For automated builds, use:

```bash
# Build optimized wheels for all platforms
maturin build --release --profile release-lto --strip --universal2 --out dist/

# Upload to PyPI (with appropriate authentication)
twine upload dist/*.whl dist/*.tar.gz
```

## Package Size Expectations

| Component | Typical Size |
|-----------|--------------|
| **Wheel (optimized)** | 15-25 MB (varies by platform) |
| **Wheel (debug)** | 50-100 MB |
| **Source distribution** | 2-5 MB |
| **Without optimization** | 30-60 MB |

## Verification

Test package contents:
```bash
# List wheel contents
unzip -l target/wheels/forge3d-*.whl

# List source distribution contents  
tar -tzf target/wheels/forge3d-*.tar.gz

# Verify package installs correctly
pip install target/wheels/forge3d-*.whl
python -c "import forge3d; print('Success:', forge3d.__version__)"
```

## Troubleshooting

### Large Wheel Size
- Ensure `strip = true` in pyproject.toml
- Use `--profile release-lto` for builds
- Check for included debug symbols: `objdump -h _forge3d.so | grep debug`

### Missing Files in Package
- Check MANIFEST.in includes required files
- Verify `include-package-data = true` in pyproject.toml
- Test with `maturin sdist` and inspect contents

### Build Performance
- Use `maturin develop` for development (faster)
- Use `--profile release-lto` only for distribution builds
- Consider `lto = "thin"` for faster builds with some optimization

## References

- [Maturin Documentation](https://maturin.rs/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Rust Cargo Book](https://doc.rust-lang.org/cargo/)
- [PyPI Classifiers](https://pypi.org/classifiers/)