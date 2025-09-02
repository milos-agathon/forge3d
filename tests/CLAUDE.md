# CLAUDE Knowledge: tests/

## Purpose

Comprehensive test suite for forge3d covering all major functionality: Python API, Rust core, rendering pipeline, terrain processing, vector graphics, colormaps, and cross-platform compatibility.

## Public API / Entry Points

* **Main test runner**: `pytest -q` (runs all tests with backend-aware skips)
* **Selective testing**: Tests support markers for targeted runs:
  - GPU tests: `pytest -m gpu`
  - Terrain tests: `pytest -m terrain`  
  - Camera tests: `pytest -m camera`
* **Environment control**: `VF_ENABLE_TERRAIN_TESTS=1` enables terrain-specific tests

## Build & Test (component-scoped)

```bash
# Run all tests
pytest -q

# Run specific test categories  
pytest tests/test_terrain*.py -v
pytest tests/test_camera.py -v
pytest tests/test_api*.py -v

# Run with terrain tests enabled
VF_ENABLE_TERRAIN_TESTS=1 pytest tests/test_b1*.py -v

# Skip GPU tests if no adapter
pytest -m "not gpu"
```

## Important Files

* `conftest.py` - pytest configuration and shared fixtures
* `test_api.py` - Core API functionality tests
* `test_terrain*.py` - Terrain processing and rendering tests
* `test_camera.py` - Camera system and transforms tests
* `test_colormap.py` - Colormap functionality tests
* `test_diagnostics.py` - Device detection and probe tests
* `test_numpy_*.py` - NumPy interoperability tests
* `test_packaging_*.py` - Python package integration tests
* `smoke_test.py` - Basic smoke tests for CI
* `_ssim.py` - Image similarity utilities for visual regression tests

## Dependencies

* **Internal**: forge3d Python package, _forge3d Rust extension
* **External**: 
  - pytest (test framework)
  - numpy (array operations)
  - PIL/image libraries (for PNG comparison tests)

## Gotchas & Limits

* **GPU dependency**: Many tests require a compatible GPU adapter (Vulkan/Metal/DX12/GL)
* **Backend skips**: Tests automatically skip if required GPU backend is unavailable
* **Platform differences**: Some visual tests may have minor pixel differences across platforms due to driver variations
* **Memory constraints**: Tests respect ≤512 MiB host-visible memory budget
* **Determinism**: Tests use fixed seeds for reproducible synthetic data generation
* **Contiguity**: NumPy arrays must be C-contiguous for GPU upload tests

## Common Tasks

* **Add new API test**: Create `test_new_feature.py` following existing patterns
* **Debug GPU test failure**: Check `device_probe()` output and available adapters
* **Add visual regression test**: Use `_ssim.py` utilities for image comparison
* **Skip GPU tests**: Use `@pytest.mark.skipif(not _gpu_ok())` decorator
* **Test terrain features**: Enable with `VF_ENABLE_TERRAIN_TESTS=1` environment variable

## Ownership / TODOs

* **Test coverage**: Comprehensive coverage across all API surfaces
* **CI integration**: Tests run on multiple platforms with appropriate backend selection
* **Performance benchmarks**: Some tests include basic performance validation
* **Visual regression**: Golden image comparison for rendering accuracy