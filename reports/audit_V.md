# Audit Report: Workstream V - Datashader Interop

**Date**: 2025-09-12  
**Auditor**: Claude Code (Audit Mode)  
**Target**: forge3d repository - Workstream V: Datashader Interop

## 1. Scope & CSV Hygiene Summary

**Workstream**: V - Datashader Interop  
**Tasks Matched**: 2 tasks found in roadmap.csv  
**CSV Headers**: ✅ All expected headers present and correctly ordered  
**CSV Hygiene**: ✅ No issues detected in Priority/Phase values or required fields

### Matched Tasks:
- **V1**: Datashader pipeline → RGBA overlay (Priority: Medium, Phase: Beyond MVP)
- **V2**: Datashader performance stress & goldens (Priority: Low, Phase: Beyond MVP)

## 2. Readiness Verdict per Task

### V1: Datashader pipeline → RGBA overlay
**Readiness**: 🔴 **Absent**

**Expected Deliverables**:
- `python/forge3d/adapters/datashader_adapter.py`
- Examples demonstrating datashader integration
- Tests on millions of points

**Acceptance Criteria**:
- Datashader RGBA arrays accepted without copy
- Overlay aligns with coordinates  
- Example notebook renders

### V2: Datashader performance stress & goldens
**Readiness**: 🔴 **Absent**

**Expected Deliverables**:
- `tests/perf/test_datashader_zoom.py`
- Golden images for validation
- CI job integration

**Acceptance Criteria**:
- SSIM≥0.98 across zoom levels
- Frame time within target
- CI artifacts on regression

## 3. Evidence Map

### Repository Structure Analysis
```
python/forge3d/adapters/
├── __init__.py           ✅ EXISTS
├── mpl_cmap.py          ✅ EXISTS  
├── rasterio_tiles.py    ✅ EXISTS
├── reproject.py         ✅ EXISTS
└── datashader_adapter.py ❌ MISSING
```

```
tests/
├── test_staging_performance.py ✅ EXISTS (unrelated perf test)
├── test_lod_perf.py            ✅ EXISTS (unrelated perf test)
├── test_perf.py                ✅ EXISTS (unrelated perf test)
└── perf/                       ❌ MISSING (no perf directory)
    └── test_datashader_zoom.py ❌ MISSING
```

### Keyword Search Results
- **"datashader"**: 1 match (only in docs/task.xml - this audit task)
- **"adapter"**: 200+ matches (extensive adapter framework exists)
- **Datashader-specific files**: 0 matches
- **Performance/stress tests**: Exist for other features, none for datashader

## 4. Blocking Gaps

### V1 Blocking Gaps:
1. **Primary Adapter Missing**: `python/forge3d/adapters/datashader_adapter.py` does not exist
2. **No Datashader Integration**: No imports, classes, or functions related to datashader found
3. **Missing Examples**: No example notebooks or scripts demonstrating datashader usage
4. **No API Surface**: No datashader-related functions exposed in `forge3d/__init__.py`

### V2 Blocking Gaps:
1. **Missing Test Infrastructure**: `tests/perf/` directory does not exist
2. **No Performance Tests**: `test_datashader_zoom.py` does not exist
3. **Missing Golden Images**: No baseline images for SSIM comparison
4. **No CI Integration**: No CI jobs configured for datashader performance testing

## 5. Minimal Change Plan

### For V1: Datashader pipeline → RGBA overlay

**Required Files**:
1. Create `python/forge3d/adapters/datashader_adapter.py`
   - Implement `DatashaderAdapter` class
   - Add functions for Canvas/shade output conversion
   - Include coordinate alignment validation
   - Support zero-copy RGBA array ingestion

2. Update `python/forge3d/adapters/__init__.py`
   - Export datashader adapter functions
   - Add availability check (`is_datashader_available()`)

3. Create example script `examples/datashader_overlay_demo.py`
   - Demonstrate integration with millions of points
   - Show coordinate alignment
   - Generate example notebook output

4. Add tests `tests/test_datashader_adapter.py`
   - Unit tests for adapter functions
   - Integration tests with mock datashader outputs
   - Coordinate validation tests

### For V2: Datashader performance stress & goldens

**Required Files**:
1. Create `tests/perf/` directory structure
2. Implement `tests/perf/test_datashader_zoom.py`
   - Performance stress tests across zoom levels
   - SSIM-based golden image validation
   - Frame time measurement and validation
   - Synthetic data generation for reproducibility

3. Generate baseline golden images
   - Store in appropriate location (tests/goldens/ or similar)
   - Include metadata for zoom levels and parameters

4. Update CI configuration
   - Add job for datashader performance testing
   - Configure artifact upload on regression
   - Set appropriate SSIM thresholds (≥0.98)

**Dependencies to Address**:
- **V1 depends on**: N2 (Advanced Rendering Systems), O3 (Resource Management)
- **V2 depends on**: R1 (Matplotlib Interop), G7 (Performance Testing)

## 6. Validation Runbook

Once implementation is complete, validate using these commands:

### Build & Development Setup
```bash
# Ensure datashader is available
pip install datashader

# Build with latest changes
maturin develop --release

# Run basic import test
python -c "from forge3d.adapters import datashader_adapter; print('✅ Datashader adapter available')"
```

### V1 Validation
```bash
# Run adapter tests
pytest tests/test_datashader_adapter.py -v

# Run example demo
python examples/datashader_overlay_demo.py

# Verify output files created
ls -la *datashader*.png *datashader*.json
```

### V2 Validation
```bash
# Run performance tests
pytest tests/perf/test_datashader_zoom.py -v

# Check SSIM validation
python -c "
import tests.perf.test_datashader_zoom as t
result = t.test_zoom_level_ssim()
print(f'SSIM validation: {\"✅ PASS\" if result >= 0.98 else \"❌ FAIL\"}')"

# Verify golden images exist
ls -la tests/goldens/datashader_*.png
```

### Integration Testing
```bash
# Full test suite (skip GPU-heavy tests if needed)
pytest tests/ -k "datashader" -v

# Build documentation
cd docs && make html

# Verify docs include datashader sections
grep -r "datashader" docs/_build/html/
```

## 7. Risk Assessment

### Implementation Risks
- **Medium**: Coordinate alignment complexity between datashader and forge3d coordinate systems
- **Medium**: Memory spike handling during large point dataset processing
- **Low**: Datashader version compatibility (mitigated by pinning versions)

### Performance Risks  
- **Medium**: Potential performance regressions with large datasets
- **Low**: SSIM threshold sensitivity across different backends/drivers

### Dependencies
- External dependency on datashader package installation
- Requires functional adapter framework (already present)
- CI infrastructure needs enhancement for performance testing

---

**Summary**: Both V1 and V2 tasks are completely **Absent** from the current implementation. The forge3d project has a robust adapter framework in place (as evidenced by mpl_cmap, rasterio_tiles, and reproject adapters), but no datashader-specific implementation exists. Implementation would require creating the adapter module, performance testing infrastructure, and appropriate examples/documentation.