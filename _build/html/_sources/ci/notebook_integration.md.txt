# Running Notebooks in CI

This guide describes how forge3d integrates Jupyter notebooks into CI pipelines for automated testing and validation of integration scenarios.

## Overview

forge3d uses Jupyter notebooks for integration testing, providing comprehensive end-to-end validation of complex workflows including:

- External library adapter integration (matplotlib, datashader, xarray, rasterio)
- Geospatial data processing and CRS validation
- Multi-step rendering pipelines with error handling
- Performance benchmarking and memory usage tracking

## CI Integration Architecture

### Notebook Execution Workflow

```
notebooks/integration/
├── matplotlib_terrain.ipynb      # Terrain + matplotlib colormaps
├── datashader_points.ipynb       # Large-scale point visualization  
├── adapter_showcase.ipynb        # Multi-adapter integration
└── data_ingestion.ipynb          # Geospatial I/O workflows

.github/workflows/notebooks.yml   # CI pipeline configuration
```

### Execution Pipeline

1. **Environment Setup**: Install forge3d and optional dependencies
2. **Headless Execution**: Run notebooks via `nbconvert --execute`
3. **Artifact Collection**: Capture rendered outputs and metadata  
4. **Validation**: Verify outputs meet quality requirements
5. **Artifact Storage**: Upload results for debugging and analysis

## Configuration

### GitHub Actions Workflow

The notebook CI pipeline is configured in `.github/workflows/notebooks.yml`:

```yaml
name: Integration Notebooks CI

on:
  push:
    branches: [ main, 'feat/*workstream*', 'feat/*integration*' ]
    paths:
      - 'notebooks/integration/*.ipynb'
      - 'python/forge3d/adapters/*.py'

env:
  NOTEBOOK_TIMEOUT_MINUTES: "10"  # 10 minute budget per notebook
  MEMORY_BUDGET_MB: "512"         # Memory limit enforcement
  JUPYTER_KERNEL_TIMEOUT: "600"   # Kernel timeout in seconds

jobs:
  notebook-execution:
    runs-on: ubuntu-latest
    timeout-minutes: 45  # Total: 4 notebooks × 10 min + setup
    
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
        notebook: [
          "matplotlib_terrain",
          "datashader_points", 
          "adapter_showcase",
          "data_ingestion"
        ]
```

### Execution Requirements

Each notebook must satisfy:

- **Runtime**: ≤10 minutes execution time
- **Memory**: ≤512 MiB peak usage
- **Dependencies**: Graceful fallback when optional libraries unavailable
- **Outputs**: At least one PNG proof image
- **Metadata**: JSON with timing and resource usage

### Error Handling

Notebooks include comprehensive error handling:

```python
# Example error handling pattern
try:
    import optional_library
    OPTIONAL_AVAILABLE = True
    print("✓ optional_library available")
except ImportError:
    OPTIONAL_AVAILABLE = False
    print("ℹ optional_library not available - skipping related features")

# Graceful feature degradation
if OPTIONAL_AVAILABLE:
    # Full functionality
    result = advanced_processing_with_optional_library(data)
else:
    # Fallback processing  
    result = basic_processing_without_optional_library(data)
```

## Notebook Structure

### Standard Cell Organization

All integration notebooks follow a consistent structure:

#### 1. Introduction & Setup
```python
"""
Notebook Description and Purpose
Expected runtime: < X minutes
Memory usage: < Y MiB  
Outputs: list of expected files
"""

import sys
from pathlib import Path

# Add repo root for imports
if Path('../..').exists():
    sys.path.insert(0, str(Path('../..').resolve()))

import forge3d as f3d
print(f"✓ forge3d {f3d.__version__} loaded successfully")
```

#### 2. Device Information
```python
# Check GPU capabilities and adapter info
try:
    device_info = f3d.device_probe()
    print("🖥️  Device Information:")
    print(f"   Backend: {device_info.get('backend', 'unknown')}")
    print(f"   Adapter: {device_info.get('adapter_name', 'unknown')}")
    
    features = device_info.get('features', '')
    if 'TIMESTAMP_QUERY' in features:
        print("   ✓ GPU timing available")
    else:
        print("   ⚠ GPU timing fallback to CPU")
        
except Exception as e:
    print(f"⚠ Device probe failed: {e}")
    print("Continuing with fallback adapter...")
```

#### 3. Data Generation/Loading
```python
# Generate deterministic synthetic data
np.random.seed(42)  # Reproducible results
data = generate_test_data(size=256, complexity='medium')

print(f"📊 Generated data: {data.shape} ({data.dtype})")
print(f"   Memory footprint: {data.nbytes / 1024:.1f} KB")
```

#### 4. Adapter Integration  
```python
# Test adapter availability
try:
    from forge3d.adapters import is_matplotlib_available
    MATPLOTLIB_AVAILABLE = is_matplotlib_available()
    print(f"✓ matplotlib adapter available: {MATPLOTLIB_AVAILABLE}")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("✗ matplotlib adapter not available")

# Use adapter if available
if MATPLOTLIB_AVAILABLE:
    from forge3d.adapters import matplotlib_to_forge3d_colormap
    colormap = matplotlib_to_forge3d_colormap('terrain')
else:
    # Fallback to built-in colormap
    colormap = 'viridis'
```

#### 5. Processing & Rendering
```python
# Time-tracked processing
import time

start_time = time.time()
processing_start = time.time()

# Main processing pipeline
renderer = f3d.Renderer(512, 512, prefer_software=False)
renderer.upload_height_r32f(height_data, spacing=0.1, exaggeration=1.5)

processing_time = (time.time() - processing_start) * 1000

# Render output
render_start = time.time()
rgba_output = renderer.render_rgba()
render_time = (time.time() - render_start) * 1000

total_time = (time.time() - start_time) * 1000
print(f"⏱️  Timing: processing={processing_time:.1f}ms, render={render_time:.1f}ms")
```

#### 6. Output Generation
```python
# Save primary output
output_path = "notebook_output.png"
save_start = time.time()
f3d.numpy_to_png(output_path, rgba_output)
save_time = (time.time() - save_start) * 1000

# Verify output
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path) / 1024
    print(f"✓ Output saved: {output_path} ({file_size:.1f} KB)")
else:
    raise FileNotFoundError(f"Expected output {output_path} not created")
```

#### 7. Performance Metrics
```python
# Export comprehensive metadata for CI validation
metadata = {
    "notebook_info": {
        "name": "example_notebook",
        "forge3d_version": f3d.__version__,
        "execution_time_ms": total_time,
        "device_info": device_info,
    },
    "timing": {
        "processing_ms": processing_time,
        "rendering_ms": render_time,
        "save_ms": save_time,
        "total_ms": total_time,
    },
    "memory": {
        "peak_mb": get_peak_memory_usage(),
        "budget_compliance": peak_mb < 512,
    },
    "outputs": {
        "primary": output_path,
        "file_size_kb": file_size,
    },
    "adapters": {
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "datashader": is_datashader_available(),
        "rasterio": is_rasterio_available(),
    }
}

# Save metadata for CI validation
with open("notebook_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

#### 8. Validation & Cleanup
```python
# Final validation
print("🔍 Final Validation:")
assert os.path.exists(output_path), "Primary output must exist"
assert file_size > 1.0, "Output file must be substantial"
assert total_time < 600000, "Must complete within 10 minute budget"  # 10 min = 600,000 ms
assert peak_mb < 512, "Must stay within memory budget"

print("✅ Notebook completed successfully!")
print(f"📁 Outputs: {output_path}")
print(f"⏱️  Total time: {total_time/1000:.1f}s")
print(f"💾 Peak memory: {peak_mb:.1f} MB")
```

## Dependency Management

### Optional Dependencies

Integration notebooks handle optional dependencies gracefully:

```python
# Comprehensive dependency checking
adapters_status = {}

try:
    import matplotlib
    from forge3d.adapters import matplotlib_to_forge3d_colormap
    adapters_status['matplotlib'] = {
        'available': True, 
        'version': matplotlib.__version__
    }
except ImportError:
    adapters_status['matplotlib'] = {'available': False, 'reason': 'ImportError'}

try:
    import datashader
    from forge3d.adapters import datashader_to_overlay_texture
    adapters_status['datashader'] = {
        'available': True,
        'version': datashader.__version__
    }
except ImportError:
    adapters_status['datashader'] = {'available': False, 'reason': 'ImportError'}

# Continue with available adapters only
print(f"Adapter availability: {adapters_status}")
```

### Fallback Strategies

When optional dependencies are missing:

1. **Skip Feature**: Skip optional processing steps
2. **Synthetic Data**: Use built-in test data instead of real datasets  
3. **Built-in Functions**: Use forge3d built-ins instead of external adapters
4. **Reduced Functionality**: Provide core functionality without enhancements

```python
# Example: Colormap fallback strategy
if adapters_status['matplotlib']['available']:
    # Use matplotlib colormap
    colormap_data = matplotlib_to_forge3d_colormap('terrain')
    colormap_source = "matplotlib"
else:
    # Fallback to built-in colormap
    colormap_data = 'terrain'  # Built-in forge3d colormap
    colormap_source = "built-in"

print(f"Using {colormap_source} colormap: {colormap_data}")
```

## CI Validation

### Success Criteria

For a notebook to pass CI validation:

1. **Execution**: Completes without exceptions
2. **Timing**: Finishes within 10-minute budget  
3. **Memory**: Stays within 512 MiB limit
4. **Outputs**: Generates expected PNG file(s)
5. **Metadata**: Exports valid JSON metadata
6. **Quality**: Output images meet minimum quality thresholds

### Failure Analysis

Common failure modes and debugging:

#### Timeout Failures
```bash
# Notebook exceeded 10-minute timeout
CAUSE: Expensive operations or infinite loops
DEBUG: Check timing metadata for bottlenecks
FIX: Optimize expensive operations or reduce data size
```

#### Memory Failures  
```bash
# Process killed due to memory usage
CAUSE: Large arrays or memory leaks
DEBUG: Check memory tracking in notebook output
FIX: Use streaming processing or reduce data size
```

#### Dependency Failures
```bash
# Import errors for optional dependencies
CAUSE: Missing optional packages in CI environment
DEBUG: Check adapter availability sections
FIX: Ensure graceful fallback is implemented
```

#### GPU Failures
```bash
# Rendering pipeline failures
CAUSE: GPU not available or driver issues
DEBUG: Check device_probe() output
FIX: Use prefer_software=True fallback
```

### Quality Metrics

Notebooks export quality metrics for validation:

```python
# Image quality validation
def validate_output_quality(image_path, min_content_ratio=0.1):
    """Validate that rendered image has substantial content."""
    img_data = f3d.png_to_numpy(image_path)
    
    # Check for non-black content
    non_zero_pixels = np.count_nonzero(img_data[:, :, :3])
    total_pixels = img_data.shape[0] * img_data.shape[1] * 3
    content_ratio = non_zero_pixels / total_pixels
    
    return {
        'content_ratio': content_ratio,
        'meets_threshold': content_ratio >= min_content_ratio,
        'image_size': img_data.shape,
        'file_exists': True
    }

# Export quality metrics
quality_metrics = validate_output_quality(output_path)
metadata['quality'] = quality_metrics

assert quality_metrics['meets_threshold'], f"Output quality too low: {quality_metrics['content_ratio']:.1%}"
```

## Performance Monitoring

### Timing Benchmarks

Notebooks track detailed timing for performance regression detection:

```python
# Detailed timing breakdown
timings = {
    'startup_ms': (import_end - import_start) * 1000,
    'data_generation_ms': (data_end - data_start) * 1000,
    'adapter_setup_ms': (adapter_end - adapter_start) * 1000,
    'upload_ms': (upload_end - upload_start) * 1000,
    'render_ms': (render_end - render_start) * 1000,
    'save_ms': (save_end - save_start) * 1000,
    'total_ms': (total_end - total_start) * 1000,
}

# Performance validation
for phase, time_ms in timings.items():
    if time_ms > 60000:  # 1 minute per phase
        print(f"⚠ {phase} took {time_ms/1000:.1f}s (>1min threshold)")
    else:
        print(f"✓ {phase}: {time_ms:.1f}ms")
```

### Memory Tracking

```python
import psutil
import os

def get_memory_usage():
    """Get current process memory usage."""
    process = psutil.Process(os.getpid())
    return {
        'rss_mb': process.memory_info().rss / 1024 / 1024,
        'vms_mb': process.memory_info().vms / 1024 / 1024,
    }

# Track memory at key points
memory_baseline = get_memory_usage()
# ... processing ...
memory_peak = get_memory_usage()

memory_metrics = {
    'baseline_mb': memory_baseline['rss_mb'],
    'peak_mb': memory_peak['rss_mb'],  
    'delta_mb': memory_peak['rss_mb'] - memory_baseline['rss_mb'],
    'budget_compliance': memory_peak['rss_mb'] < 512,
}

metadata['memory'] = memory_metrics
```

## Best Practices

### Development Guidelines

1. **Deterministic**: Always use fixed random seeds
2. **Robust**: Handle missing dependencies gracefully
3. **Efficient**: Stay within time and memory budgets
4. **Documented**: Clear explanations of processing steps
5. **Validated**: Comprehensive output checking

### Testing Locally

Before committing notebooks:

```bash
# Test individual notebook
jupyter nbconvert --execute --to notebook notebooks/integration/example.ipynb

# Test all integration notebooks
for notebook in notebooks/integration/*.ipynb; do
    echo "Testing $notebook..."
    jupyter nbconvert --execute --to notebook "$notebook" --ExecutePreprocessor.timeout=600
done

# Validate outputs
ls -la notebooks/integration/*.png
ls -la notebooks/integration/*.json
```

### Debugging Failed Runs

When notebooks fail in CI:

1. **Download Artifacts**: Get execution logs and partial outputs
2. **Check Device Info**: Verify GPU availability and capabilities  
3. **Review Timings**: Identify bottleneck operations
4. **Test Locally**: Reproduce with similar environment
5. **Add Diagnostics**: Enhanced logging for problem areas

```python
# Enhanced debugging output
import logging
logging.basicConfig(level=logging.DEBUG)

print("🔍 Debug Information:")
print(f"   Python: {sys.version}")
print(f"   Platform: {sys.platform}")
print(f"   Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"   CPU cores: {psutil.cpu_count()}")

# Detailed device information
device_info = f3d.device_probe()
print(f"   GPU: {device_info}")
```

This CI integration ensures that forge3d's complex integration scenarios remain stable and performant across different environments and dependency combinations.