# I6: Bind Group Churn Performance Analysis

**Status**: Complete  
**Experiment**: Split vs Single Bind Group Performance  
**Files**: `examples/perf/split_vs_single_bg.rs`, `examples/perf/split_vs_single_bg.py`

## Overview

This experiment measures the performance impact of bind group churn in WebGPU rendering. It compares two approaches:

1. **Split Bind Groups**: Each object has its own bind group (causes frequent bind group switches)
2. **Single Bind Group**: All objects share one bind group with dynamic offsets (minimal switching)

The goal is to quantify the CPU overhead and frame time impact of frequent bind group state changes during rendering.

## Experimental Design

### Test Scenarios

**Split Bind Groups Approach:**
- Creates N separate uniform buffers (one per object)
- Creates N separate bind groups (one per object)  
- Renders each object with `set_bind_group()` call per object
- Measures bind group switches and frame times

**Single Bind Group Approach:**
- Creates one large uniform buffer containing all object data
- Creates one bind group with dynamic offset support
- Renders all objects using `set_bind_group()` with dynamic offsets
- Measures reduced state changes and frame times

### Parameters

- **Objects**: 1,000 objects by default (configurable with `--objects`)
- **Frames**: 600 frames by default (configurable with `--frames`)
- **Geometry**: Simple quads (6 indices each)
- **Transform**: Per-object 4x4 matrix (64 bytes, aligned)

## Running the Experiment

### Command Line Usage

```bash
# Run with default settings (600 frames, 1000 objects)
RUST_LOG=info cargo run --example split_vs_single_bg --release -- --out artifacts/perf/I6_bg_churn.csv

# Run with custom parameters  
RUST_LOG=info cargo run --example split_vs_single_bg --release -- --frames 1200 --objects 2000 --out my_results.csv

# Using Python wrapper
python examples/perf/split_vs_single_bg.py --frames 600 --out artifacts/perf/I6_bg_churn.csv

# With verbose output
python examples/perf/split_vs_single_bg.py --verbose --frames 600 --objects 1000
```

### Required Environment

- **Rust**: `cargo` build system with release optimization
- **GPU**: Any WebGPU-compatible adapter (preferably discrete GPU)
- **Output**: CSV file will be created in specified path

## CSV Schema

The output CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Configuration` | String | Test configuration ("SplitBindGroups" or "SingleBindGroup") |
| `TotalTimeMs` | Float | Total time for all frames in milliseconds |
| `AvgFrameTimeMs` | Float | Average time per frame in milliseconds |
| `MinFrameTimeMs` | Float | Minimum frame time in milliseconds |
| `MaxFrameTimeMs` | Float | Maximum frame time in milliseconds |
| `BindGroupSwitches` | Integer | Total number of bind group switches |
| `DrawCalls` | Integer | Total number of draw calls |

### Example Output

```csv
Configuration,TotalTimeMs,AvgFrameTimeMs,MinFrameTimeMs,MaxFrameTimeMs,BindGroupSwitches,DrawCalls
SplitBindGroups,1523.456,2.539,2.123,3.987,600000,600000
SingleBindGroup,892.123,1.487,1.234,2.456,600,600000
```

## Interpretation

### Key Metrics

1. **Frame Time Improvement**: `split_avg_ms / single_avg_ms` ratio
2. **Bind Group Efficiency**: Switches per frame (should be 1 for single vs N for split)
3. **CPU Overhead Reduction**: Total time difference shows CPU savings

### Expected Results

- **Performance Improvement**: Single bind group should be 1.5-3x faster
- **Bind Group Switches**: Split approach switches N times per frame, single approach switches once
- **Consistency**: Single bind group should have more consistent frame times (lower variance)

### Factors Affecting Results

1. **GPU Driver**: Different vendors handle bind group switches with varying efficiency
2. **Object Count**: More objects amplify the bind group churn effect
3. **Uniform Data Size**: Larger per-object data increases the cost difference
4. **GPU Memory**: Dynamic offset approach may have different memory access patterns

## Performance Targets

Based on the roadmap acceptance criteria:

- ✅ **CSV Generation**: Must produce valid CSV with all required columns
- ✅ **Performance Delta**: Results should show measurable improvement with single bind group
- ✅ **Reproducibility**: Multiple runs should show consistent trends
- ✅ **Validation**: Python wrapper validates CSV structure and numeric data

## Integration with CI

This benchmark can be integrated into CI pipelines:

```bash
# Run benchmark in CI
python examples/perf/split_vs_single_bg.py --frames 300 --objects 500 --out ci_results.csv

# Validate performance regression  
python scripts/validate_performance.py ci_results.csv --threshold 1.2
```

## Troubleshooting

### Common Issues

1. **No GPU Available**: Test will fall back to software renderer (much slower)
2. **Validation Errors**: Check that both configurations are present in CSV
3. **Timeout**: Large object counts may exceed 5-minute Python wrapper timeout
4. **Memory Issues**: Very large object counts may hit memory limits

### Debug Steps

```bash
# Check GPU adapter
RUST_LOG=debug cargo run --example split_vs_single_bg --release -- --frames 10

# Manual CSV inspection
head -5 artifacts/perf/I6_bg_churn.csv

# Verify column structure
python -c "import csv; print(list(csv.DictReader(open('artifacts/perf/I6_bg_churn.csv')))[:2])"
```

## Technical Implementation

### Shaders

- `shaders/perf/split_bg.wgsl`: Split bind group approach (per-object uniforms)
- `shaders/perf/single_bg.wgsl`: Single bind group approach (dynamic offsets)

### Bind Group Layouts

**Split Approach:**
```rust
BindGroupLayoutEntry {
    binding: 0,
    visibility: ShaderStages::VERTEX_FRAGMENT,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,  // No dynamic offset
        min_binding_size: None,
    },
    count: None,
}
```

**Single Approach:**
```rust  
BindGroupLayoutEntry {
    binding: 0,
    visibility: ShaderStages::VERTEX_FRAGMENT,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: true,   // Enable dynamic offset
        min_binding_size: Some(std::num::NonZeroU64::new(64).unwrap()),
    },
    count: None,
}
```

This experiment provides concrete data for optimizing bind group usage in WebGPU applications and validates the effectiveness of dynamic offset strategies.