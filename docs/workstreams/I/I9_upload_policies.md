# I9: Upload Policy Benchmark and Analysis

**Status**: Complete  
**Benchmark**: Upload Strategy Performance Comparison  
**Files**: `bench/upload_policies/policies.rs`, `ci/run_benches.sh`

## Overview

This benchmark compares different WebGPU buffer upload strategies to identify the optimal approach for large per-frame data transfers. The goal is to maximize throughput (MB/s) and minimize CPU overhead when uploading data to GPU buffers.

## Upload Strategies Tested

### 1. queue.writeBuffer
- **Description**: Direct upload using `queue.write_buffer()`
- **Use Case**: Simple, immediate data transfer
- **Pros**: Easy to use, no memory management complexity
- **Cons**: May have higher CPU overhead for large transfers

### 2. mappedAtCreation
- **Description**: Create buffer with `mapped_at_creation: true`, write directly to mapped memory
- **Use Case**: One-time buffer initialization with immediate data
- **Pros**: Potentially lower overhead, direct memory access
- **Cons**: Creates new buffer each time, memory allocation overhead

### 3. stagingRing
- **Description**: Persistent ring buffer with mapped memory for staging uploads
- **Use Case**: Frequent, regular data uploads with predictable sizes
- **Pros**: Amortized allocation cost, reduced memory churn
- **Cons**: More complex implementation, requires ring buffer management

## Benchmark Configuration

The benchmark uses the following default parameters:

- **Data Size**: 1 MB per upload iteration
- **Iterations**: 100 upload operations
- **Warmup**: 10 iterations (excluded from measurements)
- **Metrics**: Throughput (MB/s), CPU time, average time per upload

## Running the Benchmark

### Manual Execution

```bash
# Build and run benchmark directly
cargo build --release --bin policies
cargo run --release --bin policies

# With specific environment override
FORGE3D_UPLOAD_POLICY=mappedAtCreation cargo run --release --bin policies
```

### CI Integration

The benchmark is integrated into CI via the `ci/run_benches.sh` script:

```bash
# Run complete CI benchmark suite
bash ci/run_benches.sh

# With environment override testing
FORGE3D_UPLOAD_POLICY=stagingRing bash ci/run_benches.sh
```

## Output Artifacts

The benchmark generates a detailed report at `artifacts/perf/I9_upload_policies.md` containing:

### Performance Table
| Policy | Throughput (MB/s) | Avg Time (ms) | CPU Time (ms) |
|--------|-------------------|---------------|---------------|
| queue.writeBuffer | X.XX | X.XXX | X.XXX |
| mappedAtCreation ⭐ | X.XX | X.XXX | X.XXX |
| stagingRing | X.XX | X.XXX | X.XXX |

*⭐ indicates the selected default policy*

### Performance Summary
- **Selected Default**: Best performing policy based on throughput
- **Performance Improvement**: Percentage improvement over slowest policy
- **Acceptance Criteria**: Must achieve ≥15% improvement over slowest approach

## Environment Override

You can override the default policy selection using the `FORGE3D_UPLOAD_POLICY` environment variable:

```bash
export FORGE3D_UPLOAD_POLICY=mappedAtCreation
# Your application will now use this policy instead of the benchmarked default
```

**Valid Policy Names:**
- `queue.writeBuffer`
- `mappedAtCreation`
- `stagingRing`

## CI Publication

The CI system automatically:

1. **Builds** the benchmark in release mode
2. **Executes** the benchmark with default settings
3. **Tests** environment override functionality
4. **Publishes** the performance report as a CI artifact
5. **Validates** that the artifact contains expected performance data

### CI Artifact Location

The performance report is published to:
```
artifacts/perf/I9_upload_policies.md
```

This artifact can be downloaded from CI runs to track performance regression over time.

## Acceptance Criteria

The benchmark must meet these criteria:

1. **I9-AC1**: `ci/run_benches.sh` exits with code 0 and creates `artifacts/perf/I9_upload_policies.md`
2. **I9-AC2**: Environment override test shows the specified policy in the MD output

## Performance Targets

Based on the workstream requirements:

- ✅ **Throughput Measurement**: All policies must report MB/s throughput
- ✅ **CPU Time Tracking**: CPU overhead must be measured separately from total time
- ✅ **Regression Detection**: ≥15% improvement required over slowest policy
- ✅ **Environment Override**: Must respect `FORGE3D_UPLOAD_POLICY` variable
- ✅ **CI Integration**: Fully automated with artifact generation

## Interpreting Results

### Performance Indicators

1. **Throughput (MB/s)**: Higher is better - indicates raw data transfer speed
2. **Avg Time (ms)**: Lower is better - per-upload latency
3. **CPU Time (ms)**: Lower is better - CPU overhead for the operation

### Factors Affecting Performance

- **GPU Driver**: Different vendors optimize upload paths differently
- **Memory Type**: Host-visible vs device-local memory impact
- **Buffer Usage**: Storage vs uniform buffer usage patterns
- **Data Size**: Larger transfers may favor different strategies
- **Frequency**: Upload frequency affects ring buffer effectiveness

## Troubleshooting

### Common Issues

1. **No GPU Available**: Benchmark falls back to software rendering (much slower)
2. **Build Failures**: Ensure `cargo build --release --bin policies` succeeds
3. **Missing Artifacts**: Check that `artifacts/perf/` directory permissions allow writes
4. **Environment Override Ignored**: Verify policy name spelling matches exactly

### Debug Steps

```bash
# Test GPU availability
RUST_LOG=debug cargo run --release --bin policies

# Validate artifact structure
ls -la artifacts/perf/I9_upload_policies.md

# Check environment variable
echo $FORGE3D_UPLOAD_POLICY

# Verify benchmark binary exists
ls -la target/release/policies
```

## Integration with forge3d

The benchmark results inform the default upload strategy used throughout the forge3d library. The selected policy optimizes:

- **Per-frame uniform buffer uploads** (camera matrices, lighting parameters)
- **Dynamic geometry data** (particle systems, procedural meshes)
- **Texture streaming** (when using buffer-to-texture copies)
- **Compute shader input data** (large datasets for processing)

This ensures optimal performance across different GPU vendors and driver versions while maintaining a simple API for forge3d users.