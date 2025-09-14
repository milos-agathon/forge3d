# Workstream A Implementation Plan - Path Tracing GPU Compute

**Target**: Complete missing GPU compute implementation for Workstream A (Path Tracing)  
**Mode**: Implementation (WRITE_CHANGES=true, USE_TESTS=true, ENSURE_CI=true)  
**Branch**: `ws-a-implementation`  
**Scope**: 25 tasks, focusing on P0 foundational components first  

## Implementation Strategy

### Phase 1: GPU Foundation (P0 Tasks)
Focus on establishing the GPU compute pipeline and core infrastructure that other tasks depend on.

### Phase 2: Advanced Features (P1/P2 Tasks)  
Build advanced path tracing features on the established foundation.

## Task-by-Task Implementation Plan

### 🎯 Priority P0 (Foundation) - 5 tasks

#### A1: Compute Path Tracer MVP - GPU Kernel Implementation
**Current Status**: Present but Partial (CPU implementation exists)  
**Missing**: WGSL compute kernels, GPU BVH traversal, Rust backend wiring  

**Implementation**:
1. **Expand WGSL Kernel** (`src/shaders/pt_kernel.wgsl`):
   - Add uniforms for width/height, camera params, scene data  
   - Implement xorshift64* RNG matching CPU version
   - Add ray generation with jitter for anti-aliasing
   - Basic ray-sphere intersection compute
   - HDR accumulation buffer with Reinhard tonemap

2. **Rust Backend Integration** (`src/path_tracing/mod.rs` - new):
   - PathTracerGPU struct with compute pipeline  
   - Buffer management for output, uniforms, scene data
   - Dispatch coordination with tiling support
   - GPU/CPU readback for RGBA output

3. **Python API Bridge** (`python/forge3d/path_tracing.py`):
   - Modify PathTracer.render_rgba() to use GPU when available
   - Fallback to CPU when GPU unavailable  
   - Maintain deterministic output with fixed seeds

**Files to Create/Modify**:
- `src/path_tracing/mod.rs` (new)
- `src/path_tracing/compute.rs` (new)  
- `src/shaders/pt_kernel.wgsl` (expand)
- `python/forge3d/path_tracing.py` (modify)
- `tests/test_path_tracing_gpu.py` (new)

#### A7: LBVH/SAH Builder & Refit - GPU Construction
**Current Status**: Present but Partial (CPU SAH exists)  
**Missing**: GPU LBVH construction, Morton codes, refit functionality

**Implementation**:
1. **Morton Code Generation** (`src/shaders/morton_codes.wgsl` - new):
   - Compute shader to generate 64-bit Morton codes from triangle centroids
   - Spatial binning and sorting preparation

2. **Radix Sort Implementation** (`src/shaders/radix_sort.wgsl` - new):
   - GPU radix sort for Morton code sorting
   - Multiple passes for 64-bit keys

3. **LBVH Construction** (`src/shaders/lbvh_build.wgsl` - new):
   - Parallel BVH node construction from sorted Morton codes
   - Bounding box computation and propagation

4. **Rust Integration** (`src/path_tracing/bvh.rs` - new):
   - BVH GPU builder with memory management
   - CPU SAH fallback for small scenes
   - BVH refitting for dynamic scenes

**Files to Create/Modify**:
- `src/path_tracing/bvh.rs` (new)
- `src/shaders/morton_codes.wgsl` (new)
- `src/shaders/radix_sort.wgsl` (new)  
- `src/shaders/lbvh_build.wgsl` (new)
- `tests/test_bvh_gpu.py` (new)

#### A12: Wavefront Path Tracer - Queue-Based Architecture  
**Current Status**: Absent
**Missing**: Queue-based PT stages, compaction, persistent threads

**Implementation**:
1. **Wavefront Architecture** (`src/shaders/wavefront_pt.wgsl` - new):
   - Separate kernels for ray generation, intersection, shading
   - Ray queue management and compaction
   - Work distribution across wavefront stages

2. **Queue Management** (`src/path_tracing/wavefront.rs` - new):
   - GPU buffer pools for ray queues
   - Dynamic scheduling between stages
   - Memory compaction and reuse

**Files to Create/Modify**:
- `src/path_tracing/wavefront.rs` (new)
- `src/shaders/wavefront_pt.wgsl` (new)
- `tests/test_wavefront_pt.py` (new)

#### A14: AOVs & Debug Outputs - Multi-Buffer Output
**Current Status**: Present but Partial (basic RGBA exists)
**Missing**: Albedo/normal/depth AOVs, EXR export, debug visualization

**Implementation**:  
1. **AOV Buffer Management** (`src/path_tracing/aov.rs` - new):
   - Multiple render targets for albedo, normal, depth, direct/indirect  
   - EXR export functionality with OpenEXR integration
   - Debug visualization modes

2. **WGSL AOV Output** (modify existing shaders):
   - Multi-target output from path tracing kernels
   - Consistent G-buffer layout for denoiser consumption

**Files to Create/Modify**:
- `src/path_tracing/aov.rs` (new)
- `python/forge3d/exr_export.py` (new)  
- Modify existing WGSL kernels for AOV output
- `tests/test_aov_output.py` (new)

#### A16: QMC & Adaptive Sampler - Low-Discrepancy Sequences
**Current Status**: Absent  
**Missing**: Sobol/Owen sequences, blue-noise, per-pixel estimator

**Implementation**:
1. **QMC Sequence Generation** (`src/path_tracing/qmc.rs` - new):
   - Sobol sequence generation with Owen scrambling
   - Blue-noise texture integration  
   - Per-pixel sample distribution

2. **Adaptive Sampling** (`src/shaders/adaptive_sample.wgsl` - new):  
   - Variance estimation per pixel/tile
   - Dynamic sample count allocation
   - Convergence criteria checking

**Files to Create/Modify**:
- `src/path_tracing/qmc.rs` (new)
- `src/shaders/adaptive_sample.wgsl` (new)
- `tests/test_qmc_sampling.py` (new)

### 🔧 Priority P1 (Advanced Features) - 19 tasks

#### A2: Material Models v1 - Complete GPU Implementation  
**Current Status**: Present but Partial (CPU BSDF exists)
**Implementation**: Port CPU BSDF evaluation to WGSL, add environment sampling

#### A3: Triangle Mesh + BVH - GPU Traversal
**Current Status**: Present but Partial (CPU BVH exists)  
**Implementation**: GPU BVH traversal kernels, watertight intersection

#### A4: Light Sampling & MIS - Next Event Estimation
**Implementation**: Area light sampling, multiple importance sampling, light selection

#### A5: Denoiser (A-trous/SVGF) - Post-Process Denoising  
**Implementation**: A-trous wavelet denoiser, SVGF temporal accumulation

#### A6: Dielectric Water (offline) - Advanced Materials
**Implementation**: Beer-Lambert absorption, Fresnel refraction  

#### A8-A25: Remaining Advanced Features
**Implementation**: Progressive implementation based on priority and dependencies

## Memory Budget Compliance

All implementations must respect the **≤512 MiB host-visible heap** constraint:

- **BVH Storage**: Compressed BVH nodes, streaming for large scenes
- **Texture Assets**: Mipmapped textures with LOD management  
- **Buffer Pools**: Reuse compute buffers, avoid allocation churn
- **Monitoring**: Add telemetry for memory usage tracking

## Testing Strategy

### Unit Tests
- Each component has dedicated test file (`test_*_gpu.py`)
- CPU/GPU parity testing for correctness validation
- Performance benchmarking with regression detection

### Integration Tests  
- Cornell box reference scene for visual validation
- Cross-platform testing (Vulkan/Metal/DX12)
- Memory budget compliance verification

### CI Integration
- GPU tests run when adapters available, skip gracefully otherwise
- Golden image comparisons for visual regression detection
- Performance benchmarks with acceptable variance thresholds

## File Structure

```
src/
├── path_tracing/
│   ├── mod.rs           # Main path tracing module
│   ├── compute.rs       # GPU compute pipeline
│   ├── bvh.rs          # BVH builder and traversal  
│   ├── wavefront.rs    # Wavefront PT architecture
│   ├── aov.rs          # AOV buffer management
│   ├── qmc.rs          # QMC sampling
│   ├── materials.rs    # Material evaluation
│   └── denoiser.rs     # Denoising algorithms

shaders/
├── pt_kernel.wgsl       # Main path tracing compute  
├── morton_codes.wgsl    # Morton code generation
├── radix_sort.wgsl      # GPU radix sort
├── lbvh_build.wgsl      # LBVH construction
├── wavefront_pt.wgsl    # Wavefront path tracer
├── adaptive_sample.wgsl # Adaptive sampling
└── denoise.wgsl         # Denoising kernels

python/forge3d/
├── path_tracing.py      # Main API (existing, modify)
└── exr_export.py        # EXR export functionality

tests/
├── test_path_tracing_gpu.py    # GPU implementation tests
├── test_bvh_gpu.py             # BVH GPU tests  
├── test_wavefront_pt.py        # Wavefront PT tests
├── test_aov_output.py          # AOV output tests
└── test_qmc_sampling.py        # QMC sampling tests
```

## Implementation Order

### Week 1-2: A1 GPU Kernel Foundation
1. Expand WGSL compute kernel with basic ray tracing
2. Rust backend integration and buffer management  
3. Python API bridging with GPU/CPU fallback

### Week 3-4: A7 LBVH Construction
1. Morton code generation and radix sort
2. LBVH construction kernels
3. Integration with A1 ray tracing

### Week 5-6: A14 AOVs & A16 QMC  
1. Multi-buffer AOV output system
2. QMC sequence generation and adaptive sampling
3. EXR export functionality

### Week 7-8: A12 Wavefront + Testing
1. Wavefront path tracer architecture  
2. Comprehensive testing and validation
3. Performance optimization and profiling

### Week 9+: P1 Advanced Features
1. A4 Light Sampling & MIS
2. A5 Denoiser implementation
3. Remaining advanced features based on priority

## Risk Mitigation

### Technical Risks
1. **GPU Memory Overflow**: Implement streaming and LOD systems early
2. **Compute Shader Portability**: Use conservative WGSL features, test on all backends
3. **Performance Regression**: Continuous benchmarking with CPU baseline

### Schedule Risks  
1. **Scope Creep**: Focus on P0 tasks first, defer P1 features if needed
2. **Integration Complexity**: Incremental integration with existing CPU path tracer
3. **Testing Overhead**: Parallel test development with implementation

## Success Criteria

### Phase 1 Success (P0 Complete)
- A1: GPU path tracer renders Cornell box scene correctly
- A7: GPU BVH construction handles 1M triangles in <1s  
- A12: Wavefront PT shows measurable performance improvement
- A14: AOV outputs available for denoiser consumption
- A16: QMC sampling reduces noise vs uniform random

### Overall Success (Full Workstream)
- All 25 tasks implemented and tested
- GPU implementation shows ≥2× speedup vs CPU baseline
- Memory usage stays within 512MiB budget  
- Cross-platform compatibility validated
- Cornell box and complex scenes render correctly within error thresholds

## Conclusion

This implementation plan provides a structured approach to completing Workstream A path tracing functionality. The focus on P0 foundation tasks first ensures that advanced features have a solid base to build upon. The extensive testing strategy and risk mitigation measures should help deliver a robust, cross-platform GPU path tracer that meets the performance and quality requirements.