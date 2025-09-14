# Workstream A (Path Tracing) - Repository Audit

**Date**: September 14, 2025  
**Workstream**: A - Path Tracing (WebGPU Compute, Offline HQ)  
**Repository**: forge3d  
**Task Mode**: Verify & Implement (WRITE_CHANGES=true, USE_TESTS=true, ENSURE_CI=true)  

## Executive Summary

Workstream A contains 25 tasks covering path tracing implementation from basic MVP (A1) to advanced features like participating media (A11) and path guiding (A13). The audit reveals **significant existing infrastructure** with CPU-based implementations already present for core features A1-A3, but **missing GPU compute implementation** and many advanced features.

**Overall Status**: ~15% Complete (4/25 tasks have meaningful artifacts)

## Task-by-Task Audit

### ✅ PRESENT & WIRED (4 tasks)

#### A1: Compute Path Tracer MVP
- **Status**: Present but Partial (CPU implementation exists, GPU compute missing)
- **Evidence**: `python/forge3d/path_tracing.py:30-103`
- **Features**: CPU RNG (xorshift64*), HDR accumulation, tiled rendering, basic ray casting
- **Missing**: WGSL compute kernels, GPU BVH traversal
- **Tests**: `tests/test_path_tracing_a1.py:9-43`

#### A2: Material Models v1  
- **Status**: Present but Partial (Lambertian, GGX metal, dielectric implemented in CPU)
- **Evidence**: `python/forge3d/path_tracing.py:272-334`
- **Features**: Lambert BSDF, GGX specular, Fresnel terms, material dispatch
- **Missing**: GPU shader implementation, env sampling
- **Tests**: Material validation in BVH trace function

#### A3: Triangle Mesh + BVH
- **Status**: Present but Partial (CPU BVH with SAH splitting implemented)
- **Evidence**: `python/forge3d/path_tracing.py:336-577`
- **Features**: CPU BVH builder, SAH splitting, watertight Möller-Trumbore intersection
- **Missing**: GPU BVH layout, traversal kernels
- **Tests**: BVH functionality tested via triangle tracing

#### A7: LBVH/SAH Builder & Refit
- **Status**: Present but Partial (SAH builder implemented, LBVH missing)
- **Evidence**: `python/forge3d/path_tracing.py:366-474`
- **Features**: CPU SAH with binning, BVH reordering for cache coherency
- **Missing**: GPU LBVH construction, Morton codes, refit functionality

### 🔄 PRESENT BUT PARTIAL (1 task)

#### A14: AOVs & Debug Outputs  
- **Status**: Present but Partial (basic output structure exists)
- **Evidence**: `python/forge3d/path_tracing.py:55-102` (RGBA output)
- **Features**: Basic RGBA output structure
- **Missing**: Albedo/normal/depth AOVs, EXR export, debug visualization

### ❌ ABSENT (20 tasks)

Missing entirely: A4 (Light Sampling), A5 (Denoiser), A6 (Dielectric Water), A8 (ReSTIR), A9 (PBR Textures), A10 (SDF Primitives), A11 (Participating Media), A12 (Wavefront), A13 (Path Guiding), A15 (Progressive), A16 (QMC Sampler), A17 (Firefly Clamp), A18 (Ground Plane), A19 (Scene Cache), A20 (Soft Lights), A21 (AO Integrator), A22 (Instanced Geometry), A23 (Hair BSDF), A24 (Anisotropic BRDF), A25 (Object Importance)

## Infrastructure Assessment

### Existing Assets
- **Python API**: Complete CPU path tracer with factory function
- **WGSL Placeholder**: `src/shaders/pt_kernel.wgsl` (8×8 workgroup, black output)
- **Test Coverage**: Basic A1 validation, RNG determinism, HDR tonemap  
- **Documentation**: Path tracing module documented in code comments

### GPU Implementation Gap
- **Compute Pipeline**: Placeholder WGSL exists but not wired to Rust backend
- **Buffer Management**: No GPU buffer allocation for BVH/triangle data  
- **Kernel Dispatch**: No compute pass integration with main rendering
- **Memory Budget**: 512MiB constraint not enforced in path tracing paths

## Platform & Backend Support

**Environment**:
- Rust: 1.88.0
- Python: 3.13.5  
- CMake: 3.31.6
- Sphinx: 8.2.3
- Maturin: 1.9.4

**Target Platforms**: win_amd64, linux_x86_64, macos_universal2  
**GPU APIs**: WebGPU/WGSL primary, Vulkan 1.2-compatible design  

## Risk Assessment

### High Priority Risks
1. **GPU Memory Budget**: Path tracing BVH/textures may exceed 512MiB limit
2. **Compute Shader Portability**: Complex PT kernels may hit driver differences  
3. **Performance Expectations**: CPU implementation sets baseline; GPU must show improvement
4. **Feature Completeness**: 20/25 tasks missing may indicate scope too large

### Mitigations
1. Implement streaming/LOD for large scenes, BVH compression
2. Use conservative WGSL features, comprehensive backend testing
3. Profile CPU vs GPU, optimize bottlenecks systematically  
4. Prioritize P0 tasks (A1, A7, A12, A14, A16) for initial GPU implementation

## Next Steps

Based on task priority and current readiness:

### Phase 1: GPU Foundation (P0 tasks)
1. **A1**: Wire WGSL compute kernel to Rust backend  
2. **A7**: GPU LBVH builder with Morton codes
3. **A12**: Wavefront path tracer architecture  
4. **A14**: AOV outputs and debug visualization
5. **A16**: QMC sampler integration

### Phase 2: Advanced Features (P1 tasks)  
1. **A4**: Light sampling with MIS
2. **A5**: Denoiser (A-trous/SVGF)
3. **A2**: Complete material models (env sampling)
4. **A9**: PBR texture integration

### Dependencies
- Most tasks depend on A1 (compute foundation)
- A5 (Denoiser) depends on A14 (AOVs)  
- A8 (ReSTIR) depends on A4 (Light Sampling)

## Validation Requirements

- All implementations must stay within 512MiB GPU memory budget
- CPU/GPU parity testing for correctness validation
- Cornell box reference scene for visual validation  
- Performance targets: ≥20 Mray/s on mid-tier GPU (A3)
- Cross-platform testing on Vulkan/Metal/DX12

## Conclusion

Workstream A has solid CPU foundations but requires substantial GPU compute implementation. The existing CPU path tracer provides excellent reference implementation and test coverage. Priority should be given to A1 GPU kernel wiring and A7 LBVH implementation to establish the compute pipeline foundation.