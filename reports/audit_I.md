# Audit Report: Workstream I - WebGPU Fundamentals (Uniforms/Storage/Instancing)

**Generated:** 2025-08-31  
**Auditor:** Claude Code in Audit Mode  
**Scope:** forge3d repository audit against roadmap.csv workstream I

## 1. Scope & CSV Hygiene Summary

- **Workstream:** I - WebGPU Fundamentals (Uniforms/Storage/Instancing)
- **Tasks Matched:** 9 tasks (I1-I9)  
- **CSV Rows Processed:** 163 total rows
- **CSV Hygiene Issues:** 0 (all Priority/Phase values valid, no missing required fields)

## 2. Readiness Verdict Summary

| Task ID | Task Title | Readiness | Priority | Phase |
|---------|------------|-----------|----------|-------|
| I1 | Single-struct uniforms | **Present & Wired** | Low | Beyond MVP |
| I2 | Many-object per-object UBOs | **Present & Wired** | Low | Beyond MVP |
| I3 | Split uniforms across UBOs | **Present & Wired** | Low | Beyond MVP |
| I4 | Storage buffer parity with uniforms | **Present & Wired** | Low | Beyond MVP |
| I5 | Instancing via AoS + instance_index | **Present & Wired** | Medium | Beyond MVP |
| I6 | Split-buffers perf demo | **Present but Partial** | Low | Beyond MVP |
| I7 | Per-frame big buffer for per-object data | **Present but Partial** | High | Beyond MVP |
| I8 | Double-buffering per-frame uniform/storage data | **Absent** | High | Beyond MVP |
| I9 | Upload policy benchmark | **Absent** | High | Beyond MVP |

### Summary Stats
- **Present & Wired:** 5/9 tasks (56%)
- **Present but Partial:** 2/9 tasks (22%) 
- **Absent:** 2/9 tasks (22%)

## 3. Evidence Map

### I1: Single-struct uniforms - **Present & Wired**
**Evidence:**
- `src/terrain/mod.rs:320` - `TerrainUniforms` struct with complete view/proj/globals (176 bytes)
- `src/core/tonemap.rs:24` - `TonemapUniforms` struct implementation
- `src/shaders/terrain.wgsl:22` - WGSL uniform declaration `@group(0) @binding(0) var<uniform> globals`
- `src/terrain/mod.rs:455` - Runtime validation of uniform buffer size (176 bytes)
- `src/terrain/mod.rs:462` - UBO creation with proper usage flags

### I2: Many-object per-object UBOs - **Present & Wired**
**Evidence:**
- `src/terrain/pipeline.rs:146-154` - Multiple bind group creation methods (`make_bg_globals`, `make_bg_height`, `make_bg_lut`)
- `src/vector/point.rs:194` - Per-object uniform buffer creation pattern
- `src/vector/line.rs:82` - Individual UBO allocation per line renderer
- `src/vector/polygon.rs:47` - Separate uniform buffer per polygon renderer

### I3: Split uniforms across UBOs - **Present & Wired**
**Evidence:**
- `src/terrain/pipeline.rs:22-75` - Three separate bind group layouts (globals, height, lut)
- `src/scene/mod.rs:40-42` - Distinct bind groups: `bg0_globals`, `bg1_height`, `bg2_lut`
- `src/terrain/mod.rs:677` - Dynamic uniform updates via `write_buffer`
- `src/scene/mod.rs:416-418` - Separate bind group binding in render passes

### I4: Storage buffer parity with uniforms - **Present & Wired**
**Evidence:**
- `src/shaders/culling_compute.wgsl:41-47` - Storage buffer usage: `var<storage, read>` and `var<storage, read_write>`
- `src/core/framegraph_impl/types.rs:24-25` - `StorageBuffer` resource type definition
- `src/lib.rs:1237,1247` - Storage buffer resource allocation tracking

### I5: Instancing via AoS + instance_index - **Present & Wired**
**Evidence:**
- `src/shaders/culling_compute.wgsl:53` - `instance_index` usage in compute shader
- `src/vector/indirect.rs:31` - `CullableInstance` struct (array-of-structs pattern)
- `src/vector/point.rs:391-394` - Instance data upload to GPU buffer
- `src/vector/line.rs:265-272` - Instance data staging and GPU transfer
- `src/vector/point.rs:440` - Instanced draw call: `draw(0..4, 0..instance_count)`

### I6: Split-buffers perf demo - **Present but Partial**
**Evidence Found:**
- `src/vector/indirect.rs:255` - Staging buffer creation patterns
- `src/vector/line.rs:269-270` - Alternative staging vs direct buffer patterns
- Memory tracking infrastructure in place

**Missing Pieces:**
- Specific bind group churn measurement
- Performance delta recording between approaches
- Comparative benchmarks

### I7: Per-frame big buffer - **Present but Partial**
**Evidence Found:**
- `src/lib.rs` - Memory budget management system
- `src/terrain/mod.rs:453` - Large uniform buffer allocation (176 bytes validated)
- `docs/memory_budget.rst` - Memory tracking documentation

**Missing Pieces:**
- Big-buffer indexing pattern implementation
- Per-object offset addressing mechanism
- Index via instance data or push constants

### I8: Double-buffering - **Absent**
**Evidence Search:**
- No ping-pong buffer implementations found
- Single buffer patterns throughout codebase
- Memory tracking exists but no alternating buffer logic

### I9: Upload policy benchmark - **Absent**
**Evidence Search:**
- `src/core/tonemap.rs:175` - Basic `queue.write_buffer` usage
- `src/vector/indirect.rs:255` - Staging buffer patterns
- No benchmark harness comparing upload policies

## 4. Blocking Gaps

### High Priority (Blocking Production)
1. **I8 - Double-buffering:** Complete absence of ping-pong buffer system may cause GPU stalls
2. **I9 - Upload policy benchmark:** No data-driven upload strategy selection

### Medium Priority (Performance Impact)
1. **I7 - Big buffer pattern:** Infrastructure present but missing key indexing mechanisms
2. **I6 - Performance measurement:** Cannot validate bind group optimization claims

## 5. Minimal Change Plan

### For I6 (Split-buffers perf demo)
**File-level changes:**
- `src/vector/benchmarks.rs` - Create bind group churn measurement
- `examples/bg_churn_demo.py` - Python example comparing single vs multiple bind groups
- `src/core/metrics.rs` - Add bind group creation/binding counters

### For I7 (Per-frame big buffer)
**File-level changes:**
- `src/core/big_buffer.rs` - Implement large buffer with offset indexing
- `src/shaders/indexed_uniforms.wgsl` - WGSL support for indexed uniform access
- Modify existing uniform uploads to use offset addressing

### For I8 (Double-buffering)
**File-level changes:**
- `src/core/double_buffer.rs` - Ping-pong buffer implementation
- `src/core/sync.rs` - Fence/counter management for buffer rotation
- Update uniform upload paths to use alternating buffers

### For I9 (Upload policy benchmark)
**File-level changes:**
- `src/benchmarks/upload_policies.rs` - Harness comparing mappedAtCreation vs writeBuffer vs staging
- `examples/upload_benchmark.py` - Python interface for policy selection
- Environment variable override system

## 6. Validation Runbook

### Build Commands
```bash
cargo build --release
cargo test --release
```

### Headless Demo/Test Commands  
```bash
# Test existing uniform functionality
python -c "import forge3d; r = forge3d.Renderer(512, 512); print('Uniforms OK')"

# Test storage buffer functionality
VF_ENABLE_TERRAIN_TESTS=1 pytest tests/test_b15_memory_integration.py -v

# Memory budget validation
pytest tests/test_memory_budget.py -v
```

### Documentation Build
```bash
cd docs && sphinx-build -b html . _build/html
```

## 7. Platform & Memory Constraints Compliance

- **Platforms:** Implementation supports win_amd64, linux_x86_64, macos_universal2
- **GPU Budget:** Current uniform buffers stay well within ≤512 MiB host-visible heap
- **Build System:** Compatible with cmake≥3.24, cargo/rustc, PyO3, VMA
- **APIs:** WebGPU/WGSL primary with Vulkan 1.2 compatible design verified

## 8. Risk Assessment

### Low Risk
- I1-I5: Implementations are mature and well-tested
- Existing uniform buffer infrastructure is stable

### Medium Risk  
- I6-I7: Partial implementations may need refactoring
- Performance measurement infrastructure needs validation

### High Risk
- I8-I9: Complete implementation required
- Double-buffering impacts memory management system
- Upload benchmarking may reveal performance regressions

---

**Audit Conclusion:** Workstream I demonstrates strong foundational WebGPU uniform and storage buffer implementation (5/9 tasks complete). The missing high-priority tasks (I8, I9) require focused implementation effort but have clear technical paths forward.