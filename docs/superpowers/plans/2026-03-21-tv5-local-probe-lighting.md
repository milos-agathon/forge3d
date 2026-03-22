# TV5 Local Probe Lighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement terrain-native diffuse GI using heightfield-analytical irradiance probes that make valleys darker and ridges brighter.

**Architecture:** CPU-side SH L2 baker ray-marches the heightfield per probe, uploads packed SH data to a storage buffer, and the fragment shader bilinearly interpolates probe irradiance to replace/blend with the global IBL diffuse term. Probes extend the existing group(6) bind group layout with two new bindings (uniform + SSBO). Fallback bind group with zeroed data ensures pixel-identical behavior when disabled.

**Tech Stack:** Rust (wgpu, bytemuck), WGSL shaders, Python (dataclass + PyO3 bridge), pytest

**Spec:** `docs/superpowers/specs/2026-03-21-tv5-local-probe-lighting-design.md`

---

## File Structure

### New Files (Rust)

| File | Responsibility |
|------|---------------|
| `src/terrain/probes/mod.rs` | Module root, re-exports |
| `src/terrain/probes/types.rs` | `ProbeGridDesc`, `ProbePlacement`, `SHL2`, `ProbeIrradianceSet`, `ProbeError` |
| `src/terrain/probes/baker.rs` | `ProbeBaker` trait |
| `src/terrain/probes/heightfield_baker.rs` | `HeightfieldAnalyticalBaker` — hemisphere ray-march, SH accumulation |
| `src/terrain/probes/gpu.rs` | `GpuProbeData`, `ProbeGridUniformsGpu`, `upload()`, memory tracking |

### New Files (Shader)

| File | Responsibility |
|------|---------------|
| `src/shaders/terrain_probes.wgsl` | `ProbeGridUniforms`, `GpuProbeData` structs, `evaluate_sh_l2()`, `sample_probe_irradiance()` |

### New Files (Python / Tests / Examples)

| File | Responsibility |
|------|---------------|
| `tests/test_terrain_probes.py` | Unit, visual regression, memory, edge blend, invalidation tests |
| `examples/terrain_tv5_probe_lighting_demo.py` | Offscreen demo with DEM, side-by-side comparison, debug viz |

### Modified Files

| File | Change |
|------|--------|
| `src/terrain/mod.rs:86` | Add `pub mod probes;` |
| `src/terrain/renderer/core.rs:86-87` | Add probe SSBO, uniform buffer, fallback buffer, memory counter fields |
| `src/terrain/renderer/constructor.rs:79-86` | Initialize probe fallback uniform + SSBO buffers |
| `src/terrain/renderer/uniforms.rs:59-70` | No change (MaterialLayerUniforms unchanged) |
| `src/terrain/renderer/bind_groups/layouts.rs:88-104` | Extend `create_material_layer_bind_group_layout()` with bindings 1 (uniform) and 2 (storage) |
| `src/terrain/renderer/bind_groups/terrain_pass.rs:317-324` | Add probe uniform + SSBO entries to material layer bind group |
| `src/terrain/renderer/pipeline_cache.rs:55-116` | Add `terrain_probes` to `preprocess_terrain_shader()` concatenation |
| `src/shaders/terrain_pbr_pom.wgsl:145,455,2967-2981` | Add debug constants (slots 50-51), `#include` directive, probe blend in `fs_main` |
| `src/terrain/renderer/py_api.rs:290+` | Add `get_probe_memory_report()` method |
| `src/terrain/renderer/probes.rs` | Probe orchestration: auto-placement, bake, upload |
| `src/terrain/render_params/native_probes.rs` | `ProbeSettingsNative` Rust struct |
| `src/terrain/render_params/decode_probes.rs` | Decode probe settings from Python config |
| `src/terrain/render_params/core.rs:27` | Add `probes: ProbeSettingsNative` to `DecodedTerrainSettings` |
| `src/terrain/render_params.rs` | Add `mod decode_probes; mod native_probes;` |
| `src/terrain/render_params/private_impl.rs` | Call `parse_probe_settings()` during decode |
| `python/forge3d/terrain_params.py` | Add `ProbeSettings` dataclass, integrate into `make_terrain_params_config()` |
| `python/forge3d/__init__.pyi` | Add `ProbeSettings` and `get_probe_memory_report` type stubs |

---

## Task 1: Probe Data Types (Rust)

**Files:**
- Create: `src/terrain/probes/mod.rs`
- Create: `src/terrain/probes/types.rs`
- Modify: `src/terrain/mod.rs:86`

- [ ] **Step 1: Write failing test for ProbeGridDesc and ProbePlacement**

Add to `src/terrain/probes/types.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_placement_invariant() {
        let grid = ProbeGridDesc {
            origin: [0.0, 0.0],
            spacing: [10.0, 10.0],
            dims: [4, 4],
            height_offset: 5.0,
            influence_radius: 0.0,
        };
        let positions: Vec<[f32; 3]> = (0..16).map(|i| [i as f32, 0.0, 0.0]).collect();
        let placement = ProbePlacement::new(grid, positions);
        assert_eq!(placement.positions_ws.len(), 16);
    }

    #[test]
    #[should_panic(expected = "positions_ws.len() must equal dims[0] * dims[1]")]
    fn test_probe_placement_invariant_fails_on_mismatch() {
        let grid = ProbeGridDesc {
            origin: [0.0, 0.0],
            spacing: [10.0, 10.0],
            dims: [4, 4],
            height_offset: 5.0,
            influence_radius: 0.0,
        };
        let positions: Vec<[f32; 3]> = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect();
        ProbePlacement::new(grid, positions); // Should panic
    }

}
// Note: SH packing roundtrip test lives in gpu.rs (Task 2) to avoid circular dependency.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib -p forge3d -- probes::types::tests --test-threads=1`
Expected: FAIL — module/types don't exist yet

- [ ] **Step 3: Implement types**

Create `src/terrain/probes/mod.rs`:

```rust
pub mod types;
pub mod baker;
pub mod heightfield_baker;
pub mod gpu;

pub use types::*;
pub use baker::*;
pub use heightfield_baker::*;
pub use gpu::*;
```

Create `src/terrain/probes/types.rs`:

```rust
/// Grid descriptor — where probes live.
#[derive(Clone, Debug)]
pub struct ProbeGridDesc {
    pub origin: [f32; 2],
    pub spacing: [f32; 2],
    pub dims: [u32; 2],
    pub height_offset: f32,
    pub influence_radius: f32,
}

/// Resolved world-space positions — derived from grid + terrain.
#[derive(Clone, Debug)]
pub struct ProbePlacement {
    pub grid: ProbeGridDesc,
    pub positions_ws: Vec<[f32; 3]>,
}

impl ProbePlacement {
    pub fn new(grid: ProbeGridDesc, positions_ws: Vec<[f32; 3]>) -> Self {
        assert_eq!(
            positions_ws.len(),
            (grid.dims[0] * grid.dims[1]) as usize,
            "positions_ws.len() must equal dims[0] * dims[1]"
        );
        Self { grid, positions_ws }
    }
}

/// SH L2 coefficients: 9 basis functions x RGB.
#[derive(Clone, Debug)]
pub struct SHL2 {
    pub coeffs: [[f32; 3]; 9],
}

/// Baked diffuse irradiance — one SHL2 per placed probe.
#[derive(Clone, Debug)]
pub struct ProbeIrradianceSet {
    pub probes: Vec<SHL2>,
}

/// Probe system errors.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("Probe bake failed: {0}")]
    BakeFailed(String),
}
```

Add to `src/terrain/mod.rs` after line 85 (`pub mod renderer;`):

```rust
pub mod probes;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib -p forge3d -- probes::types::tests --test-threads=1`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/terrain/probes/mod.rs src/terrain/probes/types.rs src/terrain/mod.rs
git commit -m "feat(tv5): add probe data types — ProbeGridDesc, ProbePlacement, SHL2"
```

---

## Task 2: GPU Mirror Types and Packing

**Files:**
- Create: `src/terrain/probes/gpu.rs`

- [ ] **Step 1: Write failing test for GPU layout sizes**

Add to `src/terrain/probes/gpu.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_gpu_layout_size() {
        assert_eq!(std::mem::size_of::<GpuProbeData>(), 144);
        assert_eq!(std::mem::size_of::<ProbeGridUniformsGpu>(), 48);
    }

    #[test]
    fn test_zeroed_gpu_probe_data() {
        let z = GpuProbeData::zeroed();
        let bytes = bytemuck::bytes_of(&z);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_sh_packing_roundtrip() {
        use crate::terrain::probes::types::SHL2;
        let mut sh = SHL2 { coeffs: [[0.0; 3]; 9] };
        for (i, c) in sh.coeffs.iter_mut().enumerate() {
            *c = [(i as f32) * 0.1, (i as f32) * 0.2, (i as f32) * 0.3];
        }
        let gpu = GpuProbeData::from_sh(&sh);
        let roundtrip = gpu.to_sh();
        for i in 0..9 {
            for c in 0..3 {
                assert!((sh.coeffs[i][c] - roundtrip.coeffs[i][c]).abs() < 1e-6,
                    "Mismatch at coeff [{i}][{c}]");
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib -p forge3d -- probes::gpu::tests --test-threads=1`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement GPU types**

Create `src/terrain/probes/gpu.rs`:

```rust
use bytemuck::{Pod, Zeroable};
use crate::terrain::probes::types::SHL2;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ProbeGridUniformsGpu {
    pub grid_origin: [f32; 4],   // xy=origin, z=height_offset, w=enabled
    pub grid_params: [f32; 4],   // x=spacing_x, y=spacing_y, z=dims_x, w=dims_y
    pub blend_params: [f32; 4],  // x=fallback_blend_distance, y=probe_count, zw=pad
}

impl ProbeGridUniformsGpu {
    pub fn disabled() -> Self {
        Self {
            grid_origin: [0.0, 0.0, 0.0, 0.0], // w=0.0 => disabled
            grid_params: [1.0, 1.0, 1.0, 1.0],
            blend_params: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuProbeData {
    pub sh_r_01: [f32; 4],
    pub sh_r_23: [f32; 4],
    pub sh_r_4:  [f32; 4],
    pub sh_g_01: [f32; 4],
    pub sh_g_23: [f32; 4],
    pub sh_g_4:  [f32; 4],
    pub sh_b_01: [f32; 4],
    pub sh_b_23: [f32; 4],
    pub sh_b_4:  [f32; 4],
}

impl GpuProbeData {
    /// Pack CPU-side SHL2 (interleaved RGB per basis) into channel-major GPU layout.
    pub fn from_sh(sh: &SHL2) -> Self {
        let c = &sh.coeffs;
        Self {
            sh_r_01: [c[0][0], c[1][0], c[2][0], c[3][0]],
            sh_r_23: [c[4][0], c[5][0], c[6][0], c[7][0]],
            sh_r_4:  [c[8][0], 0.0, 0.0, 0.0],
            sh_g_01: [c[0][1], c[1][1], c[2][1], c[3][1]],
            sh_g_23: [c[4][1], c[5][1], c[6][1], c[7][1]],
            sh_g_4:  [c[8][1], 0.0, 0.0, 0.0],
            sh_b_01: [c[0][2], c[1][2], c[2][2], c[3][2]],
            sh_b_23: [c[4][2], c[5][2], c[6][2], c[7][2]],
            sh_b_4:  [c[8][2], 0.0, 0.0, 0.0],
        }
    }

    /// Unpack channel-major GPU layout back to interleaved SHL2 (for testing roundtrip).
    pub fn to_sh(&self) -> SHL2 {
        let mut coeffs = [[0.0f32; 3]; 9];
        let r = [self.sh_r_01, self.sh_r_23, self.sh_r_4];
        let g = [self.sh_g_01, self.sh_g_23, self.sh_g_4];
        let b = [self.sh_b_01, self.sh_b_23, self.sh_b_4];
        for i in 0..9 {
            let vec_idx = i / 4;
            let comp_idx = i % 4;
            coeffs[i][0] = r[vec_idx][comp_idx];
            coeffs[i][1] = g[vec_idx][comp_idx];
            coeffs[i][2] = b[vec_idx][comp_idx];
        }
        SHL2 { coeffs }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib -p forge3d -- probes --test-threads=1`
Expected: PASS (all 5 tests from tasks 1+2)

- [ ] **Step 5: Commit**

```bash
git add src/terrain/probes/gpu.rs
git commit -m "feat(tv5): add GPU mirror types — GpuProbeData (144B), ProbeGridUniformsGpu (48B)"
```

---

## Task 3: ProbeBaker Trait and HeightfieldAnalyticalBaker

**Files:**
- Create: `src/terrain/probes/baker.rs`
- Create: `src/terrain/probes/heightfield_baker.rs`

- [ ] **Step 1: Write failing test for baker determinism**

Add to `src/terrain/probes/heightfield_baker.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::probes::types::*;

    fn flat_heightfield(dim: u32) -> Vec<f32> {
        vec![0.0; (dim * dim) as usize]
    }

    fn test_grid(dims: [u32; 2]) -> ProbeGridDesc {
        ProbeGridDesc {
            origin: [-50.0, -50.0],
            spacing: [100.0 / (dims[0].max(2) - 1) as f32, 100.0 / (dims[1].max(2) - 1) as f32],
            dims,
            height_offset: 5.0,
            influence_radius: 0.0,
        }
    }

    #[test]
    fn test_probe_bake_deterministic() {
        let dim = 64u32;
        let grid = test_grid([4, 4]);
        let baker = HeightfieldAnalyticalBaker {
            heightfield: flat_heightfield(dim),
            height_dims: (dim, dim),
            terrain_span: [100.0, 100.0],
            sky_color: [0.6, 0.75, 1.0],
            sky_intensity: 1.0,
            ray_count: 64,
            max_trace_distance: 50.0,
        };
        let positions: Vec<[f32; 3]> = (0..16)
            .map(|i| {
                let col = (i % 4) as f32;
                let row = (i / 4) as f32;
                [grid.origin[0] + grid.spacing[0] * col,
                 grid.origin[1] + grid.spacing[1] * row,
                 5.0]
            })
            .collect();
        let placement = ProbePlacement::new(grid.clone(), positions);

        let result1 = baker.bake(&placement).unwrap();
        let result2 = baker.bake(&placement).unwrap();

        for (i, (p1, p2)) in result1.probes.iter().zip(result2.probes.iter()).enumerate() {
            for j in 0..9 {
                for c in 0..3 {
                    assert_eq!(p1.coeffs[j][c], p2.coeffs[j][c],
                        "Non-deterministic at probe {i}, coeff [{j}][{c}]");
                }
            }
        }
    }

    #[test]
    fn test_flat_terrain_unoccluded() {
        // Flat terrain = no occlusion, all probes should have non-zero L0 (DC term)
        let dim = 32u32;
        let grid = test_grid([2, 2]);
        let baker = HeightfieldAnalyticalBaker {
            heightfield: flat_heightfield(dim),
            height_dims: (dim, dim),
            terrain_span: [100.0, 100.0],
            sky_color: [1.0, 1.0, 1.0],
            sky_intensity: 1.0,
            ray_count: 64,
            max_trace_distance: 50.0,
        };
        let positions = vec![
            [-50.0, -50.0, 5.0],
            [50.0, -50.0, 5.0],
            [-50.0, 50.0, 5.0],
            [50.0, 50.0, 5.0],
        ];
        let placement = ProbePlacement::new(grid, positions);
        let result = baker.bake(&placement).unwrap();

        for (i, probe) in result.probes.iter().enumerate() {
            // L0 DC term (index 0) should be positive for sky-visible probes
            assert!(probe.coeffs[0][0] > 0.0,
                "Probe {i} L0 R should be > 0, got {}", probe.coeffs[0][0]);
        }
    }

    #[test]
    fn test_nodata_heightfield_no_nan() {
        let dim = 32u32;
        let mut hf = flat_heightfield(dim);
        // Inject NaN/infinity
        hf[0] = f32::NAN;
        hf[1] = f32::INFINITY;
        hf[2] = f32::NEG_INFINITY;

        let grid = test_grid([2, 2]);
        let baker = HeightfieldAnalyticalBaker {
            heightfield: hf,
            height_dims: (dim, dim),
            terrain_span: [100.0, 100.0],
            sky_color: [1.0, 1.0, 1.0],
            sky_intensity: 1.0,
            ray_count: 32,
            max_trace_distance: 50.0,
        };
        let positions = vec![
            [-50.0, -50.0, 5.0],
            [50.0, -50.0, 5.0],
            [-50.0, 50.0, 5.0],
            [50.0, 50.0, 5.0],
        ];
        let placement = ProbePlacement::new(grid, positions);
        let result = baker.bake(&placement).unwrap();

        for (i, probe) in result.probes.iter().enumerate() {
            for j in 0..9 {
                for c in 0..3 {
                    assert!(!probe.coeffs[j][c].is_nan(),
                        "NaN at probe {i}, coeff [{j}][{c}]");
                    assert!(probe.coeffs[j][c].is_finite(),
                        "Infinite at probe {i}, coeff [{j}][{c}]");
                }
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib -p forge3d -- probes::heightfield_baker::tests --test-threads=1`
Expected: FAIL — baker not implemented

- [ ] **Step 3: Implement baker trait**

Create `src/terrain/probes/baker.rs`:

```rust
use crate::terrain::probes::types::*;

/// Abstract bake backend — placement in, irradiance out.
pub trait ProbeBaker {
    fn bake(&self, placement: &ProbePlacement) -> Result<ProbeIrradianceSet, ProbeError>;
}
```

- [ ] **Step 4: Implement HeightfieldAnalyticalBaker**

Create `src/terrain/probes/heightfield_baker.rs`:

```rust
use crate::terrain::probes::baker::ProbeBaker;
use crate::terrain::probes::types::*;

pub struct HeightfieldAnalyticalBaker {
    pub heightfield: Vec<f32>,
    pub height_dims: (u32, u32),
    pub terrain_span: [f32; 2],
    pub sky_color: [f32; 3],
    pub sky_intensity: f32,
    pub ray_count: u32,
    pub max_trace_distance: f32,
}

impl HeightfieldAnalyticalBaker {
    /// Sample heightfield with bilinear interpolation. Returns 0.0 for out-of-bounds or NaN/Inf.
    fn sample_height(&self, world_x: f32, world_y: f32) -> f32 {
        let (w, h) = self.height_dims;
        // World to UV: terrain centered at origin, span = terrain_span
        let u = (world_x / self.terrain_span[0]) + 0.5;
        let v = (world_y / self.terrain_span[1]) + 0.5;

        let fx = u * (w - 1) as f32;
        let fy = v * (h - 1) as f32;

        if fx < 0.0 || fy < 0.0 || fx >= (w - 1) as f32 || fy >= (h - 1) as f32 {
            return 0.0;
        }

        let ix = fx.floor() as u32;
        let iy = fy.floor() as u32;
        let tx = fx.fract();
        let ty = fy.fract();

        let idx = |x: u32, y: u32| -> f32 {
            let val = self.heightfield[(y * w + x) as usize];
            if val.is_nan() || val.is_infinite() { 0.0 } else { val }
        };

        let h00 = idx(ix, iy);
        let h10 = idx((ix + 1).min(w - 1), iy);
        let h01 = idx(ix, (iy + 1).min(h - 1));
        let h11 = idx((ix + 1).min(w - 1), (iy + 1).min(h - 1));

        let mix_a = h00 + (h10 - h00) * tx;
        let mix_b = h01 + (h11 - h01) * tx;
        mix_a + (mix_b - mix_a) * ty
    }

    /// Generate stratified Fibonacci spiral directions on upper hemisphere.
    fn hemisphere_directions(count: u32) -> Vec<[f32; 3]> {
        let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
        (0..count)
            .map(|i| {
                let theta = (i as f32 / golden_ratio).fract() * 2.0 * std::f32::consts::PI;
                let cos_phi = 1.0 - (2.0 * i as f32 + 1.0) / (2.0 * count as f32);
                let cos_phi = cos_phi.max(0.0); // Upper hemisphere only
                let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
                [sin_phi * theta.cos(), sin_phi * theta.sin(), cos_phi]
            })
            .collect()
    }

    /// SH L2 basis function evaluation for direction d.
    fn sh_basis(d: &[f32; 3]) -> [f32; 9] {
        let (x, y, z) = (d[0], d[1], d[2]);
        [
            0.282095,                              // Y00
            0.488603 * y,                           // Y1,-1
            0.488603 * z,                           // Y1,0
            0.488603 * x,                           // Y1,1
            1.092548 * x * y,                       // Y2,-2
            1.092548 * y * z,                       // Y2,-1
            0.315392 * (3.0 * z * z - 1.0),        // Y2,0
            1.092548 * x * z,                       // Y2,1
            0.546274 * (x * x - y * y),             // Y2,2
        ]
    }

    /// Ray-march along heightfield from probe position in given direction.
    /// Returns true if the ray is occluded.
    fn trace_ray(&self, origin: &[f32; 3], dir: &[f32; 3], max_dist: f32) -> bool {
        let step_count = 100;
        let step_size = max_dist / step_count as f32;

        for i in 1..=step_count {
            let t = step_size * i as f32;
            let ray_x = origin[0] + dir[0] * t;
            let ray_y = origin[1] + dir[1] * t;
            let ray_z = origin[2] + dir[2] * t;

            let terrain_h = self.sample_height(ray_x, ray_y);
            if terrain_h > ray_z {
                return true; // Occluded
            }
        }
        false
    }
}

impl ProbeBaker for HeightfieldAnalyticalBaker {
    fn bake(&self, placement: &ProbePlacement) -> Result<ProbeIrradianceSet, ProbeError> {
        let directions = Self::hemisphere_directions(self.ray_count);
        let solid_angle = 2.0 * std::f32::consts::PI / self.ray_count as f32;

        let probes: Vec<SHL2> = placement
            .positions_ws
            .iter()
            .map(|pos| {
                let mut coeffs = [[0.0f32; 3]; 9];

                for dir in &directions {
                    let cos_theta = dir[2].max(0.0); // z = cos(elevation)
                    if cos_theta <= 0.0 {
                        continue;
                    }

                    let occluded = self.trace_ray(pos, dir, self.max_trace_distance);

                    if !occluded {
                        let basis = Self::sh_basis(dir);
                        let weight = cos_theta * solid_angle;
                        for l in 0..9 {
                            coeffs[l][0] += self.sky_color[0] * self.sky_intensity * basis[l] * weight;
                            coeffs[l][1] += self.sky_color[1] * self.sky_intensity * basis[l] * weight;
                            coeffs[l][2] += self.sky_color[2] * self.sky_intensity * basis[l] * weight;
                        }
                    }
                }

                SHL2 { coeffs }
            })
            .collect();

        Ok(ProbeIrradianceSet { probes })
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib -p forge3d -- probes --test-threads=1`
Expected: PASS (all tests from tasks 1-3)

- [ ] **Step 6: Commit**

```bash
git add src/terrain/probes/baker.rs src/terrain/probes/heightfield_baker.rs
git commit -m "feat(tv5): implement HeightfieldAnalyticalBaker — SH L2 hemisphere ray-march"
```

---

## Task 4: WGSL Shader — terrain_probes.wgsl

**Files:**
- Create: `src/shaders/terrain_probes.wgsl`

- [ ] **Step 1: Create the probe shader include**

Create `src/shaders/terrain_probes.wgsl` with the full shader code from spec Section 4.1:

```wgsl
// TV5: Local probe lighting for terrain scenes
// Included by terrain_pbr_pom.wgsl via preprocess_terrain_shader()

struct ProbeGridUniforms {
    // xy = world origin of grid, z = height_offset, w = enabled (1.0 = on, 0.0 = off)
    grid_origin:   vec4<f32>,
    // x = spacing_x, y = spacing_y, z = f32(dims_x), w = f32(dims_y)
    grid_params:   vec4<f32>,
    // x = fallback_blend_distance (world-space meters), y = f32(probe_count), zw = pad
    blend_params:  vec4<f32>,
};

struct GpuProbeData {
    sh_r_01: vec4<f32>,
    sh_r_23: vec4<f32>,
    sh_r_4:  vec4<f32>,
    sh_g_01: vec4<f32>,
    sh_g_23: vec4<f32>,
    sh_g_4:  vec4<f32>,
    sh_b_01: vec4<f32>,
    sh_b_23: vec4<f32>,
    sh_b_4:  vec4<f32>,
};

@group(6) @binding(1)
var<uniform> probe_grid: ProbeGridUniforms;

@group(6) @binding(2)
var<storage, read> probe_data: array<GpuProbeData>;

struct ProbeIrradianceResult {
    irradiance: vec3<f32>,
    weight: f32,
};

fn evaluate_sh_l2(n: vec3<f32>, probe: GpuProbeData) -> vec3<f32> {
    let Y00  = 0.282095;
    let Y1m1 = 0.488603 * n.y;
    let Y10  = 0.488603 * n.z;
    let Y11  = 0.488603 * n.x;
    let Y2m2 = 1.092548 * n.x * n.y;
    let Y2m1 = 1.092548 * n.y * n.z;
    let Y20  = 0.315392 * (3.0 * n.z * n.z - 1.0);
    let Y21  = 1.092548 * n.x * n.z;
    let Y22  = 0.546274 * (n.x * n.x - n.y * n.y);

    let basis_01 = vec4<f32>(Y00, Y1m1, Y10, Y11);
    let basis_23 = vec4<f32>(Y2m2, Y2m1, Y20, Y21);

    var result: vec3<f32>;
    result.r = dot(probe.sh_r_01, basis_01) + dot(probe.sh_r_23, basis_23) + probe.sh_r_4.x * Y22;
    result.g = dot(probe.sh_g_01, basis_01) + dot(probe.sh_g_23, basis_23) + probe.sh_g_4.x * Y22;
    result.b = dot(probe.sh_b_01, basis_01) + dot(probe.sh_b_23, basis_23) + probe.sh_b_4.x * Y22;

    return max(result, vec3(0.0));
}

fn sample_probe_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> ProbeIrradianceResult {
    var result: ProbeIrradianceResult;
    result.irradiance = vec3(0.0);
    result.weight = 0.0;

    if (probe_grid.grid_origin.w < 0.5) {
        return result;
    }

    let dims = vec2<u32>(u32(probe_grid.grid_params.z), u32(probe_grid.grid_params.w));
    let spacing = probe_grid.grid_params.xy;

    let grid_uv = (world_pos.xy - probe_grid.grid_origin.xy) / spacing;

    let cell = clamp(grid_uv - vec2(0.5), vec2(0.0), vec2<f32>(dims - vec2(1u)));
    let i0 = vec2<u32>(floor(cell));
    let frac = fract(cell);

    let idx00 = i0.y * dims.x + i0.x;
    let idx10 = i0.y * dims.x + min(i0.x + 1u, dims.x - 1u);
    let idx01 = min(i0.y + 1u, dims.y - 1u) * dims.x + i0.x;
    let idx11 = min(i0.y + 1u, dims.y - 1u) * dims.x + min(i0.x + 1u, dims.x - 1u);

    let sh00 = evaluate_sh_l2(normal, probe_data[idx00]);
    let sh10 = evaluate_sh_l2(normal, probe_data[idx10]);
    let sh01 = evaluate_sh_l2(normal, probe_data[idx01]);
    let sh11 = evaluate_sh_l2(normal, probe_data[idx11]);

    let bilinear = mix(mix(sh00, sh10, frac.x), mix(sh01, sh11, frac.x), frac.y);

    let fallback_dist = probe_grid.blend_params.x;
    let grid_extent = vec2<f32>(dims - vec2(1u));

    let sentinel = fallback_dist + 1.0;
    let dist_x = select(min(grid_uv.x, grid_extent.x - grid_uv.x) * spacing.x,
                         sentinel, grid_extent.x < 0.5);
    let dist_y = select(min(grid_uv.y, grid_extent.y - grid_uv.y) * spacing.y,
                         sentinel, grid_extent.y < 0.5);
    let dist_to_edge = min(dist_x, dist_y);
    let edge_weight = saturate(dist_to_edge / max(fallback_dist, 1e-6));

    result.irradiance = bilinear;
    result.weight = edge_weight;
    return result;
}
```

- [ ] **Step 2: Verify file created**

Run: `ls -la src/shaders/terrain_probes.wgsl`
Expected: File exists

- [ ] **Step 3: Commit**

```bash
git add src/shaders/terrain_probes.wgsl
git commit -m "feat(tv5): add terrain_probes.wgsl — SH L2 evaluation and grid bilinear sampling"
```

---

## Task 5: Extend Group(6) Bind Group Layout

**Files:**
- Modify: `src/terrain/renderer/bind_groups/layouts.rs:88-104`

- [ ] **Step 1: Extend layout with bindings 1 and 2**

In `src/terrain/renderer/bind_groups/layouts.rs`, replace the `create_material_layer_bind_group_layout` function (lines 88-104) to add probe uniform (binding 1) and probe SSBO (binding 2):

```rust
    pub(in crate::terrain::renderer) fn create_material_layer_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_pbr_pom.material_layer_bind_group_layout"),
            entries: &[
                // Binding 0: MaterialLayerUniforms (existing)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: ProbeGridUniforms (TV5)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: probe_data SSBO (TV5)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p forge3d 2>&1 | head -20`
Expected: Compilation errors expected (bind group creation doesn't match yet — that's Task 7)

- [ ] **Step 3: Commit**

```bash
git add src/terrain/renderer/bind_groups/layouts.rs
git commit -m "feat(tv5): extend group(6) layout with probe uniform (binding 1) and SSBO (binding 2)"
```

---

## Task 6: Add Probe Fields to TerrainScene and Constructor

**Files:**
- Modify: `src/terrain/renderer/core.rs:86-87`
- Modify: `src/terrain/renderer/constructor.rs:79-86`

- [ ] **Step 1: Add fields to TerrainScene**

In `src/terrain/renderer/core.rs`, after line 87 (`material_layer_uniform_buffer`), add:

```rust
    // TV5: Probe lighting
    pub(super) probe_grid_uniform_buffer: wgpu::Buffer,
    pub(super) probe_ssbo: wgpu::Buffer,
    pub(super) probe_grid_uniform_bytes: u64,
    pub(super) probe_ssbo_bytes: u64,
```

- [ ] **Step 2: Initialize probe fallback resources in constructor**

In `src/terrain/renderer/constructor.rs`, after the `material_layer_uniform_buffer` creation (line 86), add:

```rust
        // TV5: Probe fallback resources (zeroed = disabled)
        let probe_grid_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.probe_grid.uniform_buffer.fallback"),
                contents: bytemuck::bytes_of(&crate::terrain::probes::ProbeGridUniformsGpu::disabled()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let probe_ssbo =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.probe_data.ssbo.fallback"),
                contents: bytemuck::bytes_of(&crate::terrain::probes::GpuProbeData::zeroed()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
```

Also add the new fields to the `TerrainScene { ... }` struct initialization in the constructor, setting `probe_grid_uniform_bytes: 0` and `probe_ssbo_bytes: 0`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p forge3d 2>&1 | head -20`
Expected: May have remaining errors from bind group creation — addressed in Task 7.

- [ ] **Step 4: Commit**

```bash
git add src/terrain/renderer/core.rs src/terrain/renderer/constructor.rs
git commit -m "feat(tv5): add probe buffer fields to TerrainScene, init fallback resources"
```

---

## Task 7: Wire Probe Buffers into Bind Group Creation

**Files:**
- Modify: `src/terrain/renderer/bind_groups/terrain_pass.rs:317-324`

- [ ] **Step 1: Add probe entries to material layer bind group**

In `src/terrain/renderer/bind_groups/terrain_pass.rs`, replace the bind group creation at lines 317-324:

```rust
        let material_layer = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.material_layer.bind_group"),
            layout: &self.material_layer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.material_layer_uniform_buffer.as_entire_binding(),
                },
                // TV5: Probe grid uniforms
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.probe_grid_uniform_buffer.as_entire_binding(),
                },
                // TV5: Probe SH data SSBO
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.probe_ssbo.as_entire_binding(),
                },
            ],
        });
```

- [ ] **Step 2: Verify full compilation**

Run: `cargo check -p forge3d 2>&1 | tail -5`
Expected: PASS (no errors — all bind group entries match layout)

- [ ] **Step 3: Commit**

```bash
git add src/terrain/renderer/bind_groups/terrain_pass.rs
git commit -m "feat(tv5): wire probe uniform + SSBO into group(6) bind group"
```

---

## Task 8: Shader Preprocessing — Add terrain_probes.wgsl

**Files:**
- Modify: `src/terrain/renderer/pipeline_cache.rs:55-116`

- [ ] **Step 1: Add terrain_probes to shader concatenation**

In `src/terrain/renderer/pipeline_cache.rs`, after the `terrain_noise` include (line 90), add:

```rust
        let terrain_probes = include_str!("../../shaders/terrain_probes.wgsl");
```

And update the format string (line 98) to insert `terrain_probes` before `terrain`:

Change the format from 16 to 17 arguments, inserting `terrain_probes` between `terrain_noise` and `terrain`:

```rust
        format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            lights,
            brdf_common,
            brdf_lambert,
            brdf_phong,
            brdf_oren_nayar,
            brdf_cook_torrance,
            brdf_disney_principled,
            brdf_ashikhmin_shirley,
            brdf_ward,
            brdf_toon,
            brdf_minnaert,
            brdf_dispatch,
            lighting,
            lighting_ibl,
            terrain_noise,
            terrain_probes,
            terrain
        )
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p forge3d 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/terrain/renderer/pipeline_cache.rs
git commit -m "feat(tv5): add terrain_probes.wgsl to shader preprocessing pipeline"
```

---

## Task 9: Integrate Probes in Fragment Shader (fs_main)

**Files:**
- Modify: `src/shaders/terrain_pbr_pom.wgsl:145,2967-2981`

- [ ] **Step 1: Add `#include` directive and debug mode constants**

In `src/shaders/terrain_pbr_pom.wgsl`, after line 50 (`#include "terrain_noise.wgsl"`), add for consistency with existing convention:

```wgsl
// TV5: Local probe lighting (SH L2 evaluation and grid sampling)
#include "terrain_probes.wgsl"
```

Then, after line 145 (`const DBG_VIEW_POS_XYZ: u32 = 42u;`), add:

```wgsl
// TV5: Probe lighting debug modes
const DBG_PROBE_IRRADIANCE: u32 = 50u;  // Raw probe irradiance * weight (tonemapped)
const DBG_PROBE_WEIGHT: u32 = 51u;      // Probe weight as grayscale
```

- [ ] **Step 2: Add probe blend in fs_main**

Replace lines 2979-2981 in `terrain_pbr_pom.wgsl`:

From:
```wgsl
    let ibl_diffuse_with_shadow = ibl_split.diffuse * shadow_factor;
    let ibl_with_shadow = ibl_diffuse_with_shadow + ibl_split.specular;
    ibl_contrib = ibl_with_shadow * u_ibl.intensity * ibl_occlusion;
```

To:
```wgsl
    // TV5: Blend probe irradiance into diffuse IBL term
    // input.world_position is set by the vertex shader (line 1318) — its XY is
    // sufficiently accurate for the low-frequency bilinear probe grid lookup.
    // shading_normal (line 2600) is the final surface normal used for all lighting.
    let probe_result = sample_probe_irradiance(input.world_position, shading_normal);
    let kS_ibl = ibl_split.fresnel;
    let kD_ibl = (vec3<f32>(1.0) - kS_ibl) * (1.0 - metallic);
    let global_diffuse = ibl_split.diffuse;
    let probe_diffuse = kD_ibl * ibl_albedo * probe_result.irradiance;
    let blended_diffuse = mix(global_diffuse, probe_diffuse, probe_result.weight);

    let ibl_diffuse_with_shadow = blended_diffuse * shadow_factor;
    let ibl_with_shadow = ibl_diffuse_with_shadow + ibl_split.specular;
    ibl_contrib = ibl_with_shadow * u_ibl.intensity * ibl_occlusion;
```

**Variable scope notes:**
- `input.world_position` (line 1155, `VertexOutput`) — used throughout fs_main (lines 2585, 2915, 2927). The vertex shader (line 1317-1318) sets this to `vec3(world_xy, world_z_original)`. While the comment at line 1027 warns about interpolation inaccuracy in fullscreen-triangle mode, the probe grid sampling is low-frequency (bilinear over ~10s of meters), so the error is negligible.
- `shading_normal` (line 2600) — the final surface normal after blending, water override, and detail normals. This is the correct normal for SH evaluation.

- [ ] **Step 3: Add debug mode handlers**

In the debug mode block (after the last existing debug mode handler, around line 3180+), add:

```wgsl
    } else if (debug_mode == 50u) {
        // DBG_PROBE_IRRADIANCE: Show probe contribution
        let probe_dbg = sample_probe_irradiance(input.world_position, shading_normal);
        final_color = probe_dbg.irradiance * probe_dbg.weight;
    } else if (debug_mode == 51u) {
        // DBG_PROBE_WEIGHT: Show probe weight as grayscale
        let probe_dbg = sample_probe_irradiance(input.world_position, shading_normal);
        final_color = vec3<f32>(probe_dbg.weight);
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p forge3d 2>&1 | tail -10`
Expected: PASS — shader compiles at pipeline creation time, so Rust check passes. Runtime validation happens at first render.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/terrain_pbr_pom.wgsl
git commit -m "feat(tv5): integrate probe irradiance blend in fs_main, add debug modes 50-51"
```

---

## Task 10: Probe Upload and Memory Tracking

**Files:**
- Modify: `src/terrain/probes/gpu.rs` (extend with upload logic)
- Modify: `src/terrain/renderer/core.rs` (add upload method)

- [ ] **Step 1: Write failing test for memory tracking**

Add to `src/terrain/probes/gpu.rs` tests:

```rust
    #[test]
    fn test_probe_upload_sizes() {
        let probes: Vec<SHL2> = (0..16).map(|_| SHL2 { coeffs: [[0.0; 3]; 9] }).collect();
        let set = ProbeIrradianceSet { probes };
        let gpu_data = pack_probes_for_upload(&set);
        assert_eq!(gpu_data.len(), 16);
        assert_eq!(std::mem::size_of_val(gpu_data.as_slice()), 16 * 144);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib -p forge3d -- probes::gpu::tests::test_probe_upload_sizes --test-threads=1`
Expected: FAIL — `pack_probes_for_upload` not defined

- [ ] **Step 3: Implement pack function and upload method**

Add to `src/terrain/probes/gpu.rs`:

```rust
use crate::terrain::probes::types::{ProbeIrradianceSet, SHL2};

/// Pack a ProbeIrradianceSet into GPU-ready data.
pub fn pack_probes_for_upload(set: &ProbeIrradianceSet) -> Vec<GpuProbeData> {
    set.probes.iter().map(GpuProbeData::from_sh).collect()
}
```

Add upload method to `TerrainScene` in a new file or extend `core.rs`:

```rust
impl TerrainScene {
    pub(super) fn upload_probe_data(
        &mut self,
        grid_uniforms: &ProbeGridUniformsGpu,
        probe_data: &[GpuProbeData],
    ) {
        // Write grid uniforms
        self.queue.write_buffer(
            &self.probe_grid_uniform_buffer,
            0,
            bytemuck::bytes_of(grid_uniforms),
        );

        // Recreate SSBO if size changed
        let required_bytes = (probe_data.len() * std::mem::size_of::<GpuProbeData>()) as u64;
        if required_bytes != self.probe_ssbo_bytes || required_bytes == 0 {
            // Free old tracking
            if self.probe_ssbo_bytes > 0 {
                crate::core::memory_tracker::global_tracker()
                    .free_buffer_allocation(self.probe_ssbo_bytes, false);
            }

            self.probe_ssbo = self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("terrain.probe_data.ssbo"),
                    contents: bytemuck::cast_slice(probe_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            );
            self.probe_ssbo_bytes = required_bytes;
            self.probe_grid_uniform_bytes = std::mem::size_of::<ProbeGridUniformsGpu>() as u64;

            // Track new allocation
            crate::core::memory_tracker::global_tracker()
                .track_buffer_allocation(required_bytes, false);
        } else {
            self.queue.write_buffer(
                &self.probe_ssbo,
                0,
                bytemuck::cast_slice(probe_data),
            );
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib -p forge3d -- probes::gpu::tests --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/terrain/probes/gpu.rs src/terrain/renderer/core.rs
git commit -m "feat(tv5): implement probe upload with SSBO resize and memory tracking"
```

---

## Task 11: Python API — ProbeSettings Dataclass

**Files:**
- Modify: `python/forge3d/terrain_params.py`

- [ ] **Step 1: Add ProbeSettings dataclass**

Add to `python/forge3d/terrain_params.py` after the existing dataclass imports:

```python
@dataclass
class ProbeSettings:
    """TV5: Irradiance probe configuration for terrain scenes."""
    enabled: bool = False
    grid_dims: Tuple[int, int] = (8, 8)
    origin: Optional[Tuple[float, float]] = None
    spacing: Optional[Tuple[float, float]] = None
    height_offset: float = 5.0
    ray_count: int = 64
    fallback_blend_distance: Optional[float] = None
    sky_color: Tuple[float, float, float] = (0.6, 0.75, 1.0)
    sky_intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.enabled:
            cols, rows = self.grid_dims
            if cols < 1 or rows < 1:
                raise ValueError("grid_dims must be >= (1, 1)")
            if cols * rows > 4096:
                raise ValueError("grid_dims product must be <= 4096 (probe count limit)")
```

- [ ] **Step 2: Add `probes` parameter to `make_terrain_params_config()`**

Add `probes: Optional[ProbeSettings] = None` to the function signature and include probe settings in the config dict when enabled.

- [ ] **Step 3: Commit**

```bash
git add python/forge3d/terrain_params.py
git commit -m "feat(tv5): add ProbeSettings dataclass and integrate into terrain params"
```

---

## Task 12: Rust-Side Probe Settings Decode (render_params)

**Files:**
- Create: `src/terrain/render_params/native_probes.rs`
- Create: `src/terrain/render_params/decode_probes.rs`
- Modify: `src/terrain/render_params/core.rs:4-28` (add `probes` to `DecodedTerrainSettings`)
- Modify: `src/terrain/render_params.rs` (add module declarations)
- Modify: `src/terrain/render_params/private_impl.rs` (call decoder)

- [ ] **Step 1: Create native probe settings struct**

Create `src/terrain/render_params/native_probes.rs`:

```rust
/// TV5: Native representation of probe settings, decoded from Python config.
#[derive(Clone, Debug)]
pub struct ProbeSettingsNative {
    pub enabled: bool,
    pub grid_dims: (u32, u32),
    pub origin: Option<(f32, f32)>,
    pub spacing: Option<(f32, f32)>,
    pub height_offset: f32,
    pub ray_count: u32,
    pub fallback_blend_distance: Option<f32>,
    pub sky_color: [f32; 3],
    pub sky_intensity: f32,
}

impl Default for ProbeSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            grid_dims: (8, 8),
            origin: None,
            spacing: None,
            height_offset: 5.0,
            ray_count: 64,
            fallback_blend_distance: None,
            sky_color: [0.6, 0.75, 1.0],
            sky_intensity: 1.0,
        }
    }
}
```

- [ ] **Step 2: Create decoder**

Create `src/terrain/render_params/decode_probes.rs` following the pattern in `decode_materials.rs`:

```rust
use super::*;

pub(super) fn parse_probe_settings(
    params: &Bound<'_, PyAny>,
) -> ProbeSettingsNative {
    if let Ok(probes) = params.getattr("probes") {
        let enabled: bool = probes.getattr("enabled")
            .and_then(|v| v.extract()).unwrap_or(false);
        if !enabled {
            return ProbeSettingsNative::default();
        }
        let grid_dims: (u32, u32) = probes.getattr("grid_dims")
            .and_then(|v| v.extract()).unwrap_or((8, 8));
        let origin: Option<(f32, f32)> = probes.getattr("origin")
            .and_then(|v| v.extract()).ok();
        let spacing: Option<(f32, f32)> = probes.getattr("spacing")
            .and_then(|v| v.extract()).ok();
        let height_offset: f32 = probes.getattr("height_offset")
            .and_then(|v| v.extract()).unwrap_or(5.0);
        let ray_count: u32 = probes.getattr("ray_count")
            .and_then(|v| v.extract()).unwrap_or(64);
        let fallback_blend_distance: Option<f32> = probes.getattr("fallback_blend_distance")
            .and_then(|v| v.extract()).ok();
        let sky_color_vec: Vec<f32> = probes.getattr("sky_color")
            .and_then(|v| v.extract()).unwrap_or_else(|_| vec![0.6, 0.75, 1.0]);
        let sky_color = [
            sky_color_vec.first().copied().unwrap_or(0.6),
            sky_color_vec.get(1).copied().unwrap_or(0.75),
            sky_color_vec.get(2).copied().unwrap_or(1.0),
        ];
        let sky_intensity: f32 = probes.getattr("sky_intensity")
            .and_then(|v| v.extract()).unwrap_or(1.0);

        ProbeSettingsNative {
            enabled, grid_dims, origin, spacing, height_offset,
            ray_count, fallback_blend_distance, sky_color, sky_intensity,
        }
    } else {
        ProbeSettingsNative::default()
    }
}
```

- [ ] **Step 3: Wire into DecodedTerrainSettings**

In `src/terrain/render_params/core.rs`, add after line 27 (`pub sky: SkySettingsNative`):

```rust
    pub probes: ProbeSettingsNative,
```

In `src/terrain/render_params.rs`, add module declarations:

```rust
mod decode_probes;
mod native_probes;
```

And add the import:

```rust
use native_probes::ProbeSettingsNative;
```

In `src/terrain/render_params/private_impl.rs`, add the probe decode call alongside other decode calls.

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p forge3d 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/terrain/render_params/native_probes.rs src/terrain/render_params/decode_probes.rs src/terrain/render_params/core.rs src/terrain/render_params.rs src/terrain/render_params/private_impl.rs
git commit -m "feat(tv5): add Rust-side probe settings decode from Python config"
```

---

## Task 13: PyO3 Bridge — get_probe_memory_report()
<!-- Note: Tasks renumbered after inserting Task 12 (render_params decode). -->

**Files:**
- Modify: `src/terrain/renderer/py_api.rs:290+`

- [ ] **Step 1: Add get_probe_memory_report method**

In `src/terrain/renderer/py_api.rs`, after `get_scatter_memory_report` (line 290+), add:

```rust
    #[pyo3(signature = ())]
    pub fn get_probe_memory_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let probe_count = if self.scene.probe_ssbo_bytes > 0 {
            (self.scene.probe_ssbo_bytes / 144) as u64
        } else {
            0
        };
        dict.set_item("probe_count", probe_count)?;
        dict.set_item("grid_uniform_bytes", self.scene.probe_grid_uniform_bytes)?;
        dict.set_item("probe_ssbo_bytes", self.scene.probe_ssbo_bytes)?;
        dict.set_item("total_bytes",
            self.scene.probe_grid_uniform_bytes + self.scene.probe_ssbo_bytes)?;
        Ok(dict.into())
    }
```

- [ ] **Step 2: Update type stubs**

Add to `python/forge3d/__init__.pyi` after `get_scatter_memory_report`:

```python
    def get_probe_memory_report(self) -> Dict[str, Any]: ...
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p forge3d 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/terrain/renderer/py_api.rs python/forge3d/__init__.pyi
git commit -m "feat(tv5): expose get_probe_memory_report() via PyO3"
```

---

## Task 14: Probe Bake Integration in Render Path

**Files:**
- Create: `src/terrain/renderer/probes.rs`
- Modify: `src/terrain/renderer/draw/mod.rs:9-18`

- [ ] **Step 1: Create probe orchestration module**

Create `src/terrain/renderer/probes.rs` with auto-placement resolution, bake, and upload:

```rust
use crate::terrain::probes::*;
use crate::terrain::render_params::native_probes::ProbeSettingsNative;

/// Resolve auto-placement from probe settings + heightfield.
/// Implements spec Section 7.3 rules.
pub(super) fn resolve_placement(
    settings: &ProbeSettingsNative,
    terrain_span: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
) -> ProbePlacement {
    let (cols, rows) = settings.grid_dims;
    let half_span = terrain_span / 2.0;

    // Resolve spacing and origin per-axis (spec Section 7.3)
    let (spacing_x, origin_x) = if cols > 1 {
        let s = terrain_span / (cols - 1) as f32;
        (s, -half_span)
    } else {
        (terrain_span, 0.0) // Single probe centered
    };
    let (spacing_y, origin_y) = if rows > 1 {
        let s = terrain_span / (rows - 1) as f32;
        (s, -half_span)
    } else {
        (terrain_span, 0.0)
    };

    let origin = settings.origin.unwrap_or((origin_x, origin_y));
    let spacing = settings.spacing.unwrap_or((spacing_x, spacing_y));

    let grid = ProbeGridDesc {
        origin: [origin.0, origin.1],
        spacing: [spacing.0, spacing.1],
        dims: [cols, rows],
        height_offset: settings.height_offset,
        influence_radius: 0.0,
    };

    // Compute world-space positions by sampling heightfield
    let (hw, hh) = height_dims;
    let positions: Vec<[f32; 3]> = (0..(cols * rows))
        .map(|idx| {
            let col = idx % cols;
            let row = idx / cols;
            let wx = origin.0 + spacing.0 * col as f32;
            let wy = origin.1 + spacing.1 * row as f32;

            // Sample heightfield at world XY (bilinear)
            let u = (wx / terrain_span) + 0.5;
            let v = (wy / terrain_span) + 0.5;
            let fx = (u * (hw - 1) as f32).clamp(0.0, (hw - 1) as f32);
            let fy = (v * (hh - 1) as f32).clamp(0.0, (hh - 1) as f32);
            let ix = fx.floor() as u32;
            let iy = fy.floor() as u32;
            let h = heightfield[(iy * hw + ix) as usize];
            let h = if h.is_finite() { h } else { 0.0 };
            let wz = h * z_scale + settings.height_offset;

            [wx, wy, wz]
        })
        .collect();

    ProbePlacement::new(grid, positions)
}

/// Run full probe pipeline: resolve placement, bake, upload.
pub(super) fn prepare_probes(
    scene: &mut super::TerrainScene,
    settings: &ProbeSettingsNative,
    terrain_span: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
) {
    if !settings.enabled {
        // Upload disabled uniform (weight=0, fallback to global IBL)
        scene.upload_probe_data(
            &ProbeGridUniformsGpu::disabled(),
            &[GpuProbeData::zeroed()],
        );
        return;
    }

    let placement = resolve_placement(settings, terrain_span, heightfield, height_dims, z_scale);
    let (cols, rows) = settings.grid_dims;

    let baker = HeightfieldAnalyticalBaker {
        heightfield: heightfield.to_vec(),
        height_dims,
        terrain_span: [terrain_span, terrain_span],
        sky_color: settings.sky_color,
        sky_intensity: settings.sky_intensity,
        ray_count: settings.ray_count,
        max_trace_distance: terrain_span * 0.5,
    };

    let irradiance = baker.bake(&placement).expect("Heightfield baker is infallible");
    let gpu_data = pack_probes_for_upload(&irradiance);

    let spacing = settings.spacing.unwrap_or_else(|| {
        let sx = if cols > 1 { terrain_span / (cols - 1) as f32 } else { terrain_span };
        let sy = if rows > 1 { terrain_span / (rows - 1) as f32 } else { terrain_span };
        (sx, sy)
    });
    let blend_dist = settings.fallback_blend_distance
        .unwrap_or(spacing.0.min(spacing.1) * 2.0);

    let grid_uniforms = ProbeGridUniformsGpu {
        grid_origin: [
            placement.grid.origin[0],
            placement.grid.origin[1],
            settings.height_offset,
            1.0, // enabled
        ],
        grid_params: [spacing.0, spacing.1, cols as f32, rows as f32],
        blend_params: [blend_dist, (cols * rows) as f32, 0.0, 0.0],
    };

    scene.upload_probe_data(&grid_uniforms, &gpu_data);
}
```

- [ ] **Step 2: Wire into render_internal**

In `src/terrain/renderer/draw/mod.rs`, inside `render_internal()` (line 17-18, after `self.prepare_frame_lighting(decoded)?`), add the probe preparation call:

```rust
        // TV5: Bake and upload probes if enabled
        super::probes::prepare_probes(
            self,
            &decoded.probes,
            self.terrain_span_from_params(params),
            heightfield_slice,
            (hm_width, hm_height),
            params.z_scale,
        );
```

The exact insertion point is after heightfield data is available but before bind groups are created.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p forge3d 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/terrain/renderer/probes.rs src/terrain/renderer/draw/mod.rs
git commit -m "feat(tv5): integrate probe bake + upload into terrain render path with auto-placement"
```

---

## Task 15: Python Tests

**Files:**
- Create: `tests/test_terrain_probes.py`

- [ ] **Step 1: Write test file with all spec test cases**

Create `tests/test_terrain_probes.py`:

```python
"""TV5: Local probe lighting tests."""
import numpy as np
import pytest
from pathlib import Path

# Runtime detection (skip if no GPU)
from tests._terrain_runtime import require_terrain_runtime
require_terrain_runtime()

import forge3d as f3d
from forge3d.terrain_params import ProbeSettings, make_terrain_params_config


class TestProbeSettingsValidation:
    """Spec 10.6: Out-of-bounds and degenerate tests."""

    def test_probe_grid_dims_validation_zero(self):
        with pytest.raises(ValueError, match="grid_dims must be >= \\(1, 1\\)"):
            ProbeSettings(enabled=True, grid_dims=(0, 0))

    def test_probe_grid_dims_validation_too_large(self):
        with pytest.raises(ValueError, match="probe count limit"):
            ProbeSettings(enabled=True, grid_dims=(65, 65))

    def test_probe_grid_dims_single_probe(self):
        s = ProbeSettings(enabled=True, grid_dims=(1, 1))
        assert s.grid_dims == (1, 1)

    def test_probe_defaults_disabled(self):
        s = ProbeSettings()
        assert s.enabled is False


@pytest.fixture(scope="module")
def renderer():
    session = f3d.Session(window=False)
    return f3d.TerrainRenderer(session)


@pytest.fixture(scope="module")
def test_heightmap():
    """Synthetic heightmap with a valley."""
    size = 256
    hm = np.zeros((size, size), dtype=np.float32)
    # Create valley: low center, high edges
    for y in range(size):
        for x in range(size):
            dx = (x - size / 2) / (size / 2)
            dy = (y - size / 2) / (size / 2)
            hm[y, x] = (dx * dx + dy * dy) * 500.0  # Bowl shape
    return hm


class TestProbeMemory:
    """Spec 10.4: Memory tracking."""

    def test_probe_memory_tracked(self, renderer, test_heightmap):
        # Render with probes enabled
        # ... render call with ProbeSettings(enabled=True, grid_dims=(4, 4)) ...
        report = renderer.get_probe_memory_report()
        probe_count = report["probe_count"]
        expected = 48 + probe_count * 144
        assert report["total_bytes"] == expected


class TestProbePixelIdentity:
    """Spec 10.2: Probes disabled = pixel-identical to baseline."""

    def test_probe_fallback_pixel_identical(self, renderer, test_heightmap):
        # Render baseline (no probes)
        # ... render without probes ...
        # Render with probes disabled (fallback)
        # ... render with ProbeSettings(enabled=False) ...
        # Compare: frames must be identical
        pass  # Filled in during implementation


class TestProbeVisualRegression:
    """Spec 10.3: Visual regression tests."""

    def test_probe_valley_darker(self, renderer, test_heightmap):
        # Valley scene with probes should have lower luminance in center
        pass  # Filled in during implementation with actual render calls

    def test_probe_ridge_brighter(self, renderer, test_heightmap):
        # Ridge (edge) with probes should have higher relative luminance
        pass  # Filled in during implementation with actual render calls


class TestProbeEdgeBlend:
    """Spec 10.5: Edge blend smoothness."""

    def test_probe_edge_blend_smooth(self, renderer, test_heightmap):
        # Fragments at grid boundary blend smoothly (no hard seam)
        # Render with small grid, check luminance transition at grid edge
        pass  # Filled in during implementation


class TestProbeOutOfBounds:
    """Spec 10.6: Out-of-bounds weight."""

    def test_probe_out_of_bounds_weight_zero(self, renderer, test_heightmap):
        # Probe grid covering only a small region of terrain; fragments far
        # outside should get weight=0.0 (full IBL fallback).
        # Use debug_mode=51 (DBG_PROBE_WEIGHT) and check that pixels outside
        # grid are black (weight=0).
        pass  # Filled in during implementation


class TestProbeInvalidation:
    """Spec 10.7: Invalidation behavior."""

    def test_probe_invalidation_triggers(self, renderer, test_heightmap):
        # Changing sky_color or grid_dims triggers rebake (different output);
        # changing camera position does not (same output).
        pass  # Filled in during implementation
```

- [ ] **Step 2: Run tests to verify they work**

Run: `pytest tests/test_terrain_probes.py -v --tb=short 2>&1 | tail -20`
Expected: Validation tests PASS, render tests may SKIP or PASS depending on GPU

- [ ] **Step 3: Commit**

```bash
git add tests/test_terrain_probes.py
git commit -m "test(tv5): add probe lighting test suite — validation, memory, visual regression"
```

---

## Task 16: Example Demo

**Files:**
- Create: `examples/terrain_tv5_probe_lighting_demo.py`

- [ ] **Step 1: Create demo script**

Create `examples/terrain_tv5_probe_lighting_demo.py` following the pattern from `terrain_tv4_material_variation_demo.py`:

The demo should:
1. Load a real DEM (Gore Range or synthetic)
2. Render offscreen: probes disabled vs probes enabled
3. Render probe debug mode visualization (mode 50, 51)
4. Print memory report
5. Save PNG output

- [ ] **Step 2: Run demo to verify output**

Run: `python examples/terrain_tv5_probe_lighting_demo.py --output /tmp/tv5_demo.png`
Expected: PNG files created, memory report printed

- [ ] **Step 3: Commit**

```bash
git add examples/terrain_tv5_probe_lighting_demo.py
git commit -m "feat(tv5): add probe lighting demo — offscreen comparison + debug viz"
```

---

## Task 17: End-to-End Verification

- [ ] **Step 1: Run full Rust test suite**

Run: `cargo test --workspace -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile`
Expected: All tests PASS including new probe tests

- [ ] **Step 2: Run Python test suite**

Run: `pytest tests/test_terrain_probes.py tests/test_terrain_materials.py -v --tb=short`
Expected: All PASS

- [ ] **Step 3: Run visual regression (if GPU available)**

Run: `pytest tests/test_terrain_visual_goldens.py -v --tb=short`
Expected: Existing goldens still PASS (probes disabled by default = pixel-identical)

- [ ] **Step 4: Final commit with any fixups**

```bash
git add -A
git commit -m "feat(tv5): complete local probe lighting implementation"
```
