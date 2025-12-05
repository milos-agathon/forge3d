## Terrain Rendering Improvement Plan v2

### Architecture Context

```
┌─────────────────────────────────────────────────────────────────────┐
│  TerrainRenderer (current state)                                    │
│  ├── Single forward pass (terrain_pbr_pom.wgsl)                    │
│  ├── IBL via IblGpuResources ✅                                     │
│  ├── CSM infrastructure exists, TERRAIN_USE_SHADOWS = false ⚠️     │
│  ├── SSR/SSGI not wired (P5 harness only) — SKIP for terrain       │
│  └── Specular aliasing fixed via roughness floor 0.65 (brute-force)│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Cascaded Shadow Maps (2-3 days)

**Goal:** Enable proper terrain shadows using existing CSM infrastructure.

**Impact:** ⭐⭐⭐⭐⭐ — Shadows define terrain form; currently missing entirely.

### P1.1 — Enable shadow shader path

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

```wgsl
// Line ~50: Change from false to true
const TERRAIN_USE_SHADOWS: bool = true;
```

### P1.2 — Wire CSM resources in TerrainRenderer

**File:** [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0)

Current state uses [NoopShadow](cci:2://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:68:0-77:1) placeholder. Replace with real CSM binding:

```rust
// In render_terrain_pbr_pom():
// 1. Create CSM renderer if shadows enabled
let csm = CsmRenderer::new(&device, &queue, CsmConfig {
    cascade_count: params.shadow_cascades().unwrap_or(4),
    resolution: params.shadow_map_res().unwrap_or(2048),
    max_distance: params.shadow_max_distance().unwrap_or(2000.0),
    ..Default::default()
});

// 2. Render shadow passes (one per cascade)
for cascade in 0..csm.cascade_count() {
    csm.render_cascade(&mut encoder, cascade, &terrain_mesh, &light_dir);
}

// 3. Bind shadow textures to terrain shader (group 5)
let shadow_bind_group = device.create_bind_group(&BindGroupDescriptor {
    layout: &shadow_bind_group_layout,
    entries: &[
        BindGroupEntry { binding: 0, resource: csm.shadow_maps_view() },
        BindGroupEntry { binding: 1, resource: csm.shadow_sampler() },
        BindGroupEntry { binding: 2, resource: csm.cascade_uniforms_buffer() },
    ],
});
```

### P1.3 — Terrain-scale cascade configuration

**File:** `src/csm.rs` (or terrain_renderer.rs)

```rust
// Cascade splits optimized for terrain viewing distances
// Assuming cam_radius ~1400m, max view ~5000m
const TERRAIN_CASCADE_SPLITS: [f32; 4] = [
    50.0,    // Cascade 0: 0-50m (close detail, sharp shadows)
    200.0,   // Cascade 1: 50-200m (mid-ground)
    800.0,   // Cascade 2: 200-800m (main terrain)
    3000.0,  // Cascade 3: 800-3000m (distant mountains)
];
```

### P1.4 — PCSS integration

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

Enable contact-hardening shadows:

```wgsl
fn calculate_shadow_terrain(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_depth: f32
) -> f32 {
    let cascade = select_cascade(view_depth);
    let shadow_coord = world_to_shadow_space(world_pos, cascade);
    
    // PCSS: Blocker search → penumbra estimation → variable PCF
    let blocker = find_average_blocker(shadow_coord, cascade);
    if (blocker.count == 0u) { return 1.0; }
    
    let penumbra = estimate_penumbra(shadow_coord.z, blocker.depth, u_shadow.light_size);
    return pcf_filter_variable(shadow_coord, cascade, penumbra);
}
```

### P1.5 — CLI parameters

**File:** [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0)

```python
parser.add_argument("--shadows-enabled", action="store_true", default=True)
parser.add_argument("--shadow-cascades", type=int, default=4)
parser.add_argument("--shadow-max-distance", type=float, default=3000.0)
parser.add_argument("--shadow-softness", type=float, default=0.01,
                    help="PCSS light size for soft shadows")
```

### Deliverables
- [ ] Terrain casts and receives shadows
- [ ] Contact-hardening (sharp near, soft far)
- [ ] CLI control over shadow quality
- [ ] Debug mode to visualize cascade boundaries

---

## Phase 2: Atmospheric Depth (1-2 days)

**Goal:** Distance-based haze for depth perception.

**Impact:** ⭐⭐⭐⭐⭐ — Transforms flat render into believable landscape.

### P2.1 — Height-fog with aerial perspective

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

Add after PBR composition, before tonemapping:

```wgsl
fn apply_atmospheric_fog(
    color: vec3<f32>,
    world_pos: vec3<f32>,
    cam_pos: vec3<f32>,
    sun_dir: vec3<f32>,
    fog_params: vec4<f32>  // density, height_falloff, inscatter_strength, _pad
) -> vec3<f32> {
    let view_vec = world_pos - cam_pos;
    let dist = length(view_vec);
    let view_dir = view_vec / dist;
    
    // Distance-based extinction
    let fog_density = fog_params.x;
    let extinction = 1.0 - exp(-dist * fog_density);
    
    // Height-based density modulation (valleys hazier)
    let height_falloff = fog_params.y;
    let avg_height = (world_pos.y + cam_pos.y) * 0.5;
    let height_factor = exp(-max(avg_height, 0.0) * height_falloff);
    let effective_extinction = extinction * height_factor;
    
    // Inscattering: sun-aligned views get warm tint
    let sun_alignment = max(dot(view_dir, sun_dir), 0.0);
    let inscatter = fog_params.z * pow(sun_alignment, 8.0);
    
    // Fog color: neutral blue-gray + sun tint
    let fog_color_base = vec3<f32>(0.65, 0.70, 0.80);
    let fog_color_sun = vec3<f32>(1.0, 0.85, 0.6);
    let fog_color = mix(fog_color_base, fog_color_sun, inscatter);
    
    return mix(color, fog_color, effective_extinction);
}
```

### P2.2 — Uniforms and CLI

**File:** [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0)

```rust
struct AtmosphereUniforms {
    fog_params: [f32; 4],      // density, height_falloff, inscatter, _pad
    fog_color: [f32; 4],       // rgb, _pad
    sun_direction: [f32; 4],   // xyz, _pad
}
```

**File:** [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0)

```python
parser.add_argument("--fog-density", type=float, default=0.0002)
parser.add_argument("--fog-height-falloff", type=float, default=0.0008)
parser.add_argument("--fog-inscatter", type=float, default=0.3)
```

### Deliverables
- [ ] Distant terrain fades to atmospheric haze
- [ ] Valleys hazier than peaks
- [ ] Sun-aligned views have warm glow
- [ ] CLI control over fog intensity

---

## Phase 3: Normal Anti-Aliasing Fix (2-3 days)

**Goal:** Restore specular detail by fixing flakes properly (not brute-force roughness).

**Impact:** ⭐⭐⭐⭐ — Re-enables specular highlights on wet rock, snow glints.

### P3.1 — Precompute normal variance texture

**File:** [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0)

At heightmap upload, generate variance mipchain:

```rust
fn generate_normal_variance_mips(heightmap: &[f32], width: u32, height: u32) -> Texture {
    let mut variance_data = Vec::new();
    
    for mip in 0..MAX_MIPS {
        let mip_w = width >> mip;
        let mip_h = height >> mip;
        
        for y in 0..mip_h {
            for x in 0..mip_w {
                // Compute normal at this mip
                let n = compute_sobel_normal(heightmap, x, y, mip);
                
                // Compute variance from 2x2 block of finer mip
                let variance = if mip == 0 {
                    0.0
                } else {
                    compute_normal_variance_from_children(x, y, mip)
                };
                
                variance_data.push(encode_normal_variance(n, variance));
            }
        }
    }
    
    create_texture_with_mips(variance_data, width, height)
}
```

### P3.2 — Shader-side variance-aware roughness

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

```wgsl
fn sample_normal_with_variance(uv: vec2<f32>, lod: f32) -> NormalAndVariance {
    let sample = textureSampleLevel(normal_variance_tex, sampler, uv, lod);
    let normal = decode_normal(sample.xyz);
    let variance = sample.w;
    return NormalAndVariance(normal, variance);
}

// Replace brute-force floor with variance-adjusted roughness
fn compute_specular_roughness(base_roughness: f32, normal_variance: f32) -> f32 {
    // Toksvig with precomputed variance
    let sigma2 = normal_variance;
    let r2 = base_roughness * base_roughness;
    return sqrt(r2 + sigma2 * (1.0 - r2));
}

// In main shader:
let nv = sample_normal_with_variance(uv, lod);
let specular_roughness = compute_specular_roughness(material_roughness, nv.variance);
// Use specular_roughness only for specular term, base_roughness for diffuse
```

### P3.3 — Restore lower roughness floor

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

```wgsl
// Change from 0.65 back to reasonable floor
let roughness_floor = select(0.25, 0.02, is_water);  // Was 0.65
```

### Deliverables
- [ ] Specular highlights visible on rock/snow again
- [ ] No flakes (variance-based roughness boost)
- [ ] Roughness floor reduced from 0.65 to 0.25

---

## Phase 4: Water Planar Reflections (2-3 days)

**Goal:** Water reflects surrounding terrain, not just sky.

**Impact:** ⭐⭐⭐⭐ — Water currently looks like flat gray; this makes it read as water.

### P4.1 — Reflection render pass

**File:** [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0)

```rust
fn render_water_reflection(
    &self,
    encoder: &mut CommandEncoder,
    water_plane_height: f32,
) -> TextureView {
    // 1. Create mirrored camera
    let reflected_view = self.camera.reflect_across_plane(water_plane_height);
    
    // 2. Render terrain to reflection texture (half res is fine)
    let reflection_texture = self.create_reflection_texture();
    
    // 3. Set clip plane to avoid rendering below water
    let clip_plane = vec4(0.0, 1.0, 0.0, -water_plane_height);
    
    // 4. Render pass with reflected camera
    self.render_terrain_pass(encoder, &reflected_view, &reflection_texture, Some(clip_plane));
    
    reflection_texture.create_view(&Default::default())
}
```

### P4.2 — Water shader reflection sampling

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

```wgsl
@group(6) @binding(0) var reflection_tex: texture_2d<f32>;
@group(6) @binding(1) var reflection_sampler: sampler;

fn sample_water_reflection(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>
) -> vec3<f32> {
    // Project world position to screen space
    let clip_pos = u_terrain.view_proj * vec4(world_pos, 1.0);
    var screen_uv = clip_pos.xy / clip_pos.w * 0.5 + 0.5;
    screen_uv.y = 1.0 - screen_uv.y;  // Flip Y for reflection
    
    // Perturb by wave normal for distortion
    let distortion = normal.xz * 0.02;
    screen_uv += distortion;
    
    return textureSample(reflection_tex, reflection_sampler, screen_uv).rgb;
}

// In water branch:
if (is_water) {
    let reflection = sample_water_reflection(world_pos, wave_normal, view_dir);
    let fresnel = fresnel_schlick(n_dot_v, f0);
    water_color = mix(underwater_color, reflection, fresnel);
}
```

### P4.3 — Animated Gerstner waves

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

```wgsl
struct GerstnerWave {
    direction: vec2<f32>,
    steepness: f32,
    wavelength: f32,
}

fn gerstner_wave(pos: vec2<f32>, wave: GerstnerWave, time: f32) -> vec3<f32> {
    let k = 2.0 * PI / wave.wavelength;
    let c = sqrt(9.8 / k);
    let d = normalize(wave.direction);
    let f = k * (dot(d, pos) - c * time);
    let a = wave.steepness / k;
    
    return vec3(
        d.x * a * cos(f),
        a * sin(f),
        d.y * a * cos(f)
    );
}

fn compute_water_displacement(world_pos: vec3<f32>, time: f32) -> WaterSurface {
    let waves = array<GerstnerWave, 4>(
        GerstnerWave(vec2(1.0, 0.0), 0.25, 60.0),
        GerstnerWave(vec2(0.7, 0.7), 0.15, 30.0),
        GerstnerWave(vec2(0.0, 1.0), 0.10, 15.0),
        GerstnerWave(vec2(-0.5, 0.5), 0.08, 8.0),
    );
    
    var offset = vec3(0.0);
    for (var i = 0; i < 4; i++) {
        offset += gerstner_wave(world_pos.xz, waves[i], time);
    }
    
    // Compute normal from displacement derivatives
    let normal = compute_gerstner_normal(world_pos.xz, waves, time);
    
    return WaterSurface(offset, normal);
}
```

### Deliverables
- [ ] Water reflects surrounding mountains
- [ ] Fresnel-based reflection intensity (more reflection at grazing angles)
- [ ] Animated waves with Gerstner displacement
- [ ] Wave intensity reduces near shore

---

## Phase 5: Ambient Occlusion Enhancement (1 day)

**Goal:** Verify SSAO is contributing; add heightmap-based AO fallback.

**Impact:** ⭐⭐⭐ — Crevices and valleys should be darker.

### P5.1 — Debug SSAO integration

**File:** [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0)

Add debug mode to output raw SSAO:
```rust
// Debug mode 28: Output raw SSAO
if debug_mode == 28 {
    return vec4(occlusion, occlusion, occlusion, 1.0);
}
```

### P5.2 — Heightmap horizon-based AO

**File:** [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0)

Precompute coarse AO from heightmap:

```rust
fn compute_horizon_ao(heightmap: &[f32], w: u32, h: u32, x: u32, y: u32) -> f32 {
    const DIRECTIONS: usize = 8;
    const MAX_DIST: f32 = 50.0;  // meters
    
    let center_h = heightmap[(y * w + x) as usize];
    let mut occlusion = 0.0;
    
    for dir in 0..DIRECTIONS {
        let angle = (dir as f32) * TAU / (DIRECTIONS as f32);
        let dx = angle.cos();
        let dy = angle.sin();
        
        let mut max_horizon = 0.0f32;
        for step in 1..20 {
            let sx = x as f32 + dx * step as f32 * 2.0;
            let sy = y as f32 + dy * step as f32 * 2.0;
            if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h as f32 { break; }
            
            let sample_h = heightmap[(sy as u32 * w + sx as u32) as usize];
            let dist = step as f32 * 2.0;
            let horizon = (sample_h - center_h) / dist;
            max_horizon = max_horizon.max(horizon);
        }
        
        // Convert horizon angle to occlusion
        occlusion += (1.0 - max_horizon.atan() / FRAC_PI_2).max(0.0);
    }
    
    occlusion / DIRECTIONS as f32
}
```

### Deliverables
- [ ] Verify SSAO buffer is being sampled
- [ ] Heightmap AO texture generated at upload
- [ ] Valleys visibly darker than ridges

---

## Phase 6: Micro-Detail (1-2 days)

**Goal:** Close-range surface detail without flakes.

**Impact:** ⭐⭐⭐ — Improves close-up views.

### P6.1 — Detail normal tiling

**File:** [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0)

```wgsl
fn blend_detail_normal(
    base_normal: vec3<f32>,
    world_pos: vec3<f32>,
    cam_distance: f32
) -> vec3<f32> {
    // Detail fades out with distance
    let detail_strength = saturate(1.0 - cam_distance / 300.0) * 0.4;
    if (detail_strength < 0.01) { return base_normal; }
    
    // Sample tiling detail normal (using triplanar to avoid stretching)
    let detail_scale = 2.0;  // 2 meter repeat
    let detail_n = sample_triplanar_normal(
        detail_normal_tex,
        world_pos * detail_scale,
        base_normal
    );
    
    // Reoriented normal blending
    return blend_rnm(base_normal, detail_n * detail_strength);
}
```

### P6.2 — Procedural albedo variation

```wgsl
fn add_albedo_noise(albedo: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let noise_scale = 0.01;
    let noise = simplex_noise_3d(world_pos * noise_scale);
    let variation = noise * 0.1;  // ±10% brightness
    return albedo * (1.0 + variation);
}
```

### Deliverables
- [ ] Tiling detail normal for close-up
- [ ] Procedural brightness variation
- [ ] Detail fades with distance (no LOD popping)

---

## Implementation Schedule

| Phase | Days | Priority | Dependencies |
|-------|------|----------|--------------|
| **P1: Shadows** | 2-3 | 🔴 Critical | None |
| **P2: Atmospheric fog** | 1-2 | 🔴 Critical | None |
| **P3: Normal AA fix** | 2-3 | 🟡 High | None |
| **P4: Water reflections** | 2-3 | 🟡 High | P1 (shadows) |
| **P5: AO enhancement** | 1 | 🟢 Medium | None |
| **P6: Micro-detail** | 1-2 | 🟢 Medium | P3 (normal AA) |

**Total: 10-14 days**

---

## Verification Protocol

After each phase:

```bash
# Render test image
python examples/terrain_demo.py \
  --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/hdri/snow_field_4k.hdr \
  --size 1920 1080 --render-scale 1 --msaa 8 \
  --output examples/output/phase_N.png --overwrite

# Compare with baseline
python scripts/compare_images.py \
  examples/output/terrain_high_roughness.png \
  examples/output/phase_N.png \
  --output examples/output/phase_N_diff.png --ssim
```

---

## What We're NOT Doing

| Feature | Reason |
|---------|--------|
| SSR (full screen-space) | Planar reflections better for water; terrain too rough for SSR |
| SSGI | IBL sufficient for ambient; cost/benefit poor for terrain |
| Volumetric fog | Nice-to-have but atmospheric fog covers 90% of the value |
| Vegetation | Out of scope; requires instancing system |

---