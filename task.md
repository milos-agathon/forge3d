You must implement this solution and save as examples/terrain_demo.py

---

# Goal

Render a **3D terrain** from a **local DEM (GeoTIFF)** using a **custom colormap** plus **PBR + IBL + Triplanar + POM**, then export a high-quality image (or view it interactively).

---

# 1) Install & import (Python front-end)

```bash
pip install forge3d  # or: uv add forge3d
```

```python
import forge3d as f3d
```

**Rust backend:** compiled extension (PyO3) loads the wgpu/renderer core.

---

# 2) Open a session (headless or interactive)

```python
# window=False = offscreen (perfect for batch renders)
sess = f3d.Session(window=False)
```

**Rust backend:** initializes adapter/device/queue, selects the best backend
(Metal on macOS, D3D12/Vulkan on Win, Vulkan on Linux), and creates a resource
arena sized to your GPU budget (keep host-visible ≤ ~512 MiB).

---

# 3) Load the DEM (local file)

```python
dem = f3d.io.load_dem(
    "assets/Gore_Range_Albers_1m.tif",
    fill_nodata=True,         # hole-filling
    to_local_metric=True      # reproject to a local meter CRS if needed
)
```

**Rust backend:**

* Reads GeoTIFF via GDAL/Rust readers → float32/uint16 height grid
* Applies NoData fill, optional reprojection, builds a **GPU height texture** (R16/R32F)
* Generates a **normal map** from height gradients for lighting

---

# 4) Define a custom colormap (continuous 1D LUT; actual values will depend on the quantiles from the tiff)

```python
cmap = f3d.Colormap1D.from_stops(
    stops=[
        (200,  "#e7d8a2"), 
        (800,  "#c5a06e"),
        (1500, "#995f57"),
        (2200, "#4a3c37"),
    ],
    domain=(200, 2200)  # min/max elevation in your AOI
)
```

**Rust backend:** builds a normalized **1D texture LUT** (256–1024 texels) for instant, banding-free palette lookup in the shader.

---

# 5) Pick terrain materials & environment lighting

```python
materials = f3d.MaterialSet.terrain_default(
    triplanar_scale=6.0, normal_strength=1.0, blend_sharpness=4.0
)

ibl = f3d.IBL.from_hdr(
    "assets/hdr/gravel_plaza_4k.hdr",
    intensity=1.0,
    rotate_deg=0.0
)
```

**Rust backend:**

* Creates material texture arrays & samplers
* Prefilters the HDR to **diffuse/specular mip chains** + **BRDF LUT** (compute pass) for PBR IBL

---

# 6) Configure rendering parameters (all the knobs you asked for)

```python
params = f3d.TerrainRenderParams(
    # Dimensions
    size_px=(2560, 1440),
    render_scale=1.0,
    msaa_samples=4,

    # Terrain geometry
    z_scale=1.5,

    # Camera (orbit)
    cam_target=[0.0, 0.0, 0.0],
    cam_radius=1200.0,
    cam_phi_deg=135.0,
    cam_theta_deg=45.0,
    cam_gamma_deg=0.0,
    fov_y_deg=55.0,
    clip=(0.1, 6000.0),

    # Lighting
    light=f3d.LightSettings(
        light_type="Directional",
        azimuth_deg=135.0,
        elevation_deg=35.0,
        intensity=3.0,
        color=[1.0, 1.0, 1.0]
    ),
    ibl=f3d.IblSettings(
        enabled=True, intensity=1.0, rotation_deg=0.0
    ),

    # Shadows
    shadows=f3d.ShadowSettings(
        enabled=True, technique="PCSS",
        map_resolution=4096, cascade_count=3,
        max_distance=4000.0, split_lambda=0.6,
        depth_bias=0.002, normal_bias=0.5,
        softness=1.5, intensity=0.8,
        min_variance=1e-4, bleed_reduction=0.5,
        clamp_min=0.0, clamp_max=1.0
    ),

    # Triplanar + POM
    triplanar=f3d.TriplanarSettings(
        scale=6.0, blend_sharpness=4.0, normal_strength=1.0
    ),
    pom=f3d.PomSettings(
        enabled=True, model="Occlusion",
        height_scale=0.04,
        min_steps=12, max_steps=40, refine_steps=4,
        shadowing=True, horizon_search=True
    ),

    # LOD & Sampling
    lod=f3d.LodSettings(max_mip_level=0, lod_bias=0.0, normal_mip_bias=-0.5),
    sampling=f3d.SamplingSettings(
        min_filter="Linear", mag_filter="Linear", mipmap_filter="Linear",
        anisotropy=8,
        address_mode_u="Repeat", address_mode_v="Repeat", address_mode_w="Repeat"
    ),

    # Clamp ranges (physically plausible/material safety)
    clamp=f3d.ClampSettings(
        albedo=(0.0,1.0), roughness=(0.04,1.0), metallic=(0.0,1.0),
        ao=(0.0,1.0), height=(0.0,1.0)
    ),

    # Use your custom colormap as an overlay/albedo driver
    overlays=[
        f3d.OverlayLayer.from_colormap1d(
            cmap, opacity=1.0, z_offset=0.0,
            blend_mode="Alpha", value_range=(200,4200)
        )
    ],

    # Tonemapping / exposure
    exposure=1.0, gamma=2.2,

    # Albedo mixing mode (how colormap influences PBR base color)
    albedo_mode="mix",           # "colormap" | "mix" | "material"
    colormap_strength=0.5
)
```

**Rust backend:** prepares pipelines (shadow, g-buffer or forward, lighting, compute for GTAO/denoise if enabled), allocates textures for HDR, normals, shadows, and binds all parameter buffers.

---

# 7) Render a still and save

```python
renderer = f3d.TerrainRenderer(sess.device)

frame = renderer.render_terrain_pbr_pom(
    target=None,                      # offscreen target handled internally
    heightmap=dem.texture_view,
    material_set=materials,
    env_maps=ibl,
    params=params
)

frame.save("out/swiss_pbr_colormap.png")  # supports PNG/EXR/TIFF
```

**Rust backend:**

* Updates cascaded shadow maps → main PBR+IBL pass with triplanar+POM → optional SSAO/GTAO/composite → ACES tonemap → resolves MSAA → returns an image.

---

# 8) (Optional) interactive viewer

```python
with f3d.Viewer(sess, renderer, dem.texture_view, materials, ibl, params) as view:
    view.run()  # orbit/pan/zoom; UI sliders for z_scale, light, POM steps, etc.
```

**Rust backend:** swaps to a surface-backed target; re-records command buffers as camera/light knobs change.

---

# 9) CLI equivalent (nice for reproducible runs)

```bash
forge3d render terrain \
  --dem data/swiss_dem.tif \
  --colormap colormap.json \
  --out out/swiss.png \
  --size 2560x1440 --msaa 4 --z-scale 1.5 \
  --camera "phi=135,theta=45,gamma=0,radius=1200,target=0,0,0,fov=55" \
  --light "type=dir,az=135,el=35,intensity=3.0" \
  --ibl "hdr=assets/hdr/gravel_plaza_4k.hdr,intensity=1.0,rot=0" \
  --shadows "tech=pcss,res=4096,cascades=3,dist=4000,soft=1.5,intensity=0.8" \
  --pom "on,scale=0.04,steps=12:40,refine=4" \
  --albedo-mode mix --colormap-strength 0.5
```

---

## What the **Rust backend** does at each Python call

* **`io.load_dem(...)`** → CPU decode, reprojection, NoData fill → GPU **height/normal textures**
* **`Colormap1D.from_stops(...)`** → builds a **1D LUT texture** (linear space)
* **`IBL.from_hdr(...)`** → GPU prefilter to **diffuse/specular mips + BRDF LUT**
* **`render_terrain_pbr_pom(...)`** → records wgpu commands: shadow pass → PBR+IBL with **triplanar sampling** and **POM ray-march** → optional AO/denoise → tone-map → resolve
* **Memory/tiling:** if DEM > ~8k², backend **tiles + mip-streams**; height uses R16 whenever range allows

---

## Presets (speed vs. quality)

* **Preview (very fast):** `msaa=1`, `PCF` shadows, `pom.enabled=False`, `anisotropy=2`, `render_scale=0.75`
* **Final still:** `msaa=4–8`, `PCSS`, `pom.max_steps=40`, `anisotropy=16`, `render_scale=1.0`, save **EXR** for grading
* **Huge DEMs:** enable tiling, clamp `shadows.max_distance`, keep `z_scale` physically plausible

---

## Common pitfalls & fixes

* **Colormap banding** → LUT ≥ 512, render in **RGBA16F**, enable dithering
* **Shadow acne** → tweak `depth_bias`, `normal_bias`, or switch to `EVSM` with `min_variance`
* **POM too expensive** → lower `max_steps`, keep `refine_steps`, apply only on steep slopes
* **Specular shimmer** → `anisotropy ≥ 8`, `lod_bias≈0`, use **TAA** (viewer)

---

## Minimal working example (copy-paste)

```python
import forge3d as f3d

sess = f3d.Session(window=False)
dem  = f3d.io.load_dem("data/swiss_dem.tif", fill_nodata=True, to_local_metric=True)

cmap = f3d.Colormap1D.from_stops(
    stops=[(200,"#313695"),(800,"#4575b4"),(1500,"#74add1"),
           (2200,"#fdae61"),(3000,"#f46d43"),(4200,"#a50026")],
    domain=(200,4200)
)

materials = f3d.MaterialSet.terrain_default(6.0, 1.0, 4.0)
ibl = f3d.IBL.from_hdr("assets/hdr/gravel_plaza_4k.hdr", intensity=1.0, rotate_deg=0.0)

params = f3d.TerrainRenderParams(
    size_px=(1920,1080), render_scale=1.0, msaa_samples=4, z_scale=1.5,
    cam_target=[0,0,0], cam_radius=1100, cam_phi_deg=135, cam_theta_deg=45, cam_gamma_deg=0,
    fov_y_deg=55, clip=(0.1, 6000),
    light=f3d.LightSettings("Directional", 135, 35, 3.0, [1,1,1]),
    ibl=f3d.IblSettings(True, 1.0, 0.0),
    shadows=f3d.ShadowSettings(True, "PCSS", 4096, 3, 4000, 0.6, 0.002, 0.5, 1.5, 0.8, 1e-4, 0.5, 0.0, 1.0),
    triplanar=f3d.TriplanarSettings(6.0, 4.0, 1.0),
    pom=f3d.PomSettings(True, "Occlusion", 0.04, 12, 40, 4, True, True),
    lod=f3d.LodSettings(0, 0.0, -0.5),
    sampling=f3d.SamplingSettings("Linear","Linear","Linear",8,"Repeat","Repeat","Repeat"),
    clamp=f3d.ClampSettings((0,1),(0.04,1),(0,1),(0,1),(0,1)),
    overlays=[f3d.OverlayLayer.from_colormap1d(cmap, 1.0, 0.0, "Alpha", (200,4200))],
    exposure=1.0, gamma=2.2, albedo_mode="mix", colormap_strength=0.5
)

renderer = f3d.TerrainRenderer(sess.device)
frame = renderer.render_terrain_pbr_pom(None, dem.texture_view, materials, ibl, params)
frame.save("out/swiss_pbr_colormap.png")
```

That’s the end-to-end **Python UX** with clear **Rust backend** responsibilities—ready to drop into your forge3d docs/examples.
