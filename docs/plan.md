## 1. Goals and constraints

- **Goal:** [interactive_viewer_terrain_demo.py] should show the actual Gore Range terrain (or whatever DEM/HDR you pass), not a uniform purple frame.
- **Constraints:**
  - Reuse existing terrain code (`terrain_pbr_pom`, [TerrainRenderer], `RendererConfig`) rather than duplicating shaders/pipelines.
  - Keep viewer architecture clean and testable per `AGENTS.md`.
  - Avoid breaking the existing headless `terrain_demo` and tests.

---

## 2. High-level design

Conceptually:

- The **terrain demo** already knows how to:
  - Build a `RendererConfig` from CLI args.
  - Use terrain shaders + IBL to render a heightmap scene to an image.

- The **interactive viewer** knows how to:
  - Own a `wgpu::Device/Queue`.
  - Run P5/P6 GI/screen-space pipelines.
  - Render scene geometry (meshes) into its targets.
  - Take snapshots at arbitrary resolution.

What’s missing is a **TerrainScene** that:

- Can be created inside the viewer from a `RendererConfig` (or equivalent), and
- Knows how to render the terrain into the viewer’s render targets each frame.

We’ll add that and a small Python API to call it.

---

## 3. Milestone plan

### M1 — Confirm and reuse terrain configuration

**Files:**

- [python/forge3d/terrain_demo.py]
- `python/forge3d/terrain_pbr_pom.py`
- [src/render/params.rs] (RendererConfig)
- [src/terrain_renderer.rs]

**Tasks:**

- Verify that [_build_renderer_config(args)] ultimately builds a `RendererConfig` (or something equivalent) on the Rust side.
- Identify exactly which pieces the viewer needs:
  - DEM path, HDR path.
  - Z-scale, camera radius/angles.
  - IBL options.
  - Colormap and material controls.

**Outcome:** A clear mapping from Python CLI args → a single Rust config type the viewer can understand (likely `RendererConfig`).

---

### M2 — Introduce a reusable `TerrainScene` module (Rust only)

**Files:**

- New: `src/terrain/scene.rs` (or similar)
- Existing: [src/terrain_renderer.rs]

**Tasks:**

- Factor the **GPU-facing pieces** of [TerrainRenderer] into a non-PyO3 helper:

  ```rust
  pub struct TerrainScene {
      // device/queue references (Arc<..> or borrowed)
      // terrain pipelines, bind groups, textures
      // colormap LUT, IBL resources, etc.
  }

  impl TerrainScene {
      pub fn new(device: &Device, queue: &Queue, cfg: &RendererConfig) -> Result<Self>;
      pub fn render(
          &mut self,
          encoder: &mut wgpu::CommandEncoder,
          color_view: &wgpu::TextureView,
          depth_view: &wgpu::TextureView,
      ) -> Result<()>;
  }
  ```

- Refactor [TerrainRenderer] so that it **owns a `TerrainScene` internally** instead of duplicating pipeline code:

  ```rust
  pub struct TerrainRenderer {
      device: Arc<Device>,
      queue: Arc<Queue>,
      scene: Mutex<TerrainScene>,
      // existing lights/config fields
  }
  ```

**Outcome:** A single terrain rendering implementation, reusable both by headless terrain and the interactive viewer.

---

### M3 — Attach `TerrainScene` to the viewer

**Files:**

- [src/viewer/mod.rs]

**Tasks:**

- Add a field on [Viewer]:

  ```rust
  pub struct Viewer {
      // ...
      terrain_scene: Option<TerrainScene>,
      // ...
  }
  ```

- Add a method:

  ```rust
  impl Viewer {
      pub fn load_terrain_from_config(
          &mut self,
          cfg: &RendererConfig,
      ) -> anyhow::Result<()> {
          let device = &self.device;
          let queue = &self.queue;
          let scene = TerrainScene::new(device, queue, cfg)?;
          self.terrain_scene = Some(scene);
          Ok(())
      }
  }
  ```

- In the main render loop (where the fallback pipeline currently draws the purple full-screen triangle), change logic to:

  ```rust
  if let Some(scene) = &mut self.terrain_scene {
      scene.render(&mut encoder, &color_view, &depth_view)?;
  } else {
      // existing fallback_pipeline solid color path
  }
  ```

**Outcome:** Viewer can render a terrain scene instead of falling back to the purple debug pipeline when a `TerrainScene` has been loaded.

---

### M4 — Create a dedicated Rust entrypoint: `open_terrain_viewer`

**Files:**

- [src/lib.rs] (PyO3 interface)
- Possibly [src/cli/interactive_viewer.rs] (if you want a Rust-only CLI later)

**Tasks:**

- Add a new native function:

  ```rust
  #[pyfunction]
  fn open_terrain_viewer(
      cfg: RendererConfig,
      width: u32,
      height: u32,
      title: String,
      // vsync, fov, znear/zfar, snapshot_* and initial_commands exactly like open_viewer
      // ...
  ) -> PyResult<()> {
      // 1. Build a ViewerConfig as in open_viewer.
      // 2. Construct Viewer.
      // 3. Call viewer.load_terrain_from_config(&cfg).
      // 4. Run viewer loop / snapshot as usual.
  }
  ```

- Make sure the signature is symmetrical with [open_viewer] so Python can wrap it easily.

**Outcome:** Native API that ties together viewer setup + terrain loading in one call.

---

### M5 — Python API and example changes

**Files:**

- [python/forge3d/__init__.py]
- [examples/interactive_viewer_terrain_demo.py]

**Tasks:**

1. **Python wrapper:**

   In [forge3d/__init__.py], add:

   ```python
   def open_terrain_viewer(
       config,
       *,
       width: int = 1024,
       height: int = 768,
       title: str = "forge3d Terrain Interactive Viewer",
       vsync: bool = True,
       fov_deg: float = 45.0,
       znear: float = 0.1,
       zfar: float = 1000.0,
       snapshot_path: str | None = None,
       snapshot_width: int | None = None,
       snapshot_height: int | None = None,
       initial_commands: list[str] | None = None,
   ) -> None:
       # mirror validation from open_viewer for snapshot_* and env var
       # then call _native.open_terrain_viewer(...)
   ```

   Reuse the same snapshot validation logic we already wrote for [open_viewer].

2. **Example script:**

   In [examples/interactive_viewer_terrain_demo.py], change [main()] to:

   ```python
   cfg = terrain_demo._build_renderer_config(args)
   width, height = args.size

   f3d.open_terrain_viewer(
       cfg,
       width=int(width),
       height=int(height),
       title=str(args.title),
       snapshot_path=str(args.snapshot_path) if args.snapshot_path is not None else None,
       snapshot_width=args.snapshot_width,
       snapshot_height=args.snapshot_height,
       initial_commands=[
           ":gi gtao on",
           ":fog on",
       ],
   )
   ```

**Outcome:** From Python you now explicitly say “open a viewer with this terrain config”, and the Rust side has what it needs to render a real scene.

---

### M6 — Tests and verification

**Files:**

- `tests/test_terrain_renderer.py`
- New or existing I1 tests (e.g. `tests/test_I1_open_viewer_api.py`)
- Possibly a new `tests/test_interactive_terrain_viewer.py` (optional)

**Tasks:**

- Ensure existing terrain tests still pass (`python -m pytest tests/test_terrain_renderer.py -q`).
- Add a lightweight smoke test for the terrain viewer, e.g.:

  ```python
  def test_interactive_terrain_viewer_smoke(tmp_path):
      import forge3d as f3d
      from examples import terrain_demo

      args = argparse.Namespace(
          dem=terrain_demo.DEFAULT_DEM,
          hdr=terrain_demo.DEFAULT_HDR,
          size=(320, 180),
          exposure=1.0,
          # fill required fields minimally
      )
      cfg = terrain_demo._build_renderer_config(args)
      out = tmp_path / "iv_terrain.png"

      # Use a tiny snapshot to keep test fast
      f3d.open_terrain_viewer(
          cfg,
          width=320,
          height=180,
          snapshot_path=str(out),
          snapshot_width=320,
          snapshot_height=180,
      )

      assert out.exists()
      # Optionally, check that the image is not a flat color.
  ```

---

## 4. What this gives you

Once these steps are implemented:

- [interactive_viewer_terrain_demo.py] will:
  - Build the same terrain configuration as the headless demo,
  - Pass it into a viewer that actually instantiates terrain GPU resources,
  - Render a real terrain scene with SSAO/fog/etc,
  - Take a high-res snapshot that looks like your PBR+POM terrain, not a purple fallback.