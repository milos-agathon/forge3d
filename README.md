<div align="center">
  <a href="./">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo-2000-dark.png">
      <img src="assets/logo-2000.png"
           alt="forge3d logo"
           width="224"
           height="224"
           decoding="async"
           loading="eager">
    </picture>
  </a>
</div>

# forge3d

Headless GPU rendering + PNG↔NumPy utilities (Rust + PyO3 + wgpu).

Current release: 1.2.0 — Golden-image regression testing and GPU memory-budget enforcement, plus a refreshed IBL pipeline.

## Installation

```bash
# from source
pip install -U maturin
maturin develop --release
# or via wheel (if provided)
# pip install forge3d
```

## Quick Start (< 10 minutes)

New to forge3d? Launch the interactive viewer and capture a high-resolution snapshot of Mount Rainier:

1. **Install prerequisites**: Ensure you have Python 3.8+ and Rust installed
2. **Install maturin**: `pip install -U maturin`
3. **Build forge3d**: `maturin develop --release`
4. **Build the interactive viewer**:

```bash
cargo build --release --bin interactive_viewer
```

5. **Launch the viewer with a high-quality preset**:

```bash
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --preset high_quality --width 800 --height 800
```

6. **Capture a high-resolution snapshot** (type this in the viewer's terminal):

```bash
snap highres_rainier.png 4000x4000
```

This workflow demonstrates forge3d's **supersampled snapshot** capability: the viewer runs at interactive resolution (800×800) for real-time exploration, then renders a single frame at 4000×4000 (16 megapixels) for publication-quality output. Because the GPU renders only one high-res frame on demand, you get print-ready imagery without the memory cost of a persistent 4K framebuffer.

![Mount Rainier High-Resolution Snapshot](assets/highres.png)

## Platform requirements

Runs anywhere wgpu supports: Vulkan / Metal / DX12 / GL (and Browser WebGPU for diagnostics). A discrete or integrated GPU is recommended. Examples/tests that need a GPU will skip if no compatible adapter is found.
