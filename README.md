<p align="center">
  <a href="#">
    <!-- Replace with your logo path or external URL -->
    <img src="docs/assets/logo-placeholder.svg" alt="vulkan-forge logo" width="128" height="128">
  </a>
</p>

<h1 align="center">vulkan‑forge</h1>

<p align="center">
  Rust‑first, cross‑platform <strong>wgpu/WebGPU</strong> renderer exposed to Python for fast, headless 3D rendering.
  <br>
  <em>Built in Rust, shipped as Python wheels.</em>
</p>

<p align="center">
  <!-- Badges: replace USER/REPO with your org/repo slug -->
  <a href="https://pypi.org/project/vulkan-forge/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vulkan-forge"></a>
  <a href="https://pypi.org/project/vulkan-forge/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/vulkan-forge"></a>
  <a href="https://github.com/USER/REPO/actions"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/USER/REPO/wheels.yml?branch=main"></a>
  <a href="https://pepy.tech/project/vulkan-forge"><img alt="Downloads" src="https://img.shields.io/pepy/dt/vulkan-forge"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="#"><img alt="Platform" src="https://img.shields.io/badge/platform-win%20|%20macOS%20|%20linux-lightgrey"></a>
  <a href="https://codecov.io/gh/USER/REPO"><img alt="codecov" src="https://codecov.io/gh/USER/REPO/branch/main/graph/badge.svg"></a>
</p>

<p align="center">
  <a href="#-install">Install</a> •
  <a href="#-quick-start">Quick start</a> •
  <a href="#-features">Features</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-license">License</a>
</p>

---

> ⚠️ <strong>Status:</strong> Early prototype. APIs will change during the MVP phase.

## 🚀 Install

```bash
pip install vulkan-forge
# or from source
pip install maturin
git clone https://github.com/USER/REPO.git
cd REPO
maturin develop --release
```

## 🧪 Quick start

```python
from vulkan_forge import Renderer

r = Renderer(width=512, height=512)
print(r.info())
rgba = r.render_triangle_rgba()     # H x W x 4, uint8
r.render_triangle_png("triangle.png")
```

## ✨ Features (MVP)

- Headless, cross‑platform off‑screen rendering via `wgpu` (Vulkan/Metal/DX12).
- Rust core for performance and a thin Python API.
- Deterministic readback to NumPy RGBA images.
- CI wheels for Windows, macOS, Linux (x86_64/arm64).

## 🗺️ Roadmap (short)

- Terrain layer from NumPy DEMs (lighting + tonemap)
- Geo vectors (polygons/lines/points) with tessellation
- Simple camera & sun controls
- Image tests (SSIM) across backends

## 🤝 Contributing

- Issues and PRs welcome. See `CONTRIBUTING.md` (to be added).
- Code of Conduct: `CODE_OF_CONDUCT.md` (to be added).

## 📄 License

MIT © Your Name
