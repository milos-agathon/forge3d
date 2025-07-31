# python/ — Memory for Python surface & packaging

This directory contains the thin Python API layer for `vulkan_forge`. The Rust core is exposed via **PyO3** as `vulkan_forge._vulkan_forge` and re‑exported in `vulkan_forge/__init__.py`.

## API principles
- Keep the Python API **minimal and explicit**. No hidden threads; no implicit globals.
- Validate **dtypes/shapes/contiguity** at the boundary and raise clear `PyRuntimeError`.
- Expose **metrics** and **device info** for debugging.
- Prefer **zero‑copy** or single‑copy paths from NumPy to Rust.

## Public methods (MVP)
```python
Renderer(width: int, height: int, *, prefer_software: bool = False)
Renderer.info() -> str
Renderer.report_environment() -> dict  # JSONable
Renderer.render_triangle_rgba() -> np.ndarray[H, W, 4] (uint8)
Renderer.render_triangle_png(path: str) -> None
Renderer.add_terrain(heightmap: np.ndarray, spacing: tuple[float,float], exaggeration: float = 1.0, *, colormap: str = "viridis") -> None
Renderer.set_camera_orbit(center=(0,0,0), distance: float, elevation_deg: float, azimuth_deg: float) -> None
Renderer.set_sun(elevation_deg: float, azimuth_deg: float, exposure: float = 1.0) -> None
Renderer.render_rgba() -> np.ndarray[H, W, 4] (uint8)
Renderer.render_png(path: str) -> None
```
Vectors/graph arrive in Week‑3/4 but the interface shape follows the same principles.

## Boundary checks
- Accept `float32` (preferred) and `float64` for DEM; cast to `f32` once in Rust.
- Enforce C‑contiguous arrays with shape `(H,W)` (DEM) and `(N,2)` (points) or packed formats for polygons/lines.
- Raise `PyRuntimeError` with **actionable messages** (what went wrong + how to fix).

## NumPy & PyO3 interop
- Use `numpy` crate: fetch `PyArray*` as **readonly views**, validate without copying, then copy/convert once if needed.
- Release the **GIL** during heavy GPU work: wrap render calls in `Python::allow_threads` or `py.allow_threads` blocks.
- Map Rust errors via `thiserror` → `PyErr` with a clear prefix (`[Device]`, `[Upload]`, `[Render]`).

## Examples & notebooks
- Keep examples **self‑contained** and deterministic (generate synthetic data).
- Use **fixed seeds** in any random generation.
- Save **metrics JSON** and **env.json** next to output PNGs to aid reproducibility.
- Provide CLI entry points for the three MVP examples (terrain+overlays, basemap, graph).

## Packaging (maturin)
- Use **abi3** (py>=3.10) to build one wheel per OS/arch.
- Linux: `--compatibility manylinux2014`.
- Keep `strip = true` in `pyproject.toml`; enable LTO thin in Cargo profiles.
- Build/test **once per OS/arch** in CI; no need for per‑Python builds.

## Style & lint
- Type‑hint public APIs. Keep docstrings short and imperative.
- Run `black` and `ruff` locally (`make fmt` / `make lint`).

## Testing
- `pytest -q` covers: input validation, small render correctness, SSIM/PSNR against goldens.
- Golden images live in `tests/golden/`; regenerate with a script on an authoritative platform.
- Backends: allow forcing `WGPU_BACKEND=metal|vulkan|dx12`; skip gracefully if unavailable.

## Troubleshooting
- Import failures after build: try `pip uninstall -y vulkan-forge && maturin develop --release`.
- Missing wheels on your OS: build from source with the same command.
