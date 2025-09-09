Examples

Overview
- This folder showcases advanced examples using the existing public APIs (no private hooks). Many examples write outputs to `out/` (gitignored) and/or print stats.

Run
- Ensure the extension is built:
  pip install -U maturin
  maturin develop --release

- Run any Python example, e.g.:
  python examples/advanced_terrain_shadows_pbr.py
  python examples/device_capability_probe.py

Outputs
- Generated images are written to `out/` or the current directory. The `out/` directory is ignored by Git per `.gitignore`.

Caveats
- GPU-dependent examples auto-skip functionality or fall back gracefully when no compatible adapter is present.

