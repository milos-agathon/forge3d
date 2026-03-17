# Staging Buffer Rings

Staging rings are an internal Rust implementation detail used for GPU upload and
readback workflows. There is no public Python API for manually creating or
driving staging rings.

## What Python users have today

- `Scene.render_rgba()` and viewer snapshots for readback
- memory telemetry via `forge3d.memory_metrics()`
- device capability inspection via `forge3d.device_probe()`

If you need to tune staging behavior, do it in the Rust renderer layer.
