# GPU Profiling and Performance Analysis

forge3d does not currently expose a public `forge3d.gpu_metrics` Python module.

## What you can use from Python today

- `forge3d.device_probe()` for adapter and capability inspection
- `forge3d.enumerate_adapters()` for adapter listing
- `forge3d.memory_metrics()` for memory telemetry
- `Scene.get_stats()` for scene-side runtime stats
- `ViewerHandle.get_stats()` for interactive-viewer runtime stats

## Example

```python
import forge3d as f3d

print(f3d.device_probe())
print(f3d.enumerate_adapters())
print(f3d.memory_metrics())
```

## Deeper profiling

For frame captures and GPU timelines, use external tools such as RenderDoc,
Nsight Graphics, or Radeon GPU Profiler against the native viewer or your Rust
integration. The renderer contains internal timing infrastructure, but it is
not surfaced as a stable Python metrics API today.
