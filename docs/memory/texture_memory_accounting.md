# Texture Memory Accounting

Texture memory accounting is currently surfaced to Python through aggregated
telemetry, not through a dedicated texture-accounting module.

## Public API

- `forge3d.memory_metrics()`
- `forge3d.budget_remaining()`
- `forge3d.utilization_ratio()`

These helpers report the current tracked memory state without exposing the
renderer's internal allocators directly.
