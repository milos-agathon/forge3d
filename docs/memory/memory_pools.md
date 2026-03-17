# GPU Memory Pools

The current repository contains Rust-side allocation and pooling logic, but
there is no public Python module named `forge3d.memory` for direct pool
management.

## Public Python view

From Python, memory management is currently observational:

- `forge3d.memory_metrics()`
- `forge3d.budget_remaining()`
- `forge3d.utilization_ratio()`

## Internal status

Pool allocators and related resource managers are implementation details of the
renderer. This page should be read as an internal architecture topic rather
than a stable Python feature surface.
