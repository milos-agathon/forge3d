// src/path_tracing/mod.rs
// Public GPU path tracing entry points and types for A1.
// This exists to expose a minimal compute-based tracer to Python via PyO3 and internal Rust use.
// RELEVANT FILES:src/path_tracing/compute.rs,src/shaders/pt_kernel.wgsl,python/forge3d/path_tracing.py,src/lib.rs

pub mod compute;
pub mod accel;

