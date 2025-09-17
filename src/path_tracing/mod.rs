// src/path_tracing/mod.rs
// Public GPU path tracing entry points and types for A1.
// This exists to expose a minimal compute-based tracer to Python via PyO3 and internal Rust use.
// RELEVANT FILES:src/path_tracing/compute.rs,src/shaders/pt_kernel.wgsl,python/forge3d/path_tracing.py,src/lib.rs

pub mod accel;
pub mod alias_table;
pub mod aov;
pub mod compute;
pub mod guiding;
pub mod hybrid_compute;
pub mod io;
pub mod restir;
pub mod wavefront;
pub mod mesh;

// Note: SVGF integration stubs live under src/denoise/svgf for future wiring.
// Keeping this file unchanged functionally to avoid build impact.

/// Parameters for path tracing configuration
#[derive(Clone, Debug)]
pub struct TracerParams {
    pub samples_per_pixel: u32,
    pub max_depth: u32,
    pub engine: TracerEngine,
}

/// Path tracing engine selection
#[derive(Clone, Debug, PartialEq)]
pub enum TracerEngine {
    Megakernel,
    Wavefront,
}

impl Default for TracerParams {
    fn default() -> Self {
        Self {
            samples_per_pixel: 64,
            max_depth: 8,
            engine: TracerEngine::Megakernel,
        }
    }
}
