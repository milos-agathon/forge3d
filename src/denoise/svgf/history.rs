// src/denoise/svgf/history.rs
// History buffer management notes for SVGF (stub)
// Why: clarify ping-pong layouts and reset triggers without changing runtime code yet
// RELEVANT FILES:src/shaders/svgf_history_reset.wgsl,src/denoise/svgf/pipelines.rs,src/path_tracing/mod.rs

#[allow(dead_code)]
pub enum ResetReason {
    FirstFrame = 1,
    ResolutionChange = 2,
    LargeMotion = 4,
}

