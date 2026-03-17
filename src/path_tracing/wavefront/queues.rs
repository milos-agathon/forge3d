// src/path_tracing/wavefront/queues.rs
// Queue management structures for wavefront path tracing
// Handles GPU buffers for rays, hits, scatter rays, and miss rays

mod intersect_shade;
mod raygen_shadow;
mod scatter_compact;
mod types;

pub use types::{Hit, QueueBuffers, QueueHeader, Ray, ScatterRay, ShadowRay};
