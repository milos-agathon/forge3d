// src/viewer/state/gpu_state.rs
// Core GPU resources for the Viewer

use std::sync::Arc;
use wgpu::{Adapter, Device, Queue};

/// Core GPU resources shared across the viewer
pub struct GpuCore {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter: Arc<Adapter>,
}

impl GpuCore {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, adapter: Arc<Adapter>) -> Self {
        Self {
            device,
            queue,
            adapter,
        }
    }
}
