use crate::core::error::RenderError;
use std::sync::Arc;
use wgpu::*;

/// Task for a worker thread to execute
pub trait CommandTask: Send + Sync {
    /// Error type for task execution
    type Error: std::error::Error + Send + 'static;

    /// Execute the task, recording commands into the encoder
    fn execute(
        &self,
        encoder: &mut CommandEncoder,
        device: &Device,
        queue: &Queue,
    ) -> Result<usize, Self::Error>;

    /// Get a descriptive name for this task (for debugging/profiling)
    fn name(&self) -> &str;
}

/// Example task for copying buffers
pub struct CopyTask {
    name: String,
    src_buffer: Arc<Buffer>,
    dst_buffer: Arc<Buffer>,
    size: u64,
}

impl CopyTask {
    pub fn new(name: String, src: Arc<Buffer>, dst: Arc<Buffer>, size: u64) -> Self {
        Self {
            name,
            src_buffer: src,
            dst_buffer: dst,
            size,
        }
    }
}

impl CommandTask for CopyTask {
    type Error = RenderError;

    fn execute(
        &self,
        encoder: &mut CommandEncoder,
        _device: &Device,
        _queue: &Queue,
    ) -> Result<usize, Self::Error> {
        encoder.copy_buffer_to_buffer(&self.src_buffer, 0, &self.dst_buffer, 0, self.size);
        Ok(1) // One command recorded
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Example task for clearing textures
pub struct ClearTask {
    name: String,
    texture: Arc<Texture>,
    clear_color: Color,
}

impl ClearTask {
    pub fn new(name: String, texture: Arc<Texture>, clear_color: Color) -> Self {
        Self {
            name,
            texture,
            clear_color,
        }
    }
}

impl CommandTask for ClearTask {
    type Error = RenderError;

    fn execute(
        &self,
        encoder: &mut CommandEncoder,
        _device: &Device,
        _queue: &Queue,
    ) -> Result<usize, Self::Error> {
        let view = self.texture.create_view(&TextureViewDescriptor::default());

        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(&format!("clear_pass_{}", self.name)),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(self.clear_color),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        drop(render_pass); // End the render pass
        Ok(1) // One render pass recorded
    }

    fn name(&self) -> &str {
        &self.name
    }
}
