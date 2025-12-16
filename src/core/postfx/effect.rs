use super::config::PostFxConfig;
use super::resources::PostFxResourcePool;
use crate::core::error::RenderResult;
use crate::core::gpu_timing::GpuTimingManager;
use wgpu::*;

/// Post-processing effect definition
pub trait PostFxEffect: Send + Sync {
    /// Get effect name
    fn name(&self) -> &str;

    /// Get effect configuration
    fn config(&self) -> &PostFxConfig;

    /// Set effect parameter
    fn set_parameter(&mut self, name: &str, value: f32) -> RenderResult<()>;

    /// Get effect parameter
    fn get_parameter(&self, name: &str) -> Option<f32>;

    /// Initialize effect resources
    fn initialize(
        &mut self,
        device: &Device,
        resource_pool: &mut PostFxResourcePool,
    ) -> RenderResult<()>;

    /// Execute effect compute pass
    fn execute(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        input: &TextureView,
        output: &TextureView,
        resource_pool: &PostFxResourcePool,
        timing_manager: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()>;

    /// Cleanup effect resources
    fn cleanup(&mut self) -> RenderResult<()> {
        Ok(())
    }
}
