use crate::core::error::RenderResult;
use crate::core::postfx::PostFxResourcePool;

use super::resources::{
    allocate_resource_indices, create_layouts, create_pipelines, create_uniform_buffers,
};
use super::BloomEffect;

pub(super) fn initialize_effect(
    effect: &mut BloomEffect,
    device: &wgpu::Device,
    resource_pool: &mut PostFxResourcePool,
) -> RenderResult<()> {
    let layouts = create_layouts(device);
    let pipelines = create_pipelines(device, &layouts);
    let uniforms = create_uniform_buffers(device);
    let (brightpass_idx, blur_temp_idx) = allocate_resource_indices(device, resource_pool)?;

    effect.brightpass_pipeline = Some(pipelines.brightpass);
    effect.blur_h_pipeline = Some(pipelines.blur_h);
    effect.blur_v_pipeline = Some(pipelines.blur_v);
    effect.composite_pipeline = Some(pipelines.composite);
    effect.brightpass_layout = Some(layouts.brightpass);
    effect.blur_layout = Some(layouts.blur);
    effect.composite_layout = Some(layouts.composite);
    effect.brightpass_uniform_buffer = Some(uniforms.brightpass);
    effect.blur_uniform_buffer = Some(uniforms.blur);
    effect.composite_uniform_buffer = Some(uniforms.composite);
    effect.brightpass_texture_index = Some(brightpass_idx);
    effect.blur_temp_texture_index = Some(blur_temp_idx);

    Ok(())
}
