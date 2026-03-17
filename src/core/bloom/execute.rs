use crate::core::error::{RenderError, RenderResult};
use crate::core::gpu_timing::GpuTimingManager;
use crate::core::postfx::PostFxResourcePool;
use wgpu::*;

use super::config::{BloomBlurUniforms, BloomBrightPassUniforms, BloomCompositeUniforms};
use super::BloomEffect;

pub(super) fn execute_effect(
    effect: &BloomEffect,
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    input: &TextureView,
    output: &TextureView,
    resource_pool: &PostFxResourcePool,
    mut timing_manager: Option<&mut GpuTimingManager>,
) -> RenderResult<()> {
    if !effect.bloom_config.enabled {
        return Ok(());
    }

    let timing_scope = if let Some(timer) = timing_manager.as_mut() {
        Some(timer.begin_scope(encoder, "bloom"))
    } else {
        None
    };

    let brightpass_pipeline = effect
        .brightpass_pipeline
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom brightpass pipeline not initialized".into()))?;
    let blur_h_pipeline = effect
        .blur_h_pipeline
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom blur_h pipeline not initialized".into()))?;
    let blur_v_pipeline = effect
        .blur_v_pipeline
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom blur_v pipeline not initialized".into()))?;
    let composite_pipeline = effect
        .composite_pipeline
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom composite pipeline not initialized".into()))?;
    let brightpass_layout = effect
        .brightpass_layout
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom brightpass layout not initialized".into()))?;
    let blur_layout = effect
        .blur_layout
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom blur layout not initialized".into()))?;
    let composite_layout = effect
        .composite_layout
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom composite layout not initialized".into()))?;
    let brightpass_uniform_buf = effect
        .brightpass_uniform_buffer
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom brightpass UBO not initialized".into()))?;
    let blur_uniform_buf = effect
        .blur_uniform_buffer
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom blur UBO not initialized".into()))?;
    let composite_uniform_buf = effect
        .composite_uniform_buffer
        .as_ref()
        .ok_or_else(|| RenderError::Render("Bloom composite UBO not initialized".into()))?;
    let bp_idx = effect
        .brightpass_texture_index
        .ok_or_else(|| RenderError::Render("Bloom brightpass texture index not set".into()))?;
    let bt_idx = effect
        .blur_temp_texture_index
        .ok_or_else(|| RenderError::Render("Bloom blur temp texture index not set".into()))?;

    let bright_view = resource_pool.get_current_ping_pong(bp_idx).ok_or_else(|| {
        RenderError::Render("Bloom brightpass ping-pong texture unavailable".into())
    })?;
    let blur_temp_view = resource_pool.get_current_ping_pong(bt_idx).ok_or_else(|| {
        RenderError::Render("Bloom blur temp ping-pong texture unavailable".into())
    })?;
    let blur_result_view = resource_pool
        .get_previous_ping_pong(bp_idx)
        .ok_or_else(|| {
            RenderError::Render("Bloom blur result ping-pong texture unavailable".into())
        })?;

    queue.write_buffer(
        brightpass_uniform_buf,
        0,
        bytemuck::bytes_of(&BloomBrightPassUniforms {
            threshold: effect.bloom_config.threshold,
            softness: effect.bloom_config.softness,
            _pad: [0.0; 2],
        }),
    );
    queue.write_buffer(
        blur_uniform_buf,
        0,
        bytemuck::bytes_of(&BloomBlurUniforms {
            radius: effect.bloom_config.radius,
            strength: 1.0,
            _pad: [0.0; 2],
        }),
    );
    queue.write_buffer(
        composite_uniform_buf,
        0,
        bytemuck::bytes_of(&BloomCompositeUniforms {
            intensity: effect.bloom_config.strength,
            _pad: [0.0; 3],
        }),
    );

    let pool_w = resource_pool.width().max(1);
    let pool_h = resource_pool.height().max(1);
    let workgroups_x = (pool_w + 15) / 16;
    let workgroups_y = (pool_h + 15) / 16;

    dispatch_pass(
        device,
        encoder,
        "bloom_brightpass_bg",
        "bloom_brightpass",
        brightpass_layout,
        brightpass_pipeline,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(input),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(bright_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: brightpass_uniform_buf.as_entire_binding(),
            },
        ],
        workgroups_x,
        workgroups_y,
    );
    dispatch_pass(
        device,
        encoder,
        "bloom_blur_h_bg",
        "bloom_blur_h",
        blur_layout,
        blur_h_pipeline,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(bright_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(blur_temp_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: blur_uniform_buf.as_entire_binding(),
            },
        ],
        workgroups_x,
        workgroups_y,
    );
    dispatch_pass(
        device,
        encoder,
        "bloom_blur_v_bg",
        "bloom_blur_v",
        blur_layout,
        blur_v_pipeline,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(blur_temp_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(blur_result_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: blur_uniform_buf.as_entire_binding(),
            },
        ],
        workgroups_x,
        workgroups_y,
    );
    dispatch_pass(
        device,
        encoder,
        "bloom_composite_bg",
        "bloom_composite",
        composite_layout,
        composite_pipeline,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(input),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(blur_result_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(output),
            },
            BindGroupEntry {
                binding: 3,
                resource: composite_uniform_buf.as_entire_binding(),
            },
        ],
        workgroups_x,
        workgroups_y,
    );

    log::debug!(
        target: "bloom",
        "P1.2: Bloom executed: threshold={:.2}, strength={:.2}, radius={:.1}",
        effect.bloom_config.threshold,
        effect.bloom_config.strength,
        effect.bloom_config.radius
    );

    let _ = timing_scope;
    Ok(())
}

fn dispatch_pass(
    device: &Device,
    encoder: &mut CommandEncoder,
    bind_group_label: &str,
    pass_label: &str,
    layout: &BindGroupLayout,
    pipeline: &ComputePipeline,
    entries: &[BindGroupEntry<'_>],
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some(bind_group_label),
        layout,
        entries,
    });

    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(pass_label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
