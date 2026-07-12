use super::*;
use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};

pub(super) fn create_color_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> Result<(TrackedTexture, wgpu::TextureView), RenderError> {
    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("scene-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        },
    )?;
    let view = texture.create_view(&Default::default());
    Ok((texture, view))
}

pub(super) fn create_normal_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> Result<(TrackedTexture, wgpu::TextureView), RenderError> {
    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("scene-normal"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: NORMAL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )?;
    let view = texture.create_view(&Default::default());
    Ok((texture, view))
}

pub(super) fn create_msaa_normal_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> Result<(Option<TrackedTexture>, Option<wgpu::TextureView>), RenderError> {
    if sample_count <= 1 {
        return Ok((None, None));
    }

    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("scene-msaa-normal"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: NORMAL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
    )?;
    let view = texture.create_view(&Default::default());
    Ok((Some(texture), Some(view)))
}

pub(super) fn create_msaa_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> Result<(Option<TrackedTexture>, Option<wgpu::TextureView>), RenderError> {
    if sample_count <= 1 {
        return Ok((None, None));
    }

    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("scene-msaa-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
    )?;
    let view = texture.create_view(&Default::default());
    Ok((Some(texture), Some(view)))
}

pub(super) fn create_depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> Result<(Option<TrackedTexture>, Option<wgpu::TextureView>), RenderError> {
    if sample_count <= 1 {
        return Ok((None, None));
    }

    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("scene-depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
    )?;
    let view = texture.create_view(&Default::default());
    Ok((Some(texture), Some(view)))
}
