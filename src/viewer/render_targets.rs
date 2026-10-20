//! Viewport allocation and installation. Preparation never changes the live viewer.

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::core::screen_space_effects::PreparedScreenSpaceTargets;
use crate::core::taa::PreparedTaaTargets;
use crate::viewer::init::sky_output_descriptor;
use crate::viewer::Viewer;
use wgpu::{Device, TextureDescriptor, TextureFormat, TextureUsages, TextureView};

#[derive(Clone, Copy)]
pub(crate) enum TargetKind {
    Lit,
    GiBaseline,
    GiDiffuse,
    GiSpecular,
    GiOutput,
    GiDebug,
    Sky,
    Depth,
    FogOutput,
    FogHistory,
    FogOutputHalf,
    FogHistoryHalf,
}

impl TargetKind {
    pub(crate) fn descriptor(self, width: u32, height: u32) -> TextureDescriptor<'static> {
        use TargetKind::*;
        use TextureUsages as U;
        if matches!(self, Sky) {
            return sky_output_descriptor(width, height);
        }
        let (label, format, usage) = match self {
            Lit => (
                "viewer.lit.output",
                TextureFormat::Rgba8Unorm,
                U::STORAGE_BINDING | U::TEXTURE_BINDING,
            ),
            GiDebug => (
                "viewer.gi.debug",
                TextureFormat::Rgba16Float,
                U::STORAGE_BINDING | U::TEXTURE_BINDING,
            ),
            Depth => (
                "viewer.gbuf.z",
                TextureFormat::Depth32Float,
                U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            ),
            FogHistory | FogHistoryHalf => (
                if matches!(self, FogHistory) {
                    "viewer.fog.history"
                } else {
                    "viewer.fog.history.half"
                },
                TextureFormat::Rgba16Float,
                U::TEXTURE_BINDING | U::COPY_DST,
            ),
            _ => {
                let label = match self {
                    GiBaseline => "viewer.gi.baseline.hdr",
                    GiDiffuse => "viewer.gi.baseline.diffuse.hdr",
                    GiSpecular => "viewer.gi.baseline.spec.hdr",
                    GiOutput => "viewer.gi.output.hdr",
                    FogOutput => "viewer.fog.output",
                    FogOutputHalf => "viewer.fog.output.half",
                    _ => unreachable!(),
                };
                (
                    label,
                    TextureFormat::Rgba16Float,
                    U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC,
                )
            }
        };
        let (width, height) = if matches!(self, FogOutputHalf | FogHistoryHalf) {
            ((width / 2).max(1), (height / 2).max(1))
        } else {
            (width, height)
        };
        TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        }
    }

    pub(crate) fn create(self, device: &Device, width: u32, height: u32) -> RenderResult<Target> {
        Target::create(device, &self.descriptor(width, height))
    }
}

pub(crate) struct Target {
    pub(crate) texture: TrackedTexture,
    pub(crate) view: TextureView,
}

impl Target {
    fn create(device: &Device, descriptor: &TextureDescriptor<'_>) -> RenderResult<Self> {
        let texture = tracked_create_texture(device, descriptor)?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self { texture, view })
    }
}

pub(crate) struct PreparedViewportTargets {
    width: u32,
    height: u32,
    lit: Target,
    gi_baseline: Target,
    gi_diffuse: Target,
    gi_specular: Target,
    gi_output: Target,
    gi_debug: Target,
    sky: Target,
    depth: Option<Target>,
    fog_output: Target,
    fog_history: Target,
    fog_output_half: Target,
    fog_history_half: Target,
    gi: Option<PreparedScreenSpaceTargets>,
    taa: Option<PreparedTaaTargets>,
}

impl PreparedViewportTargets {
    pub(crate) fn prepare(viewer: &Viewer, width: u32, height: u32) -> RenderResult<Self> {
        Self::prepare_with(viewer, width, height, Target::create)
    }

    fn prepare_with(
        viewer: &Viewer,
        width: u32,
        height: u32,
        mut allocate: impl FnMut(&Device, &TextureDescriptor<'_>) -> RenderResult<Target>,
    ) -> RenderResult<Self> {
        let device = &viewer.device;
        let limit = device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > limit || height > limit {
            return Err(RenderError::Render(format!(
                "Invalid viewport size {width}x{height}; device maximum is {limit}"
            )));
        }
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = (|| {
            let mut target = |kind: TargetKind| allocate(device, &kind.descriptor(width, height));
            Ok(Self {
                width,
                height,
                lit: target(TargetKind::Lit)?,
                gi_baseline: target(TargetKind::GiBaseline)?,
                gi_diffuse: target(TargetKind::GiDiffuse)?,
                gi_specular: target(TargetKind::GiSpecular)?,
                gi_output: target(TargetKind::GiOutput)?,
                gi_debug: target(TargetKind::GiDebug)?,
                sky: target(TargetKind::Sky)?,
                depth: viewer
                    .geom_pipeline
                    .as_ref()
                    .map(|_| target(TargetKind::Depth))
                    .transpose()?,
                fog_output: target(TargetKind::FogOutput)?,
                fog_history: target(TargetKind::FogHistory)?,
                fog_output_half: target(TargetKind::FogOutputHalf)?,
                fog_history_half: target(TargetKind::FogHistoryHalf)?,
                gi: viewer
                    .gi
                    .as_ref()
                    .map(|gi| gi.prepare_resize(device, width, height))
                    .transpose()?,
                taa: viewer
                    .taa_renderer
                    .as_ref()
                    .map(|taa| taa.prepare_resize(device, width, height))
                    .transpose()?,
            })
        })();
        let validation = pollster::block_on(device.pop_error_scope());
        let allocation = pollster::block_on(device.pop_error_scope());
        if let Some(error) = validation.or(allocation) {
            return Err(RenderError::Device(format!(
                "Viewport allocation {width}x{height} failed: {error}"
            )));
        }
        result
    }

    /// No allocation or fallible operation may occur after preparation succeeds.
    pub(crate) fn install(self, viewer: &mut Viewer) {
        macro_rules! install {
            ($target:ident, $texture:ident, $view:ident) => {
                viewer.$texture = self.$target.texture;
                viewer.$view = self.$target.view;
            };
        }
        install!(lit, lit_output, lit_output_view);
        install!(gi_baseline, gi_baseline_hdr, gi_baseline_hdr_view);
        install!(
            gi_diffuse,
            gi_baseline_diffuse_hdr,
            gi_baseline_diffuse_hdr_view
        );
        install!(gi_specular, gi_baseline_spec_hdr, gi_baseline_spec_hdr_view);
        install!(gi_output, gi_output_hdr, gi_output_hdr_view);
        install!(gi_debug, gi_debug, gi_debug_view);
        install!(sky, sky_output, sky_output_view);
        install!(fog_output, fog_output, fog_output_view);
        install!(fog_history, fog_history, fog_history_view);
        install!(fog_output_half, fog_output_half, fog_output_half_view);
        install!(fog_history_half, fog_history_half, fog_history_half_view);
        (viewer.z_texture, viewer.z_view) = match self.depth {
            Some(target) => (Some(target.texture), Some(target.view)),
            None => (None, None),
        };
        if let (Some(gi), Some(prepared)) = (viewer.gi.as_mut(), self.gi) {
            gi.install_resize(prepared, &viewer.queue);
        }
        if let Some(pass) = viewer.gi_pass.as_mut() {
            pass.resize(self.width, self.height);
        }
        if let (Some(taa), Some(prepared)) = (viewer.taa_renderer.as_mut(), self.taa) {
            taa.install_resize(prepared);
        }
        viewer.lit_bind_group_cache.borrow_mut().take();
        viewer.comp_bind_group_cache.borrow_mut().clear();
        viewer.sky_bg0_cache.borrow_mut().take();
        viewer.sky_present_bg_cache.borrow_mut().take();
        viewer.fog_bg0_cache.borrow_mut().take();
        viewer.fog_bg2_cache.borrow_mut().take();
        viewer.fog_bg2_half_cache.borrow_mut().take();
        viewer.fog_upsample_bg_cache.borrow_mut().take();
        viewer.invalidate_viewport_histories(self.width, self.height);
        viewer.hud.set_resolution(self.width, self.height);
    }
}

impl Viewer {
    pub(crate) fn render_targets_match(&self, width: u32, height: u32) -> bool {
        (self.lit_output.width(), self.lit_output.height()) == (width, height)
    }
}

#[cfg(test)]
mod tests;
