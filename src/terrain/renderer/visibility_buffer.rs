//! TESSELLA terrain visibility buffer.
//!
//! Zero is reserved for background. Visible primitives are encoded as
//! `1 + ((tile_id & 0x00ff_ffff) << 8) | (triangle_id & 0xff)`. The extra one
//! keeps tile zero / triangle zero distinct from background. The material
//! resolve re-rasterizes only depth-equal fragments, so POM and VT feedback
//! execute exactly once for every non-background visibility pixel.

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use bytemuck::{Pod, Zeroable};
use std::sync::{Mutex, OnceLock};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct VisibilityCounters {
    visible_pixels: u32,
    feedback_records: u32,
    material_invocations: u32,
    background_pixels: u32,
    fallback_texels: u32,
    forward_material_invocations: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisibilityStats {
    pub visible_pixels: u32,
    pub feedback_records: u32,
    pub material_invocations: u32,
    pub background_pixels: u32,
    pub fallback_texels: u32,
    pub forward_material_invocations: u32,
}

static LAST_STATS: OnceLock<Mutex<VisibilityStats>> = OnceLock::new();

pub fn publish_stats(stats: VisibilityStats) {
    if let Ok(mut current) = LAST_STATS
        .get_or_init(|| Mutex::new(VisibilityStats::default()))
        .lock()
    {
        *current = stats;
    }
}

pub fn latest_stats() -> VisibilityStats {
    LAST_STATS
        .get_or_init(|| Mutex::new(VisibilityStats::default()))
        .lock()
        .map(|stats| *stats)
        .unwrap_or_default()
}

pub struct TerrainVisibilityBuffer {
    width: u32,
    height: u32,
    _texture: TrackedTexture,
    view: wgpu::TextureView,
    stats_buffer: TrackedBuffer,
    stats_readback: TrackedBuffer,
    stats_bind_group: wgpu::BindGroup,
    stats_pipeline: wgpu::ComputePipeline,
    staged: bool,
}

impl TerrainVisibilityBuffer {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        frame_counters: &wgpu::Buffer,
    ) -> RenderResult<Self> {
        let width = width.max(1);
        let height = height.max(1);
        let texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.visibility.ids"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let stats_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.visibility.stats"),
                size: std::mem::size_of::<VisibilityCounters>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let stats_readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.visibility.stats_readback"),
                size: std::mem::size_of::<VisibilityCounters>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.visibility.stats.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let stats_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.visibility.stats.bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame_counters.as_entire_binding(),
                },
            ],
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_visbuffer_resolve",
            include_str!("../../shaders/terrain_visbuffer_resolve.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.visibility.stats.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let stats_pipeline = crate::core::shader_registry::with_error_scope(
            device,
            "terrain.visibility.stats.pipeline",
            || {
                crate::core::shader_registry::create_compute_pipeline_scoped(
                    device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("terrain.visibility.stats.pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: "cs_main",
                    },
                )
            },
        );
        Ok(Self {
            width,
            height,
            _texture: texture,
            view,
            stats_buffer,
            stats_readback,
            stats_bind_group,
            stats_pipeline,
            staged: false,
        })
    }

    pub fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width.max(1) && self.height == height.max(1)
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self._texture
    }

    pub fn stage_stats(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.stats_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.visibility.stats.pass"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("terrain_visbuffer_resolve");
            pass.set_pipeline(&self.stats_pipeline);
            pass.set_bind_group(0, &self.stats_bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.stats_buffer,
            0,
            &self.stats_readback,
            0,
            std::mem::size_of::<VisibilityCounters>() as u64,
        );
        self.staged = true;
    }

    pub fn finish_frame(&mut self, device: &wgpu::Device) -> RenderResult<VisibilityStats> {
        if !self.staged {
            return Ok(VisibilityStats::default());
        }
        let slice = self.stats_readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| RenderError::render("visibility stats callback dropped"))?
            .map_err(|error| {
                RenderError::render(format!("visibility stats map failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range();
        let counters = bytemuck::pod_read_unaligned::<VisibilityCounters>(&mapped);
        drop(mapped);
        self.stats_readback.unmap();
        self.staged = false;
        Ok(VisibilityStats {
            visible_pixels: counters.visible_pixels,
            feedback_records: counters.feedback_records,
            material_invocations: counters.material_invocations,
            background_pixels: counters.background_pixels,
            fallback_texels: counters.fallback_texels,
            forward_material_invocations: counters.forward_material_invocations,
        })
    }
}

#[cfg(feature = "extension-module")]
impl super::TerrainScene {
    pub(super) fn ensure_visibility_buffer(&self, width: u32, height: u32) -> anyhow::Result<()> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        if visibility
            .as_ref()
            .is_none_or(|buffer| !buffer.matches(width, height))
        {
            *visibility = Some(
                TerrainVisibilityBuffer::new(
                    self.device.as_ref(),
                    width,
                    height,
                    &self.vt_frame_counters_buffer,
                )
                .map_err(anyhow::Error::msg)?,
            );
        }
        Ok(())
    }

    pub(super) fn stage_visibility_stats(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<()> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain visibility buffer not initialized"))?;
        buffer.stage_stats(encoder);
        Ok(())
    }

    pub(super) fn finish_visibility_frame(&self) -> anyhow::Result<VisibilityStats> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let Some(buffer) = visibility.as_mut() else {
            let stats = VisibilityStats::default();
            publish_stats(stats);
            return Ok(stats);
        };
        let stats = buffer
            .finish_frame(self.device.as_ref())
            .map_err(anyhow::Error::msg)?;
        publish_stats(stats);
        Ok(stats)
    }

    pub(super) fn pick_visibility_pixels(
        &self,
        pixels: &[(u32, u32)],
    ) -> anyhow::Result<Vec<Option<(u32, u32)>>> {
        let visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no completed visibility render is available"))?;
        let picking =
            crate::picking::UnifiedPickingSystem::new(self.device.clone(), self.queue.clone());
        picking
            .pick_visibility_pixels(buffer.texture(), buffer.width, buffer.height, pixels)
            .map_err(anyhow::Error::msg)
    }
}
