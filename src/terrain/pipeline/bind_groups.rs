use super::TerrainPipeline;
use crate::core::reflections::PlanarReflectionRenderer;
use wgpu::*;

pub fn make_bg_globals(pipeline: &TerrainPipeline, device: &Device, ubo: &Buffer) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vf.Terrain.bg.globals"),
        layout: &pipeline.bgl_globals,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: ubo.as_entire_binding(),
        }],
    })
}

/// E2/E1: Per-tile uniform + page table bind group helper
pub fn make_bg_tile(
    pipeline: &TerrainPipeline,
    device: &Device,
    tile_ubo: &Buffer,
    page_table: Option<&Buffer>,
    tile_slot_ubo: &Buffer,
    mosaic_params_ubo: &Buffer,
) -> BindGroup {
    let pt_dummy = device.create_buffer(&BufferDescriptor {
        label: Some("vf.Terrain.page_table.dummy"),
        // Must be at least the size of one PageTableEntry (8 u32 = 32 bytes)
        size: 32,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pt_binding = page_table
        .map(|b| b.as_entire_binding())
        .unwrap_or_else(|| pt_dummy.as_entire_binding());

    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vf.Terrain.bg.tile"),
        layout: &pipeline.bgl_tile,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: tile_ubo.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: pt_binding,
            },
            BindGroupEntry {
                binding: 2,
                resource: tile_slot_ubo.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: mosaic_params_ubo.as_entire_binding(),
            },
        ],
    })
}

/// Bind group for height texture/sampler
pub fn make_bg_height(pipeline: &TerrainPipeline, device: &Device, view: &TextureView, samp: &Sampler) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vf.Terrain.bg.height"),
        layout: &pipeline.bgl_height,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(samp),
            },
        ],
    })
}

pub fn make_bg_lut(pipeline: &TerrainPipeline, device: &Device, view: &TextureView, samp: &Sampler) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vf.Terrain.bg.lut"),
        layout: &pipeline.bgl_lut,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(samp),
            },
        ],
    })
}

/// Create bind group with texture array for descriptor indexing mode
pub fn make_bg_lut_array(
    pipeline: &TerrainPipeline,
    device: &Device,
    views: &[&TextureView],
    samp: &Sampler,
) -> BindGroup {
    if !pipeline.descriptor_indexing {
        panic!("make_bg_lut_array called but descriptor indexing is not available");
    }

    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vf.Terrain.bg.lut.array"),
        layout: &pipeline.bgl_lut,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(views),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(samp),
            },
        ],
    })
}

// B7: Cloud shadow bind group helper
pub fn make_bg_cloud_shadows(
    pipeline: &TerrainPipeline,
    device: &Device,
    view: &TextureView,
    samp: &Sampler,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vf.Terrain.bg.cloud_shadows"),
        layout: &pipeline.bgl_cloud_shadows,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(samp),
            },
        ],
    })
}

pub fn make_bg_reflection(
    pipeline: &TerrainPipeline,
    device: &Device,
    renderer: &PlanarReflectionRenderer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vf.Terrain.bg.reflection"),
        layout: &pipeline.bgl_reflection,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: renderer.uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&renderer.reflection_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&renderer.reflection_sampler),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&renderer.reflection_depth_view),
            },
        ],
    })
}
