// src/viewer/init/gi_baseline_init.rs
// GI baseline pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{BindGroupLayout, ComputePipeline, Device, Texture, TextureView};

/// GI baseline shader source (copy lit to HDR)
pub const GI_BASELINE_SHADER: &str = r#"
@group(0) @binding(0) var lit_tex : texture_2d<f32>;
@group(0) @binding(1) var hdr_tex : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(lit_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coord = vec2<i32>(gid.xy);
    let c = textureLoad(lit_tex, coord, 0);
    textureStore(hdr_tex, coord, c);
}
"#;

/// GI split shader source (split lit into diffuse/spec)
pub const GI_SPLIT_SHADER: &str = r#"
@group(0) @binding(0) var lit_tex     : texture_2d<f32>;
@group(0) @binding(1) var normal_tex  : texture_2d<f32>;
@group(0) @binding(2) var material_tex: texture_2d<f32>;
@group(0) @binding(3) var diff_tex    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var spec_tex    : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(lit_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coord = vec2<i32>(gid.xy);
    let lit = textureLoad(lit_tex, coord, 0).rgb;
    let mat = textureLoad(material_tex, coord, 0);
    let metallic = mat.a;
    // Approximate split: specular contribution scales with metallic
    let spec = lit * metallic * 0.3;
    let diff = lit - spec;
    textureStore(diff_tex, coord, vec4<f32>(diff, 1.0));
    textureStore(spec_tex, coord, vec4<f32>(spec, 1.0));
}
"#;

/// Resources created during GI baseline initialization
pub struct GiBaselineResources {
    pub gi_baseline_bgl: BindGroupLayout,
    pub gi_baseline_pipeline: ComputePipeline,
    pub gi_split_bgl: BindGroupLayout,
    pub gi_split_pipeline: ComputePipeline,
    pub gi_baseline_hdr: Texture,
    pub gi_baseline_hdr_view: TextureView,
    pub gi_baseline_diffuse_hdr: Texture,
    pub gi_baseline_diffuse_hdr_view: TextureView,
    pub gi_baseline_spec_hdr: Texture,
    pub gi_baseline_spec_hdr_view: TextureView,
    pub gi_output_hdr: Texture,
    pub gi_output_hdr_view: TextureView,
    pub gi_debug: Texture,
    pub gi_debug_view: TextureView,
}

/// Create GI baseline resources
pub fn create_gi_baseline_resources(device: &Arc<Device>, width: u32, height: u32) -> GiBaselineResources {
    // GI baseline bind group layout
    let gi_baseline_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.gi.baseline.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let gi_baseline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.gi.baseline.shader"),
        source: wgpu::ShaderSource::Wgsl(GI_BASELINE_SHADER.into()),
    });

    let gi_baseline_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.gi.baseline.pl"),
        bind_group_layouts: &[&gi_baseline_bgl],
        push_constant_ranges: &[],
    });

    let gi_baseline_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.gi.baseline.pipeline"),
        layout: Some(&gi_baseline_pl),
        module: &gi_baseline_shader,
        entry_point: "cs_main",
    });

    // GI split bind group layout
    let gi_split_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.gi.split.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let gi_split_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.gi.split.shader"),
        source: wgpu::ShaderSource::Wgsl(GI_SPLIT_SHADER.into()),
    });

    let gi_split_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.gi.split.pl"),
        bind_group_layouts: &[&gi_split_bgl],
        push_constant_ranges: &[],
    });

    let gi_split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.gi.split.pipeline"),
        layout: Some(&gi_split_pl),
        module: &gi_split_shader,
        entry_point: "cs_main",
    });

    // Create HDR textures
    let gi_baseline_hdr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.gi.baseline.hdr"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let gi_baseline_hdr_view = gi_baseline_hdr.create_view(&wgpu::TextureViewDescriptor::default());

    let gi_baseline_diffuse_hdr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.gi.baseline.diffuse.hdr"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let gi_baseline_diffuse_hdr_view = gi_baseline_diffuse_hdr.create_view(&wgpu::TextureViewDescriptor::default());

    let gi_baseline_spec_hdr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.gi.baseline.spec.hdr"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let gi_baseline_spec_hdr_view = gi_baseline_spec_hdr.create_view(&wgpu::TextureViewDescriptor::default());

    let gi_output_hdr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.gi.output.hdr"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let gi_output_hdr_view = gi_output_hdr.create_view(&wgpu::TextureViewDescriptor::default());

    let gi_debug = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.gi.debug"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let gi_debug_view = gi_debug.create_view(&wgpu::TextureViewDescriptor::default());

    GiBaselineResources {
        gi_baseline_bgl,
        gi_baseline_pipeline,
        gi_split_bgl,
        gi_split_pipeline,
        gi_baseline_hdr,
        gi_baseline_hdr_view,
        gi_baseline_diffuse_hdr,
        gi_baseline_diffuse_hdr_view,
        gi_baseline_spec_hdr,
        gi_baseline_spec_hdr_view,
        gi_output_hdr,
        gi_output_hdr_view,
        gi_debug,
        gi_debug_view,
    }
}
