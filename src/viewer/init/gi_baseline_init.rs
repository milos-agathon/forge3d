// src/viewer/init/gi_baseline_init.rs
// GI baseline pipeline initialization for the Viewer

use crate::viewer::render_targets::{Target, TargetKind};
use std::sync::Arc;
use wgpu::{BindGroupLayout, ComputePipeline, Device, TextureView};

use crate::core::error::RenderResult;
use crate::core::resource_tracker::TrackedTexture;

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
    pub gi_baseline_hdr: TrackedTexture,
    pub gi_baseline_hdr_view: TextureView,
    pub gi_baseline_diffuse_hdr: TrackedTexture,
    pub gi_baseline_diffuse_hdr_view: TextureView,
    pub gi_baseline_spec_hdr: TrackedTexture,
    pub gi_baseline_spec_hdr_view: TextureView,
    pub gi_output_hdr: TrackedTexture,
    pub gi_output_hdr_view: TextureView,
    pub gi_debug: TrackedTexture,
    pub gi_debug_view: TextureView,
}

/// Create GI baseline resources
pub fn create_gi_baseline_resources(
    device: &Arc<Device>,
    width: u32,
    height: u32,
) -> RenderResult<GiBaselineResources> {
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

    let gi_baseline_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.gi.baseline.shader",
        GI_BASELINE_SHADER,
    );

    let gi_baseline_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.gi.baseline.pl"),
        bind_group_layouts: &[&gi_baseline_bgl],
        push_constant_ranges: &[],
    });

    let gi_baseline_pipeline = crate::core::shader_registry::with_error_scope(
        device,
        "viewer.gi.baseline.pipeline",
        || {
            crate::core::shader_registry::create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("viewer.gi.baseline.pipeline"),
                    layout: Some(&gi_baseline_pl),
                    module: &gi_baseline_shader,
                    entry_point: "cs_main",
                },
            )
        },
    );

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

    let gi_split_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.gi.split.shader",
        GI_SPLIT_SHADER,
    );

    let gi_split_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.gi.split.pl"),
        bind_group_layouts: &[&gi_split_bgl],
        push_constant_ranges: &[],
    });

    let gi_split_pipeline =
        crate::core::shader_registry::with_error_scope(device, "viewer.gi.split.pipeline", || {
            crate::core::shader_registry::create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("viewer.gi.split.pipeline"),
                    layout: Some(&gi_split_pl),
                    module: &gi_split_shader,
                    entry_point: "cs_main",
                },
            )
        });

    // Create HDR textures
    let Target {
        texture: gi_baseline_hdr,
        view: gi_baseline_hdr_view,
    } = TargetKind::GiBaseline.create(device, width, height)?;

    let Target {
        texture: gi_baseline_diffuse_hdr,
        view: gi_baseline_diffuse_hdr_view,
    } = TargetKind::GiDiffuse.create(device, width, height)?;

    let Target {
        texture: gi_baseline_spec_hdr,
        view: gi_baseline_spec_hdr_view,
    } = TargetKind::GiSpecular.create(device, width, height)?;

    let Target {
        texture: gi_output_hdr,
        view: gi_output_hdr_view,
    } = TargetKind::GiOutput.create(device, width, height)?;

    let Target {
        texture: gi_debug,
        view: gi_debug_view,
    } = TargetKind::GiDebug.create(device, width, height)?;

    Ok(GiBaselineResources {
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
    })
}
