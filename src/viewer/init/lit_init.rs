// src/viewer/init/lit_init.rs
// Lit compute pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupLayout, Buffer, ComputePipeline, Device, Sampler, Texture, TextureView,
};

use super::super::viewer_constants::LIT_WGSL_VERSION;

/// Lit shader source
pub const LIT_SHADER: &str = r#"
struct LitParams {
    // x,y,z = sun_dir_vs, w = sun_intensity
    sun_dir_and_intensity: vec4<f32>,
    // x = ibl_intensity, y = use_ibl (1.0|0.0), z = brdf index, w = pad
    ibl_use_brdf_pad: vec4<f32>,
    // x = roughness [0,1], y = debug_mode (0=off,1=roughness,2=NDF), z/w pad
    debug_extra: vec4<f32>,
};
@group(0) @binding(0) var normal_tex : texture_2d<f32>;
@group(0) @binding(1) var albedo_tex : texture_2d<f32>;
@group(0) @binding(2) var depth_tex  : texture_2d<f32>;
@group(0) @binding(3) var out_tex    : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var env_cube   : texture_cube<f32>;
@group(0) @binding(5) var env_samp   : sampler;
@group(0) @binding(6) var<uniform> P : LitParams;

const BRDF_LAMBERT: f32 = 0.0;
const BRDF_PHONG: f32 = 1.0;
const BRDF_GGX: f32 = 4.0;
const BRDF_DISNEY: f32 = 6.0;

fn approx_eq(a: f32, b: f32) -> bool { return abs(a - b) < 0.5; }

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(normal_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coord = vec2<i32>(gid.xy);
    var n = textureLoad(normal_tex, coord, 0).xyz;
    n = normalize(n);
    let a = textureLoad(albedo_tex, coord, 0).rgb;
    let l = -normalize(P.sun_dir_and_intensity.xyz);
    let rough = clamp(P.debug_extra.x, 0.0, 1.0);
    let dbg = u32(P.debug_extra.y + 0.5);

    if (dbg == 1u) {
        textureStore(out_tex, coord, vec4<f32>(rough, 0.0, 0.0, 1.0));
        return;
    }
    let ndl = max(dot(n, l), 0.0);
    var col = vec3<f32>(0.0);
    if (ndl > 0.0) {
        if (approx_eq(P.ibl_use_brdf_pad.z, BRDF_LAMBERT)) {
            let diffuse = a * (1.0 / 3.14159265);
            col = diffuse * P.sun_dir_and_intensity.w * ndl;
        } else if (approx_eq(P.ibl_use_brdf_pad.z, BRDF_PHONG)) {
            let v = vec3<f32>(0.0, 0.0, 1.0);
            let h = normalize(l + v);
            let shininess = 64.0;
            let spec = pow(max(dot(n, h), 0.0), shininess);
            let spec_c = mix(vec3<f32>(0.04), a, 0.0) * spec;
            let diffuse = a * (1.0 / 3.14159265);
            col = (diffuse + spec_c) * P.sun_dir_and_intensity.w * ndl;
        } else {
            let v = vec3<f32>(0.0, 0.0, 1.0);
            let h = normalize(l + v);
            let n_dot_h = max(dot(n, h), 0.0);
            let v_dot_h = max(dot(v, h), 0.0);
            let r = rough;
            let alpha = r * r;
            let denom = n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0;
            let D = (alpha * alpha) / (3.14159265 * denom * denom + 1e-6);
            let F0 = mix(vec3<f32>(0.04), a, 0.0);
            let F = F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - v_dot_h, 5.0);
            let kS = F;
            let kD = (vec3<f32>(1.0) - kS);
            let diffuse = kD * a * (1.0 / 3.14159265);
            let specular = F * D;
            col = (diffuse + specular) * P.sun_dir_and_intensity.w * ndl;
        }
    }
    col += 0.1 * a;
    if (dbg == 2u) {
        let v = vec3<f32>(0.0, 0.0, 1.0);
        let h = normalize(l + v);
        let n_dot_h = max(dot(n, h), 0.0);
        let alpha = max(1e-3, rough * rough);
        let a2 = alpha * alpha;
        let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        let D = a2 / max(3.14159265 * denom * denom, 1e-6);
        textureStore(out_tex, coord, vec4<f32>(D, D, D, 1.0));
        return;
    }
    if (P.ibl_use_brdf_pad.y > 0.5) {
        let env = textureSampleLevel(env_cube, env_samp, n, 0.0).rgb;
        col += a * env * P.ibl_use_brdf_pad.x;
    }
    textureStore(out_tex, coord, vec4<f32>(col, 1.0));
}
"#;

/// Resources created during lit pipeline initialization
pub struct LitResources {
    pub lit_bind_group_layout: BindGroupLayout,
    pub lit_pipeline: ComputePipeline,
    pub lit_uniform: Buffer,
    pub lit_output: Texture,
    pub lit_output_view: TextureView,
    pub dummy_env_view: TextureView,
    pub dummy_env_sampler: Sampler,
}

/// Create lit compute pipeline and resources
pub fn create_lit_resources(device: &Arc<Device>, width: u32, height: u32) -> LitResources {
    let lit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.lit.bgl"),
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
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let lit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.lit.compute.shader"),
        source: wgpu::ShaderSource::Wgsl(LIT_SHADER.into()),
    });

    let lit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.lit.pl"),
        bind_group_layouts: &[&lit_bgl],
        push_constant_ranges: &[],
    });

    let lit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("viewer.lit.pipeline"),
        layout: Some(&lit_pl),
        module: &lit_shader,
        entry_point: "cs_main",
    });

    println!("[viewer] lit compute WGSL version {} compiled", LIT_WGSL_VERSION);

    let lit_params: [f32; 12] = [
        0.3, 0.6, -1.0, 1.0,  // sun_dir_vs.xyz, sun_intensity
        0.6, 1.0, 4.0, 0.0,   // ibl_intensity, use_ibl, brdf, pad
        0.5, 0.0, 0.0, 0.0,   // roughness, debug_mode, pad, pad
    ];
    let lit_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewer.lit.uniform"),
        contents: bytemuck::cast_slice(&lit_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Dummy IBL cube (1x1x6) and sampler
    let dummy_env = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.lit.dummy.env"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let dummy_env_view = dummy_env.create_view(&wgpu::TextureViewDescriptor {
        label: Some("viewer.lit.dummy.env.view"),
        format: Some(wgpu::TextureFormat::Rgba8Unorm),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: Some(6),
    });
    let dummy_env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewer.lit.dummy.env.sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let lit_output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewer.lit.output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let lit_output_view = lit_output.create_view(&wgpu::TextureViewDescriptor::default());

    LitResources {
        lit_bind_group_layout: lit_bgl,
        lit_pipeline,
        lit_uniform,
        lit_output,
        lit_output_view,
        dummy_env_view,
        dummy_env_sampler,
    }
}
