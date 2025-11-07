// src/offscreen/brdf_tile.rs
// P7-01: Offscreen BRDF tile renderer scaffold
// Renders a single BRDF tile (UV-sphere or analytic patch) for gallery generation
// Headless rendering with no viewer dependencies, suitable for CI goldens
// RELEVANT FILES: src/renderer/readback.rs, src/pipeline/pbr.rs, src/shaders/pbr.wgsl, src/lighting/types.rs

use anyhow::{ensure, Result};
use wgpu::util::DeviceExt;

/// Create identity matrix
fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Create look-at view matrix
fn create_look_at_matrix(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = [
        center[0] - eye[0],
        center[1] - eye[1],
        center[2] - eye[2],
    ];
    let f_len = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
    let f = [f[0] / f_len, f[1] / f_len, f[2] / f_len];

    let s = [
        f[1] * up[2] - f[2] * up[1],
        f[2] * up[0] - f[0] * up[2],
        f[0] * up[1] - f[1] * up[0],
    ];
    let s_len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    let s = [s[0] / s_len, s[1] / s_len, s[2] / s_len];

    let u = [
        s[1] * f[2] - s[2] * f[1],
        s[2] * f[0] - s[0] * f[2],
        s[0] * f[1] - s[1] * f[0],
    ];

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-s[0] * eye[0] - s[1] * eye[1] - s[2] * eye[2],
         -u[0] * eye[0] - u[1] * eye[1] - u[2] * eye[2],
         f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2],
         1.0],
    ]
}

/// Create perspective projection matrix
fn create_perspective_matrix(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), -1.0],
        [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
    ]
}

/// Render a BRDF tile offscreen and return tight RGBA8 buffer.
///
/// # Arguments
/// * `device` - wgpu device for GPU resource creation
/// * `queue` - wgpu queue for command submission
/// * `model_u32` - BRDF model index (Lambert=0, Phong=1, GGX=4, Disney=6)
/// * `roughness` - Material roughness in [0,1]; shader uses alpha = roughness^2
/// * `width` - Output image width in pixels
/// * `height` - Output image height in pixels
/// * `ndf_only` - Debug mode: when true, output vec4(D, D, D, 1) where D is NDF value
/// * `g_only` - Milestone 0: Debug mode: output Smith G as grayscale
/// * `dfg_only` - Milestone 0: Debug mode: output D*F*G (pre-division by 4*nl*nv)
/// * `spec_only` - Milestone 0: Debug mode: output specular term only
/// * `roughness_visualize` - Milestone 0: Debug mode: output vec3(r) for uniform validation
/// * `exposure` - Exposure multiplier (default 1.0); tone mapping disabled for stability
/// * `light_intensity` - Light intensity multiplier (default 0.8)
/// * `clearcoat` - Disney Principled clearcoat amount (default 0.0)
/// * `clearcoat_roughness` - Disney Principled clearcoat roughness (default 0.0)
/// * `sheen` - Disney Principled sheen amount (default 0.0)
/// * `sheen_tint` - Disney Principled sheen tint (default 0.0)
/// * `specular_tint` - Disney Principled specular tint (default 0.0)
/// * `debug_dot_products` - Debug mode: when true, output min/max N·L, N·V to console
///
/// # Returns
/// Tight row-major RGBA8 buffer of shape (height, width, 4) with no padding.
///
/// # BRDF Model Indices
/// Per src/lighting/types.rs and src/shaders/lighting.wgsl:
/// - 0: Lambert (Lambertian diffuse)
/// - 1: Phong
/// - 2: Blinn-Phong
/// - 3: Oren-Nayar
/// - 4: Cook-Torrance GGX (most common)
/// - 5: Cook-Torrance Beckmann
/// - 6: Disney Principled
/// - 7: Ashikhmin-Shirley
/// - 8: Ward
/// - 9: Toon
/// - 10: Minnaert
/// - 11: Subsurface
/// - 12: Hair
///
/// # Notes
/// - Roughness is clamped to [0.0, 1.0] and mapped to alpha = roughness^2 in shader
/// - Tone mapping is disabled and exposure defaults to 1.0 for reproducible output
/// - NDF-only mode matches viewer math for D(α) to enable visual comparisons
pub fn render_brdf_tile_offscreen(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model_u32: u32,
    roughness: f32,
    width: u32,
    height: u32,
    ndf_only: bool,
    g_only: bool,
    dfg_only: bool,
    spec_only: bool,
    roughness_visualize: bool,
    exposure: f32,
    light_intensity: f32,
    // M2: Base color override (for energy tests)
    base_color: [f32; 3],
    // M4: Disney Principled BRDF extensions
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen: f32,
    sheen_tint: f32,
    specular_tint: f32,
    // M1: Debug dot products flag
    debug_dot_products: bool,
    // M2: Additional debug toggles and controls
    debug_lambert_only: bool,
    debug_diffuse_only: bool,
    debug_d: bool,
    debug_spec_no_nl: bool,
    debug_energy: bool,
    debug_angle_sweep: bool,
    debug_angle_component: u32, // 0=spec,1=diffuse,2=combined
    debug_no_srgb: bool,
    output_mode: u32, // 0=linear, 1=srgb
    metallic_override: f32,
    wi3_debug_mode: u32,
    wi3_debug_roughness: f32,
) -> Result<Vec<u8>> {
    // M0.1: Log GPU adapter info once per run (use static flag to log only once)
    use std::sync::Once;
    static LOG_GPU_INFO: Once = Once::new();
    LOG_GPU_INFO.call_once(|| {
        // Get adapter info from global GPU context
        let gpu_ctx = crate::gpu::ctx();
        let adapter_info = gpu_ctx.adapter.get_info();
        log::info!("[M0] GPU Adapter: {} ({})", adapter_info.name, adapter_info.backend.to_str());
        log::info!("[M0] Device Type: {:?}", adapter_info.device_type);
    });
    
    ensure!(width > 0 && height > 0, "tile dimensions must be positive");
    ensure!(width <= 4096 && height <= 4096, "tile dimensions must be <= 4096 to avoid GPU timeouts");
    
    let roughness = roughness.clamp(0.0, 1.0);
    let _exposure = exposure.max(1e-6);
    let light_intensity = light_intensity.max(1e-6);  // Milestone 4: Clamp light intensity
    let wi3_mode = wi3_debug_mode;
    let wi3_roughness = if wi3_mode != 0 {
        wi3_debug_roughness.clamp(0.0, 1.0)
    } else {
        roughness
    };

    // M0: Validate BRDF model index (restrict to baseline set {0,1,4,6})
    ensure!(matches!(model_u32, 0 | 1 | 4 | 6),
        "invalid BRDF model index: {}. Allowed: 0(Lambert),1(Phong),4(GGX),6(Disney)",
        model_u32);

    // M0: Log model and flags for each render
    log::info!(
        "[M0] BRDF Tile Render: model={} roughness={:.3} size={}x{} flags: ndf_only={} g_only={} dfg_only={} spec_only={} r_vis={} exposure={:.3} light={:.3} base=({:.2},{:.2},{:.2})",
        model_u32, roughness, width, height, ndf_only, g_only, dfg_only, spec_only, roughness_visualize, _exposure, light_intensity, base_color[0], base_color[1], base_color[2]
    );

    // Create render target texture (linear target; shader applies sRGB if requested)
    let render_target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen.brdf_tile.render_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Create depth buffer for PBR mesh pass
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen.brdf_tile.depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let render_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // P7-02: Generate UV-sphere mesh with normals and tangents
    let (vertices, indices) = crate::offscreen::sphere::generate_uv_sphere(64, 32, 1.0);
    
    // Create vertex buffer
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen.brdf_tile.vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Create index buffer
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen.brdf_tile.index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let num_indices = indices.len() as u32;

    // P7-03: Wire PBR pipeline with BRDF model, roughness, ndf_only, and exposure
    let pipeline = crate::offscreen::pipeline::BrdfTilePipeline::new(device)?;

    // M1: Camera V = normalize(0, 0, 1). Use z=2.0 and FOV 60° so sphere fits with high coverage.
    let camera_pos = [0.0f32, 0.0, 2.0];
    let look_at = [0.0f32, 0.0, 0.0];
    let up = [0.0f32, 1.0, 0.0];
    
    let view_matrix = create_look_at_matrix(camera_pos, look_at, up);
    let aspect = width as f32 / height as f32;
    let projection_matrix = create_perspective_matrix(60.0_f32.to_radians(), aspect, 0.1, 100.0);
    let model_matrix = identity_matrix();

    // Create uniforms
    let uniforms = crate::offscreen::pipeline::Uniforms {
        model_matrix,
        view_matrix,
        projection_matrix,
    };
    
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen.brdf_tile.uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Compute metallic override and resulting F0 mix for debug views
    let metallic = metallic_override.clamp(0.0, 1.0);
    let dielectric_f0 = [0.04f32, 0.04, 0.04];
    let f0 = [
        dielectric_f0[0] * (1.0 - metallic) + base_color[0] * metallic,
        dielectric_f0[1] * (1.0 - metallic) + base_color[1] * metallic,
        dielectric_f0[2] * (1.0 - metallic) + base_color[2] * metallic,
    ];

    // Set up lighting (M1): directional L = normalize(0.5, 0.5, 1)
    // Shader uses light_dir as a direction that is negated in the fragment: light_dir = normalize(-params.light_dir)
    // Therefore, set params.light_dir = -L.
    let light_dir = [-0.5, -0.5, -1.0];
    let params = crate::offscreen::pipeline::BrdfTileParams {
        light_dir,
        _pad0: 0.0,
        light_color: [1.0, 1.0, 1.0],
        light_intensity,  // Milestone 4: Use parameter instead of hardcoded 0.8
        camera_pos,
        _pad1: 0.0,
        // M1/M2: Base color override
        base_color,
        metallic,
        roughness,
        ndf_only: if ndf_only { 1 } else { 0 },
        g_only: if g_only { 1 } else { 0 },
        dfg_only: if dfg_only { 1 } else { 0 },
        spec_only: if spec_only { 1 } else { 0 },
        roughness_visualize: if roughness_visualize { 1 } else { 0 },
        // M0/M3: Provide explicit F0 (mix dielectric 0.04 with base color by metallic)
        f0,
        _pad_f0: 0.0,
        // M4: Disney Principled BRDF extensions
        clearcoat: clearcoat.clamp(0.0, 1.0),
        clearcoat_roughness: clearcoat_roughness.clamp(0.0, 1.0),
        sheen: sheen.clamp(0.0, 1.0),
        sheen_tint: sheen_tint.clamp(0.0, 1.0),
        specular_tint: specular_tint.clamp(0.0, 1.0),
        // M2: Debug toggles
        debug_lambert_only: if debug_lambert_only { 1 } else { 0 },
        debug_diffuse_only: if debug_diffuse_only { 1 } else { 0 },
        debug_energy: if debug_energy { 1 } else { 0 },
        debug_d: if debug_d { 1 } else { 0 },
        debug_g_dbg: 0, // reserved (we use g_only above for the primary G)
        debug_spec_no_nl: if debug_spec_no_nl { 1 } else { 0 },
        debug_angle_sweep: if debug_angle_sweep { 1 } else { 0 },
        debug_angle_component: debug_angle_component,
        debug_no_srgb: if debug_no_srgb { 1 } else { 0 },
        // padding to keep total struct size stable
        _pad2: 0,
        _pad3: 0,
        _pad4: 0,
        _pad5: 0,
        _pad6: 0,
        _pad7: 0,
    };
    
    // Pad the uniform buffer to the WGSL-expected size to avoid size drift issues.
    let params_bytes = bytemuck::bytes_of(&params);
    const PARAMS_MIN_SIZE: usize = 256; // Overprovision to 256 bytes to avoid WGSL/Rust drift
    let target_len = params_bytes.len().max(PARAMS_MIN_SIZE);
    let mut padded = vec![0u8; target_len];
    padded[..params_bytes.len()].copy_from_slice(params_bytes);

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen.brdf_tile.params"),
        contents: &padded,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shading parameters for BRDF dispatch
    let shading = crate::offscreen::pipeline::ShadingParamsGPU {
        brdf: model_u32,
        metallic,
        roughness, // α = roughness^2 computed in shader
        sheen: 0.0,
        clearcoat: 0.0,
        subsurface: 0.0,
        anisotropy: 0.0,
        exposure: _exposure,
        output_mode,
        _pad_out0: 0,
        _pad_out1: 0,
        _pad_out2: 0,
    };
    
    let shading_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen.brdf_tile.shading"),
        contents: bytemuck::cast_slice(&[shading]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let debug_push = crate::offscreen::pipeline::DebugPush {
        mode: wi3_mode,
        roughness: wi3_roughness,
        _pad: [0.0, 0.0],
    };
    let debug_push_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen.brdf_tile.debug_push"),
        contents: bytemuck::bytes_of(&debug_push),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // M1: Create debug buffer for min/max N·L, N·V tracking
    // Initialize with sentinel values in fixed-point u32 space:
    // min set to u32::MAX (so atomicMin will decrease), max set to 0 (so atomicMax will increase)
    let debug_init: [u32; 4] = [
        u32::MAX,  // min_nl
        0,         // max_nl
        u32::MAX,  // min_nv
        0,         // max_nv
    ];
    let debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen.brdf_tile.debug_buffer"),
        contents: bytemuck::cast_slice(&debug_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Create bind group
    let bind_group = pipeline.create_bind_group(
        device,
        &uniforms_buffer,
        &params_buffer,
        &shading_buffer,
        &debug_buffer,
        &debug_push_buffer,
    );

    // Render the sphere
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen.brdf_tile.encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen.brdf_tile.render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        render_pass.set_pipeline(pipeline.pipeline());
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..num_indices, 0, 0..1);
    }

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Read back RGBA8 pixels
    let buffer = crate::renderer::readback::read_texture_tight(
        device,
        queue,
        &render_target,
        (width, height),
        wgpu::TextureFormat::Rgba8Unorm,
    )?;

    // M1: Read back debug buffer and log min/max N·L, N·V if enabled
    if debug_dot_products {
        // Create staging buffer for readback
        let debug_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen.brdf_tile.debug_staging"),
            size: 16, // 4 x u32
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen.brdf_tile.debug_readback"),
        });
        encoder.copy_buffer_to_buffer(&debug_buffer, 0, &debug_staging, 0, 16);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        
        let slice = debug_staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        
        if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let data = slice.get_mapped_range();
            let values: &[u32; 4] = bytemuck::from_bytes(&data[..16]);
            // Convert fixed-point back to [0,1]
            let denom = 4294967295.0_f32;
            let min_nl = (values[0] as f32) / denom;
            let max_nl = (values[1] as f32) / denom;
            let min_nv = (values[2] as f32) / denom;
            let max_nv = (values[3] as f32) / denom;
            
            log::info!("[M1] Debug Dot Products:");
            log::info!("  N·L range: [{:.4}, {:.4}]", min_nl, max_nl);
            log::info!("  N·V range: [{:.4}, {:.4}]", min_nv, max_nv);
            
            drop(data);
            debug_staging.unmap();
        }
    }
    
    // Verify buffer size matches expected tight layout (H x W x 4 bytes)
    let expected_size = (height * width * 4) as usize;
    ensure!(
        buffer.len() == expected_size,
        "readback size mismatch: got {} bytes, expected {} for {}x{} RGBA8",
        buffer.len(),
        expected_size,
        width,
        height
    );

    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brdf_tile_scaffold_returns_tight_buffer() {
        // Skip if no GPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }));
        
        let Some(adapter) = adapter else {
            eprintln!("No GPU adapter available, skipping test");
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )).expect("Failed to create device");

        // Test GGX at roughness 0.5
        let result = render_brdf_tile_offscreen(
            &device,
            &queue,
            4,    // GGX
            0.5,  // roughness
            256,  // width
            256,  // height
            false, // ndf_only
            false, // g_only
            false, // dfg_only
            false, // spec_only
            false, // roughness_visualize
            1.0,  // exposure
            0.8,  // light_intensity
            [0.5, 0.5, 0.5], // base_color
            0.0,  // clearcoat
            0.0,  // clearcoat_roughness
            0.0,  // sheen
            0.0,  // sheen_tint
            0.0,  // specular_tint
            false, // debug_dot_products
            // M2 defaults
            false, // debug_lambert_only
            false, // debug_diffuse_only
            false, // debug_d
            false, // debug_spec_no_nl
            false, // debug_energy
            false, // debug_angle_sweep
            2,     // debug_angle_component
            false, // debug_no_srgb
            1,     // output_mode (srgb)
            0.0,   // metallic_override
            0,     // wi3_debug_mode
            0.5,   // wi3_debug_roughness
        );

        assert!(result.is_ok(), "render_brdf_tile_offscreen failed: {:?}", result.err());
        let buffer = result.unwrap();
        
        // Verify tight buffer shape: (H, W, 4)
        assert_eq!(buffer.len(), 256 * 256 * 4, "buffer size mismatch");
        
        // Scaffold encodes model/roughness in clear color; verify it's non-zero
        assert!(buffer.iter().any(|&b| b > 0), "buffer is all zeros");
    }

    #[test]
    fn test_brdf_tile_validates_inputs() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
        let Some(adapter) = adapter else { return; };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("Failed to create device");

        // Invalid BRDF model
        let result = render_brdf_tile_offscreen(
            &device,
            &queue,
            99,
            0.5,
            256,
            256,
            false,
            false,
            false,
            false,
            false,
            1.0,
            0.8,
            [0.5, 0.5, 0.5],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            false, // debug_dot_products
            false, // debug_lambert_only
            false, // debug_diffuse_only
            false, // debug_d
            false, // debug_spec_no_nl
            false, // debug_energy
            false, // debug_angle_sweep
            2,     // debug_angle_component
            false, // debug_no_srgb
            1,     // output_mode
            0.0,
            0,
            0.5,
        );
        assert!(result.is_err(), "should reject invalid BRDF model");

        // Zero dimensions
        let result = render_brdf_tile_offscreen(
            &device,
            &queue,
            4,
            0.5,
            0,
            256,
            false,
            false,
            false,
            false,
            false,
            1.0,
            0.8,
            [0.5, 0.5, 0.5],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            false, // debug_dot_products
            false, // debug_lambert_only
            false, // debug_diffuse_only
            false, // debug_d
            false, // debug_spec_no_nl
            false, // debug_energy
            false, // debug_angle_sweep
            2,     // debug_angle_component
            false, // debug_no_srgb
            1,     // output_mode
            0.0,
            0,
            0.5,
        );
        assert!(result.is_err(), "should reject zero width");
    }

    /// P7-04: CPU readback validation
    /// This test demonstrates that the returned buffer is suitable for PNG export
    #[test]
    fn test_brdf_tile_png_compatible_output() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }));
        
        let Some(adapter) = adapter else {
            eprintln!("No GPU adapter available, skipping PNG compatibility test");
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )).expect("Failed to create device");

        // Render GGX at roughness 0.3
        let width = 128u32;
        let height = 128u32;
        let result = render_brdf_tile_offscreen(
            &device,
            &queue,
            4,     // GGX
            0.3,   // roughness
            width,
            height,
            false, // ndf_only
            false, // g_only
            false, // dfg_only
            false, // spec_only
            false, // roughness_visualize
            1.0,   // exposure
            0.8,   // light_intensity
            [0.5, 0.5, 0.5], // base_color
            0.0,   // clearcoat
            0.0,   // clearcoat_roughness
            0.0,   // sheen
            0.0,   // sheen_tint
            0.0,   // specular_tint
            false, // debug_dot_products
            false, // debug_lambert_only
            false, // debug_diffuse_only
            false, // debug_d
            false, // debug_spec_no_nl
            false, // debug_energy
            false, // debug_angle_sweep
            2,     // debug_angle_component
            false, // debug_no_srgb
            1,     // output_mode
            0.0,   // metallic_override
            0,     // wi3_debug_mode
            0.3,   // wi3_debug_roughness
        );

        assert!(result.is_ok(), "render_brdf_tile_offscreen failed: {:?}", result.err());
        let buffer = result.unwrap();
        
        // P7-04 Exit Criteria: Verify buffer is PNG-compatible
        // 1. Tight layout: H × W × 4 bytes (no padding)
        assert_eq!(buffer.len(), (height * width * 4) as usize, "buffer must be tight row-major RGBA8");
        
        // 2. Non-trivial content: should have some variation (not all black or all white)
        let unique_pixels = buffer.chunks_exact(4)
            .map(|rgba| (rgba[0], rgba[1], rgba[2], rgba[3]))
            .collect::<std::collections::HashSet<_>>();
        assert!(unique_pixels.len() > 10, "rendered output should have variation, got {} unique pixels", unique_pixels.len());
        
        // 3. Valid RGBA range: all bytes in [0, 255] (automatically true for u8)
        // 4. Alpha channel: should be 255 (opaque)
        let all_opaque = buffer.chunks_exact(4).all(|rgba| rgba[3] == 255);
        assert!(all_opaque, "all pixels should be opaque (alpha=255)");
        
        // 5. Some pixels should be non-black (sphere should be lit)
        let has_lighting = buffer.chunks_exact(4).any(|rgba| rgba[0] > 10 || rgba[1] > 10 || rgba[2] > 10);
        assert!(has_lighting, "rendered sphere should have visible lighting");
        
        // Buffer is now suitable for PNG export via python/forge3d/__init__.py:numpy_to_png()
        // In Python: numpy_to_png("output.png", buffer.reshape(height, width, 4))
    }
}
