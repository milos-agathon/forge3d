// examples/wavefront_instances.rs
// Demonstrates TLAS-style instancing in the Wavefront Path Tracer by calling upload_instances()
// and rendering a single frame to a PNG.
//
// This example uses minimal scene data (one sphere material, a tiny mesh BLAS, and two instances).
// It is primarily a wiring example for the API and may produce a simple image.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use image::{ImageBuffer, Rgba};
use pollster;
use wgpu::util::DeviceExt;

use forge3d::accel::cpu_bvh::{build_bvh_cpu, BuildOptions, MeshCPU};
use forge3d::path_tracing::mesh::{build_mesh_atlas, MeshAtlas, MeshBuilder};
use forge3d::path_tracing::wavefront::WavefrontScheduler;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    frame_index: u32,
    spp: u32,
    cam_origin: [f32; 3],
    cam_fov_y: f32,
    cam_right: [f32; 3],
    cam_aspect: f32,
    cam_up: [f32; 3],
    cam_exposure: f32,
    cam_forward: [f32; 3],
    seed_hi: u32,
    seed_lo: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Sphere {
    center: [f32; 3],
    radius: f32,
    albedo: [f32; 3],
    metallic: f32,
    roughness: f32,
    ior: f32,
    emissive: [f32; 3],
    ax: f32,
    ay: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AreaLightWGSL {
    position: [f32; 3],
    radius: f32,
    normal: [f32; 3],
    intensity: f32,
    color: [f32; 3],
    importance: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DirectionalLightWGSL {
    direction: [f32; 3],
    intensity: f32,
    color: [f32; 3],
    importance: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HairSegment {
    p0: [f32; 3],
    r0: f32,
    p1: [f32; 3],
    r1: f32,
    material_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // CLI flags (very simple):
    //   --restir         Enable ReSTIR passes (init/temporal/spatial)
    //   --restir-debug   Enable ReSTIR debug AOV preview in shading
    //   --restir-spatial Enable ReSTIR spatial reuse stage (default off)
    let mut restir_enabled = false;
    let mut restir_debug = false;
    let mut restir_spatial = false;
    let mut swap_materials = false; // P2: validation toggle
    let mut skinny_blas1 = false;   // P2: near-edge skinny triangles in BLAS 1
    let mut camera_jitter: f32 = 0.0; // P2: small camera jitter along X
    let mut force_blas: Option<u32> = None; // P2: force all instances to use the same BLAS
    let mut dump_aov_depth: Option<String> = None; // P2: dump raw RGBA32F AOV depth to file
    let mut dump_aov_albedo: Option<String> = None; // P2: dump raw RGBA32F AOV albedo to file
    let mut dump_aov_normal: Option<String> = None; // P2: dump raw RGBA32F AOV normal to file
    let mut dump_aov_with_header: bool = false; // P2: prefix dumps with header containing (width,height,channels)
    // P4: medium HG (single scatter) parameters
    let mut medium_enable: bool = false;
    let mut medium_g: f32 = 0.0;
    let mut medium_sigma_t: f32 = 0.0;
    let mut medium_density: f32 = 0.0;
    // P5: AO compute from AOVs
    let mut compute_ao: bool = false;
    let mut ao_samples: u32 = 16;
    let mut ao_intensity: f32 = 1.0;
    let mut ao_bias: f32 = 0.025;
    // P5: Hair demo
    let mut hair_demo: bool = false;
    let mut hair_width: f32 = 0.02;
    let mut hair_mat: u32 = 1;
    // P6: QMC/Owen + adaptive SPP in raygen
    let mut qmc_mode: u32 = 0; // 0=Halton/VDC, 1=Sobol
    let mut spp_limit: Option<u32> = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--restir" | "--restir=on" => restir_enabled = true,
            "--restir-debug" => restir_debug = true,
            "--restir-spatial" => restir_spatial = true,
            "--swap-materials" => {
                swap_materials = true;
            }
            s if s.starts_with("--camera-jitter=") => {
                if let Some(eq) = s.split_once('=') {
                    if let Ok(val) = eq.1.parse::<f32>() {
                        camera_jitter = val;
                    }
                }
            }
            "--skinny-blas1" => {
                skinny_blas1 = true;
            }
            s if s.starts_with("--force-blas=") => {
                if let Some(eq) = s.split_once('=') {
                    if let Ok(idx) = eq.1.parse::<u32>() {
                        if idx <= 1 { force_blas = Some(idx); }
                    }
                }
            }
            s if s.starts_with("--dump-aov-depth=") => {
                if let Some((_, path)) = s.split_once('=') {
                    dump_aov_depth = Some(path.to_string());
                }
            }
            s if s.starts_with("--dump-aov-albedo=") => {
                if let Some((_, path)) = s.split_once('=') {
                    dump_aov_albedo = Some(path.to_string());
                }
            }
            s if s.starts_with("--dump-aov-normal=") => {
                if let Some((_, path)) = s.split_once('=') {
                    dump_aov_normal = Some(path.to_string());
                }
            }
            "--dump-aov-with-header" => {
                dump_aov_with_header = true;
            }
            "--medium-enable" | "--medium=on" => { medium_enable = true; }
            s if s.starts_with("--medium-g=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(f) = v.parse::<f32>() { medium_g = f; } }
            }
            s if s.starts_with("--medium-sigma-t=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(f) = v.parse::<f32>() { medium_sigma_t = f; } }
            }
            s if s.starts_with("--medium-density=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(f) = v.parse::<f32>() { medium_density = f; } }
            }
            "--compute-ao" => { compute_ao = true; }
            s if s.starts_with("--ao-samples=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(u) = v.parse::<u32>() { ao_samples = u.max(1); } }
            }
            s if s.starts_with("--ao-intensity=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(f) = v.parse::<f32>() { ao_intensity = f; } }
            }
            s if s.starts_with("--ao-bias=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(f) = v.parse::<f32>() { ao_bias = f; } }
            }
            "--hair-demo" => { hair_demo = true; }
            s if s.starts_with("--hair-width=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(f) = v.parse::<f32>() { hair_width = f.max(0.0); } }
            }
            s if s.starts_with("--hair-mat=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(u) = v.parse::<u32>() { hair_mat = if u > 0 { 1 } else { 0 }; } }
            }
            s if s.starts_with("--qmc-mode=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(u) = v.parse::<u32>() { qmc_mode = if u > 0 { 1 } else { 0 }; } }
            }
            s if s.starts_with("--spp-limit=") => {
                if let Some((_, v)) = s.split_once('=') { if let Ok(u) = v.parse::<u32>() { spp_limit = Some(u); } }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --example wavefront_instances [-- --restir] [--restir-debug] [--restir-spatial] \
                     [--swap-materials] [--skinny-blas1] [--camera-jitter=<f>] [--force-blas=<0|1>] \
                     [--dump-aov-depth=<path>] [--dump-aov-albedo=<path>] [--dump-aov-normal=<path>] [--dump-aov-with-header] \
                     [--medium-enable] [--medium-g=<f>] [--medium-sigma-t=<f>] [--medium-density=<f>] \
                     [--compute-ao] [--ao-samples=<u>] [--ao-intensity=<f>] [--ao-bias=<f>] [--hair-demo] [--hair-width=<f>] [--hair-mat=<0|1>] \
                     [--qmc-mode=<0|1>] [--spp-limit=<u>]\n\
                     Defaults: ReSTIR disabled; spatial disabled; swap-materials off; skinny-blas1 off; jitter 0; no force-blas"
                );
                return Ok(());
            }
            _ => {}
        }
    }

    // Create wgpu device/queue
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .ok_or("No suitable GPU adapter")?;
    // Request the adapter's supported limits to enable enough storage buffers per stage
    let adapter_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: adapter_limits,
            label: Some("forge3d-wavefront-instance-device"),
        },
        None,
    ))?;

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Image size and camera
    let width: u32 = 512;
    let height: u32 = 512;
    let aspect = width as f32 / height as f32;
    let mut cam_origin = Vec3::new(0.0, 1.5, 4.0);
    let mut cam_target = Vec3::new(0.0, 0.6, 0.0);
    if camera_jitter != 0.0 {
        cam_origin.x += camera_jitter;
        cam_target.x += camera_jitter;
    }
    let cam_forward = (cam_target - cam_origin).normalize();
    let world_up = Vec3::Y;
    let cam_right = cam_forward.cross(world_up).normalize();
    let cam_up = cam_right.cross(cam_forward).normalize();
    let cam_fov_y = 45.0_f32.to_radians();
    let cam_exposure = 1.0_f32;

    // Uniforms buffer
    let uniforms = Uniforms {
        width,
        height,
        frame_index: 0,
        spp: 1,
        cam_origin: cam_origin.to_array(),
        cam_fov_y,
        cam_right: cam_right.to_array(),
        cam_aspect: aspect,
        cam_up: cam_up.to_array(),
        cam_exposure,
        cam_forward: cam_forward.to_array(),
        seed_hi: 1337,
        seed_lo: 42,
        _pad: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Two sphere materials to demonstrate per-instance material_id selection
    let mat0 = Sphere {
        center: [0.0, 0.6, 0.0],
        radius: 0.5,
        albedo: [0.8, 0.2, 0.2], // red-ish
        metallic: 0.0,
        roughness: 0.5,
        ior: 1.0,
        emissive: [0.0, 0.0, 0.0],
        ax: 0.2,
        ay: 0.4,
    };
    let mat1 = Sphere {
        center: [0.0, 0.6, 0.0],
        radius: 0.5,
        albedo: [0.2, 0.8, 0.2], // green-ish
        metallic: 0.0,
        roughness: 0.3,
        ior: 1.0,
        emissive: [0.0, 0.0, 0.0],
        ax: 0.2,
        ay: 0.4,
    };
    // Hair materials (append to sphere material table): 2=dark, 3=blond
    let hair_dark = Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 0.0,
        albedo: [0.30, 0.20, 0.12],   // dark brown base color
        metallic: 0.1,                // slight tinting of spec for demo
        roughness: 0.35,
        ior: 1.55,                    // hair-like IOR (not used in hair branch yet)
        emissive: [0.0, 0.0, 0.0],
        ax: 0.2,
        ay: 0.3,
    };
    let hair_blond = Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 0.0,
        albedo: [0.90, 0.80, 0.50],   // warm blond
        metallic: 0.1,
        roughness: 0.28,
        ior: 1.55,
        emissive: [0.0, 0.0, 0.0],
        ax: 0.15,
        ay: 0.25,
    };
    let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene_spheres"),
        contents: bytemuck::cast_slice(&[mat0, mat1, hair_dark, hair_blond]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Build two BLAS entries and pack them into a mesh atlas (multi-BLAS)
    let mesh0 = MeshBuilder::quad();
    let bvh0 = build_bvh_cpu(&mesh0, &BuildOptions::default())?;
    let mesh1 = if skinny_blas1 {
        // Create a very skinny quad (near-degenerate along X) to stress near-edge intersections
        let v: Vec<[f32; 3]> = vec![
            [-0.01, -1.0, 0.0],
            [ 0.01, -1.0, 0.0],
            [ 0.01,  1.0, 0.0],
            [-0.01,  1.0, 0.0],
        ];
        let idx: Vec<[u32; 3]> = vec![ [0,1,2], [0,2,3] ];
        MeshCPU::new(v, idx)
    } else {
        MeshBuilder::quad()
    };
    let bvh1 = build_bvh_cpu(&mesh1, &BuildOptions::default())?;
    let atlas = build_mesh_atlas(device.as_ref(), &[(mesh0, bvh0), (mesh1, bvh1)])?;
    // Move atlas buffers out so we can hand descs to the scheduler and bind V/I/BVH
    let MeshAtlas { vertex_buffer, index_buffer, bvh_buffer, descs_buffer, .. } = atlas;

    // Instances: create two transforms placing the quad at different positions
    let t_left = Mat4::from_translation(Vec3::new(-0.8, 0.3, 0.0)) * Mat4::from_scale(Vec3::splat(0.75));
    let t_right = Mat4::from_translation(Vec3::new(0.9, 0.7, -0.2)) * Mat4::from_scale(Vec3::splat(0.5));

    // Area and directional lights (disabled by intensity=0 to keep image stable)
    let area_l = AreaLightWGSL {
        position: [0.0, 3.0, 0.0],
        radius: 0.5,
        normal: [0.0, -1.0, 0.0],
        intensity: 0.0,
        color: [1.0, 1.0, 1.0],
        importance: 1.0,
    };
    let dir_l = DirectionalLightWGSL {
        direction: [0.0, -1.0, 0.0],
        intensity: 0.0,
        color: [1.0, 1.0, 1.0],
        importance: 1.0,
    };
    let area_lights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("area_lights"),
        contents: bytemuck::cast_slice(&[area_l]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let directional_lights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("directional_lights"),
        contents: bytemuck::cast_slice(&[dir_l]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Per-object importance (one entry corresponding to our single material)
    let object_importance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("object_importance"),
        contents: bytemuck::cast_slice(&[1.0_f32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Accumulation HDR buffer (vec4<f32>) and bind group
    let accum_size = (width as usize * height as usize * std::mem::size_of::<[f32; 4]>()) as u64;
    let accum_hdr = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accum-hdr"),
        size: accum_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Wavefront scheduler
    let mut sched = WavefrontScheduler::new(device.clone(), queue.clone(), width, height)?;
    // Initialize minimal scene BG for ReSTIR spatial (lights + G-buffers)
    sched.init_restir_scene_spatial_bind_group(&area_lights_buf, &directional_lights_buf)?;
    // Apply medium parameters
    if medium_enable || medium_sigma_t > 0.0 {
        sched.set_medium_params(medium_g, medium_sigma_t, medium_density, medium_enable);
    }
    // Apply CLI-controlled toggles for ReSTIR
    if restir_enabled {
        sched.set_restir_enabled(true);
    }
    if restir_debug {
        sched.set_restir_debug_aov_mode(true);
    }
    if restir_spatial {
        sched.set_restir_spatial_enabled(true);
    }
    // Apply QMC/Owen mode and adaptive SPP clamp for raygen (P6)
    sched.set_qmc_mode(qmc_mode);
    if let Some(limit) = spp_limit { sched.set_adaptive_spp_limit(limit); } else { sched.set_adaptive_spp_limit(0); }
    // Provide BLAS descriptor table to scheduler (binding 15)
    sched.set_blas_descs_buffer(descs_buffer);
    // Optional: hair segments demo (upload before creating scene bind group so binding 20 points to correct buffer)
    if hair_demo {
        // Create a simple arc of segments above the origin using material 0
        let mut segs: Vec<HairSegment> = Vec::new();
        let base_y = 0.5f32;
        let radius = hair_width.max(0.0);
        let count = 12;
        for i in 0..count {
            let t0 = i as f32 / count as f32;
            let t1 = (i + 1) as f32 / count as f32;
            // Quadratic bezier-like arc in XZ plane
            let x0 = -0.6 + 1.2 * t0;
            let z0 = 0.4 * (1.0 - (2.0 * t0 - 1.0).abs());
            let x1 = -0.6 + 1.2 * t1;
            let z1 = 0.4 * (1.0 - (2.0 * t1 - 1.0).abs());
            segs.push(HairSegment {
                p0: [x0, base_y + 0.2 * z0, 0.1 * z0],
                r0: radius,
                p1: [x1, base_y + 0.2 * z1, 0.1 * z1],
                r1: radius,
                material_id: (2 + hair_mat.min(1)),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            });
        }
        let hair_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hair-segments"),
            contents: bytemuck::cast_slice(&segs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        sched.set_hair_segments_buffer(hair_buf);
    }
    // Upload instances (two copies) selecting different BLAS and material IDs
    // Default: instance 0 => (BLAS 0, MAT 0), instance 1 => (BLAS 1, MAT 1)
    // With --swap-materials: swap material ids to validate per-instance material indexing.
    // Determine BLAS indices per instance
    let (i0_blas, i1_blas) = match force_blas {
        Some(idx) => (idx, idx),
        None => (0, 1),
    };
    if swap_materials {
        sched.upload_instances_with_meta(&[(t_left, i0_blas, 1), (t_right, i1_blas, 0)]);
    } else {
        sched.upload_instances_with_meta(&[(t_left, i0_blas, 0), (t_right, i1_blas, 1)]);
    }

    // Create scene bind group (Group 1)
    let scene_bg = sched.create_scene_bind_group(
        &spheres_buffer,
        &vertex_buffer,
        &index_buffer,
        &bvh_buffer,
        &area_lights_buf,
        &directional_lights_buf,
        &object_importance_buf,
    )?;

    // Create accumulation bind group (Group 3)
    let accum_bg = sched.create_accum_bind_group(&accum_hdr);

    // Render one frame
    sched.render_frame_simple(&uniforms_buffer, &scene_bg, &accum_bg)?;

    // Read back HDR buffer and write a PNG (simple tonemap)
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accum-readback"),
        size: accum_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(&accum_hdr, 0, &readback, 0, accum_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    // Convert float RGBA to u8 PNG with simple exposure and gamma
    let px: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    let gamma = 1.0 / 2.2;
    // Iterate manually to avoid itertools dependency
    let mut i = 0usize;
    for y in 0..height {
        for x in 0..width {
            let c = px[i];
            i += 1;
            let r = (c[0] * cam_exposure).max(0.0);
            let g = (c[1] * cam_exposure).max(0.0);
            let b = (c[2] * cam_exposure).max(0.0);
            let r8 = (r.powf(gamma).min(1.0) * 255.0 + 0.5) as u8;
            let g8 = (g.powf(gamma).min(1.0) * 255.0 + 0.5) as u8;
            let b8 = (b.powf(gamma).min(1.0) * 255.0 + 0.5) as u8;
            img.put_pixel(x, y, Rgba([r8, g8, b8, 255]));
        }
    }
    drop(data);
    readback.unmap();

    std::fs::create_dir_all("out").ok();
    let out_path = "out/wavefront_instances.png";
    img.save(out_path)?;
    println!("Saved: {}", out_path);

    // Optional: dump AOV buffers (RGBA32F) to raw files for numeric analysis in tests
    if let Some(path) = dump_aov_depth {
        let aov_bytes = (sched.aov_pixel_count() * core::mem::size_of::<[f32;4]>()) as u64;
        let depth_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-depth-readback"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut aov_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aov-depth-readback-encoder"),
        });
        sched.copy_aov_depth_to(&mut aov_encoder, &depth_readback);
        queue.submit(Some(aov_encoder.finish()));

        let aov_slice = depth_readback.slice(..);
        aov_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let aov_data = aov_slice.get_mapped_range();
        if dump_aov_with_header {
            // Header: b"AOV0" + width(u32 LE) + height(u32 LE) + channels(u32 LE)
            let mut blob = Vec::with_capacity(16 + aov_bytes as usize);
            blob.extend_from_slice(b"AOV0");
            blob.extend_from_slice(&width.to_le_bytes());
            blob.extend_from_slice(&height.to_le_bytes());
            blob.extend_from_slice(&(4u32).to_le_bytes());
            blob.extend_from_slice(&aov_data);
            std::fs::write(&path, &blob)?;
        } else {
            // Write raw bytes to file
            std::fs::write(&path, &aov_data)?;
        }
        drop(aov_data);
        depth_readback.unmap();
        println!("Saved AOV depth RGBA32F: {} ({} bytes)", path, aov_bytes);
    }

    if let Some(path) = dump_aov_albedo {
        let aov_bytes = (sched.aov_pixel_count() * core::mem::size_of::<[f32;4]>()) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-albedo-readback"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("aov-albedo-enc") });
        sched.copy_aov_albedo_to(&mut enc, &rb);
        queue.submit(Some(enc.finish()));
        let sl = rb.slice(..);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = sl.get_mapped_range();
        if dump_aov_with_header {
            let mut blob = Vec::with_capacity(16 + aov_bytes as usize);
            blob.extend_from_slice(b"AOV0");
            blob.extend_from_slice(&width.to_le_bytes());
            blob.extend_from_slice(&height.to_le_bytes());
            blob.extend_from_slice(&(4u32).to_le_bytes());
            blob.extend_from_slice(&data);
            std::fs::write(&path, &blob)?;
        } else {
            std::fs::write(&path, &data)?;
        }
        drop(data);
        rb.unmap();
        println!("Saved AOV albedo RGBA32F: {} ({} bytes)", path, aov_bytes);
    }

    if let Some(path) = dump_aov_normal {
        let aov_bytes = (sched.aov_pixel_count() * core::mem::size_of::<[f32;4]>()) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-normal-readback"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("aov-normal-enc") });
        sched.copy_aov_normal_to(&mut enc, &rb);
        queue.submit(Some(enc.finish()));
        let sl = rb.slice(..);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = sl.get_mapped_range();
        if dump_aov_with_header {
            let mut blob = Vec::with_capacity(16 + aov_bytes as usize);
            blob.extend_from_slice(b"AOV0");
            blob.extend_from_slice(&width.to_le_bytes());
            blob.extend_from_slice(&height.to_le_bytes());
            blob.extend_from_slice(&(4u32).to_le_bytes());
            blob.extend_from_slice(&data);
            std::fs::write(&path, &blob)?;
        } else {
            std::fs::write(&path, &data)?;
        }
        drop(data);
        rb.unmap();
        println!("Saved AOV normal RGBA32F: {} ({} bytes)", path, aov_bytes);
    }

    // Optional: compute AO from AOVs and write grayscale PNG
    if compute_ao {
        let ao_bytes = (sched.aov_pixel_count() * core::mem::size_of::<[f32;4]>()) as u64;
        let ao_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ao-out"),
            size: ao_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut ao_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ao-encoder") });
        sched.dispatch_ao_from_aovs(&mut ao_encoder, ao_samples, ao_intensity, ao_bias, 1337, &ao_out)?;
        queue.submit(Some(ao_encoder.finish()));
        let sl = ao_out.slice(..);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = sl.get_mapped_range();
        let px: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let mut img_ao: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
        let mut i = 0usize;
        for y in 0..height { for x in 0..width { let v = px[i][0].clamp(0.0, 1.0); let g = (v * 255.0 + 0.5) as u8; img_ao.put_pixel(x, y, Rgba([g, g, g, 255])); i += 1; }}
        drop(data);
        ao_out.unmap();
        std::fs::create_dir_all("out").ok();
        let ao_path = "out/wavefront_ao.png";
        img_ao.save(ao_path)?;
        println!("Saved AO PNG: {}", ao_path);
    }

    Ok(())
}
