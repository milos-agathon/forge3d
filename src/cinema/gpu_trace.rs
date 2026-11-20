//! Batch query dispatch of the unchanged PROMETHEUS production terrain_trace.
use super::solver::Terrain;
use crate::core::{
    gpu::try_ctx,
    resource_tracker::{tracked_create_buffer, tracked_create_buffer_init},
};
use crate::path_tracing::hybrid_compute::terrain_heightfield::{
    EarthCurvatureUniforms, TerrainMinMaxPyramid, TerrainPtUniforms,
};
use bytemuck::{Pod, Zeroable};
use serde::Serialize;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Ray {
    pub origin_tmin: [f32; 4],
    pub direction_tmax: [f32; 4],
}
#[derive(Serialize)]
pub struct TraceReport {
    pub hits: Vec<u32>,
    pub adapter: String,
    pub backend: String,
    pub software_fallback: bool,
    pub ray_count: usize,
}

const ENTRY: &str = r#"
struct CinemaRay { origin_tmin: vec4<f32>, direction_tmax: vec4<f32> }
@group(3) @binding(8) var<storage, read_write> cinema_hits: array<u32>;
@group(3) @binding(9) var<storage, read> cinema_rays: array<CinemaRay>;
@compute @workgroup_size(64,1,1)
fn main_cinema_trace(@builtin(global_invocation_id) gid:vec3<u32>) {
    if (gid.x >= arrayLength(&cinema_rays)) { return; }
    let input=cinema_rays[gid.x];
    var ray:Ray;
    ray.origin=input.origin_tmin.xyz; ray.tmin=input.origin_tmin.w;
    ray.direction=input.direction_tmax.xyz; ray.tmax=input.direction_tmax.w;
    let hit=terrain_trace(ray,true,false);
    cinema_hits[gid.x]=select(0u,1u,hit.hit != 0u) | (hybrid_sdf_error << 1u) | (terrain_count_overflow << 2u);
}
"#;

pub fn sample_capacity() -> Result<usize, String> {
    let context = try_ctx().map_err(|e| e.to_string())?;
    let limits = context.device.limits();
    let buffer_rays = limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size))
        / std::mem::size_of::<Ray>() as u64;
    let dispatch_rays = u64::from(limits.max_compute_workgroups_per_dimension) * 64;
    Ok((buffer_rays.min(dispatch_rays) / 19) as usize)
}

pub fn trace(terrain: &Terrain, rays: &[Ray]) -> Result<TraceReport, String> {
    let context = try_ctx().map_err(|e| e.to_string())?;
    let info = context.adapter.get_info();
    if context.software_fallback || info.device_type == wgpu::DeviceType::Cpu {
        return Err("AUTOCHIRON requires a physical GPU adapter".into());
    }
    if rays.is_empty() {
        return Err("trace query must contain rays".into());
    }
    let device = &context.device;
    let queue = &context.queue;
    if rays.iter().any(|r| {
        !r.origin_tmin
            .iter()
            .chain(&r.direction_tmax)
            .all(|v| v.is_finite())
            || r.origin_tmin[3] < 0.0
            || r.direction_tmax[3] < r.origin_tmin[3]
            || r.direction_tmax[..3].iter().all(|v| *v == 0.0)
    }) {
        return Err(
            "trace rays must have finite positions, nonzero directions and ordered distances"
                .into(),
        );
    }
    let texture_limit = device.limits().max_texture_dimension_2d;
    if terrain.w > texture_limit as usize
        || terrain.h > texture_limit as usize
        || terrain.pyramid.dims[0].0 > texture_limit
        || terrain.pyramid.dims[0].1 > texture_limit
    {
        return Err("heightfield exceeds negotiated texture dimensions".into());
    }
    if rays.len().div_ceil(64) > device.limits().max_compute_workgroups_per_dimension as usize {
        return Err("trace batch exceeds negotiated dispatch limit".into());
    }
    let pyramid = TerrainMinMaxPyramid::from_heightfield(
        device,
        queue,
        &terrain.heights,
        terrain.w as u32,
        terrain.h as u32,
    )
    .map_err(|e| e.to_string())?;
    let uniform = TerrainPtUniforms {
        origin_spacing: [
            0.0,
            0.0,
            (terrain.width / (terrain.w - 1) as f64) as f32,
            (terrain.width / (terrain.h - 1) as f64) as f32,
        ],
        h_params: [pyramid.h_min, pyramid.h_max, 1.0, 0.0],
        albedo_pad: [0.0; 4],
        dims: [
            terrain.w as u32,
            terrain.h as u32,
            (terrain.w - 1) as u32,
            (terrain.h - 1) as u32,
        ],
        mips: [pyramid.mip_count, 1, 0, 0],
        extra: [0; 4],
    };
    if !uniform.origin_spacing.iter().all(|v| v.is_finite())
        || uniform.origin_spacing[2] < f32::MIN_POSITIVE
        || uniform.origin_spacing[3] < f32::MIN_POSITIVE
    {
        return Err("heightfield spacing is outside the GPU float32 range".into());
    }
    let curvature = EarthCurvatureUniforms {
        inv_two_r_prime: 0.0,
        _pad0: 0.0,
        ray_origin_geodetic: [0.0; 2],
        enabled: 0,
        _pad1: 0,
    };
    let buffer = |label, contents, usage| {
        tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            },
        )
        .map_err(|e| e.to_string())
    };
    let uniforms = buffer(
        "cinema.trace.terrain",
        bytemuck::bytes_of(&uniform),
        wgpu::BufferUsages::UNIFORM,
    )?;
    let earth = buffer(
        "cinema.trace.curvature",
        bytemuck::bytes_of(&curvature),
        wgpu::BufferUsages::UNIFORM,
    )?;
    let inputs = buffer(
        "cinema.trace.rays",
        bytemuck::cast_slice(rays),
        wgpu::BufferUsages::STORAGE,
    )?;
    let size = (rays.len() * std::mem::size_of::<u32>()) as u64;
    let allocation = |label, usage| {
        tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            },
        )
        .map_err(|e| e.to_string())
    };
    let output = allocation(
        "cinema.trace.hits",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;
    let readback = allocation(
        "cinema.trace.readback",
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    )?;
    let source = format!("{}\n{}", crate::shader_sources::hybrid_kernel(), ENTRY);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cinema.trace.shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        return Err(error.to_string());
    }
    let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("cinema.trace.pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main_cinema_trace",
        },
    )?;
    let empty0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cinema.trace.empty0"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[],
    });
    let empty1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cinema.trace.empty1"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[],
    });
    let height_view = pyramid
        .height_texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let minmax_view = pyramid
        .minmax_texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cinema.trace.terrain.group"),
        layout: &pipeline.get_bind_group_layout(2),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&height_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&minmax_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: earth.as_entire_binding(),
            },
        ],
    });
    let io = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cinema.trace.io"),
        layout: &pipeline.get_bind_group_layout(3),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 8,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: inputs.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cinema.trace.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cinema.trace.dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty0, &[]);
        pass.set_bind_group(1, &empty1, &[]);
        pass.set_bind_group(2, &group, &[]);
        pass.set_bind_group(3, &io, &[]);
        pass.dispatch_workgroups(rays.len().div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (send, receive) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = send.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receive
        .recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = slice.get_mapped_range();
    let hits = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    if hits.iter().any(|v| *v > 1) {
        return Err(
            "production terrain_trace reported a traversal error or counter overflow".into(),
        );
    }
    Ok(TraceReport {
        hits,
        adapter: info.name,
        backend: format!("{:?}", info.backend),
        software_fallback: context.software_fallback,
        ray_count: rays.len(),
    })
}
