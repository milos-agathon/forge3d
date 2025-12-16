// src/viewer/render/sky.rs
// Sky rendering helpers for the interactive viewer

use glam::Mat4;
use wgpu::{Buffer, CommandEncoder, Device, Queue};

use super::super::util;
use super::super::viewer_types::SkyUniforms;

/// Sky rendering state needed for the compute pass
pub struct SkyRenderState<'a> {
    pub sky_params: &'a Buffer,
    pub sky_camera: &'a Buffer,
    pub sky_output_view: &'a wgpu::TextureView,
    pub sky_bind_group_layout0: &'a wgpu::BindGroupLayout,
    pub sky_bind_group_layout1: &'a wgpu::BindGroupLayout,
    pub sky_pipeline: &'a wgpu::ComputePipeline,
}

/// Camera state for sky rendering
pub struct SkyCameraState {
    pub view_matrix: Mat4,
    pub proj_matrix: Mat4,
    pub eye: glam::Vec3,
}

/// Sky parameters for the frame
pub struct SkyParams {
    pub model_id: u32,
    pub turbidity: f32,
    pub ground_albedo: f32,
    pub exposure: f32,
    pub sun_intensity: f32,
}

/// Update sky camera uniforms
pub fn update_sky_camera(queue: &Queue, sky_camera: &Buffer, camera: &SkyCameraState) {
    let inv_view = camera.view_matrix.inverse();
    let inv_proj = camera.proj_matrix.inverse();
    let cam_buf: [[[f32; 4]; 4]; 4] = [
        util::mat4_to_arr4(camera.view_matrix),
        util::mat4_to_arr4(camera.proj_matrix),
        util::mat4_to_arr4(inv_view),
        util::mat4_to_arr4(inv_proj),
    ];
    queue.write_buffer(sky_camera, 0, bytemuck::cast_slice(&cam_buf));

    let eye4: [f32; 4] = [camera.eye.x, camera.eye.y, camera.eye.z, 0.0];
    let base = (std::mem::size_of::<[[f32; 4]; 4]>() * 4) as u64;
    queue.write_buffer(sky_camera, base, bytemuck::cast_slice(&eye4));
}

/// Update sky parameters uniform
pub fn update_sky_params(
    queue: &Queue,
    sky_params_buf: &Buffer,
    params: &SkyParams,
    inv_view: Mat4,
) {
    let sun_dir_vs = glam::Vec3::new(0.3, 0.6, -1.0).normalize();
    let sun_dir_ws = (inv_view * glam::Vec4::new(sun_dir_vs.x, sun_dir_vs.y, sun_dir_vs.z, 0.0))
        .truncate()
        .normalize();

    let sky_uniforms = SkyUniforms {
        sun_direction: [sun_dir_ws.x, sun_dir_ws.y, sun_dir_ws.z],
        turbidity: params.turbidity.clamp(1.0, 10.0),
        ground_albedo: params.ground_albedo.clamp(0.0, 1.0),
        model: params.model_id,
        sun_intensity: params.sun_intensity.max(0.0),
        exposure: params.exposure.max(0.0),
        _pad: [0.0; 4],
    };
    queue.write_buffer(sky_params_buf, 0, bytemuck::bytes_of(&sky_uniforms));
}

/// Dispatch sky compute pass
pub fn dispatch_sky_compute(
    device: &Device,
    encoder: &mut CommandEncoder,
    state: &SkyRenderState,
    width: u32,
    height: u32,
) {
    let sky_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewer.sky.bg0"),
        layout: state.sky_bind_group_layout0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.sky_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(state.sky_output_view),
            },
        ],
    });
    let sky_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewer.sky.bg1"),
        layout: state.sky_bind_group_layout1,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: state.sky_camera.as_entire_binding(),
        }],
    });

    let gx = (width + 7) / 8;
    let gy = (height + 7) / 8;
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("viewer.sky.compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(state.sky_pipeline);
        cpass.set_bind_group(0, &sky_bg0, &[]);
        cpass.set_bind_group(1, &sky_bg1, &[]);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
}
