use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_buffer_init};
use crate::geo::refraction::{principal_radii_m, EarthModel, RefractionModel};
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug)]
pub struct ViewshedOptions {
    pub width: u32,
    pub height: u32,
    pub observer_x: f32,
    pub observer_y: f32,
    pub observer_height_m: f32,
    pub target_height_m: f32,
    pub max_distance_m: f32,
    pub observer_latitude_rad: f32,
    pub observer_longitude_rad: f32,
    pub left_unwrapped_deg: f32,
    pub top_deg: f32,
    pub longitude_step_deg: f32,
    pub latitude_step_deg: f32,
    pub geodesic_sphere_radius_m: f32,
    pub earth_model: EarthModel,
    pub refraction_model: RefractionModel,
}

#[derive(Debug)]
pub struct ViewshedOutput {
    pub visibility: Vec<bool>,
    pub curvature_drop_m: Vec<f32>,
    pub refraction_gain_m: Vec<f32>,
    pub horizon_distance_m: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewshedUniforms {
    dimensions: [u32; 4],
    observer: [f32; 4],
    metric: [f32; 4],
    physics: [f32; 4],
    geodetic: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewshedCell {
    visible: u32,
    curvature_drop_m: f32,
    refraction_gain_m: f32,
    horizon_distance_m: f32,
}

fn physics_terms(options: &ViewshedOptions) -> Result<[f32; 4], String> {
    let one_minus_k = 1.0 - options.refraction_model.k()?;
    let (inv_meridional, inv_prime_vertical) = match options.earth_model {
        EarthModel::Flat => (0.0, 0.0),
        EarthModel::Sphere { radius_m } if radius_m.is_finite() && radius_m > 0.0 => {
            (radius_m.recip(), radius_m.recip())
        }
        EarthModel::Sphere { .. } => return Err("sphere radius must be finite and positive".into()),
        EarthModel::Ellipsoid { latitude_deg } => {
            let (meridional, prime_vertical) = principal_radii_m(latitude_deg)?;
            (meridional.recip(), prime_vertical.recip())
        }
    };
    Ok([
        inv_meridional as f32,
        inv_prime_vertical as f32,
        one_minus_k as f32,
        f32::from(!matches!(options.earth_model, EarthModel::Flat)),
    ])
}

fn validate(
    heights: &[f32],
    positions_m: &[[f32; 2]],
    options: &ViewshedOptions,
) -> Result<(), String> {
    let expected = options.width as usize * options.height as usize;
    if options.width < 2
        || options.height < 2
        || heights.len() != expected
        || positions_m.len() != expected
    {
        return Err(format!(
            "DEM/position lengths do not match dimensions {}x{} (both dimensions must be at least 2)",
            options.width,
            options.height
        ));
    }
    if heights.iter().any(|height| !height.is_finite())
        || positions_m
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err(format!(
            "DEM heights and geodesic positions must be finite ({} heights)",
            heights.len(),
        ));
    }
    let finite = [
        options.observer_x,
        options.observer_y,
        options.observer_height_m,
        options.target_height_m,
        options.max_distance_m,
        options.observer_latitude_rad,
        options.observer_longitude_rad,
        options.left_unwrapped_deg,
        options.top_deg,
        options.longitude_step_deg,
        options.latitude_step_deg,
        options.geodesic_sphere_radius_m,
    ]
    .iter()
    .all(|value| value.is_finite());
    if !finite
        || options.observer_x < -0.5
        || options.observer_x > options.width as f32 - 0.5
        || options.observer_y < -0.5
        || options.observer_y > options.height as f32 - 0.5
        || options.observer_height_m < 0.0
        || options.target_height_m < 0.0
        || options.max_distance_m <= 0.0
        || options.longitude_step_deg <= 0.0
        || options.latitude_step_deg <= 0.0
        || options.geodesic_sphere_radius_m < 0.0
    {
        return Err(
            "viewshed dimensions, observer, heights, spacing, and distance are invalid".into(),
        );
    }
    physics_terms(options)?;
    Ok(())
}

pub fn compute_viewshed(
    heights: &[f32],
    positions_m: &[[f32; 2]],
    options: &ViewshedOptions,
) -> RenderResult<ViewshedOutput> {
    validate(heights, positions_m, options).map_err(RenderError::render)?;
    let physics = physics_terms(options).map_err(RenderError::render)?;
    let context = crate::core::gpu::try_ctx()?;
    let device = &context.device;
    let queue = &context.queue;
    let uniforms = ViewshedUniforms {
        dimensions: [options.width, options.height, 0, 0],
        observer: [
            options.observer_x,
            options.observer_y,
            options.observer_height_m,
            options.target_height_m,
        ],
        metric: [
            options.max_distance_m,
            options.longitude_step_deg,
            options.latitude_step_deg,
            options.geodesic_sphere_radius_m,
        ],
        physics,
        geodetic: [
            options.observer_latitude_rad,
            options.observer_longitude_rad,
            options.left_unwrapped_deg,
            options.top_deg,
        ],
    };
    let height_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.viewshed.heights"),
            contents: bytemuck::cast_slice(heights),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )?;
    let uniform_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.viewshed.uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )?;
    let position_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.viewshed.geodesic_positions"),
            contents: bytemuck::cast_slice(positions_m),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )?;
    let output_size = (heights.len() * std::mem::size_of::<ViewshedCell>()) as u64;
    let output = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("helios.viewshed.output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )?;
    let readback = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("helios.viewshed.readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("helios.viewshed.shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/terrain_viewshed.wgsl").into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("helios.viewshed.pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("helios.viewshed.bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: height_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: position_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("helios.viewshed.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("helios.viewshed.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(options.width.div_ceil(8), options.height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|error| RenderError::readback(format!("viewshed callback failed: {error}")))?
        .map_err(|error| RenderError::readback(format!("viewshed map failed: {error}")))?;
    let mapped = slice.get_mapped_range();
    let cells = bytemuck::cast_slice::<u8, ViewshedCell>(&mapped);
    if cells.iter().any(|cell| cell.visible > 1) {
        drop(mapped);
        readback.unmap();
        return Err(RenderError::render(
            "viewshed geodesic leaves the DEM footprint",
        ));
    }
    let result = ViewshedOutput {
        visibility: cells.iter().map(|cell| cell.visible != 0).collect(),
        curvature_drop_m: cells.iter().map(|cell| cell.curvature_drop_m).collect(),
        refraction_gain_m: cells.iter().map(|cell| cell.refraction_gain_m).collect(),
        horizon_distance_m: cells.iter().map(|cell| cell.horizon_distance_m).collect(),
    };
    drop(mapped);
    readback.unmap();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewshed_validation_rejects_bad_models_and_dimensions() {
        let mut options = ViewshedOptions {
            width: 2,
            height: 2,
            observer_x: 0.0,
            observer_y: 0.0,
            observer_height_m: 1.7,
            target_height_m: 0.0,
            max_distance_m: 10.0,
            observer_latitude_rad: 0.0,
            observer_longitude_rad: 0.0,
            left_unwrapped_deg: 0.0,
            top_deg: 1.0,
            longitude_step_deg: 1.0,
            latitude_step_deg: 1.0,
            geodesic_sphere_radius_m: 0.0,
            earth_model: EarthModel::Flat,
            refraction_model: RefractionModel::None,
        };
        assert!(validate(&[0.0; 4], &[[0.0; 2]; 4], &options).is_ok());
        options.width = 1;
        assert!(validate(&[0.0; 2], &[[0.0; 2]; 2], &options).is_err());
        options.width = 2;
        options.refraction_model = RefractionModel::EffectiveRadius { k: 1.0 };
        assert!(validate(&[0.0; 4], &[[0.0; 2]; 4], &options).is_err());
    }
}
