//! P2.3: GPU LOD selection with frustum culling.
//!
//! Performs per-tile frustum culling and LOD selection on the GPU using
//! a compute shader, outputting a compact list of visible tiles.

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_buffer_init};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};

#[cfg(feature = "enable-globe")]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GpuLodEncodeError {
    #[error(transparent)]
    GlobeFrame(#[from] crate::terrain::clipmap::globe::GlobeFrameError),
    #[error("planetary LOD tile count mismatch: expected {expected}, got {actual}")]
    TileCountMismatch { expected: usize, actual: usize },
}

/// Configuration for GPU LOD selection.
#[derive(Debug, Clone)]
pub struct GpuLodConfig {
    /// Target pixel error budget.
    pub pixel_error_budget: f32,
    /// Viewport width in pixels.
    pub viewport_width: u32,
    /// Viewport height in pixels.
    pub viewport_height: u32,
    /// Camera field of view in radians.
    pub fov_y: f32,
    /// Maximum LOD level.
    pub max_lod: u32,
    /// Terrain width in world units.
    pub terrain_width: f32,
    /// Tile size in world units.
    pub tile_size: f32,
}

impl Default for GpuLodConfig {
    fn default() -> Self {
        Self {
            pixel_error_budget: 2.0,
            viewport_width: 1920,
            viewport_height: 1080,
            fov_y: std::f32::consts::FRAC_PI_4,
            max_lod: 4,
            terrain_width: 10000.0,
            tile_size: 256.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlanetLodParams {
    radius: f32,
    altitude: f32,
    camera_up: Vec3,
}

impl PlanetLodParams {
    #[cfg(feature = "enable-globe")]
    fn from_globe(
        frame: &crate::terrain::clipmap::globe::GlobeFrame,
    ) -> Result<Self, crate::terrain::clipmap::globe::GlobeFrameError> {
        let camera_anchor = frame.camera_anchor();
        let radius = frame.radius();
        let camera_up = frame
            .ecef_to_local_vector(camera_anchor.normalize())
            ?
            .as_vec3()
            .normalize();
        Ok(Self {
            radius: radius as f32,
            altitude: (camera_anchor.length() - radius).max(0.0) as f32,
            camera_up,
        })
    }
}

fn horizon_visible(
    tile_center_relative: Vec3,
    tile_angular_radius: f32,
    planet: PlanetLodParams,
) -> bool {
    if !planet.radius.is_finite()
        || planet.radius <= 0.0
        || !planet.altitude.is_finite()
        || !planet.camera_up.is_finite()
    {
        return true;
    }
    let camera_up = planet.camera_up.normalize_or_zero();
    let center_from_planet =
        tile_center_relative + camera_up * (planet.radius + planet.altitude.max(0.0));
    if camera_up == Vec3::ZERO
        || !center_from_planet.is_finite()
        || center_from_planet.length_squared() == 0.0
    {
        return true;
    }

    let tangent_cos = (planet.radius / (planet.radius + planet.altitude.max(0.0))).clamp(0.0, 1.0);
    let angular_radius = tile_angular_radius.max(0.0).min(std::f32::consts::PI);
    let expanded_horizon = (tangent_cos.acos() + angular_radius).min(std::f32::consts::PI);
    let conservative_threshold = expanded_horizon.cos();
    camera_up.dot(center_from_planet.normalize()) >= conservative_threshold
}

/// Uniform buffer for LOD selection parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LodSelectParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub frustum_planes: [[f32; 4]; 6],
    pub lod_params: [f32; 4], // pixel_error_budget, viewport_height, fov_y, max_lod
    pub terrain_params: [f32; 4], // tile_size, num_tiles, variant_count, first_instance
    pub height_params: [f32; 4], // conservative world-space min/max
    pub planet_params: [f32; 4], // radius, camera altitude, globe enabled, reserved
    pub camera_up: [f32; 4],
}

impl LodSelectParams {
    fn new(
        view_proj: Mat4,
        camera_pos: Vec3,
        frustum: &FrustumPlanes,
        config: &GpuLodConfig,
        num_tiles: u32,
        variant_count: u32,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
        planet: Option<PlanetLodParams>,
    ) -> Self {
        let planet_params = planet
            .map(|planet| [planet.radius, planet.altitude, 1.0, 0.0])
            .unwrap_or([0.0; 4]);
        let camera_up = planet
            .map(|planet| planet.camera_up.extend(0.0).to_array())
            .unwrap_or([0.0, 0.0, 1.0, 0.0]);
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            frustum_planes: frustum.to_array(),
            lod_params: [
                config.pixel_error_budget,
                config.viewport_height as f32,
                config.fov_y,
                config.max_lod as f32,
            ],
            terrain_params: [
                config.tile_size,
                num_tiles as f32,
                variant_count as f32,
                u32::from(first_instance) as f32,
            ],
            height_params: [
                height_bounds.0,
                height_bounds.1,
                u32::from(frustum_culling) as f32,
                0.0,
            ],
            planet_params,
            camera_up,
        }
    }
}

/// Tile information for GPU processing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileInfo {
    pub tile_id: u32,
    pub height_min: f32,
    pub bounds_min: [f32; 2],
    pub bounds_max: [f32; 2],
    pub distance: f32,
    pub selected_lod: u32,
    pub visible: u32,
    pub height_max: f32,
    /// Tile center after f64 camera-anchor subtraction. Flat tiles retain
    /// their historical world-space XY center with Z = 0.
    pub camera_relative_center: [f32; 3],
    /// Conservative angular half-width of a planetary tile in radians.
    pub angular_radius: f32,
}

impl TileInfo {
    pub fn new(lod: u32, x: u32, y: u32, bounds_min: Vec2, bounds_max: Vec2) -> Self {
        let center = (bounds_min + bounds_max) * 0.5;
        Self {
            tile_id: Self::pack_id(lod, x, y),
            height_min: f32::INFINITY,
            bounds_min: [bounds_min.x, bounds_min.y],
            bounds_max: [bounds_max.x, bounds_max.y],
            distance: 0.0,
            selected_lod: lod,
            visible: 1,
            height_max: f32::NEG_INFINITY,
            camera_relative_center: [center.x, center.y, 0.0],
            angular_radius: 0.0,
        }
    }

    pub fn with_height_bounds(mut self, height_min: f32, height_max: f32) -> Self {
        self.height_min = height_min;
        self.height_max = height_max;
        self
    }

    #[cfg(feature = "enable-globe")]
    pub(crate) fn with_globe_bounds(
        mut self,
        frame: &crate::terrain::clipmap::globe::GlobeFrame,
        bounds_min: Vec3,
        bounds_max: Vec3,
    ) -> Result<Self, crate::terrain::clipmap::globe::GlobeFrameError> {
        let center = (bounds_min + bounds_max) * 0.5;
        self.bounds_min = bounds_min.truncate().to_array();
        self.bounds_max = bounds_max.truncate().to_array();
        self.height_min = bounds_min.z;
        self.height_max = bounds_max.z;
        self.camera_relative_center = center.to_array();

        let anchor = frame.camera_anchor();
        let local_to_ecef = crate::terrain::clipmap::globe::GlobeFrame::tangent_to_ecef(anchor)
            .ok_or(crate::terrain::clipmap::globe::GlobeFrameError::EcefOutOfRange)?;
        let center_world = anchor + local_to_ecef.transform_vector3(center.as_dvec3());
        let center_direction = center_world.normalize();
        let mut angular_radius = 0.0_f64;
        for x in [bounds_min.x, bounds_max.x] {
            for y in [bounds_min.y, bounds_max.y] {
                for z in [bounds_min.z, bounds_max.z] {
                    let world = anchor
                        + local_to_ecef
                            .transform_vector3(glam::DVec3::new(x as f64, y as f64, z as f64));
                    angular_radius = angular_radius.max(
                        center_direction
                            .dot(world.normalize())
                            .clamp(-1.0, 1.0)
                            .acos(),
                    );
                }
            }
        }
        self.angular_radius = angular_radius.min(std::f64::consts::PI) as f32;
        Ok(self)
    }

    #[cfg(feature = "enable-globe")]
    pub fn with_globe_center(
        mut self,
        frame: &crate::terrain::clipmap::globe::GlobeFrame,
        center_ecef: glam::DVec3,
        angular_radius: f32,
    ) -> Result<Self, crate::terrain::clipmap::globe::GlobeFrameError> {
        let relative = frame.camera_relative(center_ecef)?.position;
        self.camera_relative_center = relative.to_array();
        self.angular_radius = angular_radius.max(0.0);
        Ok(self)
    }

    pub fn pack_id(lod: u32, x: u32, y: u32) -> u32 {
        (lod << 24) | ((x & 0xFFF) << 12) | (y & 0xFFF)
    }

    pub fn unpack_id(packed: u32) -> (u32, u32, u32) {
        let lod = packed >> 24;
        let x = (packed >> 12) & 0xFFF;
        let y = packed & 0xFFF;
        (lod, x, y)
    }
}

/// Output header with atomic counters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OutputHeader {
    pub visible_count: u32,
    pub total_triangles: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Per-region indexed draw metadata consumed by the LOD compute pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IndirectDrawTemplate {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub tile_id: u32,
}

/// WGPU `DrawIndexedIndirect` argument layout (20 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// GPU-written instance data for the compact indirect draw list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClipmapDrawInstance {
    pub transform: [[f32; 4]; 4],
    pub tile_id_lod: [u32; 2],
    pub _pad: [u32; 2],
}

impl ClipmapDrawInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Uint32x2];

    pub fn identity(tile_id: u32, lod: u32) -> Self {
        Self {
            transform: Mat4::IDENTITY.to_cols_array_2d(),
            tile_id_lod: [tile_id, lod],
            _pad: [0; 2],
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

mod readback;
pub use readback::GpuLodDrawResources;
use readback::SelectionReadbackRuntime;
#[cfg(test)]
use readback::{CompletedLodSelection, SelectionReadbackTicketState, SelectionReadbackTickets};
pub(crate) use readback::{LodSelectionProvenance, SelectionReadbackTicket};
/// Frustum planes for culling (Ax + By + Cz + D = 0 format).
#[derive(Debug, Clone)]
pub struct FrustumPlanes {
    pub left: Vec4,
    pub right: Vec4,
    pub bottom: Vec4,
    pub top: Vec4,
    pub near: Vec4,
    pub far: Vec4,
}

impl FrustumPlanes {
    /// Extract frustum planes from view-projection matrix.
    pub fn from_view_proj(vp: Mat4) -> Self {
        let rows = [
            Vec4::new(vp.x_axis.x, vp.y_axis.x, vp.z_axis.x, vp.w_axis.x),
            Vec4::new(vp.x_axis.y, vp.y_axis.y, vp.z_axis.y, vp.w_axis.y),
            Vec4::new(vp.x_axis.z, vp.y_axis.z, vp.z_axis.z, vp.w_axis.z),
            Vec4::new(vp.x_axis.w, vp.y_axis.w, vp.z_axis.w, vp.w_axis.w),
        ];

        Self {
            left: normalize_plane(rows[3] + rows[0]),
            right: normalize_plane(rows[3] - rows[0]),
            bottom: normalize_plane(rows[3] + rows[1]),
            top: normalize_plane(rows[3] - rows[1]),
            // WGPU/glam perspective matrices use a zero-to-one depth range.
            // The near clip inequality is therefore row2 >= 0, not the
            // OpenGL-style row3 + row2 >= 0.  The latter rejected every
            // clipmap tile on the NVIDIA Vulkan acceptance lane.
            near: normalize_plane(rows[2]),
            far: normalize_plane(rows[3] - rows[2]),
        }
    }

    pub fn to_array(&self) -> [[f32; 4]; 6] {
        [
            self.left.to_array(),
            self.right.to_array(),
            self.bottom.to_array(),
            self.top.to_array(),
            self.near.to_array(),
            self.far.to_array(),
        ]
    }
}

fn normalize_plane(plane: Vec4) -> Vec4 {
    let normal = plane.xyz();
    let normal_len = normal.length();
    if normal_len > 0.0 {
        plane / normal_len
    } else {
        plane
    }
}

/// GPU LOD selector using compute shaders.
pub struct GpuLodSelector {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    config: GpuLodConfig,
}

impl GpuLodSelector {
    /// Create a new GPU LOD selector.
    pub fn new(device: &wgpu::Device, config: GpuLodConfig) -> Self {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "clipmap_lod_select",
            include_str!("../../shaders/clipmap_lod_select.wgsl"),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lod_select_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lod_select_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("lod_select_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_main",
            },
        );

        Self {
            pipeline,
            bind_group_layout,
            config,
        }
    }

    /// Allocate the static inputs and reusable outputs for a clipmap mesh.
    pub fn create_draw_resources(
        &self,
        device: &wgpu::Device,
        tiles: &[TileInfo],
        draw_templates: &[IndirectDrawTemplate],
    ) -> RenderResult<GpuLodDrawResources> {
        let variant_count = self.config.max_lod + 1;
        if tiles.is_empty() || draw_templates.len() != tiles.len() * variant_count as usize {
            return Err(crate::core::error::RenderError::render(
                "GPU LOD selection requires every tile to have every LOD draw template",
            ));
        }

        let params = LodSelectParams::new(
            Mat4::IDENTITY,
            Vec3::ZERO,
            &FrustumPlanes::from_view_proj(Mat4::IDENTITY),
            &self.config,
            tiles.len() as u32,
            variant_count,
            false,
            (-f32::MAX, f32::MAX),
            true,
            None,
        );
        let params_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_params_buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let input_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_input_tiles"),
                contents: bytemuck::cast_slice(tiles),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let template_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_draw_templates"),
                contents: bytemuck::cast_slice(draw_templates),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;

        let output_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("lod_output_tiles"),
                size: (tiles.len() * std::mem::size_of::<TileInfo>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let indirect_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.clipmap.indirect_args"),
                size: (tiles.len() * std::mem::size_of::<DrawIndexedIndirectArgs>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let instance_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.clipmap.draw_instances"),
                size: (tiles.len() * std::mem::size_of::<ClipmapDrawInstance>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;

        let header = OutputHeader {
            visible_count: 0,
            total_triangles: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let header_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("lod_output_header"),
                contents: bytemuck::bytes_of(&header),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lod_select_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: template_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });
        let selection_readback_size = std::mem::size_of::<OutputHeader>() as u64
            + tiles.len() as u64 * std::mem::size_of::<TileInfo>() as u64;
        let selection_readbacks = [
            tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("lod_selection_readback_0"),
                    size: selection_readback_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )?,
            tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("lod_selection_readback_1"),
                    size: selection_readback_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )?,
        ];

        Ok(GpuLodDrawResources {
            indirect_buffer,
            instance_buffer,
            output_tiles: output_buffer,
            output_header: header_buffer,
            max_draw_count: tiles.len() as u32,
            variant_count,
            params: params_buffer,
            input_tiles: input_buffer,
            _draw_templates: template_buffer,
            selection_readbacks,
            selection_readback: std::sync::Mutex::new(SelectionReadbackRuntime::default()),
            bind_group,
        })
    }

    /// Cull clipmap regions and write a compact indirect draw/instance list.
    pub fn encode_indirect(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuLodDrawResources,
        view_proj: Mat4,
        camera_pos: Vec3,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
    ) {
        let _ = self.encode_indirect_in_space(
            queue,
            encoder,
            resources,
            None,
            view_proj,
            camera_pos,
            first_instance,
            height_bounds,
            frustum_culling,
            None,
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_indirect_tracked(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuLodDrawResources,
        view_proj: Mat4,
        camera_pos: Vec3,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
        provenance: LodSelectionProvenance,
    ) -> Option<SelectionReadbackTicket> {
        self.encode_indirect_in_space(
            queue,
            encoder,
            resources,
            None,
            view_proj,
            camera_pos,
            first_instance,
            height_bounds,
            frustum_culling,
            None,
            Some(provenance),
        )
    }

    #[cfg(feature = "enable-globe")]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_indirect_globe(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuLodDrawResources,
        camera_relative_tiles: &[TileInfo],
        view_proj: Mat4,
        frame: &crate::terrain::clipmap::globe::GlobeFrame,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
    ) -> Result<bool, GpuLodEncodeError> {
        let expected = resources.max_draw_count as usize;
        if camera_relative_tiles.len() != expected {
            return Err(GpuLodEncodeError::TileCountMismatch {
                expected,
                actual: camera_relative_tiles.len(),
            });
        }
        let planet = PlanetLodParams::from_globe(frame)?;
        let _ = self.encode_indirect_in_space(
            queue,
            encoder,
            resources,
            Some(camera_relative_tiles),
            view_proj,
            Vec3::ZERO,
            first_instance,
            height_bounds,
            frustum_culling,
            Some(planet),
            None,
        );
        Ok(true)
    }

    #[cfg(feature = "enable-globe")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_indirect_globe_tracked(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuLodDrawResources,
        camera_relative_tiles: &[TileInfo],
        view_proj: Mat4,
        frame: &crate::terrain::clipmap::globe::GlobeFrame,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
        provenance: LodSelectionProvenance,
    ) -> Result<
        Option<SelectionReadbackTicket>,
        GpuLodEncodeError,
    > {
        let expected = resources.max_draw_count as usize;
        if camera_relative_tiles.len() != expected {
            return Err(GpuLodEncodeError::TileCountMismatch {
                expected,
                actual: camera_relative_tiles.len(),
            });
        }
        let planet = PlanetLodParams::from_globe(frame)?;
        Ok(self.encode_indirect_in_space(
            queue,
            encoder,
            resources,
            Some(camera_relative_tiles),
            view_proj,
            Vec3::ZERO,
            first_instance,
            height_bounds,
            frustum_culling,
            Some(planet),
            Some(provenance),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_indirect_in_space(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuLodDrawResources,
        camera_relative_tiles: Option<&[TileInfo]>,
        view_proj: Mat4,
        camera_pos: Vec3,
        first_instance: bool,
        height_bounds: (f32, f32),
        frustum_culling: bool,
        planet: Option<PlanetLodParams>,
        provenance: Option<LodSelectionProvenance>,
    ) -> Option<SelectionReadbackTicket> {
        let frustum = FrustumPlanes::from_view_proj(view_proj);
        let params = LodSelectParams::new(
            view_proj,
            camera_pos,
            &frustum,
            &self.config,
            resources.max_draw_count,
            resources.variant_count,
            first_instance,
            height_bounds,
            frustum_culling,
            planet,
        );
        queue.write_buffer(&resources.params, 0, bytemuck::bytes_of(&params));
        if let Some(tiles) = camera_relative_tiles {
            queue.write_buffer(&resources.input_tiles, 0, bytemuck::cast_slice(tiles));
        }
        encoder.clear_buffer(&resources.output_header, 0, None);
        encoder.clear_buffer(&resources.indirect_buffer, 0, None);
        encoder.clear_buffer(&resources.instance_buffer, 0, None);
        crate::core::shader_registry::record_shader_use("clipmap_lod_select");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lod_select_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        provenance.and_then(|provenance| resources.stage_selection_readback(encoder, provenance))
    }
}

/// Result of GPU LOD selection.
#[derive(Debug, Clone)]
pub struct LodSelectionResult {
    /// Visible tiles after frustum culling with selected LOD.
    pub visible_tiles: Vec<TileInfo>,
    /// Total triangle count across all visible tiles.
    pub total_triangles: u32,
    /// Number of tiles culled.
    pub culled_count: u32,
}

impl LodSelectionResult {
    /// Calculate triangle reduction percentage.
    pub fn triangle_reduction(&self, full_res_triangles: u32) -> f32 {
        if full_res_triangles == 0 {
            return 0.0;
        }
        let reduction =
            (full_res_triangles as f32 - self.total_triangles as f32) / full_res_triangles as f32;
        (reduction * 100.0).max(0.0)
    }
}

/// Use the exact GPU-submitted tile/LOD list when it is available, falling
/// back to an independent CPU selection only for paths without that provenance.
pub(crate) fn prefer_submitted_tiles<F>(
    submitted_tiles: Option<&[TileInfo]>,
    cpu_fallback: F,
) -> Vec<TileInfo>
where
    F: FnOnce() -> Vec<TileInfo>,
{
    submitted_tiles.map_or_else(cpu_fallback, <[_]>::to_vec)
}

/// CPU fallback for LOD selection (used when GPU compute is unavailable).
pub fn cpu_lod_select(
    tiles: &[TileInfo],
    view_proj: Mat4,
    camera_pos: Vec3,
    config: &GpuLodConfig,
    height_bounds: (f32, f32),
) -> LodSelectionResult {
    cpu_lod_select_in_space(tiles, view_proj, camera_pos, config, height_bounds, None)
}

#[cfg(feature = "enable-globe")]
pub fn cpu_lod_select_globe(
    tiles: &[TileInfo],
    view_proj: Mat4,
    config: &GpuLodConfig,
    height_bounds: (f32, f32),
    frame: &crate::terrain::clipmap::globe::GlobeFrame,
) -> Result<LodSelectionResult, crate::terrain::clipmap::globe::GlobeFrameError> {
    Ok(cpu_lod_select_in_space(
        tiles,
        view_proj,
        Vec3::ZERO,
        config,
        height_bounds,
        Some(PlanetLodParams::from_globe(frame)?),
    ))
}

fn cpu_lod_select_in_space(
    tiles: &[TileInfo],
    view_proj: Mat4,
    camera_pos: Vec3,
    config: &GpuLodConfig,
    height_bounds: (f32, f32),
    planet: Option<PlanetLodParams>,
) -> LodSelectionResult {
    let frustum = FrustumPlanes::from_view_proj(view_proj);
    let camera_pos_2d = Vec2::new(camera_pos.x, camera_pos.y);

    let mut visible_tiles = Vec::new();
    let mut total_triangles = 0u32;
    let mut culled_count = 0u32;

    for tile in tiles {
        let bounds_min = Vec2::from(tile.bounds_min);
        let bounds_max = Vec2::from(tile.bounds_max);
        let center = (bounds_min + bounds_max) * 0.5;

        let (height_min, height_max) = if tile.height_min <= tile.height_max {
            (tile.height_min, tile.height_max)
        } else {
            height_bounds
        };
        let center_relative = Vec3::from(tile.camera_relative_center);
        let (frustum_min, frustum_max, frustum_height_min, frustum_height_max) =
            if planet.is_some() {
                let half_xy = (bounds_max - bounds_min).abs() * 0.5;
                let half_z = ((height_max - height_min).abs() * 0.5).max(0.0);
                (
                    center_relative.truncate() - half_xy,
                    center_relative.truncate() + half_xy,
                    center_relative.z - half_z,
                    center_relative.z + half_z,
                )
            } else {
                (bounds_min, bounds_max, height_min, height_max)
            };
        let visible = frustum_test_aabb(
            frustum_min,
            frustum_max,
            frustum_height_min,
            frustum_height_max,
            &frustum,
        )
            && planet.map_or(true, |planet| {
                horizon_visible(center_relative, tile.angular_radius, planet)
            });

        if !visible {
            culled_count += 1;
            continue;
        }

        // Calculate distance and select LOD
        let distance = if planet.is_some() {
            center_relative.length()
        } else {
            camera_pos_2d.distance(center)
        };
        let selected_lod = select_lod_cpu(distance, config);

        let mut selected_tile = *tile;
        selected_tile.distance = distance;
        selected_tile.selected_lod = selected_lod;
        selected_tile.visible = 1;

        // Calculate triangle count for this LOD
        let base_triangles = 128 * 128 * 2;
        let reduction = 1u32 << (selected_lod * 2);
        total_triangles += base_triangles / reduction.max(1);

        visible_tiles.push(selected_tile);
    }

    LodSelectionResult {
        visible_tiles,
        total_triangles,
        culled_count,
    }
}

fn frustum_test_aabb(
    bounds_min: Vec2,
    bounds_max: Vec2,
    height_min: f32,
    height_max: f32,
    frustum: &FrustumPlanes,
) -> bool {
    let planes = [
        frustum.left,
        frustum.right,
        frustum.bottom,
        frustum.top,
        frustum.near,
        frustum.far,
    ];

    for plane in planes {
        let positive = Vec3::new(
            if plane.x >= 0.0 {
                bounds_max.x
            } else {
                bounds_min.x
            },
            if plane.y >= 0.0 {
                bounds_max.y
            } else {
                bounds_min.y
            },
            if plane.z >= 0.0 {
                height_max
            } else {
                height_min
            },
        );
        if plane.xyz().dot(positive) + plane.w < 0.0 {
            return false;
        }
    }
    true
}

fn select_lod_cpu(distance: f32, config: &GpuLodConfig) -> u32 {
    let safe_distance = distance.max(0.1);
    let half_fov = config.fov_y * 0.5;
    let pixels_per_unit = (config.viewport_height as f32 * 0.5) / (safe_distance * half_fov.tan());

    for lod in (0..=config.max_lod).rev() {
        let error = config.tile_size * (1 << lod) as f32 * pixels_per_unit;
        if error <= config.pixel_error_budget {
            return lod;
        }
    }

    0
}

#[cfg(test)]
mod tests;
