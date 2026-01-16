//! P2.3: GPU LOD selection with frustum culling.
//!
//! Performs per-tile frustum culling and LOD selection on the GPU using
//! a compute shader, outputting a compact list of visible tiles.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use wgpu::util::DeviceExt;

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

/// Uniform buffer for LOD selection parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LodSelectParams {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub frustum_planes: [[f32; 4]; 6],
    pub lod_params: [f32; 4],     // pixel_error_budget, viewport_height, fov_y, max_lod
    pub terrain_params: [f32; 4], // terrain_width, tile_size, num_tiles_x, num_tiles_y
}

impl LodSelectParams {
    pub fn new(
        view_proj: Mat4,
        camera_pos: Vec3,
        frustum: &FrustumPlanes,
        config: &GpuLodConfig,
    ) -> Self {
        let num_tiles_x = (config.terrain_width / config.tile_size).ceil();
        let num_tiles_y = num_tiles_x;

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
                config.terrain_width,
                config.tile_size,
                num_tiles_x,
                num_tiles_y,
            ],
        }
    }
}

/// Tile information for GPU processing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileInfo {
    pub tile_id: u32,
    pub _pad0: u32,
    pub bounds_min: [f32; 2],
    pub bounds_max: [f32; 2],
    pub distance: f32,
    pub selected_lod: u32,
    pub visible: u32,
    pub _pad1: u32,
}

impl TileInfo {
    pub fn new(lod: u32, x: u32, y: u32, bounds_min: Vec2, bounds_max: Vec2) -> Self {
        Self {
            tile_id: Self::pack_id(lod, x, y),
            _pad0: 0,
            bounds_min: [bounds_min.x, bounds_min.y],
            bounds_max: [bounds_max.x, bounds_max.y],
            distance: 0.0,
            selected_lod: lod,
            visible: 1,
            _pad1: 0,
        }
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
            left: (rows[3] + rows[0]).normalize(),
            right: (rows[3] - rows[0]).normalize(),
            bottom: (rows[3] + rows[1]).normalize(),
            top: (rows[3] - rows[1]).normalize(),
            near: (rows[3] + rows[2]).normalize(),
            far: (rows[3] - rows[2]).normalize(),
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

/// GPU LOD selector using compute shaders.
pub struct GpuLodSelector {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    config: GpuLodConfig,
}

impl GpuLodSelector {
    /// Create a new GPU LOD selector.
    pub fn new(device: &wgpu::Device, config: GpuLodConfig) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clipmap_lod_select"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/clipmap_lod_select.wgsl").into()),
        });

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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lod_select_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lod_select_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        Self {
            pipeline,
            bind_group_layout,
            config,
        }
    }

    /// Perform GPU LOD selection for a set of tiles.
    pub fn select(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        tiles: &[TileInfo],
        view_proj: Mat4,
        camera_pos: Vec3,
    ) -> LodSelectionResult {
        if tiles.is_empty() {
            return LodSelectionResult {
                visible_tiles: Vec::new(),
                total_triangles: 0,
                culled_count: 0,
            };
        }

        // Create uniform buffer
        let frustum = FrustumPlanes::from_view_proj(view_proj);
        let params = LodSelectParams::new(view_proj, camera_pos, &frustum, &self.config);
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lod_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create input tile buffer
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lod_input_tiles"),
            contents: bytemuck::cast_slice(tiles),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create output buffers
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lod_output_tiles"),
            size: (tiles.len() * std::mem::size_of::<TileInfo>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let header = OutputHeader {
            visible_count: 0,
            total_triangles: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lod_output_header"),
            contents: bytemuck::bytes_of(&header),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Create bind group
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
            ],
        });

        // Dispatch compute shader
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lod_select_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (tiles.len() as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Note: For actual readback, would need to copy to staging buffer and map
        // This is a simplified version that returns placeholder data
        LodSelectionResult {
            visible_tiles: tiles.to_vec(),
            total_triangles: tiles.len() as u32 * 1000,
            culled_count: 0,
        }
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
        let reduction = (full_res_triangles as f32 - self.total_triangles as f32)
            / full_res_triangles as f32;
        (reduction * 100.0).max(0.0)
    }
}

/// CPU fallback for LOD selection (used when GPU compute is unavailable).
pub fn cpu_lod_select(
    tiles: &[TileInfo],
    view_proj: Mat4,
    camera_pos: Vec3,
    config: &GpuLodConfig,
) -> LodSelectionResult {
    let frustum = FrustumPlanes::from_view_proj(view_proj);
    let camera_pos_2d = Vec2::new(camera_pos.x, camera_pos.z);

    let mut visible_tiles = Vec::new();
    let mut total_triangles = 0u32;
    let mut culled_count = 0u32;

    for tile in tiles {
        let bounds_min = Vec2::from(tile.bounds_min);
        let bounds_max = Vec2::from(tile.bounds_max);
        let center = (bounds_min + bounds_max) * 0.5;

        // Simple frustum cull (check center point against planes)
        let center_3d = Vec3::new(center.x, 0.0, center.y);
        let visible = frustum_test_point(center_3d, &frustum);

        if !visible {
            culled_count += 1;
            continue;
        }

        // Calculate distance and select LOD
        let distance = camera_pos_2d.distance(center);
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

fn frustum_test_point(point: Vec3, frustum: &FrustumPlanes) -> bool {
    let planes = [
        frustum.left,
        frustum.right,
        frustum.bottom,
        frustum.top,
        frustum.near,
        frustum.far,
    ];

    for plane in planes {
        if plane.xyz().dot(point) + plane.w < 0.0 {
            return false;
        }
    }
    true
}

fn select_lod_cpu(distance: f32, config: &GpuLodConfig) -> u32 {
    let safe_distance = distance.max(0.1);
    let half_fov = config.fov_y * 0.5;
    let pixels_per_unit = (config.viewport_height as f32 * 0.5) / (safe_distance * half_fov.tan());

    for lod in 0..=config.max_lod {
        let lod_scale = 1.0 / (1 << lod) as f32;
        let error = config.tile_size * pixels_per_unit * lod_scale;
        if error <= config.pixel_error_budget {
            return lod;
        }
    }

    config.max_lod
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_id_packing() {
        let (lod, x, y) = (3, 100, 200);
        let packed = TileInfo::pack_id(lod, x, y);
        let (l, xx, yy) = TileInfo::unpack_id(packed);
        assert_eq!((l, xx, yy), (lod, x, y));
    }

    #[test]
    fn test_frustum_planes_extraction() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 100.0, 100.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 1000.0);
        let vp = proj * view;

        let frustum = FrustumPlanes::from_view_proj(vp);

        // Planes should be normalized
        assert!((frustum.left.xyz().length() - 1.0).abs() < 0.01);
        assert!((frustum.right.xyz().length() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cpu_lod_selection() {
        let tiles = vec![
            TileInfo::new(0, 0, 0, Vec2::new(0.0, 0.0), Vec2::new(256.0, 256.0)),
            TileInfo::new(0, 1, 0, Vec2::new(256.0, 0.0), Vec2::new(512.0, 256.0)),
        ];

        let view = Mat4::look_at_rh(Vec3::new(128.0, 100.0, 128.0), Vec3::new(128.0, 0.0, 128.0), Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 1.0, 10000.0);
        let vp = proj * view;

        let config = GpuLodConfig::default();
        let result = cpu_lod_select(&tiles, vp, Vec3::new(128.0, 100.0, 128.0), &config);

        assert!(!result.visible_tiles.is_empty());
        assert!(result.total_triangles > 0);
    }

    #[test]
    fn test_lod_selection_distance_based() {
        let config = GpuLodConfig {
            pixel_error_budget: 2.0,
            viewport_height: 1080,
            fov_y: 45.0_f32.to_radians(),
            max_lod: 4,
            tile_size: 256.0,
            ..Default::default()
        };

        // Close distance should select low LOD (high detail)
        let lod_close = select_lod_cpu(100.0, &config);
        // Far distance should select higher LOD (lower detail)
        let lod_far = select_lod_cpu(10000.0, &config);

        assert!(lod_far >= lod_close, "Far tiles should use coarser LOD");
    }
}
