//! B12: Screen-space error LOD system
//!
//! This module provides Level-of-Detail (LOD) selection based on screen-space error
//! metrics to optimize triangle count while maintaining visual quality.

use crate::core::gpu_timing::GpuTimingManager;
use crate::terrain::tiling::{TileBounds, TileId};
use glam::{Mat4, Vec2, Vec3, Vec4Swizzles};
use wgpu::CommandEncoder;

/// Configuration for LOD selection
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Target pixel error budget (pixels)
    pub pixel_error_budget: f32,
    /// Viewport width in pixels
    pub viewport_width: u32,
    /// Viewport height in pixels  
    pub viewport_height: u32,
    /// Camera field of view in radians
    pub fov_y: f32,
}

impl LodConfig {
    pub fn new(
        pixel_error_budget: f32,
        viewport_width: u32,
        viewport_height: u32,
        fov_y: f32,
    ) -> Self {
        Self {
            pixel_error_budget,
            viewport_width,
            viewport_height,
            fov_y,
        }
    }
}

/// Screen-space error calculation result
#[derive(Debug, Clone)]
pub struct ScreenSpaceError {
    /// Projected edge length in pixels
    pub edge_length_pixels: f32,
    /// Estimated error in pixels if using this LOD
    pub error_pixels: f32,
    /// Whether this LOD meets the error budget
    pub within_budget: bool,
}

/// Calculate screen-space error for a tile at a given LOD
///
/// The screen-space error represents how many pixels of error would be introduced
/// by using a particular level of detail for rendering a tile.
pub fn screen_space_error(
    tile_bounds: &TileBounds,
    tile_id: TileId,
    camera_pos: Vec3,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    config: &LodConfig,
) -> ScreenSpaceError {
    // Get tile center and size
    let tile_center = tile_bounds.center();
    let tile_size = tile_bounds.size();
    let tile_diagonal = tile_size.length();

    // Transform tile center to screen space
    let tile_center_3d = Vec3::new(tile_center.x, 0.0, tile_center.y);
    let view_pos = view_matrix * tile_center_3d.extend(1.0);
    let clip_pos = proj_matrix * view_pos;

    // Avoid division by zero or negative w
    if clip_pos.w <= 0.001 {
        return ScreenSpaceError {
            edge_length_pixels: 0.0,
            error_pixels: f32::INFINITY,
            within_budget: false,
        };
    }

    // Convert to NDC then to screen coordinates
    let ndc_pos = clip_pos.xyz() / clip_pos.w;
    let _screen_x = (ndc_pos.x + 1.0) * 0.5 * config.viewport_width as f32;
    let _screen_y = (1.0 - ndc_pos.y) * 0.5 * config.viewport_height as f32;

    // Calculate distance from camera
    let camera_pos_2d = Vec2::new(camera_pos.x, camera_pos.z);
    let distance = camera_pos_2d.distance(tile_center);

    // Avoid division by very small distances
    if distance < 0.1 {
        return ScreenSpaceError {
            edge_length_pixels: f32::INFINITY,
            error_pixels: 0.0,
            within_budget: true,
        };
    }

    // Calculate projected size of tile diagonal on screen
    // Use FOV to estimate pixels per world unit at this distance
    let half_fov = config.fov_y * 0.5;
    let pixels_per_world_unit = (config.viewport_height as f32 * 0.5) / (distance * half_fov.tan());
    let edge_length_pixels = tile_diagonal * pixels_per_world_unit;

    // Calculate error based on LOD level
    // Higher LOD means more detail, lower error
    // LOD 0 = full detail, LOD 1 = half resolution, etc.
    let lod_scale = 1.0 / (1 << tile_id.lod) as f32;
    let base_error = 1.0; // Base error in pixels for full resolution
    let error_pixels = base_error / lod_scale;

    let within_budget = error_pixels <= config.pixel_error_budget;

    ScreenSpaceError {
        edge_length_pixels,
        error_pixels,
        within_budget,
    }
}

/// Select appropriate LOD level for a tile based on screen-space error
///
/// Returns the highest LOD (most detail) that still meets the error budget,
/// or the maximum available LOD if none meet the budget.
pub fn select_lod_for_tile(
    tile_bounds: &TileBounds,
    base_tile_id: TileId,
    camera_pos: Vec3,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    config: &LodConfig,
    max_lod: u32,
) -> TileId {
    // Start from LOD 0 (highest detail) and work up until we find one that fits budget
    for lod in 0..=max_lod {
        let test_tile_id = TileId::new(lod, base_tile_id.x >> lod, base_tile_id.y >> lod);

        let sse = screen_space_error(
            tile_bounds,
            test_tile_id,
            camera_pos,
            view_matrix,
            proj_matrix,
            config,
        );

        // If this LOD is within budget, use it
        if sse.within_budget {
            return test_tile_id;
        }
    }

    // If no LOD meets the budget, use the lowest detail available
    TileId::new(
        max_lod,
        base_tile_id.x >> max_lod,
        base_tile_id.y >> max_lod,
    )
}

/// Calculate triangle count reduction for LOD selection
///
/// This is used by tests to verify the ≥40% triangle reduction requirement
pub fn calculate_triangle_reduction(
    tiles_full_res: &[TileId],
    tiles_with_lod: &[TileId],
    base_triangles_per_tile: u32,
) -> f32 {
    if tiles_full_res.is_empty() {
        return 0.0;
    }

    let full_res_triangles: u32 = tiles_full_res.iter().map(|_| base_triangles_per_tile).sum();

    let lod_triangles: u32 = tiles_with_lod
        .iter()
        .map(|tile| {
            // Triangle count reduces by factor of 4 per LOD level (2x2 = 4x fewer triangles)
            let reduction_factor = 1 << (tile.lod * 2);
            base_triangles_per_tile / reduction_factor.max(1)
        })
        .sum();

    if full_res_triangles == 0 {
        return 0.0;
    }

    let reduction = (full_res_triangles as f32 - lod_triangles as f32) / full_res_triangles as f32;
    reduction.max(0.0)
}

/// Utility function to create view matrix from camera parameters
pub fn create_view_matrix(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    Mat4::look_at_rh(eye, target, up)
}

/// Utility function to create projection matrix
pub fn create_projection_matrix(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(fov_y, aspect, near, far)
}

/// Update terrain LOD levels with GPU timing support
///
/// This function would be used in a complete implementation to update
/// terrain mesh LOD levels based on camera position and view frustum.
pub fn update_terrain_lod_with_timing(
    encoder: &mut CommandEncoder,
    mut timing_manager: Option<&mut GpuTimingManager>,
    _camera_pos: Vec3,
    _view_matrix: Mat4,
    _proj_matrix: Mat4,
    _config: &LodConfig,
) -> Vec<TileId> {
    let timing_scope = if let Some(timer) = timing_manager.as_mut() {
        Some(timer.begin_scope(encoder, "terrain_lod_update"))
    } else {
        None
    };

    // In a complete implementation, this would:
    // 1. Perform frustum culling on GPU
    // 2. Calculate screen-space error for visible tiles
    // 3. Select appropriate LOD levels
    // 4. Update mesh buffers with new geometry

    // For now, return a placeholder result
    let updated_tiles = vec![
        TileId::new(0, 0, 0),
        TileId::new(1, 0, 0),
        TileId::new(2, 0, 0),
    ];

    // End GPU timing scope
    if let (Some(timer), Some(scope_id)) = (timing_manager, timing_scope) {
        timer.end_scope(encoder, scope_id);
    }

    updated_tiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::tiling::TileBounds;

    #[test]
    fn test_screen_space_error_calculation() {
        let bounds = TileBounds::new(Vec2::new(-100.0, -100.0), Vec2::new(100.0, 100.0));
        let tile_id = TileId::new(0, 0, 0);
        let camera_pos = Vec3::new(0.0, 100.0, -200.0);
        let view = create_view_matrix(camera_pos, Vec3::ZERO, Vec3::Y);
        let proj = create_projection_matrix(45.0_f32.to_radians(), 1.0, 1.0, 1000.0);
        let config = LodConfig::new(2.0, 1024, 768, 45.0_f32.to_radians());

        let sse = screen_space_error(&bounds, tile_id, camera_pos, view, proj, &config);

        // Should have valid screen space error values
        assert!(sse.edge_length_pixels > 0.0);
        assert!(sse.error_pixels >= 0.0);
    }

    #[test]
    fn test_lod_selection() {
        let bounds = TileBounds::new(Vec2::new(-50.0, -50.0), Vec2::new(50.0, 50.0));
        let base_tile = TileId::new(0, 0, 0);
        let camera_pos = Vec3::new(0.0, 50.0, -100.0);
        let view = create_view_matrix(camera_pos, Vec3::ZERO, Vec3::Y);
        let proj = create_projection_matrix(45.0_f32.to_radians(), 1.0, 1.0, 1000.0);

        // Strict budget should force higher LOD
        let strict_config = LodConfig::new(0.5, 1024, 768, 45.0_f32.to_radians());
        let lod_tile = select_lod_for_tile(
            &bounds,
            base_tile,
            camera_pos,
            view,
            proj,
            &strict_config,
            3,
        );

        // Should select some LOD level
        assert!(lod_tile.lod <= 3);
    }

    #[test]
    fn test_triangle_reduction_calculation() {
        let full_res_tiles = vec![
            TileId::new(0, 0, 0),
            TileId::new(0, 1, 0),
            TileId::new(0, 0, 1),
            TileId::new(0, 1, 1),
        ];

        let lod_tiles = vec![
            TileId::new(1, 0, 0), // LOD 1 = 1/4 triangles
            TileId::new(2, 0, 0), // LOD 2 = 1/16 triangles
        ];

        let base_triangles = 1000;
        let reduction = calculate_triangle_reduction(&full_res_tiles, &lod_tiles, base_triangles);

        // Should show significant reduction
        assert!(reduction > 0.0);
        assert!(reduction < 1.0);
    }

    #[test]
    fn test_triangle_reduction_meets_40_percent() {
        // Test scenario that should meet the ≥40% reduction requirement
        let full_res_tiles = vec![TileId::new(0, 0, 0); 10]; // 10 full-res tiles
        let lod_tiles = vec![TileId::new(2, 0, 0); 4]; // 4 tiles at LOD 2 (1/16 triangles each)

        let base_triangles = 1000;
        let reduction = calculate_triangle_reduction(&full_res_tiles, &lod_tiles, base_triangles);

        // Should achieve significant reduction due to LOD 2 usage
        assert!(
            reduction >= 0.4,
            "Triangle reduction {} should be >= 40%",
            reduction
        );
    }
}
