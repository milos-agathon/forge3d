use glam::Vec3;
use super::types::{GroundPlaneParams, GroundPlaneMode};

/// Create preset configurations
pub struct GroundPlanePresets;

impl GroundPlanePresets {
    pub fn create_engineering_grid() -> GroundPlaneParams {
        GroundPlaneParams {
            mode: GroundPlaneMode::Grid,
            major_spacing: 10.0,
            minor_spacing: 1.0,
            major_grid_color: Vec3::new(0.0, 0.8, 0.0), // Green major lines
            minor_grid_color: Vec3::new(0.0, 0.4, 0.0), // Dark green minor lines
            albedo: Vec3::new(0.05, 0.05, 0.05),        // Nearly black background
            ..Default::default()
        }
    }

    pub fn create_architectural_grid() -> GroundPlaneParams {
        GroundPlaneParams {
            mode: GroundPlaneMode::Grid,
            major_spacing: 5.0, // 5 meter grid for architecture
            minor_spacing: 1.0, // 1 meter subdivisions
            major_grid_color: Vec3::new(0.6, 0.6, 0.8), // Light blue major lines
            minor_grid_color: Vec3::new(0.4, 0.4, 0.6), // Darker blue minor lines
            albedo: Vec3::new(0.9, 0.9, 0.95), // Light background
            major_width: 1.5,
            minor_width: 0.8,
            ..Default::default()
        }
    }

    pub fn create_simple_ground() -> GroundPlaneParams {
        GroundPlaneParams {
            mode: GroundPlaneMode::Solid,
            albedo: Vec3::new(0.3, 0.25, 0.2), // Brown earth color
            alpha: 1.0,
            ..Default::default()
        }
    }
}
