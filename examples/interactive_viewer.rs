// examples/interactive_viewer.rs
// Workstream I1: Interactive Viewer demonstration
// Simple example showing the windowed viewer with orbit and FPS camera modes

use forge3d::viewer::{run_viewer, ViewerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create viewer configuration
    let config = ViewerConfig {
        width: 1280,
        height: 720,
        title: "forge3d Interactive Viewer Demo".to_string(),
        vsync: true,
        fov_deg: 60.0,
        znear: 0.1,
        zfar: 1000.0,
    };

    // Run the viewer
    run_viewer(config)
}
