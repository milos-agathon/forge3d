// examples/p5_ao_export.rs
// P5.1: Headless SSAO/GTAO artifact exporter (viewer-driven)
// This example launches the interactive viewer and queues capture commands
// to generate required artifacts under reports/p5_1/. It follows the
// single-terminal workflow by reusing the viewer command parser.
//
// Artifacts:
//   - reports/p5_1/ao_cornell_off_on.png
//   - reports/p5_1/ao_buffers_grid.png
//   - reports/p5_1/ao_params_sweep.png
//   - reports/p5_1/p5_1_meta.json
//
// Usage:
//   cargo run --release --example p5_ao_export
//
// Note: This will open a window. The captures will be queued automatically.
// You can close the window after files are written, or type :quit in the
// terminal to exit.

use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Queue initial commands for the viewer to generate artifacts
    // Enable SSAO and set reasonable defaults
    let mut cmds: Vec<String> = vec![
        ":gi ssao on".to_string(),
        ":viz material".to_string(),
        ":ssao-radius 0.5".to_string(),
        ":ssao-intensity 1.0".to_string(),
        ":ao-blur on".to_string(),
        ":ao-temporal-alpha 0.2".to_string(),
        // Capture set
        ":p5 cornell".to_string(),
        ":p5 grid".to_string(),
        ":p5 sweep".to_string(),
    ];
    // Allow optional size override via env (useful for CI)
    let size = std::env::var("P51_SIZE").ok();
    if let Some(dim) = size {
        if let Some((w, h)) = dim.split_once('x') {
            if let (Ok(wi), Ok(hi)) = (w.parse::<u32>(), h.parse::<u32>()) {
                cmds.push(format!(":size {} {}", wi, hi));
            }
        }
    }
    set_initial_commands(cmds);

    // Viewer config
    let config = ViewerConfig {
        width: 1280,
        height: 720,
        title: "forge3d P5.1 AO Export".to_string(),
        vsync: true,
        fov_deg: 60.0,
        znear: 0.1,
        zfar: 1000.0,
    };

    // Run viewer; captures will be processed after first frames
    // Close the window when done to exit, or type :quit in terminal
    run_viewer(config)
}
