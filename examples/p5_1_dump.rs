// examples/p5_1_dump.rs
// P5.1 SSAO/GTAO Artifact Generator (viewer-driven)
// Launches the interactive viewer once and queues capture commands to generate:
//   - reports/p5/p5_ssao_cornell.png
//   - reports/p5/p5_ssao_buffers_grid.png
//   - reports/p5/p5_ssao_params_grid.png
//   - reports/p5/p5_meta.json
// This preserves the single-terminal workflow and uses the existing capture
// pipeline implemented in the viewer. No viewer code changes are required.
//
// Usage:
//   cargo run --release --example p5_1_dump
// Optional env: P51_SIZE="1280x720" to override default viewer size.

use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Queue initial commands for the viewer to generate P5.1 artifacts
    // Enable AO path, prefer GTAO (viewer uses GTAO implementation when available)
    let mut cmds: Vec<String> = vec![
        ":gi gtao on".to_string(),
        ":viz material".to_string(),
        ":ssao-radius 0.5".to_string(),
        ":ssao-intensity 1.5".to_string(), // Higher intensity for stronger AO effect
        ":ssao-mul 1.0".to_string(), // Ensure full AO composite strength
        ":ao-blur on".to_string(),
        ":ao-temporal-alpha 0.0".to_string(), // P5.1 strict: temporal disabled
        // Generate the three artifacts and metadata in sequence
        ":p5 cornell".to_string(),
        ":p5 grid".to_string(),
        ":p5 sweep".to_string(),
    ];

    // Optional size override via env (useful for CI)
    if let Ok(dim) = std::env::var("P51_SIZE") {
        if let Some((w, h)) = dim.split_once('x') {
            if let (Ok(wi), Ok(hi)) = (w.parse::<u32>(), h.parse::<u32>()) {
                cmds.push(format!(":size {} {}", wi, hi));
            }
        }
    }
    set_initial_commands(cmds);

    // Viewer config (window can be closed after files are written)
    let config = ViewerConfig {
        width: 1280,
        height: 720,
        title: "forge3d P5.1 AO Export".to_string(),
        vsync: true,
        fov_deg: 60.0,
        znear: 0.1,
        zfar: 1000.0,
    };

    // Run viewer; captures will be processed on first frames.
    // Close the window to exit, or type :quit in the terminal.
    run_viewer(config)
}
