// examples/p5_ssgi_generate.rs
// P5.2 SSGI Artifact Generator (viewer-driven)
// Launches the interactive forge3d viewer once and queues capture commands to generate:
//   - reports/p5/p5_ssgi_cornell.png
//   - reports/p5/p5_ssgi_temporal_compare.png
//   - reports/p5/p5_meta.json (merged/updated with SSGI metrics)
// This preserves the single-terminal workflow and reuses the existing viewer capture
// implementations that write PNGs and metadata asynchronously on the first frames.
//
// Usage:
//   cargo run --release --example p5_ssgi_generate
// Optional env: P52_SIZE="1280x720" to override default viewer window size.

use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Queue initial commands for the viewer to generate P5.2 artifacts
    // Enable SSGI path and configure parameters to match acceptance spec
    let mut cmds: Vec<String> = vec![
        ":viz material".to_string(),
        ":gi ssgi on".to_string(),
        ":ssgi-steps 24".to_string(),
        ":ssgi-radius 1.0".to_string(),
        ":ssgi-temporal-alpha 0.15".to_string(),
        ":ssgi-temporal on".to_string(),
        ":ssgi-half off".to_string(),
        ":ssgi-edges on".to_string(),
        ":ssgi-upsample-sigma-depth 0.02".to_string(),
        ":ssgi-upsample-sigma-normal 0.25".to_string(),
        // Generate the SSGI deliverables
        ":p5 ssgi-cornell".to_string(),
        ":p5 ssgi-temporal".to_string(),
    ];

    // Optional size override via env (useful for CI)
    if let Ok(dim) = std::env::var("P52_SIZE") {
        if let Some((w, h)) = dim.split_once('x') {
            if let (Ok(wi), Ok(hi)) = (w.parse::<u32>(), h.parse::<u32>()) {
                cmds.push(format!(":size {} {}", wi, hi));
            }
        }
    }
    set_initial_commands(cmds);

    // Viewer config (window can be closed after files are written)
    let config = ViewerConfig {
        width: 1920,
        height: 1080,
        title: "forge3d P5.2 SSGI Export".to_string(),
        vsync: true,
        fov_deg: 60.0,
        znear: 0.1,
        zfar: 1000.0,
    };

    // Run viewer; captures will be processed on first frames.
    // Close the window to exit, or type :quit in the terminal.
    run_viewer(config)
}
