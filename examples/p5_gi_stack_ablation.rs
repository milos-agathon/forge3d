// examples/p5_gi_stack_ablation.rs
// P5.4 GI Stack Ablation (viewer-driven)
//
// Launches the interactive forge3d viewer once and queues a capture command
// that renders the P5 reference lighting setup from a fixed camera and
// produces a 4-column ablation image under reports/p5/:
//   - p5_gi_stack_ablation.png
// Columns (left to right):
//   1) Baseline (no AO, no SSGI, no SSR)
//   2) +AO only
//   3) +AO + SSGI
//   4) +AO + SSGI + SSR
//
// Usage:
//   cargo run --release --example p5_gi_stack_ablation
//
// The example relies on the viewer's internal capture helper
// `capture_p54_gi_stack_ablation`, which uses the current scene, camera,
// exposure, tone mapping, and post-processing settings. All four columns are
// captured in a single viewer session so they share identical camera and post
// settings; only the GI toggles differ.

use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut cmds: Vec<String> = vec![
        // Render in lit mode so GI affects the final lighting buffer.
        ":viz lit".to_string(),
        // Load the P5.3 SSR scene preset so geometry, camera, and lighting
        // match the reference SSR spheres setup before we run the GI stack.
        ":load-ssr-preset".to_string(),
        // Queue the GI stack ablation capture. The viewer-side helper will
        // toggle AO / SSGI / SSR internally and stitch the four columns.
        ":p5 gi-stack".to_string(),
    ];

    // Optional size override via env (useful for CI)
    if let Ok(dim) = std::env::var("P54_SIZE") {
        if let Some((w, h)) = dim.split_once('x') {
            if let (Ok(wi), Ok(hi)) = (w.parse::<u32>(), h.parse::<u32>()) {
                cmds.push(format!(":size {} {}", wi, hi));
            }
        }
    }

    set_initial_commands(cmds);

    let config = ViewerConfig {
        width: 1920,
        height: 1080,
        title: "forge3d P5.4 GI Stack Ablation".to_string(),
        vsync: true,
        fov_deg: 55.0,
        znear: 0.1,
        zfar: 1000.0,
    };

    run_viewer(config)
}
