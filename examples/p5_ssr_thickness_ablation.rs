// examples/p5_ssr_thickness_ablation.rs
// Launches the viewer to capture the SSR thickness ablation comparison.
// Usage:
//   cargo run --release --example p5_ssr_thickness_ablation
// Optional env: P53_SIZE="1920x1080" to override viewer size.

use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut cmds: Vec<String> = vec![
        ":viz lit".to_string(),
        ":gi ssr on".to_string(),
        ":ssr-max-steps 64".to_string(),
        ":ssr-thickness 0.08".to_string(),
        ":load-ssr-preset".to_string(),
        ":p5 ssr-thickness".to_string(),
    ];

    if let Ok(dim) = std::env::var("P53_SIZE") {
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
        title: "forge3d P5.3 SSR Thickness Ablation".to_string(),
        vsync: true,
        fov_deg: 55.0,
        znear: 0.1,
        zfar: 1000.0,
    };

    run_viewer(config)
}
