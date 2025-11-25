// examples/gi_showcase.rs
// P5.8 GI Showcase: minimal interactive scene with AO/SSGI/SSR enabled
// and an on-screen performance HUD driven by the main viewer.
//
// Usage:
//   cargo run --release --example gi_showcase -- [GI FLAGS...]
//
// GI-related CLI flags are parsed via GiCliConfig and translated into
// viewer commands (e.g. ":gi ssao on", ":ssgi-steps 16"). If no
// GI flags are provided, this example enables SSAO, SSGI, and SSR with
// reasonable defaults so you immediately see their effect and timings.
//
// The viewer HUD renders GI performance numbers per pass when GI is
// active; see docs/p5_gi_passes.md for details.

use forge3d::cli::args::GiCliConfig;
use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};
use std::env;

fn gi_cli_config_to_commands(cfg: &GiCliConfig) -> Vec<String> {
    cfg.to_commands()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable logging for GI timing and viewer events.
    let _ = env_logger::try_init();

    // Collect all CLI arguments (excluding argv[0]) so we can parse
    // GI-related flags via the central schema in src/cli/args.rs.
    let all_args: Vec<String> = env::args().skip(1).collect();

    let gi_cfg = match GiCliConfig::parse(&all_args) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("[forge3d gi_showcase] error parsing GI flags: {e}");
            std::process::exit(1);
        }
    };

    // Seed initial commands from parsed GI configuration.
    let mut cmds: Vec<String> = gi_cli_config_to_commands(&gi_cfg);

    // If the user did not provide any explicit GI toggles, enable all
    // three passes so the showcase starts with a fully-stacked GI path.
    if !cmds.iter().any(|c| c.starts_with(":gi ")) {
        cmds.push(":gi ssao on".to_string());
        cmds.push(":gi ssgi on".to_string());
        cmds.push(":gi ssr on".to_string());
    }

    // SSAO/GTAO defaults (only apply if not already specified via CLI).
    if !cmds.iter().any(|c| c.starts_with(":ssao-radius")) {
        cmds.push(":ssao-radius 0.5".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssao-intensity")) {
        cmds.push(":ssao-intensity 1.0".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssao-technique")) {
        // Default to GTAO for crisper horizon-based occlusion.
        cmds.push(":ssao-technique gtao".to_string());
    }

    // SSGI defaults.
    if !cmds.iter().any(|c| c.starts_with(":ssgi-steps")) {
        cmds.push(":ssgi-steps 16".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssgi-radius")) {
        cmds.push(":ssgi-radius 1.0".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssgi-temporal-alpha")) {
        cmds.push(":ssgi-temporal-alpha 0.15".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssgi-temporal")) {
        cmds.push(":ssgi-temporal on".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssgi-half")) {
        cmds.push(":ssgi-half on".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssgi-edges")) {
        cmds.push(":ssgi-edges on".to_string());
    }

    // SSR defaults.
    if !cmds.iter().any(|c| c.starts_with(":ssr-max-steps")) {
        cmds.push(":ssr-max-steps 32".to_string());
    }
    if !cmds.iter().any(|c| c.starts_with(":ssr-thickness")) {
        cmds.push(":ssr-thickness 0.2".to_string());
    }

    // Default visualization: lit final image so GI affects the lighting
    // buffer. Users can switch to GI debug view with ":viz gi composite".
    if !cmds.iter().any(|c| c.starts_with(":viz ")) {
        cmds.push(":viz lit".to_string());
    }

    // Provide a default IBL unless the user already set one via viewer
    // commands or script. This matches the quick-start P5 examples.
    if !cmds.iter().any(|c| c.starts_with(":ibl ")) {
        cmds.push(":ibl assets/snow_field_4k.hdr".to_string());
    }

    // Optional render size override via env (e.g., P58_SIZE="1280x720").
    if let Ok(dim) = std::env::var("P58_SIZE") {
        if let Some((w, h)) = dim.split_once('x') {
            if let (Ok(wi), Ok(hi)) = (w.parse::<u32>(), h.parse::<u32>()) {
                cmds.push(format!(":size {} {}", wi, hi));
            }
        }
    }

    if !cmds.is_empty() {
        set_initial_commands(cmds);
    }

    let config = ViewerConfig {
        width: 1920,
        height: 1080,
        title: "forge3d P5.8 GI Showcase".to_string(),
        vsync: true,
        fov_deg: 60.0,
        znear: 0.1,
        zfar: 1000.0,
    };

    run_viewer(config)
}
