    //   --lit-sun <float>
    //   --lit-ibl <float>
// examples/interactive_viewer.rs
// Workstream I1: Interactive Viewer demonstration
// Simple example showing the windowed viewer with orbit and FPS camera modes

use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Map simple CLI flags to viewer commands
    // Supported:
    //   --gi <ssao:on|ssao:off|ssgi:on|ssgi:off|ssr:on|ssr:off>  (repeatable)
    //   --snapshot <path>
    //   --obj <path>
    //   --gltf <path>
    //   --ibl <path.hdr|path.exr>
    //   --viz <material|normal|depth|gi|lit>
    //   --ssao-radius <float>
    //   --ssao-intensity <float>
    //   --ssao-composite <on|off>
    //   --ssao-mul <0..1>
    //   --ssgi-steps <u32>
    //   --ssgi-radius <float>
    //   --ssgi-half <on|off>
    //   --ssgi-temporal-alpha <float>
    //   --ssr-max-steps <u32>
    //   --ssr-thickness <float>
    //   --ssao-technique <ssao|gtao>
    //   --size <WxH>
    //   --fov <degrees>
    //   --cam-lookat ex,ey,ez,tx,ty,tz[,ux,uy,uz]
    let mut args = env::args().skip(1);
    let mut cmds: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--size" => {
                if let Some(dim) = args.next() {
                    if let Some((w, h)) = dim.split_once('x') {
                        if let (Ok(wi), Ok(hi)) = (w.parse::<u32>(), h.parse::<u32>()) {
                            cmds.push(format!(":size {} {}", wi, hi));
                        }
                    }
                }
            }
            "--fov" => {
                if let Some(val) = args.next() {
                    cmds.push(format!(":fov {}", val));
                }
            }
            "--cam-lookat" => {
                if let Some(spec) = args.next() {
                    let parts: Vec<&str> = spec.split(',').collect();
                    if parts.len() == 6 || parts.len() == 9 {
                        let ex = parts[0]; let ey = parts[1]; let ez = parts[2];
                        let tx = parts[3]; let ty = parts[4]; let tz = parts[5];
                        if parts.len() == 9 {
                            let ux = parts[6]; let uy = parts[7]; let uz = parts[8];
                            cmds.push(format!(":cam-lookat {} {} {} {} {} {} {} {} {}", ex, ey, ez, tx, ty, tz, ux, uy, uz));
                        } else {
                            cmds.push(format!(":cam-lookat {} {} {} {} {} {}", ex, ey, ez, tx, ty, tz));
                        }
                    }
                }
            }
            "--gi" => {
                if let Some(spec) = args.next() {
                    if let Some((eff, state)) = spec.split_once(':') {
                        let eff_l = eff.to_lowercase();
                        let state_l = state.to_lowercase();
                        if ["ssao", "ssgi", "ssr"].contains(&eff_l.as_str())
                            && ["on", "off"].contains(&state_l.as_str())
                        {
                            cmds.push(format!(":gi {} {}", eff_l, state_l));
                        }
                    }
                }
            }
            "--snapshot" => {
                if let Some(path) = args.next() {
                    cmds.push(format!("snapshot {}", path));
                }
            }
            "--obj" => {
                if let Some(path) = args.next() {
                    cmds.push(format!(":obj {}", path));
                }
            }
            "--gltf" => {
                if let Some(path) = args.next() {
                    cmds.push(format!(":gltf {}", path));
                }
            }
            "--viz" => {
                if let Some(mode) = args.next() { cmds.push(format!(":viz {}", mode.to_lowercase())); }
            }
            "--lit-sun" => {
                if let Some(val) = args.next() { cmds.push(format!(":lit-sun {}", val)); }
            }
            "--lit-ibl" => {
                if let Some(val) = args.next() { cmds.push(format!(":lit-ibl {}", val)); }
            }
            "--ibl" => {
                if let Some(path) = args.next() { cmds.push(format!(":ibl {}", path)); }
            }
            "--ssao-radius" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssao-radius {}", val)); }
            }
            "--ssao-intensity" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssao-intensity {}", val)); }
            }
            "--ssao-composite" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssao-composite {}", val)); }
            }
            "--ssao-mul" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssao-mul {}", val)); }
            }
            "--ssgi-steps" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssgi-steps {}", val)); }
            }
            "--ssgi-radius" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssgi-radius {}", val)); }
            }
            "--ssgi-half" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssgi-half {}", val)); }
            }
            "--ssgi-temporal-alpha" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssgi-temporal-alpha {}", val)); }
            }
            "--ssr-max-steps" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssr-max-steps {}", val)); }
            }
            "--ssr-thickness" => {
                if let Some(val) = args.next() { cmds.push(format!(":ssr-thickness {}", val)); }
            }
            "--ssao-technique" => {
                if let Some(mode) = args.next() { cmds.push(format!(":ssao-technique {}", mode.to_lowercase())); }
            }
            _ => {}
        }
    }

    if !cmds.is_empty() {
        set_initial_commands(cmds);
    }

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
