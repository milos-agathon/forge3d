//   --lit-sun <float>
//   --lit-ibl <float>
// examples/interactive_viewer.rs
// Workstream I1: Interactive Viewer demonstration
// Simple example showing the windowed viewer with orbit and FPS camera modes
//
use forge3d::cli::args::GiCliConfig;
use forge3d::viewer::{run_viewer, set_initial_commands, ViewerConfig};
use std::env;

fn gi_cli_config_to_commands(cfg: &GiCliConfig) -> Vec<String> {
    cfg.to_commands()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Map simple CLI flags to viewer commands
    // Supported:
    //   --gi <ssao:on|ssao:off|ssgi:on|ssgi:off|ssr:on|ssr:off>  (repeatable)
    //   --snapshot <path>
    //   --obj <path>
    //   --gltf <path>
    //   --ibl <path.hdr|path.exr>
    //   --viz <material|normal|depth|gi|lit>
    //   --brdf <lambert|phong|ggx|disney>
    //   --sky <off|on|preetham|hosek-wilkie>
    //   --sky-turbidity <float>
    //   --sky-ground <float>
    //   --sky-exposure <float>
    //   --sky-sun <float>
    //   --fog <on|off>
    //   --fog-density <float>
    //   --fog-g <float>
    //   --fog-steps <u32>
    //   --fog-shadow <on|off>
    //   --fog-temporal <float 0..0.9>
    //   --ssao-radius <float>
    //   --ssao-intensity <float>
    //   --ssao-composite <on|off>
    //   --ssao-mul <0..1>
    //   --ssao-bias <float>
    //   --ao-blur <on|off>
    //   --ao-temporal-alpha <0..1>
    //   --ssgi-steps <u32>
    //   --ssgi-radius <float>
    //   --ssgi-half <on|off>
    //   --ssgi-temporal-alpha <float>
    //   --ssr-enable <on|off>
    //   --ssr-max-steps <u32>
    //   --ssr-thickness <float>
    //   --ssao-technique <ssao|gtao>
    //   --ssao-samples <u32>
    //   --ssao-directions <u32>
    //   --ssao-temporal-alpha <0..1>
    //   --size <WxH>
    //   --fov <degrees>
    //   --cam-lookat ex,ey,ez,tx,ty,tz[,ux,uy,uz]

    // Collect all CLI arguments (excluding argv[0]) so we can validate
    // GI-related flags using the central schema in src/cli/args.rs.
    let all_args: Vec<String> = env::args().skip(1).collect();

    let gi_cfg = match GiCliConfig::parse(&all_args) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("[forge3d CLI] error parsing GI flags: {e}");
            std::process::exit(1);
        }
    };

    // Seed initial commands with GI configuration derived from GiCliConfig.
    let mut cmds: Vec<String> = gi_cli_config_to_commands(&gi_cfg);

    let mut args = all_args.into_iter();
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
                        let ex = parts[0];
                        let ey = parts[1];
                        let ez = parts[2];
                        let tx = parts[3];
                        let ty = parts[4];
                        let tz = parts[5];
                        if parts.len() == 9 {
                            let ux = parts[6];
                            let uy = parts[7];
                            let uz = parts[8];
                            cmds.push(format!(
                                ":cam-lookat {} {} {} {} {} {} {} {} {}",
                                ex, ey, ez, tx, ty, tz, ux, uy, uz
                            ));
                        } else {
                            cmds.push(format!(
                                ":cam-lookat {} {} {} {} {} {}",
                                ex, ey, ez, tx, ty, tz
                            ));
                        }
                    }
                }
            }
            "--gi" => {
                // GI flags are fully handled by GiCliConfig; skip here.
                let _ = args.next();
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
                if let Some(mode) = args.next() {
                    cmds.push(format!(":viz {}", mode.to_lowercase()));
                }
            }
            "--brdf" => {
                if let Some(model) = args.next() {
                    let m = model.to_lowercase();
                    if [
                        "lambert",
                        "lam",
                        "phong",
                        "ggx",
                        "disney",
                        "disney-principled",
                        "principled",
                    ]
                    .contains(&m.as_str())
                    {
                        cmds.push(format!(":brdf {}", m));
                    }
                }
            }
            "--lit-sun" => {
                if let Some(val) = args.next() {
                    cmds.push(format!(":lit-sun {}", val));
                }
            }
            "--lit-ibl" => {
                if let Some(val) = args.next() {
                    cmds.push(format!(":lit-ibl {}", val));
                }
            }
            "--ibl" => {
                if let Some(path) = args.next() {
                    cmds.push(format!(":ibl {}", path));
                }
            }
            "--sky" => {
                if let Some(mode) = args.next() {
                    let m = mode.to_lowercase();
                    cmds.push(format!(":sky {}", m));
                }
            }
            "--sky-turbidity" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":sky-turbidity {}", v));
                }
            }
            "--sky-ground" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":sky-ground {}", v));
                }
            }
            "--sky-exposure" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":sky-exposure {}", v));
                }
            }
            "--sky-sun" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":sky-sun {}", v));
                }
            }
            "--fog" => {
                if let Some(arg) = args.next() {
                    let on = matches!(arg.as_str(), "on" | "1" | "true");
                    cmds.push(format!(":fog {}", if on { "on" } else { "off" }));
                }
            }
            "--fog-density" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":fog-density {}", v));
                }
            }
            "--fog-g" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":fog-g {}", v));
                }
            }
            "--fog-steps" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":fog-steps {}", v));
                }
            }
            "--fog-shadow" => {
                if let Some(arg) = args.next() {
                    let on = matches!(arg.as_str(), "on" | "1" | "true");
                    cmds.push(format!(":fog-shadow {}", if on { "on" } else { "off" }));
                }
            }
            "--fog-temporal" => {
                if let Some(v) = args.next() {
                    cmds.push(format!(":fog-temporal {}", v));
                }
            }
            // GI parameter flags are handled via GiCliConfig; skip here.
            "--ssao-radius"
            | "--ssao-intensity"
            | "--ssao-bias"
            | "--ssao-samples"
            | "--ssao-directions"
            | "--ssao-composite"
            | "--ssao-mul"
            | "--ssgi-steps"
            | "--ssgi-radius"
            | "--ssgi-half"
            | "--ssgi-temporal-alpha"
            | "--ssgi-temporal-enable"
            | "--ssgi-edges"
            | "--ssgi-upsigma-depth"
            | "--ssgi-upsample-sigma-depth"
            | "--ssgi-upsigma-normal"
            | "--ssgi-upsample-sigma-normal"
            | "--ssr-enable"
            | "--ssr-max-steps"
            | "--ssr-thickness"
            | "--ao-blur"
            | "--ao-temporal-alpha"
            | "--ssao-temporal-alpha"
            | "--ssao-technique" => {
                let _ = args.next();
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
