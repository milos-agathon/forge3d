use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::GiStatus => {
            if let Some(ref gi) = viewer.gi {
                let ssao_on = gi.is_enabled(SSE::SSAO);
                let ssgi_on = gi.is_enabled(SSE::SSGI);
                let ssr_on = gi.is_enabled(SSE::SSR) && viewer.ssr_params.ssr_enable;

                let ssao = gi.ssao_settings();
                println!(
                    "GI: ssao={} radius={:.6} intensity={:.6}",
                    if ssao_on { "on" } else { "off" },
                    ssao.radius,
                    ssao.intensity
                );

                if let Some(ssgi) = gi.ssgi_settings() {
                    println!(
                        "GI: ssgi={} steps={} radius={:.6}",
                        if ssgi_on { "on" } else { "off" },
                        ssgi.num_steps,
                        ssgi.radius
                    );
                } else {
                    println!("GI: ssgi=<unavailable>");
                }

                println!(
                    "GI: ssr={} max_steps={} thickness={:.6}",
                    if ssr_on { "on" } else { "off" },
                    viewer.ssr_params.ssr_max_steps,
                    viewer.ssr_params.ssr_thickness
                );

                println!(
                    "GI: weights ao={:.6} ssgi={:.6} ssr={:.6}",
                    viewer.gi_ao_weight, viewer.gi_ssgi_weight, viewer.gi_ssr_weight
                );
            } else {
                println!("GI: <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::SetGiSeed(seed) => {
            viewer.gi_seed = Some(*seed);
            if let Some(ref mut gi) = viewer.gi {
                if let Err(e) = gi.set_gi_seed(&viewer.device, &viewer.queue, *seed) {
                    eprintln!("Failed to set GI seed {}: {}", seed, e);
                } else {
                    println!("GI seed set to {}", seed);
                }
            } else {
                eprintln!("GI manager not available");
            }
            true
        }
        ViewerCmd::QueryGiSeed => {
            if let Some(seed) = viewer.gi_seed {
                println!("gi-seed = {}", seed);
            } else {
                println!("gi-seed = <unset>");
            }
            true
        }
        ViewerCmd::GiToggle(effect, on) => {
            let eff = match *effect {
                "ssao" => SSE::SSAO,
                "ssgi" => SSE::SSGI,
                "ssr" => SSE::SSR,
                _ => return true,
            };
            if *effect == "ssr" {
                viewer.ssr_params.set_enabled(*on);
                println!(
                    "[SSR] enable={}, max_steps={}, thickness={:.3}",
                    viewer.ssr_params.ssr_enable,
                    viewer.ssr_params.ssr_max_steps,
                    viewer.ssr_params.ssr_thickness
                );
            }
            if let Some(ref mut gi) = viewer.gi {
                if *on {
                    if let Err(e) = gi.enable_effect(&viewer.device, eff) {
                        eprintln!("Failed to enable {:?}: {}", eff, e);
                    } else {
                        println!("Enabled {:?}", eff);
                    }
                } else {
                    gi.disable_effect(eff);
                    println!("Disabled {:?}", eff);
                }
            }
            if *effect == "ssr" {
                viewer.sync_ssr_params_to_gi();
            }
            true
        }
        ViewerCmd::DumpGbuffer => {
            viewer.dump_p5_requested = true;
            true
        }
        ViewerCmd::SetSsaoSamples(n) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssao_settings(&viewer.queue, |s| {
                    s.num_samples = (*n).max(1);
                });
            }
            true
        }
        ViewerCmd::SetSsaoRadius(r) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssao_settings(&viewer.queue, |s| {
                    s.radius = r.max(0.0);
                });
            }
            true
        }
        ViewerCmd::SetSsaoIntensity(v) => {
            viewer.ssao_composite_mul = v.max(0.0);
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssao_settings(&viewer.queue, |s| {
                    s.intensity = v.max(0.0);
                });
                gi.set_ssao_composite_multiplier(&viewer.queue, *v);
            }
            true
        }
        ViewerCmd::SetSsaoBias(b) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.set_ssao_bias(&viewer.queue, *b);
            }
            true
        }
        ViewerCmd::SetSsaoDirections(dirs) => {
            if let Some(ref mut gi) = viewer.gi {
                let ns = dirs.saturating_mul(4).max(8);
                gi.update_ssao_settings(&viewer.queue, |s| {
                    s.num_samples = ns;
                });
            }
            true
        }
        ViewerCmd::SetSsaoTemporalAlpha(a) | ViewerCmd::SetAoTemporalAlpha(a) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.set_ssao_temporal_alpha(&viewer.queue, *a);
            }
            true
        }
        ViewerCmd::SetSsaoTemporalEnabled(on) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.set_ssao_temporal(*on);
            }
            true
        }
        ViewerCmd::SetSsaoTechnique(tech) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssao_settings(&viewer.queue, |s| {
                    s.technique = if *tech != 0 { 1 } else { 0 };
                });
            }
            true
        }
        ViewerCmd::SetAoBlur(on) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.set_ssao_blur(*on);
            }
            viewer.ssao_blur_enabled = *on;
            true
        }
        ViewerCmd::SetSsaoComposite(on) => {
            viewer.use_ssao_composite = *on;
            true
        }
        ViewerCmd::SetSsaoCompositeMul(v) => {
            viewer.ssao_composite_mul = v.max(0.0);
            if let Some(ref mut gi) = viewer.gi {
                gi.set_ssao_composite_multiplier(&viewer.queue, *v);
            }
            true
        }
        ViewerCmd::QuerySsaoRadius => {
            if let Some(ref gi) = viewer.gi {
                let s = gi.ssao_settings();
                println!("ssao-radius = {:.6}", s.radius);
            } else {
                println!("ssao-radius = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoIntensity => {
            if let Some(ref gi) = viewer.gi {
                let s = gi.ssao_settings();
                println!("ssao-intensity = {:.6}", s.intensity);
            } else {
                println!("ssao-intensity = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoBias => {
            if let Some(ref gi) = viewer.gi {
                let s = gi.ssao_settings();
                println!("ssao-bias = {:.6}", s.bias);
            } else {
                println!("ssao-bias = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoSamples => {
            if let Some(ref gi) = viewer.gi {
                let s = gi.ssao_settings();
                println!("ssao-samples = {}", s.num_samples);
            } else {
                println!("ssao-samples = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoDirections => {
            if let Some(ref gi) = viewer.gi {
                let s = gi.ssao_settings();
                let dirs = (s.num_samples / 4).max(1);
                println!("ssao-directions = {}", dirs);
            } else {
                println!("ssao-directions = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoTemporalAlpha => {
            if let Some(ref gi) = viewer.gi {
                let a = gi.ssao_temporal_alpha();
                println!("ssao-temporal-alpha = {:.6}", a);
            } else {
                println!("ssao-temporal-alpha = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoTemporalEnabled => {
            if let Some(ref gi) = viewer.gi {
                let on = gi.ssao_temporal_enabled();
                println!("ssao-temporal = {}", if on { "on" } else { "off" });
            } else {
                println!("ssao-temporal = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoBlur => {
            if viewer.gi.is_some() {
                println!(
                    "ssao-blur = {}",
                    if viewer.ssao_blur_enabled {
                        "on"
                    } else {
                        "off"
                    }
                );
            } else {
                println!("ssao-blur = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoComposite => {
            if viewer.gi.is_some() {
                println!(
                    "ssao-composite = {}",
                    if viewer.use_ssao_composite {
                        "on"
                    } else {
                        "off"
                    }
                );
            } else {
                println!("ssao-composite = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoMul => {
            if viewer.gi.is_some() {
                println!("ssao-mul = {:.6}", viewer.ssao_composite_mul);
            } else {
                println!("ssao-mul = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsaoTechnique => {
            if let Some(ref gi) = viewer.gi {
                let s = gi.ssao_settings();
                let name = if s.technique != 0 { "gtao" } else { "ssao" };
                println!("ssao-technique = {}", name);
            } else {
                println!("ssao-technique = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiSteps => {
            if let Some(ref gi) = viewer.gi {
                if let Some(s) = gi.ssgi_settings() {
                    println!("ssgi-steps = {}", s.num_steps);
                } else {
                    println!("ssgi-steps = <unavailable>");
                }
            } else {
                println!("ssgi-steps = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiRadius => {
            if let Some(ref gi) = viewer.gi {
                if let Some(s) = gi.ssgi_settings() {
                    println!("ssgi-radius = {:.6}", s.radius);
                } else {
                    println!("ssgi-radius = <unavailable>");
                }
            } else {
                println!("ssgi-radius = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiHalf => {
            if let Some(ref gi) = viewer.gi {
                if let Some(on) = gi.ssgi_half_res() {
                    println!("ssgi-half = {}", if on { "on" } else { "off" });
                } else {
                    println!("ssgi-half = <unavailable>");
                }
            } else {
                println!("ssgi-half = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiTemporalAlpha => {
            if let Some(ref gi) = viewer.gi {
                if let Some(s) = gi.ssgi_settings() {
                    println!("ssgi-temporal-alpha = {:.6}", s.temporal_alpha);
                } else {
                    println!("ssgi-temporal-alpha = <unavailable>");
                }
            } else {
                println!("ssgi-temporal-alpha = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiTemporalEnabled => {
            if let Some(ref gi) = viewer.gi {
                if let Some(s) = gi.ssgi_settings() {
                    println!(
                        "ssgi-temporal = {}",
                        if s.temporal_enabled != 0 { "on" } else { "off" }
                    );
                } else {
                    println!("ssgi-temporal = <unavailable>");
                }
            } else {
                println!("ssgi-temporal = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiEdges => {
            if let Some(ref gi) = viewer.gi {
                if let Some(s) = gi.ssgi_settings() {
                    println!(
                        "ssgi-edges = {}",
                        if s.use_edge_aware != 0 { "on" } else { "off" }
                    );
                } else {
                    println!("ssgi-edges = <unavailable>");
                }
            } else {
                println!("ssgi-edges = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiUpsampleSigmaDepth => {
            if let Some(ref gi) = viewer.gi {
                if let Some(s) = gi.ssgi_settings() {
                    println!("ssgi-upsample-sigma-depth = {:.6}", s.upsample_depth_sigma);
                } else {
                    println!("ssgi-upsample-sigma-depth = <unavailable>");
                }
            } else {
                println!("ssgi-upsample-sigma-depth = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsgiUpsampleSigmaNormal => {
            if let Some(ref gi) = viewer.gi {
                if let Some(s) = gi.ssgi_settings() {
                    println!(
                        "ssgi-upsample-sigma-normal = {:.6}",
                        s.upsample_normal_sigma
                    );
                } else {
                    println!("ssgi-upsample-sigma-normal = <unavailable>");
                }
            } else {
                println!("ssgi-upsample-sigma-normal = <unavailable: GI manager not initialized>");
            }
            true
        }
        ViewerCmd::QuerySsrEnable => {
            println!(
                "ssr-enable = {}",
                if viewer.ssr_params.ssr_enable {
                    "on"
                } else {
                    "off"
                }
            );
            true
        }
        ViewerCmd::QuerySsrMaxSteps => {
            println!("ssr-max-steps = {}", viewer.ssr_params.ssr_max_steps);
            true
        }
        ViewerCmd::QuerySsrThickness => {
            println!("ssr-thickness = {:.6}", viewer.ssr_params.ssr_thickness);
            true
        }
        ViewerCmd::SetGiAoWeight(w) => {
            viewer.gi_ao_weight = *w;
            true
        }
        ViewerCmd::SetGiSsgiWeight(w) => {
            viewer.gi_ssgi_weight = *w;
            true
        }
        ViewerCmd::SetGiSsrWeight(w) => {
            viewer.gi_ssr_weight = *w;
            true
        }
        ViewerCmd::QueryGiAoWeight => {
            println!("ao-weight = {:.6}", viewer.gi_ao_weight);
            true
        }
        ViewerCmd::QueryGiSsgiWeight => {
            println!("ssgi-weight = {:.6}", viewer.gi_ssgi_weight);
            true
        }
        ViewerCmd::QueryGiSsrWeight => {
            println!("ssr-weight = {:.6}", viewer.gi_ssr_weight);
            true
        }
        ViewerCmd::SetSsgiSteps(n) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssgi_settings(&viewer.queue, |s| {
                    s.num_steps = *n;
                });
            }
            true
        }
        ViewerCmd::SetSsgiRadius(r) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssgi_settings(&viewer.queue, |s| {
                    s.radius = r.max(0.0);
                });
            }
            true
        }
        ViewerCmd::SetSsgiHalf(on) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.set_ssgi_half_res_with_queue(&viewer.device, &viewer.queue, *on);
            }
            true
        }
        ViewerCmd::SetSsgiTemporalAlpha(a) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssgi_settings(&viewer.queue, |s| {
                    s.temporal_alpha = a.clamp(0.0, 1.0);
                });
            }
            true
        }
        ViewerCmd::SetSsgiTemporalEnabled(on) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssgi_settings(&viewer.queue, |s| {
                    s.temporal_enabled = if *on { 1 } else { 0 };
                });
                let _ = gi.ssgi_reset_history(&viewer.device, &viewer.queue);
            }
            true
        }
        ViewerCmd::SetSsgiEdges(on) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssgi_settings(&viewer.queue, |s| {
                    s.use_edge_aware = if *on { 1 } else { 0 };
                });
            }
            true
        }
        ViewerCmd::SetSsgiUpsampleSigmaDepth(sig) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssgi_settings(&viewer.queue, |s| {
                    s.upsample_depth_sigma = sig.max(1e-4);
                });
            }
            true
        }
        ViewerCmd::SetSsgiUpsampleSigmaNormal(sig) => {
            if let Some(ref mut gi) = viewer.gi {
                gi.update_ssgi_settings(&viewer.queue, |s| {
                    s.upsample_normal_sigma = sig.max(1e-4);
                });
            }
            true
        }
        ViewerCmd::SetSsrMaxSteps(steps) => {
            viewer.ssr_params.set_max_steps(*steps);
            println!("[SSR] max steps set to {}", viewer.ssr_params.ssr_max_steps);
            viewer.sync_ssr_params_to_gi();
            true
        }
        ViewerCmd::SetSsrThickness(thickness) => {
            viewer.ssr_params.set_thickness(*thickness);
            println!(
                "[SSR] thickness set to {:.3}",
                viewer.ssr_params.ssr_thickness
            );
            viewer.sync_ssr_params_to_gi();
            true
        }
        _ => false,
    }
}
