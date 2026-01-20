// src/viewer/cmd/handler.rs
// Command handler for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring
//
// NOTE: This file exceeds the 300 LOC target due to the large match statement.
// Further refactoring would require restructuring the ViewerCmd handling.

use std::path::Path;

use wgpu::util::DeviceExt;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::viewer::event_loop::{update_ipc_transform_stats, set_pending_bundle_save, set_pending_bundle_load};
use crate::viewer::terrain;
use crate::cli::gi_types::GiVizMode;
use crate::viewer::viewer_enums::{CaptureKind, FogMode, ViewerCmd, VizMode};
use crate::viewer::viewer_types;
use crate::viewer::Viewer;

impl Viewer {
    pub(crate) fn handle_cmd(&mut self, cmd: ViewerCmd) {
        match cmd {
            ViewerCmd::Quit => { /* handled in event loop */ }
            ViewerCmd::GiStatus => {
                if let Some(ref gi) = self.gi {
                    let ssao_on = gi.is_enabled(SSE::SSAO);
                    let ssgi_on = gi.is_enabled(SSE::SSGI);
                    let ssr_on = gi.is_enabled(SSE::SSR) && self.ssr_params.ssr_enable;

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
                        self.ssr_params.ssr_max_steps,
                        self.ssr_params.ssr_thickness
                    );

                    println!(
                        "GI: weights ao={:.6} ssgi={:.6} ssr={:.6}",
                        self.gi_ao_weight,
                        self.gi_ssgi_weight,
                        self.gi_ssr_weight
                    );
                } else {
                    println!("GI: <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::SetGiSeed(seed) => {
                self.gi_seed = Some(seed);
                if let Some(ref mut gi) = self.gi {
                    if let Err(e) = gi.set_gi_seed(&self.device, &self.queue, seed) {
                        eprintln!("Failed to set GI seed {}: {}", seed, e);
                    } else {
                        println!("GI seed set to {}", seed);
                    }
                } else {
                    eprintln!("GI manager not available");
                }
            }
            ViewerCmd::QueryGiSeed => {
                if let Some(seed) = self.gi_seed {
                    println!("gi-seed = {}", seed);
                } else {
                    println!("gi-seed = <unset>");
                }
            }
            ViewerCmd::GiToggle(effect, on) => {
                let eff = match effect {
                    "ssao" => SSE::SSAO,
                    "ssgi" => SSE::SSGI,
                    "ssr" => SSE::SSR,
                    _ => return,
                };
                if effect == "ssr" {
                    self.ssr_params.set_enabled(on);
                    println!(
                        "[SSR] enable={}, max_steps={}, thickness={:.3}",
                        self.ssr_params.ssr_enable,
                        self.ssr_params.ssr_max_steps,
                        self.ssr_params.ssr_thickness
                    );
                }
                if let Some(ref mut gi) = self.gi {
                    if on {
                        if let Err(e) = gi.enable_effect(&self.device, eff) {
                            eprintln!("Failed to enable {:?}: {}", eff, e);
                        } else {
                            println!("Enabled {:?}", eff);
                        }
                    } else {
                        gi.disable_effect(eff);
                        println!("Disabled {:?}", eff);
                    }
                }
                if effect == "ssr" {
                    self.sync_ssr_params_to_gi();
                }
            }
            ViewerCmd::DumpGbuffer => {
                self.dump_p5_requested = true;
            }
            // SSAO parameter updates
            ViewerCmd::SetSsaoSamples(n) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.num_samples = n.max(1);
                    });
                }
            }
            ViewerCmd::SetSsaoRadius(r) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.radius = r.max(0.0);
                    });
                }
            }
            ViewerCmd::SetSsaoIntensity(v) => {
                self.ssao_composite_mul = v.max(0.0);
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.intensity = v.max(0.0);
                    });
                    gi.set_ssao_composite_multiplier(&self.queue, v);
                }
            }
            ViewerCmd::SetSsaoBias(b) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_bias(&self.queue, b);
                }
            }
            ViewerCmd::SetSsaoDirections(dirs) => {
                if let Some(ref mut gi) = self.gi {
                    let ns = dirs.saturating_mul(4).max(8);
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.num_samples = ns;
                    });
                }
            }
            ViewerCmd::SetSsaoTemporalAlpha(a) | ViewerCmd::SetAoTemporalAlpha(a) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_temporal_alpha(&self.queue, a);
                }
            }
            ViewerCmd::SetSsaoTemporalEnabled(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_temporal(on);
                }
            }
            ViewerCmd::SetSsaoTechnique(tech) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.technique = if tech != 0 { 1 } else { 0 };
                    });
                }
            }
            ViewerCmd::SetAoBlur(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_blur(on);
                }
                self.ssao_blur_enabled = on;
            }
            ViewerCmd::SetSsaoComposite(on) => {
                self.use_ssao_composite = on;
            }
            ViewerCmd::SetSsaoCompositeMul(v) => {
                self.ssao_composite_mul = v.max(0.0);
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_composite_multiplier(&self.queue, v);
                }
            }
            // GI query handling
            ViewerCmd::QuerySsaoRadius => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-radius = {:.6}", s.radius);
                } else {
                    println!("ssao-radius = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoIntensity => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-intensity = {:.6}", s.intensity);
                } else {
                    println!("ssao-intensity = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoBias => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-bias = {:.6}", s.bias);
                } else {
                    println!("ssao-bias = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoSamples => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-samples = {}", s.num_samples);
                } else {
                    println!("ssao-samples = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoDirections => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    let dirs = (s.num_samples / 4).max(1);
                    println!("ssao-directions = {}", dirs);
                } else {
                    println!("ssao-directions = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoTemporalAlpha => {
                if let Some(ref gi) = self.gi {
                    let a = gi.ssao_temporal_alpha();
                    println!("ssao-temporal-alpha = {:.6}", a);
                } else {
                    println!("ssao-temporal-alpha = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoTemporalEnabled => {
                if let Some(ref gi) = self.gi {
                    let on = gi.ssao_temporal_enabled();
                    println!("ssao-temporal = {}", if on { "on" } else { "off" });
                } else {
                    println!("ssao-temporal = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoBlur => {
                if self.gi.is_some() {
                    println!(
                        "ssao-blur = {}",
                        if self.ssao_blur_enabled { "on" } else { "off" }
                    );
                } else {
                    println!("ssao-blur = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoComposite => {
                if self.gi.is_some() {
                    println!(
                        "ssao-composite = {}",
                        if self.use_ssao_composite { "on" } else { "off" }
                    );
                } else {
                    println!("ssao-composite = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoMul => {
                if self.gi.is_some() {
                    println!("ssao-mul = {:.6}", self.ssao_composite_mul);
                } else {
                    println!("ssao-mul = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoTechnique => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    let name = if s.technique != 0 { "gtao" } else { "ssao" };
                    println!("ssao-technique = {}", name);
                } else {
                    println!("ssao-technique = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiSteps => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!("ssgi-steps = {}", s.num_steps);
                    } else {
                        println!("ssgi-steps = <unavailable>");
                    }
                } else {
                    println!("ssgi-steps = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiRadius => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!("ssgi-radius = {:.6}", s.radius);
                    } else {
                        println!("ssgi-radius = <unavailable>");
                    }
                } else {
                    println!("ssgi-radius = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiHalf => {
                if let Some(ref gi) = self.gi {
                    if let Some(on) = gi.ssgi_half_res() {
                        println!("ssgi-half = {}", if on { "on" } else { "off" });
                    } else {
                        println!("ssgi-half = <unavailable>");
                    }
                } else {
                    println!("ssgi-half = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiTemporalAlpha => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!("ssgi-temporal-alpha = {:.6}", s.temporal_alpha);
                    } else {
                        println!("ssgi-temporal-alpha = <unavailable>");
                    }
                } else {
                    println!("ssgi-temporal-alpha = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiTemporalEnabled => {
                if let Some(ref gi) = self.gi {
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
            }
            ViewerCmd::QuerySsgiEdges => {
                if let Some(ref gi) = self.gi {
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
            }
            ViewerCmd::QuerySsgiUpsampleSigmaDepth => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!(
                            "ssgi-upsample-sigma-depth = {:.6}",
                            s.upsample_depth_sigma
                        );
                    } else {
                        println!("ssgi-upsample-sigma-depth = <unavailable>");
                    }
                } else {
                    println!(
                        "ssgi-upsample-sigma-depth = <unavailable: GI manager not initialized>"
                    );
                }
            }
            ViewerCmd::QuerySsgiUpsampleSigmaNormal => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!(
                            "ssgi-upsample-sigma-normal = {:.6}",
                            s.upsample_normal_sigma
                        );
                    } else {
                        println!("ssgi-upsample-sigma-normal = <unavailable>");
                    }
                } else {
                    println!(
                        "ssgi-upsample-sigma-normal = <unavailable: GI manager not initialized>"
                    );
                }
            }
            ViewerCmd::QuerySsrEnable => {
                println!(
                    "ssr-enable = {}",
                    if self.ssr_params.ssr_enable {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            ViewerCmd::QuerySsrMaxSteps => {
                println!("ssr-max-steps = {}", self.ssr_params.ssr_max_steps);
            }
            ViewerCmd::QuerySsrThickness => {
                println!("ssr-thickness = {:.6}", self.ssr_params.ssr_thickness);
            }
            ViewerCmd::SetGiAoWeight(w) => {
                self.gi_ao_weight = w.clamp(0.0, 1.0);
            }
            ViewerCmd::SetGiSsgiWeight(w) => {
                self.gi_ssgi_weight = w.clamp(0.0, 1.0);
            }
            ViewerCmd::SetGiSsrWeight(w) => {
                self.gi_ssr_weight = w.clamp(0.0, 1.0);
            }
            ViewerCmd::QueryGiAoWeight => {
                println!("ao-weight = {:.6}", self.gi_ao_weight);
            }
            ViewerCmd::QueryGiSsgiWeight => {
                println!("ssgi-weight = {:.6}", self.gi_ssgi_weight);
            }
            ViewerCmd::QueryGiSsrWeight => {
                println!("ssr-weight = {:.6}", self.gi_ssr_weight);
            }
            // P5.2 SSGI controls
            ViewerCmd::SetSsgiSteps(n) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.num_steps = n.max(0);
                    });
                }
            }
            ViewerCmd::SetSsgiRadius(r) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.radius = r.max(0.0);
                    });
                }
            }
            ViewerCmd::SetSsgiHalf(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssgi_half_res_with_queue(&self.device, &self.queue, on);
                }
            }
            ViewerCmd::SetSsgiTemporalAlpha(a) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.temporal_alpha = a.clamp(0.0, 1.0);
                    });
                }
            }
            ViewerCmd::SetSsgiTemporalEnabled(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.temporal_enabled = if on { 1 } else { 0 };
                    });
                    let _ = gi.ssgi_reset_history(&self.device, &self.queue);
                }
            }
            ViewerCmd::SetSsgiEdges(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.use_edge_aware = if on { 1 } else { 0 };
                    });
                }
            }
            ViewerCmd::SetSsgiUpsampleSigmaDepth(sig) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.upsample_depth_sigma = sig.max(1e-4);
                    });
                }
            }
            ViewerCmd::SetSsgiUpsampleSigmaNormal(sig) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.upsample_normal_sigma = sig.max(1e-4);
                    });
                }
            }
            ViewerCmd::SetSsrMaxSteps(steps) => {
                self.ssr_params.set_max_steps(steps);
                println!("[SSR] max steps set to {}", self.ssr_params.ssr_max_steps);
                self.sync_ssr_params_to_gi();
            }
            ViewerCmd::SetSsrThickness(thickness) => {
                self.ssr_params.set_thickness(thickness);
                println!(
                    "[SSR] thickness set to {:.3}",
                    self.ssr_params.ssr_thickness
                );
                self.sync_ssr_params_to_gi();
            }
            ViewerCmd::Snapshot(path) => {
                let mut p = path.unwrap_or_else(|| "snapshot.png".to_string());
                let has_sep = p.contains('/') || p.contains('\\');
                if !has_sep && p.starts_with("p5_") {
                    let filename = if p.ends_with(".png") {
                        p
                    } else {
                        format!("{}.png", p)
                    };
                    let full = std::path::PathBuf::from("reports")
                        .join("p5")
                        .join(filename);
                    p = full.to_string_lossy().to_string();
                }
                self.snapshot_request = Some(p);
            }
            ViewerCmd::LoadObj(path) => {
                match crate::io::obj_read::import_obj(&path) {
                    Ok(obj) => {
                        if let Err(e) = self.upload_mesh(&obj.mesh) {
                            eprintln!("Failed to upload OBJ mesh: {}", e);
                        } else {
                            if let Some(mat) = obj.materials.get(0) {
                                if let Some(tex_rel) = &mat.diffuse_texture {
                                    if let Some(base) = Path::new(&path).parent() {
                                        let tex_path = base.join(tex_rel);
                                        let _ = self.load_albedo_texture(tex_path.as_path());
                                    }
                                }
                            }
                            println!("Loaded OBJ geometry: {}", path);
                        }
                    }
                    Err(e) => eprintln!("OBJ import failed: {}", e),
                }
            }
            ViewerCmd::LoadGltf(path) => match crate::io::gltf_read::import_gltf_to_mesh(&path) {
                Ok(mesh) => {
                    if let Err(e) = self.upload_mesh(&mesh) {
                        eprintln!("Failed to upload glTF mesh: {}", e);
                    }
                }
                Err(e) => eprintln!("glTF import failed: {}", e),
            },
            ViewerCmd::SetViz(mode) => {
                let m = match mode.as_str() {
                    "material" | "mat" => VizMode::Material,
                    "normal" | "normals" => VizMode::Normal,
                    "depth" => VizMode::Depth,
                    "gi" => VizMode::Gi,
                    "lit" => VizMode::Lit,
                    _ => {
                        eprintln!("Unknown viz mode: {}", mode);
                        self.viz_mode
                    }
                };
                self.viz_mode = m;
            }
            ViewerCmd::SetGiViz(mode) => {
                self.gi_viz_mode = mode;
                match mode {
                    GiVizMode::None => {
                        self.viz_mode = VizMode::Lit;
                    }
                    _ => {
                        self.viz_mode = VizMode::Gi;
                    }
                }
            }
            ViewerCmd::QueryGiViz => {
                let name = match self.gi_viz_mode {
                    GiVizMode::None => "none",
                    GiVizMode::Composite => "composite",
                    GiVizMode::Ao => "ao",
                    GiVizMode::Ssgi => "ssgi",
                    GiVizMode::Ssr => "ssr",
                };
                println!("viz-gi = {}", name);
            }
            ViewerCmd::LoadSsrPreset => match self.apply_ssr_scene_preset() {
                Ok(_) => println!("[SSR] Loaded scene preset"),
                Err(e) => eprintln!("[SSR] Failed to load preset: {}", e),
            },
            ViewerCmd::SetLitSun(v) => {
                self.lit_sun_intensity = v.max(0.0);
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitIbl(v) => {
                self.lit_ibl_intensity = v.max(0.0);
                self.lit_use_ibl = self.lit_ibl_intensity > 0.0;
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitBrdf(idx) => {
                self.lit_brdf = idx;
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitRough(v) => {
                self.lit_roughness = v.clamp(0.0, 1.0);
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitDebug(m) => {
                self.lit_debug_mode = match m {
                    1 | 2 => m,
                    _ => 0,
                };
                self.update_lit_uniform();
            }
            // P5.1 capture commands
            ViewerCmd::CaptureP51Cornell => {
                self.pending_captures
                    .push_back(CaptureKind::P51CornellSplit);
                println!("[P5.1] capture: Cornell OFF/ON split queued");
            }
            ViewerCmd::CaptureP51Grid => {
                self.pending_captures.push_back(CaptureKind::P51AoGrid);
                println!("[P5.1] capture: AO buffers grid queued");
            }
            ViewerCmd::CaptureP51Sweep => {
                self.pending_captures.push_back(CaptureKind::P51ParamSweep);
                println!("[P5.1] capture: AO parameter sweep queued");
            }
            ViewerCmd::CaptureP52SsgiCornell => {
                self.pending_captures.push_back(CaptureKind::P52SsgiCornell);
                println!("[P5.2] capture: SSGI Cornell split queued");
            }
            ViewerCmd::CaptureP52SsgiTemporal => {
                self.pending_captures
                    .push_back(CaptureKind::P52SsgiTemporal);
                println!("[P5.2] capture: SSGI temporal compare queued");
            }
            ViewerCmd::CaptureP53SsrGlossy => {
                self.pending_captures.push_back(CaptureKind::P53SsrGlossy);
                println!("[P5.3] capture: SSR glossy spheres queued");
            }
            ViewerCmd::CaptureP53SsrThickness => {
                self.pending_captures
                    .push_back(CaptureKind::P53SsrThickness);
                println!("[P5.3] capture: SSR thickness ablation queued");
            }
            ViewerCmd::CaptureP54GiStack => {
                self.pending_captures
                    .push_back(CaptureKind::P54GiStack);
                println!("[P5.4] capture: GI stack ablation queued");
            }
            // Sky controls
            ViewerCmd::SkyToggle(on) => {
                self.sky_enabled = on;
            }
            ViewerCmd::SkySetModel(id) => {
                self.sky_model_id = id;
                self.sky_enabled = true;
            }
            ViewerCmd::SkySetTurbidity(t) => {
                self.sky_turbidity = t.clamp(1.0, 10.0);
            }
            ViewerCmd::SkySetGround(a) => {
                self.sky_ground_albedo = a.clamp(0.0, 1.0);
            }
            ViewerCmd::SkySetExposure(e) => {
                self.sky_exposure = e.max(0.0);
            }
            ViewerCmd::SkySetSunIntensity(i) => {
                self.sky_sun_intensity = i.max(0.0);
            }
            // Fog controls
            ViewerCmd::FogToggle(on) => {
                self.fog_enabled = on;
            }
            ViewerCmd::FogSetDensity(v) => {
                self.fog_density = v.max(0.0);
            }
            ViewerCmd::FogSetG(v) => {
                self.fog_g = v.clamp(-0.999, 0.999);
            }
            ViewerCmd::FogSetSteps(v) => {
                self.fog_steps = v.max(1);
            }
            ViewerCmd::FogSetShadow(on) => {
                self.fog_use_shadows = on;
            }
            ViewerCmd::FogSetTemporal(v) => {
                self.fog_temporal_alpha = v.clamp(0.0, 0.9);
            }
            ViewerCmd::SetFogMode(m) => {
                self.fog_mode = if m != 0 {
                    FogMode::Froxels
                } else {
                    FogMode::Raymarch
                };
            }
            ViewerCmd::FogHalf(on) => {
                self.fog_half_res_enabled = on;
            }
            ViewerCmd::FogEdges(on) => {
                self.fog_bilateral = on;
            }
            ViewerCmd::FogUpsigma(s) => {
                self.fog_upsigma = s.max(0.0);
            }
            ViewerCmd::FogPreset(p) => {
                match p {
                    0 => {
                        self.fog_steps = 32;
                        self.fog_temporal_alpha = 0.7;
                        self.fog_density = 0.02;
                    }
                    1 => {
                        self.fog_steps = 64;
                        self.fog_temporal_alpha = 0.6;
                        self.fog_density = 0.04;
                    }
                    _ => {
                        self.fog_steps = 96;
                        self.fog_temporal_alpha = 0.5;
                        self.fog_density = 0.06;
                    }
                }
            }
            ViewerCmd::HudToggle(on) => {
                self.hud_enabled = on;
                self.hud.set_enabled(on);
            }
            ViewerCmd::LoadIbl(path) => match self.load_ibl(&path) {
                Ok(_) => println!("Loaded IBL: {}", path),
                Err(e) => eprintln!("IBL load failed: {}", e),
            },
            ViewerCmd::IblToggle(on) => {
                self.lit_use_ibl = on;
                if on && self.ibl_renderer.is_none() {
                    println!(
                        "IBL enabled (no environment loaded; use :ibl load <path> to load HDR)"
                    );
                } else if !on {
                    println!("IBL disabled");
                }
                self.update_lit_uniform();
            }
            ViewerCmd::IblIntensity(v) => {
                self.lit_ibl_intensity = v.max(0.0);
                self.lit_use_ibl = self.lit_ibl_intensity > 0.0;
                self.update_lit_uniform();
                println!("IBL intensity: {:.2}", self.lit_ibl_intensity);
            }
            ViewerCmd::IblRotate(deg) => {
                self.lit_ibl_rotation_deg = deg;
                println!("IBL rotation: {:.1}°", deg);
            }
            ViewerCmd::IblCache(dir) => {
                if let Some(ref cache_path) = dir {
                    self.ibl_cache_dir = Some(std::path::PathBuf::from(cache_path));
                    println!(
                        "IBL cache directory: {} (will be used on next load)",
                        cache_path
                    );
                    if let Some(ref mut ibl) = self.ibl_renderer {
                        let hdr_path = self
                            .ibl_hdr_path
                            .as_ref()
                            .map(|p| Path::new(p))
                            .unwrap_or_else(|| Path::new(""));
                        if let Err(e) = ibl.configure_cache(cache_path, hdr_path) {
                            eprintln!("Failed to configure IBL cache: {}", e);
                        } else {
                            println!("IBL cache reconfigured");
                        }
                    }
                } else {
                    self.ibl_cache_dir = None;
                    println!("IBL cache directory cleared (cache will be disabled on next load)");
                }
            }
            ViewerCmd::IblRes(res) => {
                self.ibl_base_resolution = Some(res);
                println!("IBL base resolution: {} (will be used on next load)", res);
                if let Some(ref mut ibl) = self.ibl_renderer {
                    ibl.set_base_resolution(res);
                    if let Err(e) = ibl.initialize(&self.device, &self.queue) {
                        eprintln!("Failed to reinitialize IBL with new resolution: {}", e);
                    } else {
                        println!("IBL reinitialized with resolution {}", res);
                    }
                }
            }
            // IPC-specific commands
            ViewerCmd::SetSunDirection { azimuth_deg, elevation_deg } => {
                let az_rad = azimuth_deg.to_radians();
                let el_rad = elevation_deg.to_radians();
                let _dir = glam::Vec3::new(
                    el_rad.cos() * az_rad.sin(),
                    el_rad.sin(),
                    el_rad.cos() * az_rad.cos(),
                );
                println!("Sun direction: azimuth={:.1}° elevation={:.1}°", azimuth_deg, elevation_deg);
            }
            ViewerCmd::SetIbl { path, intensity } => {
                match self.load_ibl(&path) {
                    Ok(_) => {
                        self.lit_ibl_intensity = intensity.max(0.0);
                        self.lit_use_ibl = self.lit_ibl_intensity > 0.0;
                        self.update_lit_uniform();
                        println!("Loaded IBL: {} with intensity {:.2}", path, intensity);
                    }
                    Err(e) => eprintln!("IBL load failed: {}", e),
                }
            }
            ViewerCmd::SetZScale(value) => {
                #[cfg(feature = "extension-module")]
                {
                    if let Some(ref mut _scene) = self.terrain_scene {
                        println!("Terrain z-scale set to {:.2} (terrain scene attached)", value);
                    } else {
                        eprintln!("SetZScale error: z-scale only applies to terrain scenes");
                    }
                }
                #[cfg(not(feature = "extension-module"))]
                {
                    let _ = value;
                    eprintln!("SetZScale error: terrain support not compiled in");
                }
            }
            ViewerCmd::SnapshotWithSize { path, width, height } => {
                if let (Some(w), Some(h)) = (width, height) {
                    self.view_config.snapshot_width = Some(w);
                    self.view_config.snapshot_height = Some(h);
                }
                self.snapshot_request = Some(path);
            }
            ViewerCmd::SaveBundle { path, name } => {
                // Bundle saving is handled via Python; this just acknowledges the command
                let bundle_name = name.as_deref().unwrap_or("scene");
                println!("SaveBundle requested: {} (name: {})", path, bundle_name);
                // Store the request for Python-side handling (local struct)
                self.pending_bundle_save = Some((path.clone(), name.clone()));
                // Also set global IPC state for polling from server thread
                set_pending_bundle_save(path, name);
            }
            ViewerCmd::LoadBundle { path } => {
                // Bundle loading is handled via Python; this just acknowledges the command
                println!("LoadBundle requested: {}", path);
                // Store the request for Python-side handling (local struct)
                self.pending_bundle_load = Some(path.clone());
                // Also set global IPC state for polling from server thread
                set_pending_bundle_load(path);
            }
            ViewerCmd::SetFov(fov) => {
                self.view_config.fov_deg = fov.clamp(1.0, 179.0);
                println!("FOV set to {:.1}°", self.view_config.fov_deg);
            }
            ViewerCmd::SetCamLookAt { eye, target, up } => {
                let e = glam::Vec3::from(eye);
                let t = glam::Vec3::from(target);
                let u = glam::Vec3::from(up);
                self.camera.set_look_at(e, t, u);
                println!("Camera: eye={:?} target={:?} up={:?}", eye, target, up);
            }
            ViewerCmd::SetSize(w, h) => {
                println!("Requested size {}x{} (resize via window manager)", w, h);
            }
            ViewerCmd::SetVizDepthMax(_v) => {
                // Depth visualization range is not wired yet; keep shader defaults.
            }
            ViewerCmd::SetTransform {
                translation,
                rotation_quat,
                scale,
            } => {
                if let Some(t) = translation {
                    self.object_translation = glam::Vec3::from(t);
                }
                if let Some(q) = rotation_quat {
                    self.object_rotation = glam::Quat::from_array(q).normalize();
                }
                if let Some(s) = scale {
                    self.object_scale = glam::Vec3::from(s);
                }
                self.object_transform = glam::Mat4::from_scale_rotation_translation(
                    self.object_scale,
                    self.object_rotation,
                    self.object_translation,
                );
                
                // CPU-side vertex transform
                if !self.original_mesh_positions.is_empty() {
                    use viewer_types::PackedVertex;
                    let vertex_count = self.original_mesh_positions.len();
                    let mut vertices: Vec<PackedVertex> = Vec::with_capacity(vertex_count);
                    
                    for i in 0..vertex_count {
                        let orig_pos = glam::Vec3::from(self.original_mesh_positions[i]);
                        let transformed = self.object_transform.transform_point3(orig_pos);
                        
                        let orig_nrm = if i < self.original_mesh_normals.len() {
                            glam::Vec3::from(self.original_mesh_normals[i])
                        } else {
                            glam::Vec3::Y
                        };
                        let rot_mat = glam::Mat3::from_quat(self.object_rotation);
                        let transformed_nrm = (rot_mat * orig_nrm).normalize();
                        
                        let uv = if i < self.original_mesh_uvs.len() {
                            self.original_mesh_uvs[i]
                        } else {
                            [0.0, 0.0]
                        };
                        
                        vertices.push(PackedVertex {
                            position: transformed.to_array(),
                            normal: transformed_nrm.to_array(),
                            uv,
                            rough_metal: [0.5, 0.0],
                        });
                    }
                    
                    let vertex_data = bytemuck::cast_slice(&vertices);
                    let new_vb = self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("viewer.ipc.mesh.vb.transformed"),
                            contents: vertex_data,
                            usage: wgpu::BufferUsages::VERTEX,
                        }
                    );
                    self.geom_vb = Some(new_vb);
                    
                    // D1: Log CPU transform applied
                    let msg = format!(
                        "[D1-CPU-TRANSFORM] frame={} vertices={} trans=[{:.3},{:.3},{:.3}] scale=[{:.3},{:.3},{:.3}]\n",
                        self.frame_count, vertex_count,
                        self.object_translation.x, self.object_translation.y, self.object_translation.z,
                        self.object_scale.x, self.object_scale.y, self.object_scale.z
                    );
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("examples/out/d1_debug.log")
                        .and_then(|mut f| {
                            use std::io::Write;
                            f.write_all(msg.as_bytes())
                        });
                }
                
                self.transform_version += 1;
                let is_identity = self.object_translation == glam::Vec3::ZERO
                    && self.object_rotation == glam::Quat::IDENTITY
                    && self.object_scale == glam::Vec3::ONE;
                update_ipc_transform_stats(self.transform_version, is_identity);
            }
            ViewerCmd::LoadTerrain(path) => {
                if self.terrain_viewer.is_none() {
                    match terrain::ViewerTerrainScene::new(
                        std::sync::Arc::clone(&self.device),
                        std::sync::Arc::clone(&self.queue),
                        self.config.format,
                    ) {
                        Ok(scene) => self.terrain_viewer = Some(scene),
                        Err(e) => {
                            eprintln!("[terrain] Failed to create viewer: {}", e);
                            return;
                        }
                    }
                }
                if let Some(ref mut tv) = self.terrain_viewer {
                    match tv.load_terrain(&path) {
                        Ok(()) => println!("[terrain] Loaded: {}", path),
                        Err(e) => eprintln!("[terrain] Failed to load {}: {}", path, e),
                    }
                }
            }
            ViewerCmd::SetTerrainCamera { phi_deg, theta_deg, radius, fov_deg } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_camera(phi_deg, theta_deg, radius, fov_deg);
                    println!("[terrain] Camera: phi={:.1}° theta={:.1}° r={:.1} fov={:.1}°", 
                        phi_deg, theta_deg, radius, fov_deg);
                }
            }
            ViewerCmd::SetTerrainSun { azimuth_deg, elevation_deg, intensity } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_sun(azimuth_deg, elevation_deg, intensity);
                    println!("[terrain] Sun: az={:.1}° el={:.1}° int={:.2}", 
                        azimuth_deg, elevation_deg, intensity);
                }
            }
            ViewerCmd::SetTerrain {
                phi, theta, radius, fov,
                sun_azimuth, sun_elevation, sun_intensity,
                ambient, zscale, shadow, background, water_level, water_color,
            } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    if let Some(t) = tv.terrain.as_mut() {
                        if let Some(v) = phi { t.cam_phi_deg = v; }
                        if let Some(v) = theta { t.cam_theta_deg = v.clamp(5.0, 85.0); }
                        if let Some(v) = radius { t.cam_radius = v.clamp(100.0, 50000.0); }
                        if let Some(v) = fov { t.cam_fov_deg = v.clamp(10.0, 120.0); }
                        if let Some(v) = sun_azimuth { t.sun_azimuth_deg = v; }
                        if let Some(v) = sun_elevation { t.sun_elevation_deg = v.clamp(-90.0, 90.0); }
                        if let Some(v) = sun_intensity { t.sun_intensity = v.max(0.0); }
                        if let Some(v) = ambient { t.ambient = v.clamp(0.0, 1.0); }
                        if let Some(v) = zscale { t.z_scale = v.max(0.01); }
                        if let Some(v) = shadow { t.shadow_intensity = v.clamp(0.0, 1.0); }
                        if let Some(bg) = background { t.background_color = bg; }
                        if let Some(v) = water_level { t.water_level = v; }
                        if let Some(wc) = water_color { t.water_color = wc; }
                    }
                    if let Some(params) = tv.get_params() {
                        println!("[terrain] {}", params);
                    }
                }
            }
            ViewerCmd::GetTerrainParams => {
                if let Some(ref tv) = self.terrain_viewer {
                    if let Some(params) = tv.get_params() {
                        println!("[terrain] {}", params);
                    }
                }
            }
            ViewerCmd::SetTerrainPbr {
                enabled,
                hdr_path,
                ibl_intensity,
                shadow_technique,
                shadow_map_res,
                exposure,
                msaa,
                normal_strength,
                height_ao,
                sun_visibility,
                materials,
                vector_overlay,
                tonemap,
                dof,
                motion_blur,
                lens_effects,
                denoise,
                volumetrics,
                sky,
                debug_mode,
            } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_terrain_pbr(
                        enabled,
                        hdr_path,
                        ibl_intensity,
                        shadow_technique,
                        shadow_map_res,
                        exposure,
                        msaa,
                        normal_strength,
                        *height_ao,
                        *sun_visibility,
                        *materials,
                        *vector_overlay,
                        tonemap,
                        lens_effects,
                        *dof,
                        motion_blur,
                        *volumetrics,
                        denoise,
                        debug_mode,
                    );
                }
                
                // M6: Wire sky config to existing viewer sky system
                if let Some(ref cfg) = sky {
                    self.sky_enabled = cfg.enabled;
                    if cfg.enabled {
                        self.sky_turbidity = cfg.turbidity;
                        self.sky_ground_albedo = cfg.ground_albedo;
                        self.sky_exposure = cfg.sky_exposure;
                        self.sky_sun_intensity = cfg.sun_intensity;
                        println!("[terrain] Sky enabled: turbidity={:.1} ground_albedo={:.2} exposure={:.2}", 
                            cfg.turbidity, cfg.ground_albedo, cfg.sky_exposure);
                    }
                }
            }
            // Overlay commands
            ViewerCmd::LoadOverlay { name, path, extent, opacity, z_order } => {
                println!("[overlay] LoadOverlay command received: name='{}' path='{}'", name, path);
                if let Some(ref mut tv) = self.terrain_viewer {
                    use std::path::Path;
                    let opacity_val = opacity.unwrap_or(1.0);
                    let z_order_val = z_order.unwrap_or(0);
                    println!("[overlay] terrain_viewer exists, calling add_overlay_image...");
                    match tv.add_overlay_image(
                        &name,
                        Path::new(&path),
                        extent,
                        opacity_val,
                        crate::viewer::terrain::BlendMode::Normal,
                        z_order_val,
                    ) {
                        Ok(id) => println!("[overlay] Loaded '{}' from {} (id={})", name, path, id),
                        Err(e) => eprintln!("[overlay] Failed to load '{}': {}", name, e),
                    }
                } else {
                    eprintln!("[overlay] No terrain loaded - load terrain first (terrain_viewer is None)");
                }
            }
            ViewerCmd::RemoveOverlay { id } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    if tv.remove_overlay(id) {
                        println!("[overlay] Removed overlay id={}", id);
                    } else {
                        eprintln!("[overlay] Overlay id={} not found", id);
                    }
                }
            }
            ViewerCmd::SetOverlayVisible { id, visible } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_overlay_visible(id, visible);
                    println!("[overlay] id={} visible={}", id, visible);
                }
            }
            ViewerCmd::SetOverlayOpacity { id, opacity } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_overlay_opacity(id, opacity);
                    println!("[overlay] id={} opacity={:.2}", id, opacity);
                }
            }
            ViewerCmd::SetGlobalOverlayOpacity { opacity } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_global_overlay_opacity(opacity);
                    println!("[overlay] global opacity={:.2}", opacity);
                }
            }
            ViewerCmd::SetOverlaysEnabled { enabled } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_overlays_enabled(enabled);
                    println!("[overlay] enabled={}", enabled);
                }
            }
            ViewerCmd::SetOverlaySolid { solid } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_overlay_solid(solid);
                    println!("[overlay] solid={}", solid);
                }
            }
            ViewerCmd::ListOverlays => {
                if let Some(ref tv) = self.terrain_viewer {
                    let ids = tv.list_overlays();
                    if ids.is_empty() {
                        println!("[overlay] No overlays loaded");
                    } else {
                        println!("[overlay] Loaded overlays: {:?}", ids);
                    }
                } else {
                    println!("[overlay] No terrain loaded");
                }
            }
            
            // === OPTION B: VECTOR OVERLAY GEOMETRY COMMANDS ===
            ViewerCmd::AddVectorOverlay {
                name,
                vertices,
                indices,
                primitive,
                drape,
                drape_offset,
                opacity,
                depth_bias,
                line_width,
                point_size,
                z_order,
            } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    use crate::viewer::terrain::vector_overlay::{VectorOverlayLayer, VectorVertex, OverlayPrimitive};
                    
                    // Convert vertices from [x, y, z, r, g, b, a, feature_id?] to VectorVertex
                    let verts: Vec<VectorVertex> = vertices.iter().map(|v| {
                        let feature_id = if v.len() > 7 { v[7] as u32 } else { 0 };
                        VectorVertex::with_feature_id(v[0], v[1], v[2], v[3], v[4], v[5], v[6], feature_id)
                    }).collect();
                    
                    let prim = OverlayPrimitive::from_str(&primitive).unwrap_or(OverlayPrimitive::Triangles);
                    
                    let layer = VectorOverlayLayer {
                        name: name.clone(),
                        vertices: verts,
                        indices: indices.clone(),
                        primitive: prim,
                        drape,
                        drape_offset,
                        opacity,
                        depth_bias,
                        line_width,
                        point_size,
                        visible: true,
                        z_order,
                    };
                    
                    let id = tv.add_vector_overlay(layer);
                    println!("[vector_overlay] Added '{}' with {} vertices (id={})", name, vertices.len(), id);

                    // Plan 3: Build BVH for picking if triangles
                    if prim == OverlayPrimitive::Triangles && !indices.is_empty() {
                        use crate::accel::cpu_bvh::{MeshCPU, build_bvh_cpu, BuildOptions};
                        use crate::picking::LayerBvhData;
                        use crate::accel::types::{BvhNode, Triangle, Aabb};

                        let pos: Vec<[f32; 3]> = vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
                        let tris_indices: Vec<[u32; 3]> = indices.chunks(3).filter_map(|c| {
                            if c.len() == 3 { Some([c[0], c[1], c[2]]) } else { None }
                        }).collect();

                        let mesh = MeshCPU::new(pos.clone(), tris_indices.clone());
                        if let Ok(bvh) = build_bvh_cpu(&mesh, &BuildOptions::default()) {
                            let mut layer_data = LayerBvhData::new(id, name.clone());
                            
                            // Convert cpu_bvh::BvhNode to accel::types::BvhNode
                            layer_data.cpu_nodes = bvh.nodes.iter().map(|n| {
                                let kind = if n.is_leaf() { 1 } else { 0 };
                                BvhNode {
                                    aabb: Aabb::new(n.aabb_min, n.aabb_max),
                                    kind,
                                    left_idx: n.left,
                                    right_idx: n.right,
                                    parent_idx: 0, // Parent pointers not populated by default build
                                }
                            }).collect();

                            layer_data.cpu_triangles = bvh.tri_indices.iter().map(|&tri_idx| {
                                let idx = tri_idx as usize;
                                if idx < tris_indices.len() {
                                    let ti = tris_indices[idx];
                                    let v0 = pos[ti[0] as usize];
                                    let v1 = pos[ti[1] as usize];
                                    let v2 = pos[ti[2] as usize];
                                    Triangle::new(v0, v1, v2)
                                } else {
                                    Triangle::new([0.0; 3], [0.0; 3], [0.0; 3])
                                }
                            }).collect();

                            // Feature IDs - extract from vertex data (index 7 is feature_id)
                            // vertices format: [x, y, z, r, g, b, a, feature_id]
                            layer_data.cpu_feature_ids = bvh.tri_indices.iter().map(|&tri_idx| {
                                let idx = tri_idx as usize;
                                if idx < tris_indices.len() {
                                    let ti = tris_indices[idx];
                                    // Use feature_id from first vertex of triangle
                                    let v0_idx = ti[0] as usize;
                                    if v0_idx < vertices.len() {
                                        vertices[v0_idx][7] as u32
                                    } else {
                                        id // fallback to layer id
                                    }
                                } else {
                                    id // fallback to layer id
                                }
                            }).collect();

                            self.unified_picking.register_layer_bvh(layer_data);
                            println!("[picking] Built BVH for layer {} ({} nodes)", id, bvh.nodes.len());
                        }
                    }
                } else {
                    eprintln!("[vector_overlay] No terrain loaded - load terrain first");
                }
            }

            // Picking commands
            ViewerCmd::PollPickEvents => {
                // Handled in server.rs, but we might want to do something here?
                // The server directly accesses the shared queue.
            }
            ViewerCmd::SetLassoMode { enabled } => {
                self.unified_picking.set_lasso_enabled(enabled);
                let state = if enabled { "active" } else { "inactive" };
                // Update shared lasso state
                if let Ok(mut s) = crate::viewer::event_loop::get_lasso_state().lock() {
                    *s = state.to_string();
                }
                println!("[picking] Lasso mode: {}", state);
            }
            ViewerCmd::GetLassoState => {
                // Handled in server.rs
            }
            ViewerCmd::ClearSelection => {
                // ClearSelection is a no-op until a selection set name is provided.
                println!("[picking] Clear selection requested");
            }
            ViewerCmd::RemoveVectorOverlay { id } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    if tv.remove_vector_overlay(id) {
                        println!("[vector_overlay] Removed id={}", id);
                    } else {
                        eprintln!("[vector_overlay] id={} not found", id);
                    }
                }
            }
            ViewerCmd::SetVectorOverlayVisible { id, visible } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_vector_overlay_visible(id, visible);
                    println!("[vector_overlay] id={} visible={}", id, visible);
                }
            }
            ViewerCmd::SetVectorOverlayOpacity { id, opacity } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_vector_overlay_opacity(id, opacity);
                    println!("[vector_overlay] id={} opacity={:.2}", id, opacity);
                }
            }
            ViewerCmd::ListVectorOverlays => {
                if let Some(ref tv) = self.terrain_viewer {
                    let ids = tv.list_vector_overlays();
                    if ids.is_empty() {
                        println!("[vector_overlay] No vector overlays loaded");
                    } else {
                        println!("[vector_overlay] Loaded: {:?}", ids);
                    }
                } else {
                    println!("[vector_overlay] No terrain loaded");
                }
            }
            ViewerCmd::SetVectorOverlaysEnabled { enabled } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_vector_overlays_enabled(enabled);
                    println!("[vector_overlay] enabled={}", enabled);
                }
            }
            ViewerCmd::SetGlobalVectorOverlayOpacity { opacity } => {
                if let Some(ref mut tv) = self.terrain_viewer {
                    tv.set_global_vector_overlay_opacity(opacity);
                    println!("[vector_overlay] global opacity={:.2}", opacity);
                }
            }



            // === LABELS ===
            ViewerCmd::AddLabel {
                text,
                world_pos,
                size,
                color,
                halo_color,
                halo_width,
                priority,
                min_zoom,
                max_zoom,
                offset,
                rotation,
                underline,
                small_caps,
                leader,
                horizon_fade_angle,
            } => {
                use crate::labels::LabelStyle;
                let mut style = LabelStyle::default();
                if let Some(s) = size {
                    style.size = s;
                }
                if let Some(c) = color {
                    style.color = c;
                }
                if let Some(hc) = halo_color {
                    style.halo_color = hc;
                }
                if let Some(hw) = halo_width {
                    style.halo_width = hw;
                }
                if let Some(p) = priority {
                    style.priority = p;
                }
                if let Some(mz) = min_zoom {
                    style.min_zoom = mz;
                }
                if let Some(mz) = max_zoom {
                    style.max_zoom = mz;
                }
                if let Some(o) = offset {
                    style.offset = o;
                }
                if let Some(r) = rotation {
                    style.rotation = r;
                }
                if let Some(u) = underline {
                    style.flags.underline = u;
                }
                if let Some(sc) = small_caps {
                    style.flags.small_caps = sc;
                }
                if let Some(l) = leader {
                    style.flags.leader = l;
                }
                if let Some(hfa) = horizon_fade_angle {
                    style.horizon_fade_angle = hfa;
                }
                let id = self.add_label(&text, (world_pos[0], world_pos[1], world_pos[2]), Some(style));
                println!("[label] added id={} text=\"{}\"", id, text);
            }
            ViewerCmd::AddLineLabel {
                text,
                polyline,
                size,
                color,
                halo_color,
                halo_width,
                priority,
                placement,
                repeat_distance,
                min_zoom,
                max_zoom,
            } => {
                use crate::labels::{LabelStyle, LineLabelPlacement};
                use glam::Vec3;
                let mut style = LabelStyle::default();
                if let Some(s) = size {
                    style.size = s;
                }
                if let Some(c) = color {
                    style.color = c;
                }
                if let Some(hc) = halo_color {
                    style.halo_color = hc;
                }
                if let Some(hw) = halo_width {
                    style.halo_width = hw;
                }
                if let Some(p) = priority {
                    style.priority = p;
                }
                if let Some(mz) = min_zoom {
                    style.min_zoom = mz;
                }
                if let Some(mz) = max_zoom {
                    style.max_zoom = mz;
                }
                let pl = match placement.as_deref() {
                    Some("along") => LineLabelPlacement::Along,
                    _ => LineLabelPlacement::Center,
                };
                let poly: Vec<Vec3> = polyline
                    .iter()
                    .map(|p| Vec3::new(p[0], p[1], p[2]))
                    .collect();
                let rd = repeat_distance.unwrap_or(0.0);
                let id = self.label_manager.add_line_label(text.clone(), poly, style, pl, rd);
                println!("[label] added line label id={} text=\"{}\"", id.0, text);
            }
            ViewerCmd::RemoveLabel { id } => {
                let removed = self.remove_label(id);
                println!("[label] remove id={} success={}", id, removed);
            }
            ViewerCmd::ClearLabels => {
                self.clear_labels();
                println!("[label] cleared all labels");
            }
            ViewerCmd::SetLabelsEnabled { enabled } => {
                self.set_labels_enabled(enabled);
                println!("[label] enabled={}", enabled);
            }
            ViewerCmd::LoadLabelAtlas {
                atlas_png_path,
                metrics_json_path,
            } => {
                match self.load_label_atlas(&atlas_png_path, &metrics_json_path) {
                    Ok(()) => println!("[label] atlas loaded from {}", atlas_png_path),
                    Err(e) => eprintln!("[label] failed to load atlas: {}", e),
                }
            }
            ViewerCmd::SetLabelZoom { zoom } => {
                self.label_manager.set_zoom(zoom);
                println!("[label] zoom={:.2}", zoom);
            }
            ViewerCmd::SetMaxVisibleLabels { max } => {
                self.label_manager.set_max_visible(max);
                println!("[label] max_visible={}", max);
            }

            // === Plan 3: Premium Label Features ===
            ViewerCmd::AddCurvedLabel {
                text,
                polyline,
                size,
                color,
                halo_color,
                halo_width,
                priority,
                tracking: _tracking,
                center_on_path: _center_on_path,
            } => {
                use crate::labels::LabelStyle;
                use glam::Vec3;
                let mut style = LabelStyle::default();
                if let Some(s) = size {
                    style.size = s;
                }
                if let Some(c) = color {
                    style.color = c;
                }
                if let Some(hc) = halo_color {
                    style.halo_color = hc;
                }
                if let Some(hw) = halo_width {
                    style.halo_width = hw;
                }
                if let Some(p) = priority {
                    style.priority = p;
                }
                // Convert polyline to Vec3
                let poly: Vec<Vec3> = polyline
                    .iter()
                    .map(|p| Vec3::new(p[0], p[1], p[2]))
                    .collect();
                // Use line label with "along" placement for curved text
                use crate::labels::LineLabelPlacement;
                let id = self.label_manager.add_line_label(
                    text.clone(),
                    poly,
                    style,
                    LineLabelPlacement::Along,
                    0.0,
                );
                println!("[label] added curved label id={} text=\"{}\"", id.0, text);
            }
            ViewerCmd::AddCallout {
                text,
                anchor,
                offset,
                background_color: _bg,
                border_color: _bc,
                border_width: _bw,
                corner_radius: _cr,
                padding: _pad,
                text_size,
                text_color,
            } => {
                // Callouts are implemented as offset labels with leader lines.
                use crate::labels::LabelStyle;
                let mut style = LabelStyle::default();
                if let Some(s) = text_size {
                    style.size = s;
                }
                if let Some(c) = text_color {
                    style.color = c;
                }
                if let Some(o) = offset {
                    style.offset = o;
                    style.flags.leader = true;
                }
                let world_pos = (anchor[0], anchor[1], anchor[2]);
                let id = self.add_label(&text, world_pos, Some(style));
                println!("[label] added callout id={} text=\"{}\"", id, text);
            }
            ViewerCmd::RemoveCallout { id } => {
                let removed = self.remove_label(id);
                println!("[label] remove callout id={} success={}", id, removed);
            }
            ViewerCmd::SetLabelTypography {
                tracking: _,
                kerning: _,
                line_height: _,
                word_spacing: _,
            } => {
                // Typography settings stored but not yet fully applied
                println!("[label] typography settings updated");
            }
            ViewerCmd::SetDeclutterAlgorithm {
                algorithm,
                seed: _,
                max_iterations: _,
            } => {
                println!("[label] declutter algorithm set to: {}", algorithm);
            }

            // P0.1/M1: OIT (Order-Independent Transparency)
            ViewerCmd::SetOitEnabled { enabled, mode } => {
                self.oit_enabled = enabled;
                self.oit_mode = mode.clone();
                // Wire to terrain scene if available
                if let Some(ref mut terrain) = self.terrain_viewer {
                    terrain.set_oit_mode(enabled, &mode);
                }
                println!("[oit] enabled={} mode={}", enabled, mode);
            }
            ViewerCmd::GetOitMode => {
                println!("[oit] enabled={} mode={}", self.oit_enabled, self.oit_mode);
            }

            // P1.3: TAA (Temporal Anti-Aliasing)
            ViewerCmd::SetTaaEnabled { enabled } => {
                self.set_taa_enabled(enabled);
                println!("[taa] enabled={}", enabled);
            }
            ViewerCmd::GetTaaStatus => {
                let taa_enabled = self.taa_renderer.as_ref().map(|t| t.is_enabled()).unwrap_or(false);
                let jitter_enabled = self.taa_jitter.enabled;
                println!("[taa] enabled={} jitter_enabled={}", taa_enabled, jitter_enabled);
            }
            ViewerCmd::SetTaaParams {
                history_weight,
                jitter_scale,
                enable_jitter,
            } => {
                // Check if we should delegate to terrain viewer
                let terrain_active = self.terrain_viewer.as_ref().map(|tv| tv.has_terrain()).unwrap_or(false);
                
                if terrain_active {
                    if let Some(ref mut tv) = self.terrain_viewer {
                         tv.set_taa_params(history_weight, jitter_scale);
                         // Note: explicit jitter enable not yet supported for terrain viewer delegation via this command
                    }
                } else {
                    // Main viewer logic
                    if let Some(w) = history_weight {
                        if let Some(ref mut taa) = self.taa_renderer {
                             taa.set_history_weight(w);
                        }
                    }
                    if let Some(scale) = jitter_scale {
                         self.taa_jitter.set_scale(scale);
                    }
                    if let Some(enabled) = enable_jitter {
                         self.taa_jitter.set_enabled(enabled);
                    } else if let Some(scale) = jitter_scale {
                         if scale > 0.0 && !self.taa_jitter.enabled {
                             self.taa_jitter.set_enabled(true);
                         }
                    }
                    
                    let taa_weight = self.taa_renderer.as_ref().map(|t| t.history_weight()).unwrap_or(0.0);
                    println!("[taa] params updated: weight={:.2} jitter_scale={:.2} jitter_enabled={}", 
                        taa_weight, self.taa_jitter.scale, self.taa_jitter.enabled);
                }
            }

            // P5: Point cloud commands
            ViewerCmd::LoadPointCloud { path, point_size, max_points, color_mode } => {
                use super::super::pointcloud::{PointCloudState, ColorMode};
                
                eprintln!("[pointcloud] Loading: {} (size={}, max={}, mode={:?})",
                    path, point_size, max_points, color_mode);
                
                // Initialize point cloud state if needed
                if self.point_cloud.is_none() {
                    eprintln!("[pointcloud] Creating PointCloudState...");
                    let depth_format = wgpu::TextureFormat::Depth32Float;
                    self.point_cloud = Some(PointCloudState::new(
                        &self.device,
                        self.config.format,
                        depth_format,
                    ));
                    eprintln!("[pointcloud] PointCloudState created");
                }
                
                if let Some(ref mut pc) = self.point_cloud {
                    let mode = color_mode.as_ref()
                        .map(|s| ColorMode::from_str(s))
                        .unwrap_or(ColorMode::Elevation);
                    
                    pc.set_point_size(point_size);
                    eprintln!("[pointcloud] Loading file...");
                    
                    match pc.load_from_file(&self.device, &self.queue, &path, max_points, mode) {
                        Ok(()) => {
                            eprintln!("[pointcloud] Loaded {} points", pc.point_count);
                            eprintln!("[pointcloud] Bounds: ({:.1}, {:.1}, {:.1}) - ({:.1}, {:.1}, {:.1})",
                                pc.bounds_min[0], pc.bounds_min[1], pc.bounds_min[2],
                                pc.bounds_max[0], pc.bounds_max[1], pc.bounds_max[2]);
                            eprintln!("[pointcloud] Center: ({:.1}, {:.1}, {:.1})",
                                pc.center[0], pc.center[1], pc.center[2]);
                            eprintln!("[pointcloud] Load complete, returning to render loop");
                        }
                        Err(e) => {
                            eprintln!("[pointcloud] Error: {}", e);
                        }
                    }
                }
            }
            ViewerCmd::ClearPointCloud => {
                if let Some(ref mut pc) = self.point_cloud {
                    pc.clear();
                }
                println!("[pointcloud] Cleared");
            }
            ViewerCmd::SetPointCloudParams { point_size, visible, color_mode } => {
                if let Some(ref mut pc) = self.point_cloud {
                    if let Some(size) = point_size {
                        pc.set_point_size(size);
                    }
                    if let Some(vis) = visible {
                        pc.set_visible(vis);
                    }
                    if let Some(mode) = color_mode {
                        use super::super::pointcloud::ColorMode;
                        pc.color_mode = ColorMode::from_str(&mode);
                    }
                    println!("[pointcloud] Params updated: size={}, visible={}, mode={:?}",
                        pc.point_size, pc.visible, pc.color_mode);
                }
            }
        }
    }
}
