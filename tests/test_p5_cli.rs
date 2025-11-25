use forge3d::cli::args::{GiCliConfig, GiEffect, GiEntry, GiVizMode, Toggle};

fn split_cli_string(s: &str) -> Vec<String> {
    s.split_whitespace().map(|t| t.to_string()).collect()
}

fn make_args(spec: &[&str]) -> Vec<String> {
    spec.iter().map(|s| s.to_string()).collect()
}

#[derive(Debug, Default, PartialEq)]
struct MockViewerGiConfig {
    gi_ssao: Option<bool>,
    gi_ssgi: Option<bool>,
    gi_ssr: Option<bool>,
    ssao_radius: Option<f32>,
    ssao_intensity: Option<f32>,
    ssao_technique: Option<String>,
    ssao_mul: Option<f32>,
    ssao_bias: Option<f32>,
    ssao_samples: Option<u32>,
    ssao_directions: Option<u32>,
    ssao_temporal_alpha: Option<f32>,
    ao_blur: Option<bool>,
    ssgi_steps: Option<u32>,
    ssgi_radius: Option<f32>,
    ssgi_half: Option<bool>,
    ssgi_temporal_alpha: Option<f32>,
    ssgi_temporal_enabled: Option<bool>,
    ssgi_edges: Option<bool>,
    ssgi_upsample_sigma_depth: Option<f32>,
    ssgi_upsample_sigma_normal: Option<f32>,
    ssr_enable: Option<bool>,
    ssr_max_steps: Option<u32>,
    ssr_thickness: Option<f32>,
    gi_seed: Option<u32>,
    gi_viz: Option<GiVizMode>,
    ao_weight: Option<f32>,
    ssgi_weight: Option<f32>,
    ssr_weight: Option<f32>,
}

fn apply_gi_commands(cmds: &[String]) -> MockViewerGiConfig {
    let mut cfg = MockViewerGiConfig::default();
    for raw in cmds {
        cfg.apply_command(raw);
    }
    cfg
}

impl MockViewerGiConfig {
    fn apply_command(&mut self, raw: &str) {
        let line = raw.trim();
        if line.is_empty() {
            return;
        }
        let l = line.strip_prefix(':').unwrap_or(line);
        let mut parts = l.split_whitespace();
        let cmd = match parts.next() {
            Some(c) => c,
            None => return,
        };
        match cmd {
            "gi" => {
                let eff = parts.next().unwrap_or("");
                let state = parts.next().unwrap_or("");
                let on = matches!(state, "on" | "1" | "true");
                match eff {
                    "ssao" => self.gi_ssao = Some(on),
                    "ssgi" => self.gi_ssgi = Some(on),
                    "ssr" => self.gi_ssr = Some(on),
                    "gtao" => {
                        self.gi_ssao = Some(on);
                        self.ssao_technique = Some("gtao".to_string());
                    }
                    _ => {}
                }
            }
            "gi-seed" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<u32>() {
                        self.gi_seed = Some(v);
                    }
                }
            }
            "ssao-radius" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssao_radius = Some(v);
                    }
                }
            }
            "ssao-intensity" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssao_intensity = Some(v);
                    }
                }
            }
            "ssao-technique" => {
                if let Some(tok) = parts.next() {
                    self.ssao_technique = Some(tok.to_string());
                }
            }
            "ssao-mul" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssao_mul = Some(v);
                    }
                }
            }
            "ssao-bias" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssao_bias = Some(v);
                    }
                }
            }
            "ssao-samples" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<u32>() {
                        self.ssao_samples = Some(v);
                    }
                }
            }
            "ssao-directions" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<u32>() {
                        self.ssao_directions = Some(v);
                    }
                }
            }
            "ssao-temporal-alpha" | "ao-temporal-alpha" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssao_temporal_alpha = Some(v);
                    }
                }
            }
            "ao-blur" => {
                if let Some(tok) = parts.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    self.ao_blur = Some(on);
                }
            }
            "ssgi-steps" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<u32>() {
                        self.ssgi_steps = Some(v);
                    }
                }
            }
            "ssgi-radius" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssgi_radius = Some(v);
                    }
                }
            }
            "ssgi-half" => {
                if let Some(tok) = parts.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    self.ssgi_half = Some(on);
                }
            }
            "ssgi-temporal-alpha" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssgi_temporal_alpha = Some(v);
                    }
                }
            }
            "ssgi-temporal" => {
                if let Some(tok) = parts.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    self.ssgi_temporal_enabled = Some(on);
                }
            }
            "ssgi-edges" => {
                if let Some(tok) = parts.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    self.ssgi_edges = Some(on);
                }
            }
            "ssgi-upsample-sigma-depth" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssgi_upsample_sigma_depth = Some(v);
                    }
                }
            }
            "ssgi-upsample-sigma-normal" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssgi_upsample_sigma_normal = Some(v);
                    }
                }
            }
            "ssr-max-steps" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<u32>() {
                        self.ssr_max_steps = Some(v);
                    }
                }
            }
            "ssr-thickness" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssr_thickness = Some(v);
                    }
                }
            }
            "ao-weight" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ao_weight = Some(v);
                    }
                }
            }
            "ssgi-weight" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssgi_weight = Some(v);
                    }
                }
            }
            "ssr-weight" => {
                if let Some(s) = parts.next() {
                    if let Ok(v) = s.parse::<f32>() {
                        self.ssr_weight = Some(v);
                    }
                }
            }
            "viz" => {
                if let Some(sub) = parts.next() {
                    if sub == "gi" {
                        if let Some(mode) = parts.next() {
                            let m = match mode {
                                "none" => Some(GiVizMode::None),
                                "composite" => Some(GiVizMode::Composite),
                                "ao" => Some(GiVizMode::Ao),
                                "ssgi" => Some(GiVizMode::Ssgi),
                                "ssr" => Some(GiVizMode::Ssr),
                                _ => None,
                            };
                            if m.is_some() {
                                self.gi_viz = m;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[test]
fn parse_all_gi_flags_into_config() {
    let args = vec![
        "--gi".to_string(),
        "ssao:on".to_string(),
        "--gi".to_string(),
        "ssgi:off".to_string(),
        "--gi".to_string(),
        "ssr:on".to_string(),
        "--gi-seed".to_string(),
        "42".to_string(),
        "--viz-gi".to_string(),
        "composite".to_string(),
        "--ssao-radius".to_string(),
        "0.5".to_string(),
        "--ssao-intensity".to_string(),
        "1.0".to_string(),
        "--ssao-technique".to_string(),
        "gtao".to_string(),
        "--ssao-composite".to_string(),
        "on".to_string(),
        "--ssao-mul".to_string(),
        "0.8".to_string(),
        "--ssao-bias".to_string(),
        "0.03".to_string(),
        "--ssao-samples".to_string(),
        "16".to_string(),
        "--ssao-directions".to_string(),
        "4".to_string(),
        "--ssao-temporal-alpha".to_string(),
        "0.2".to_string(),
        "--ao-temporal-alpha".to_string(),
        "0.3".to_string(),
        "--ao-blur".to_string(),
        "off".to_string(),
        "--ssgi-steps".to_string(),
        "24".to_string(),
        "--ssgi-radius".to_string(),
        "1.5".to_string(),
        "--ssgi-half".to_string(),
        "on".to_string(),
        "--ssgi-temporal-alpha".to_string(),
        "0.1".to_string(),
        "--ssgi-temporal-enable".to_string(),
        "on".to_string(),
        "--ssgi-edges".to_string(),
        "on".to_string(),
        "--ssgi-upsample-sigma-depth".to_string(),
        "0.02".to_string(),
        "--ssgi-upsample-sigma-normal".to_string(),
        "0.25".to_string(),
        "--ssr-enable".to_string(),
        "on".to_string(),
        "--ssr-max-steps".to_string(),
        "48".to_string(),
        "--ssr-thickness".to_string(),
        "0.08".to_string(),
        "--ao-weight".to_string(),
        "0.6".to_string(),
        "--ssgi-weight".to_string(),
        "0.7".to_string(),
        "--ssr-weight".to_string(),
        "0.8".to_string(),
    ];

    let cfg = GiCliConfig::parse(&args).expect("GI CLI parse failed");

    assert_eq!(
        cfg.entries,
        vec![
            GiEntry::Effect(GiEffect::Ssao, Toggle::On),
            GiEntry::Effect(GiEffect::Ssgi, Toggle::Off),
            GiEntry::Effect(GiEffect::Ssr, Toggle::On),
        ]
    );
    assert_eq!(cfg.gi_seed, Some(42));
    assert_eq!(cfg.gi_viz, Some(GiVizMode::Composite));

    assert_eq!(cfg.ssao.radius, Some(0.5));
    assert_eq!(cfg.ssao.intensity, Some(1.0));
    assert_eq!(cfg.ssao.technique.as_deref(), Some("gtao"));
    assert_eq!(cfg.ssao.composite_enabled, Some(true));
    assert_eq!(cfg.ssao.composite_mul, Some(0.8));
    assert_eq!(cfg.ssao.bias, Some(0.03));
    assert_eq!(cfg.ssao.samples, Some(16));
    assert_eq!(cfg.ssao.directions, Some(4));
    assert_eq!(cfg.ssao.temporal_alpha, Some(0.3));
    assert_eq!(cfg.ssao.blur_enabled, Some(false));

    assert_eq!(cfg.ssgi.steps, Some(24));
    assert_eq!(cfg.ssgi.radius, Some(1.5));
    assert_eq!(cfg.ssgi.half_res, Some(true));
    assert_eq!(cfg.ssgi.temporal_alpha, Some(0.1));
    assert_eq!(cfg.ssgi.temporal_enabled, Some(true));
    assert_eq!(cfg.ssgi.edges, Some(true));
    assert_eq!(cfg.ssgi.upsample_sigma_depth, Some(0.02));
    assert_eq!(cfg.ssgi.upsample_sigma_normal, Some(0.25));

    assert_eq!(cfg.ssr.enable, Some(true));
    assert_eq!(cfg.ssr.max_steps, Some(48));
    assert_eq!(cfg.ssr.thickness, Some(0.08));

    assert_eq!(cfg.ao_weight, Some(0.6));
    assert_eq!(cfg.ssgi_weight, Some(0.7));
    assert_eq!(cfg.ssr_weight, Some(0.8));
}

#[test]
fn parse_gi_gtao_and_apply_to_mock_viewer() {
    let args = make_args(&["--gi", "gtao:on"]);
    let cfg = GiCliConfig::parse(&args).expect("GI CLI parse failed");

    assert_eq!(
        cfg.entries,
        vec![GiEntry::Effect(GiEffect::Gtao, Toggle::On)],
    );

    let cmds = cfg.to_commands();
    assert!(cmds.iter().any(|c| c == ":gi gtao on"));

    let mock = apply_gi_commands(&cmds);
    assert_eq!(mock.gi_ssao, Some(true));
    assert_eq!(mock.ssao_technique.as_deref(), Some("gtao"));
}

#[test]
fn gi_cli_round_trip_preserves_effects_and_params() {
    let args = vec![
        "--gi".to_string(),
        "ssao:on".to_string(),
        "--gi".to_string(),
        "ssr:off".to_string(),
        "--ssao-radius".to_string(),
        "0.6".to_string(),
        "--ssao-intensity".to_string(),
        "1.3".to_string(),
        "--ssgi-steps".to_string(),
        "12".to_string(),
        "--ssgi-radius".to_string(),
        "0.9".to_string(),
        "--ssr-enable".to_string(),
        "on".to_string(),
        "--ssr-max-steps".to_string(),
        "24".to_string(),
        "--ssr-thickness".to_string(),
        "0.2".to_string(),
        "--ao-weight".to_string(),
        "0.4".to_string(),
        "--ssgi-weight".to_string(),
        "0.5".to_string(),
        "--ssr-weight".to_string(),
        "0.6".to_string(),
        "--viz-gi".to_string(),
        "ao".to_string(),
    ];

    let cfg = GiCliConfig::parse(&args).expect("GI CLI parse failed");
    let cli = cfg.to_cli_string();
    let reparsed = GiCliConfig::parse(&split_cli_string(&cli)).expect("round-trip parse failed");

    assert_eq!(reparsed.entries, cfg.entries);
    assert_eq!(reparsed.ssao.radius, cfg.ssao.radius);
    assert_eq!(reparsed.ssao.intensity, cfg.ssao.intensity);
    assert_eq!(reparsed.ssgi.steps, cfg.ssgi.steps);
    assert_eq!(reparsed.ssgi.radius, cfg.ssgi.radius);
    assert_eq!(reparsed.ssr.enable, cfg.ssr.enable);
    assert_eq!(reparsed.ssr.max_steps, cfg.ssr.max_steps);
    assert_eq!(reparsed.ssr.thickness, cfg.ssr.thickness);
    assert_eq!(reparsed.gi_viz, cfg.gi_viz);

    assert_eq!(reparsed.ao_weight, cfg.ao_weight);
    assert_eq!(reparsed.ssgi_weight, cfg.ssgi_weight);
    assert_eq!(reparsed.ssr_weight, cfg.ssr_weight);
}

#[test]
fn parse_ssgi_alias_upsigma_flags() {
    let args = make_args(&["--ssgi-upsigma-depth", "0.03", "--ssgi-upsigma-normal", "0.40"]);
    let cfg = GiCliConfig::parse(&args).expect("GI CLI parse failed");

    assert_eq!(cfg.ssgi.upsample_sigma_depth, Some(0.03));
    assert_eq!(cfg.ssgi.upsample_sigma_normal, Some(0.40));

    let cli = cfg.to_cli_string();
    let parts = split_cli_string(&cli);
    assert!(parts.windows(2).any(|w| w[0] == "--ssgi-upsample-sigma-depth"));
    assert!(parts.windows(2).any(|w| w[0] == "--ssgi-upsample-sigma-normal"));
}

#[test]
fn to_commands_emits_expected_viewer_commands() {
    let mut cfg = GiCliConfig::default();
    cfg.entries.push(GiEntry::Effect(GiEffect::Ssao, Toggle::On));
    cfg.entries.push(GiEntry::Effect(GiEffect::Ssgi, Toggle::Off));
    cfg.ssao.radius = Some(0.5);
    cfg.ssao.intensity = Some(1.0);
    cfg.ssao.technique = Some("ssao".to_string());
    cfg.ssao.composite_enabled = Some(true);
    cfg.ssgi.steps = Some(16);
    cfg.ssgi.radius = Some(1.2);
    cfg.ssgi.half_res = Some(true);
    cfg.ssr.enable = Some(true);
    cfg.ssr.max_steps = Some(32);
    cfg.ssr.thickness = Some(0.1);
    cfg.gi_viz = Some(GiVizMode::Ao);
    cfg.gi_seed = Some(7);
    cfg.ao_weight = Some(0.4);
    cfg.ssgi_weight = Some(0.5);
    cfg.ssr_weight = Some(0.6);

    let cmds = cfg.to_commands();

    assert!(cmds.iter().any(|c| c == ":gi ssao on"));
    assert!(cmds.iter().any(|c| c == ":ssao-radius 0.500000"));
    assert!(cmds.iter().any(|c| c == ":ssao-intensity 1.000000"));
    assert!(cmds.iter().any(|c| c == ":ssao-technique ssao"));
    assert!(cmds.iter().any(|c| c == ":ssao-composite on"));
    assert!(cmds.iter().any(|c| c == ":ssgi-steps 16"));
    assert!(cmds.iter().any(|c| c == ":ssgi-radius 1.200000"));
    assert!(cmds.iter().any(|c| c == ":ssgi-half on"));
    assert!(cmds.iter().any(|c| c == ":gi ssr on"));
    assert!(cmds.iter().any(|c| c == ":ssr-max-steps 32"));
    assert!(cmds.iter().any(|c| c == ":ssr-thickness 0.100000"));
    assert!(cmds.iter().any(|c| c == ":gi-seed 7"));
    assert!(cmds.iter().any(|c| c == ":viz gi ao"));
    assert!(cmds.iter().any(|c| c == ":ao-weight 0.400000"));
    assert!(cmds.iter().any(|c| c == ":ssgi-weight 0.500000"));
    assert!(cmds.iter().any(|c| c == ":ssr-weight 0.600000"));
}

#[test]
fn gi_off_disables_all_effects_in_commands() {
    let mut cfg = GiCliConfig::default();
    cfg.entries.push(GiEntry::Off);

    let cmds = cfg.to_commands();

    assert!(cmds.iter().any(|c| c == ":gi ssao off"));
    assert!(cmds.iter().any(|c| c == ":gi ssgi off"));
    assert!(cmds.iter().any(|c| c == ":gi ssr off"));
}

#[test]
fn cli_to_commands_to_mock_viewer_config_matches_gi_config() {
    let args = make_args(&[
        "--gi",
        "ssao:on",
        "--gi",
        "ssgi:on",
        "--gi",
        "ssr:on",
        "--ssao-radius",
        "0.7",
        "--ssao-intensity",
        "1.2",
        "--ssao-mul",
        "0.9",
        "--ao-blur",
        "off",
        "--ssgi-steps",
        "18",
        "--ssgi-radius",
        "0.8",
        "--ssgi-half",
        "on",
        "--ssgi-temporal-alpha",
        "0.2",
        "--ssgi-temporal-enable",
        "on",
        "--ssgi-edges",
        "on",
        "--ssgi-upsample-sigma-depth",
        "0.03",
        "--ssgi-upsample-sigma-normal",
        "0.40",
        "--ssr-enable",
        "on",
        "--ssr-max-steps",
        "32",
        "--ssr-thickness",
        "0.15",
        "--ao-weight",
        "0.25",
        "--ssgi-weight",
        "0.5",
        "--ssr-weight",
        "0.75",
        "--viz-gi",
        "ssgi",
        "--gi-seed",
        "99",
    ]);

    let cfg = GiCliConfig::parse(&args).expect("GI CLI parse failed");
    let cmds = cfg.to_commands();
    let mock = apply_gi_commands(&cmds);

    assert_eq!(mock.gi_ssao, Some(true));
    assert_eq!(mock.gi_ssgi, Some(true));
    assert_eq!(mock.gi_ssr, Some(true));

    assert_eq!(mock.ssao_radius, cfg.ssao.radius);
    assert_eq!(mock.ssao_intensity, cfg.ssao.intensity);
    assert_eq!(mock.ssao_mul, cfg.ssao.composite_mul);
    assert_eq!(mock.ao_blur, cfg.ssao.blur_enabled);

    assert_eq!(mock.ssgi_steps, cfg.ssgi.steps);
    assert_eq!(mock.ssgi_radius, cfg.ssgi.radius);
    assert_eq!(mock.ssgi_half, cfg.ssgi.half_res);
    assert_eq!(mock.ssgi_temporal_alpha, cfg.ssgi.temporal_alpha);
    assert_eq!(mock.ssgi_temporal_enabled, cfg.ssgi.temporal_enabled);
    assert_eq!(mock.ssgi_edges, cfg.ssgi.edges);
    assert_eq!(mock.ssgi_upsample_sigma_depth, cfg.ssgi.upsample_sigma_depth);
    assert_eq!(
        mock.ssgi_upsample_sigma_normal,
        cfg.ssgi.upsample_sigma_normal
    );

    assert_eq!(mock.ssr_max_steps, cfg.ssr.max_steps);
    assert_eq!(mock.ssr_thickness, cfg.ssr.thickness);
    assert_eq!(mock.gi_seed, cfg.gi_seed);
    assert_eq!(mock.gi_viz, cfg.gi_viz);

    assert_eq!(mock.ao_weight, cfg.ao_weight);
    assert_eq!(mock.ssgi_weight, cfg.ssgi_weight);
    assert_eq!(mock.ssr_weight, cfg.ssr_weight);
}
