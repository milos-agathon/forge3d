// src/cli/gi_config_parse.rs
// GiCliConfig parsing implementation
// Extracted from args.rs for maintainability (<300 lines)

use super::gi_params::GiCliConfig;
use super::gi_parsing::{
    clamp_ssr_max_steps, clamp_to_positive_with_warning, clamp_with_warning, parse_bool, parse_f32,
    parse_gi_value, parse_u32,
};
use super::gi_types::{GiCliError, GiVizMode};

impl GiCliConfig {
    /// Parse GI-related CLI flags from a flat argument list (excluding argv[0]).
    ///
    /// Unknown non-GI flags are ignored; missing or invalid values for GI flags
    /// return a `GiCliError`.
    pub fn parse(args: &[String]) -> Result<Self, GiCliError> {
        let mut cfg = GiCliConfig::default();
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--gi" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| GiCliError::new("missing value for --gi"))?;
                    let entry = parse_gi_value(value)?;
                    cfg.entries.push(entry);
                    i += 2;
                }
                "--gi-seed" => {
                    let v = parse_u32(args, i, "--gi-seed")?;
                    cfg.gi_seed = Some(v);
                    i += 2;
                }
                "--viz-gi" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| GiCliError::new("missing value for --viz-gi"))?;
                    let mode = GiVizMode::from_str(value).ok_or_else(|| {
                        GiCliError::new(format!(
                            "unknown --viz-gi value '{value}'; expected one of none, composite, ao, ssgi, ssr"
                        ))
                    })?;
                    cfg.gi_viz = Some(mode);
                    i += 2;
                }
                "--ssao-radius" => {
                    let v = parse_f32(args, i, "--ssao-radius")?;
                    if v < 0.0 {
                        eprintln!("[forge3d CLI] clamping --ssao-radius from {} to 0.0", v);
                        cfg.ssao.radius = Some(0.0);
                    } else {
                        cfg.ssao.radius = Some(v);
                    }
                    i += 2;
                }
                "--ssao-intensity" => {
                    let v = parse_f32(args, i, "--ssao-intensity")?;
                    if v < 0.0 {
                        eprintln!("[forge3d CLI] clamping --ssao-intensity from {} to 0.0", v);
                        cfg.ssao.intensity = Some(0.0);
                    } else {
                        cfg.ssao.intensity = Some(v);
                    }
                    i += 2;
                }
                "--ssao-technique" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| GiCliError::new("missing value for --ssao-technique"))?;
                    let key = value.to_ascii_lowercase();
                    if key != "ssao" && key != "gtao" {
                        return Err(GiCliError::new(
                            "unknown --ssao-technique; expected 'ssao' or 'gtao'",
                        ));
                    }
                    cfg.ssao.technique = Some(key);
                    i += 2;
                }
                "--ssao-composite" => {
                    let b = parse_bool(args, i, "--ssao-composite")?;
                    cfg.ssao.composite_enabled = Some(b);
                    i += 2;
                }
                "--ssao-mul" => {
                    let v = parse_f32(args, i, "--ssao-mul")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ssao-mul");
                    cfg.ssao.composite_mul = Some(clamped);
                    i += 2;
                }
                "--ssao-bias" => {
                    let v = parse_f32(args, i, "--ssao-bias")?;
                    cfg.ssao.bias = Some(v);
                    i += 2;
                }
                "--ssao-samples" => {
                    let v = parse_u32(args, i, "--ssao-samples")?;
                    cfg.ssao.samples = Some(v.max(1));
                    i += 2;
                }
                "--ssao-directions" => {
                    let v = parse_u32(args, i, "--ssao-directions")?;
                    cfg.ssao.directions = Some(v.max(1));
                    i += 2;
                }
                "--ssao-temporal-alpha" => {
                    let v = parse_f32(args, i, "--ssao-temporal-alpha")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ssao-temporal-alpha");
                    cfg.ssao.temporal_alpha = Some(clamped);
                    i += 2;
                }
                "--ao-temporal-alpha" => {
                    let v = parse_f32(args, i, "--ao-temporal-alpha")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ao-temporal-alpha");
                    cfg.ssao.temporal_alpha = Some(clamped);
                    i += 2;
                }
                "--ao-blur" => {
                    let b = parse_bool(args, i, "--ao-blur")?;
                    cfg.ssao.blur_enabled = Some(b);
                    i += 2;
                }
                "--ssgi-steps" => {
                    let v = parse_u32(args, i, "--ssgi-steps")?;
                    cfg.ssgi.steps = Some(v);
                    i += 2;
                }
                "--ssgi-radius" => {
                    let v = parse_f32(args, i, "--ssgi-radius")?;
                    if v < 0.0 {
                        eprintln!("[forge3d CLI] clamping --ssgi-radius from {} to 0.0", v);
                        cfg.ssgi.radius = Some(0.0);
                    } else {
                        cfg.ssgi.radius = Some(v);
                    }
                    i += 2;
                }
                "--ssgi-half" => {
                    let b = parse_bool(args, i, "--ssgi-half")?;
                    cfg.ssgi.half_res = Some(b);
                    i += 2;
                }
                "--ssgi-temporal-alpha" => {
                    let v = parse_f32(args, i, "--ssgi-temporal-alpha")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ssgi-temporal-alpha");
                    cfg.ssgi.temporal_alpha = Some(clamped);
                    i += 2;
                }
                "--ssgi-temporal-enable" => {
                    let b = parse_bool(args, i, "--ssgi-temporal-enable")?;
                    cfg.ssgi.temporal_enabled = Some(b);
                    i += 2;
                }
                "--ssgi-edges" => {
                    let b = parse_bool(args, i, "--ssgi-edges")?;
                    cfg.ssgi.edges = Some(b);
                    i += 2;
                }
                "--ssgi-upsigma-depth" | "--ssgi-upsample-sigma-depth" => {
                    let v = parse_f32(args, i, args[i].as_str())?;
                    let clamped = clamp_to_positive_with_warning(v, args[i].as_str());
                    cfg.ssgi.upsample_sigma_depth = Some(clamped);
                    i += 2;
                }
                "--ssgi-upsigma-normal" | "--ssgi-upsample-sigma-normal" => {
                    let v = parse_f32(args, i, args[i].as_str())?;
                    let clamped = clamp_to_positive_with_warning(v, args[i].as_str());
                    cfg.ssgi.upsample_sigma_normal = Some(clamped);
                    i += 2;
                }
                "--ssr-enable" => {
                    let b = parse_bool(args, i, "--ssr-enable")?;
                    cfg.ssr.enable = Some(b);
                    i += 2;
                }
                "--ssr-max-steps" => {
                    let v = parse_u32(args, i, "--ssr-max-steps")?;
                    let clamped = clamp_ssr_max_steps(v);
                    cfg.ssr.max_steps = Some(clamped);
                    i += 2;
                }
                "--ssr-thickness" => {
                    let v = parse_f32(args, i, "--ssr-thickness")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ssr-thickness");
                    cfg.ssr.thickness = Some(clamped);
                    i += 2;
                }
                "--ao-weight" => {
                    let v = parse_f32(args, i, "--ao-weight")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ao-weight");
                    cfg.ao_weight = Some(clamped);
                    i += 2;
                }
                "--ssgi-weight" => {
                    let v = parse_f32(args, i, "--ssgi-weight")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ssgi-weight");
                    cfg.ssgi_weight = Some(clamped);
                    i += 2;
                }
                "--ssr-weight" => {
                    let v = parse_f32(args, i, "--ssr-weight")?;
                    let clamped = clamp_with_warning(v, 0.0, 1.0, "--ssr-weight");
                    cfg.ssr_weight = Some(clamped);
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }
        Ok(cfg)
    }
}
