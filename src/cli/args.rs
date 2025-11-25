use std::fmt;

use crate::render::params::SsrParams;

/// Toggle state used by GI-related CLI flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Toggle {
    On,
    Off,
}

impl Toggle {
    fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "on" | "1" | "true" => Some(Toggle::On),
            "off" | "0" | "false" => Some(Toggle::Off),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Toggle::On => "on",
            Toggle::Off => "off",
        }
    }
}

/// GI effects supported by CLI flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GiEffect {
    Ssao,
    Ssgi,
    Ssr,
    Gtao,
}

/// Parsed `--gi` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GiEntry {
    Off,
    Effect(GiEffect, Toggle),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GiVizMode {
    None,
    Composite,
    Ao,
    Ssgi,
    Ssr,
}

/// SSAO-related CLI parameters.
#[derive(Clone, Debug, Default)]
pub struct SsaoCliParams {
    pub radius: Option<f32>,
    pub intensity: Option<f32>,
    pub technique: Option<String>,
    pub composite_enabled: Option<bool>,
    pub composite_mul: Option<f32>,
    pub bias: Option<f32>,
    pub samples: Option<u32>,
    pub directions: Option<u32>,
    pub temporal_alpha: Option<f32>,
    pub temporal_enabled: Option<bool>,
    pub blur_enabled: Option<bool>,
}

/// SSGI-related CLI parameters.
#[derive(Clone, Debug, Default)]
pub struct SsgiCliParams {
    pub steps: Option<u32>,
    pub radius: Option<f32>,
    pub half_res: Option<bool>,
    pub temporal_alpha: Option<f32>,
    pub temporal_enabled: Option<bool>,
    pub edges: Option<bool>,
    pub upsample_sigma_depth: Option<f32>,
    pub upsample_sigma_normal: Option<f32>,
}

/// SSR-related CLI parameters.
#[derive(Clone, Debug, Default)]
pub struct SsrCliParams {
    pub enable: Option<bool>,
    pub max_steps: Option<u32>,
    pub thickness: Option<f32>,
}

/// Aggregated GI CLI configuration.
#[derive(Clone, Debug, Default)]
pub struct GiCliConfig {
    pub entries: Vec<GiEntry>,
    pub ssao: SsaoCliParams,
    pub ssgi: SsgiCliParams,
    pub ssr: SsrCliParams,
    pub ao_weight: Option<f32>,
    pub ssgi_weight: Option<f32>,
    pub ssr_weight: Option<f32>,
    pub gi_viz: Option<GiVizMode>,
    pub gi_seed: Option<u32>,
}

/// Error raised when parsing GI CLI flags.
#[derive(Debug)]
pub struct GiCliError {
    msg: String,
}

impl GiCliError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl fmt::Display for GiCliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.msg.fmt(f)
    }
}

impl std::error::Error for GiCliError {}

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
                    let mode = match value.to_ascii_lowercase().as_str() {
                        "none" => GiVizMode::None,
                        "composite" => GiVizMode::Composite,
                        "ao" => GiVizMode::Ao,
                        "ssgi" => GiVizMode::Ssgi,
                        "ssr" => GiVizMode::Ssr,
                        other => {
                            return Err(GiCliError::new(format!(
                                "unknown --viz-gi value '{other}'; expected one of none, composite, ao, ssgi, ssr"
                            )));
                        }
                    };
                    cfg.gi_viz = Some(mode);
                    i += 2;
                }
                "--ssao-radius" => {
                    let v = parse_f32(args, i, "--ssao-radius")?;
                    if v < 0.0 {
                        eprintln!(
                            "[forge3d CLI] clamping --ssao-radius from {} to 0.0",
                            v
                        );
                        cfg.ssao.radius = Some(0.0);
                    } else {
                        cfg.ssao.radius = Some(v);
                    }
                    i += 2;
                }
                "--ssao-intensity" => {
                    let v = parse_f32(args, i, "--ssao-intensity")?;
                    if v < 0.0 {
                        eprintln!(
                            "[forge3d CLI] clamping --ssao-intensity from {} to 0.0",
                            v
                        );
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
                        eprintln!(
                            "[forge3d CLI] clamping --ssgi-radius from {} to 0.0",
                            v
                        );
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

    /// Serialize this configuration into a list of canonical viewer colon
    /// commands (e.g. ":gi ssao on", ":ssao-radius 0.500000").
    ///
    /// This is used by the interactive viewer example to seed its initial GI
    /// state from CLI flags while keeping semantics in one place.
    pub fn to_commands(&self) -> Vec<String> {
        let mut cmds: Vec<String> = Vec::new();

        // High-level GI mode toggles
        for entry in &self.entries {
            match entry {
                GiEntry::Off => {
                    // Disable all GI effects
                    cmds.push(":gi ssao off".to_string());
                    cmds.push(":gi ssgi off".to_string());
                    cmds.push(":gi ssr off".to_string());
                }
                GiEntry::Effect(effect, toggle) => {
                    let state = toggle.as_str();
                    match effect {
                        GiEffect::Ssao => cmds.push(format!(":gi ssao {}", state)),
                        GiEffect::Ssgi => cmds.push(format!(":gi ssgi {}", state)),
                        GiEffect::Ssr => cmds.push(format!(":gi ssr {}", state)),
                        GiEffect::Gtao => cmds.push(format!(":gi gtao {}", state)),
                    }
                }
            }
        }

        // SSAO-related parameters
        if let Some(v) = self.ssao.radius {
            cmds.push(format!(":ssao-radius {}", format_float(v)));
        }
        if let Some(v) = self.ssao.intensity {
            cmds.push(format!(":ssao-intensity {}", format_float(v)));
        }
        if let Some(ref t) = self.ssao.technique {
            cmds.push(format!(":ssao-technique {}", t));
        }
        if let Some(b) = self.ssao.composite_enabled {
            cmds.push(format!(":ssao-composite {}", format_bool_word(b)));
        }
        if let Some(v) = self.ssao.composite_mul {
            cmds.push(format!(":ssao-mul {}", format_float(v)));
        }
        if let Some(v) = self.ssao.bias {
            cmds.push(format!(":ssao-bias {}", format_float(v)));
        }
        if let Some(v) = self.ssao.samples {
            cmds.push(format!(":ssao-samples {}", v));
        }
        if let Some(v) = self.ssao.directions {
            cmds.push(format!(":ssao-directions {}", v));
        }
        if let Some(v) = self.ssao.temporal_alpha {
            let val = format_float(v);
            // Expose both SSAO-specific and AO-agnostic aliases.
            cmds.push(format!(":ssao-temporal-alpha {}", val));
            cmds.push(format!(":ao-temporal-alpha {}", val));
        }
        if let Some(b) = self.ssao.temporal_enabled {
            cmds.push(format!(":ssao-temporal {}", format_bool_word(b)));
        }
        if let Some(b) = self.ssao.blur_enabled {
            cmds.push(format!(":ao-blur {}", format_bool_word(b)));
        }

        // SSGI-related parameters
        if let Some(v) = self.ssgi.steps {
            cmds.push(format!(":ssgi-steps {}", v));
        }
        if let Some(v) = self.ssgi.radius {
            cmds.push(format!(":ssgi-radius {}", format_float(v)));
        }
        if let Some(b) = self.ssgi.half_res {
            cmds.push(format!(":ssgi-half {}", format_bool_word(b)));
        }
        if let Some(v) = self.ssgi.temporal_alpha {
            cmds.push(format!(":ssgi-temporal-alpha {}", format_float(v)));
        }
        if let Some(b) = self.ssgi.temporal_enabled {
            cmds.push(format!(":ssgi-temporal {}", format_bool_word(b)));
        }
        if let Some(b) = self.ssgi.edges {
            cmds.push(format!(":ssgi-edges {}", format_bool_word(b)));
        }
        if let Some(v) = self.ssgi.upsample_sigma_depth {
            cmds.push(format!(":ssgi-upsample-sigma-depth {}", format_float(v)));
        }
        if let Some(v) = self.ssgi.upsample_sigma_normal {
            cmds.push(format!(":ssgi-upsample-sigma-normal {}", format_float(v)));
        }

        // SSR-related parameters
        if let Some(b) = self.ssr.enable {
            cmds.push(format!(":gi ssr {}", format_bool_word(b)));
        }
        if let Some(v) = self.ssr.max_steps {
            cmds.push(format!(":ssr-max-steps {}", v));
        }
        if let Some(v) = self.ssr.thickness {
            cmds.push(format!(":ssr-thickness {}", format_float(v)));
        }

        if let Some(v) = self.ao_weight {
            cmds.push(format!(":ao-weight {}", format_float(v)));
        }
        if let Some(v) = self.ssgi_weight {
            cmds.push(format!(":ssgi-weight {}", format_float(v)));
        }
        if let Some(v) = self.ssr_weight {
            cmds.push(format!(":ssr-weight {}", format_float(v)));
        }

        if let Some(seed) = self.gi_seed {
            cmds.push(format!(":gi-seed {}", seed));
        }

        if let Some(mode) = self.gi_viz {
            let name = match mode {
                GiVizMode::None => "none",
                GiVizMode::Composite => "composite",
                GiVizMode::Ao => "ao",
                GiVizMode::Ssgi => "ssgi",
                GiVizMode::Ssr => "ssr",
            };
            cmds.push(format!(":viz gi {}", name));
        }

        cmds
    }

    /// Serialize this configuration into a canonical CLI flag string.
    ///
    /// This is primarily intended for round-trip testing.
    pub fn to_cli_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for entry in &self.entries {
            match entry {
                GiEntry::Off => {
                    parts.push("--gi".to_string());
                    parts.push("off".to_string());
                }
                GiEntry::Effect(effect, toggle) => {
                    let name = match effect {
                        GiEffect::Ssao => "ssao",
                        GiEffect::Ssgi => "ssgi",
                        GiEffect::Ssr => "ssr",
                        GiEffect::Gtao => "gtao",
                    };
                    parts.push("--gi".to_string());
                    parts.push(format!("{name}:{}", toggle.as_str()));
                }
            }
        }
        if let Some(v) = self.ssao.radius {
            parts.push("--ssao-radius".to_string());
            parts.push(format_float(v));
        }
        if let Some(v) = self.ssao.intensity {
            parts.push("--ssao-intensity".to_string());
            parts.push(format_float(v));
        }
        if let Some(ref t) = self.ssao.technique {
            parts.push("--ssao-technique".to_string());
            parts.push(t.clone());
        }
        if let Some(b) = self.ssao.composite_enabled {
            parts.push("--ssao-composite".to_string());
            parts.push(format_bool_word(b).to_string());
        }
        if let Some(v) = self.ssao.composite_mul {
            parts.push("--ssao-mul".to_string());
            parts.push(format_float(v));
        }
        if let Some(v) = self.ssao.bias {
            parts.push("--ssao-bias".to_string());
            parts.push(format_float(v));
        }
        if let Some(v) = self.ssao.samples {
            parts.push("--ssao-samples".to_string());
            parts.push(v.to_string());
        }
        if let Some(v) = self.ssao.directions {
            parts.push("--ssao-directions".to_string());
            parts.push(v.to_string());
        }
        if let Some(v) = self.ssao.temporal_alpha {
            parts.push("--ssao-temporal-alpha".to_string());
            parts.push(format_float(v));
        }
        if let Some(b) = self.ssao.blur_enabled {
            parts.push("--ao-blur".to_string());
            parts.push(format_bool_word(b).to_string());
        }
        if let Some(v) = self.ssgi.steps {
            parts.push("--ssgi-steps".to_string());
            parts.push(v.to_string());
        }
        if let Some(v) = self.ssgi.radius {
            parts.push("--ssgi-radius".to_string());
            parts.push(format_float(v));
        }
        if let Some(b) = self.ssgi.half_res {
            parts.push("--ssgi-half".to_string());
            parts.push(format_bool_word(b).to_string());
        }
        if let Some(v) = self.ssgi.temporal_alpha {
            parts.push("--ssgi-temporal-alpha".to_string());
            parts.push(format_float(v));
        }
        if let Some(b) = self.ssgi.edges {
            parts.push("--ssgi-edges".to_string());
            parts.push(format_bool_word(b).to_string());
        }
        if let Some(v) = self.ssgi.upsample_sigma_depth {
            parts.push("--ssgi-upsample-sigma-depth".to_string());
            parts.push(format_float(v));
        }
        if let Some(v) = self.ssgi.upsample_sigma_normal {
            parts.push("--ssgi-upsample-sigma-normal".to_string());
            parts.push(format_float(v));
        }
        if let Some(b) = self.ssr.enable {
            parts.push("--ssr-enable".to_string());
            parts.push(format_bool_word(b).to_string());
        }
        if let Some(v) = self.ssr.max_steps {
            parts.push("--ssr-max-steps".to_string());
            parts.push(v.to_string());
        }
        if let Some(v) = self.ssr.thickness {
            parts.push("--ssr-thickness".to_string());
            parts.push(format_float(v));
        }
        if let Some(v) = self.ao_weight {
            parts.push("--ao-weight".to_string());
            parts.push(format_float(v));
        }
        if let Some(v) = self.ssgi_weight {
            parts.push("--ssgi-weight".to_string());
            parts.push(format_float(v));
        }
        if let Some(v) = self.ssr_weight {
            parts.push("--ssr-weight".to_string());
            parts.push(format_float(v));
        }
        if let Some(mode) = self.gi_viz {
            let name = match mode {
                GiVizMode::None => "none",
                GiVizMode::Composite => "composite",
                GiVizMode::Ao => "ao",
                GiVizMode::Ssgi => "ssgi",
                GiVizMode::Ssr => "ssr",
            };
            parts.push("--viz-gi".to_string());
            parts.push(name.to_string());
        }
        if let Some(seed) = self.gi_seed {
            parts.push("--gi-seed".to_string());
            parts.push(seed.to_string());
        }
        parts.join(" ")
    }
}

fn parse_gi_value(value: &str) -> Result<GiEntry, GiCliError> {
    let v = value.trim();
    if v.eq_ignore_ascii_case("off") {
        return Ok(GiEntry::Off);
    }
    let (mode_str, toggle) = if let Some((mode, state)) = v.split_once(':') {
        let t = Toggle::from_str(state).ok_or_else(|| {
            GiCliError::new(format!(
                "invalid --gi state '{state}'; expected 'on' or 'off'"
            ))
        })?;
        (mode, t)
    } else {
        (v, Toggle::On)
    };
    let normalized = mode_str.to_ascii_lowercase();
    let effect = match normalized.as_str() {
        "ssao" => GiEffect::Ssao,
        "ssgi" => GiEffect::Ssgi,
        "ssr" => GiEffect::Ssr,
        "gtao" => GiEffect::Gtao,
        other => {
            return Err(GiCliError::new(format!(
                "unknown --gi value '{other}'; expected one of ssao, ssgi, ssr, gtao, off"
            )));
        }
    };
    Ok(GiEntry::Effect(effect, toggle))
}

fn parse_f32(args: &[String], idx: usize, flag: &str) -> Result<f32, GiCliError> {
    let raw = args
        .get(idx + 1)
        .ok_or_else(|| GiCliError::new(format!("missing value for {flag}")))?;
    raw.parse::<f32>()
        .map_err(|_| GiCliError::new(format!("invalid float value '{raw}' for {flag}")))
}

fn parse_u32(args: &[String], idx: usize, flag: &str) -> Result<u32, GiCliError> {
    let raw = args
        .get(idx + 1)
        .ok_or_else(|| GiCliError::new(format!("missing value for {flag}")))?;
    raw.parse::<u32>()
        .map_err(|_| GiCliError::new(format!("invalid integer value '{raw}' for {flag}")))
}

fn parse_bool(args: &[String], idx: usize, flag: &str) -> Result<bool, GiCliError> {
    let raw = args
        .get(idx + 1)
        .ok_or_else(|| GiCliError::new(format!("missing value for {flag}")))?;
    match raw.to_ascii_lowercase().as_str() {
        "on" | "1" | "true" | "yes" => Ok(true),
        "off" | "0" | "false" | "no" => Ok(false),
        other => Err(GiCliError::new(format!(
            "invalid boolean value '{other}' for {flag}; expected on/off or true/false"
        ))),
    }
}

fn clamp_with_warning(value: f32, min: f32, max: f32, flag: &str) -> f32 {
    if value < min || value > max {
        let clamped = value.clamp(min, max);
        eprintln!(
            "[forge3d CLI] clamping {flag} from {} to {}",
            value, clamped
        );
        clamped
    } else {
        value
    }
}

fn clamp_to_positive_with_warning(value: f32, flag: &str) -> f32 {
    if value <= 0.0 {
        eprintln!("[forge3d CLI] clamping {flag} from {} to 1e-4", value);
        1e-4
    } else {
        value
    }
}

fn clamp_ssr_max_steps(steps: u32) -> u32 {
    let mut params = SsrParams::default();
    params.set_max_steps(steps);
    if params.ssr_max_steps != steps {
        eprintln!(
            "[forge3d CLI] clamping --ssr-max-steps from {} to {}",
            steps, params.ssr_max_steps
        );
    }
    params.ssr_max_steps
}

fn format_bool_word(v: bool) -> &'static str {
    if v { "on" } else { "off" }
}

fn format_float(v: f32) -> String {
    format!("{:.6}", v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_gi_and_ssao() {
        let args = vec![
            "--gi".to_string(),
            "ssao:on".to_string(),
            "--ssao-radius".to_string(),
            "0.5".to_string(),
            "--ssao-intensity".to_string(),
            "1.0".to_string(),
        ];
        let cfg = GiCliConfig::parse(&args).unwrap();
        assert_eq!(cfg.entries.len(), 1);
        assert_eq!(
            cfg.entries[0],
            GiEntry::Effect(GiEffect::Ssao, Toggle::On)
        );
        assert_eq!(cfg.ssao.radius, Some(0.5));
        assert_eq!(cfg.ssao.intensity, Some(1.0));
    }

    #[test]
    fn reject_invalid_gi_mode() {
        let args = vec!["--gi".to_string(), "xyz".to_string()];
        let err = GiCliConfig::parse(&args).unwrap_err();
        assert!(err.to_string().contains("unknown --gi value"));
    }

    #[test]
    fn round_trip_cli_string() {
        let args = vec![
            "--gi".to_string(),
            "ssr:on".to_string(),
            "--ssr-max-steps".to_string(),
            "24".to_string(),
            "--ssr-thickness".to_string(),
            "0.2".to_string(),
        ];
        let cfg = GiCliConfig::parse(&args).unwrap();
        let s = cfg.to_cli_string();
        let reparsed_args: Vec<String> = s.split_whitespace().map(|s| s.to_string()).collect();
        let cfg2 = GiCliConfig::parse(&reparsed_args).unwrap();
        assert_eq!(cfg2.ssr.max_steps, cfg.ssr.max_steps);
        assert_eq!(cfg2.ssr.thickness, cfg.ssr.thickness);
        assert_eq!(cfg2.entries, cfg.entries);
    }

    #[test]
    fn parse_viz_gi_valid() {
        let args = vec![
            "--viz-gi".to_string(),
            "ao".to_string(),
        ];
        let cfg = GiCliConfig::parse(&args).unwrap();
        assert_eq!(cfg.gi_viz, Some(GiVizMode::Ao));
    }

    #[test]
    fn round_trip_viz_gi_cli_string() {
        let args = vec![
            "--viz-gi".to_string(),
            "composite".to_string(),
        ];
        let cfg = GiCliConfig::parse(&args).unwrap();
        let s = cfg.to_cli_string();
        let reparsed_args: Vec<String> = s.split_whitespace().map(|s| s.to_string()).collect();
        let cfg2 = GiCliConfig::parse(&reparsed_args).unwrap();
        assert_eq!(cfg2.gi_viz, cfg.gi_viz);
    }

    #[test]
    fn viz_gi_to_cli_and_commands() {
        let mut cfg = GiCliConfig::default();
        cfg.gi_viz = Some(GiVizMode::Ao);

        let cli = cfg.to_cli_string();
        assert!(cli.split_whitespace().collect::<Vec<_>>().windows(2).any(|w| {
            w[0] == "--viz-gi" && w[1] == "ao"
        }));

        let cmds = cfg.to_commands();
        assert!(cmds.iter().any(|c| c == ":viz gi ao"));
    }

    #[test]
    fn reject_invalid_viz_gi_mode() {
        let args = vec![
            "--viz-gi".to_string(),
            "foo".to_string(),
        ];
        let err = GiCliConfig::parse(&args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown --viz-gi value"));
        assert!(msg.contains("none, composite, ao, ssgi, ssr"));
    }
}
