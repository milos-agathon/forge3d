// src/cli/gi_parsing.rs
// GI CLI parsing helper functions
// Extracted from args.rs for maintainability (<300 lines)

use super::gi_types::{GiCliError, GiEffect, GiEntry, Toggle};
use crate::render::params::SsrParams;

/// Parse a --gi value into a GiEntry.
pub fn parse_gi_value(value: &str) -> Result<GiEntry, GiCliError> {
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

/// Parse a float argument value.
pub fn parse_f32(args: &[String], idx: usize, flag: &str) -> Result<f32, GiCliError> {
    let raw = args
        .get(idx + 1)
        .ok_or_else(|| GiCliError::new(format!("missing value for {flag}")))?;
    raw.parse::<f32>()
        .map_err(|_| GiCliError::new(format!("invalid float value '{raw}' for {flag}")))
}

/// Parse an unsigned integer argument value.
pub fn parse_u32(args: &[String], idx: usize, flag: &str) -> Result<u32, GiCliError> {
    let raw = args
        .get(idx + 1)
        .ok_or_else(|| GiCliError::new(format!("missing value for {flag}")))?;
    raw.parse::<u32>()
        .map_err(|_| GiCliError::new(format!("invalid integer value '{raw}' for {flag}")))
}

/// Parse a boolean argument value.
pub fn parse_bool(args: &[String], idx: usize, flag: &str) -> Result<bool, GiCliError> {
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

/// Clamp a value to a range with a warning message.
pub fn clamp_with_warning(value: f32, min: f32, max: f32, flag: &str) -> f32 {
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

/// Clamp a value to positive with a warning message.
pub fn clamp_to_positive_with_warning(value: f32, flag: &str) -> f32 {
    if value <= 0.0 {
        eprintln!("[forge3d CLI] clamping {flag} from {} to 1e-4", value);
        1e-4
    } else {
        value
    }
}

/// Clamp SSR max steps using SsrParams validation.
pub fn clamp_ssr_max_steps(steps: u32) -> u32 {
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
