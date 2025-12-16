// src/cli/gi_formatting.rs
// GI CLI formatting helper functions
// Extracted from args.rs for maintainability (<300 lines)

/// Format a boolean as "on"/"off" for CLI output.
pub fn format_bool_word(v: bool) -> &'static str {
    if v {
        "on"
    } else {
        "off"
    }
}

/// Format a float with 6 decimal places for CLI output.
pub fn format_float(v: f32) -> String {
    format!("{:.6}", v)
}

/// Push a flag and its f32 value to a CLI args list if the value is Some.
#[allow(dead_code)]
pub fn push_opt_f32(parts: &mut Vec<String>, flag: &str, value: Option<f32>) {
    if let Some(v) = value {
        parts.push(flag.to_string());
        parts.push(format_float(v));
    }
}

/// Push a flag and its u32 value to a CLI args list if the value is Some.
#[allow(dead_code)]
pub fn push_opt_u32(parts: &mut Vec<String>, flag: &str, value: Option<u32>) {
    if let Some(v) = value {
        parts.push(flag.to_string());
        parts.push(v.to_string());
    }
}

/// Push a flag and its bool value (as "on"/"off") to a CLI args list if the value is Some.
#[allow(dead_code)]
pub fn push_opt_bool(parts: &mut Vec<String>, flag: &str, value: Option<bool>) {
    if let Some(v) = value {
        parts.push(flag.to_string());
        parts.push(format_bool_word(v).to_string());
    }
}

/// Push a flag and its string value to a CLI args list if the value is Some.
#[allow(dead_code)]
pub fn push_opt_str(parts: &mut Vec<String>, flag: &str, value: &Option<String>) {
    if let Some(ref v) = value {
        parts.push(flag.to_string());
        parts.push(v.clone());
    }
}
