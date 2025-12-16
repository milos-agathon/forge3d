// src/cli/gi_config_output.rs
// GiCliConfig serialization methods (to_commands, to_cli_string)
// Extracted from args.rs for maintainability (<300 lines)

use super::gi_formatting::{format_bool_word, format_float};
use super::gi_params::GiCliConfig;
use super::gi_types::GiEntry;

impl GiCliConfig {
    /// Serialize this configuration into a list of canonical viewer colon
    /// commands (e.g. ":gi ssao on", ":ssao-radius 0.500000").
    pub fn to_commands(&self) -> Vec<String> {
        let mut cmds: Vec<String> = Vec::new();

        // High-level GI mode toggles
        for entry in &self.entries {
            match entry {
                GiEntry::Off => {
                    cmds.push(":gi ssao off".to_string());
                    cmds.push(":gi ssgi off".to_string());
                    cmds.push(":gi ssr off".to_string());
                }
                GiEntry::Effect(effect, toggle) => {
                    let state = toggle.as_str();
                    cmds.push(format!(":gi {} {}", effect.as_str(), state));
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
            cmds.push(format!(":viz gi {}", mode.as_str()));
        }

        cmds
    }

    /// Serialize this configuration into a canonical CLI flag string.
    pub fn to_cli_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for entry in &self.entries {
            match entry {
                GiEntry::Off => {
                    parts.push("--gi".to_string());
                    parts.push("off".to_string());
                }
                GiEntry::Effect(effect, toggle) => {
                    parts.push("--gi".to_string());
                    parts.push(format!("{}:{}", effect.as_str(), toggle.as_str()));
                }
            }
        }
        push_ssao_params(&mut parts, &self.ssao);
        push_ssgi_params(&mut parts, &self.ssgi);
        push_ssr_params(&mut parts, &self.ssr);
        push_weights(
            &mut parts,
            self.ao_weight,
            self.ssgi_weight,
            self.ssr_weight,
        );
        if let Some(mode) = self.gi_viz {
            parts.push("--viz-gi".to_string());
            parts.push(mode.as_str().to_string());
        }
        if let Some(seed) = self.gi_seed {
            parts.push("--gi-seed".to_string());
            parts.push(seed.to_string());
        }
        parts.join(" ")
    }
}

fn push_ssao_params(parts: &mut Vec<String>, ssao: &super::gi_params::SsaoCliParams) {
    if let Some(v) = ssao.radius {
        parts.push("--ssao-radius".to_string());
        parts.push(format_float(v));
    }
    if let Some(v) = ssao.intensity {
        parts.push("--ssao-intensity".to_string());
        parts.push(format_float(v));
    }
    if let Some(ref t) = ssao.technique {
        parts.push("--ssao-technique".to_string());
        parts.push(t.clone());
    }
    if let Some(b) = ssao.composite_enabled {
        parts.push("--ssao-composite".to_string());
        parts.push(format_bool_word(b).to_string());
    }
    if let Some(v) = ssao.composite_mul {
        parts.push("--ssao-mul".to_string());
        parts.push(format_float(v));
    }
    if let Some(v) = ssao.bias {
        parts.push("--ssao-bias".to_string());
        parts.push(format_float(v));
    }
    if let Some(v) = ssao.samples {
        parts.push("--ssao-samples".to_string());
        parts.push(v.to_string());
    }
    if let Some(v) = ssao.directions {
        parts.push("--ssao-directions".to_string());
        parts.push(v.to_string());
    }
    if let Some(v) = ssao.temporal_alpha {
        parts.push("--ssao-temporal-alpha".to_string());
        parts.push(format_float(v));
    }
    if let Some(b) = ssao.blur_enabled {
        parts.push("--ao-blur".to_string());
        parts.push(format_bool_word(b).to_string());
    }
}

fn push_ssgi_params(parts: &mut Vec<String>, ssgi: &super::gi_params::SsgiCliParams) {
    if let Some(v) = ssgi.steps {
        parts.push("--ssgi-steps".to_string());
        parts.push(v.to_string());
    }
    if let Some(v) = ssgi.radius {
        parts.push("--ssgi-radius".to_string());
        parts.push(format_float(v));
    }
    if let Some(b) = ssgi.half_res {
        parts.push("--ssgi-half".to_string());
        parts.push(format_bool_word(b).to_string());
    }
    if let Some(v) = ssgi.temporal_alpha {
        parts.push("--ssgi-temporal-alpha".to_string());
        parts.push(format_float(v));
    }
    if let Some(b) = ssgi.edges {
        parts.push("--ssgi-edges".to_string());
        parts.push(format_bool_word(b).to_string());
    }
    if let Some(v) = ssgi.upsample_sigma_depth {
        parts.push("--ssgi-upsample-sigma-depth".to_string());
        parts.push(format_float(v));
    }
    if let Some(v) = ssgi.upsample_sigma_normal {
        parts.push("--ssgi-upsample-sigma-normal".to_string());
        parts.push(format_float(v));
    }
}

fn push_ssr_params(parts: &mut Vec<String>, ssr: &super::gi_params::SsrCliParams) {
    if let Some(b) = ssr.enable {
        parts.push("--ssr-enable".to_string());
        parts.push(format_bool_word(b).to_string());
    }
    if let Some(v) = ssr.max_steps {
        parts.push("--ssr-max-steps".to_string());
        parts.push(v.to_string());
    }
    if let Some(v) = ssr.thickness {
        parts.push("--ssr-thickness".to_string());
        parts.push(format_float(v));
    }
}

fn push_weights(
    parts: &mut Vec<String>,
    ao_weight: Option<f32>,
    ssgi_weight: Option<f32>,
    ssr_weight: Option<f32>,
) {
    if let Some(v) = ao_weight {
        parts.push("--ao-weight".to_string());
        parts.push(format_float(v));
    }
    if let Some(v) = ssgi_weight {
        parts.push("--ssgi-weight".to_string());
        parts.push(format_float(v));
    }
    if let Some(v) = ssr_weight {
        parts.push("--ssr-weight".to_string());
        parts.push(format_float(v));
    }
}
