// src/cli/args.rs
// GI CLI configuration and argument parsing
// Refactored: types, params, parsing, formatting extracted to submodules

// Re-exports for backward compatibility
pub use super::gi_formatting::{format_bool_word, format_float};
pub use super::gi_params::{GiCliConfig, SsaoCliParams, SsgiCliParams, SsrCliParams};
pub use super::gi_parsing::{
    clamp_ssr_max_steps, clamp_to_positive_with_warning, clamp_with_warning, parse_bool, parse_f32,
    parse_gi_value, parse_u32,
};
pub use super::gi_types::{GiCliError, GiEffect, GiEntry, GiVizMode, Toggle};

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
        assert_eq!(cfg.entries[0], GiEntry::Effect(GiEffect::Ssao, Toggle::On));
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
        let args = vec!["--viz-gi".to_string(), "ao".to_string()];
        let cfg = GiCliConfig::parse(&args).unwrap();
        assert_eq!(cfg.gi_viz, Some(GiVizMode::Ao));
    }

    #[test]
    fn round_trip_viz_gi_cli_string() {
        let args = vec!["--viz-gi".to_string(), "composite".to_string()];
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
        assert!(cli
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| { w[0] == "--viz-gi" && w[1] == "ao" }));

        let cmds = cfg.to_commands();
        assert!(cmds.iter().any(|c| c == ":viz gi ao"));
    }

    #[test]
    fn reject_invalid_viz_gi_mode() {
        let args = vec!["--viz-gi".to_string(), "foo".to_string()];
        let err = GiCliConfig::parse(&args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown --viz-gi value"));
        assert!(msg.contains("none, composite, ao, ssgi, ssr"));
    }
}
