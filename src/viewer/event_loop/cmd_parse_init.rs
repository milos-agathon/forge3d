//! Startup command translation through the same parser used by stdin.

use crate::viewer::viewer_enums::ViewerCmd;

use super::stdin_reader::{translate_text_command, TextCommandCapabilities};

/// Parse startup commands with explicit startup capabilities. Invalid
/// metadata is returned to the caller instead of being silently dropped.
pub fn parse_initial_commands(cmds: &[String]) -> Result<Vec<ViewerCmd>, String> {
    let capabilities = TextCommandCapabilities::startup();
    let mut pending = Vec::new();
    for raw in cmds {
        if let Some(commands) = translate_text_command(raw, capabilities)? {
            pending.extend(commands);
        }
    }
    Ok(pending)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_uses_runtime_translator_for_fog_controls() {
        let commands =
            parse_initial_commands(&[":fog on".to_string(), ":fog-temporal 0.8".to_string()])
                .unwrap();
        assert!(
            matches!(commands.as_slice(), [ViewerCmd::FogToggle(true), ViewerCmd::FogSetTemporal(alpha)] if (*alpha - 0.8).abs() < f32::EPSILON)
        );
    }
}
