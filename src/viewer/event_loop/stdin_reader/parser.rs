mod environment;
mod gi;
mod render;

use crate::viewer::viewer_enums::ViewerCmd;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TextCommandCapabilities {
    allow_terrain_load: bool,
}

impl TextCommandCapabilities {
    pub(crate) const fn startup() -> Self {
        Self {
            allow_terrain_load: true,
        }
    }

    pub(crate) const fn runtime() -> Self {
        Self {
            allow_terrain_load: true,
        }
    }
}

#[cfg(test)]
pub(crate) fn parse_stdin_command(line: &str) -> Option<Vec<ViewerCmd>> {
    translate_text_command(line, TextCommandCapabilities::runtime())
        .ok()
        .flatten()
}

/// Single text-command ingress for startup and stdin. The command token is
/// normalized while path spelling remains byte-for-byte intact.
pub(crate) fn translate_text_command(
    line: &str,
    capabilities: TextCommandCapabilities,
) -> Result<Option<Vec<ViewerCmd>>, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let split = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
    let normalized = format!("{}{}", trimmed[..split].to_lowercase(), &trimmed[split..]);
    let line = normalized.as_str();

    if matches!(line.split_whitespace().next(), Some(":terrain" | "terrain")) {
        if !capabilities.allow_terrain_load {
            return Err("terrain loading is not enabled for this command ingress".to_string());
        }
        let path = line
            .split_once(char::is_whitespace)
            .map(|(_, path)| path.trim())
            .filter(|path| !path.is_empty())
            .ok_or_else(|| "terrain command requires a raster path".to_string())?;
        crate::viewer::terrain::ViewerTerrainScene::preflight_terrain_path(path)
            .map_err(|err| err.to_string())?;
        return Ok(Some(vec![ViewerCmd::LoadTerrain(path.to_string())]));
    }

    if let Some(cmds) = gi::parse_gi_command(line) {
        return Ok(Some(cmds));
    }
    if let Some(cmds) = render::parse_render_command(line) {
        return Ok(Some(cmds));
    }
    if let Some(cmds) = environment::parse_environment_command(line) {
        return Ok(Some(cmds));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_tiff_path() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("Forge3D-M06-Case-{}.tif", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let mut encoder = tiff::encoder::TiffEncoder::new(file).unwrap();
        let image = encoder
            .new_image::<tiff::encoder::colortype::Gray32Float>(2, 2)
            .unwrap();
        image.write_data(&[0.0, 1.0, 2.0, 3.0]).unwrap();
        path
    }

    #[test]
    fn startup_and_runtime_return_identical_terrain_rejection() {
        let missing = std::env::temp_dir().join("Forge3D-M06-Missing-Case.tif");
        let payload = format!(":TeRrAiN {}", missing.display());
        let startup =
            translate_text_command(&payload, TextCommandCapabilities::startup()).unwrap_err();
        let runtime =
            translate_text_command(&payload, TextCommandCapabilities::runtime()).unwrap_err();
        assert_eq!(startup, runtime);
        assert!(startup.contains("Forge3D-M06-Missing-Case.tif"));
    }

    #[test]
    fn command_token_normalization_preserves_path_case_exactly() {
        let path = plain_tiff_path();
        let payload = format!(":TeRrAiN {}", path.display());
        for capabilities in [
            TextCommandCapabilities::startup(),
            TextCommandCapabilities::runtime(),
        ] {
            let commands = translate_text_command(&payload, capabilities)
                .unwrap()
                .unwrap();
            assert!(matches!(
                commands.as_slice(),
                [ViewerCmd::LoadTerrain(actual)] if actual == path.to_str().unwrap()
            ));
        }
        let _ = std::fs::remove_file(path);
    }
}
