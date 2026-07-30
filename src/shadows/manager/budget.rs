use super::types::*;
use crate::core::error::{RenderError, RenderResult};
use crate::lighting::types::ShadowTechnique;

pub fn enforce_memory_budget(config: &mut ShadowManagerConfig) -> RenderResult<()> {
    let initial_resolution = config.csm.shadow_map_size;
    let budget_mib = config.max_memory_bytes as f64 / (1024.0 * 1024.0);

    loop {
        let usage = estimate_memory_bytes(
            config.csm.shadow_map_size,
            config.csm.cascade_count,
            config.technique,
        );

        if usage <= config.max_memory_bytes {
            // Log final allocation summary
            if config.csm.shadow_map_size != initial_resolution {
                log::info!(
                    "Shadow atlas: downscaled from {}px to {}px to fit {:.1} MiB budget (using {:.2} MiB, technique: {:?}, cascades: {})",
                    initial_resolution,
                    config.csm.shadow_map_size,
                    budget_mib,
                    usage as f64 / (1024.0 * 1024.0),
                    config.technique.name(),
                    config.csm.cascade_count
                );
            } else {
                log::debug!(
                    "Shadow atlas: using {}px maps ({:.2} MiB / {:.1} MiB budget, technique: {:?}, cascades: {})",
                    config.csm.shadow_map_size,
                    usage as f64 / (1024.0 * 1024.0),
                    budget_mib,
                    config.technique.name(),
                    config.csm.cascade_count
                );
            }
            return Ok(());
        }

        let next_res = (config.csm.shadow_map_size / 2).max(MIN_SHADOW_RESOLUTION);
        if next_res == config.csm.shadow_map_size {
            return Err(RenderError::budget(format!(
                "shadow atlas exceeds {:.1} MiB budget at minimum resolution ({}px, {:.2} MiB, technique: {:?}, cascades: {})",
                budget_mib,
                next_res,
                usage as f64 / (1024.0 * 1024.0),
                config.technique.name(),
                config.csm.cascade_count
            )));
        }

        // Single downscaling step
        log::debug!(
            "Shadow budget exceeded ({:.2} MiB > {:.1} MiB); downscaling {}px -> {}px",
            usage as f64 / (1024.0 * 1024.0),
            budget_mib,
            config.csm.shadow_map_size,
            next_res
        );
        config.csm.shadow_map_size = next_res;
    }
}

/// Estimate GPU memory usage for shadow atlas and moment textures.
///
/// Memory breakdown:
/// - Depth atlas: Depth32Float = 4 bytes/pixel × resolution² × cascades
/// - Moment atlas (VSM/EVSM/MSM): Rgba16Float = 8 bytes/pixel
/// - Persistent blur intermediate: Rgba16Float = 8 bytes/pixel
///
/// Does not account for texture padding/alignment; actual GPU usage may be slightly higher.
pub(super) fn estimate_memory_bytes(
    map_resolution: u32,
    cascades: u32,
    technique: ShadowTechnique,
) -> u64 {
    let res = map_resolution as u64;
    let casc = cascades as u64;

    // Depth32Float: 4 bytes per pixel
    let depth_bytes = res * res * casc * 4;

    // Moment atlas and the persistent separable-blur intermediate.
    let moment_bytes = if technique.requires_moments() {
        res * res * casc * 16
    } else {
        0
    };

    depth_bytes + moment_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_includes_persistent_rgba16float_blur_intermediate() {
        assert_eq!(
            estimate_memory_bytes(4096, 2, ShadowTechnique::VSM),
            640 * 1024 * 1024
        );
    }

    #[test]
    fn budget_never_downscales_below_the_supported_shadow_minimum() {
        let mut config = ShadowManagerConfig::default();
        config.technique = ShadowTechnique::VSM;
        config.csm.shadow_map_size = 512;
        config.csm.cascade_count = 1;
        config.max_memory_bytes = 1;

        assert!(enforce_memory_budget(&mut config).is_err());
        assert_eq!(config.csm.shadow_map_size, MIN_SHADOW_RESOLUTION);
    }
}
