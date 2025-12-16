use super::types::*;
use crate::lighting::types::ShadowTechnique;
use log::warn;

pub fn enforce_memory_budget(config: &mut ShadowManagerConfig) {
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
            break;
        }

        let next_res = (config.csm.shadow_map_size / 2).max(MIN_SHADOW_RESOLUTION);
        if next_res == config.csm.shadow_map_size {
            // Hit minimum resolution; cannot downscale further
            warn!(
                "Shadow atlas exceeds {:.1} MiB budget at minimum resolution ({}px, {:.2} MiB, technique: {:?}, cascades: {})",
                budget_mib,
                next_res,
                usage as f64 / (1024.0 * 1024.0),
                config.technique.name(),
                config.csm.cascade_count
            );
            break;
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
/// - Moment textures (VSM/EVSM/MSM): Rgba32Float = 16 bytes/pixel × resolution² × cascades
///
/// Note: VSM technically only needs 2 channels (mean, variance), but we use Rgba32Float
/// for all moment techniques to simplify the implementation and allow future extensions.
/// Does not account for texture padding/alignment; actual GPU usage may be slightly higher.
fn estimate_memory_bytes(map_resolution: u32, cascades: u32, technique: ShadowTechnique) -> u64 {
    let res = map_resolution as u64;
    let casc = cascades as u64;

    // Depth32Float: 4 bytes per pixel
    let depth_bytes = res * res * casc * 4;

    // Moment texture bytes (all use Rgba32Float in current implementation)
    let moment_bytes = if technique.requires_moments() {
        // Rgba32Float: 4 channels × 4 bytes = 16 bytes per pixel
        // Used for VSM (2 channels used), EVSM (4 channels), MSM (4 channels)
        res * res * casc * 16
    } else {
        0
    };

    depth_bytes + moment_bytes
}
