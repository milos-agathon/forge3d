use super::*;

fn extract_rgb3(obj: &Bound<'_, PyAny>, attr_name: &str, fallback: [f32; 3]) -> [f32; 3] {
    let values: Vec<f32> = obj
        .getattr(attr_name)
        .and_then(|v| v.extract())
        .unwrap_or_else(|_| fallback.to_vec());
    [
        values.first().copied().unwrap_or(fallback[0]),
        values.get(1).copied().unwrap_or(fallback[1]),
        values.get(2).copied().unwrap_or(fallback[2]),
    ]
}

pub(super) fn parse_probe_settings(params: &Bound<'_, PyAny>) -> ProbeSettingsNative {
    if let Ok(probes) = params.getattr("probes") {
        let enabled: bool = probes
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        if !enabled {
            return ProbeSettingsNative::default();
        }

        let grid_dims: (u32, u32) = probes
            .getattr("grid_dims")
            .and_then(|v| v.extract())
            .unwrap_or((8, 8));
        let origin: Option<(f32, f32)> =
            probes.getattr("origin").ok().and_then(|v| v.extract().ok());
        let spacing: Option<(f32, f32)> = probes
            .getattr("spacing")
            .ok()
            .and_then(|v| v.extract().ok());
        let height_offset: f32 = probes
            .getattr("height_offset")
            .and_then(|v| v.extract())
            .unwrap_or(5.0);
        let ray_count: u32 = probes
            .getattr("ray_count")
            .and_then(|v| v.extract())
            .unwrap_or(64);
        let fallback_blend_distance: Option<f32> = probes
            .getattr("fallback_blend_distance")
            .ok()
            .and_then(|v| v.extract().ok());
        let sky_color = extract_rgb3(&probes, "sky_color", [0.6, 0.75, 1.0]);
        let sky_intensity: f32 = probes
            .getattr("sky_intensity")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);

        ProbeSettingsNative {
            enabled,
            grid_dims,
            origin,
            spacing,
            height_offset,
            ray_count,
            fallback_blend_distance,
            sky_color,
            sky_intensity,
        }
    } else {
        ProbeSettingsNative::default()
    }
}

pub(super) fn parse_reflection_probe_settings(
    params: &Bound<'_, PyAny>,
) -> ReflectionProbeSettingsNative {
    if let Ok(probes) = params.getattr("reflection_probes") {
        let enabled: bool = probes
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        if !enabled {
            return ReflectionProbeSettingsNative::default();
        }

        let grid_dims: (u32, u32) = probes
            .getattr("grid_dims")
            .and_then(|v| v.extract())
            .unwrap_or((4, 4));
        let origin: Option<(f32, f32)> =
            probes.getattr("origin").ok().and_then(|v| v.extract().ok());
        let spacing: Option<(f32, f32)> = probes
            .getattr("spacing")
            .ok()
            .and_then(|v| v.extract().ok());
        let height_offset: f32 = probes
            .getattr("height_offset")
            .and_then(|v| v.extract())
            .unwrap_or(5.0);
        let resolution: u32 = probes
            .getattr("resolution")
            .and_then(|v| v.extract())
            .unwrap_or(16);
        let ray_count: u32 = probes
            .getattr("ray_count")
            .and_then(|v| v.extract())
            .unwrap_or(64);
        let trace_steps: u32 = probes
            .getattr("trace_steps")
            .and_then(|v| v.extract())
            .unwrap_or(192);
        let trace_refine_steps: u32 = probes
            .getattr("trace_refine_steps")
            .and_then(|v| v.extract())
            .unwrap_or(5);
        let fallback_blend_distance =
            probes
                .getattr("fallback_blend_distance")
                .ok()
                .and_then(|value| {
                    if let Ok(single) = value.extract::<f32>() {
                        Some((single, single))
                    } else {
                        value.extract::<(f32, f32)>().ok()
                    }
                });
        let _unused_sky_color = extract_rgb3(&probes, "sky_color", [0.6, 0.75, 1.0]);
        let _unused_sky_intensity: f32 = probes
            .getattr("sky_intensity")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        let strength: f32 = probes
            .getattr("strength")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);

        ReflectionProbeSettingsNative {
            enabled,
            grid_dims,
            origin,
            spacing,
            height_offset,
            resolution,
            ray_count,
            trace_steps,
            trace_refine_steps,
            fallback_blend_distance,
            strength,
        }
    } else {
        ReflectionProbeSettingsNative::default()
    }
}
