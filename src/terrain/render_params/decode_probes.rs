use super::*;

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
        let sky_color_vec: Vec<f32> = probes
            .getattr("sky_color")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| vec![0.6, 0.75, 1.0]);
        let sky_color = [
            sky_color_vec.first().copied().unwrap_or(0.6),
            sky_color_vec.get(1).copied().unwrap_or(0.75),
            sky_color_vec.get(2).copied().unwrap_or(1.0),
        ];
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
