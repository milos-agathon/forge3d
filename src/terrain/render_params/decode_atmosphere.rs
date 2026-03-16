use super::*;

pub(super) fn parse_volumetrics_settings(params: &Bound<'_, PyAny>) -> VolumetricsSettingsNative {
    if let Ok(vol) = params.getattr("volumetrics") {
        if !vol.is_none() {
            let mode_str: String = vol
                .getattr("mode")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "uniform".to_string());
            VolumetricsSettingsNative {
                enabled: vol
                    .getattr("enabled")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                mode: match mode_str.as_str() {
                    "height" => VolumetricsModeNative::Height,
                    "exponential" => VolumetricsModeNative::Exponential,
                    _ => VolumetricsModeNative::Uniform,
                },
                density: vol
                    .getattr("density")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.01),
                height_falloff: vol
                    .getattr("height_falloff")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
                base_height: vol
                    .getattr("base_height")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                scattering: vol
                    .getattr("scattering")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.5),
                absorption: vol
                    .getattr("absorption")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
                phase_g: vol
                    .getattr("phase_g")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                light_shafts: vol
                    .getattr("light_shafts")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                shaft_intensity: vol
                    .getattr("shaft_intensity")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                shaft_samples: vol
                    .getattr("shaft_samples")
                    .and_then(|v| v.extract())
                    .unwrap_or(32),
                use_shadows: vol
                    .getattr("use_shadows")
                    .and_then(|v| v.extract())
                    .unwrap_or(true),
                half_res: vol
                    .getattr("half_res")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
            }
        } else {
            VolumetricsSettingsNative::default()
        }
    } else {
        VolumetricsSettingsNative::default()
    }
}

pub(super) fn parse_sky_settings(params: &Bound<'_, PyAny>) -> SkySettingsNative {
    if let Ok(sky) = params.getattr("sky") {
        if !sky.is_none() {
            SkySettingsNative {
                enabled: sky
                    .getattr("enabled")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                turbidity: sky
                    .getattr("turbidity")
                    .and_then(|v| v.extract())
                    .unwrap_or(2.0),
                ground_albedo: sky
                    .getattr("ground_albedo")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.3),
                sun_intensity: sky
                    .getattr("sun_intensity")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                sun_size: sky
                    .getattr("sun_size")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                aerial_perspective: sky
                    .getattr("aerial_perspective")
                    .and_then(|v| v.extract())
                    .unwrap_or(true),
                aerial_density: sky
                    .getattr("aerial_density")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                sky_exposure: sky
                    .getattr("sky_exposure")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
            }
        } else {
            SkySettingsNative::default()
        }
    } else {
        SkySettingsNative::default()
    }
}
