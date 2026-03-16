use super::*;

pub(super) fn parse_fog_settings(params: &Bound<'_, PyAny>) -> FogSettingsNative {
    if let Ok(fog) = params.getattr("fog") {
        let inscatter_vec: Vec<f32> = fog
            .getattr("inscatter")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| vec![1.0, 1.0, 1.0]);
        let inscatter = [
            inscatter_vec.first().copied().unwrap_or(1.0),
            inscatter_vec.get(1).copied().unwrap_or(1.0),
            inscatter_vec.get(2).copied().unwrap_or(1.0),
        ];
        let density: f32 = fog
            .getattr("density")
            .and_then(|v| v.extract())
            .unwrap_or(0.0);
        let height_falloff: f32 = fog
            .getattr("height_falloff")
            .and_then(|v| v.extract())
            .unwrap_or(0.0);
        let base_height: f32 = fog
            .getattr("base_height")
            .and_then(|v| v.extract())
            .unwrap_or(0.0);
        let aerial_perspective: f32 = fog
            .getattr("aerial_perspective")
            .and_then(|v| v.extract())
            .unwrap_or(0.0);
        FogSettingsNative {
            density,
            height_falloff,
            base_height,
            inscatter,
            aerial_perspective,
        }
    } else {
        FogSettingsNative::default()
    }
}

pub(super) fn parse_reflection_settings(params: &Bound<'_, PyAny>) -> ReflectionSettingsNative {
    if let Ok(refl) = params.getattr("reflection") {
        let enabled: bool = refl
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let intensity: f32 = refl
            .getattr("intensity")
            .and_then(|v| v.extract())
            .unwrap_or(0.8);
        let fresnel_power: f32 = refl
            .getattr("fresnel_power")
            .and_then(|v| v.extract())
            .unwrap_or(5.0);
        let wave_strength: f32 = refl
            .getattr("wave_strength")
            .and_then(|v| v.extract())
            .unwrap_or(0.02);
        let shore_atten_width: f32 = refl
            .getattr("shore_atten_width")
            .and_then(|v| v.extract())
            .unwrap_or(0.3);
        let water_plane_height: f32 = refl
            .getattr("water_plane_height")
            .and_then(|v| v.extract())
            .unwrap_or(0.0);
        ReflectionSettingsNative {
            enabled,
            intensity,
            fresnel_power,
            wave_strength,
            shore_atten_width,
            water_plane_height,
        }
    } else {
        ReflectionSettingsNative::default()
    }
}

pub(super) fn parse_detail_settings(params: &Bound<'_, PyAny>) -> DetailSettingsNative {
    if let Ok(det) = params.getattr("detail") {
        let enabled: bool = det
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let detail_scale: f32 = det
            .getattr("detail_scale")
            .and_then(|v| v.extract())
            .unwrap_or(2.0);
        let normal_strength: f32 = det
            .getattr("normal_strength")
            .and_then(|v| v.extract())
            .unwrap_or(0.3);
        let albedo_noise: f32 = det
            .getattr("albedo_noise")
            .and_then(|v| v.extract())
            .unwrap_or(0.1);
        let fade_start: f32 = det
            .getattr("fade_start")
            .and_then(|v| v.extract())
            .unwrap_or(50.0);
        let fade_end: f32 = det
            .getattr("fade_end")
            .and_then(|v| v.extract())
            .unwrap_or(200.0);
        DetailSettingsNative {
            enabled,
            detail_scale,
            normal_strength,
            albedo_noise,
            fade_start,
            fade_end,
        }
    } else {
        DetailSettingsNative::default()
    }
}

pub(super) fn parse_height_ao_settings(params: &Bound<'_, PyAny>) -> HeightAoSettingsNative {
    if let Ok(hao) = params.getattr("height_ao") {
        let enabled: bool = hao
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let resolution_scale: f32 = hao
            .getattr("resolution_scale")
            .and_then(|v| v.extract())
            .unwrap_or(0.5);
        let directions: u32 = hao
            .getattr("directions")
            .and_then(|v| v.extract())
            .unwrap_or(6);
        let steps: u32 = hao.getattr("steps").and_then(|v| v.extract()).unwrap_or(16);
        let max_distance: f32 = hao
            .getattr("max_distance")
            .and_then(|v| v.extract())
            .unwrap_or(200.0);
        let strength: f32 = hao
            .getattr("strength")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        let blur: bool = hao
            .getattr("blur")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        HeightAoSettingsNative {
            enabled,
            resolution_scale,
            directions,
            steps,
            max_distance,
            strength,
            blur,
        }
    } else {
        HeightAoSettingsNative::default()
    }
}

pub(super) fn parse_sun_visibility_settings(
    params: &Bound<'_, PyAny>,
) -> SunVisibilitySettingsNative {
    if let Ok(sv) = params.getattr("sun_visibility") {
        let enabled: bool = sv
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let mode: String = sv
            .getattr("mode")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| "hard".to_string());
        let resolution_scale: f32 = sv
            .getattr("resolution_scale")
            .and_then(|v| v.extract())
            .unwrap_or(0.5);
        let samples: u32 = sv.getattr("samples").and_then(|v| v.extract()).unwrap_or(4);
        let steps: u32 = sv.getattr("steps").and_then(|v| v.extract()).unwrap_or(24);
        let max_distance: f32 = sv
            .getattr("max_distance")
            .and_then(|v| v.extract())
            .unwrap_or(400.0);
        let softness: f32 = sv
            .getattr("softness")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        let bias: f32 = sv.getattr("bias").and_then(|v| v.extract()).unwrap_or(0.01);
        SunVisibilitySettingsNative {
            enabled,
            mode,
            resolution_scale,
            samples,
            steps,
            max_distance,
            softness,
            bias,
        }
    } else {
        SunVisibilitySettingsNative::default()
    }
}

pub(super) fn parse_bloom_settings(params: &Bound<'_, PyAny>) -> BloomSettingsNative {
    if let Ok(bloom) = params.getattr("bloom") {
        let enabled: bool = bloom
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let threshold: f32 = bloom
            .getattr("threshold")
            .and_then(|v| v.extract())
            .unwrap_or(1.5);
        let softness: f32 = bloom
            .getattr("softness")
            .and_then(|v| v.extract())
            .unwrap_or(0.5);
        let intensity: f32 = bloom
            .getattr("intensity")
            .and_then(|v| v.extract())
            .unwrap_or(0.3);
        let radius: f32 = bloom
            .getattr("radius")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        BloomSettingsNative {
            enabled,
            threshold,
            softness,
            intensity,
            radius,
        }
    } else {
        BloomSettingsNative::default()
    }
}
