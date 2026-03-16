use super::parse::*;
use super::*;

pub(super) fn parse_light_settings(light: &Bound<'_, PyAny>) -> PyResult<LightSettingsNative> {
    let light_type: String = light.getattr("light_type")?.extract()?;
    let azimuth = to_finite_f32(
        light.getattr("azimuth_deg")?.as_gil_ref(),
        "light.azimuth_deg",
    )?;
    let elevation = to_finite_f32(
        light.getattr("elevation_deg")?.as_gil_ref(),
        "light.elevation_deg",
    )?;
    let intensity =
        to_finite_f32(light.getattr("intensity")?.as_gil_ref(), "light.intensity")?.max(0.0);
    let color: Vec<f32> = light
        .getattr("color")?
        .extract()
        .map_err(|_| PyValueError::new_err("light.color must be a sequence of three floats"))?;
    if color.len() != 3 {
        return Err(PyValueError::new_err(
            "light.color must contain exactly three components",
        ));
    }

    let azimuth_rad = azimuth.to_radians();
    let elevation_rad = elevation.to_radians();
    let cos_el = elevation_rad.cos();
    let sin_el = elevation_rad.sin();
    let direction = match light_type.as_str() {
        "Directional" | "directional" => normalize_direction(
            cos_el * azimuth_rad.cos(),
            cos_el * azimuth_rad.sin(),
            sin_el,
        ),
        _ => normalize_direction(
            cos_el * azimuth_rad.cos(),
            cos_el * azimuth_rad.sin(),
            sin_el,
        ),
    };

    Ok(LightSettingsNative {
        direction,
        intensity,
        color: [color[0], color[1], color[2]],
    })
}

pub(super) fn parse_triplanar_settings(
    triplanar: &Bound<'_, PyAny>,
) -> PyResult<TriplanarSettingsNative> {
    Ok(TriplanarSettingsNative {
        scale: to_finite_f32(triplanar.getattr("scale")?.as_gil_ref(), "triplanar.scale")?,
        blend_sharpness: to_finite_f32(
            triplanar.getattr("blend_sharpness")?.as_gil_ref(),
            "triplanar.blend_sharpness",
        )?,
        normal_strength: to_finite_f32(
            triplanar.getattr("normal_strength")?.as_gil_ref(),
            "triplanar.normal_strength",
        )?,
    })
}

pub(super) fn parse_pom_settings(pom: &Bound<'_, PyAny>) -> PyResult<PomSettingsNative> {
    Ok(PomSettingsNative {
        enabled: pom.getattr("enabled")?.extract()?,
        scale: to_finite_f32(pom.getattr("scale")?.as_gil_ref(), "pom.scale")?,
        min_steps: pom.getattr("min_steps")?.extract::<i64>()? as u32,
        max_steps: pom.getattr("max_steps")?.extract::<i64>()? as u32,
        refine_steps: pom.getattr("refine_steps")?.extract::<i64>()? as u32,
        shadow: pom.getattr("shadow")?.extract()?,
        occlusion: pom.getattr("occlusion")?.extract()?,
    })
}

pub(super) fn parse_lod_settings(lod: &Bound<'_, PyAny>) -> PyResult<LodSettingsNative> {
    Ok(LodSettingsNative {
        level: lod.getattr("level")?.extract::<i64>()? as i32,
        bias: to_finite_f32(lod.getattr("bias")?.as_gil_ref(), "lod.bias")?,
        lod0_bias: to_finite_f32(lod.getattr("lod0_bias")?.as_gil_ref(), "lod.lod0_bias")?,
    })
}

pub(super) fn parse_clamp_settings(clamp: &Bound<'_, PyAny>) -> PyResult<ClampSettingsNative> {
    Ok(ClampSettingsNative {
        height_range: tuple_to_f32_pair(
            clamp.getattr("height_range")?.as_gil_ref(),
            "clamp.height_range",
        )?,
        slope_range: tuple_to_f32_pair(
            clamp.getattr("slope_range")?.as_gil_ref(),
            "clamp.slope_range",
        )?,
        ambient_range: tuple_to_f32_pair(
            clamp.getattr("ambient_range")?.as_gil_ref(),
            "clamp.ambient_range",
        )?,
        shadow_range: tuple_to_f32_pair(
            clamp.getattr("shadow_range")?.as_gil_ref(),
            "clamp.shadow_range",
        )?,
        occlusion_range: tuple_to_f32_pair(
            clamp.getattr("occlusion_range")?.as_gil_ref(),
            "clamp.occlusion_range",
        )?,
    })
}

pub(super) fn parse_sampling_settings(
    sampling: &Bound<'_, PyAny>,
) -> PyResult<SamplingSettingsNative> {
    Ok(SamplingSettingsNative {
        mag_filter: parse_filter_mode(
            &sampling.getattr("mag_filter")?.extract::<String>()?,
            "sampling.mag_filter",
        )?,
        min_filter: parse_filter_mode(
            &sampling.getattr("min_filter")?.extract::<String>()?,
            "sampling.min_filter",
        )?,
        mip_filter: parse_filter_mode(
            &sampling.getattr("mip_filter")?.extract::<String>()?,
            "sampling.mip_filter",
        )?,
        anisotropy: sampling
            .getattr("anisotropy")?
            .extract::<i64>()?
            .clamp(1, 16) as u32,
        address_u: parse_address_mode(
            &sampling.getattr("address_u")?.extract::<String>()?,
            "sampling.address_u",
        )?,
        address_v: parse_address_mode(
            &sampling.getattr("address_v")?.extract::<String>()?,
            "sampling.address_v",
        )?,
        address_w: parse_address_mode(
            &sampling.getattr("address_w")?.extract::<String>()?,
            "sampling.address_w",
        )?,
    })
}

pub(super) fn parse_shadow_settings(shadows: &Bound<'_, PyAny>) -> PyResult<ShadowSettingsNative> {
    let softness = shadows.getattr("softness")?.extract().unwrap_or(0.01);
    let pcss_light_radius = shadows
        .getattr("pcss_light_radius")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or(0.0);
    Ok(ShadowSettingsNative {
        enabled: shadows.getattr("enabled")?.extract().unwrap_or(true),
        technique: shadows
            .getattr("technique")?
            .extract::<String>()
            .unwrap_or_else(|_| "PCSS".to_string()),
        resolution: shadows
            .getattr("resolution")?
            .extract::<i64>()
            .unwrap_or(2048) as u32,
        cascades: shadows.getattr("cascades")?.extract::<i64>().unwrap_or(1) as u32,
        max_distance: shadows.getattr("max_distance")?.extract().unwrap_or(3000.0),
        softness,
        pcss_light_radius,
        intensity: shadows.getattr("intensity")?.extract().unwrap_or(1.0),
        slope_scale_bias: shadows
            .getattr("slope_scale_bias")?
            .extract()
            .unwrap_or(0.001),
        depth_bias: shadows.getattr("depth_bias")?.extract().unwrap_or(0.0005),
        normal_bias: shadows.getattr("normal_bias")?.extract().unwrap_or(0.0002),
    })
}
