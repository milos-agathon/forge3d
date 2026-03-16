use super::*;

pub(super) fn parse_material_layer_settings(
    params: &Bound<'_, PyAny>,
) -> MaterialLayerSettingsNative {
    if let Ok(materials) = params.getattr("materials") {
        let snow_enabled: bool = materials
            .getattr("snow_enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let snow_altitude_min: f32 = materials
            .getattr("snow_altitude_min")
            .and_then(|v| v.extract())
            .unwrap_or(2000.0);
        let snow_altitude_blend: f32 = materials
            .getattr("snow_altitude_blend")
            .and_then(|v| v.extract())
            .unwrap_or(500.0);
        let snow_slope_max: f32 = materials
            .getattr("snow_slope_max")
            .and_then(|v| v.extract())
            .unwrap_or(45.0);
        let snow_slope_blend: f32 = materials
            .getattr("snow_slope_blend")
            .and_then(|v| v.extract())
            .unwrap_or(15.0);
        let snow_aspect_influence: f32 = materials
            .getattr("snow_aspect_influence")
            .and_then(|v| v.extract())
            .unwrap_or(0.3);
        let snow_color_vec: Vec<f32> = materials
            .getattr("snow_color")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| vec![0.95, 0.95, 0.98]);
        let snow_color = [
            snow_color_vec.first().copied().unwrap_or(0.95),
            snow_color_vec.get(1).copied().unwrap_or(0.95),
            snow_color_vec.get(2).copied().unwrap_or(0.98),
        ];
        let snow_roughness: f32 = materials
            .getattr("snow_roughness")
            .and_then(|v| v.extract())
            .unwrap_or(0.4);

        let rock_enabled: bool = materials
            .getattr("rock_enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let rock_slope_min: f32 = materials
            .getattr("rock_slope_min")
            .and_then(|v| v.extract())
            .unwrap_or(45.0);
        let rock_slope_blend: f32 = materials
            .getattr("rock_slope_blend")
            .and_then(|v| v.extract())
            .unwrap_or(10.0);
        let rock_color_vec: Vec<f32> = materials
            .getattr("rock_color")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| vec![0.35, 0.32, 0.28]);
        let rock_color = [
            rock_color_vec.first().copied().unwrap_or(0.35),
            rock_color_vec.get(1).copied().unwrap_or(0.32),
            rock_color_vec.get(2).copied().unwrap_or(0.28),
        ];
        let rock_roughness: f32 = materials
            .getattr("rock_roughness")
            .and_then(|v| v.extract())
            .unwrap_or(0.8);

        let wetness_enabled: bool = materials
            .getattr("wetness_enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let wetness_strength: f32 = materials
            .getattr("wetness_strength")
            .and_then(|v| v.extract())
            .unwrap_or(0.3);
        let wetness_slope_influence: f32 = materials
            .getattr("wetness_slope_influence")
            .and_then(|v| v.extract())
            .unwrap_or(0.5);

        MaterialLayerSettingsNative {
            snow_enabled,
            snow_altitude_min,
            snow_altitude_blend,
            snow_slope_max,
            snow_slope_blend,
            snow_aspect_influence,
            snow_color,
            snow_roughness,
            rock_enabled,
            rock_slope_min,
            rock_slope_blend,
            rock_color,
            rock_roughness,
            wetness_enabled,
            wetness_strength,
            wetness_slope_influence,
        }
    } else {
        MaterialLayerSettingsNative::default()
    }
}

pub(super) fn parse_vector_overlay_settings(
    params: &Bound<'_, PyAny>,
) -> VectorOverlaySettingsNative {
    if let Ok(vo) = params.getattr("vector_overlay") {
        let depth_test: bool = vo
            .getattr("depth_test")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let depth_bias: f32 = vo
            .getattr("depth_bias")
            .and_then(|v| v.extract())
            .unwrap_or(0.001);
        let depth_bias_slope: f32 = vo
            .getattr("depth_bias_slope")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        let halo_enabled: bool = vo
            .getattr("halo_enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let halo_width: f32 = vo
            .getattr("halo_width")
            .and_then(|v| v.extract())
            .unwrap_or(2.0);
        let halo_color_vec: Vec<f32> = vo
            .getattr("halo_color")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| vec![0.0, 0.0, 0.0, 0.5]);
        let halo_color = [
            halo_color_vec.first().copied().unwrap_or(0.0),
            halo_color_vec.get(1).copied().unwrap_or(0.0),
            halo_color_vec.get(2).copied().unwrap_or(0.0),
            halo_color_vec.get(3).copied().unwrap_or(0.5),
        ];
        let halo_blur: f32 = vo
            .getattr("halo_blur")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        let contour_enabled: bool = vo
            .getattr("contour_enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let contour_width: f32 = vo
            .getattr("contour_width")
            .and_then(|v| v.extract())
            .unwrap_or(1.0);
        let contour_color_vec: Vec<f32> = vo
            .getattr("contour_color")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| vec![0.0, 0.0, 0.0, 0.8]);
        let contour_color = [
            contour_color_vec.first().copied().unwrap_or(0.0),
            contour_color_vec.get(1).copied().unwrap_or(0.0),
            contour_color_vec.get(2).copied().unwrap_or(0.0),
            contour_color_vec.get(3).copied().unwrap_or(0.8),
        ];
        VectorOverlaySettingsNative {
            depth_test,
            depth_bias,
            depth_bias_slope,
            halo_enabled,
            halo_width,
            halo_color,
            halo_blur,
            contour_enabled,
            contour_width,
            contour_color,
        }
    } else {
        VectorOverlaySettingsNative::default()
    }
}
