use super::native_vt::{TerrainVTSettingsNative, VTLayerFamilyNative};
use pyo3::prelude::*;

pub(super) fn parse_vt_settings(params: &Bound<'_, PyAny>) -> TerrainVTSettingsNative {
    // First check if vt attribute exists and is not None
    if let Ok(vt_obj) = params.getattr("vt") {
        if vt_obj.is_none() {
            return TerrainVTSettingsNative::default();
        }

        // Get enabled flag
        let enabled: bool = vt_obj
            .getattr("enabled")
            .and_then(|v| v.extract())
            .unwrap_or(false);

        if !enabled {
            return TerrainVTSettingsNative::default();
        }

        // Extract fields
        let atlas_size: u32 = vt_obj
            .getattr("atlas_size")
            .and_then(|v| v.extract())
            .unwrap_or(4096);

        let residency_budget_mb: f32 = vt_obj
            .getattr("residency_budget_mb")
            .and_then(|v| v.extract())
            .unwrap_or(256.0);

        let max_mip_levels: u32 = vt_obj
            .getattr("max_mip_levels")
            .and_then(|v| v.extract())
            .unwrap_or(8);

        let use_feedback: bool = vt_obj
            .getattr("use_feedback")
            .and_then(|v| v.extract())
            .unwrap_or(true);

        // Extract layers
        let mut layers = vec![];
        if let Ok(layers_obj) = vt_obj.getattr("layers") {
            if let Ok(layers_list) = layers_obj.extract::<Vec<Bound<'_, PyAny>>>() {
                for layer_obj in layers_list {
                    if let Ok(family) = layer_obj
                        .getattr("family")
                        .and_then(|v| v.extract::<String>())
                    {
                        let virtual_size_px: (u32, u32) = layer_obj
                            .getattr("virtual_size_px")
                            .and_then(|v| v.extract())
                            .unwrap_or((4096, 4096));

                        let tile_size: u32 = layer_obj
                            .getattr("tile_size")
                            .and_then(|v| v.extract())
                            .unwrap_or(248);

                        let tile_border: u32 = layer_obj
                            .getattr("tile_border")
                            .and_then(|v| v.extract())
                            .unwrap_or(4);

                        let fallback: [f32; 4] = layer_obj
                            .getattr("fallback")
                            .and_then(|v| {
                                let list: Vec<f32> = v.extract()?;
                                if list.len() >= 4 {
                                    Ok([list[0], list[1], list[2], list[3]])
                                } else {
                                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(""))
                                }
                            })
                            .unwrap_or([0.5, 0.5, 0.5, 1.0]);

                        layers.push(VTLayerFamilyNative {
                            family,
                            virtual_size: virtual_size_px,
                            tile_size,
                            tile_border,
                            fallback,
                        });
                    }
                }
            }
        }

        return TerrainVTSettingsNative {
            enabled,
            layers,
            atlas_size,
            residency_budget_mb,
            max_mip_levels,
            use_feedback,
        };
    }

    TerrainVTSettingsNative::default()
}
