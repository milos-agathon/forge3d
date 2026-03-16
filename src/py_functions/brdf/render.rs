use super::*;

pub(crate) fn render_brdf_tile_impl<'py>(
    py: Python<'py>,
    model: &str,
    roughness: f32,
    width: u32,
    height: u32,
    ndf_only: bool,
    g_only: bool,
    dfg_only: bool,
    spec_only: bool,
    roughness_visualize: bool,
    exposure: f32,
    light_intensity: f32,
    base_color: (f32, f32, f32),
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen: f32,
    sheen_tint: f32,
    specular_tint: f32,
    debug_dot_products: bool,
    debug_lambert_only: bool,
    debug_diffuse_only: bool,
    debug_d: bool,
    debug_spec_no_nl: bool,
    debug_energy: bool,
    debug_angle_sweep: bool,
    debug_angle_component: u32,
    debug_no_srgb: bool,
    output_mode: u32,
    metallic_override: f32,
    mode: Option<&str>,
    wi3_debug_mode: u32,
    wi3_debug_roughness: f32,
    sphere_sectors: u32,
    sphere_stacks: u32,
    light_dir: Option<(f32, f32, f32)>,
    debug_kind: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let model_u32 = match model.to_lowercase().as_str() {
        "lambert" => 0,
        "phong" => 1,
        "ggx" => 4,
        "disney" => 6,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid BRDF model '{}'. Expected one of: lambert, phong, ggx, disney",
                model
            )));
        }
    };

    let roughness = roughness.clamp(0.0, 1.0);
    let sphere_sectors = sphere_sectors.clamp(8, 1024);
    let sphere_stacks = sphere_stacks.clamp(4, 512);
    let debug_kind = debug_kind.min(3);

    let (
        mut ndf_only,
        mut g_only,
        mut dfg_only,
        mut spec_only,
        mut roughness_visualize,
        mut debug_lambert_only,
        mut debug_diffuse_only,
        mut debug_d,
        mut debug_spec_no_nl,
        mut debug_energy,
        mut debug_angle_sweep,
        mut debug_angle_component,
        mut debug_no_srgb,
        mut output_mode,
    ) = (
        ndf_only,
        g_only,
        dfg_only,
        spec_only,
        roughness_visualize,
        debug_lambert_only,
        debug_diffuse_only,
        debug_d,
        debug_spec_no_nl,
        debug_energy,
        debug_angle_sweep,
        debug_angle_component,
        debug_no_srgb,
        output_mode,
    );

    if let Some(mode_str) = mode {
        let mapped = mode_str.to_lowercase();
        match mapped.as_str() {
            "full" => {
                ndf_only = false;
                g_only = false;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = false;
            }
            "ndf" => {
                ndf_only = true;
                g_only = false;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = false;
            }
            "g" => {
                ndf_only = false;
                g_only = true;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = false;
            }
            "dfg" => {
                ndf_only = false;
                g_only = false;
                dfg_only = true;
                spec_only = false;
                roughness_visualize = false;
            }
            "spec" => {
                ndf_only = false;
                g_only = false;
                dfg_only = false;
                spec_only = true;
                roughness_visualize = false;
            }
            "roughness" => {
                ndf_only = false;
                g_only = false;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = true;
            }
            "lambert" | "flatness" => debug_lambert_only = true,
            "diffuse" | "diffuse_only" => debug_diffuse_only = true,
            "d" | "ndf_only" | "debug_d" => debug_d = true,
            "spec_no_nl" => {
                spec_only = true;
                debug_spec_no_nl = true;
            }
            "energy" | "kskd" => debug_energy = true,
            "angle_spec" => {
                debug_angle_sweep = true;
                debug_angle_component = 0;
            }
            "angle_diffuse" => {
                debug_angle_sweep = true;
                debug_angle_component = 1;
            }
            "angle_combined" | "angle" => {
                debug_angle_sweep = true;
                debug_angle_component = 2;
            }
            "linear" => {
                output_mode = 0;
                debug_no_srgb = true;
            }
            "srgb" => {
                output_mode = 1;
                debug_no_srgb = false;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid mode '{}'. Expected one of: full, ndf, g, dfg, spec, roughness, lambert, d, spec_no_nl, energy, angle_spec, angle_diffuse, angle_combined, linear, srgb",
                    mode_str
                )));
            }
        }
        log::info!(
            "[M4/M2] Mode mapping applied: mode={} -> ndf_only={} g_only={} dfg_only={} spec_only={} roughness_visualize={} lambert_only={} diffuse_only={} debug_d={} spec_no_nl={} energy={} angle_sweep={} angle_comp={} no_srgb={} out_mode={}",
            mapped,
            ndf_only,
            g_only,
            dfg_only,
            spec_only,
            roughness_visualize,
            debug_lambert_only,
            debug_diffuse_only,
            debug_d,
            debug_spec_no_nl,
            debug_energy,
            debug_angle_sweep,
            debug_angle_component,
            debug_no_srgb,
            output_mode
        );
    }

    let mut wi3_debug_roughness = wi3_debug_roughness;
    if wi3_debug_mode != 0 && wi3_debug_roughness <= 0.0 {
        wi3_debug_roughness = roughness;
    }
    wi3_debug_roughness = wi3_debug_roughness.clamp(0.0, 1.0);

    let overrides = crate::offscreen::brdf_tile::BrdfTileOverrides {
        light_dir: light_dir.map(|(x, y, z)| [x, y, z]),
        debug_kind: Some(debug_kind),
    };

    let ctx = crate::core::gpu::ctx();
    let buffer = crate::offscreen::brdf_tile::render_brdf_tile_with_overrides(
        ctx.device.as_ref(),
        ctx.queue.as_ref(),
        model_u32,
        roughness,
        width,
        height,
        ndf_only,
        g_only,
        dfg_only,
        spec_only,
        roughness_visualize,
        exposure,
        light_intensity,
        [base_color.0, base_color.1, base_color.2],
        clearcoat,
        clearcoat_roughness,
        sheen,
        sheen_tint,
        specular_tint,
        debug_dot_products,
        debug_lambert_only,
        debug_diffuse_only,
        debug_d,
        debug_spec_no_nl,
        debug_energy,
        debug_angle_sweep,
        debug_angle_component,
        debug_no_srgb,
        output_mode,
        metallic_override,
        wi3_debug_mode,
        wi3_debug_roughness,
        sphere_sectors,
        sphere_stacks,
        &overrides,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to render BRDF tile: {}", e)))?;

    let expected_size = (height * width * 4) as usize;
    if buffer.len() != expected_size {
        return Err(PyRuntimeError::new_err(format!(
            "Buffer size mismatch: got {} bytes, expected {}",
            buffer.len(),
            expected_size
        )));
    }

    let array = ndarray::Array3::from_shape_vec((height as usize, width as usize, 4), buffer)
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to reshape buffer to array: {}", e))
        })?;

    Ok(array.into_pyarray_bound(py))
}
