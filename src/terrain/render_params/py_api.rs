use super::*;

#[pymethods]
impl TerrainRenderParams {
    #[new]
    #[pyo3(signature = (params))]
    pub fn new(py: Python<'_>, params: Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_python_params(py, params)
    }

    /// Project terrain UV, raw height and an elevation offset through the exact
    /// camera and height transform used by the mesh/flat-clipmap terrain pass.
    pub fn project_terrain_points(&self, points: Vec<[f64; 4]>) -> PyResult<Vec<Option<[f64; 3]>>> {
        let (_, view, proj) = crate::terrain::renderer::TerrainScene::build_camera_matrices(self);
        let (view, proj) = (view.as_dmat4(), proj.as_dmat4());
        points
            .into_iter()
            .map(|point| {
                if !point.iter().all(|value| value.is_finite()) {
                    return Err(PyValueError::new_err(
                        "terrain projection requires finite UV/height/offset",
                    ));
                }
                Ok(self.project_point(point, view, proj).map(|mut projected| {
                    // Compare final device depth in the same f32 representation
                    // as the GPU/proxy depth buffer. World/camera math and screen
                    // positions above retain f64 precision.
                    projected[2] = f64::from(projected[2] as f32);
                    projected
                }))
            })
            .collect()
    }

    /// Rasterize the DEM triangles once without crossing Python for every cell.
    #[pyo3(signature = (heightmap, size_px, nodata_height_below=None))]
    pub fn project_terrain_depth<'py>(
        &self,
        py: Python<'py>,
        heightmap: numpy::PyReadonlyArray2<'py, f32>,
        size_px: (usize, usize),
        nodata_height_below: Option<f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let dem = heightmap.as_array();
        let (rows, cols) = dem.dim();
        let (width, height) = size_px;
        if rows < 2 || cols < 2 || width == 0 || height == 0 || !dem.iter().all(|v| v.is_finite()) {
            return Err(PyValueError::new_err(
                "depth projection requires a finite 2D terrain and positive viewport",
            ));
        }
        let count = width
            .checked_mul(height)
            .ok_or_else(|| PyValueError::new_err("depth viewport overflow"))?;
        let (_, view, proj) = crate::terrain::renderer::TerrainScene::build_camera_matrices(self);
        let (view, proj) = (view.as_dmat4(), proj.as_dmat4());
        // Only two projected rows are live, independent of DEM size.
        let mut previous = Vec::with_capacity(cols);
        let mut current = Vec::with_capacity(cols);
        let mut depth = vec![1.0f32; count];
        for row in 0..rows {
            current.clear();
            for col in 0..cols {
                let value = self.project_point(
                    [
                        col as f64 / (cols - 1) as f64,
                        row as f64 / (rows - 1) as f64,
                        f64::from(dem[[row, col]]),
                        0.0,
                    ],
                    view,
                    proj,
                );
                current.push(value.map(|p| {
                    [
                        p[0] * width as f64 / f64::from(self.size_px.0),
                        p[1] * height as f64 / f64::from(self.size_px.1),
                        p[2],
                    ]
                }));
            }
            if row > 0 {
                for col in 0..cols - 1 {
                    for indices in [
                        [(row - 1, col), (row - 1, col + 1), (row, col)],
                        [(row - 1, col + 1), (row, col + 1), (row, col)],
                    ] {
                        // Fully nodata triangles leave the renderer's holes empty.
                        if nodata_height_below
                            .is_some_and(|limit| indices.iter().all(|&(y, x)| dem[[y, x]] < limit))
                        {
                            continue;
                        }
                        let triangle =
                            indices.map(|(y, x)| if y == row { current[x] } else { previous[x] });
                        if let [Some(a), Some(b), Some(c)] = triangle {
                            super::projection::raster_triangle(
                                &mut depth,
                                width,
                                height,
                                [a, b, c],
                            );
                        }
                    }
                }
            }
            std::mem::swap(&mut previous, &mut current);
        }
        use numpy::IntoPyArray;
        let array = ndarray::Array2::from_shape_vec((height, width), depth)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(array.into_pyarray_bound(py))
    }

    /// Recover the visible terrain UV at each depth pixel using the same camera.
    pub fn unproject_terrain_depth<'py>(
        &self,
        py: Python<'py>,
        depth: numpy::PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
        let depth = depth.as_array();
        let (height, width) = depth.dim();
        if width == 0
            || height == 0
            || !depth
                .iter()
                .all(|z| z.is_finite() && (0.0..=1.0).contains(z))
        {
            return Err(PyValueError::new_err(
                "unprojection requires finite normalized device depth",
            ));
        }
        let (_, view, proj) = crate::terrain::renderer::TerrainScene::build_camera_matrices(self);
        let inverse = (proj.as_dmat4() * view.as_dmat4()).inverse();
        let mut uv = ndarray::Array3::from_elem((height, width, 2), f32::NAN);
        for row in 0..height {
            for col in 0..width {
                let z = depth[[row, col]];
                if z == 1.0 {
                    continue;
                }
                let world = inverse
                    * glam::DVec4::new(
                        (col as f64 + 0.5) * 2.0 / width as f64 - 1.0,
                        1.0 - (row as f64 + 0.5) * 2.0 / height as f64,
                        f64::from(z),
                        1.0,
                    );
                if world.is_finite() && world.w != 0.0 {
                    // These are final normalized texture UVs for screen pixels,
                    // not absolute or terrain-local world-coordinate storage.
                    uv[[row, col, 0]] =
                        (world.x / world.w / f64::from(self.terrain_span) + 0.5) as f32;
                    uv[[row, col, 1]] =
                        (world.y / world.w / f64::from(self.terrain_span) + 0.5) as f32;
                }
            }
        }
        use numpy::IntoPyArray;
        Ok(uv.into_pyarray_bound(py))
    }

    #[getter]
    pub fn size_px(&self) -> (u32, u32) {
        self.size_px
    }

    #[getter]
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    #[getter]
    pub fn msaa_samples(&self) -> u32 {
        self.msaa_samples
    }

    #[getter]
    pub fn z_scale(&self) -> f32 {
        self.z_scale
    }

    #[getter]
    pub fn cam_target(&self) -> [f32; 3] {
        self.cam_target
    }

    #[getter]
    pub fn cam_radius(&self) -> f32 {
        self.cam_radius
    }

    #[getter]
    pub fn cam_phi_deg(&self) -> f32 {
        self.cam_phi_deg
    }

    #[getter]
    pub fn cam_theta_deg(&self) -> f32 {
        self.cam_theta_deg
    }

    #[getter]
    pub fn cam_gamma_deg(&self) -> f32 {
        self.cam_gamma_deg
    }

    #[getter]
    pub fn fov_y_deg(&self) -> f32 {
        self.fov_y_deg
    }

    #[getter]
    pub fn clip(&self) -> (f32, f32) {
        self.clip
    }

    #[getter]
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    #[getter]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    #[getter]
    pub fn albedo_mode(&self) -> &str {
        &self.albedo_mode
    }

    #[getter]
    pub fn colormap_strength(&self) -> f32 {
        self.colormap_strength
    }

    #[getter]
    pub fn hue_variation_strength(&self) -> f32 {
        self.hue_variation_strength
    }

    #[getter]
    pub fn material_slope_bias(&self) -> f32 {
        self.material_slope_bias
    }

    #[getter]
    pub fn material_layer_centers(&self) -> Option<Vec<f32>> {
        self.material_layer_centers.clone()
    }

    #[getter]
    pub fn nodata_height_below(&self) -> Option<f32> {
        self.nodata_height_below
    }

    /// P5: Get AO weight (0.0 = no AO, 1.0 = full AO)
    #[getter]
    pub fn ao_weight(&self) -> f32 {
        self.ao_weight
    }

    #[getter]
    pub fn height_curve_mode(&self) -> &str {
        &self.height_curve_mode
    }

    #[getter]
    pub fn height_curve_strength(&self) -> f32 {
        self.height_curve_strength
    }

    #[getter]
    pub fn height_curve_power(&self) -> f32 {
        self.height_curve_power
    }

    /// P5-L: Lambert contrast parameter [0,1] for gradient enhancement
    #[getter]
    pub fn lambert_contrast(&self) -> f32 {
        self.lambert_contrast
    }

    /// P6.1: Use Rgba8UnormSrgb for colormap texture (correct color space sampling)
    #[getter]
    pub fn colormap_srgb(&self) -> bool {
        self.colormap_srgb
    }

    /// P6.1: Use exact linear_to_srgb() instead of pow-gamma for output encoding
    #[getter]
    pub fn output_srgb_eotf(&self) -> bool {
        self.output_srgb_eotf
    }

    /// P7: Camera projection mode ("screen" = fullscreen triangle, "mesh" = perspective grid)
    #[getter]
    pub fn camera_mode(&self) -> &str {
        &self.camera_mode
    }

    #[getter]
    pub fn culling(&self) -> &str {
        &self.culling
    }

    #[getter]
    pub fn shading(&self) -> &str {
        &self.shading
    }

    #[getter]
    pub fn vt_store(&self) -> Option<String> {
        self.vt_store_path.clone()
    }

    #[getter]
    pub fn prefetch_horizon_ms(&self) -> f32 {
        self.prefetch_horizon_ms
    }

    #[getter]
    pub fn vt_upload_budget_bytes(&self) -> u64 {
        self.vt_upload_budget_bytes
    }

    /// P7: Debug mode for projection probes (0=normal, 40=view-depth, 41=NDC depth, 42=view-pos XYZ)
    #[getter]
    pub fn debug_mode(&self) -> u32 {
        self.debug_mode
    }

    /// M1: Accumulation AA sample count (1 = no AA, 16/64/256 typical for offline)
    #[getter]
    pub fn aa_samples(&self) -> u32 {
        self.aa_samples
    }

    /// M1: Accumulation AA seed for deterministic jitter (None = default sequence)
    #[getter]
    pub fn aa_seed(&self) -> Option<u64> {
        self.aa_seed
    }

    #[getter]
    pub fn terrain_data_revision(&self) -> Option<u64> {
        self.terrain_data_revision
    }

    #[getter]
    pub fn height_curve_lut(&self) -> Option<Vec<f32>> {
        self.height_curve_lut
            .as_ref()
            .map(|lut| lut.as_ref().clone())
    }

    #[getter]
    pub fn overlays(&self) -> Vec<Py<crate::core::overlay_layer::OverlayLayer>> {
        self.overlays.clone()
    }

    #[getter]
    pub fn light<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.light.clone_ref(py)
    }

    #[getter]
    pub fn ibl<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.ibl.clone_ref(py)
    }

    #[getter]
    pub fn shadows<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.shadows.clone_ref(py)
    }

    #[getter]
    pub fn triplanar<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.triplanar.clone_ref(py)
    }

    #[getter]
    pub fn pom<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.pom.clone_ref(py)
    }

    #[getter]
    pub fn lod<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.lod.clone_ref(py)
    }

    #[getter]
    pub fn sampling<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.sampling.clone_ref(py)
    }

    #[getter]
    pub fn clamp<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.clamp.clone_ref(py)
    }

    #[getter]
    pub fn python_object<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.python_object.clone_ref(py)
    }

    #[getter]
    pub fn material_map_paths(&self) -> std::collections::BTreeMap<String, String> {
        let mut paths = std::collections::BTreeMap::new();
        if let Some(path) = self.decoded.materials.albedo_path.as_ref() {
            paths.insert("albedo".to_string(), path.clone());
        }
        if let Some(path) = self.decoded.materials.normal_path.as_ref() {
            paths.insert("normal".to_string(), path.clone());
        }
        if let Some(path) = self.decoded.materials.roughness_path.as_ref() {
            paths.insert("roughness".to_string(), path.clone());
        }
        if let Some(path) = self.decoded.materials.mask_path.as_ref() {
            paths.insert("mask".to_string(), path.clone());
        }
        paths
    }

    pub fn screen_space_settings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let settings = &self.decoded.screen_space;
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("enabled", settings.enabled)?;
        dict.set_item("ssao_enabled", settings.ssao_enabled)?;
        dict.set_item("ssao_radius", settings.ssao_radius)?;
        dict.set_item("ssao_intensity", settings.ssao_intensity)?;
        dict.set_item("ssgi_enabled", settings.ssgi_enabled)?;
        dict.set_item("ssgi_intensity", settings.ssgi_intensity)?;
        dict.set_item("ssr_enabled", settings.ssr_enabled)?;
        dict.set_item("ssr_intensity", settings.ssr_intensity)?;
        dict.set_item("taa_enabled", settings.taa_enabled)?;
        dict.set_item("temporal_alpha", settings.temporal_alpha)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainRenderParams(size_px=({},{}) , overlays={}, msaa_samples={})",
            self.size_px.0,
            self.size_px.1,
            self.overlays.len(),
            self.msaa_samples
        )
    }
}
