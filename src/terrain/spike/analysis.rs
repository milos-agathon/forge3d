use super::*;

#[pymethods]
impl TerrainSpike {
    // B12: Calculate screen-space error for a tile
    #[pyo3(
        text_signature = "($self, tile_lod, tile_x, tile_y, camera_pos, camera_target, camera_up, fov_deg=45.0, viewport_width=1024, viewport_height=768, pixel_error_budget=2.0)"
    )]
    pub fn calculate_screen_space_error(
        &self,
        tile_lod: u32,
        tile_x: u32,
        tile_y: u32,
        camera_pos: (f32, f32, f32),
        camera_target: (f32, f32, f32),
        camera_up: (f32, f32, f32),
        fov_deg: Option<f32>,
        viewport_width: Option<u32>,
        viewport_height: Option<u32>,
        pixel_error_budget: Option<f32>,
    ) -> PyResult<(f32, f32, bool)> {
        use crate::terrain::lod::{screen_space_error, LodConfig};
        use glam::Vec3;

        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let tile_id = TileId::new(tile_lod, tile_x, tile_y);
        let root_bounds = &tiling_system.root_bounds;
        let tile_bounds =
            QuadTreeNode::calculate_bounds(root_bounds, tile_id, tiling_system.tile_size);

        let eye = Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2);
        let target = Vec3::new(camera_target.0, camera_target.1, camera_target.2);
        let up = Vec3::new(camera_up.0, camera_up.1, camera_up.2);

        let view = crate::terrain::lod::create_view_matrix(eye, target, up);
        let fov_rad = fov_deg.unwrap_or(45.0).to_radians();
        let vp_width = viewport_width.unwrap_or(1024);
        let vp_height = viewport_height.unwrap_or(768);
        let aspect = vp_width as f32 / vp_height as f32;
        let proj = crate::terrain::lod::create_projection_matrix(fov_rad, aspect, 0.1, 1000.0);

        let config = LodConfig::new(
            pixel_error_budget.unwrap_or(2.0),
            vp_width,
            vp_height,
            fov_rad,
        );

        let sse = screen_space_error(&tile_bounds, tile_id, eye, view, proj, &config);

        Ok((sse.edge_length_pixels, sse.error_pixels, sse.within_budget))
    }

    // B12: Select appropriate LOD for a tile based on screen-space error
    #[pyo3(
        text_signature = "($self, base_tile_lod, base_tile_x, base_tile_y, camera_pos, camera_target, camera_up, fov_deg=45.0, viewport_width=1024, viewport_height=768, pixel_error_budget=2.0, max_lod=4)"
    )]
    pub fn select_lod_for_tile(
        &self,
        base_tile_lod: u32,
        base_tile_x: u32,
        base_tile_y: u32,
        camera_pos: (f32, f32, f32),
        camera_target: (f32, f32, f32),
        camera_up: (f32, f32, f32),
        fov_deg: Option<f32>,
        viewport_width: Option<u32>,
        viewport_height: Option<u32>,
        pixel_error_budget: Option<f32>,
        max_lod: Option<u32>,
    ) -> PyResult<(u32, u32, u32)> {
        use crate::terrain::lod::{select_lod_for_tile, LodConfig};
        use glam::Vec3;

        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let base_tile_id = TileId::new(base_tile_lod, base_tile_x, base_tile_y);
        let root_bounds = &tiling_system.root_bounds;
        let tile_bounds =
            QuadTreeNode::calculate_bounds(root_bounds, base_tile_id, tiling_system.tile_size);

        let eye = Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2);
        let target = Vec3::new(camera_target.0, camera_target.1, camera_target.2);
        let up = Vec3::new(camera_up.0, camera_up.1, camera_up.2);

        let view = crate::terrain::lod::create_view_matrix(eye, target, up);
        let fov_rad = fov_deg.unwrap_or(45.0).to_radians();
        let vp_width = viewport_width.unwrap_or(1024);
        let vp_height = viewport_height.unwrap_or(768);
        let aspect = vp_width as f32 / vp_height as f32;
        let proj = crate::terrain::lod::create_projection_matrix(fov_rad, aspect, 0.1, 1000.0);

        let config = LodConfig::new(
            pixel_error_budget.unwrap_or(2.0),
            vp_width,
            vp_height,
            fov_rad,
        );

        let selected_tile = select_lod_for_tile(
            &tile_bounds,
            base_tile_id,
            eye,
            view,
            proj,
            &config,
            max_lod.unwrap_or(4),
        );

        Ok((selected_tile.lod, selected_tile.x, selected_tile.y))
    }

    // B12: Calculate triangle count reduction for LOD comparison
    #[pyo3(text_signature = "($self, full_res_tiles, lod_tiles, base_triangles_per_tile=1000)")]
    pub fn calculate_triangle_reduction(
        &self,
        full_res_tiles: Vec<(u32, u32, u32)>,
        lod_tiles: Vec<(u32, u32, u32)>,
        base_triangles_per_tile: Option<u32>,
    ) -> PyResult<f32> {
        use crate::terrain::lod::calculate_triangle_reduction;

        let full_res: Vec<TileId> = full_res_tiles
            .into_iter()
            .map(|(lod, x, y)| TileId::new(lod, x, y))
            .collect();

        let lod: Vec<TileId> = lod_tiles
            .into_iter()
            .map(|(lod, x, y)| TileId::new(lod, x, y))
            .collect();

        let base_triangles = base_triangles_per_tile.unwrap_or(1000);
        let reduction = calculate_triangle_reduction(&full_res, &lod, base_triangles);

        Ok(reduction)
    }

    // B13: Compute slope and aspect for height field
    #[pyo3(text_signature = "($self, heights, width, height, dx=1.0, dy=1.0)")]
    pub fn slope_aspect_compute<'py>(
        &self,
        py: pyo3::Python<'py>,
        heights: numpy::PyReadonlyArray1<f32>,
        width: u32,
        height: u32,
        dx: Option<f32>,
        dy: Option<f32>,
    ) -> pyo3::PyResult<(
        pyo3::Bound<'py, numpy::PyArray1<f32>>,
        pyo3::Bound<'py, numpy::PyArray1<f32>>,
    )> {
        use crate::terrain::analysis::slope_aspect_compute;
        use numpy::PyUntypedArrayMethods;

        // Validate input array
        if !heights.is_c_contiguous() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Heights array must be C-contiguous. Use np.ascontiguousarray().",
            ));
        }

        let heights_slice = heights.as_slice()?;
        let expected_len = (width * height) as usize;

        if heights_slice.len() != expected_len {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Heights array length {} does not match dimensions {}x{}={}",
                heights_slice.len(),
                width,
                height,
                expected_len
            )));
        }

        let dx = dx.unwrap_or(1.0);
        let dy = dy.unwrap_or(1.0);

        // Compute slope and aspect
        let result = slope_aspect_compute(heights_slice, width as usize, height as usize, dx, dy)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Extract slopes and aspects into separate arrays
        let mut slopes = Vec::with_capacity(result.len());
        let mut aspects = Vec::with_capacity(result.len());

        for sa in result {
            slopes.push(sa.slope_deg);
            aspects.push(sa.aspect_deg);
        }

        // Convert to NumPy arrays
        let slopes_arr = ndarray::Array1::from_vec(slopes);
        let aspects_arr = ndarray::Array1::from_vec(aspects);

        Ok((
            slopes_arr.into_pyarray_bound(py),
            aspects_arr.into_pyarray_bound(py),
        ))
    }

    // B14: Extract contour lines from height field
    #[pyo3(signature = (heights, width, height, /, dx=1.0, dy=1.0, *, levels))]
    pub fn contour_extract<'py>(
        &self,
        py: pyo3::Python<'py>,
        heights: numpy::PyReadonlyArray1<f32>,
        width: u32,
        height: u32,
        dx: Option<f32>,
        dy: Option<f32>,
        levels: Vec<f32>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use crate::terrain::analysis::contour_extract;
        use numpy::PyUntypedArrayMethods;

        // Validate input array
        if !heights.is_c_contiguous() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Heights array must be C-contiguous. Use np.ascontiguousarray().",
            ));
        }

        let heights_slice = heights.as_slice()?;
        let expected_len = (width * height) as usize;

        if heights_slice.len() != expected_len {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Heights array length {} does not match dimensions {}x{}={}",
                heights_slice.len(),
                width,
                height,
                expected_len
            )));
        }

        if levels.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "At least one contour level must be specified",
            ));
        }

        let dx = dx.unwrap_or(1.0);
        let dy = dy.unwrap_or(1.0);

        // Extract contours
        let result = contour_extract(
            heights_slice,
            width as usize,
            height as usize,
            dx,
            dy,
            &levels,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Build Python result dictionary
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("polyline_count", result.polyline_count)?;
        dict.set_item("total_points", result.total_points)?;

        // Convert polylines to Python format
        let polylines_list = pyo3::types::PyList::empty_bound(py);
        for polyline in result.polylines {
            let polyline_dict = pyo3::types::PyDict::new_bound(py);
            polyline_dict.set_item("level", polyline.level)?;

            // Convert points to NumPy array (Nx2)
            let points_flat: Vec<f32> = polyline
                .points
                .iter()
                .flat_map(|(x, y)| vec![*x, *y])
                .collect();

            if !points_flat.is_empty() {
                let points_arr =
                    ndarray::Array2::from_shape_vec((polyline.points.len(), 2), points_flat)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                polyline_dict.set_item("points", points_arr.into_pyarray_bound(py))?;
            } else {
                // Empty points array (0x2)
                let empty_arr = ndarray::Array2::<f32>::zeros((0, 2));
                polyline_dict.set_item("points", empty_arr.into_pyarray_bound(py))?;
            }

            polylines_list.append(polyline_dict)?;
        }

        dict.set_item("polylines", polylines_list)?;

        Ok(dict)
    }
}
