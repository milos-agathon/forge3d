use super::*;

#[pymethods]
impl TerrainSpike {
    // B11: Enable tiled DEM system
    #[pyo3(
        text_signature = "($self, bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y, cache_capacity=4, max_lod=4)"
    )]
    pub fn enable_tiling(
        &mut self,
        bounds_min_x: f32,
        bounds_min_y: f32,
        bounds_max_x: f32,
        bounds_max_y: f32,
        cache_capacity: Option<usize>,
        max_lod: Option<u32>,
    ) -> PyResult<()> {
        use glam::Vec2;

        let root_bounds = TileBounds::new(
            Vec2::new(bounds_min_x, bounds_min_y),
            Vec2::new(bounds_max_x, bounds_max_y),
        );

        let capacity = cache_capacity.unwrap_or(4);
        let max_lod = max_lod.unwrap_or(4);
        let tile_size = Vec2::new(1000.0, 1000.0); // Default 1km tiles

        let tiling_system = TilingSystem::new(root_bounds, capacity, max_lod, tile_size);
        self.tiling_system = Some(tiling_system);

        Ok(())
    }

    // B11: Naming shim for deliverable requirement - forwards to enable_tiling()
    #[pyo3(
        text_signature = "($self, bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y, cache_capacity=4, max_lod=4)"
    )]
    pub fn set_height_tiled(
        &mut self,
        bounds_min_x: f32,
        bounds_min_y: f32,
        bounds_max_x: f32,
        bounds_max_y: f32,
        cache_capacity: Option<usize>,
        max_lod: Option<u32>,
    ) -> PyResult<()> {
        // Forward to existing implementation
        self.enable_tiling(
            bounds_min_x,
            bounds_min_y,
            bounds_max_x,
            bounds_max_y,
            cache_capacity,
            max_lod,
        )
    }

    // B11: Get visible tiles for a camera position
    #[pyo3(
        text_signature = "($self, camera_pos, camera_dir, fov_deg=45.0, aspect=1.0, near=0.1, far=1000.0)"
    )]
    pub fn get_visible_tiles(
        &mut self,
        camera_pos: (f32, f32, f32),
        camera_dir: (f32, f32, f32),
        fov_deg: Option<f32>,
        aspect: Option<f32>,
        near: Option<f32>,
        far: Option<f32>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        use glam::Vec3;

        let tiling_system = self.tiling_system.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let frustum = Frustum::new(
            Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2),
            Vec3::new(camera_dir.0, camera_dir.1, camera_dir.2).normalize(),
            fov_deg.unwrap_or(45.0).to_radians(),
            aspect.unwrap_or(1.0),
            near.unwrap_or(0.1),
            far.unwrap_or(1000.0),
        );

        let visible_tiles = tiling_system.get_visible_tiles(&frustum);

        // Convert TileId to Python-friendly tuples (lod, x, y)
        let result: Vec<(u32, u32, u32)> = visible_tiles
            .into_iter()
            .map(|tile_id| (tile_id.lod, tile_id.x, tile_id.y))
            .collect();

        Ok(result)
    }

    // B11: Load a specific tile
    #[pyo3(text_signature = "($self, lod, x, y)")]
    pub fn load_tile(&mut self, lod: u32, x: u32, y: u32) -> PyResult<()> {
        let tiling_system = self.tiling_system.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let tile_id = TileId::new(lod, x, y);
        tiling_system.load_tile(tile_id).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load tile: {}", e))
        })?;

        Ok(())
    }

    // B11: Get cache statistics
    #[pyo3(text_signature = "($self)")]
    pub fn get_cache_stats<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let stats = tiling_system.get_cache_stats();
        let dict = pyo3::types::PyDict::new_bound(py);

        dict.set_item("capacity", stats.capacity)?;
        dict.set_item("current_size", stats.current_size)?;
        dict.set_item("memory_usage_bytes", stats.memory_usage_bytes)?;

        Ok(dict)
    }

    // B11: Stream and load visible tiles for a camera
    #[pyo3(
        text_signature = "($self, camera_pos, camera_dir, fov_deg=45.0, aspect=1.0, near=0.1, far=1000.0)"
    )]
    pub fn stream_visible_tiles(
        &mut self,
        camera_pos: (f32, f32, f32),
        camera_dir: (f32, f32, f32),
        fov_deg: Option<f32>,
        aspect: Option<f32>,
        near: Option<f32>,
        far: Option<f32>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        // Get visible tiles
        let visible_tiles =
            self.get_visible_tiles(camera_pos, camera_dir, fov_deg, aspect, near, far)?;

        // Load each visible tile
        for (lod, x, y) in &visible_tiles {
            if let Err(e) = self.load_tile(*lod, *x, *y) {
                // Log error but continue with other tiles
                eprintln!(
                    "Warning: Failed to load tile ({}, {}, {}): {}",
                    lod, x, y, e
                );
            }
        }

        Ok(visible_tiles)
    }
}
