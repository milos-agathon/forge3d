use super::*;

#[pymethods]
impl TerrainSpike {
    // E3/E1 parity: stream overlay tiles asynchronously into ColorMosaic
    #[pyo3(
        text_signature = "($self, camera_pos, camera_dir, lod, fov_deg=45.0, aspect=1.0, near=0.1, far=1000.0, max_uploads=8)"
    )]
    pub fn stream_tiles_to_overlay_mosaic_at_lod(
        &mut self,
        camera_pos: (f32, f32, f32),
        camera_dir: (f32, f32, f32),
        lod: u32,
        fov_deg: Option<f32>,
        aspect: Option<f32>,
        near: Option<f32>,
        far: Option<f32>,
        max_uploads: Option<usize>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        use glam::Vec3;
        let tiling = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;
        let mosaic = self.overlay_mosaic.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay mosaic not enabled. Call enable_overlay_mosaic() first.",
            )
        })?;
        let loader = self.async_overlay_loader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Async overlay loader not enabled. Call enable_async_overlay_loader() first.",
            )
        })?;

        // Build frustum
        let frustum = Frustum::new(
            Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2),
            Vec3::new(camera_dir.0, camera_dir.1, camera_dir.2).normalize(),
            fov_deg.unwrap_or(45.0).to_radians(),
            aspect.unwrap_or(1.0),
            near.unwrap_or(0.1),
            far.unwrap_or(1000.0),
        );

        // Visible tiles (priority-ordered: near-to-far)
        let mut visible_ids = tiling.get_visible_tiles_at_lod(&frustum, lod);
        visible_ids.sort_by(|a, b| {
            let ba = QuadTreeNode::calculate_bounds(&tiling.root_bounds, *a, tiling.tile_size);
            let bb = QuadTreeNode::calculate_bounds(&tiling.root_bounds, *b, tiling.tile_size);
            let da = glam::Vec2::new(frustum.position.x, frustum.position.z).distance(ba.center());
            let db = glam::Vec2::new(frustum.position.x, frustum.position.z).distance(bb.center());
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Cancel requests that are no longer visible
        {
            let prev: HashSet<TileId> = self.prev_visible_overlay.clone();
            let curr: HashSet<TileId> = visible_ids.iter().copied().collect();
            let to_cancel: Vec<TileId> = prev.difference(&curr).copied().collect();
            if !to_cancel.is_empty() {
                let _ = loader.cancel(&to_cancel);
            }
        }

        // Request loads for missing tiles and drain/upload under budget
        let max_uploads = max_uploads.unwrap_or(8).max(0) as usize;
        let mut uploaded = 0usize;

        for id in &visible_ids {
            if mosaic.slot_of(id).is_none() {
                let _ = loader.request(*id); // dedup/backpressure inside
            } else {
                mosaic.mark_used(*id);
            }
        }

        // Drain and upload
        let completed = loader.drain_completed(max_uploads);
        for td in completed {
            if uploaded >= max_uploads {
                break;
            }
            if visible_ids.iter().any(|t| *t == td.tile_id) && mosaic.slot_of(&td.tile_id).is_none()
            {
                mosaic
                    .upload_tile(&self.queue, td.tile_id, &td.rgba_data)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                mosaic.mark_used(td.tile_id);
                uploaded += 1;
            }
        }

        // Neighbor prefetch (overlay) under in-flight budget
        let (pending, max_in_flight, _pool) = loader.stats();
        let mut in_flight_budget = max_in_flight.saturating_sub(pending);
        if in_flight_budget > 0 {
            let vis_set: HashSet<(u32, u32, u32)> =
                visible_ids.iter().map(|t| (t.lod, t.x, t.y)).collect();
            let n = 1u32 << lod;
            for id in &visible_ids {
                if in_flight_budget == 0 {
                    break;
                }
                let l = id.lod;
                let x = id.x;
                let y = id.y;
                let neighbors = [
                    (l, x.saturating_sub(1), y),
                    (l, x + 1, y),
                    (l, x, y.saturating_sub(1)),
                    (l, x, y + 1),
                ];
                for (nl, nx, ny) in neighbors {
                    if in_flight_budget == 0 {
                        break;
                    }
                    if nx >= n || ny >= n {
                        continue;
                    }
                    if vis_set.contains(&(nl, nx, ny)) {
                        continue;
                    }
                    let nid = TileId::new(nl, nx, ny);
                    if mosaic.slot_of(&nid).is_some() {
                        continue;
                    }
                    if loader.request(nid) {
                        in_flight_budget = in_flight_budget.saturating_sub(1);
                    }
                }
            }
        }

        // Update previous visible overlay set
        self.prev_visible_overlay = visible_ids.iter().copied().collect();
        Ok(visible_ids.iter().map(|t| (t.lod, t.x, t.y)).collect())
    }

    // E1: Enable page table scaffolding (GPU buffer updated from current height mosaic)
    #[pyo3(text_signature = "($self)")]
    pub fn enable_page_table(&mut self) -> PyResult<()> {
        let mosaic = self.height_mosaic.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Height mosaic not enabled. Call enable_height_mosaic() first.",
            )
        })?;
        let capacity = (mosaic.config.tiles_x * mosaic.config.tiles_y) as usize;
        let pt = PageTable::new(&self.device, capacity);
        // initial sync
        let mut pt = pt;
        pt.sync_from_mosaic(&self.queue, mosaic);
        self.page_table = Some(pt);
        // Recreate tile bind group to include the page table buffer at binding(1)
        let pt_buf_opt = self.page_table.as_ref().map(|pt| &pt.buffer);
        self.bg5_tile = self.tp.make_bg_tile(
            &self.device,
            &self.tile_ubo,
            pt_buf_opt,
            &self.tile_slot_ubo,
            &self.mosaic_params_ubo,
        );
        Ok(())
    }
}
