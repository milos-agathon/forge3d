use super::*;

#[pymethods]
impl TerrainSpike {
    // E1: Enable GPU height mosaic (R32Float atlas) and bind it as the active height texture
    #[pyo3(
        text_signature = "($self, tile_px, tiles_x, tiles_y, fixed_lod=None, filter_linear=True)"
    )]
    pub fn enable_height_mosaic(
        &mut self,
        tile_px: u32,
        tiles_x: u32,
        tiles_y: u32,
        fixed_lod: Option<u32>,
        filter_linear: Option<bool>,
    ) -> PyResult<()> {
        let want_linear = filter_linear.unwrap_or(true);
        let cfg = crate::terrain::stream::MosaicConfig {
            tile_size_px: tile_px,
            tiles_x,
            tiles_y,
            fixed_lod,
        };
        let mosaic = crate::terrain::stream::HeightMosaic::new(&self.device, cfg, want_linear);

        // If filter policy changed (e.g., RG16F fallback enables filtering), recreate pipeline accordingly
        let mosaic_filterable = want_linear;
        if mosaic_filterable != self.height_filterable {
            self.tp = crate::terrain::pipeline::TerrainPipeline::create(
                &self.device,
                TEXTURE_FORMAT,
                NORMAL_FORMAT,
                1,
                None,
                mosaic_filterable,
            );
            // Recreate dependent bind groups with the new layouts
            self.bg0_globals = self.tp.make_bg_globals(&self.device, &self.ubo);
            self.bg2_lut = self.tp.make_bg_lut(
                &self.device,
                &self.colormap_lut.view,
                &self.colormap_lut.sampler,
            );
            let pt_buf_opt = self.page_table.as_ref().map(|pt| &pt.buffer);
            self.bg5_tile = self.tp.make_bg_tile(
                &self.device,
                &self.tile_ubo,
                pt_buf_opt,
                &self.tile_slot_ubo,
                &self.mosaic_params_ubo,
            );
            self.height_filterable = mosaic_filterable;
        }

        // Rebind group(1) to the mosaic texture/sampler
        self.bg1_height = self
            .tp
            .make_bg_height(&self.device, &mosaic.view, &mosaic.sampler);
        // Upload mosaic params (inv tiles and dims)
        let inv_tiles_x = if tiles_x > 0 {
            1.0 / tiles_x as f32
        } else {
            1.0
        };
        let inv_tiles_y = if tiles_y > 0 {
            1.0 / tiles_y as f32
        } else {
            1.0
        };
        let mp = MosaicParamsCPU {
            inv_tiles_x,
            inv_tiles_y,
            tiles_x,
            tiles_y,
        };
        self.queue
            .write_buffer(&self.mosaic_params_ubo, 0, bytemuck::bytes_of(&mp));
        // We no longer mirror handles in height_view/sampler (bind group is authoritative)
        self.height_view = None;
        self.height_sampler = None;
        self.height_mosaic = Some(mosaic);
        Ok(())
    }

    // E1: Stream tiles visible at a fixed LOD into the height mosaic; returns list of visible tiles
    #[pyo3(
        text_signature = "($self, camera_pos, camera_dir, lod, fov_deg=45.0, aspect=1.0, near=0.1, far=1000.0, max_uploads=8)"
    )]
    pub fn stream_tiles_to_height_mosaic_at_lod(
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

        let tiling = self.tiling_system.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;
        let mosaic = self.height_mosaic.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Height mosaic not enabled. Call enable_height_mosaic() first.",
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

        // Enumerate visible tiles at requested LOD
        let mut visible_ids = tiling.get_visible_tiles_at_lod(&frustum, lod);
        // Priority: near-to-far (distance to tile center)
        visible_ids.sort_by(|a, b| {
            let ba = QuadTreeNode::calculate_bounds(&tiling.root_bounds, *a, tiling.tile_size);
            let bb = QuadTreeNode::calculate_bounds(&tiling.root_bounds, *b, tiling.tile_size);
            let da = glam::Vec2::new(frustum.position.x, frustum.position.z).distance(ba.center());
            let db = glam::Vec2::new(frustum.position.x, frustum.position.z).distance(bb.center());
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        // E1e: cancellation of height requests that left visibility
        if let Some(loader) = self.async_loader.as_ref() {
            let to_cancel: Vec<TileId> = self
                .prev_visible_height
                .difference(&visible_ids.iter().copied().collect())
                .copied()
                .collect();
            if !to_cancel.is_empty() {
                let _ = loader.cancel(&to_cancel);
            }
        }
        let max_uploads = max_uploads.unwrap_or(8).max(0) as usize;
        let mut uploaded = 0usize;

        // Request loads for missing tiles (non-blocking) and upload any cached tiles immediately
        for id in &visible_ids {
            let present = mosaic.slot_of(id).is_some();
            if !present {
                // Request async load if not in cache yet
                let in_cache = tiling.get_tile_data(id).is_some();
                if !in_cache {
                    if let Some(loader) = self.async_loader.as_ref() {
                        let _ = loader.request(*id);
                    } else {
                        // Fallback: synchronous load
                        tiling
                            .load_tile(*id)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    }
                }
                // If now in cache and budget allows, upload
                if uploaded < max_uploads {
                    if let Some(td) = tiling.get_tile_data(id) {
                        if (td.width * td.height) as usize != td.height_data.len() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "Tile height buffer size mismatch",
                            ));
                        }
                        mosaic
                            .upload_tile(&self.queue, *id, &td.height_data)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                        uploaded += 1;
                    }
                }
            }
            if mosaic.slot_of(id).is_some() {
                mosaic.mark_used(*id);
            }
        }

        // Drain async completions and upload under remaining budget
        if let Some(loader) = self.async_loader.as_ref() {
            let to_drain = max_uploads.saturating_sub(uploaded);
            if to_drain > 0 {
                let completed = loader.drain_completed(to_drain);
                for td in completed {
                    let id = td.tile_id;
                    // Insert into cache (ignore errors, continue)
                    let _ = tiling.insert_tile_data(td);
                    // Upload if visible and not yet present
                    if uploaded >= max_uploads {
                        break;
                    }
                    if visible_ids.iter().any(|t| *t == id) && mosaic.slot_of(&id).is_none() {
                        if let Some(td2) = tiling.get_tile_data(&id) {
                            if (td2.width * td2.height) as usize == td2.height_data.len() {
                                mosaic
                                    .upload_tile(&self.queue, id, &td2.height_data)
                                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                                uploaded += 1;
                                mosaic.mark_used(id);
                            }
                        }
                    }
                }
            }

            // E1g: Prefetch heuristics — request 4-neighborhood tiles around visible set (same LOD)
            use std::collections::HashSet;
            let vis_set: HashSet<(u32, u32, u32)> =
                visible_ids.iter().map(|t| (t.lod, t.x, t.y)).collect();
            let n = 1u32 << lod;
            let (pending, max_in_flight, _pool) = loader.stats();
            let mut in_flight_budget = max_in_flight.saturating_sub(pending);
            if in_flight_budget > 0 {
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
                        if tiling.get_tile_data(&nid).is_some() {
                            continue;
                        }
                        if loader.request(nid) {
                            in_flight_budget = in_flight_budget.saturating_sub(1);
                        }
                    }
                }
            }
        }

        // E1: If page table is enabled, sync it from the current mosaic state
        if let Some(ref mut pt) = self.page_table {
            if let Some(ref mosaic) = self.height_mosaic {
                pt.sync_from_mosaic(&self.queue, mosaic);
            }
        }

        // Update previous visible set for cancellation
        let metrics = global_tracker().get_metrics();
        if !metrics.within_budget {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Host-visible memory budget exceeded: {} / {} bytes",
                metrics.host_visible_bytes, metrics.limit_bytes
            )));
        }

        self.prev_visible_height = visible_ids.iter().copied().collect();
        let result: Vec<(u32, u32, u32)> = visible_ids.iter().map(|t| (t.lod, t.x, t.y)).collect();
        Ok(result)
    }
}
