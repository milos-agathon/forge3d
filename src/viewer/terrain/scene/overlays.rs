use super::*;

impl ViewerTerrainScene {
    fn ensure_overlay_stack(&mut self) {
        if self.overlay_stack.is_none() {
            self.overlay_stack = Some(crate::viewer::terrain::overlay::OverlayStack::new(
                self.device.clone(),
                self.queue.clone(),
            ));
        }
    }

    pub(crate) fn stage_review_raster_replacement(
        &self,
        removed_ids: &[u32],
        overlays: &[crate::viewer::scene_review::ViewerRasterOverlayConfig],
    ) -> Result<(super::super::overlay::OverlayStack, Vec<u32>)> {
        let additions = overlays
            .iter()
            .map(|overlay| {
                super::super::overlay::OverlayStack::load_image_config(
                    &overlay.name,
                    std::path::Path::new(&overlay.path),
                    overlay.extent,
                    overlay.opacity.unwrap_or(1.0),
                    crate::viewer::terrain::overlay::BlendMode::Normal,
                    overlay.z_order.unwrap_or(0),
                )
                .map_err(anyhow::Error::msg)
            })
            .collect::<Result<Vec<_>>>()?;
        let (mut candidate, ids) = if let Some(stack) = self.overlay_stack.as_ref() {
            stack.stage_replacement(removed_ids, additions)
        } else {
            let empty =
                super::super::overlay::OverlayStack::new(self.device.clone(), self.queue.clone());
            empty.stage_replacement(removed_ids, additions)
        };
        if let Some(terrain) = self.terrain.as_ref() {
            candidate.build_composite(terrain.dimensions.0, terrain.dimensions.1)?;
        }
        Ok((candidate, ids))
    }

    pub(crate) fn stage_review_vector_replacement(
        &self,
        removed_ids: &[u32],
        mut layers: Vec<VectorOverlayLayer>,
        anchor: &crate::camera::Anchor,
    ) -> Result<(VectorOverlayStack, Vec<u32>)> {
        for layer in &mut layers {
            crate::viewer::terrain::vector_overlay::repack_source_vertices(
                layer,
                self.terrain.as_ref(),
                anchor,
            );
        }
        if let Some(stack) = self.vector_overlay_stack.as_ref() {
            stack.stage_replacement(removed_ids, layers)
        } else {
            VectorOverlayStack::new(self.device.clone(), self.queue.clone())
                .stage_replacement(removed_ids, layers)
        }
    }

    pub(crate) fn commit_review_stacks(
        &mut self,
        raster: super::super::overlay::OverlayStack,
        vectors: VectorOverlayStack,
    ) {
        self.pbr_config.overlay.enabled = raster.has_visible_layers();
        self.overlay_stack = Some(raster);
        self.vector_overlay_stack = Some(vectors);
    }

    /// Add an overlay layer from an image file. Returns layer ID or error.
    pub fn add_overlay_image(
        &mut self,
        name: &str,
        path: &std::path::Path,
        extent: Option<[f32; 4]>,
        opacity: f32,
        blend_mode: crate::viewer::terrain::overlay::BlendMode,
        z_order: i32,
    ) -> Result<u32> {
        self.ensure_overlay_stack();
        if let Some(ref mut stack) = self.overlay_stack {
            let id = stack
                .add_image(name, path, extent, opacity, blend_mode, z_order)
                .map_err(|e| anyhow::anyhow!(e))?;
            // Rebuild composite after adding layer
            if let Some(ref terrain) = self.terrain {
                if let Err(e) = stack.build_composite(terrain.dimensions.0, terrain.dimensions.1) {
                    eprintln!("[terrain] overlay composite rebuild failed: {e}");
                }
            }
            // Enable overlay system in config
            self.pbr_config.overlay.enabled = true;
            Ok(id)
        } else {
            Err(anyhow::anyhow!("Overlay stack not initialized"))
        }
    }

    /// Remove an overlay by ID. Returns true if found and removed.
    pub fn remove_overlay(&mut self, id: u32) -> bool {
        if let Some(ref mut stack) = self.overlay_stack {
            let removed = stack.remove(id);
            if removed {
                // Rebuild composite after removing layer
                if let Some(ref terrain) = self.terrain {
                    if let Err(e) =
                        stack.build_composite(terrain.dimensions.0, terrain.dimensions.1)
                    {
                        eprintln!("[terrain] overlay composite rebuild failed: {e}");
                    }
                }
                // Disable overlay system if no visible layers
                if !stack.has_visible_layers() {
                    self.pbr_config.overlay.enabled = false;
                }
            }
            removed
        } else {
            false
        }
    }

    /// Set overlay visibility
    pub fn set_overlay_visible(&mut self, id: u32, visible: bool) {
        if let Some(ref mut stack) = self.overlay_stack {
            stack.set_visible(id, visible);
            // Rebuild composite
            if let Some(ref terrain) = self.terrain {
                if let Err(e) = stack.build_composite(terrain.dimensions.0, terrain.dimensions.1) {
                    eprintln!("[terrain] overlay composite rebuild failed: {e}");
                }
            }
            // Update enabled state
            self.pbr_config.overlay.enabled = stack.has_visible_layers();
        }
    }

    /// Set overlay opacity (0.0 - 1.0)
    pub fn set_overlay_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(ref mut stack) = self.overlay_stack {
            stack.set_opacity(id, opacity);
            // Rebuild composite
            if let Some(ref terrain) = self.terrain {
                if let Err(e) = stack.build_composite(terrain.dimensions.0, terrain.dimensions.1) {
                    eprintln!("[terrain] overlay composite rebuild failed: {e}");
                }
            }
        }
    }

    /// Get list of all overlay IDs in z-order
    pub fn list_overlays(&self) -> Vec<u32> {
        if let Some(ref stack) = self.overlay_stack {
            stack.list_ids()
        } else {
            Vec::new()
        }
    }

    /// Set global overlay opacity multiplier (0.0 - 1.0)
    pub fn set_global_overlay_opacity(&mut self, opacity: f32) {
        self.pbr_config.overlay.global_opacity = opacity.clamp(0.0, 1.0);
    }

    /// Enable or disable the overlay system
    pub fn set_overlays_enabled(&mut self, enabled: bool) {
        self.pbr_config.overlay.enabled = enabled;
    }

    /// Set overlay solid surface mode (true=show base surface, false=hide where alpha=0)
    pub fn set_overlay_solid(&mut self, solid: bool) {
        self.pbr_config.overlay.solid = solid;
    }

    /// Preserve source raster colors by compositing after terrain lighting.
    pub fn set_overlay_preserve_colors(&mut self, preserve_colors: bool) {
        self.pbr_config.overlay.preserve_colors = preserve_colors;
    }

    // === VECTOR OVERLAY (OPTION B) MANAGEMENT API ===

    /// Initialize the vector overlay stack if not already initialized
    fn ensure_vector_overlay_stack(&mut self) {
        if self.vector_overlay_stack.is_none() {
            self.vector_overlay_stack = Some(VectorOverlayStack::new(
                self.device.clone(),
                self.queue.clone(),
            ));
        }
    }

    /// Add a vector overlay layer with an externally allocated ID.
    /// If drape is true and terrain is loaded, vertices will be draped onto terrain.
    pub fn add_vector_overlay_with_id(
        &mut self,
        id: Option<u32>,
        mut layer: VectorOverlayLayer,
        anchor: &crate::camera::Anchor,
    ) -> Result<u32> {
        self.ensure_vector_overlay_stack();
        crate::viewer::terrain::vector_overlay::repack_source_vertices(
            &mut layer,
            self.terrain.as_ref(),
            anchor,
        );

        if let Some(ref mut stack) = self.vector_overlay_stack {
            match id {
                Some(id) => stack.add_layer_with_id(Some(id), layer),
                None => stack.add_layer(layer),
            }
        } else {
            Ok(0)
        }
    }

    pub(crate) fn repack_vector_overlays(&mut self, anchor: &crate::camera::Anchor) {
        if let Some(stack) = self.vector_overlay_stack.as_mut() {
            stack.repack_for_anchor(self.terrain.as_ref(), anchor);
        }
    }

    pub(crate) fn vector_overlay_render_data(
        &self,
        id: u32,
    ) -> Option<(
        String,
        Vec<crate::viewer::terrain::vector_overlay::VectorVertex>,
        Vec<u32>,
        crate::viewer::terrain::vector_overlay::OverlayPrimitive,
    )> {
        self.vector_overlay_stack
            .as_ref()?
            .render_layer_data()
            .find_map(|(layer_id, name, vertices, indices, primitive)| {
                (layer_id == id).then(|| {
                    (
                        name.to_string(),
                        vertices.to_vec(),
                        indices.to_vec(),
                        primitive,
                    )
                })
            })
    }

    pub(crate) fn all_vector_overlay_render_data(
        &self,
    ) -> Vec<(
        u32,
        String,
        Vec<crate::viewer::terrain::vector_overlay::VectorVertex>,
        Vec<u32>,
        crate::viewer::terrain::vector_overlay::OverlayPrimitive,
    )> {
        self.vector_overlay_stack
            .as_ref()
            .map(|stack| {
                stack
                    .render_layer_data()
                    .map(|(id, name, vertices, indices, primitive)| {
                        (
                            id,
                            name.to_string(),
                            vertices.to_vec(),
                            indices.to_vec(),
                            primitive,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn all_vector_overlay_source_points(&self) -> Vec<glam::DVec3> {
        self.vector_overlay_stack
            .as_ref()
            .map_or_else(Vec::new, |stack| stack.source_world_points())
    }

    pub(crate) fn vector_overlay_memory_evidence(&self) -> (u64, u64, u64, Vec<u64>) {
        self.vector_overlay_stack
            .as_ref()
            .map_or_else(|| (0, 0, 0, Vec::new()), |stack| stack.memory_evidence())
    }

    /// Remove a vector overlay by ID. Returns true if found and removed.
    pub fn remove_vector_overlay(&mut self, id: u32) -> bool {
        if let Some(ref mut stack) = self.vector_overlay_stack {
            stack.remove(id)
        } else {
            false
        }
    }

    /// Set vector overlay visibility
    pub fn set_vector_overlay_visible(&mut self, id: u32, visible: bool) {
        if let Some(ref mut stack) = self.vector_overlay_stack {
            stack.set_visible(id, visible);
        }
    }

    /// Set vector overlay opacity (0.0 - 1.0)
    pub fn set_vector_overlay_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(ref mut stack) = self.vector_overlay_stack {
            stack.set_opacity(id, opacity);
        }
    }

    /// List all vector overlay IDs in z-order
    pub fn list_vector_overlays(&self) -> Vec<u32> {
        if let Some(ref stack) = self.vector_overlay_stack {
            stack.list_ids()
        } else {
            Vec::new()
        }
    }

    /// Enable or disable the vector overlay system
    pub fn set_vector_overlays_enabled(&mut self, enabled: bool) {
        if let Some(ref mut stack) = self.vector_overlay_stack {
            stack.set_enabled(enabled);
        }
    }

    /// Set global vector overlay opacity multiplier (0.0 - 1.0)
    pub fn set_global_vector_overlay_opacity(&mut self, opacity: f32) {
        if let Some(ref mut stack) = self.vector_overlay_stack {
            stack.set_global_opacity(opacity);
        }
    }

    // ensure_depth and render moved to terrain/render.rs
}
