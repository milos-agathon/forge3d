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

    /// Add an overlay layer from raw RGBA data. Returns layer ID.
    pub fn add_overlay_raster(
        &mut self,
        name: &str,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        extent: Option<[f32; 4]>,
        opacity: f32,
        blend_mode: crate::viewer::terrain::overlay::BlendMode,
        z_order: i32,
    ) -> u32 {
        self.ensure_overlay_stack();
        if let Some(ref mut stack) = self.overlay_stack {
            // Build composite at terrain resolution
            if let Some(ref terrain) = self.terrain {
                stack.build_composite(terrain.dimensions.0, terrain.dimensions.1);
            }
            let id = stack.add_raster(
                name, rgba, width, height, extent, opacity, blend_mode, z_order,
            );
            // Rebuild composite after adding layer
            if let Some(ref terrain) = self.terrain {
                stack.build_composite(terrain.dimensions.0, terrain.dimensions.1);
            }
            // Enable overlay system in config
            self.pbr_config.overlay.enabled = true;
            id
        } else {
            0
        }
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
                stack.build_composite(terrain.dimensions.0, terrain.dimensions.1);
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
                    stack.build_composite(terrain.dimensions.0, terrain.dimensions.1);
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
                stack.build_composite(terrain.dimensions.0, terrain.dimensions.1);
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
                stack.build_composite(terrain.dimensions.0, terrain.dimensions.1);
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

    /// Get number of overlay layers
    pub fn overlay_count(&self) -> usize {
        if let Some(ref stack) = self.overlay_stack {
            stack.len()
        } else {
            0
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

    /// Add a vector overlay layer. Returns layer ID.
    /// If drape is true and terrain is loaded, vertices will be draped onto terrain.
    pub fn add_vector_overlay(&mut self, mut layer: VectorOverlayLayer) -> u32 {
        self.ensure_vector_overlay_stack();

        // If draping requested and terrain is loaded, drape the vertices
        if layer.drape {
            if let Some(ref terrain) = self.terrain {
                let terrain_width = terrain.dimensions.0.max(terrain.dimensions.1) as f32;
                let height_range = terrain.domain.1 - terrain.domain.0;
                // Match terrain shader formula: world_y = (h - min_h) / h_range * terrain_width * z_scale * 0.001
                let height_scale = terrain_width * terrain.z_scale * 0.001 / height_range.max(1.0);

                drape_vertices(crate::viewer::terrain::vector_overlay::DrapeParams {
                    vertices: &mut layer.vertices,
                    heightmap: &terrain.heightmap,
                    dims: terrain.dimensions,
                    terrain_width,
                    terrain_origin: (0.0, 0.0),
                    height_offset: layer.drape_offset,
                    height_min: terrain.domain.0,
                    height_scale,
                });
            }
        }

        if let Some(ref mut stack) = self.vector_overlay_stack {
            stack.add_layer(layer)
        } else {
            0
        }
    }

    /// Update vertices for an existing layer (for animation)
    pub fn update_vector_overlay(
        &mut self,
        id: u32,
        mut vertices: Vec<VectorVertex>,
        drape: bool,
        drape_offset: f32,
    ) {
        // If draping requested and terrain is loaded, drape the vertices
        if drape {
            if let Some(ref terrain) = self.terrain {
                let terrain_width = terrain.dimensions.0.max(terrain.dimensions.1) as f32;
                let height_range = terrain.domain.1 - terrain.domain.0;
                // Match terrain shader formula: world_y = (h - min_h) / h_range * terrain_width * z_scale * 0.001
                let height_scale = terrain_width * terrain.z_scale * 0.001 / height_range.max(1.0);

                drape_vertices(crate::viewer::terrain::vector_overlay::DrapeParams {
                    vertices: &mut vertices,
                    heightmap: &terrain.heightmap,
                    dims: terrain.dimensions,
                    terrain_width,
                    terrain_origin: (0.0, 0.0),
                    height_offset: drape_offset,
                    height_min: terrain.domain.0,
                    height_scale,
                });
            }
        }

        if let Some(ref mut stack) = self.vector_overlay_stack {
            stack.update_vertices(id, vertices);
        }
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

    /// Get number of vector overlay layers
    pub fn vector_overlay_count(&self) -> usize {
        if let Some(ref stack) = self.vector_overlay_stack {
            stack.len()
        } else {
            0
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

    /// Check if vector overlays are available
    pub fn has_vector_overlays(&self) -> bool {
        if let Some(ref stack) = self.vector_overlay_stack {
            stack.has_visible_layers()
        } else {
            false
        }
    }

    // ensure_depth and render moved to terrain/render.rs
}
