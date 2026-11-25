use super::OverlayStack;
use crate::gis::raster_write::{CrsSpec, RasterData};
use crate::gis::types::{PixelRegistration, RasterInfo};
use crate::viewer::terrain::overlay::{BlendMode, OverlayData, OverlayLayer, OverlayLayerGpu};

impl OverlayStack {
    /// Create a new overlay stack
    pub fn new(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay_stack.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        Self {
            device,
            queue,
            layers: Vec::new(),
            registered_grids: Default::default(),
            next_id: 0,
            composite_texture: None,
            composite_view: None,
            composite_dimensions: (0, 0),
            dirty: true,
            sampler,
        }
    }

    /// Add an overlay layer from an image file. Returns layer ID or error.
    pub fn add_image(
        &mut self,
        name: &str,
        path: &std::path::Path,
        extent: Option<[f32; 4]>,
        opacity: f32,
        blend_mode: BlendMode,
        z_order: i32,
    ) -> Result<u32, String> {
        let config = Self::load_image_config(name, path, extent, opacity, blend_mode, z_order)?;
        Ok(self.add_layer_config(config))
    }

    pub(crate) fn load_image_config(
        name: &str,
        path: &std::path::Path,
        extent: Option<[f32; 4]>,
        opacity: f32,
        blend_mode: BlendMode,
        z_order: i32,
    ) -> Result<OverlayLayer, String> {
        let (rgba, width, height) = if is_tiff(path) {
            read_tiff_rgba(path).map_err(|err| err.to_string())?
        } else {
            let img = image::open(path)
                .map_err(|e| format!("Failed to load overlay image '{}': {e}", path.display()))?;
            let rgba_img = img.to_rgba8();
            let (width, height) = rgba_img.dimensions();
            (rgba_img.into_raw(), width, height)
        };
        Ok(OverlayLayer {
            name: name.to_string(),
            data: OverlayData::Raster {
                rgba,
                width,
                height,
            },
            extent,
            opacity,
            blend_mode,
            visible: true,
            z_order,
        })
    }

    fn add_layer_config(&mut self, config: OverlayLayer) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let (width, height) = match &config.data {
            OverlayData::Raster { width, height, .. } => (*width, *height),
        };
        self.layers.push(OverlayLayerGpu { id, config });
        self.dirty = true;
        println!(
            "[overlay] Added layer '{}' (id={}, {}x{})",
            self.layers.last().unwrap().config.name,
            id,
            width,
            height
        );
        id
    }

    /// Build a detached full-stack candidate, retaining unmanaged layers and
    /// replacing only the requested managed IDs. The live allocator and
    /// composite remain untouched until the caller swaps this candidate.
    pub(crate) fn stage_replacement(
        &self,
        removed_ids: &[u32],
        additions: Vec<OverlayLayer>,
    ) -> (Self, Vec<u32>) {
        let mut candidate = Self::new(self.device.clone(), self.queue.clone());
        candidate.next_id = self.next_id;
        candidate.layers = self
            .layers
            .iter()
            .filter(|layer| !removed_ids.contains(&layer.id))
            .map(|layer| OverlayLayerGpu {
                id: layer.id,
                config: layer.config.clone(),
            })
            .collect();
        candidate.registered_grids = self
            .registered_grids
            .iter()
            .filter(|(id, _)| !removed_ids.contains(id))
            .map(|(&id, info)| (id, info.clone()))
            .collect();
        let ids = additions
            .into_iter()
            .map(|config| candidate.add_layer_config(config))
            .collect();
        (candidate, ids)
    }

    /// Remove an overlay by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.layers.iter().position(|l| l.id == id) {
            let removed = self.layers.remove(pos);
            self.registered_grids.remove(&id);
            println!(
                "[overlay] Removed layer '{}' (id={})",
                removed.config.name, id
            );
            self.dirty = true;
            true
        } else {
            false
        }
    }

    pub(crate) fn set_registered_grid(&mut self, id: u32, grid: Option<RasterInfo>) {
        if let Some(grid) = grid {
            self.registered_grids.insert(id, grid);
        }
    }

    pub(crate) fn validate_registered_grids(&self, terrain: &RasterInfo) -> anyhow::Result<()> {
        for grid in self.registered_grids.values() {
            check_terrain_grid(grid, terrain)?;
        }
        Ok(())
    }

    pub(crate) fn geotiff_grid(
        path: &std::path::Path,
        terrain: Option<&RasterInfo>,
        extent: Option<[f32; 4]>,
    ) -> anyhow::Result<Option<RasterInfo>> {
        use anyhow::{ensure, Context};
        if !is_tiff(path) {
            return Ok(None);
        }
        let grid = crate::gis::raster_info::read_raster_info(path)?;
        if !grid.is_georeferenced && grid.registration == PixelRegistration::Area {
            return Ok(None);
        }
        let terrain = terrain.context("GeoTIFF draping requires loaded terrain")?;
        check_terrain_grid(&grid, terrain)?;
        ensure!(
            extent.is_none() || extent == Some([0.0, 0.0, 1.0, 1.0]),
            "GeoTIFF draping uses its grid extent"
        );
        Ok(Some(grid))
    }

    /// Set overlay visibility
    pub fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            if layer.config.visible != visible {
                layer.config.visible = visible;
                self.dirty = true;
            }
        }
    }

    /// Set overlay opacity (0.0 - 1.0)
    pub fn set_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            let clamped = opacity.clamp(0.0, 1.0);
            if (layer.config.opacity - clamped).abs() > 0.001 {
                layer.config.opacity = clamped;
                self.dirty = true;
            }
        }
    }

    /// Get list of all layer IDs in z-order
    pub fn list_ids(&self) -> Vec<u32> {
        let mut sorted: Vec<_> = self.layers.iter().collect();
        sorted.sort_by_key(|l| l.config.z_order);
        sorted.iter().map(|l| l.id).collect()
    }

    /// Check if any visible layers exist
    pub fn has_visible_layers(&self) -> bool {
        self.layers
            .iter()
            .any(|l| l.config.visible && l.config.opacity > 0.001)
    }

    /// Check if composite needs rebuild
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get the composite texture view (for binding to shader)
    /// Returns None if no layers or composite not built yet
    pub fn composite_view(&self) -> Option<&wgpu::TextureView> {
        self.composite_view.as_ref()
    }

    /// Get the overlay sampler
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}

fn is_tiff(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("tif") || extension.eq_ignore_ascii_case("tiff")
        })
}

fn read_tiff_rgba(path: &std::path::Path) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    use anyhow::{bail, ensure};
    let loaded = crate::gis::raster_info::read_raster_data(path)?;
    let RasterData::U8(samples) = loaded.array.data else {
        bail!("GeoTIFF draping requires 8-bit image bands");
    };
    let bands = loaded.array.bands;
    ensure!(
        matches!(bands, 1 | 3 | 4),
        "GeoTIFF draping requires grayscale, RGB or RGBA"
    );
    let pixels = samples.len() / bands;
    let rgba = (0..pixels)
        .flat_map(|i| match bands {
            1 => [samples[i], samples[i], samples[i], 255],
            3 => [
                samples[i],
                samples[pixels + i],
                samples[2 * pixels + i],
                255,
            ],
            _ => [
                samples[i],
                samples[pixels + i],
                samples[2 * pixels + i],
                samples[3 * pixels + i],
            ],
        })
        .collect();
    Ok((rgba, loaded.info.width, loaded.info.height))
}

fn check_terrain_grid(grid: &RasterInfo, terrain: &RasterInfo) -> anyhow::Result<()> {
    use anyhow::ensure;
    ensure!(
        grid.registration == terrain.registration,
        "raster registration mismatch ({} to {}); prepare with registration_conversion='preserve_sample_locations'",
        grid.registration.name(), terrain.registration.name()
    );
    let same_crs = CrsSpec::from_raster_info(grid)
        .zip(CrsSpec::from_raster_info(terrain))
        .is_some_and(|(left, right)| left.equivalent_to(&right));
    ensure!(
        same_crs
            && grid.transform == terrain.transform
            && (grid.width, grid.height) == (terrain.width, terrain.height),
        "GeoTIFF drape must be prepared on the terrain grid"
    );
    Ok(())
}
