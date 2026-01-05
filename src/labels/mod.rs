//! Labels module for screen-space text labels with MSDF rendering.
//!
//! Provides:
//! - `LabelManager` for managing labels lifecycle
//! - `LabelStyle` for styling configuration
//! - Grid-based collision detection
//! - World-to-screen projection with depth occlusion

mod atlas;
mod collision;
mod projection;
mod types;

pub use atlas::{GlyphMetrics, MsdfAtlas};
pub use collision::CollisionGrid;
pub use projection::LabelProjector;
pub use types::{LabelData, LabelId, LabelStyle};

use crate::core::text_overlay::{TextInstance, TextOverlayRenderer};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use wgpu::{Device, Queue};

/// Manages screen-space labels with collision detection and depth occlusion.
pub struct LabelManager {
    labels: HashMap<LabelId, LabelData>,
    next_id: u64,
    atlas: Option<MsdfAtlas>,
    collision_grid: CollisionGrid,
    projector: LabelProjector,
    visible_instances: Vec<TextInstance>,
    enabled: bool,
}

impl LabelManager {
    /// Create a new label manager with default settings.
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            labels: HashMap::new(),
            next_id: 1,
            atlas: None,
            collision_grid: CollisionGrid::new(screen_width, screen_height, 10),
            projector: LabelProjector::new(screen_width, screen_height),
            visible_instances: Vec::new(),
            enabled: true,
        }
    }

    /// Load an MSDF atlas from font data.
    pub fn load_atlas(
        &mut self,
        device: &Device,
        queue: &Queue,
        atlas_image: &[u8],
        atlas_width: u32,
        atlas_height: u32,
        metrics_json: &str,
    ) -> Result<(), String> {
        let atlas = MsdfAtlas::load(device, queue, atlas_image, atlas_width, atlas_height, metrics_json)?;
        self.atlas = Some(atlas);
        Ok(())
    }

    /// Load atlas from PNG file and JSON metrics file.
    pub fn load_atlas_from_files(
        &mut self,
        device: &Device,
        queue: &Queue,
        atlas_png_path: &str,
        metrics_json_path: &str,
    ) -> Result<(), String> {
        let atlas = MsdfAtlas::load_from_files(device, queue, atlas_png_path, metrics_json_path)?;
        self.atlas = Some(atlas);
        Ok(())
    }

    /// Add a label at a world position.
    pub fn add_label(&mut self, text: String, world_pos: Vec3, style: LabelStyle) -> LabelId {
        let id = LabelId(self.next_id);
        self.next_id += 1;

        let label = LabelData {
            id,
            text,
            world_pos,
            style,
            screen_pos: None,
            visible: true,
            depth: 0.0,
        };
        self.labels.insert(id, label);
        id
    }

    /// Remove a label by ID.
    pub fn remove_label(&mut self, id: LabelId) -> bool {
        self.labels.remove(&id).is_some()
    }

    /// Update label style.
    pub fn set_label_style(&mut self, id: LabelId, style: LabelStyle) -> bool {
        if let Some(label) = self.labels.get_mut(&id) {
            label.style = style;
            true
        } else {
            false
        }
    }

    /// Get a label by ID.
    pub fn get_label(&self, id: LabelId) -> Option<&LabelData> {
        self.labels.get(&id)
    }

    /// Get mutable label by ID.
    pub fn get_label_mut(&mut self, id: LabelId) -> Option<&mut LabelData> {
        self.labels.get_mut(&id)
    }

    /// Clear all labels.
    pub fn clear(&mut self) {
        self.labels.clear();
    }

    /// Set enabled state.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if labels are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get number of labels.
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    /// Resize for new screen dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.collision_grid = CollisionGrid::new(width, height, 10);
        self.projector = LabelProjector::new(width, height);
    }

    /// Update label positions and visibility based on current view.
    /// Returns the number of visible labels.
    pub fn update(&mut self, view_proj: Mat4) -> usize {
        if !self.enabled {
            return 0;
        }

        if self.atlas.is_none() {
            self.visible_instances.clear();
            return 0;
        }

        let atlas = self.atlas.as_ref().unwrap();
        self.collision_grid.clear();
        self.visible_instances.clear();

        // Collect labels and sort by priority (higher priority first)
        let mut sorted_labels: Vec<_> = self.labels.values_mut().collect();
        sorted_labels.sort_by(|a, b| b.style.priority.cmp(&a.style.priority));

        for label in sorted_labels {
            // Project world position to screen
            let projected = self.projector.project(label.world_pos, view_proj);

            if let Some((screen_pos, depth)) = projected {
                label.screen_pos = Some(screen_pos);
                label.depth = depth;

                // Calculate label bounds
                let (width, height) = atlas.measure_text(&label.text, label.style.size);
                let half_w = width * 0.5;
                let half_h = height * 0.5;

                let bounds = [
                    screen_pos[0] - half_w,
                    screen_pos[1] - half_h,
                    screen_pos[0] + half_w,
                    screen_pos[1] + half_h,
                ];

                // Check collision
                if self.collision_grid.try_insert(bounds) {
                    label.visible = true;

                    // Generate text instances for this label
                    let instances = atlas.layout_text(
                        &label.text,
                        screen_pos,
                        label.style.size,
                        label.style.color,
                        label.style.halo_color,
                        label.style.halo_width,
                    );
                    self.visible_instances.extend(instances);
                } else {
                    label.visible = false;
                }
            } else {
                label.screen_pos = None;
                label.visible = false;
            }
        }

        self.visible_instances.len()
    }

    /// Upload instances to the text overlay renderer.
    pub fn upload_to_renderer(
        &self,
        device: &Device,
        queue: &Queue,
        renderer: &mut TextOverlayRenderer,
    ) {
        if let Some(atlas) = &self.atlas {
            // Recreate bind group with the atlas view
            renderer.recreate_bind_group(device, Some(&atlas.view));
        }

        // Use SDF mode (1 channel) for bitmap fonts, MSDF (3 channels) for proper MSDF atlases
        // For now, default to SDF mode which works with simple bitmap fonts
        renderer.set_channels(1);
        renderer.set_smoothing(2.0);

        renderer.upload_instances(device, queue, &self.visible_instances);
    }

    /// Get reference to the atlas view if loaded.
    pub fn atlas_view(&self) -> Option<&wgpu::TextureView> {
        self.atlas.as_ref().map(|a| a.view.as_ref())
    }

    /// Get visible instance count.
    pub fn visible_count(&self) -> usize {
        self.visible_instances.len()
    }
}
