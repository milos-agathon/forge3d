//! Labels module for screen-space text labels with MSDF rendering.
//!
//! Provides:
//! - `LabelManager` for managing labels lifecycle
//! - `LabelStyle` for styling configuration  
//! - Grid-based and R-tree collision detection
//! - World-to-screen projection with depth occlusion
//! - Line labels along polylines
//! - Leader lines for offset labels
//! - Scale-dependent visibility (min/max zoom)
//! - Horizon fade for labels near the horizon

mod atlas;
pub mod callout;
mod collision;
pub mod curved;
pub mod declutter;
pub mod layer;
pub mod leader;
pub mod line_label;
pub mod optimal;
mod projection;
#[cfg(feature = "extension-module")]
pub mod py_bindings;
pub mod rtree;
mod types;
pub mod typography;

pub use atlas::{GlyphMetrics, MsdfAtlas};
pub use callout::{Callout, CalloutStyle, PointerDirection};
pub use collision::CollisionGrid;
pub use curved::{CurvedGlyphInstance, CurvedTextLayout, SampledPath};
pub use declutter::{DeclutterAlgorithm, DeclutterConfig, DeclutterResult, PlacementCandidate};
pub use layer::{
    FeatureGeometry, FeatureType, LabelFeature, LabelLayer, LabelLayerConfig, PlacementStrategy,
};
pub use leader::{create_leader_line, generate_leader_vertices};
pub use line_label::{compute_glyph_advances, compute_line_label_placement};
pub use optimal::{
    declutter_optimal, ladder_candidates, OptimalOutcome, RationaleRecord, SolverCandidate,
};
pub use projection::LabelProjector;
pub use rtree::LabelRTree;
pub use types::{
    GlyphPlacement, LabelData, LabelFlags, LabelId, LabelStyle, LeaderLine, LineLabelData,
    LineLabelPlacement,
};
pub use typography::{KerningTable, TextCase, TypographySettings};

use crate::core::text_overlay::{TextInstance, TextOverlayRenderer};
use glam::{DVec3, Mat4, Vec3};
use std::collections::HashMap;
use wgpu::{Device, Queue};

/// Manages screen-space labels with collision detection and depth occlusion.
pub struct LabelManager {
    labels: HashMap<LabelId, LabelData>,
    line_labels: HashMap<LabelId, LineLabelData>,
    next_id: u64,
    atlas: Option<MsdfAtlas>,
    collision_rtree: LabelRTree,
    projector: LabelProjector,
    visible_instances: Vec<TextInstance>,
    leader_lines: Vec<LeaderLine>,
    enabled: bool,
    current_zoom: f32,
    max_visible_labels: usize,
    typography: TypographySettings,
    declutter_algorithm: DeclutterAlgorithm,
    declutter_config: DeclutterConfig,
}

impl LabelManager {
    /// Create a new label manager with default settings.
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            labels: HashMap::new(),
            line_labels: HashMap::new(),
            next_id: 1,
            atlas: None,
            collision_rtree: LabelRTree::new(screen_width, screen_height),
            projector: LabelProjector::new(screen_width, screen_height),
            visible_instances: Vec::new(),
            leader_lines: Vec::new(),
            enabled: true,
            current_zoom: 1.0,
            max_visible_labels: 500,
            typography: TypographySettings::default(),
            declutter_algorithm: DeclutterAlgorithm::default(),
            declutter_config: DeclutterConfig::default(),
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
        let atlas = MsdfAtlas::load(
            device,
            queue,
            atlas_image,
            atlas_width,
            atlas_height,
            metrics_json,
        )?;
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

    fn allocate_id(&mut self, requested: Option<LabelId>) -> LabelId {
        let id = requested.unwrap_or(LabelId(self.next_id));
        self.next_id = self.next_id.max(id.0.saturating_add(1));
        id
    }

    /// Add a label at a world position.
    pub fn add_label(&mut self, text: String, world_pos: Vec3, style: LabelStyle) -> LabelId {
        self.add_label_with_id(None, text, world_pos, style)
    }

    /// Add a label at a world position, preserving an externally allocated ID.
    pub fn add_label_with_id<P: Into<DVec3>>(
        &mut self,
        id: Option<LabelId>,
        text: String,
        world_pos: P,
        style: LabelStyle,
    ) -> LabelId {
        let id = self.allocate_id(id);

        let world_pos = world_pos.into();
        let label = LabelData {
            id,
            text,
            world_pos,
            render_pos: Vec3::ZERO,
            style,
            screen_pos: None,
            visible: true,
            depth: 0.0,
            horizon_angle: 0.0,
            computed_alpha: 1.0,
        };
        self.labels.insert(id, label);
        id
    }

    /// Add a line label along a polyline.
    pub fn add_line_label<P: Into<DVec3>>(
        &mut self,
        text: String,
        polyline: Vec<P>,
        style: LabelStyle,
        placement: LineLabelPlacement,
        repeat_distance: f32,
    ) -> LabelId {
        self.add_line_label_with_id(None, text, polyline, style, placement, repeat_distance)
    }

    /// Add a line label along a polyline, preserving an externally allocated ID.
    pub fn add_line_label_with_id<P: Into<DVec3>>(
        &mut self,
        id: Option<LabelId>,
        text: String,
        polyline: Vec<P>,
        style: LabelStyle,
        placement: LineLabelPlacement,
        repeat_distance: f32,
    ) -> LabelId {
        let id = self.allocate_id(id);

        let polyline: Vec<DVec3> = polyline.into_iter().map(Into::into).collect();
        let render_polyline = vec![Vec3::ZERO; polyline.len()];
        let line_label = LineLabelData {
            id,
            text,
            polyline,
            render_polyline,
            style,
            placement,
            repeat_distance,
            glyph_positions: Vec::new(),
            visible: true,
        };
        self.line_labels.insert(id, line_label);
        id
    }

    /// Remove a label by ID.
    pub fn remove_label(&mut self, id: LabelId) -> bool {
        self.labels.remove(&id).is_some() || self.line_labels.remove(&id).is_some()
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
        self.line_labels.clear();
        self.leader_lines.clear();
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
        self.labels.len() + self.line_labels.len()
    }

    /// Absolute f64 source positions retained for prospective-frame checks.
    pub(crate) fn world_points(&self) -> Vec<DVec3> {
        self.labels
            .values()
            .map(|label| label.world_pos)
            .chain(
                self.line_labels
                    .values()
                    .flat_map(|label| label.polyline.iter().copied()),
            )
            .collect()
    }

    /// Set the current zoom level for scale-dependent visibility.
    pub fn set_zoom(&mut self, zoom: f32) {
        self.current_zoom = zoom;
    }

    /// Get the current zoom level.
    pub fn get_zoom(&self) -> f32 {
        self.current_zoom
    }

    /// Set maximum number of visible labels.
    pub fn set_max_visible(&mut self, max: usize) {
        self.max_visible_labels = max;
    }

    /// Set global typography state for future label layout.
    pub fn set_typography(
        &mut self,
        tracking: Option<f32>,
        kerning: Option<bool>,
        line_height: Option<f32>,
        word_spacing: Option<f32>,
    ) -> TypographySettings {
        if let Some(value) = tracking {
            self.typography.tracking = value;
        }
        if let Some(value) = kerning {
            self.typography.kerning = value;
        }
        if let Some(value) = line_height {
            self.typography.line_height = value;
        }
        if let Some(value) = word_spacing {
            self.typography.word_spacing = value;
        }
        self.typography
    }

    /// Return current typography settings.
    pub fn typography(&self) -> TypographySettings {
        self.typography
    }

    /// Deterministic layout metric used by tests and validation paths.
    pub fn layout_metric_width(text: &str, font_size: f32, settings: &TypographySettings) -> f32 {
        let base_advances: Vec<f32> = text
            .chars()
            .map(|ch| if ch == ' ' { 0.3 } else { 0.5 })
            .collect();
        let mut kerning_table = KerningTable::new();
        kerning_table.load_common_latin_pairs();
        typography::compute_advances_with_typography(
            text,
            &base_advances,
            font_size,
            settings,
            Some(&kerning_table),
        )
        .iter()
        .sum()
    }

    /// Set label declutter policy state.
    pub fn set_declutter_algorithm(
        &mut self,
        algorithm: DeclutterAlgorithm,
        seed: Option<u64>,
        max_iterations: Option<usize>,
    ) -> (DeclutterAlgorithm, DeclutterConfig) {
        self.declutter_algorithm = algorithm;
        if let Some(value) = seed {
            self.declutter_config.seed = value;
        }
        if let Some(value) = max_iterations {
            self.declutter_config.max_iterations = value;
        }
        (self.declutter_algorithm, self.declutter_config.clone())
    }

    /// Resize for new screen dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.collision_rtree.resize(width, height);
        self.projector = LabelProjector::new(width, height);
    }

    /// Get leader lines for rendering.
    pub fn leader_lines(&self) -> &[LeaderLine] {
        &self.leader_lines
    }

    /// Pick a label at the given screen coordinates.
    pub fn pick_at(&self, x: f32, y: f32) -> Option<LabelId> {
        // Query a small box around the cursor (e.g. 4x4 pixels)
        let bounds = [x - 2.0, y - 2.0, x + 2.0, y + 2.0];
        let hits = self.collision_rtree.query_intersecting(bounds);

        // Return the first hit
        // In a real implementation we might want to sort by depth or priority
        hits.first().map(|h| LabelId(h.id))
    }

    /// Update label positions and visibility based on current view.
    /// Returns the number of visible labels.
    pub fn update(&mut self, view_proj: Mat4) -> usize {
        self.update_with_camera(view_proj, None, None)
    }

    /// Update with camera position for horizon fade calculation.
    pub fn update_with_camera(
        &mut self,
        view_proj: Mat4,
        camera_pos: Option<Vec3>,
        selected_ids: Option<&std::collections::HashSet<u64>>,
    ) -> usize {
        self.update_with_camera_anchored(
            view_proj,
            camera_pos,
            selected_ids,
            &crate::camera::Anchor::new(),
        )
    }

    /// Update using the viewer's frozen frame anchor.
    pub fn update_with_camera_anchored(
        &mut self,
        view_proj: Mat4,
        camera_pos: Option<Vec3>,
        selected_ids: Option<&std::collections::HashSet<u64>>,
        anchor: &crate::camera::Anchor,
    ) -> usize {
        if !self.enabled {
            return 0;
        }

        if self.atlas.is_none() {
            self.visible_instances.clear();
            return 0;
        }

        let atlas = self.atlas.as_ref().unwrap();
        self.collision_rtree.clear();
        self.visible_instances.clear();
        self.leader_lines.clear();

        let (screen_w, screen_h) = self.projector.screen_size();
        let mut visible_count = 0;

        // Collect labels and sort by priority (higher priority first)
        let mut sorted_labels: Vec<_> = self.labels.values_mut().collect();
        sorted_labels.sort_by_key(|label| std::cmp::Reverse(label.style.priority));

        for label in sorted_labels {
            label.render_pos = anchor.to_render_vec3(label.world_pos);
            // Skip if we've reached max visible
            if visible_count >= self.max_visible_labels {
                label.visible = false;
                continue;
            }

            // Scale filtering: check zoom range
            if self.current_zoom < label.style.min_zoom || self.current_zoom > label.style.max_zoom
            {
                label.visible = false;
                label.screen_pos = None;
                continue;
            }

            // Project world position to screen
            let projected = self.projector.project(label.render_pos, view_proj);

            if let Some((mut screen_pos, depth)) = projected {
                label.depth = depth;

                // Compute horizon angle for fade
                let horizon_alpha = if let Some(cam_pos) = camera_pos {
                    let to_label = label.render_pos - cam_pos;
                    let horizontal_dist =
                        (to_label.x * to_label.x + to_label.z * to_label.z).sqrt();
                    let angle_deg = (to_label.y / horizontal_dist.max(0.001))
                        .atan()
                        .to_degrees();
                    label.horizon_angle = angle_deg;

                    // Fade based on horizon angle
                    let fade_start = label.style.horizon_fade_angle;
                    if angle_deg.abs() < fade_start {
                        (angle_deg.abs() / fade_start).clamp(0.0, 1.0)
                    } else {
                        1.0
                    }
                } else {
                    label.horizon_angle = 90.0;
                    1.0
                };

                label.computed_alpha = horizon_alpha * label.style.color[3];

                // Apply offset
                let anchor_screen = screen_pos;
                screen_pos[0] += label.style.offset[0];
                screen_pos[1] += label.style.offset[1];
                label.screen_pos = Some(screen_pos);

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

                // Check collision using R-tree
                if self.collision_rtree.try_insert(label.id.0, bounds) {
                    label.visible = true;
                    visible_count += 1;

                    // Generate leader line if offset and flag set
                    if label.style.flags.leader
                        && (label.style.offset[0].abs() > 1.0 || label.style.offset[1].abs() > 1.0)
                    {
                        self.leader_lines.push(create_leader_line(
                            anchor_screen,
                            screen_pos,
                            label.style.halo_color,
                            1.5,
                        ));
                    }

                    // Apply color with computed alpha
                    let mut color = label.style.color;

                    // Highlight if selected
                    if let Some(selected) = selected_ids {
                        if selected.contains(&label.id.0) {
                            // Gold highlight color
                            color = [1.0, 0.8, 0.0, 1.0];
                            // Also ensure fully opaque if highlighted
                            label.computed_alpha = 1.0;
                        }
                    }

                    color[3] = label.computed_alpha;

                    // Generate text instances for this label
                    let instances = atlas.layout_text(
                        &label.text,
                        screen_pos,
                        label.style.size,
                        color,
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

        // Process line labels
        for line_label in self.line_labels.values_mut() {
            if line_label.render_polyline.len() != line_label.polyline.len() {
                line_label
                    .render_polyline
                    .resize(line_label.polyline.len(), Vec3::ZERO);
            }
            for (render, world) in line_label
                .render_polyline
                .iter_mut()
                .zip(line_label.polyline.iter().copied())
            {
                *render = anchor.to_render_vec3(world);
            }
            if visible_count >= self.max_visible_labels {
                line_label.visible = false;
                continue;
            }

            // Scale filtering
            if self.current_zoom < line_label.style.min_zoom
                || self.current_zoom > line_label.style.max_zoom
            {
                line_label.visible = false;
                continue;
            }

            // Compute glyph advances
            let advances = compute_glyph_advances(&line_label.text, line_label.style.size);

            // Compute placements
            let placements = compute_line_label_placement(
                &line_label.render_polyline,
                &line_label.text,
                &advances,
                view_proj,
                screen_w,
                screen_h,
                line_label.placement,
                line_label.style.size,
            );

            if placements.is_empty() {
                line_label.visible = false;
                continue;
            }

            line_label.glyph_positions = placements;
            line_label.visible = true;
            visible_count += 1;

            // Emit one rotated atlas glyph quad per placed line-label glyph.
            let mut color = line_label.style.color;
            color[3] = color[3].clamp(0.0, 1.0);
            for (ch, placement) in line_label
                .text
                .chars()
                .zip(line_label.glyph_positions.iter())
            {
                if ch == ' ' {
                    continue;
                }
                let mut instances = atlas.layout_text(
                    &ch.to_string(),
                    placement.screen_pos,
                    line_label.style.size,
                    color,
                    line_label.style.halo_color,
                    line_label.style.halo_width,
                );
                for instance in &mut instances {
                    instance.rotation = placement.rotation;
                }
                self.visible_instances.extend(instances);
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
            // Recreate bind group with the atlas view. Best-effort HUD label path:
            // the internal allocations were infallible before the tracked-wrapper
            // migration, so a rare allocation failure is discarded rather than
            // aborting the frame (this fn returns a count, not a Result).
            let _ = renderer.recreate_bind_group(device, Some(&atlas.view));
        }

        // Use SDF mode (1 channel) for bitmap fonts, MSDF (3 channels) for proper MSDF atlases.
        // Default to SDF because current atlases are single-channel bitmap fonts.
        renderer.set_channels(1);
        renderer.set_smoothing(2.0);

        let _ = renderer.upload_instances(device, queue, &self.visible_instances);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_manager_typography_and_declutter_state_mutate() {
        let mut manager = LabelManager::new(800, 600);

        let typography = manager.set_typography(Some(0.25), Some(true), Some(1.3), Some(2.0));
        assert_eq!(typography.tracking, 0.25);
        assert!(typography.kerning);
        assert_eq!(typography.line_height, 1.3);
        assert_eq!(typography.word_spacing, 2.0);

        let default_width =
            LabelManager::layout_metric_width("AV label", 16.0, &TypographySettings::default());
        let typography_width = LabelManager::layout_metric_width("AV label", 16.0, &typography);
        assert!(typography_width > default_width);

        let (algorithm, config) =
            manager.set_declutter_algorithm(DeclutterAlgorithm::Annealing, Some(123), Some(50));
        assert_eq!(algorithm, DeclutterAlgorithm::Annealing);
        assert_eq!(config.seed, 123);
        assert_eq!(config.max_iterations, 50);
    }
}
