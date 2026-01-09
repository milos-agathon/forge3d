// src/picking/mod.rs
// Picking system for interactive feature selection in forge3d viewer
// Implements ID buffer rendering, ray picking, hover support, multi-select, BVH, and lasso
// Plan 1: MVP ID buffer + Plan 2: Standard ray picking + Plan 3: Premium BVH + Lasso

mod id_buffer;
mod ray;
mod bounds;
mod tile_id;
mod terrain_query;
mod selection;
mod unified;
mod heightfield_ray;
mod lasso;
mod highlight;

pub use id_buffer::{IdBufferPass, IdVertex};
pub use ray::{Ray, unproject_cursor, invert_matrix};
pub use bounds::{AABB, LayerBounds, BoundsManager};
pub use tile_id::{TileIdPass, TileIdVertex, TileIdUniforms, DEFAULT_TILE_SIZE};
pub use terrain_query::{TerrainQueryEngine, TerrainQueryResult, TerrainQueryConfig};
pub use selection::{SelectionManager, SelectionSet, SelectionStyle};
pub use unified::{
    UnifiedPickingSystem, UnifiedPickingConfig, RichPickResult, 
    TerrainHitInfo, PickEvent, PickEventType, LayerBvhData
};
pub use heightfield_ray::{HeightfieldRayEngine, HeightfieldConfig, HeightfieldHit};
pub use lasso::{LassoSelection, LassoConfig, LassoState, BoxSelection, Point2D, BoundingBox2D};
pub use highlight::{
    HighlightManager, HighlightStyle, HighlightEffect, HighlightUniforms,
    HIGHLIGHT_SHADER_FUNCTIONS
};

use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device, Queue};

/// Result of a pick operation
#[derive(Debug, Clone)]
pub struct PickResult {
    /// Feature ID (0 = no feature/background)
    pub feature_id: u32,
    /// Screen position where the pick occurred
    pub screen_pos: (u32, u32),
    /// World position if depth was available
    pub world_pos: Option<[f32; 3]>,
    /// Layer name if available
    pub layer_name: Option<String>,
}

/// Configuration for the picking system
#[derive(Debug, Clone)]
pub struct PickingConfig {
    /// Whether picking is enabled (default: false to avoid overhead)
    pub enabled: bool,
    /// Highlight color for selected features (RGBA)
    pub highlight_color: [f32; 4],
    /// Whether hover highlighting is enabled
    pub hover_enabled: bool,
    /// Hover delay in milliseconds before triggering hover callback
    pub hover_delay_ms: u32,
    /// Tile size for hover picking (smaller = faster but less accurate)
    pub tile_size: u32,
    /// Whether multi-select mode is enabled
    pub multi_select: bool,
}

impl Default for PickingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            highlight_color: [1.0, 0.8, 0.0, 0.5], // Yellow semi-transparent
            hover_enabled: false,
            hover_delay_ms: 100,
            tile_size: DEFAULT_TILE_SIZE,
            multi_select: false,
        }
    }
}

/// Manages picking operations for the viewer
pub struct PickingManager {
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    config: PickingConfig,
    id_buffer_pass: Option<IdBufferPass>,
    tile_id_pass: Option<TileIdPass>,
    readback_buffer: Option<Buffer>,
    selection_manager: SelectionManager,
    bounds_manager: BoundsManager,
    terrain_query: TerrainQueryEngine,
    pending_pick: Option<(u32, u32)>,
    pending_hover: Option<(u32, u32, std::time::Instant)>,
    last_hover_pos: Option<(u32, u32)>,
    width: u32,
    height: u32,
}

impl PickingManager {
    /// Create a new picking manager
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            config: PickingConfig::default(),
            id_buffer_pass: None,
            tile_id_pass: None,
            readback_buffer: None,
            selection_manager: SelectionManager::new(),
            bounds_manager: BoundsManager::new(),
            terrain_query: TerrainQueryEngine::new(TerrainQueryConfig::default()),
            pending_pick: None,
            pending_hover: None,
            last_hover_pos: None,
            width: 0,
            height: 0,
        }
    }

    /// Initialize or resize the ID buffer
    pub fn init(&mut self, width: u32, height: u32) {
        if !self.config.enabled {
            return;
        }

        if self.width == width && self.height == height && self.id_buffer_pass.is_some() {
            return;
        }

        self.width = width;
        self.height = height;

        self.id_buffer_pass = Some(IdBufferPass::new(
            &self.device,
            width,
            height,
        ));

        // Initialize tile ID pass for hover picking
        if self.config.hover_enabled {
            self.tile_id_pass = Some(TileIdPass::new(&self.device, self.config.tile_size));
        }

        self.readback_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("picking_readback_buffer"),
            size: 4, // Single u32 pixel
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }

    /// Enable or disable picking
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
        if !enabled {
            self.id_buffer_pass = None;
            self.readback_buffer = None;
        }
    }

    /// Check if picking is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Set highlight color for selected features
    pub fn set_highlight_color(&mut self, color: [f32; 4]) {
        self.config.highlight_color = color;
    }

    /// Get highlight color
    pub fn highlight_color(&self) -> [f32; 4] {
        self.config.highlight_color
    }

    /// Request a pick at the given screen coordinates
    pub fn request_pick(&mut self, x: u32, y: u32) {
        if self.config.enabled {
            self.pending_pick = Some((x, y));
        }
    }

    /// Check if there's a pending pick request
    pub fn has_pending_pick(&self) -> bool {
        self.pending_pick.is_some()
    }

    /// Get the pending pick coordinates
    pub fn pending_pick_coords(&self) -> Option<(u32, u32)> {
        self.pending_pick
    }

    /// Clear pending pick request
    pub fn clear_pending_pick(&mut self) {
        self.pending_pick = None;
    }

    /// Set the selected feature ID (adds to primary selection)
    pub fn set_selected(&mut self, feature_id: Option<u32>) {
        if let Some(id) = feature_id {
            self.selection_manager.handle_pick(id, false);
        } else {
            self.selection_manager.clear_set("primary");
        }
    }

    /// Get the currently selected feature ID (first from primary selection)
    pub fn selected_feature(&self) -> Option<u32> {
        self.selection_manager.get_selection().first().copied()
    }
    
    /// Get all selected feature IDs
    pub fn get_selection(&self) -> Vec<u32> {
        self.selection_manager.get_selection()
    }
    
    /// Enable or disable multi-select mode
    pub fn set_multi_select(&mut self, enabled: bool) {
        self.config.multi_select = enabled;
        self.selection_manager.set_multi_select(enabled);
    }
    
    /// Check if multi-select is enabled
    pub fn is_multi_select(&self) -> bool {
        self.config.multi_select
    }
    
    /// Get the selection manager
    pub fn selection_manager(&self) -> &SelectionManager {
        &self.selection_manager
    }
    
    /// Get mutable selection manager
    pub fn selection_manager_mut(&mut self) -> &mut SelectionManager {
        &mut self.selection_manager
    }
    
    /// Get the bounds manager
    pub fn bounds_manager(&self) -> &BoundsManager {
        &self.bounds_manager
    }
    
    /// Get mutable bounds manager
    pub fn bounds_manager_mut(&mut self) -> &mut BoundsManager {
        &mut self.bounds_manager
    }
    
    /// Get the terrain query engine
    pub fn terrain_query(&self) -> &TerrainQueryEngine {
        &self.terrain_query
    }
    
    /// Get mutable terrain query engine
    pub fn terrain_query_mut(&mut self) -> &mut TerrainQueryEngine {
        &mut self.terrain_query
    }

    /// Get the ID buffer pass for rendering
    pub fn id_buffer_pass(&self) -> Option<&IdBufferPass> {
        self.id_buffer_pass.as_ref()
    }

    /// Get mutable reference to ID buffer pass
    pub fn id_buffer_pass_mut(&mut self) -> Option<&mut IdBufferPass> {
        self.id_buffer_pass.as_mut()
    }

    /// Get the readback buffer
    pub fn readback_buffer(&self) -> Option<&Buffer> {
        self.readback_buffer.as_ref()
    }

    /// Copy a single pixel from the ID buffer to the readback buffer
    pub fn copy_pixel_to_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: u32,
        y: u32,
    ) {
        let id_pass = match &self.id_buffer_pass {
            Some(pass) => pass,
            None => return,
        };

        let readback = match &self.readback_buffer {
            Some(buf) => buf,
            None => return,
        };

        // Clamp coordinates
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: id_pass.id_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Read back the feature ID from the readback buffer (blocking)
    /// Call this after the GPU commands have been submitted
    pub fn read_feature_id(&self) -> Option<u32> {
        let readback = self.readback_buffer.as_ref()?;

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        if rx.recv().ok()?.is_ok() {
            let data = slice.get_mapped_range();
            let id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            readback.unmap();
            Some(id)
        } else {
            None
        }
    }

    /// Process a completed pick and return the result
    pub fn complete_pick(&mut self, x: u32, y: u32, shift_held: bool) -> Option<PickResult> {
        let feature_id = self.read_feature_id()?;
        
        // ID 0 means no feature (background)
        if feature_id == 0 {
            if !shift_held {
                self.selection_manager.clear_set("primary");
            }
            return None;
        }

        self.selection_manager.handle_pick(feature_id, shift_held);

        Some(PickResult {
            feature_id,
            screen_pos: (x, y),
            world_pos: None,
            layer_name: None,
        })
    }
    
    /// Enable or disable hover highlighting
    pub fn set_hover_enabled(&mut self, enabled: bool) {
        self.config.hover_enabled = enabled;
        if enabled && self.config.enabled && self.tile_id_pass.is_none() {
            self.tile_id_pass = Some(TileIdPass::new(&self.device, self.config.tile_size));
        } else if !enabled {
            self.tile_id_pass = None;
        }
    }
    
    /// Check if hover is enabled
    pub fn is_hover_enabled(&self) -> bool {
        self.config.hover_enabled
    }
    
    /// Set hover delay in milliseconds
    pub fn set_hover_delay(&mut self, delay_ms: u32) {
        self.config.hover_delay_ms = delay_ms;
    }
    
    /// Request hover check at position (throttled)
    pub fn request_hover(&mut self, x: u32, y: u32) {
        if !self.config.hover_enabled {
            return;
        }
        
        // Check if position changed significantly
        if let Some((last_x, last_y)) = self.last_hover_pos {
            if (x as i32 - last_x as i32).abs() < 2 && (y as i32 - last_y as i32).abs() < 2 {
                return;
            }
        }
        
        self.last_hover_pos = Some((x, y));
        self.pending_hover = Some((x, y, std::time::Instant::now()));
    }
    
    /// Check if hover delay has elapsed and return pending hover coordinates
    pub fn check_hover_ready(&self) -> Option<(u32, u32)> {
        if let Some((x, y, time)) = self.pending_hover {
            if time.elapsed().as_millis() >= self.config.hover_delay_ms as u128 {
                return Some((x, y));
            }
        }
        None
    }
    
    /// Clear pending hover
    pub fn clear_pending_hover(&mut self) {
        self.pending_hover = None;
    }
    
    /// Set hover feature (for highlighting)
    pub fn set_hover_feature(&mut self, feature_id: Option<u32>) {
        self.selection_manager.set_hover(feature_id);
    }
    
    /// Get hover feature
    pub fn hover_feature(&self) -> Option<u32> {
        self.selection_manager.hover_feature()
    }
    
    /// Get the tile ID pass for hover rendering
    pub fn tile_id_pass(&self) -> Option<&TileIdPass> {
        self.tile_id_pass.as_ref()
    }
    
    /// Get picking config
    pub fn config(&self) -> &PickingConfig {
        &self.config
    }

    /// Get current buffer dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
