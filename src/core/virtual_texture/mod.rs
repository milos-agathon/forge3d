//! Virtual texture streaming system.

use crate::core::feedback_buffer::FeedbackBuffer;
use crate::core::memory_tracker::global_tracker;
use crate::core::tile_cache::{TileCache, TileData, TileId};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
#[cfg(feature = "enable-staging-rings")]
use std::sync::{Arc, Mutex};
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, Device, Queue, Sampler, Texture,
};

#[cfg(feature = "enable-staging-rings")]
use crate::core::staging_rings::StagingRing;

mod constructor;
#[cfg(test)]
mod tests;
mod types;
mod update;
mod upload;

pub use types::{CameraInfo, PageTableEntry, VirtualTextureConfig, VirtualTextureStats};

/// Virtual texture streaming system
pub struct VirtualTexture {
    pub(super) config: VirtualTextureConfig,
    pub(super) atlas_texture: Texture,
    pub(super) page_table: Texture,
    pub(super) page_table_data: Vec<PageTableEntry>,
    pub(super) feedback_buffer: Option<FeedbackBuffer>,
    pub(super) tile_cache: TileCache,
    pub(super) requested_tiles: HashSet<TileId>,
    pub(super) stats: VirtualTextureStats,
    #[cfg(feature = "enable-staging-rings")]
    pub(super) staging_ring: Option<Arc<Mutex<StagingRing>>>,
}
