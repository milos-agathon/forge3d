use glam::Vec2;
use std::sync::Arc;

use super::common::CoalescePolicy;
use super::readers::OverlayReader;
use super::scheduler::{SchedulingPolicy, TileRequestScheduler};
use crate::terrain::tiling::{TileBounds, TileId};

#[derive(Debug)]
pub struct OverlayTileData {
    pub tile_id: TileId,
    pub rgba_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub host_memory_size: u64,
}

impl OverlayTileData {
    pub fn new(tile_id: TileId, rgba_data: Vec<u8>, width: u32, height: u32) -> Self {
        let host_memory_size = rgba_data.len() as u64;
        Self {
            tile_id,
            rgba_data,
            width,
            height,
            host_memory_size,
        }
    }
}

pub struct AsyncOverlayLoader {
    scheduler: TileRequestScheduler<OverlayTileData>,
}

impl AsyncOverlayLoader {
    pub fn new_with_reader(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn OverlayReader>,
        policy: CoalescePolicy,
    ) -> Self {
        Self {
            scheduler: TileRequestScheduler::new(
                max_in_flight,
                pool_size,
                policy,
                SchedulingPolicy::Coalescing,
                move |id| {
                    let rgba = reader.read(
                        &root_bounds,
                        tile_size,
                        id,
                        tile_resolution,
                        tile_resolution,
                    );
                    Ok(OverlayTileData::new(
                        id,
                        rgba,
                        tile_resolution,
                        tile_resolution,
                    ))
                },
            ),
        }
    }

    pub fn request(&self, id: TileId) -> bool {
        self.scheduler.request(id)
    }
    pub fn cancel(&self, ids: &[TileId]) -> usize {
        self.scheduler.cancel(ids)
    }
    pub fn drain_completed(&self, limit: usize) -> Vec<OverlayTileData> {
        self.scheduler.drain_completed(limit)
    }
    pub fn stats(&self) -> (usize, usize, usize) {
        self.scheduler.stats()
    }
    pub fn counters(&self) -> (usize, usize, usize, usize, usize, usize) {
        self.scheduler.counters()
    }
}
