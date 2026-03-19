//! LRU tile cache for virtual texture streaming.

mod allocator;
mod cache;
#[cfg(test)]
mod tests;
mod types;

pub use cache::{CacheStats, TileCache};
pub use types::{AtlasSlot, TileData, TileId};
