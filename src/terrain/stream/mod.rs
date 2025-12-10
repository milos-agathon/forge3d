// E1/E3/E4: Terrain height tile streaming into a GPU R32Float mosaic texture
// - LRU-managed atlas of height tiles (tile_size_px × tile_size_px), arranged in a fixed grid (tiles_x × tiles_y)
// - Per-frame upload budget to avoid long stalls
// - Integrates with TerrainSpike by rebinding group(1) height texture/sampler to mosaic

mod color;
mod config;
mod height;
mod util;

pub use color::ColorMosaic;
pub use config::MosaicConfig;
pub use height::HeightMosaic;
