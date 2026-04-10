use super::helpers::{build_grid_xyuv, build_view_matrices};
use super::*;
// ---------- Render spike object used by tests ----------

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[pyclass(module = "forge3d._forge3d", name = "TerrainSpike")]
pub struct TerrainSpike {
    width: u32,
    height: u32,
    _grid: u32,

    device: wgpu::Device,
    queue: wgpu::Queue,

    // T33-BEGIN:tp-and-bgs
    tp: crate::terrain::pipeline::TerrainPipeline,
    bg0_globals: wgpu::BindGroup,
    bg1_height: wgpu::BindGroup,
    bg2_lut: wgpu::BindGroup,
    // T33-END:tp-and-bgs
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,

    ubo: wgpu::Buffer,
    tile_ubo: wgpu::Buffer,          // E2: per-tile uniforms buffer
    tile_slot_ubo: wgpu::Buffer,     // E1b: per-draw tile slot buffer
    mosaic_params_ubo: wgpu::Buffer, // E1b: mosaic params buffer
    colormap_lut: ColormapLUT,
    lut_format: &'static str,

    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    _normal: wgpu::Texture,
    normal_view: wgpu::TextureView,

    globals: Globals,
    last_uniforms: TerrainUniforms,

    // T33: optional height texture state
    height_view: Option<wgpu::TextureView>,
    height_sampler: Option<wgpu::Sampler>,
    // E6: whether the current pipeline expects a filterable height sampler
    height_filterable: bool,

    // B11: Tiling system for large DEMs
    tiling_system: Option<TilingSystem>,

    // E1: GPU height mosaic (R32Float) for streamed tiles
    height_mosaic: Option<crate::terrain::stream::HeightMosaic>,
    // E3: Optional overlay mosaic (RGBA8) for basemap streaming
    overlay_mosaic: Option<crate::terrain::stream::ColorMosaic>,
    // E3: Overlay compositor
    overlay_renderer: Option<crate::core::overlays::OverlayRenderer>,
    // E2: tile uniforms bind group
    bg5_tile: wgpu::BindGroup,
    // E1: Optional page table buffer
    page_table: Option<PageTable>,
    // E1c: Optional background async tile loader
    async_loader: Option<crate::terrain::page_table::AsyncTileLoader>,
    // E1c/E1e: Optional background async overlay loader
    async_overlay_loader: Option<crate::terrain::page_table::AsyncOverlayLoader>,
    // E1e: track previous visible tiles for cancellation (height/overlay)
    prev_visible_height: HashSet<TileId>,
    prev_visible_overlay: HashSet<TileId>,
}

mod analysis;
mod async_loader;
mod constructor;
mod height_mosaic;
mod overlay_stream;
mod render;
mod terrain_ops;
mod tiling;
