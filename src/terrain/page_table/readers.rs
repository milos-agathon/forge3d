use glam::Vec2;

use crate::terrain::tiling::{QuadTreeNode, TileBounds, TileId};

/// Simple file-backed overlay reader that expands a template like
/// "/data/tiles/{lod}/{x}/{y}.png" and returns RGBA8 bytes.
pub struct FileOverlayReader {
    template: String,
}

impl FileOverlayReader {
    pub fn new(template: String) -> Self {
        Self { template }
    }

    fn expand(&self, id: TileId) -> String {
        self.template
            .replace("{lod}", &id.lod.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string())
    }
}

impl OverlayReader for FileOverlayReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let path = self.expand(tile_id);
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                if rgba.width() != width || rgba.height() != height {
                    let resized = image::imageops::resize(
                        &rgba,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    resized.into_raw()
                } else {
                    rgba.into_raw()
                }
            }
            Err(_) => vec![0u8; (width * height * 4) as usize],
        }
    }
}

/// Simple file-backed height reader using PNG grayscale (8/16-bit) to f32 with scale/offset
pub struct FileHeightReader {
    template: String,
    scale: f32,
    offset: f32,
}

impl FileHeightReader {
    pub fn new(template: String, scale: f32, offset: f32) -> Self {
        Self {
            template,
            scale,
            offset,
        }
    }

    fn expand(&self, id: TileId) -> String {
        self.template
            .replace("{lod}", &id.lod.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string())
    }
}

impl HeightReader for FileHeightReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let path = self.expand(tile_id);
        let expected = (width * height) as usize;
        match image::open(&path) {
            Ok(img) => {
                let gray = img.to_luma16();
                let (w, h) = gray.dimensions();
                if w != width || h != height {
                    let resized = image::imageops::resize(
                        &gray,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    let mut out = Vec::with_capacity(expected);
                    for &v16 in resized.as_raw() {
                        let v = (v16 as f32) / 65535.0;
                        out.push(v * self.scale + self.offset);
                    }
                    out
                } else {
                    let mut out = Vec::with_capacity(expected);
                    for &v16 in gray.as_raw() {
                        let v = (v16 as f32) / 65535.0;
                        out.push(v * self.scale + self.offset);
                    }
                    out
                }
            }
            Err(_) => vec![0.0f32; expected],
        }
    }
}

pub trait HeightReader: Send + Sync + 'static {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32>;
}

pub struct SyntheticHeightReader;

impl HeightReader for SyntheticHeightReader {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let mut heights = Vec::with_capacity((width * height) as usize);
        let bounds = QuadTreeNode::calculate_bounds(root_bounds, tile_id, tile_size);
        for y in 0..height {
            for x in 0..width {
                let u = x as f32 / (width - 1) as f32;
                let v = y as f32 / (height - 1) as f32;
                let world_x = bounds.min.x + u * bounds.size().x;
                let world_y = bounds.min.y + v * bounds.size().y;
                let h = (world_x * 0.1).sin() * 10.0 + (world_y * 0.1).cos() * 10.0;
                heights.push(h);
            }
        }
        heights
    }
}

pub trait OverlayReader: Send + Sync + 'static {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8>;
}

pub struct SyntheticOverlayReader;

impl OverlayReader for SyntheticOverlayReader {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut px = Vec::with_capacity((width * height * 4) as usize);
        let bounds = QuadTreeNode::calculate_bounds(root_bounds, tile_id, tile_size);
        for y in 0..height {
            for x in 0..width {
                let u = x as f32 / (width - 1) as f32;
                let v = y as f32 / (height - 1) as f32;
                let wx = bounds.min.x + u * bounds.size().x;
                let wy = bounds.min.y + v * bounds.size().y;
                let r = (((wx * 0.01).sin() * 0.5 + 0.5) * 255.0) as u8;
                let g = (((wy * 0.01).cos() * 0.5 + 0.5) * 255.0) as u8;
                let b = (((wx * 0.02 + wy * 0.02).sin() * 0.5 + 0.5) * 255.0) as u8;
                px.extend_from_slice(&[r, g, b, 255]);
            }
        }
        px
    }
}
