//! Sprite atlas loading for Mapbox Style Spec icons.
//!
//! Mapbox styles reference sprites via a base URL that resolves to:
//! - `{sprite}.json` - JSON metadata with icon positions
//! - `{sprite}.png` - Sprite atlas image
//! - `{sprite}@2x.json` / `{sprite}@2x.png` - High-DPI variants

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Error type for sprite operations.
#[derive(Debug, thiserror::Error)]
pub enum SpriteError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Image error: {0}")]
    Image(String),
    #[error("Sprite not found: {0}")]
    NotFound(String),
}

/// A single sprite entry in the atlas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpriteEntry {
    /// X position in atlas (pixels).
    pub x: u32,
    /// Y position in atlas (pixels).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel ratio (1 for standard, 2 for @2x).
    #[serde(default = "default_pixel_ratio")]
    pub pixel_ratio: f32,
    /// Whether the sprite is an SDF (signed distance field).
    #[serde(default)]
    pub sdf: bool,
    /// Optional content area for 9-slice scaling.
    #[serde(default)]
    pub content: Option<[u32; 4]>,
    /// Optional stretch areas for 9-slice scaling.
    #[serde(rename = "stretchX")]
    #[serde(default)]
    pub stretch_x: Option<Vec<[u32; 2]>>,
    #[serde(rename = "stretchY")]
    #[serde(default)]
    pub stretch_y: Option<Vec<[u32; 2]>>,
}

fn default_pixel_ratio() -> f32 {
    1.0
}

/// Sprite atlas containing metadata and optional image data.
#[derive(Debug, Clone)]
pub struct SpriteAtlas {
    /// Sprite entries by name.
    pub entries: HashMap<String, SpriteEntry>,
    /// Atlas image data (RGBA).
    pub image_data: Option<Vec<u8>>,
    /// Atlas image width.
    pub image_width: u32,
    /// Atlas image height.
    pub image_height: u32,
    /// Pixel ratio (1 or 2).
    pub pixel_ratio: f32,
}

impl SpriteAtlas {
    /// Create empty atlas.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            image_data: None,
            image_width: 0,
            image_height: 0,
            pixel_ratio: 1.0,
        }
    }

    /// Get a sprite entry by name.
    pub fn get(&self, name: &str) -> Option<&SpriteEntry> {
        self.entries.get(name)
    }

    /// Check if sprite exists.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// List all sprite names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    /// Get sprite UV coordinates (normalized 0-1).
    pub fn get_uvs(&self, name: &str) -> Option<[f32; 4]> {
        let entry = self.entries.get(name)?;
        if self.image_width == 0 || self.image_height == 0 {
            return None;
        }
        let u0 = entry.x as f32 / self.image_width as f32;
        let v0 = entry.y as f32 / self.image_height as f32;
        let u1 = (entry.x + entry.width) as f32 / self.image_width as f32;
        let v1 = (entry.y + entry.height) as f32 / self.image_height as f32;
        Some([u0, v0, u1, v1])
    }

    /// Extract sprite pixels from atlas image.
    pub fn extract_sprite(&self, name: &str) -> Option<(Vec<u8>, u32, u32)> {
        let entry = self.entries.get(name)?;
        let image_data = self.image_data.as_ref()?;
        
        let mut pixels = Vec::with_capacity((entry.width * entry.height * 4) as usize);
        
        for y in 0..entry.height {
            let src_y = entry.y + y;
            let src_offset = ((src_y * self.image_width + entry.x) * 4) as usize;
            let row_bytes = (entry.width * 4) as usize;
            
            if src_offset + row_bytes <= image_data.len() {
                pixels.extend_from_slice(&image_data[src_offset..src_offset + row_bytes]);
            }
        }
        
        Some((pixels, entry.width, entry.height))
    }
}

impl Default for SpriteAtlas {
    fn default() -> Self {
        Self::new()
    }
}

/// Load sprite atlas from local files.
///
/// Expects `{base_path}.json` and optionally `{base_path}.png`.
pub fn load_sprite_atlas(base_path: &Path) -> Result<SpriteAtlas, SpriteError> {
    // Try @2x first, then standard
    let (json_path, png_path, pixel_ratio) = if base_path.to_string_lossy().ends_with("@2x") {
        (
            base_path.with_extension("json"),
            base_path.with_extension("png"),
            2.0,
        )
    } else {
        let path_2x_str = format!("{}@2x", base_path.display());
        let path_2x = std::path::PathBuf::from(&path_2x_str);
        let json_2x = path_2x.with_extension("json");
        
        if json_2x.exists() {
            (json_2x, path_2x.with_extension("png"), 2.0)
        } else {
            (
                base_path.with_extension("json"),
                base_path.with_extension("png"),
                1.0,
            )
        }
    };

    // Load JSON metadata
    let json_content = fs::read_to_string(&json_path)?;
    let entries: HashMap<String, SpriteEntry> = serde_json::from_str(&json_content)?;

    // Load PNG if exists
    let (image_data, image_width, image_height) = if png_path.exists() {
        load_png_rgba(&png_path)?
    } else {
        (None, 0, 0)
    };

    Ok(SpriteAtlas {
        entries,
        image_data,
        image_width,
        image_height,
        pixel_ratio,
    })
}

/// Load PNG file as RGBA bytes.
fn load_png_rgba(path: &Path) -> Result<(Option<Vec<u8>>, u32, u32), SpriteError> {
    let data = fs::read(path)?;
    
    // Simple PNG decoder - just get dimensions for now
    // In production, use image crate or lodepng
    if data.len() < 24 || &data[0..8] != b"\x89PNG\r\n\x1a\n" {
        return Err(SpriteError::Image("Invalid PNG signature".to_string()));
    }

    // Read IHDR chunk (must be first)
    let chunk_length = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let chunk_type = &data[12..16];
    
    if chunk_type != b"IHDR" || chunk_length < 13 {
        return Err(SpriteError::Image("Missing IHDR chunk".to_string()));
    }

    let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

    // For full image decoding, we'd need a proper PNG decoder
    // Return dimensions only for now; image loading happens in Python
    Ok((None, width, height))
}

/// Glyph range for PBF font loading.
#[derive(Debug, Clone)]
pub struct GlyphRange {
    /// Start codepoint (inclusive).
    pub start: u32,
    /// End codepoint (inclusive).
    pub end: u32,
}

impl GlyphRange {
    /// Create glyph range from codepoint.
    pub fn from_codepoint(cp: u32) -> Self {
        let range_start = (cp / 256) * 256;
        Self {
            start: range_start,
            end: range_start + 255,
        }
    }

    /// Get range filename (e.g., "0-255.pbf").
    pub fn filename(&self) -> String {
        format!("{}-{}.pbf", self.start, self.end)
    }
}

/// Glyph metrics from PBF font.
#[derive(Debug, Clone, Default)]
pub struct GlyphMetrics {
    /// Glyph ID / codepoint.
    pub id: u32,
    /// Bitmap width.
    pub width: u32,
    /// Bitmap height.
    pub height: u32,
    /// Left bearing.
    pub left: i32,
    /// Top bearing.
    pub top: i32,
    /// Horizontal advance.
    pub advance: u32,
}

/// Font stack for glyph loading.
#[derive(Debug, Clone)]
pub struct FontStack {
    /// Font names in priority order.
    pub fonts: Vec<String>,
    /// Loaded glyph metrics by codepoint.
    pub glyphs: HashMap<u32, GlyphMetrics>,
    /// SDF glyph bitmap data (indexed by codepoint).
    pub glyph_data: HashMap<u32, Vec<u8>>,
}

impl FontStack {
    pub fn new(fonts: Vec<String>) -> Self {
        Self {
            fonts,
            glyphs: HashMap::new(),
            glyph_data: HashMap::new(),
        }
    }

    /// Get glyph metrics for codepoint.
    pub fn get_glyph(&self, codepoint: u32) -> Option<&GlyphMetrics> {
        self.glyphs.get(&codepoint)
    }

    /// Check if glyph is loaded.
    pub fn has_glyph(&self, codepoint: u32) -> bool {
        self.glyphs.contains_key(&codepoint)
    }

    /// Get required glyph ranges for a string.
    pub fn required_ranges(&self, text: &str) -> Vec<GlyphRange> {
        let mut ranges: Vec<GlyphRange> = Vec::new();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for ch in text.chars() {
            let cp = ch as u32;
            let range_start = (cp / 256) * 256;
            
            if !seen.contains(&range_start) && !self.has_glyph(cp) {
                seen.insert(range_start);
                ranges.push(GlyphRange::from_codepoint(cp));
            }
        }

        ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprite_entry_serde() {
        let json = r#"{"x": 0, "y": 0, "width": 32, "height": 32}"#;
        let entry: SpriteEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.width, 32);
        assert_eq!(entry.pixel_ratio, 1.0);
        assert!(!entry.sdf);
    }

    #[test]
    fn test_sprite_atlas_uvs() {
        let mut atlas = SpriteAtlas::new();
        atlas.image_width = 256;
        atlas.image_height = 256;
        atlas.entries.insert(
            "test".to_string(),
            SpriteEntry {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
                pixel_ratio: 1.0,
                sdf: false,
                content: None,
                stretch_x: None,
                stretch_y: None,
            },
        );

        let uvs = atlas.get_uvs("test").unwrap();
        assert!((uvs[0] - 0.0).abs() < 0.001);
        assert!((uvs[1] - 0.0).abs() < 0.001);
        assert!((uvs[2] - 0.125).abs() < 0.001); // 32/256
        assert!((uvs[3] - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_glyph_range() {
        let range = GlyphRange::from_codepoint(65); // 'A'
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 255);
        assert_eq!(range.filename(), "0-255.pbf");

        let range = GlyphRange::from_codepoint(0x4E00); // CJK
        assert_eq!(range.start, 19712);
        assert_eq!(range.filename(), "19712-19967.pbf");
    }

    #[test]
    fn test_font_stack_ranges() {
        let stack = FontStack::new(vec!["Open Sans Regular".to_string()]);
        let ranges = stack.required_ranges("Hello");
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0);
    }
}
