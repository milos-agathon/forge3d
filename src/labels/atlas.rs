//! MSDF font atlas loading and text layout.

use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::core::text_overlay::TextInstance;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Device, Queue, TextureView};

/// Metrics for a single glyph in the atlas.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetrics {
    /// Unicode codepoint.
    pub codepoint: u32,
    /// UV coordinates in atlas [u0, v0, u1, v1].
    pub uv: [f32; 4],
    /// Glyph width in atlas pixels.
    pub width: f32,
    /// Glyph height in atlas pixels.
    pub height: f32,
    /// Horizontal offset from cursor to glyph origin.
    pub offset_x: f32,
    /// Vertical offset from baseline to glyph top.
    pub offset_y: f32,
    /// Horizontal advance after this glyph.
    pub advance: f32,
}

/// MSDF font atlas with glyph metrics.
pub struct MsdfAtlas {
    pub texture: Arc<TrackedTexture>,
    pub view: Arc<TextureView>,
    pub width: u32,
    pub height: u32,
    /// Glyph metrics indexed by Unicode codepoint.
    glyphs: HashMap<u32, GlyphMetrics>,
    /// Font size used when generating the atlas.
    pub atlas_font_size: f32,
    /// Line height in atlas pixels.
    pub line_height: f32,
    /// Baseline offset from top of line.
    pub baseline: f32,
}

#[derive(Debug, Deserialize)]
struct AtlasMetricsJson {
    font_size: Option<f32>,
    line_height: Option<f32>,
    baseline: Option<f32>,
    glyphs: HashMap<String, GlyphMetricsJson>,
}

#[derive(Debug, Deserialize)]
struct GlyphMetricsJson {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    #[serde(default)]
    ox: f32,
    #[serde(default)]
    oy: f32,
    adv: Option<f32>,
}

impl MsdfAtlas {
    /// Load an MSDF atlas from raw image data and JSON metrics.
    pub fn load(
        device: &Device,
        queue: &Queue,
        atlas_image: &[u8],
        atlas_width: u32,
        atlas_height: u32,
        metrics_json: &str,
    ) -> Result<Self, String> {
        // Allocate the atlas texture through the tracked wrapper.
        let texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("msdf_atlas"),
                size: wgpu::Extent3d {
                    width: atlas_width,
                    height: atlas_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )
        .map_err(|e| e.to_string())?;

        // Upload image data
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            atlas_image,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * atlas_width),
                rows_per_image: Some(atlas_height),
            },
            wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Parse metrics
        let (glyphs, atlas_font_size, line_height, baseline) =
            Self::parse_metrics(metrics_json, atlas_width, atlas_height)?;

        Ok(Self {
            texture: Arc::new(texture),
            view: Arc::new(view),
            width: atlas_width,
            height: atlas_height,
            glyphs,
            atlas_font_size,
            line_height,
            baseline,
        })
    }

    /// Load atlas from PNG file and JSON metrics file.
    pub fn load_from_files(
        device: &Device,
        queue: &Queue,
        atlas_png_path: &str,
        metrics_json_path: &str,
    ) -> Result<Self, String> {
        // Load PNG using the image crate
        let img =
            image::open(atlas_png_path).map_err(|e| format!("Failed to load atlas PNG: {}", e))?;

        let width = img.width();
        let height = img.height();
        let rgba_data = img.to_rgba8().into_raw();

        // Load JSON metrics
        let metrics_json = std::fs::read_to_string(metrics_json_path)
            .map_err(|e| format!("Failed to read metrics JSON: {}", e))?;

        Self::load(device, queue, &rgba_data, width, height, &metrics_json)
    }

    /// Parse metrics from JSON.
    /// Supports a simplified format:
    /// {
    ///   "font_size": 32,
    ///   "line_height": 40,
    ///   "baseline": 32,
    ///   "glyphs": {
    ///     "65": { "x": 0, "y": 0, "w": 20, "h": 30, "ox": 0, "oy": 0, "adv": 22 },
    ///     ...
    ///   }
    /// }
    fn parse_metrics(
        json: &str,
        atlas_width: u32,
        atlas_height: u32,
    ) -> Result<(HashMap<u32, GlyphMetrics>, f32, f32, f32), String> {
        let parsed: AtlasMetricsJson =
            serde_json::from_str(json).map_err(|e| format!("Invalid atlas metrics JSON: {e}"))?;
        if parsed.glyphs.is_empty() {
            return Err("Atlas metrics must contain at least one glyph".to_string());
        }

        let font_size = parsed.font_size.unwrap_or(32.0);
        let line_height = parsed.line_height.unwrap_or(font_size * 1.25);
        let baseline = parsed.baseline.unwrap_or(font_size);
        let mut glyphs = HashMap::with_capacity(parsed.glyphs.len());

        for (codepoint_str, glyph) in parsed.glyphs {
            let codepoint = codepoint_str.parse::<u32>().map_err(|_| {
                format!("Atlas glyph key is not a Unicode codepoint: {codepoint_str}")
            })?;
            if glyph.w <= 0.0 || glyph.h <= 0.0 {
                return Err(format!(
                    "Atlas glyph {codepoint} has non-positive dimensions"
                ));
            }
            let u0 = glyph.x / atlas_width as f32;
            let v0 = glyph.y / atlas_height as f32;
            let u1 = (glyph.x + glyph.w) / atlas_width as f32;
            let v1 = (glyph.y + glyph.h) / atlas_height as f32;
            glyphs.insert(
                codepoint,
                GlyphMetrics {
                    codepoint,
                    uv: [u0, v0, u1, v1],
                    width: glyph.w,
                    height: glyph.h,
                    offset_x: glyph.ox,
                    offset_y: glyph.oy,
                    advance: glyph.adv.unwrap_or(glyph.w),
                },
            );
        }

        Ok((glyphs, font_size, line_height, baseline))
    }

    /// Get glyph metrics for a character.
    pub fn get_glyph(&self, c: char) -> Option<&GlyphMetrics> {
        self.glyphs.get(&(c as u32))
    }

    /// Measure text dimensions at a given size.
    /// Returns (width, height) in pixels.
    pub fn measure_text(&self, text: &str, size: f32) -> (f32, f32) {
        let scale = size / self.atlas_font_size;
        let mut width = 0.0f32;
        let mut max_height = 0.0f32;

        for c in text.chars() {
            if let Some(glyph) = self.get_glyph(c) {
                width += glyph.advance * scale;
                max_height = max_height.max(glyph.height * scale);
            } else if c == ' ' {
                // Space fallback
                width += size * 0.3;
            }
        }

        (width, max_height.max(size))
    }

    /// Layout text into TextInstance quads.
    /// Generates one SDF/MSDF instance per glyph; the shader expands halos.
    pub fn layout_text(
        &self,
        text: &str,
        center_pos: [f32; 2],
        size: f32,
        color: [f32; 4],
        halo_color: [f32; 4],
        halo_width: f32,
    ) -> Vec<TextInstance> {
        let mut instances = Vec::new();
        let scale = size / self.atlas_font_size;

        // Measure total width to center
        let (total_width, total_height) = self.measure_text(text, size);
        let start_x = center_pos[0] - total_width * 0.5;
        let start_y = center_pos[1] - total_height * 0.5;

        let mut cursor_x = start_x;
        for c in text.chars() {
            if let Some(glyph) = self.get_glyph(c) {
                let x0 = cursor_x + glyph.offset_x * scale;
                let y0 = start_y + glyph.offset_y * scale;
                let x1 = x0 + glyph.width * scale;
                let y1 = y0 + glyph.height * scale;

                instances.push(
                    TextInstance::new(
                        [x0, y0],
                        [x1, y1],
                        [glyph.uv[0], glyph.uv[1]],
                        [glyph.uv[2], glyph.uv[3]],
                        color,
                    )
                    .with_halo(halo_color, halo_width),
                );

                cursor_x += glyph.advance * scale;
            } else if c == ' ' {
                cursor_x += size * 0.3;
            }
        }

        instances
    }
}

impl Clone for MsdfAtlas {
    fn clone(&self) -> Self {
        Self {
            texture: Arc::clone(&self.texture),
            view: Arc::clone(&self.view),
            width: self.width,
            height: self.height,
            glyphs: self.glyphs.clone(),
            atlas_font_size: self.atlas_font_size,
            line_height: self.line_height,
            baseline: self.baseline,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MsdfAtlas;

    #[test]
    fn parses_metrics_with_serde_json() {
        let json = r#"{
            "font_size": 24,
            "line_height": 32,
            "baseline": 20,
            "glyphs": {
                "65": {"x": 4, "y": 8, "w": 16, "h": 18, "ox": -2, "oy": 1, "adv": 17}
            }
        }"#;

        let (glyphs, font_size, line_height, baseline) =
            MsdfAtlas::parse_metrics(json, 64, 64).expect("metrics should parse");
        let glyph = glyphs.get(&65).expect("A glyph should be present");

        assert_eq!(font_size, 24.0);
        assert_eq!(line_height, 32.0);
        assert_eq!(baseline, 20.0);
        assert_eq!(glyph.width, 16.0);
        assert_eq!(glyph.height, 18.0);
        assert_eq!(glyph.advance, 17.0);
        assert_eq!(glyph.uv, [4.0 / 64.0, 8.0 / 64.0, 20.0 / 64.0, 26.0 / 64.0]);
    }

    #[test]
    fn rejects_empty_or_malformed_metrics() {
        assert!(MsdfAtlas::parse_metrics(r#"{"glyphs": {}}"#, 64, 64).is_err());
        assert!(MsdfAtlas::parse_metrics(
            r#"{"font_size": 12, "line_height": 16, "baseline": 12, "glyphs": {"A": {"x": 0, "y": 0, "w": 1, "h": 1}}}"#,
            64,
            64
        )
        .is_err());
    }
}
