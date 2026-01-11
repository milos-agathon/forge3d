//! MSDF font atlas loading and text layout.

use crate::core::text_overlay::TextInstance;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Device, Queue, Texture, TextureView};

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
    pub texture: Arc<Texture>,
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
        // Create texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
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
        });

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
        let img = image::open(atlas_png_path)
            .map_err(|e| format!("Failed to load atlas PNG: {}", e))?;

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
        // Simple JSON parsing without external dependencies
        let mut glyphs = HashMap::new();
        let mut font_size = 32.0f32;
        let mut line_height = 40.0f32;
        let mut baseline = 32.0f32;

        // Try to parse as JSON
        // This is a minimal parser - in production, use serde_json
        let json = json.trim();

        // Extract font_size
        if let Some(pos) = json.find("\"font_size\"") {
            if let Some(colon) = json[pos..].find(':') {
                let start = pos + colon + 1;
                let end = json[start..]
                    .find(|c: char| c == ',' || c == '}')
                    .map(|p| start + p)
                    .unwrap_or(json.len());
                if let Ok(v) = json[start..end].trim().parse::<f32>() {
                    font_size = v;
                }
            }
        }

        // Extract line_height
        if let Some(pos) = json.find("\"line_height\"") {
            if let Some(colon) = json[pos..].find(':') {
                let start = pos + colon + 1;
                let end = json[start..]
                    .find(|c: char| c == ',' || c == '}')
                    .map(|p| start + p)
                    .unwrap_or(json.len());
                if let Ok(v) = json[start..end].trim().parse::<f32>() {
                    line_height = v;
                }
            }
        }

        // Extract baseline
        if let Some(pos) = json.find("\"baseline\"") {
            if let Some(colon) = json[pos..].find(':') {
                let start = pos + colon + 1;
                let end = json[start..]
                    .find(|c: char| c == ',' || c == '}')
                    .map(|p| start + p)
                    .unwrap_or(json.len());
                if let Ok(v) = json[start..end].trim().parse::<f32>() {
                    baseline = v;
                }
            }
        }

        // Extract glyphs section
        if let Some(glyphs_start) = json.find("\"glyphs\"") {
            if let Some(obj_start) = json[glyphs_start..].find('{') {
                let glyphs_str = &json[glyphs_start + obj_start..];
                // Find matching closing brace
                let mut depth = 0;
                let mut end_pos = 0;
                for (i, c) in glyphs_str.char_indices() {
                    match c {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end_pos = i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let glyphs_obj = &glyphs_str[1..end_pos - 1]; // Remove outer braces

                // Parse each glyph entry: "65": { ... }
                let mut pos = 0;
                while pos < glyphs_obj.len() {
                    // Find next key (codepoint)
                    if let Some(key_start) = glyphs_obj[pos..].find('"') {
                        let key_start = pos + key_start + 1;
                        if let Some(key_end) = glyphs_obj[key_start..].find('"') {
                            let key_end = key_start + key_end;
                            let codepoint_str = &glyphs_obj[key_start..key_end];

                            // Parse codepoint
                            if let Ok(codepoint) = codepoint_str.parse::<u32>() {
                                // Find the glyph object
                                if let Some(obj_start) = glyphs_obj[key_end..].find('{') {
                                    let obj_start = key_end + obj_start;
                                    if let Some(obj_end) = glyphs_obj[obj_start..].find('}') {
                                        let obj_end = obj_start + obj_end + 1;
                                        let glyph_str = &glyphs_obj[obj_start..obj_end];

                                        // Parse glyph properties
                                        let x = parse_json_num(glyph_str, "x").unwrap_or(0.0);
                                        let y = parse_json_num(glyph_str, "y").unwrap_or(0.0);
                                        let w = parse_json_num(glyph_str, "w").unwrap_or(0.0);
                                        let h = parse_json_num(glyph_str, "h").unwrap_or(0.0);
                                        let ox = parse_json_num(glyph_str, "ox").unwrap_or(0.0);
                                        let oy = parse_json_num(glyph_str, "oy").unwrap_or(0.0);
                                        let adv = parse_json_num(glyph_str, "adv").unwrap_or(w);

                                        // Convert to UV coordinates
                                        let u0 = x / atlas_width as f32;
                                        let v0 = y / atlas_height as f32;
                                        let u1 = (x + w) / atlas_width as f32;
                                        let v1 = (y + h) / atlas_height as f32;

                                        glyphs.insert(
                                            codepoint,
                                            GlyphMetrics {
                                                codepoint,
                                                uv: [u0, v0, u1, v1],
                                                width: w,
                                                height: h,
                                                offset_x: ox,
                                                offset_y: oy,
                                                advance: adv,
                                            },
                                        );

                                        pos = obj_end;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    pos += 1;
                }
            }
        }

        // If no glyphs were parsed, create a basic ASCII set with fallback metrics
        if glyphs.is_empty() {
            // Fallback ASCII metrics keep layout functional when glyph parsing fails.
            let glyph_w = font_size * 0.6;
            let glyph_h = font_size;
            for c in 32u32..127 {
                glyphs.insert(
                    c,
                    GlyphMetrics {
                        codepoint: c,
                        uv: [0.0, 0.0, 0.1, 0.1], // Small region
                        width: glyph_w,
                        height: glyph_h,
                        offset_x: 0.0,
                        offset_y: 0.0,
                        advance: glyph_w,
                    },
                );
            }
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
    /// Generates instances for each glyph, optionally with halo.
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

        // If halo is enabled, render halo glyphs first (behind main text)
        if halo_width > 0.0 {
            let halo_offset = halo_width;
            let offsets = [
                (-halo_offset, -halo_offset),
                (halo_offset, -halo_offset),
                (-halo_offset, halo_offset),
                (halo_offset, halo_offset),
                (-halo_offset, 0.0),
                (halo_offset, 0.0),
                (0.0, -halo_offset),
                (0.0, halo_offset),
            ];

            for (dx, dy) in offsets {
                let mut halo_cursor_x = start_x;
                for c in text.chars() {
                    if let Some(glyph) = self.get_glyph(c) {
                        let x0 = halo_cursor_x + glyph.offset_x * scale + dx;
                        let y0 = start_y + glyph.offset_y * scale + dy;
                        let x1 = x0 + glyph.width * scale;
                        let y1 = y0 + glyph.height * scale;

                        instances.push(TextInstance {
                            rect_min: [x0, y0],
                            rect_max: [x1, y1],
                            uv_min: [glyph.uv[0], glyph.uv[1]],
                            uv_max: [glyph.uv[2], glyph.uv[3]],
                            color: halo_color,
                        });

                        halo_cursor_x += glyph.advance * scale;
                    } else if c == ' ' {
                        halo_cursor_x += size * 0.3;
                    }
                }
            }
        }

        // Render main text
        let mut cursor_x = start_x;
        for c in text.chars() {
            if let Some(glyph) = self.get_glyph(c) {
                let x0 = cursor_x + glyph.offset_x * scale;
                let y0 = start_y + glyph.offset_y * scale;
                let x1 = x0 + glyph.width * scale;
                let y1 = y0 + glyph.height * scale;

                instances.push(TextInstance {
                    rect_min: [x0, y0],
                    rect_max: [x1, y1],
                    uv_min: [glyph.uv[0], glyph.uv[1]],
                    uv_max: [glyph.uv[2], glyph.uv[3]],
                    color,
                });

                cursor_x += glyph.advance * scale;
            } else if c == ' ' {
                cursor_x += size * 0.3;
            }
        }

        instances
    }
}

/// Helper to parse a numeric value from a JSON-like string.
fn parse_json_num(s: &str, key: &str) -> Option<f32> {
    let search = format!("\"{}\"", key);
    if let Some(pos) = s.find(&search) {
        if let Some(colon) = s[pos..].find(':') {
            let start = pos + colon + 1;
            let end = s[start..]
                .find(|c: char| c == ',' || c == '}')
                .map(|p| start + p)
                .unwrap_or(s.len());
            return s[start..end].trim().parse::<f32>().ok();
        }
    }
    None
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
