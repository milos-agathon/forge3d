//! SVG label export from the same positioned font outlines used by raster text.

use crate::labels::font::{FontCollection, FontRequest};
use crate::labels::positioned::{outline_bounds, positioned_outlines, svg_path_data};
use crate::labels::shape;
use crate::labels::{LabelData, LabelStyle};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LabelSvgConfig {
    pub font_family: String,
    pub font_weight: String,
    pub use_text_shadow: bool,
    pub precision: u8,
}

impl Default for LabelSvgConfig {
    fn default() -> Self {
        Self {
            font_family: "sans-serif".to_owned(),
            font_weight: "normal".to_owned(),
            use_text_shadow: false,
            precision: 2,
        }
    }
}

fn color_to_css(color: [f32; 4]) -> String {
    let r = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
    let g = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
    let b = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
    let a = color[3].clamp(0.0, 1.0);
    if (a - 1.0).abs() < 0.001 {
        format!("#{r:02x}{g:02x}{b:02x}")
    } else {
        format!("rgba({r},{g},{b},{a:.2})")
    }
}

fn format_coord(value: f32, precision: u8) -> String {
    format!("{value:.precision$}", precision = precision as usize)
}

fn default_fonts() -> Result<Arc<FontCollection>, String> {
    let requests = [
        FontRequest::from_bytes(
            "NotoSansLatin-subset.ttf",
            &include_bytes!("../../assets/fonts/NotoSansLatin-subset.ttf")[..],
        ),
        FontRequest::from_bytes(
            "NotoSansArabic-subset.ttf",
            &include_bytes!("../../assets/fonts/NotoSansArabic-subset.ttf")[..],
        ),
        FontRequest::from_bytes(
            "NotoSansHebrew-subset.ttf",
            &include_bytes!("../../assets/fonts/NotoSansHebrew-subset.ttf")[..],
        ),
        FontRequest::from_bytes(
            "NotoSansDevanagari-subset.ttf",
            &include_bytes!("../../assets/fonts/NotoSansDevanagari-subset.ttf")[..],
        ),
        FontRequest::from_bytes(
            "NotoSansSC-subset.ttf",
            &include_bytes!("../../assets/fonts/NotoSansSC-subset.ttf")[..],
        ),
    ];
    FontCollection::load(&requests)
        .map(Arc::new)
        .map_err(|error| error.to_string())
}

fn outline_geometry(text: &str, size: f32, precision: u8) -> Result<(String, [f32; 4]), String> {
    let shaped = shape::shape(text, default_fonts()?, size, None, None, &[])
        .map_err(|error| error.to_string())?;
    let range = 0..text.chars().count();
    let outlines = positioned_outlines(&shaped, std::slice::from_ref(&range))
        .map_err(|error| error.to_string())?;
    let bounds = outline_bounds(&outlines).unwrap_or([0.0; 4]);
    Ok((svg_path_data(&outlines, precision), bounds))
}

fn path_elements(
    text: &str,
    x: f32,
    y: f32,
    style: &LabelStyle,
    config: &LabelSvgConfig,
) -> Result<String, String> {
    let (path, bounds) = outline_geometry(text, style.size, config.precision)?;
    let translate_x = x - (bounds[0] + bounds[2]) * 0.5;
    let translate_y = y - (bounds[1] + bounds[3]) * 0.5;
    let transform = format!(
        "translate({} {})",
        format_coord(translate_x, config.precision),
        format_coord(translate_y, config.precision)
    );
    let mut output = String::new();
    if style.halo_width > 0.0 && style.halo_color[3] > 0.001 {
        output.push_str(&format!(
            "  <path d=\"{path}\" transform=\"{transform}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" stroke-linejoin=\"round\"/>\n",
            color_to_css(style.halo_color),
            format_coord(style.halo_width * 2.0, config.precision)
        ));
    }
    output.push_str(&format!(
        "  <path d=\"{path}\" transform=\"{transform}\" fill=\"{}\" fill-rule=\"nonzero\"/>\n",
        color_to_css(style.color)
    ));
    Ok(output)
}

fn label_to_svg_elements(label: &LabelData, config: &LabelSvgConfig) -> String {
    if !label.visible {
        return String::new();
    }
    let Some([x, y]) = label.screen_pos else {
        return String::new();
    };
    path_elements(&label.text, x, y, &label.style, config)
        .unwrap_or_else(|error| panic!("failed to outline SVG label {:?}: {error}", label.text))
}

pub fn labels_to_svg_text(labels: &[LabelData]) -> String {
    labels_to_svg_text_with_config(labels, &LabelSvgConfig::default())
}

pub fn labels_to_svg_text_with_config(labels: &[LabelData], config: &LabelSvgConfig) -> String {
    labels
        .iter()
        .map(|label| label_to_svg_elements(label, config))
        .collect()
}

pub fn labels_to_svg_document(
    labels: &[LabelData],
    width: u32,
    height: u32,
    config: &LabelSvgConfig,
) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">\n{}</svg>",
        labels_to_svg_text_with_config(labels, config)
    )
}

pub fn label_at_position(
    text: &str,
    x: f32,
    y: f32,
    style: &LabelStyle,
    config: &LabelSvgConfig,
) -> String {
    path_elements(text, x, y, style, config)
        .unwrap_or_else(|error| panic!("failed to outline SVG label {text:?}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelId;
    use glam::{DVec3, Vec3};

    fn label(text: &str) -> LabelData {
        LabelData {
            id: LabelId(1),
            text: text.to_string(),
            world_pos: DVec3::ZERO,
            render_pos: Vec3::ZERO,
            style: LabelStyle::default(),
            screen_pos: Some([50.0, 50.0]),
            visible: true,
            depth: 0.5,
            horizon_angle: 0.0,
            computed_alpha: 1.0,
        }
    }

    #[test]
    fn labels_are_outline_paths_with_shared_halo_geometry() {
        let svg = labels_to_svg_text(&[label("Map")]);
        assert!(!svg.contains("<text"));
        let paths: Vec<_> = svg
            .lines()
            .filter_map(|line| line.split("d=\"").nth(1)?.split('"').next())
            .collect();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], paths[1]);
    }

    #[test]
    fn invisible_labels_emit_nothing() {
        let mut hidden = label("Map");
        hidden.visible = false;
        assert!(labels_to_svg_text(&[hidden]).is_empty());
    }
}
