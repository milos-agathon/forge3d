//! SVG label export from the same positioned font outlines used by raster text.

use crate::labels::font::{FontCollection, FontRequest};
use crate::labels::positioned::{outline_bounds, positioned_outlines, svg_path_data};
use crate::labels::shape;
use crate::labels::{LabelData, LabelStyle};
use std::fmt;
use std::sync::{Arc, OnceLock};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvgTextError {
    UnsupportedFontFamily(String),
    UnsupportedFontWeight(String),
    Outline(String),
}

impl fmt::Display for SvgTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFontFamily(value) => {
                write!(formatter, "unsupported SVG font family {value:?}")
            }
            Self::UnsupportedFontWeight(value) => {
                write!(formatter, "unsupported SVG font weight {value:?}")
            }
            Self::Outline(value) => formatter.write_str(value),
        }
    }
}

impl std::error::Error for SvgTextError {}

fn validate_config(config: &LabelSvgConfig) -> Result<(), SvgTextError> {
    if !matches!(
        config.font_family.trim().to_ascii_lowercase().as_str(),
        "sans-serif" | "noto sans" | "noto-sans"
    ) {
        return Err(SvgTextError::UnsupportedFontFamily(
            config.font_family.clone(),
        ));
    }
    if !matches!(
        config.font_weight.trim().to_ascii_lowercase().as_str(),
        "normal" | "400"
    ) {
        return Err(SvgTextError::UnsupportedFontWeight(
            config.font_weight.clone(),
        ));
    }
    Ok(())
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
    static FONTS: OnceLock<Result<Arc<FontCollection>, String>> = OnceLock::new();
    FONTS
        .get_or_init(|| {
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
        })
        .clone()
}

fn outline_geometry(
    text: &str,
    size: f32,
    precision: u8,
) -> Result<(String, [f32; 4]), SvgTextError> {
    let fonts = default_fonts().map_err(SvgTextError::Outline)?;
    let shaped = shape::shape(text, fonts, size, None, None, &[])
        .map_err(|error| SvgTextError::Outline(error.to_string()))?;
    let range = 0..text.chars().count();
    let outlines = positioned_outlines(&shaped, std::slice::from_ref(&range))
        .map_err(|error| SvgTextError::Outline(error.to_string()))?;
    let bounds = outline_bounds(&outlines).unwrap_or([0.0; 4]);
    Ok((svg_path_data(&outlines, precision), bounds))
}

fn path_elements(
    text: &str,
    x: f32,
    y: f32,
    style: &LabelStyle,
    config: &LabelSvgConfig,
) -> Result<String, SvgTextError> {
    validate_config(config)?;
    let (path, bounds) = outline_geometry(text, style.size, config.precision)?;
    let translate_x = x - (bounds[0] + bounds[2]) * 0.5;
    let translate_y = y - (bounds[1] + bounds[3]) * 0.5;
    let transform = format!(
        "translate({} {})",
        format_coord(translate_x, config.precision),
        format_coord(translate_y, config.precision)
    );
    let mut output = String::new();
    if config.use_text_shadow {
        let shadow_transform = format!(
            "translate({} {})",
            format_coord(translate_x + 1.0, config.precision),
            format_coord(translate_y + 1.0, config.precision)
        );
        output.push_str(&format!(
            "  <path d=\"{path}\" transform=\"{shadow_transform}\" fill=\"rgba(0,0,0,0.35)\" fill-rule=\"nonzero\"/>\n"
        ));
    }
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

fn try_label_to_svg_elements(
    label: &LabelData,
    config: &LabelSvgConfig,
) -> Result<String, SvgTextError> {
    if !label.visible {
        return Ok(String::new());
    }
    let Some([x, y]) = label.screen_pos else {
        return Ok(String::new());
    };
    path_elements(&label.text, x, y, &label.style, config)
}

pub fn labels_to_svg_text(labels: &[LabelData]) -> String {
    labels_to_svg_text_with_config(labels, &LabelSvgConfig::default())
}

pub fn labels_to_svg_text_with_config(labels: &[LabelData], config: &LabelSvgConfig) -> String {
    try_labels_to_svg_text_with_config(labels, config)
        .unwrap_or_else(|error| panic!("SVG label configuration rejected: {error}"))
}

pub fn try_labels_to_svg_text_with_config(
    labels: &[LabelData],
    config: &LabelSvgConfig,
) -> Result<String, SvgTextError> {
    labels
        .iter()
        .map(|label| try_label_to_svg_elements(label, config))
        .collect()
}

pub fn labels_to_svg_document(
    labels: &[LabelData],
    width: u32,
    height: u32,
    config: &LabelSvgConfig,
) -> String {
    try_labels_to_svg_document(labels, width, height, config)
        .unwrap_or_else(|error| panic!("SVG label configuration rejected: {error}"))
}

pub fn try_labels_to_svg_document(
    labels: &[LabelData],
    width: u32,
    height: u32,
    config: &LabelSvgConfig,
) -> Result<String, SvgTextError> {
    Ok(format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">\n{}</svg>",
        try_labels_to_svg_text_with_config(labels, config)?
    ))
}

pub fn label_at_position(
    text: &str,
    x: f32,
    y: f32,
    style: &LabelStyle,
    config: &LabelSvgConfig,
) -> String {
    try_label_at_position(text, x, y, style, config)
        .unwrap_or_else(|error| panic!("failed to outline SVG label {text:?}: {error}"))
}

pub fn try_label_at_position(
    text: &str,
    x: f32,
    y: f32,
    style: &LabelStyle,
    config: &LabelSvgConfig,
) -> Result<String, SvgTextError> {
    path_elements(text, x, y, style, config)
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

    #[test]
    fn shadow_configuration_emits_distinct_offset_geometry() {
        let plain = labels_to_svg_text(&[label("Map")]);
        let config = LabelSvgConfig {
            use_text_shadow: true,
            ..LabelSvgConfig::default()
        };
        let shadowed = try_labels_to_svg_text_with_config(&[label("Map")], &config).unwrap();

        assert_ne!(plain, shadowed);
        assert!(shadowed.contains("rgba(0,0,0,0.35)"));
        let transforms: Vec<_> = shadowed
            .lines()
            .filter_map(|line| line.split("transform=\"").nth(1)?.split('"').next())
            .collect();
        assert_eq!(transforms.len(), 3);
        assert_ne!(transforms[0], transforms[1]);
        assert_eq!(transforms[1], transforms[2]);
    }

    #[test]
    fn unsupported_font_semantics_are_explicit_errors() {
        let family = LabelSvgConfig {
            font_family: "Host Font".to_owned(),
            ..LabelSvgConfig::default()
        };
        assert_eq!(
            try_labels_to_svg_text_with_config(&[label("Map")], &family).unwrap_err(),
            SvgTextError::UnsupportedFontFamily("Host Font".to_owned())
        );

        let weight = LabelSvgConfig {
            font_weight: "bold".to_owned(),
            ..LabelSvgConfig::default()
        };
        assert_eq!(
            try_labels_to_svg_text_with_config(&[label("Map")], &weight).unwrap_err(),
            SvgTextError::UnsupportedFontWeight("bold".to_owned())
        );
    }
}
