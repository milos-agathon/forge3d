//! SVG text generation for labels.
//!
//! Generates SVG `<text>` elements from label data, including
//! halo (outline) effects using text-shadow or duplicate elements.

use crate::labels::{LabelData, LabelStyle};

/// Configuration for label SVG export.
#[derive(Debug, Clone)]
pub struct LabelSvgConfig {
    /// Font family (default: "sans-serif").
    pub font_family: String,
    /// Font weight (default: "normal").
    pub font_weight: String,
    /// Use text-shadow for halo (CSS, may not work in all SVG viewers).
    /// If false, uses duplicate text elements for better compatibility.
    pub use_text_shadow: bool,
    /// Decimal precision for coordinate values.
    pub precision: u8,
}

impl Default for LabelSvgConfig {
    fn default() -> Self {
        Self {
            font_family: "sans-serif".to_string(),
            font_weight: "normal".to_string(),
            use_text_shadow: false,
            precision: 2,
        }
    }
}

/// Convert RGBA color (0..1) to CSS color string.
fn color_to_css(c: [f32; 4]) -> String {
    let r = (c[0].clamp(0.0, 1.0) * 255.0) as u8;
    let g = (c[1].clamp(0.0, 1.0) * 255.0) as u8;
    let b = (c[2].clamp(0.0, 1.0) * 255.0) as u8;
    let a = c[3].clamp(0.0, 1.0);
    if (a - 1.0).abs() < 0.001 {
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    } else {
        format!("rgba({},{},{},{:.2})", r, g, b, a)
    }
}

/// Format coordinate with specified precision.
fn format_coord(value: f32, precision: u8) -> String {
    format!("{:.prec$}", value, prec = precision as usize)
}

/// Escape text for XML/SVG.
fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Generate SVG text elements for a single label.
fn label_to_svg_elements(label: &LabelData, config: &LabelSvgConfig) -> String {
    let Some(screen_pos) = label.screen_pos else {
        return String::new();
    };

    if !label.visible {
        return String::new();
    }

    let x = format_coord(screen_pos[0], config.precision);
    let y = format_coord(screen_pos[1], config.precision);
    let font_size = format_coord(label.style.size, config.precision);
    let text = escape_xml(&label.text);

    let mut result = String::new();

    // Common text attributes
    let text_attrs = format!(
        r#"x="{}" y="{}" font-family="{}" font-size="{}" font-weight="{}" text-anchor="middle" dominant-baseline="middle""#,
        x, y, config.font_family, font_size, config.font_weight
    );

    // Halo (outline) effect
    if label.style.halo_width > 0.0 && label.style.halo_color[3] > 0.001 {
        if config.use_text_shadow {
            // Use stroke for halo (works in most SVG viewers)
            let halo_color = color_to_css(label.style.halo_color);
            let stroke_width = format_coord(label.style.halo_width * 2.0, config.precision);
            result.push_str(&format!(
                r#"  <text {} fill="{}" stroke="{}" stroke-width="{}" stroke-linejoin="round">{}</text>
"#,
                text_attrs, halo_color, halo_color, stroke_width, text
            ));
        } else {
            // Duplicate text with stroke for halo (better compatibility)
            let halo_color = color_to_css(label.style.halo_color);
            let stroke_width = format_coord(label.style.halo_width * 2.0, config.precision);
            result.push_str(&format!(
                r#"  <text {} fill="none" stroke="{}" stroke-width="{}" stroke-linejoin="round">{}</text>
"#,
                text_attrs, halo_color, stroke_width, text
            ));
        }
    }

    // Main text
    let fill_color = color_to_css(label.style.color);
    result.push_str(&format!(
        r#"  <text {} fill="{}">{}</text>
"#,
        text_attrs, fill_color, text
    ));

    result
}

/// Generate SVG text elements for all visible labels.
///
/// # Arguments
/// * `labels` - Slice of label data with computed screen positions
///
/// # Returns
/// SVG fragment containing `<text>` elements (not a complete SVG document).
///
/// # Example
/// ```ignore
/// let label_svg = labels_to_svg_text(&labels);
/// let complete_svg = format!(
///     r#"<?xml version="1.0"?>
/// <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
/// {}
/// </svg>"#,
///     label_svg
/// );
/// ```
pub fn labels_to_svg_text(labels: &[LabelData]) -> String {
    labels_to_svg_text_with_config(labels, &LabelSvgConfig::default())
}

/// Generate SVG text elements with custom configuration.
pub fn labels_to_svg_text_with_config(labels: &[LabelData], config: &LabelSvgConfig) -> String {
    let mut result = String::with_capacity(256 * labels.len());

    for label in labels {
        result.push_str(&label_to_svg_elements(label, config));
    }

    result
}

/// Generate a complete SVG document with labels.
///
/// # Arguments
/// * `labels` - Slice of label data with computed screen positions
/// * `width` - SVG width
/// * `height` - SVG height
/// * `config` - Label rendering configuration
///
/// # Returns
/// Complete SVG document as a string.
pub fn labels_to_svg_document(
    labels: &[LabelData],
    width: u32,
    height: u32,
    config: &LabelSvgConfig,
) -> String {
    let label_elements = labels_to_svg_text_with_config(labels, config);

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">
{}
</svg>"#,
        width, height, label_elements
    )
}

/// Generate SVG text for a label at a specific position (without pre-computed screen_pos).
///
/// Useful when you want to place labels programmatically without using the label manager.
pub fn label_at_position(
    text: &str,
    x: f32,
    y: f32,
    style: &LabelStyle,
    config: &LabelSvgConfig,
) -> String {
    let x_str = format_coord(x, config.precision);
    let y_str = format_coord(y, config.precision);
    let font_size = format_coord(style.size, config.precision);
    let escaped_text = escape_xml(text);

    let text_attrs = format!(
        r#"x="{}" y="{}" font-family="{}" font-size="{}" font-weight="{}" text-anchor="middle" dominant-baseline="middle""#,
        x_str, y_str, config.font_family, font_size, config.font_weight
    );

    let mut result = String::new();

    // Halo
    if style.halo_width > 0.0 && style.halo_color[3] > 0.001 {
        let halo_color = color_to_css(style.halo_color);
        let stroke_width = format_coord(style.halo_width * 2.0, config.precision);
        result.push_str(&format!(
            r#"  <text {} fill="none" stroke="{}" stroke-width="{}" stroke-linejoin="round">{}</text>
"#,
            text_attrs, halo_color, stroke_width, escaped_text
        ));
    }

    // Main text
    let fill_color = color_to_css(style.color);
    result.push_str(&format!(
        r#"  <text {} fill="{}">{}</text>
"#,
        text_attrs, fill_color, escaped_text
    ));

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::LabelId;
    use glam::Vec3;

    fn make_test_label(text: &str, x: f32, y: f32) -> LabelData {
        LabelData {
            id: LabelId(1),
            text: text.to_string(),
            world_pos: Vec3::ZERO,
            style: LabelStyle::default(),
            screen_pos: Some([x, y]),
            visible: true,
            depth: 0.5,
            horizon_angle: 0.0,
            computed_alpha: 1.0,
        }
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("Hello & World"), "Hello &amp; World");
        assert_eq!(escape_xml("<test>"), "&lt;test&gt;");
        assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
    }

    #[test]
    fn test_single_label() {
        let labels = vec![make_test_label("Test", 100.0, 200.0)];
        let svg = labels_to_svg_text(&labels);

        assert!(svg.contains("<text"));
        assert!(svg.contains("Test"));
        assert!(svg.contains("x=\"100.00\""));
        assert!(svg.contains("y=\"200.00\""));
    }

    #[test]
    fn test_label_with_halo() {
        let mut label = make_test_label("Halo Test", 50.0, 50.0);
        label.style.halo_width = 2.0;
        label.style.halo_color = [1.0, 1.0, 1.0, 0.8];

        let svg = labels_to_svg_text(&[label]);

        // Should have two text elements: halo and main
        assert_eq!(svg.matches("<text").count(), 2);
        assert!(svg.contains("stroke="));
    }

    #[test]
    fn test_invisible_label() {
        let mut label = make_test_label("Hidden", 100.0, 100.0);
        label.visible = false;

        let svg = labels_to_svg_text(&[label]);
        assert!(svg.is_empty());
    }

    #[test]
    fn test_label_without_screen_pos() {
        let mut label = make_test_label("No Position", 0.0, 0.0);
        label.screen_pos = None;

        let svg = labels_to_svg_text(&[label]);
        assert!(svg.is_empty());
    }

    #[test]
    fn test_complete_document() {
        let labels = vec![make_test_label("Document Test", 400.0, 300.0)];
        let svg = labels_to_svg_document(&labels, 800, 600, &LabelSvgConfig::default());

        assert!(svg.contains("<?xml version"));
        assert!(svg.contains("viewBox=\"0 0 800 600\""));
        assert!(svg.contains("Document Test"));
        assert!(svg.contains("</svg>"));
    }
}
