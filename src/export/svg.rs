//! SVG generation for vector geometry.
//!
//! Generates SVG documents from polygon and polyline definitions,
//! suitable for print-grade vector export.

use crate::vector::api::{PolygonDef, PolylineDef, VectorStyle};

use super::projection::{project_2d_to_screen, Bounds2D};

/// Configuration options for SVG export.
#[derive(Debug, Clone)]
pub struct SvgExportConfig {
    /// Decimal precision for coordinate values (default: 2).
    pub precision: u8,
    /// Optional background color (RGBA).
    pub background: Option<[f32; 4]>,
    /// Whether to include XML declaration (default: true).
    pub xml_declaration: bool,
    /// Whether to include a viewBox attribute (default: true).
    pub include_viewbox: bool,
    /// Stroke line cap style (default: "round").
    pub stroke_linecap: String,
    /// Stroke line join style (default: "round").
    pub stroke_linejoin: String,
}

impl Default for SvgExportConfig {
    fn default() -> Self {
        Self {
            precision: 2,
            background: None,
            xml_declaration: true,
            include_viewbox: true,
            stroke_linecap: "round".to_string(),
            stroke_linejoin: "round".to_string(),
        }
    }
}

impl SvgExportConfig {
    /// Create config with custom precision.
    pub fn with_precision(mut self, precision: u8) -> Self {
        self.precision = precision;
        self
    }

    /// Set background color.
    pub fn with_background(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.background = Some([r, g, b, a]);
        self
    }

    /// Disable XML declaration for embedding in HTML.
    pub fn without_xml_declaration(mut self) -> Self {
        self.xml_declaration = false;
        self
    }
}

/// Convert RGBA color (0..1) to CSS hex color string (#RRGGBB).
fn color_to_hex(c: [f32; 4]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8
    )
}

/// Convert RGBA color (0..1) to CSS rgba() string.
fn color_to_rgba(c: [f32; 4]) -> String {
    let r = (c[0].clamp(0.0, 1.0) * 255.0) as u8;
    let g = (c[1].clamp(0.0, 1.0) * 255.0) as u8;
    let b = (c[2].clamp(0.0, 1.0) * 255.0) as u8;
    let a = c[3].clamp(0.0, 1.0);
    if (a - 1.0).abs() < 0.001 {
        // Fully opaque, use hex
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    } else {
        format!("rgba({},{},{},{:.2})", r, g, b, a)
    }
}

/// Format a coordinate with specified precision.
fn format_coord(value: f32, precision: u8) -> String {
    format!("{:.prec$}", value, prec = precision as usize)
}

/// Generate style attributes for a vector element.
fn style_attributes(style: &VectorStyle, config: &SvgExportConfig, is_polygon: bool) -> String {
    let mut attrs = Vec::new();

    if is_polygon {
        // Fill for polygons
        if style.fill_color[3] > 0.001 {
            attrs.push(format!("fill=\"{}\"", color_to_rgba(style.fill_color)));
        } else {
            attrs.push("fill=\"none\"".to_string());
        }
    } else {
        // No fill for polylines
        attrs.push("fill=\"none\"".to_string());
    }

    // Stroke
    if style.stroke_width > 0.0 && style.stroke_color[3] > 0.001 {
        attrs.push(format!("stroke=\"{}\"", color_to_rgba(style.stroke_color)));
        attrs.push(format!(
            "stroke-width=\"{}\"",
            format_coord(style.stroke_width, config.precision)
        ));
        attrs.push(format!("stroke-linecap=\"{}\"", config.stroke_linecap));
        attrs.push(format!("stroke-linejoin=\"{}\"", config.stroke_linejoin));
    } else if !is_polygon {
        // Polylines need a default stroke
        attrs.push("stroke=\"#000000\"".to_string());
        attrs.push("stroke-width=\"1\"".to_string());
    }

    attrs.join(" ")
}

/// Generate SVG from vector geometry.
///
/// # Arguments
/// * `polygons` - Polygon definitions with exterior rings and optional holes
/// * `polylines` - Polyline definitions with path coordinates
/// * `bounds` - Bounding box for coordinate mapping
/// * `width` - Output SVG width in pixels
/// * `height` - Output SVG height in pixels
/// * `config` - Export configuration options
///
/// # Returns
/// Complete SVG document as a string.
///
/// # Example
/// ```ignore
/// let svg = vectors_to_svg(
///     &polygons,
///     &polylines,
///     &bounds,
///     800,
///     600,
///     &SvgExportConfig::default()
/// );
/// std::fs::write("output.svg", svg)?;
/// ```
pub fn vectors_to_svg(
    polygons: &[PolygonDef],
    polylines: &[PolylineDef],
    bounds: &Bounds2D,
    width: u32,
    height: u32,
    config: &SvgExportConfig,
) -> String {
    let viewport = (width, height);
    let precision = config.precision;

    let mut svg = String::with_capacity(1024 * polygons.len() + 256 * polylines.len() + 512);

    // XML declaration
    if config.xml_declaration {
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    }

    // SVG root element
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\"{}>\n",
        if config.include_viewbox {
            format!(" viewBox=\"0 0 {} {}\"", width, height)
        } else {
            format!(" width=\"{}\" height=\"{}\"", width, height)
        }
    ));

    // Background rectangle
    if let Some(bg) = config.background {
        svg.push_str(&format!(
            "  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
            width,
            height,
            color_to_rgba(bg)
        ));
    }

    // Polygons
    for polygon in polygons {
        // Project exterior ring
        let exterior_points: String = polygon
            .exterior
            .iter()
            .map(|p| {
                let (x, y) = project_2d_to_screen(*p, bounds, viewport);
                format!(
                    "{},{}",
                    format_coord(x, precision),
                    format_coord(y, precision)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        if polygon.holes.is_empty() {
            // Simple polygon without holes
            svg.push_str(&format!(
                "  <polygon points=\"{}\" {}/>\n",
                exterior_points,
                style_attributes(&polygon.style, config, true)
            ));
        } else {
            // Polygon with holes - use path element with fill-rule
            let mut d = String::new();

            // Exterior ring (as move-to + line-to + close)
            for (i, p) in polygon.exterior.iter().enumerate() {
                let (x, y) = project_2d_to_screen(*p, bounds, viewport);
                if i == 0 {
                    d.push_str(&format!(
                        "M{},{}",
                        format_coord(x, precision),
                        format_coord(y, precision)
                    ));
                } else {
                    d.push_str(&format!(
                        " L{},{}",
                        format_coord(x, precision),
                        format_coord(y, precision)
                    ));
                }
            }
            d.push_str(" Z");

            // Holes (in reverse winding for evenodd fill rule)
            for hole in &polygon.holes {
                for (i, p) in hole.iter().enumerate() {
                    let (x, y) = project_2d_to_screen(*p, bounds, viewport);
                    if i == 0 {
                        d.push_str(&format!(
                            " M{},{}",
                            format_coord(x, precision),
                            format_coord(y, precision)
                        ));
                    } else {
                        d.push_str(&format!(
                            " L{},{}",
                            format_coord(x, precision),
                            format_coord(y, precision)
                        ));
                    }
                }
                d.push_str(" Z");
            }

            svg.push_str(&format!(
                "  <path d=\"{}\" fill-rule=\"evenodd\" {}/>\n",
                d,
                style_attributes(&polygon.style, config, true)
            ));
        }
    }

    // Polylines
    for polyline in polylines {
        let points: String = polyline
            .path
            .iter()
            .map(|p| {
                let (x, y) = project_2d_to_screen(*p, bounds, viewport);
                format!(
                    "{},{}",
                    format_coord(x, precision),
                    format_coord(y, precision)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        svg.push_str(&format!(
            "  <polyline points=\"{}\" {}/>\n",
            points,
            style_attributes(&polyline.style, config, false)
        ));
    }

    svg.push_str("</svg>");

    svg
}

/// Generate SVG from pre-projected screen coordinates.
///
/// Use this when you have already projected 3D coordinates to screen space.
pub fn vectors_to_svg_screen_coords(
    polygons: &[(Vec<(f32, f32)>, VectorStyle)],
    polylines: &[(Vec<(f32, f32)>, VectorStyle)],
    width: u32,
    height: u32,
    config: &SvgExportConfig,
) -> String {
    let precision = config.precision;

    let mut svg = String::with_capacity(1024 * polygons.len() + 256 * polylines.len() + 512);

    // XML declaration
    if config.xml_declaration {
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    }

    // SVG root element
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\"{}>\n",
        if config.include_viewbox {
            format!(" viewBox=\"0 0 {} {}\"", width, height)
        } else {
            format!(" width=\"{}\" height=\"{}\"", width, height)
        }
    ));

    // Background
    if let Some(bg) = config.background {
        svg.push_str(&format!(
            "  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
            width,
            height,
            color_to_rgba(bg)
        ));
    }

    // Polygons
    for (vertices, style) in polygons {
        let points: String = vertices
            .iter()
            .map(|(x, y)| {
                format!(
                    "{},{}",
                    format_coord(*x, precision),
                    format_coord(*y, precision)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        svg.push_str(&format!(
            "  <polygon points=\"{}\" {}/>\n",
            points,
            style_attributes(style, config, true)
        ));
    }

    // Polylines
    for (vertices, style) in polylines {
        let points: String = vertices
            .iter()
            .map(|(x, y)| {
                format!(
                    "{},{}",
                    format_coord(*x, precision),
                    format_coord(*y, precision)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        svg.push_str(&format!(
            "  <polyline points=\"{}\" {}/>\n",
            points,
            style_attributes(style, config, false)
        ));
    }

    svg.push_str("</svg>");

    svg
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn test_color_to_hex() {
        assert_eq!(color_to_hex([1.0, 0.0, 0.0, 1.0]), "#ff0000");
        assert_eq!(color_to_hex([0.0, 1.0, 0.0, 1.0]), "#00ff00");
        assert_eq!(color_to_hex([0.0, 0.0, 1.0, 1.0]), "#0000ff");
        assert_eq!(color_to_hex([0.5, 0.5, 0.5, 1.0]), "#7f7f7f");
    }

    #[test]
    fn test_color_to_rgba() {
        assert_eq!(color_to_rgba([1.0, 0.0, 0.0, 1.0]), "#ff0000");
        assert_eq!(color_to_rgba([1.0, 0.0, 0.0, 0.5]), "rgba(255,0,0,0.50)");
    }

    #[test]
    fn test_empty_svg() {
        let svg = vectors_to_svg(
            &[],
            &[],
            &Bounds2D::default(),
            100,
            100,
            &SvgExportConfig::default(),
        );
        assert!(svg.contains("<?xml version"));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_polygon_with_holes() {
        let polygon = PolygonDef {
            exterior: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(100.0, 0.0),
                Vec2::new(100.0, 100.0),
                Vec2::new(0.0, 100.0),
            ],
            holes: vec![vec![
                Vec2::new(25.0, 25.0),
                Vec2::new(75.0, 25.0),
                Vec2::new(75.0, 75.0),
                Vec2::new(25.0, 75.0),
            ]],
            style: VectorStyle::default(),
        };

        let bounds = Bounds2D::from_extents(0.0, 0.0, 100.0, 100.0);
        let svg = vectors_to_svg(&[polygon], &[], &bounds, 100, 100, &SvgExportConfig::default());

        assert!(svg.contains("<path"));
        assert!(svg.contains("fill-rule=\"evenodd\""));
        assert!(svg.contains(" M")); // Hole starts with M
    }
}
