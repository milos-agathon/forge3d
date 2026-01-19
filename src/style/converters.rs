//! Converters from Mapbox Style properties to forge3d native styles.

use crate::labels::LabelStyle;
use crate::style::types::{LayoutProps, PaintProps, StyleLayer};
use crate::vector::api::VectorStyle;

/// Convert paint properties to VectorStyle for polygons/lines.
pub fn paint_to_vector_style(paint: &PaintProps) -> VectorStyle {
    let mut style = VectorStyle::default();

    // Fill color
    if let Some(ref color) = paint.fill_color {
        if let Some(rgba) = color.to_rgba() {
            style.fill_color = rgba;
        }
    }

    // Fill opacity modifies alpha
    if let Some(opacity) = paint.fill_opacity {
        style.fill_color[3] *= opacity;
    }

    // Fill outline color -> stroke color
    if let Some(ref color) = paint.fill_outline_color {
        if let Some(rgba) = color.to_rgba() {
            style.stroke_color = rgba;
        }
    }

    // Line color
    if let Some(ref color) = paint.line_color {
        if let Some(rgba) = color.to_rgba() {
            style.stroke_color = rgba;
            // For lines, stroke is primary, use same for fill
            style.fill_color = rgba;
        }
    }

    // Line opacity modifies stroke alpha
    if let Some(opacity) = paint.line_opacity {
        style.stroke_color[3] *= opacity;
    }

    // Line width
    if let Some(ref width) = paint.line_width {
        if let Some(w) = width.as_f32() {
            style.stroke_width = w;
        }
    }

    // Circle color -> fill color (for points)
    if let Some(ref color) = paint.circle_color {
        if let Some(rgba) = color.to_rgba() {
            style.fill_color = rgba;
        }
    }

    // Circle radius -> point size
    if let Some(radius) = paint.circle_radius {
        style.point_size = radius * 2.0; // diameter
    }

    // Circle opacity
    if let Some(opacity) = paint.circle_opacity {
        style.fill_color[3] *= opacity;
    }

    style
}

/// Convert layout properties to LabelStyle for text rendering.
pub fn layout_to_label_style(layout: &LayoutProps, paint: &PaintProps) -> LabelStyle {
    let mut style = LabelStyle::default();

    // Text size
    if let Some(ref size) = layout.text_size {
        if let Some(s) = size.as_f32() {
            style.size = s;
        }
    }

    // Text color
    if let Some(ref color) = paint.text_color {
        if let Some(rgba) = color.to_rgba() {
            style.color = rgba;
        }
    }

    // Text opacity modifies alpha
    if let Some(opacity) = paint.text_opacity {
        style.color[3] *= opacity;
    }

    // Halo color
    if let Some(ref color) = paint.text_halo_color {
        if let Some(rgba) = color.to_rgba() {
            style.halo_color = rgba;
        }
    }

    // Halo width
    if let Some(width) = paint.text_halo_width {
        style.halo_width = width;
    }

    // Text offset
    if let Some(ref offset) = layout.text_offset {
        if offset.len() >= 2 {
            // Mapbox uses ems; approximate to pixels (1em ≈ text size)
            let em_to_px = style.size;
            style.offset = [offset[0] * em_to_px, offset[1] * em_to_px];
        }
    }

    style
}

/// Convert a full StyleLayer to VectorStyle.
pub fn layer_to_vector_style(layer: &StyleLayer) -> VectorStyle {
    paint_to_vector_style(&layer.paint)
}

/// Convert a full StyleLayer to LabelStyle (for symbol layers).
pub fn layer_to_label_style(layer: &StyleLayer) -> LabelStyle {
    layout_to_label_style(&layer.layout, &layer.paint)
}

/// Extract the text property name from a symbol layer.
pub fn layer_text_property(layer: &StyleLayer) -> Option<&str> {
    layer.layout.text_field.as_ref()?.as_property_name()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::types::{ColorValue, NumberValue};

    #[test]
    fn test_paint_to_vector_style_fill() {
        let paint = PaintProps {
            fill_color: Some(ColorValue::String("#ff0000".to_string())),
            fill_opacity: Some(0.5),
            ..Default::default()
        };

        let style = paint_to_vector_style(&paint);
        assert!((style.fill_color[0] - 1.0).abs() < 0.01);
        assert!((style.fill_color[3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_paint_to_vector_style_line() {
        let paint = PaintProps {
            line_color: Some(ColorValue::String("#00ff00".to_string())),
            line_width: Some(NumberValue::Number(3.0)),
            ..Default::default()
        };

        let style = paint_to_vector_style(&paint);
        assert!((style.stroke_color[1] - 1.0).abs() < 0.01);
        assert!((style.stroke_width - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_layout_to_label_style() {
        let layout = LayoutProps {
            text_size: Some(NumberValue::Number(16.0)),
            ..Default::default()
        };
        let paint = PaintProps {
            text_color: Some(ColorValue::String("#333333".to_string())),
            text_halo_color: Some(ColorValue::String("#ffffff".to_string())),
            text_halo_width: Some(2.0),
            ..Default::default()
        };

        let style = layout_to_label_style(&layout, &paint);
        assert!((style.size - 16.0).abs() < 0.01);
        assert!((style.halo_width - 2.0).abs() < 0.01);
        assert!((style.halo_color[0] - 1.0).abs() < 0.01);
    }
}
