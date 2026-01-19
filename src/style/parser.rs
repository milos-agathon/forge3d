//! Mapbox Style Spec JSON parser.

use std::fs;
use std::path::Path;

use crate::style::types::{LayerType, StyleLayer, StyleSpec};

/// Error type for style parsing.
#[derive(Debug, thiserror::Error)]
pub enum StyleError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid style: {0}")]
    Invalid(String),
}

/// Parse a Mapbox GL Style Spec JSON file.
pub fn parse_style(path: &Path) -> Result<StyleSpec, StyleError> {
    let content = fs::read_to_string(path)?;
    parse_style_str(&content)
}

/// Parse a Mapbox GL Style Spec from a JSON string.
pub fn parse_style_str(json: &str) -> Result<StyleSpec, StyleError> {
    let spec: StyleSpec = serde_json::from_str(json)?;
    validate_style(&spec)?;
    Ok(spec)
}

/// Validate a parsed style specification.
fn validate_style(spec: &StyleSpec) -> Result<(), StyleError> {
    if spec.version != 8 {
        return Err(StyleError::Invalid(format!(
            "Unsupported style version: {} (expected 8)",
            spec.version
        )));
    }
    Ok(())
}

impl StyleSpec {
    /// Get all layers of a specific type.
    pub fn layers_by_type(&self, layer_type: LayerType) -> Vec<&StyleLayer> {
        self.layers
            .iter()
            .filter(|l| l.layer_type == layer_type)
            .collect()
    }

    /// Get all fill layers.
    pub fn fill_layers(&self) -> Vec<&StyleLayer> {
        self.layers_by_type(LayerType::Fill)
    }

    /// Get all line layers.
    pub fn line_layers(&self) -> Vec<&StyleLayer> {
        self.layers_by_type(LayerType::Line)
    }

    /// Get all symbol (text/icon) layers.
    pub fn symbol_layers(&self) -> Vec<&StyleLayer> {
        self.layers_by_type(LayerType::Symbol)
    }

    /// Find a layer by ID.
    pub fn layer_by_id(&self, id: &str) -> Option<&StyleLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Get layers for a specific source-layer name.
    pub fn layers_for_source_layer(&self, source_layer: &str) -> Vec<&StyleLayer> {
        self.layers
            .iter()
            .filter(|l| l.source_layer.as_deref() == Some(source_layer))
            .collect()
    }

    /// Count of layers by type.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl StyleLayer {
    /// Check if layer is visible (default true if not specified).
    pub fn is_visible(&self) -> bool {
        self.layout
            .visibility
            .as_ref()
            .map(|v| v != "none")
            .unwrap_or(true)
    }

    /// Check if layer passes zoom range filter.
    pub fn in_zoom_range(&self, zoom: f32) -> bool {
        let min_ok = self.minzoom.map(|z| zoom >= z).unwrap_or(true);
        let max_ok = self.maxzoom.map(|z| zoom <= z).unwrap_or(true);
        min_ok && max_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_style_json() -> &'static str {
        r##"{"version":8,"name":"Test Style","sources":{},"layers":[{"id":"background","type":"background","paint":{"background-color":"#f0f0f0"}},{"id":"water","type":"fill","source":"composite","source-layer":"water","paint":{"fill-color":"#0066ff","fill-opacity":0.8}},{"id":"roads","type":"line","source":"composite","source-layer":"road","paint":{"line-color":"#ffffff","line-width":2},"filter":["==","class","motorway"]},{"id":"labels","type":"symbol","source":"composite","source-layer":"place_label","layout":{"text-field":"{name}","text-size":14},"paint":{"text-color":"#333333","text-halo-color":"#ffffff","text-halo-width":1.5}},{"id":"hidden-layer","type":"fill","source":"composite","source-layer":"landuse","layout":{"visibility":"none"}}]}"##
    }

    #[test]
    fn test_parse_minimal_style() {
        let spec = parse_style_str(minimal_style_json()).unwrap();
        assert_eq!(spec.version, 8);
        assert_eq!(spec.name, "Test Style");
        assert_eq!(spec.layers.len(), 5);
    }

    #[test]
    fn test_layers_by_type() {
        let spec = parse_style_str(minimal_style_json()).unwrap();
        assert_eq!(spec.fill_layers().len(), 2);
        assert_eq!(spec.line_layers().len(), 1);
        assert_eq!(spec.symbol_layers().len(), 1);
    }

    #[test]
    fn test_layer_visibility() {
        let spec = parse_style_str(minimal_style_json()).unwrap();
        let water = spec.layer_by_id("water").unwrap();
        let hidden = spec.layer_by_id("hidden-layer").unwrap();
        assert!(water.is_visible());
        assert!(!hidden.is_visible());
    }

    #[test]
    fn test_fill_paint_props() {
        let spec = parse_style_str(minimal_style_json()).unwrap();
        let water = spec.layer_by_id("water").unwrap();

        let color = water.paint.fill_color.as_ref().unwrap().to_rgba().unwrap();
        assert!((color[0] - 0.0).abs() < 0.01);
        assert!((color[1] - 0.4).abs() < 0.01);
        assert!((color[2] - 1.0).abs() < 0.01);

        let opacity = water.paint.fill_opacity.unwrap();
        assert!((opacity - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_line_paint_props() {
        let spec = parse_style_str(minimal_style_json()).unwrap();
        let roads = spec.layer_by_id("roads").unwrap();

        let color = roads.paint.line_color.as_ref().unwrap().to_rgba().unwrap();
        assert!((color[0] - 1.0).abs() < 0.01);

        let width = roads.paint.line_width.as_ref().unwrap().as_f32().unwrap();
        assert!((width - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_symbol_layout_props() {
        let spec = parse_style_str(minimal_style_json()).unwrap();
        let labels = spec.layer_by_id("labels").unwrap();

        let prop_name = labels.layout.text_field.as_ref().unwrap().as_property_name();
        assert_eq!(prop_name, Some("name"));

        let size = labels.layout.text_size.as_ref().unwrap().as_f32().unwrap();
        assert!((size - 14.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_exists() {
        let spec = parse_style_str(minimal_style_json()).unwrap();
        let roads = spec.layer_by_id("roads").unwrap();
        assert!(roads.filter.is_some());
    }

    #[test]
    fn test_invalid_version() {
        let json = r#"{"version": 7, "layers": []}"#;
        let result = parse_style_str(json);
        assert!(result.is_err());
    }
}
