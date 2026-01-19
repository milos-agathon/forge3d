//! Style spec types for Mapbox GL Style Spec.

use serde::{Deserialize, Serialize};

/// Complete style specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleSpec {
    /// Style version (always 8 for Mapbox GL).
    #[serde(default = "default_version")]
    pub version: u32,
    /// Style name.
    #[serde(default)]
    pub name: String,
    /// Style layers.
    #[serde(default)]
    pub layers: Vec<StyleLayer>,
    /// Data sources (informational, not parsed in detail).
    #[serde(default)]
    pub sources: serde_json::Value,
    /// Sprite URL (optional).
    #[serde(default)]
    pub sprite: Option<String>,
    /// Glyphs URL template (optional).
    #[serde(default)]
    pub glyphs: Option<String>,
}

fn default_version() -> u32 {
    8
}

/// A single style layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleLayer {
    /// Unique layer ID.
    pub id: String,
    /// Layer type (fill, line, symbol, background).
    #[serde(rename = "type")]
    pub layer_type: LayerType,
    /// Source ID (for data layers).
    #[serde(default)]
    pub source: Option<String>,
    /// Source layer name (for vector tile sources).
    #[serde(rename = "source-layer")]
    #[serde(default)]
    pub source_layer: Option<String>,
    /// Paint properties (colors, widths, opacities).
    #[serde(default)]
    pub paint: PaintProps,
    /// Layout properties (visibility, text settings).
    #[serde(default)]
    pub layout: LayoutProps,
    /// Filter expression (optional).
    #[serde(default)]
    pub filter: Option<FilterExpr>,
    /// Minimum zoom level.
    #[serde(default)]
    pub minzoom: Option<f32>,
    /// Maximum zoom level.
    #[serde(default)]
    pub maxzoom: Option<f32>,
}

/// Layer types supported by Mapbox GL Style Spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayerType {
    Fill,
    Line,
    Symbol,
    Background,
    Circle,
    Raster,
    Hillshade,
    #[serde(other)]
    Unknown,
}

/// Paint properties for styling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PaintProps {
    // Fill properties
    #[serde(rename = "fill-color")]
    pub fill_color: Option<ColorValue>,
    #[serde(rename = "fill-opacity")]
    pub fill_opacity: Option<f32>,
    #[serde(rename = "fill-outline-color")]
    pub fill_outline_color: Option<ColorValue>,

    // Line properties
    #[serde(rename = "line-color")]
    pub line_color: Option<ColorValue>,
    #[serde(rename = "line-width")]
    pub line_width: Option<NumberValue>,
    #[serde(rename = "line-opacity")]
    pub line_opacity: Option<f32>,

    // Symbol/text properties
    #[serde(rename = "text-color")]
    pub text_color: Option<ColorValue>,
    #[serde(rename = "text-halo-color")]
    pub text_halo_color: Option<ColorValue>,
    #[serde(rename = "text-halo-width")]
    pub text_halo_width: Option<f32>,
    #[serde(rename = "text-opacity")]
    pub text_opacity: Option<f32>,

    // Circle properties
    #[serde(rename = "circle-color")]
    pub circle_color: Option<ColorValue>,
    #[serde(rename = "circle-radius")]
    pub circle_radius: Option<f32>,
    #[serde(rename = "circle-opacity")]
    pub circle_opacity: Option<f32>,

    // Background properties
    #[serde(rename = "background-color")]
    pub background_color: Option<ColorValue>,
}

/// Layout properties for text and visibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LayoutProps {
    /// Visibility ("visible" or "none").
    pub visibility: Option<String>,
    /// Text field expression (e.g., "{name}" or ["get", "name"]).
    #[serde(rename = "text-field")]
    pub text_field: Option<TextFieldValue>,
    /// Text size in pixels.
    #[serde(rename = "text-size")]
    pub text_size: Option<NumberValue>,
    /// Text font stack.
    #[serde(rename = "text-font")]
    pub text_font: Option<Vec<String>>,
    /// Text anchor position.
    #[serde(rename = "text-anchor")]
    pub text_anchor: Option<String>,
    /// Text offset from anchor.
    #[serde(rename = "text-offset")]
    pub text_offset: Option<Vec<f32>>,
    /// Text max width before wrapping.
    #[serde(rename = "text-max-width")]
    pub text_max_width: Option<f32>,
    /// Line cap style.
    #[serde(rename = "line-cap")]
    pub line_cap: Option<String>,
    /// Line join style.
    #[serde(rename = "line-join")]
    pub line_join: Option<String>,
}

/// Color value - can be a simple string or an expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ColorValue {
    /// Simple color string (e.g., "#ff0000", "rgb(255,0,0)", "red").
    String(String),
    /// Expression (e.g., ["interpolate", ...]).
    Expression(serde_json::Value),
}

impl ColorValue {
    /// Parse color to RGBA [0..1].
    pub fn to_rgba(&self) -> Option<[f32; 4]> {
        match self {
            ColorValue::String(s) => parse_color_string(s),
            ColorValue::Expression(_) => None,
        }
    }

    /// Evaluate color with expression context.
    pub fn evaluate(&self, ctx: &crate::style::expressions::EvalContext) -> Option<[f32; 4]> {
        match self {
            ColorValue::String(s) => parse_color_string(s),
            ColorValue::Expression(expr) => crate::style::expressions::evaluate_color(expr, ctx),
        }
    }
}

/// Number value - can be a literal or an expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NumberValue {
    /// Literal number.
    Number(f32),
    /// Expression (e.g., ["interpolate", ...]).
    Expression(serde_json::Value),
}

impl NumberValue {
    /// Get literal value or None for expressions.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            NumberValue::Number(n) => Some(*n),
            NumberValue::Expression(_) => None,
        }
    }

    /// Evaluate number with expression context.
    pub fn evaluate(&self, ctx: &crate::style::expressions::EvalContext) -> Option<f32> {
        match self {
            NumberValue::Number(n) => Some(*n),
            NumberValue::Expression(expr) => {
                crate::style::expressions::evaluate_number(expr, ctx).map(|v| v as f32)
            }
        }
    }
}

/// Text field value - can be a string template or expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextFieldValue {
    /// Template string (e.g., "{name}").
    String(String),
    /// Expression (e.g., ["get", "name"]).
    Expression(serde_json::Value),
}

impl TextFieldValue {
    /// Extract the property name if this is a simple field reference.
    pub fn as_property_name(&self) -> Option<&str> {
        match self {
            TextFieldValue::String(s) => {
                // Parse "{property}" format
                if s.starts_with('{') && s.ends_with('}') {
                    Some(&s[1..s.len() - 1])
                } else {
                    None
                }
            }
            TextFieldValue::Expression(expr) => {
                // Parse ["get", "property"] format
                if let Some(arr) = expr.as_array() {
                    if arr.len() == 2 {
                        if let (Some("get"), Some(prop)) =
                            (arr[0].as_str(), arr[1].as_str())
                        {
                            return Some(prop);
                        }
                    }
                }
                None
            }
        }
    }
}

/// Filter expression (subset of Mapbox filter syntax).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterExpr {
    /// Array-based expression.
    Array(Vec<serde_json::Value>),
    /// Boolean literal.
    Bool(bool),
}

impl FilterExpr {
    /// Evaluate filter against a feature's properties.
    pub fn evaluate(&self, properties: &serde_json::Map<String, serde_json::Value>) -> bool {
        match self {
            FilterExpr::Bool(b) => *b,
            FilterExpr::Array(arr) => evaluate_filter_array(arr, properties),
        }
    }
}

fn evaluate_filter_array(
    arr: &[serde_json::Value],
    props: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    if arr.is_empty() {
        return true;
    }

    let op = match arr[0].as_str() {
        Some(s) => s,
        None => return true,
    };

    match op {
        "==" | "eq" => {
            if arr.len() != 3 {
                return true;
            }
            let key = arr[1].as_str().unwrap_or("");
            let expected = &arr[2];
            props.get(key).map(|v| v == expected).unwrap_or(false)
        }
        "!=" | "neq" => {
            if arr.len() != 3 {
                return true;
            }
            let key = arr[1].as_str().unwrap_or("");
            let expected = &arr[2];
            props.get(key).map(|v| v != expected).unwrap_or(true)
        }
        "all" => arr[1..].iter().all(|sub| {
            if let Some(sub_arr) = sub.as_array() {
                evaluate_filter_array(sub_arr, props)
            } else {
                true
            }
        }),
        "any" => arr[1..].iter().any(|sub| {
            if let Some(sub_arr) = sub.as_array() {
                evaluate_filter_array(sub_arr, props)
            } else {
                false
            }
        }),
        "has" => {
            if arr.len() != 2 {
                return true;
            }
            let key = arr[1].as_str().unwrap_or("");
            props.contains_key(key)
        }
        "!" | "not" => {
            if arr.len() != 2 {
                return true;
            }
            if let Some(sub_arr) = arr[1].as_array() {
                !evaluate_filter_array(sub_arr, props)
            } else {
                true
            }
        }
        "in" => {
            if arr.len() < 3 {
                return true;
            }
            let key = arr[1].as_str().unwrap_or("");
            if let Some(val) = props.get(key) {
                arr[2..].iter().any(|v| v == val)
            } else {
                false
            }
        }
        _ => true, // Unknown operators pass through
    }
}

/// Parse a CSS color string to RGBA.
pub fn parse_color_string(s: &str) -> Option<[f32; 4]> {
    let s = s.trim();

    // Hex colors
    if s.starts_with('#') {
        return parse_hex_color(s);
    }

    // RGB/RGBA functions
    if s.starts_with("rgb") {
        return parse_rgb_color(s);
    }

    // HSL/HSLA functions
    if s.starts_with("hsl") {
        return parse_hsl_color(s);
    }

    // Named colors (subset)
    match s.to_lowercase().as_str() {
        "black" => Some([0.0, 0.0, 0.0, 1.0]),
        "white" => Some([1.0, 1.0, 1.0, 1.0]),
        "red" => Some([1.0, 0.0, 0.0, 1.0]),
        "green" => Some([0.0, 0.5, 0.0, 1.0]),
        "blue" => Some([0.0, 0.0, 1.0, 1.0]),
        "yellow" => Some([1.0, 1.0, 0.0, 1.0]),
        "cyan" => Some([0.0, 1.0, 1.0, 1.0]),
        "magenta" => Some([1.0, 0.0, 1.0, 1.0]),
        "gray" | "grey" => Some([0.5, 0.5, 0.5, 1.0]),
        "orange" => Some([1.0, 0.647, 0.0, 1.0]),
        "transparent" => Some([0.0, 0.0, 0.0, 0.0]),
        _ => None,
    }
}

fn parse_hex_color(s: &str) -> Option<[f32; 4]> {
    let hex = s.trim_start_matches('#');
    match hex.len() {
        3 => {
            // #RGB -> #RRGGBB
            let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
            let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
            let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
            Some([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0])
        }
        4 => {
            // #RGBA -> #RRGGBBAA
            let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
            let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
            let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
            let a = u8::from_str_radix(&hex[3..4].repeat(2), 16).ok()?;
            Some([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ])
        }
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0])
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
            Some([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ])
        }
        _ => None,
    }
}

fn parse_rgb_color(s: &str) -> Option<[f32; 4]> {
    let inner = s
        .trim_start_matches("rgba(")
        .trim_start_matches("rgb(")
        .trim_end_matches(')');
    let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();

    if parts.len() < 3 {
        return None;
    }

    let r: f32 = parts[0].trim_end_matches('%').parse().ok()?;
    let g: f32 = parts[1].trim_end_matches('%').parse().ok()?;
    let b: f32 = parts[2].trim_end_matches('%').parse().ok()?;

    let (r, g, b) = if parts[0].contains('%') {
        (r / 100.0, g / 100.0, b / 100.0)
    } else {
        (r / 255.0, g / 255.0, b / 255.0)
    };

    let a = if parts.len() >= 4 {
        parts[3].parse().unwrap_or(1.0)
    } else {
        1.0
    };

    Some([r, g, b, a])
}

fn parse_hsl_color(s: &str) -> Option<[f32; 4]> {
    let inner = s
        .trim_start_matches("hsla(")
        .trim_start_matches("hsl(")
        .trim_end_matches(')');
    let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();

    if parts.len() < 3 {
        return None;
    }

    let h: f32 = parts[0].parse().ok()?;
    let s_val: f32 = parts[1].trim_end_matches('%').parse::<f32>().ok()? / 100.0;
    let l: f32 = parts[2].trim_end_matches('%').parse::<f32>().ok()? / 100.0;

    let a = if parts.len() >= 4 {
        parts[3].parse().unwrap_or(1.0)
    } else {
        1.0
    };

    // HSL to RGB conversion
    let (r, g, b) = hsl_to_rgb(h / 360.0, s_val, l);
    Some([r, g, b, a])
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_colors() {
        assert_eq!(parse_hex_color("#fff"), Some([1.0, 1.0, 1.0, 1.0]));
        assert_eq!(parse_hex_color("#000"), Some([0.0, 0.0, 0.0, 1.0]));
        assert_eq!(parse_hex_color("#ff0000"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(parse_hex_color("#00ff00ff"), Some([0.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn test_parse_rgb_colors() {
        let rgba = parse_rgb_color("rgb(255, 0, 0)").unwrap();
        assert!((rgba[0] - 1.0).abs() < 0.01);
        assert!((rgba[1] - 0.0).abs() < 0.01);
        assert!((rgba[2] - 0.0).abs() < 0.01);

        let rgba = parse_rgb_color("rgba(0, 255, 0, 0.5)").unwrap();
        assert!((rgba[1] - 1.0).abs() < 0.01);
        assert!((rgba[3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_filter_evaluation() {
        let mut props = serde_json::Map::new();
        props.insert("class".to_string(), serde_json::json!("road"));
        props.insert("level".to_string(), serde_json::json!(1));

        // Simple equality
        let filter = FilterExpr::Array(vec![
            serde_json::json!("=="),
            serde_json::json!("class"),
            serde_json::json!("road"),
        ]);
        assert!(filter.evaluate(&props));

        // All combinator
        let filter = FilterExpr::Array(vec![
            serde_json::json!("all"),
            serde_json::json!(["==", "class", "road"]),
            serde_json::json!(["==", "level", 1]),
        ]);
        assert!(filter.evaluate(&props));

        // Any combinator
        let filter = FilterExpr::Array(vec![
            serde_json::json!("any"),
            serde_json::json!(["==", "class", "highway"]),
            serde_json::json!(["==", "class", "road"]),
        ]);
        assert!(filter.evaluate(&props));
    }
}
