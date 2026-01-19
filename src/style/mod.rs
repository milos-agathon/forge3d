//! Mapbox Style Spec import module.
//!
//! This module provides parsing and conversion of Mapbox GL Style Spec JSON
//! into forge3d's native vector and label styles.
//!
//! Supported layer types (v1):
//! - `fill`: Polygon fill with color, opacity, outline
//! - `line`: Polyline with color, width, opacity
//! - `symbol`: Text labels with size, color, halo
//! - `background`: Background color (informational only)
//!
//! See <https://docs.mapbox.com/mapbox-gl-js/style-spec/> for full spec.

pub mod converters;
pub mod expressions;
pub mod parser;
pub mod sprite;
pub mod types;

pub use converters::{layout_to_label_style, paint_to_vector_style};
pub use expressions::{evaluate_expression, evaluate_color, evaluate_number, EvalContext};
pub use parser::parse_style;
pub use sprite::{SpriteAtlas, SpriteEntry, load_sprite_atlas};
pub use types::{
    FilterExpr, LayerType, LayoutProps, PaintProps, StyleLayer, StyleSpec,
};
