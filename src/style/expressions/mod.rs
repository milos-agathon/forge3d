//! Mapbox Style Spec expression evaluation.
//!
//! Implements evaluation of data-driven expressions including:
//! - `interpolate`: Linear/exponential interpolation between stops
//! - `step`: Stepped/discrete values at breakpoints
//! - `match`: Pattern matching on property values
//! - `get`: Property value lookup
//! - `coalesce`: First non-null value
//! - Math operators: `+`, `-`, `*`, `/`, `%`, `^`
//! - Comparison: `<`, `<=`, `>`, `>=`
//! - Logic: `all`, `any`, `!`, `case`

use serde_json::Value;

mod comparison;
mod control;
mod dispatch;
mod logic;
mod math;
mod property;
mod strings;

#[cfg(test)]
mod tests;

/// Expression evaluation context containing feature properties and zoom level.
#[derive(Debug, Clone)]
pub struct EvalContext<'a> {
    /// Feature properties map.
    pub properties: &'a serde_json::Map<String, Value>,
    /// Current zoom level.
    pub zoom: f64,
    /// Geometry type (optional).
    pub geometry_type: Option<&'a str>,
}

impl<'a> EvalContext<'a> {
    pub fn new(properties: &'a serde_json::Map<String, Value>, zoom: f64) -> Self {
        Self {
            properties,
            zoom,
            geometry_type: None,
        }
    }

    pub fn with_geometry_type(mut self, geom_type: &'a str) -> Self {
        self.geometry_type = Some(geom_type);
        self
    }
}

pub use dispatch::{evaluate_color, evaluate_expression, evaluate_number};
