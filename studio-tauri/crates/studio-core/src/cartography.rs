//! The cartography: the map's own layer stack.
//!
//! A map is a stack of layers, drawn back to front. Each layer carries its
//! own style, its own name and its own eye, and the order is the author's:
//! the terrain's relief at the bottom, the elevation ramp and the draped
//! imagery over it, the vector layers and their labels above that, and the
//! plate around the map at the very front.
//!
//! The stack replaced three separate homes for the same decisions — the
//! terrain object's draped raster and vector layer, and the render settings'
//! colormap and plate. One stack means one order, one place to look, and one
//! place to change.
//!
//! Two kinds share the engine's single terrain-overlay slot: the elevation
//! ramp and the draped imagery. Only one can reach the terrain shader, so the
//! renderer takes the uppermost visible one and reports the other as not
//! composited rather than dropping it silently.

use crate::ids::{new_id, AssetId};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A layer's identity. Stable across save, load and undo.
pub type CartographyLayerId = Uuid;

/// One layer of the map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CartographyLayer {
    pub id: CartographyLayerId,
    /// What the layer is called in the stack.
    pub name: String,
    /// The eye: a hidden layer is not drawn.
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(flatten)]
    pub kind: CartographyLayerKind,
}

/// What a layer draws.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum CartographyLayerKind {
    /// The terrain's own relief: the DEM, shaded. The base of the stack.
    Relief,
    /// The elevation ramp painted over the relief, by the engine's colormap.
    ElevationRamp {
        /// The engine's colormap name (`viridis`, `terrain`, ...).
        colormap: String,
        /// How strongly the ramp covers the material, 0..=1.
        strength: f32,
        /// Draw the engine's elevation legend in the frame's corner.
        legend: bool,
    },
    /// A raster draped on the terrain (land cover, an orthophoto).
    Imagery {
        asset: AssetId,
        opacity: f32,
        /// Composite source colours after lighting (categorical palettes survive).
        preserve_colors: bool,
        /// Clean a single-band classification before it is paletted.
        despeckle: bool,
    },
    /// A vector layer: lines draped on the terrain.
    Vector {
        asset: AssetId,
        /// The line colour, RGBA in 0..=1.
        colour: [f32; 4],
        /// The line's symbol width in pixels.
        width_px: f32,
    },
    /// A vector layer: polygons filled on the terrain, holes and all, drawn
    /// by the engine's own polygon rasterizer.
    Polygon {
        asset: AssetId,
        /// The fill colour, RGBA in 0..=1.
        colour: [f32; 4],
    },
    /// The labels of a vector layer: its named features, drawn by the engine's
    /// own label plan and text compositor.
    Labels {
        /// The vector layer whose features are named.
        source: CartographyLayerId,
        /// The attribute that names a feature (`name`, `name.en`, ...).
        field: String,
        /// The label colour, RGBA in 0..=1.
        colour: [f32; 4],
        /// The label size in pixels.
        size_px: f32,
    },
    /// The graticule: the grid of meridians and parallels, with its
    /// coordinate labels. The engine generates the grid and places the labels;
    /// the layer only says how often the grid repeats and how it is drawn.
    Graticule {
        /// The grid's interval in degrees.
        interval_deg: f32,
        /// The line colour, RGBA in 0..=1.
        colour: [f32; 4],
        /// The line's symbol width in pixels.
        width_px: f32,
        /// Draw the coordinate labels along the grid.
        labels: bool,
    },
    /// The map plate: the page the map is cut out on, with its furniture.
    Plate {
        title: String,
        /// The line along the bottom of the page.
        caption: String,
        /// The north arrow (turned to the map's orientation).
        north_arrow: bool,
        /// The scale bar (only where the ground scale is known).
        scale_bar: bool,
        /// The legends of the layers below.
        legend: bool,
    },
}

fn default_true() -> bool {
    true
}

impl CartographyLayerKind {
    /// The name a new layer of this kind gets.
    pub fn default_name(&self) -> String {
        match self {
            CartographyLayerKind::Relief => "Terrain relief".to_owned(),
            CartographyLayerKind::ElevationRamp { .. } => "Elevation ramp".to_owned(),
            CartographyLayerKind::Imagery { .. } => "Imagery".to_owned(),
            CartographyLayerKind::Vector { .. } => "Vector layer".to_owned(),
            CartographyLayerKind::Polygon { .. } => "Polygon layer".to_owned(),
            CartographyLayerKind::Labels { .. } => "Labels".to_owned(),
            CartographyLayerKind::Graticule { .. } => "Graticule".to_owned(),
            CartographyLayerKind::Plate { .. } => "Map plate".to_owned(),
        }
    }

    /// The asset this layer draws, when it draws a file.
    pub fn asset(&self) -> Option<AssetId> {
        match self {
            CartographyLayerKind::Imagery { asset, .. }
            | CartographyLayerKind::Vector { asset, .. }
            | CartographyLayerKind::Polygon { asset, .. } => Some(*asset),
            _ => None,
        }
    }

    /// Whether the layer's values are inside the ranges the engine takes.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            CartographyLayerKind::Relief => Ok(()),
            CartographyLayerKind::ElevationRamp {
                colormap, strength, ..
            } => {
                if colormap.trim().is_empty() {
                    return Err("the elevation ramp needs a colormap".to_owned());
                }
                if !strength.is_finite() || !(0.0..=1.0).contains(strength) {
                    return Err("the elevation ramp's strength must be between 0 and 1".to_owned());
                }
                Ok(())
            }
            CartographyLayerKind::Imagery { asset, opacity, .. } => {
                if asset.is_nil() {
                    return Err("the imagery layer has no asset".to_owned());
                }
                if !opacity.is_finite() || !(0.0..=1.0).contains(opacity) {
                    return Err("the imagery's opacity must be between 0 and 1".to_owned());
                }
                Ok(())
            }
            CartographyLayerKind::Polygon { colour, .. } => {
                if !colour
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                {
                    return Err("the polygon colour channels must be between 0 and 1".to_owned());
                }
                Ok(())
            }
            CartographyLayerKind::Vector {
                colour, width_px, ..
            } => {
                if !colour
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                {
                    return Err("the vector colour channels must be between 0 and 1".to_owned());
                }
                if !width_px.is_finite() || !(1.0..=20.0).contains(width_px) {
                    return Err("the vector's width must be between 1 and 20 pixels".to_owned());
                }
                Ok(())
            }
            CartographyLayerKind::Labels {
                source,
                field,
                colour,
                size_px,
            } => {
                if source.is_nil() {
                    return Err("the label layer has no source layer".to_owned());
                }
                if field.trim().is_empty() {
                    return Err(
                        "the label layer needs the attribute that names a feature".to_owned()
                    );
                }
                if !colour
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                {
                    return Err("the label colour channels must be between 0 and 1".to_owned());
                }
                if !size_px.is_finite() || !(8.0..=48.0).contains(size_px) {
                    return Err("the label's size must be between 8 and 48 pixels".to_owned());
                }
                Ok(())
            }
            CartographyLayerKind::Graticule {
                interval_deg,
                colour,
                width_px,
                ..
            } => {
                // The engine takes any positive interval; the studio's range
                // is inside it (a tenth of a degree to half a hemisphere).
                if !interval_deg.is_finite() || !(0.05..=30.0).contains(interval_deg) {
                    return Err(
                        "the graticule's interval must be between 0.05 and 30 degrees".to_owned(),
                    );
                }
                if !colour
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                {
                    return Err(
                        "the graticule's colour channels must be between 0 and 1".to_owned()
                    );
                }
                if !width_px.is_finite() || !(1.0..=20.0).contains(width_px) {
                    return Err("the graticule's width must be between 1 and 20 pixels".to_owned());
                }
                Ok(())
            }
            CartographyLayerKind::Plate { .. } => Ok(()),
        }
    }
}

impl CartographyLayer {
    /// A layer of the given kind, with the studio's defaults.
    pub fn new(kind: CartographyLayerKind) -> Self {
        let name = kind.default_name();
        Self {
            id: new_id(),
            name,
            visible: true,
            kind,
        }
    }

    /// The layer's own validation.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("a layer needs a name".to_owned());
        }
        self.kind.validate()
    }
}

/// The stack: the map's layers, in draw order (the first is drawn first, so
/// it is at the bottom of the panel).
pub type CartographyStack = Vec<CartographyLayer>;

/// The stack a fresh project starts with: the relief, and nothing else.
pub fn default_stack() -> CartographyStack {
    vec![CartographyLayer::new(CartographyLayerKind::Relief)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_map_is_its_relief() {
        let stack = default_stack();
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].kind, CartographyLayerKind::Relief);
        assert!(stack[0].visible);
    }

    #[test]
    fn the_layers_round_trip_through_json() {
        let layer = CartographyLayer::new(CartographyLayerKind::Vector {
            asset: new_id(),
            colour: [0.9, 0.29, 0.1, 1.0],
            width_px: 3.0,
        });
        let text = serde_json::to_string(&layer).expect("serialize");
        let back: CartographyLayer = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(layer, back);
        assert!(text.contains("\"type\":\"vector\""), "{text}");
    }

    #[test]
    fn the_polygon_layer_round_trips_through_json() {
        let layer = CartographyLayer::new(CartographyLayerKind::Polygon {
            asset: new_id(),
            colour: [0.2, 0.5, 0.7, 0.6],
        });
        let text = serde_json::to_string(&layer).expect("serialize");
        let back: CartographyLayer = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(layer, back);
        assert!(text.contains("\"type\":\"polygon\""), "{text}");
        assert!(layer.validate().is_ok());
    }

    #[test]
    fn the_graticule_keeps_the_engine_own_ranges() {
        let grid = |interval_deg: f32, width_px: f32| CartographyLayerKind::Graticule {
            interval_deg,
            colour: [0.1, 0.1, 0.12, 0.75],
            width_px,
            labels: true,
        };
        assert!(grid(1.0, 1.0).validate().is_ok());
        // The engine takes any positive interval; the studio's range is
        // inside it, and it refuses what the engine would refuse.
        assert!(grid(0.0, 1.0).validate().is_err());
        assert!(grid(45.0, 1.0).validate().is_err());
        assert!(grid(1.0, 0.0).validate().is_err());
    }

    #[test]
    fn the_graticule_round_trips_through_json() {
        let layer = CartographyLayer::new(CartographyLayerKind::Graticule {
            interval_deg: 0.5,
            colour: [0.1, 0.1, 0.12, 0.75],
            width_px: 1.0,
            labels: true,
        });
        let text = serde_json::to_string(&layer).expect("serialize");
        let back: CartographyLayer = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(layer, back);
        assert!(text.contains("\"type\":\"graticule\""), "{text}");
        assert!(text.contains("\"intervalDeg\":0.5"), "{text}");
    }

    #[test]
    fn the_ranges_the_engine_takes_are_enforced() {
        let labels = CartographyLayerKind::Labels {
            source: new_id(),
            field: "name".to_owned(),
            colour: [0.0, 0.0, 0.0, 1.0],
            size_px: 14.0,
        };
        assert!(labels.validate().is_ok());
        let (source, field, colour) = match &labels {
            CartographyLayerKind::Labels {
                source,
                field,
                colour,
                ..
            } => (*source, field.clone(), *colour),
            _ => unreachable!("the layer is a label layer"),
        };
        assert!(CartographyLayerKind::Labels {
            source,
            field,
            colour,
            size_px: 4.0,
        }
        .validate()
        .is_err());
        assert!(CartographyLayerKind::Vector {
            asset: new_id(),
            colour: [1.4, 0.0, 0.0, 1.0],
            width_px: 3.0,
        }
        .validate()
        .is_err());
    }
}
