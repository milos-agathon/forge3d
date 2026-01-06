//! Label layers for batch labeling from feature collections.
//!
//! Provides a high-level API for labeling geographic features
//! with automatic style functions and placement strategies.

use crate::labels::types::{LabelId, LabelStyle};
use crate::labels::declutter::{DeclutterAlgorithm, DeclutterConfig};
use crate::labels::typography::TypographySettings;
use glam::Vec3;

/// Feature type for labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    /// Point feature (city, peak, etc.).
    Point,
    /// Line feature (road, river, etc.).
    Line,
    /// Polygon feature (lake, country, etc.).
    Polygon,
}

/// A geographic feature to be labeled.
#[derive(Debug, Clone)]
pub struct LabelFeature {
    /// Unique feature identifier.
    pub id: u64,
    /// Feature type.
    pub feature_type: FeatureType,
    /// Feature geometry (point, or vertices for line/polygon).
    pub geometry: FeatureGeometry,
    /// Properties/attributes for style functions.
    pub properties: std::collections::HashMap<String, String>,
}

/// Feature geometry variants.
#[derive(Debug, Clone)]
pub enum FeatureGeometry {
    /// Single point.
    Point(Vec3),
    /// Polyline vertices.
    Line(Vec<Vec3>),
    /// Polygon vertices (exterior ring).
    Polygon(Vec<Vec3>),
}

impl FeatureGeometry {
    /// Get centroid of the geometry.
    pub fn centroid(&self) -> Vec3 {
        match self {
            FeatureGeometry::Point(p) => *p,
            FeatureGeometry::Line(pts) => {
                if pts.is_empty() {
                    Vec3::ZERO
                } else {
                    let sum: Vec3 = pts.iter().copied().sum();
                    sum / pts.len() as f32
                }
            }
            FeatureGeometry::Polygon(pts) => {
                if pts.is_empty() {
                    Vec3::ZERO
                } else {
                    let sum: Vec3 = pts.iter().copied().sum();
                    sum / pts.len() as f32
                }
            }
        }
    }
}

/// Placement strategy for labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlacementStrategy {
    /// Place at feature centroid.
    #[default]
    Centroid,
    /// Place along line (for line features).
    AlongLine,
    /// Place with curved text along path.
    Curved,
    /// Automatic based on feature type.
    Auto,
}

/// Label layer configuration.
#[derive(Debug, Clone)]
pub struct LabelLayerConfig {
    /// Field name to use for label text.
    pub label_field: String,
    /// Base style for labels.
    pub base_style: LabelStyle,
    /// Typography settings.
    pub typography: TypographySettings,
    /// Placement strategy.
    pub placement: PlacementStrategy,
    /// Declutter algorithm.
    pub declutter: DeclutterAlgorithm,
    /// Declutter configuration.
    pub declutter_config: DeclutterConfig,
    /// Minimum zoom for visibility.
    pub min_zoom: f32,
    /// Maximum zoom for visibility.
    pub max_zoom: f32,
    /// Whether to allow curved text.
    pub allow_curved: bool,
    /// Repeat distance for line labels (0 = no repeat).
    pub repeat_distance: f32,
}

impl Default for LabelLayerConfig {
    fn default() -> Self {
        Self {
            label_field: "name".to_string(),
            base_style: LabelStyle::default(),
            typography: TypographySettings::default(),
            placement: PlacementStrategy::Auto,
            declutter: DeclutterAlgorithm::Greedy,
            declutter_config: DeclutterConfig::default(),
            min_zoom: 0.0,
            max_zoom: f32::MAX,
            allow_curved: true,
            repeat_distance: 0.0,
        }
    }
}

impl LabelLayerConfig {
    /// Set the label field.
    pub fn with_label_field(mut self, field: &str) -> Self {
        self.label_field = field.to_string();
        self
    }

    /// Set the base style.
    pub fn with_style(mut self, style: LabelStyle) -> Self {
        self.base_style = style;
        self
    }

    /// Set the placement strategy.
    pub fn with_placement(mut self, placement: PlacementStrategy) -> Self {
        self.placement = placement;
        self
    }

    /// Set the declutter algorithm.
    pub fn with_declutter(mut self, algorithm: DeclutterAlgorithm) -> Self {
        self.declutter = algorithm;
        self
    }

    /// Set zoom range.
    pub fn with_zoom_range(mut self, min: f32, max: f32) -> Self {
        self.min_zoom = min;
        self.max_zoom = max;
        self
    }

    /// Enable/disable curved text.
    pub fn with_curved(mut self, allow: bool) -> Self {
        self.allow_curved = allow;
        self
    }
}

/// A label generated from a feature.
#[derive(Debug, Clone)]
pub struct GeneratedLabel {
    /// Feature ID this label came from.
    pub feature_id: u64,
    /// Label text.
    pub text: String,
    /// Label position (world space).
    pub position: Vec3,
    /// Computed style.
    pub style: LabelStyle,
    /// Placement type.
    pub placement_type: LabelPlacementType,
    /// Line geometry if line label.
    pub line_geometry: Option<Vec<Vec3>>,
}

/// Type of label placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelPlacementType {
    /// Point label at centroid.
    Point,
    /// Line label (horizontal at center).
    LineCenter,
    /// Line label (following line direction).
    LineAlong,
    /// Curved text along path.
    Curved,
}

/// A label layer managing labels for a set of features.
#[derive(Debug, Clone)]
pub struct LabelLayer {
    /// Layer identifier.
    pub id: u64,
    /// Layer name.
    pub name: String,
    /// Features in this layer.
    pub features: Vec<LabelFeature>,
    /// Configuration.
    pub config: LabelLayerConfig,
    /// Generated labels.
    pub labels: Vec<GeneratedLabel>,
    /// Whether the layer is visible.
    pub visible: bool,
    /// Assigned label IDs from the label manager.
    pub label_ids: Vec<LabelId>,
}

impl LabelLayer {
    /// Create a new label layer.
    pub fn new(id: u64, name: &str, config: LabelLayerConfig) -> Self {
        Self {
            id,
            name: name.to_string(),
            features: Vec::new(),
            config,
            labels: Vec::new(),
            visible: true,
            label_ids: Vec::new(),
        }
    }

    /// Add a feature to the layer.
    pub fn add_feature(&mut self, feature: LabelFeature) {
        self.features.push(feature);
    }

    /// Add multiple features.
    pub fn add_features(&mut self, features: Vec<LabelFeature>) {
        self.features.extend(features);
    }

    /// Generate labels for all features.
    pub fn generate_labels(&mut self) {
        self.labels.clear();

        for feature in &self.features {
            // Get label text from properties
            let text = feature
                .properties
                .get(&self.config.label_field)
                .cloned()
                .unwrap_or_default();

            if text.is_empty() {
                continue;
            }

            // Determine placement type
            let placement_type = match self.config.placement {
                PlacementStrategy::Auto => match feature.feature_type {
                    FeatureType::Point => LabelPlacementType::Point,
                    FeatureType::Line => {
                        if self.config.allow_curved {
                            LabelPlacementType::Curved
                        } else {
                            LabelPlacementType::LineAlong
                        }
                    }
                    FeatureType::Polygon => LabelPlacementType::Point,
                },
                PlacementStrategy::Centroid => LabelPlacementType::Point,
                PlacementStrategy::AlongLine => LabelPlacementType::LineAlong,
                PlacementStrategy::Curved => LabelPlacementType::Curved,
            };

            // Get position
            let position = feature.geometry.centroid();

            // Get line geometry if needed
            let line_geometry = match &feature.geometry {
                FeatureGeometry::Line(pts) => Some(pts.clone()),
                FeatureGeometry::Polygon(pts) => Some(pts.clone()),
                _ => None,
            };

            // Apply style with priority based on feature properties
            let mut style = self.config.base_style.clone();
            style.min_zoom = self.config.min_zoom;
            style.max_zoom = self.config.max_zoom;

            // Priority from properties if available
            if let Some(pop_str) = feature.properties.get("population") {
                if let Ok(pop) = pop_str.parse::<i32>() {
                    style.priority = pop / 1000; // Higher population = higher priority
                }
            }

            self.labels.push(GeneratedLabel {
                feature_id: feature.id,
                text,
                position,
                style,
                placement_type,
                line_geometry,
            });
        }
    }

    /// Get number of features.
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    /// Get number of generated labels.
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    /// Clear all features and labels.
    pub fn clear(&mut self) {
        self.features.clear();
        self.labels.clear();
        self.label_ids.clear();
    }

    /// Set visibility.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
}

/// Create features from simple point data.
pub fn features_from_points(
    points: &[(Vec3, String)],
    label_field: &str,
) -> Vec<LabelFeature> {
    points
        .iter()
        .enumerate()
        .map(|(i, (pos, name))| {
            let mut properties = std::collections::HashMap::new();
            properties.insert(label_field.to_string(), name.clone());

            LabelFeature {
                id: i as u64,
                feature_type: FeatureType::Point,
                geometry: FeatureGeometry::Point(*pos),
                properties,
            }
        })
        .collect()
}

/// Create features from polylines.
pub fn features_from_lines(
    lines: &[(Vec<Vec3>, String)],
    label_field: &str,
) -> Vec<LabelFeature> {
    lines
        .iter()
        .enumerate()
        .map(|(i, (pts, name))| {
            let mut properties = std::collections::HashMap::new();
            properties.insert(label_field.to_string(), name.clone());

            LabelFeature {
                id: i as u64,
                feature_type: FeatureType::Line,
                geometry: FeatureGeometry::Line(pts.clone()),
                properties,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_layer_creation() {
        let config = LabelLayerConfig::default();
        let layer = LabelLayer::new(1, "test", config);
        assert_eq!(layer.feature_count(), 0);
    }

    #[test]
    fn test_feature_from_points() {
        let points = vec![
            (Vec3::new(0.0, 0.0, 0.0), "Point A".to_string()),
            (Vec3::new(10.0, 0.0, 0.0), "Point B".to_string()),
        ];
        let features = features_from_points(&points, "name");
        assert_eq!(features.len(), 2);
    }

    #[test]
    fn test_generate_labels() {
        let mut layer = LabelLayer::new(1, "test", LabelLayerConfig::default());
        
        let mut props = std::collections::HashMap::new();
        props.insert("name".to_string(), "Test Point".to_string());
        
        layer.add_feature(LabelFeature {
            id: 1,
            feature_type: FeatureType::Point,
            geometry: FeatureGeometry::Point(Vec3::new(0.0, 0.0, 0.0)),
            properties: props,
        });

        layer.generate_labels();
        assert_eq!(layer.label_count(), 1);
    }
}
