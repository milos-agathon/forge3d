//! Tileset parsing and management for 3D Tiles

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use super::bounds::BoundingVolume;
use super::error::Tiles3dResult;
use super::tile::{Tile, TileRefine};

/// Asset metadata for the tileset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilesetAsset {
    /// 3D Tiles version
    pub version: String,
    /// Application-specific version
    #[serde(rename = "tilesetVersion")]
    pub tileset_version: Option<String>,
}

/// Properties metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilesetProperties {
    /// Minimum value
    pub minimum: Option<f64>,
    /// Maximum value
    pub maximum: Option<f64>,
}

/// Root tileset.json structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilesetJson {
    /// Asset metadata
    pub asset: TilesetAsset,
    /// Geometric error of the tileset
    #[serde(rename = "geometricError")]
    pub geometric_error: f32,
    /// Root tile
    pub root: Tile,
    /// Optional properties
    pub properties: Option<std::collections::HashMap<String, TilesetProperties>>,
    /// Optional extensions used
    #[serde(rename = "extensionsUsed")]
    pub extensions_used: Option<Vec<String>>,
    /// Optional extensions required
    #[serde(rename = "extensionsRequired")]
    pub extensions_required: Option<Vec<String>>,
}

/// A loaded 3D Tiles tileset
#[derive(Debug)]
pub struct Tileset {
    /// Base path for resolving relative URIs
    pub base_path: PathBuf,
    /// Parsed tileset.json
    pub json: TilesetJson,
}

impl Tileset {
    /// Load a tileset from a file path
    pub fn load<P: AsRef<Path>>(path: P) -> Tiles3dResult<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let json: TilesetJson = serde_json::from_str(&content)?;
        
        let base_path = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        
        Ok(Self { base_path, json })
    }

    /// Load a tileset from JSON string with a base path
    pub fn from_json(json_str: &str, base_path: PathBuf) -> Tiles3dResult<Self> {
        let json: TilesetJson = serde_json::from_str(json_str)?;
        Ok(Self { base_path, json })
    }

    /// Get the root tile
    pub fn root(&self) -> &Tile {
        &self.json.root
    }

    /// Get the tileset version
    pub fn version(&self) -> &str {
        &self.json.asset.version
    }

    /// Get the root geometric error
    pub fn geometric_error(&self) -> f32 {
        self.json.geometric_error
    }

    /// Resolve a content URI to an absolute path
    pub fn resolve_uri(&self, uri: &str) -> PathBuf {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            PathBuf::from(uri)
        } else {
            self.base_path.join(uri)
        }
    }

    /// Get default refinement strategy for the tileset
    pub fn default_refine(&self) -> TileRefine {
        self.json.root.refine.unwrap_or(TileRefine::Replace)
    }

    /// Get total tile count
    pub fn tile_count(&self) -> usize {
        self.json.root.count_tiles()
    }

    /// Get maximum depth of the tile hierarchy
    pub fn max_depth(&self) -> usize {
        self.json.root.max_depth()
    }

    /// Get the bounding volume of the root tile
    pub fn bounding_volume(&self) -> &BoundingVolume {
        &self.json.root.bounding_volume
    }

    /// Check if any required extensions are present
    pub fn has_required_extensions(&self) -> bool {
        self.json.extensions_required.as_ref().map_or(false, |e| !e.is_empty())
    }

    /// Get list of required extensions
    pub fn required_extensions(&self) -> &[String] {
        self.json.extensions_required.as_deref().unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_tileset() {
        let json = r#"{
            "asset": { "version": "1.0" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {
                    "sphere": [0.0, 0.0, 0.0, 100.0]
                },
                "geometricError": 100.0,
                "refine": "REPLACE"
            }
        }"#;
        
        let tileset = Tileset::from_json(json, PathBuf::from(".")).unwrap();
        assert_eq!(tileset.version(), "1.0");
        assert_eq!(tileset.geometric_error(), 500.0);
        assert_eq!(tileset.tile_count(), 1);
    }

    #[test]
    fn test_parse_tileset_with_children() {
        let json = r#"{
            "asset": { "version": "1.0" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 100] },
                "geometricError": 100.0,
                "children": [
                    {
                        "boundingVolume": { "sphere": [-50, 0, 0, 50] },
                        "geometricError": 10.0,
                        "content": { "uri": "tile1.b3dm" }
                    },
                    {
                        "boundingVolume": { "sphere": [50, 0, 0, 50] },
                        "geometricError": 10.0,
                        "content": { "uri": "tile2.b3dm" }
                    }
                ]
            }
        }"#;
        
        let tileset = Tileset::from_json(json, PathBuf::from("/data")).unwrap();
        assert_eq!(tileset.tile_count(), 3);
        assert_eq!(tileset.max_depth(), 2);
        
        let children = &tileset.root().children;
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].content_uri(), Some("tile1.b3dm"));
    }
}
