//! Bundle manifest schema and I/O.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::BUNDLE_VERSION;

/// Result type for bundle operations
pub type BundleResult<T> = Result<T, BundleError>;

/// Bundle operation errors
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Invalid bundle: {0}")]
    Invalid(String),
    
    #[error("Version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: u32, got: u32 },
    
    #[error("Checksum mismatch for {path}")]
    ChecksumMismatch { path: String },
}

/// Bundle manifest containing metadata and checksums.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Schema version (currently 1)
    pub version: u32,
    
    /// Human-readable bundle name
    pub name: String,
    
    /// ISO 8601 creation timestamp
    pub created_at: String,
    
    /// Optional description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    
    /// SHA-256 checksums for bundle files (path -> hex digest)
    #[serde(default)]
    pub checksums: HashMap<String, String>,
    
    /// Terrain metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terrain: Option<TerrainMeta>,
    
    /// Camera bookmarks
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub camera_bookmarks: Vec<CameraBookmark>,
}

/// Terrain metadata in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainMeta {
    /// Path to DEM file within bundle (e.g., "terrain/dem.tif")
    pub dem_path: String,
    
    /// Coordinate reference system (e.g., "EPSG:32610")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crs: Option<String>,
    
    /// Elevation domain [min, max]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<[f64; 2]>,
    
    /// Colormap name or path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colormap: Option<String>,
}

/// Camera bookmark for saved viewpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraBookmark {
    /// Bookmark name
    pub name: String,
    
    /// Camera eye position [x, y, z]
    pub eye: [f64; 3],
    
    /// Camera target position [x, y, z]
    pub target: [f64; 3],
    
    /// Camera up vector [x, y, z]
    #[serde(default = "default_up")]
    pub up: [f64; 3],
    
    /// Field of view in degrees
    #[serde(default = "default_fov")]
    pub fov_deg: f64,
}

fn default_up() -> [f64; 3] {
    [0.0, 1.0, 0.0]
}

fn default_fov() -> f64 {
    45.0
}

impl BundleManifest {
    /// Create a new manifest with the current version
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: BUNDLE_VERSION,
            name: name.into(),
            created_at: chrono_timestamp(),
            description: None,
            checksums: HashMap::new(),
            terrain: None,
            camera_bookmarks: Vec::new(),
        }
    }
    
    /// Load manifest from a JSON file
    pub fn load(path: &Path) -> BundleResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let manifest: Self = serde_json::from_str(&data)?;
        
        if manifest.version > BUNDLE_VERSION {
            return Err(BundleError::VersionMismatch {
                expected: BUNDLE_VERSION,
                got: manifest.version,
            });
        }
        
        Ok(manifest)
    }
    
    /// Save manifest to a JSON file
    pub fn save(&self, path: &Path) -> BundleResult<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
    
    /// Add a checksum for a file path
    pub fn add_checksum(&mut self, rel_path: impl Into<String>, hash: impl Into<String>) {
        self.checksums.insert(rel_path.into(), hash.into());
    }
    
    /// Verify a file's checksum
    pub fn verify_checksum(&self, rel_path: &str, actual_hash: &str) -> BundleResult<()> {
        if let Some(expected) = self.checksums.get(rel_path) {
            if expected != actual_hash {
                return Err(BundleError::ChecksumMismatch {
                    path: rel_path.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Generate ISO 8601 timestamp
fn chrono_timestamp() -> String {
    // Use simple UTC timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    
    // Convert to rough ISO 8601 format
    let days_since_epoch = secs / 86400;
    let years = 1970 + (days_since_epoch / 365);
    let remaining_days = days_since_epoch % 365;
    let month = (remaining_days / 30).min(11) + 1;
    let day = (remaining_days % 30) + 1;
    
    let day_secs = secs % 86400;
    let hour = day_secs / 3600;
    let minute = (day_secs % 3600) / 60;
    let second = day_secs % 60;
    
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, month, day, hour, minute, second
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manifest_roundtrip() {
        let mut manifest = BundleManifest::new("test_bundle");
        manifest.description = Some("Test description".into());
        manifest.add_checksum("terrain/dem.tif", "abc123");
        manifest.terrain = Some(TerrainMeta {
            dem_path: "terrain/dem.tif".into(),
            crs: Some("EPSG:32610".into()),
            domain: Some([0.0, 1000.0]),
            colormap: Some("viridis".into()),
        });
        
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let loaded: BundleManifest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(loaded.version, BUNDLE_VERSION);
        assert_eq!(loaded.name, "test_bundle");
        assert_eq!(loaded.checksums.get("terrain/dem.tif"), Some(&"abc123".into()));
    }
}
