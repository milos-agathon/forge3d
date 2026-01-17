//! EPT (Entwine Point Tile) format support

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use super::error::{PointCloudError, PointCloudResult};
use super::octree::{OctreeKey, OctreeBounds, OctreeNode};
use glam::Vec3;

/// EPT schema dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EptDimension {
    pub name: String,
    #[serde(rename = "type")]
    pub dtype: String,
    pub size: u32,
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub offset: Option<f64>,
}

/// EPT schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EptSchema(pub Vec<EptDimension>);

/// EPT dataset info from ept.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EptInfo {
    pub bounds: [f64; 6],
    #[serde(rename = "boundsConforming")]
    pub bounds_conforming: Option<[f64; 6]>,
    pub points: u64,
    pub schema: EptSchema,
    pub span: u32,
    #[serde(rename = "dataType")]
    pub data_type: String,
    #[serde(rename = "hierarchyType")]
    pub hierarchy_type: String,
    pub srs: Option<serde_json::Value>,
}

/// EPT dataset handle
pub struct EptDataset {
    base_path: PathBuf,
    pub info: EptInfo,
    hierarchy: HashMap<OctreeKey, u64>,
    root_bounds: OctreeBounds,
}

impl EptDataset {
    /// Open an EPT dataset from ept.json path
    pub fn open<P: AsRef<Path>>(path: P) -> PointCloudResult<Self> {
        let path = path.as_ref();
        let base_path = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        
        let content = std::fs::read_to_string(path)?;
        let info: EptInfo = serde_json::from_str(&content)?;
        
        let root_bounds = OctreeBounds::new(
            Vec3::new(info.bounds[0] as f32, info.bounds[1] as f32, info.bounds[2] as f32),
            Vec3::new(info.bounds[3] as f32, info.bounds[4] as f32, info.bounds[5] as f32),
        );

        let mut dataset = Self {
            base_path,
            info,
            hierarchy: HashMap::new(),
            root_bounds,
        };
        
        dataset.load_hierarchy()?;
        Ok(dataset)
    }

    fn load_hierarchy(&mut self) -> PointCloudResult<()> {
        match self.info.hierarchy_type.as_str() {
            "json" => self.load_json_hierarchy(),
            "gzip" => self.load_json_hierarchy(), // Same format, just compressed
            _ => Err(PointCloudError::Unsupported(
                format!("Hierarchy type: {}", self.info.hierarchy_type)
            )),
        }
    }

    fn load_json_hierarchy(&mut self) -> PointCloudResult<()> {
        self.load_hierarchy_node(&OctreeKey::root())
    }

    fn load_hierarchy_node(&mut self, key: &OctreeKey) -> PointCloudResult<()> {
        let hier_path = self.hierarchy_path(key);
        
        if !hier_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&hier_path)?;
        let hier: HashMap<String, i64> = serde_json::from_str(&content)?;
        
        for (key_str, point_count) in hier {
            if point_count > 0 {
                if let Some(node_key) = OctreeKey::from_str(&key_str) {
                    self.hierarchy.insert(node_key, point_count as u64);
                }
            } else if point_count == -1 {
                // -1 means subtree exists, load it
                if let Some(node_key) = OctreeKey::from_str(&key_str) {
                    self.load_hierarchy_node(&node_key)?;
                }
            }
        }
        Ok(())
    }

    fn hierarchy_path(&self, key: &OctreeKey) -> PathBuf {
        self.base_path.join("ept-hierarchy").join(format!("{}.json", key.to_string()))
    }

    fn data_path(&self, key: &OctreeKey) -> PathBuf {
        let ext = match self.info.data_type.as_str() {
            "laszip" => "laz",
            "binary" => "bin",
            "zstandard" => "zst",
            _ => "bin",
        };
        self.base_path.join("ept-data").join(format!("{}.{}", key.to_string(), ext))
    }

    /// Get root node
    pub fn root_node(&self) -> OctreeNode {
        let key = OctreeKey::root();
        let point_count = self.hierarchy.get(&key).copied().unwrap_or(0);
        let mut node = OctreeNode::new(key, self.root_bounds, point_count);
        
        for octant in 0..8 {
            let child_key = node.key.child(octant);
            if self.hierarchy.contains_key(&child_key) {
                node.children.push(child_key);
            }
        }
        node
    }

    /// Get child nodes
    pub fn children(&self, key: &OctreeKey) -> Vec<OctreeNode> {
        let mut children = Vec::new();
        for octant in 0..8 {
            let child_key = key.child(octant);
            if let Some(&point_count) = self.hierarchy.get(&child_key) {
                let bounds = self.bounds_for_key(&child_key);
                let mut node = OctreeNode::new(child_key, bounds, point_count);
                
                for o in 0..8 {
                    let grandchild = node.key.child(o);
                    if self.hierarchy.contains_key(&grandchild) {
                        node.children.push(grandchild);
                    }
                }
                children.push(node);
            }
        }
        children
    }

    fn bounds_for_key(&self, key: &OctreeKey) -> OctreeBounds {
        let mut bounds = self.root_bounds;
        for d in 0..key.depth {
            let shift = key.depth - d - 1;
            let octant = (((key.x >> shift) & 1) |
                         (((key.y >> shift) & 1) << 1) |
                         (((key.z >> shift) & 1) << 2)) as u8;
            bounds = bounds.child_bounds(octant);
        }
        bounds
    }

    /// Read points for a node
    pub fn read_points(&self, key: &OctreeKey) -> PointCloudResult<PointData> {
        let path = self.data_path(key);
        if !path.exists() {
            return Err(PointCloudError::InvalidEpt(format!("Data file not found: {:?}", path)));
        }
        
        let data = std::fs::read(&path)?;
        self.decode_points(&data)
    }

    fn decode_points(&self, data: &[u8]) -> PointCloudResult<PointData> {
        let schema = &self.info.schema.0;
        let record_size: usize = schema.iter().map(|d| d.size as usize).sum();
        let point_count = data.len() / record_size;
        
        let mut positions = Vec::with_capacity(point_count * 3);
        let mut colors = None;
        
        let x_dim = schema.iter().find(|d| d.name == "X");
        let y_dim = schema.iter().find(|d| d.name == "Y");
        let z_dim = schema.iter().find(|d| d.name == "Z");
        let r_dim = schema.iter().find(|d| d.name == "Red");
        
        let has_color = r_dim.is_some();
        if has_color {
            colors = Some(Vec::with_capacity(point_count * 3));
        }
        
        let mut offset_map: HashMap<&str, usize> = HashMap::new();
        let mut off = 0;
        for dim in schema {
            offset_map.insert(&dim.name, off);
            off += dim.size as usize;
        }
        
        for i in 0..point_count {
            let base = i * record_size;
            
            if let (Some(x), Some(y), Some(z)) = (x_dim, y_dim, z_dim) {
                let x_off = base + offset_map["X"];
                let y_off = base + offset_map["Y"];
                let z_off = base + offset_map["Z"];
                
                let xv = read_dim_value(&data[x_off..], x);
                let yv = read_dim_value(&data[y_off..], y);
                let zv = read_dim_value(&data[z_off..], z);
                
                positions.push(xv as f32);
                positions.push(yv as f32);
                positions.push(zv as f32);
            }
            
            if let Some(ref mut cols) = colors {
                let r_off = base + offset_map.get("Red").copied().unwrap_or(0);
                let g_off = base + offset_map.get("Green").copied().unwrap_or(0);
                let b_off = base + offset_map.get("Blue").copied().unwrap_or(0);
                
                cols.push((u16::from_le_bytes([data[r_off], data[r_off+1]]) >> 8) as u8);
                cols.push((u16::from_le_bytes([data[g_off], data[g_off+1]]) >> 8) as u8);
                cols.push((u16::from_le_bytes([data[b_off], data[b_off+1]]) >> 8) as u8);
            }
        }
        
        Ok(PointData { positions, colors, intensities: None })
    }

    pub fn node_count(&self) -> usize { self.hierarchy.len() }
    pub fn total_points(&self) -> u64 { self.info.points }
    pub fn bounds(&self) -> OctreeBounds { self.root_bounds }
}

/// Decoded point data (shared with COPC)
#[derive(Debug)]
pub struct PointData {
    pub positions: Vec<f32>,
    pub colors: Option<Vec<u8>>,
    pub intensities: Option<Vec<u16>>,
}

fn read_dim_value(data: &[u8], dim: &EptDimension) -> f64 {
    let raw = match (dim.dtype.as_str(), dim.size) {
        ("signed", 4) => i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f64,
        ("unsigned", 4) => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f64,
        ("float", 8) => f64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]),
        ("float", 4) => f32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f64,
        _ => 0.0,
    };
    
    let scale = dim.scale.unwrap_or(1.0);
    let offset = dim.offset.unwrap_or(0.0);
    raw * scale + offset
}
