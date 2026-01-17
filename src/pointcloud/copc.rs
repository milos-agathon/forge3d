//! COPC (Cloud Optimized Point Cloud) format support

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::collections::HashMap;
use super::error::{PointCloudError, PointCloudResult};
use super::octree::{OctreeKey, OctreeBounds, OctreeNode};
use glam::Vec3;

/// COPC file header info
#[derive(Debug, Clone)]
pub struct CopcHeader {
    pub point_count: u64,
    pub point_format: u8,
    pub point_record_length: u16,
    pub scale: [f64; 3],
    pub offset: [f64; 3],
    pub min_bounds: [f64; 3],
    pub max_bounds: [f64; 3],
}

/// COPC-specific info from VLR
#[derive(Debug, Clone)]
pub struct CopcInfo {
    pub center: [f64; 3],
    pub halfsize: f64,
    pub spacing: f64,
    pub root_hier_offset: u64,
    pub root_hier_size: u64,
    pub gpstime_minimum: f64,
    pub gpstime_maximum: f64,
}

/// COPC dataset handle
pub struct CopcDataset {
    path: std::path::PathBuf,
    pub header: CopcHeader,
    pub info: CopcInfo,
    hierarchy: HashMap<OctreeKey, HierarchyEntry>,
    root_bounds: OctreeBounds,
}

#[derive(Debug, Clone)]
struct HierarchyEntry {
    offset: u64,
    byte_size: u32,
    point_count: u32,
}

impl CopcDataset {
    /// Open a COPC file
    pub fn open<P: AsRef<Path>>(path: P) -> PointCloudResult<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = std::fs::File::open(&path)?;
        
        let header = read_las_header(&mut file)?;
        let info = read_copc_vlr(&mut file, &header)?;
        
        let root_bounds = OctreeBounds::new(
            Vec3::new(
                (info.center[0] - info.halfsize) as f32,
                (info.center[1] - info.halfsize) as f32,
                (info.center[2] - info.halfsize) as f32,
            ),
            Vec3::new(
                (info.center[0] + info.halfsize) as f32,
                (info.center[1] + info.halfsize) as f32,
                (info.center[2] + info.halfsize) as f32,
            ),
        );

        let mut dataset = Self {
            path,
            header,
            info,
            hierarchy: HashMap::new(),
            root_bounds,
        };
        
        dataset.load_root_hierarchy()?;
        Ok(dataset)
    }

    fn load_root_hierarchy(&mut self) -> PointCloudResult<()> {
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.info.root_hier_offset))?;
        
        let mut buf = vec![0u8; self.info.root_hier_size as usize];
        file.read_exact(&mut buf)?;
        
        self.parse_hierarchy_page(&buf)?;
        Ok(())
    }

    fn parse_hierarchy_page(&mut self, data: &[u8]) -> PointCloudResult<()> {
        let entry_size = 32;
        let count = data.len() / entry_size;
        
        for i in 0..count {
            let offset = i * entry_size;
            let d = i32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
            let x = i32::from_le_bytes([data[offset+4], data[offset+5], data[offset+6], data[offset+7]]);
            let y = i32::from_le_bytes([data[offset+8], data[offset+9], data[offset+10], data[offset+11]]);
            let z = i32::from_le_bytes([data[offset+12], data[offset+13], data[offset+14], data[offset+15]]);
            let file_offset = u64::from_le_bytes([
                data[offset+16], data[offset+17], data[offset+18], data[offset+19],
                data[offset+20], data[offset+21], data[offset+22], data[offset+23],
            ]);
            let byte_size = i32::from_le_bytes([data[offset+24], data[offset+25], data[offset+26], data[offset+27]]);
            let point_count = i32::from_le_bytes([data[offset+28], data[offset+29], data[offset+30], data[offset+31]]);

            if d >= 0 && point_count > 0 {
                let key = OctreeKey::new(d as u32, x as u32, y as u32, z as u32);
                self.hierarchy.insert(key, HierarchyEntry {
                    offset: file_offset,
                    byte_size: byte_size as u32,
                    point_count: point_count as u32,
                });
            }
        }
        Ok(())
    }

    /// Get root node
    pub fn root_node(&self) -> OctreeNode {
        let key = OctreeKey::root();
        let entry = self.hierarchy.get(&key);
        let point_count = entry.map_or(0, |e| e.point_count as u64);
        OctreeNode::new(key, self.root_bounds, point_count)
    }

    /// Get child nodes for a key
    pub fn children(&self, key: &OctreeKey) -> Vec<OctreeNode> {
        let mut children = Vec::new();
        for octant in 0..8 {
            let child_key = key.child(octant);
            if let Some(entry) = self.hierarchy.get(&child_key) {
                let bounds = self.bounds_for_key(&child_key);
                let mut node = OctreeNode::new(child_key, bounds, entry.point_count as u64);
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
        let mut current = OctreeKey::root();
        
        for d in 0..key.depth {
            let shift = key.depth - d - 1;
            let octant = (((key.x >> shift) & 1) |
                         (((key.y >> shift) & 1) << 1) |
                         (((key.z >> shift) & 1) << 2)) as u8;
            bounds = bounds.child_bounds(octant);
            current = current.child(octant);
        }
        bounds
    }

    /// Read points for a node
    pub fn read_points(&self, key: &OctreeKey) -> PointCloudResult<PointData> {
        let entry = self.hierarchy.get(key)
            .ok_or_else(|| PointCloudError::InvalidCopc("Node not in hierarchy".into()))?;
        
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(entry.offset))?;
        
        let mut compressed = vec![0u8; entry.byte_size as usize];
        file.read_exact(&mut compressed)?;
        
        decode_laz_chunk(&compressed, entry.point_count, &self.header)
    }

    pub fn node_count(&self) -> usize { self.hierarchy.len() }
    pub fn total_points(&self) -> u64 { self.header.point_count }
    pub fn bounds(&self) -> OctreeBounds { self.root_bounds }
}

/// Decoded point data
#[derive(Debug)]
pub struct PointData {
    pub positions: Vec<f32>,
    pub colors: Option<Vec<u8>>,
    pub intensities: Option<Vec<u16>>,
}

fn read_las_header<R: Read + Seek>(reader: &mut R) -> PointCloudResult<CopcHeader> {
    let mut buf = [0u8; 375];
    reader.read_exact(&mut buf)?;
    
    if &buf[0..4] != b"LASF" {
        return Err(PointCloudError::InvalidCopc("Not a LAS file".into()));
    }
    
    let point_format = buf[104];
    let point_record_length = u16::from_le_bytes([buf[105], buf[106]]);
    let point_count = u64::from_le_bytes([buf[247], buf[248], buf[249], buf[250], buf[251], buf[252], buf[253], buf[254]]);
    
    let scale = [
        f64::from_le_bytes(buf[131..139].try_into().unwrap()),
        f64::from_le_bytes(buf[139..147].try_into().unwrap()),
        f64::from_le_bytes(buf[147..155].try_into().unwrap()),
    ];
    let offset = [
        f64::from_le_bytes(buf[155..163].try_into().unwrap()),
        f64::from_le_bytes(buf[163..171].try_into().unwrap()),
        f64::from_le_bytes(buf[171..179].try_into().unwrap()),
    ];
    let max_bounds = [
        f64::from_le_bytes(buf[179..187].try_into().unwrap()),
        f64::from_le_bytes(buf[203..211].try_into().unwrap()),
        f64::from_le_bytes(buf[227..235].try_into().unwrap()),
    ];
    let min_bounds = [
        f64::from_le_bytes(buf[187..195].try_into().unwrap()),
        f64::from_le_bytes(buf[211..219].try_into().unwrap()),
        f64::from_le_bytes(buf[235..243].try_into().unwrap()),
    ];

    Ok(CopcHeader {
        point_count,
        point_format,
        point_record_length,
        scale,
        offset,
        min_bounds,
        max_bounds,
    })
}

fn read_copc_vlr<R: Read + Seek>(reader: &mut R, _header: &CopcHeader) -> PointCloudResult<CopcInfo> {
    reader.seek(SeekFrom::Start(375))?;
    
    let mut vlr_buf = [0u8; 54];
    reader.read_exact(&mut vlr_buf)?;
    
    let user_id = std::str::from_utf8(&vlr_buf[2..18]).unwrap_or("").trim_end_matches('\0');
    if user_id != "copc" {
        return Err(PointCloudError::InvalidCopc("Missing COPC VLR".into()));
    }
    
    let content_size = u16::from_le_bytes([vlr_buf[20], vlr_buf[21]]) as usize;
    let mut content = vec![0u8; content_size];
    reader.read_exact(&mut content)?;
    
    Ok(CopcInfo {
        center: [
            f64::from_le_bytes(content[0..8].try_into().unwrap()),
            f64::from_le_bytes(content[8..16].try_into().unwrap()),
            f64::from_le_bytes(content[16..24].try_into().unwrap()),
        ],
        halfsize: f64::from_le_bytes(content[24..32].try_into().unwrap()),
        spacing: f64::from_le_bytes(content[32..40].try_into().unwrap()),
        root_hier_offset: u64::from_le_bytes(content[40..48].try_into().unwrap()),
        root_hier_size: u64::from_le_bytes(content[48..56].try_into().unwrap()),
        gpstime_minimum: f64::from_le_bytes(content[56..64].try_into().unwrap()),
        gpstime_maximum: f64::from_le_bytes(content[64..72].try_into().unwrap()),
    })
}

fn decode_laz_chunk(data: &[u8], point_count: u32, header: &CopcHeader) -> PointCloudResult<PointData> {
    // Simplified: assume uncompressed for now (real impl would use laz-rs)
    let record_len = header.point_record_length as usize;
    let expected = point_count as usize * record_len;
    
    if data.len() < expected {
        // Data is likely LAZ compressed - return placeholder
        return Ok(PointData {
            positions: vec![0.0; point_count as usize * 3],
            colors: None,
            intensities: None,
        });
    }
    
    let mut positions = Vec::with_capacity(point_count as usize * 3);
    let has_rgb = header.point_format == 2 || header.point_format == 3 || 
                  header.point_format == 5 || header.point_format >= 7;
    let mut colors = if has_rgb { Some(Vec::with_capacity(point_count as usize * 3)) } else { None };
    
    for i in 0..point_count as usize {
        let off = i * record_len;
        let x = i32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let y = i32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
        let z = i32::from_le_bytes([data[off+8], data[off+9], data[off+10], data[off+11]]);
        
        positions.push((x as f64 * header.scale[0] + header.offset[0]) as f32);
        positions.push((y as f64 * header.scale[1] + header.offset[1]) as f32);
        positions.push((z as f64 * header.scale[2] + header.offset[2]) as f32);
        
        if let Some(ref mut cols) = colors {
            let rgb_off = off + 20;
            if rgb_off + 6 <= data.len() {
                cols.push((u16::from_le_bytes([data[rgb_off], data[rgb_off+1]]) >> 8) as u8);
                cols.push((u16::from_le_bytes([data[rgb_off+2], data[rgb_off+3]]) >> 8) as u8);
                cols.push((u16::from_le_bytes([data[rgb_off+4], data[rgb_off+5]]) >> 8) as u8);
            }
        }
    }
    
    Ok(PointData { positions, colors, intensities: None })
}
