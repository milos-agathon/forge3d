//! Octree data structures for point cloud LOD

use glam::Vec3;

/// Octree node key (Morton code style: D-X-Y-Z)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OctreeKey {
    pub depth: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl OctreeKey {
    pub fn root() -> Self {
        Self { depth: 0, x: 0, y: 0, z: 0 }
    }

    pub fn new(depth: u32, x: u32, y: u32, z: u32) -> Self {
        Self { depth, x, y, z }
    }

    /// Get child key for given octant (0-7)
    pub fn child(&self, octant: u8) -> Self {
        let cx = (self.x << 1) | ((octant & 1) as u32);
        let cy = (self.y << 1) | (((octant >> 1) & 1) as u32);
        let cz = (self.z << 1) | (((octant >> 2) & 1) as u32);
        Self { depth: self.depth + 1, x: cx, y: cy, z: cz }
    }

    /// Get parent key
    pub fn parent(&self) -> Option<Self> {
        if self.depth == 0 { return None; }
        Some(Self {
            depth: self.depth - 1,
            x: self.x >> 1,
            y: self.y >> 1,
            z: self.z >> 1,
        })
    }

    /// Convert to string representation (e.g., "0-0-0-0" for root)
    pub fn to_string(&self) -> String {
        format!("{}-{}-{}-{}", self.depth, self.x, self.y, self.z)
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 4 { return None; }
        Some(Self {
            depth: parts[0].parse().ok()?,
            x: parts[1].parse().ok()?,
            y: parts[2].parse().ok()?,
            z: parts[3].parse().ok()?,
        })
    }
}

/// Axis-aligned bounding box for octree nodes
#[derive(Debug, Clone, Copy)]
pub struct OctreeBounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl OctreeBounds {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn radius(&self) -> f32 {
        self.size().length() * 0.5
    }

    /// Get bounds for child octant
    pub fn child_bounds(&self, octant: u8) -> Self {
        let center = self.center();
        let min = Vec3::new(
            if octant & 1 == 0 { self.min.x } else { center.x },
            if octant & 2 == 0 { self.min.y } else { center.y },
            if octant & 4 == 0 { self.min.z } else { center.z },
        );
        let max = Vec3::new(
            if octant & 1 == 0 { center.x } else { self.max.x },
            if octant & 2 == 0 { center.y } else { self.max.y },
            if octant & 4 == 0 { center.z } else { self.max.z },
        );
        Self { min, max }
    }

    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn intersects_frustum(&self, view_proj: &glam::Mat4) -> bool {
        let center = self.center();
        let radius = self.radius();
        let clip = *view_proj * center.extend(1.0);
        if clip.w <= 0.0 { return radius > clip.z.abs(); }
        let ndc = clip.truncate() / clip.w;
        let margin = radius / clip.w;
        ndc.x.abs() <= 1.0 + margin && ndc.y.abs() <= 1.0 + margin && ndc.z >= -margin
    }
}

/// Octree node with metadata
#[derive(Debug, Clone)]
pub struct OctreeNode {
    pub key: OctreeKey,
    pub bounds: OctreeBounds,
    pub point_count: u64,
    pub spacing: f32,
    pub children: Vec<OctreeKey>,
}

impl OctreeNode {
    pub fn new(key: OctreeKey, bounds: OctreeBounds, point_count: u64) -> Self {
        let spacing = bounds.size().x / (1 << key.depth) as f32;
        Self { key, bounds, point_count, spacing, children: Vec::new() }
    }

    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}
