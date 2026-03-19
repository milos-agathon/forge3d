use bytemuck::{Pod, Zeroable};

/// Triangle mesh for BVH construction - simple vertex/index representation
#[derive(Debug, Clone)]
pub struct MeshCPU {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>, // triangle indices (CCW winding)
}

impl MeshCPU {
    pub fn new(vertices: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> Self {
        Self { vertices, indices }
    }

    pub fn triangle_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Get triangle vertices by index
    pub fn get_triangle(&self, tri_idx: usize) -> Option<([f32; 3], [f32; 3], [f32; 3])> {
        if tri_idx >= self.indices.len() {
            return None;
        }
        let indices = self.indices[tri_idx];
        let v0 = *self.vertices.get(indices[0] as usize)?;
        let v1 = *self.vertices.get(indices[1] as usize)?;
        let v2 = *self.vertices.get(indices[2] as usize)?;
        Some((v0, v1, v2))
    }

    /// Compute triangle centroid
    pub fn triangle_centroid(&self, tri_idx: usize) -> Option<[f32; 3]> {
        let (v0, v1, v2) = self.get_triangle(tri_idx)?;
        Some([
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        ])
    }

    /// Compute triangle AABB
    pub fn triangle_aabb(&self, tri_idx: usize) -> Option<Aabb> {
        let (v0, v1, v2) = self.get_triangle(tri_idx)?;
        let mut aabb = Aabb::empty();
        aabb.expand_point(v0);
        aabb.expand_point(v1);
        aabb.expand_point(v2);
        Some(aabb)
    }
}

/// GPU-compatible AABB layout (16-byte aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Aabb {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            _pad0: 0.0,
            max: [f32::NEG_INFINITY; 3],
            _pad1: 0.0,
        }
    }

    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            min,
            _pad0: 0.0,
            max,
            _pad1: 0.0,
        }
    }

    pub fn expand_point(&mut self, point: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(point[i]);
            self.max[i] = self.max[i].max(point[i]);
        }
    }

    pub fn expand_aabb(&mut self, other: &Aabb) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    pub fn surface_area(&self) -> f32 {
        let extent = self.extent();
        if extent[0] < 0.0 || extent[1] < 0.0 || extent[2] < 0.0 {
            return 0.0;
        }
        2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])
    }

    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }
}

/// GPU-compatible BVH node layout as specified in task A3.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BvhNode {
    pub aabb_min: [f32; 3],
    pub left: u32, // if internal: left child index; if leaf: first triangle index
    pub aabb_max: [f32; 3],
    pub right: u32, // if internal: right child index; if leaf: triangle count
    pub flags: u32, // bit 0: leaf flag (1 = leaf, 0 = internal)
    pub _pad: u32,  // padding for 16-byte alignment
}

impl BvhNode {
    pub fn internal(aabb: Aabb, left_idx: u32, right_idx: u32) -> Self {
        Self {
            aabb_min: aabb.min,
            left: left_idx,
            aabb_max: aabb.max,
            right: right_idx,
            flags: 0,
            _pad: 0,
        }
    }

    pub fn leaf(aabb: Aabb, first_tri: u32, tri_count: u32) -> Self {
        Self {
            aabb_min: aabb.min,
            left: first_tri,
            aabb_max: aabb.max,
            right: tri_count,
            flags: 1,
            _pad: 0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        (self.flags & 1) != 0
    }

    pub fn is_internal(&self) -> bool {
        (self.flags & 1) == 0
    }

    pub fn aabb(&self) -> Aabb {
        Aabb {
            min: self.aabb_min,
            _pad0: 0.0,
            max: self.aabb_max,
            _pad1: 0.0,
        }
    }

    /// Get child indices for internal nodes
    pub fn children(&self) -> Option<(u32, u32)> {
        if self.is_internal() {
            Some((self.left, self.right))
        } else {
            None
        }
    }

    /// Get triangle range for leaf nodes
    pub fn triangles(&self) -> Option<(u32, u32)> {
        if self.is_leaf() {
            Some((self.left, self.right))
        } else {
            None
        }
    }
}

const _: () = {
    assert!(std::mem::size_of::<BvhNode>() == 40);
    assert!(std::mem::align_of::<BvhNode>() == 4);
};

/// Build options for BVH construction
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub max_leaf_size: u32,
    pub method: BuildMethod,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            max_leaf_size: 4,
            method: BuildMethod::MedianSplit,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuildMethod {
    MedianSplit,
}

/// CPU BVH data with flattened layout suitable for GPU upload
#[derive(Debug, Clone)]
pub struct BvhCPU {
    pub nodes: Vec<BvhNode>,
    pub tri_indices: Vec<u32>, // reordered triangle indices
    pub world_aabb: Aabb,
    pub build_stats: BuildStats,
}

impl BvhCPU {
    pub fn node_count(&self) -> u32 {
        self.nodes.len() as u32
    }

    pub fn triangle_count(&self) -> u32 {
        self.build_stats.triangle_count
    }
}

/// Build statistics
#[derive(Debug, Clone, Default)]
pub struct BuildStats {
    pub build_time_ms: f32,
    pub triangle_count: u32,
    pub node_count: u32,
    pub leaf_count: u32,
    pub internal_count: u32,
    pub max_depth: u32,
    pub avg_leaf_size: f32,
    pub memory_usage_bytes: u64,
}
