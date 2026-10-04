// src/splat/bvh.rs
// Bounding-volume hierarchies for the fused scene: one top-level tree over
// heterogeneous leaves (splat pages, COPC/LiDAR octree nodes, terrain tiles)
// and a small per-page tree over the primitives of a resident page. Both use
// the same 32-byte node the WGSL traversal in
// src/shaders/fusion/unified_occlusion.wgsl reads.
// RELEVANT FILES: src/splat/fusion.rs, src/splat/stream.rs,
//                 src/shaders/fusion/unified_occlusion.wgsl

use bytemuck::{Pod, Zeroable};

/// Leaf marker in `FusionBvhNode::b`.
pub const LEAF_FLAG: u32 = 0x8000_0000;
/// Leaf kind: a page of Gaussian splats.
pub const KIND_SPLAT: u32 = 1;
/// Leaf kind: a page of LiDAR/COPC points.
pub const KIND_LIDAR: u32 = 2;
/// Leaf kind: a terrain heightfield tile.
pub const KIND_TERRAIN: u32 = 3;
const KIND_SHIFT: u32 = 28;
const COUNT_MASK: u32 = (1 << KIND_SHIFT) - 1;
/// Maximum primitives per leaf of a per-page tree.
pub const BLAS_LEAF_SIZE: usize = 4;

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    pub fn union(self, other: Self) -> Self {
        let mut out = self;
        for axis in 0..3 {
            out.min[axis] = out.min[axis].min(other.min[axis]);
            out.max[axis] = out.max[axis].max(other.max[axis]);
        }
        out
    }

    pub fn centroid(self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    pub fn is_valid(self) -> bool {
        (0..3).all(|axis| {
            self.min[axis].is_finite()
                && self.max[axis].is_finite()
                && self.min[axis] <= self.max[axis]
        })
    }

    /// Slab test: the ray parameter span `[enter, exit]` clipped to
    /// `[tmin, tmax]`, or `None` when the ray misses the box.
    pub fn ray_span(self, origin: [f32; 3], dir: [f32; 3], tmin: f32, tmax: f32) -> Option<(f32, f32)> {
        let (mut enter, mut exit) = (tmin, tmax);
        for axis in 0..3 {
            let d = dir[axis];
            let inv = if d.abs() < 1e-12 {
                if d < 0.0 {
                    -1e12
                } else {
                    1e12
                }
            } else {
                1.0 / d
            };
            let mut t0 = (self.min[axis] - origin[axis]) * inv;
            let mut t1 = (self.max[axis] - origin[axis]) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            enter = enter.max(t0);
            exit = exit.min(t1);
            if enter > exit {
                return None;
            }
        }
        Some((enter, exit))
    }
}

/// 32-byte BVH node shared by the top-level and per-page trees.
///
/// Interior: `a` = left child, `b` = right child (indices relative to the
/// tree's first node). Leaf: `b` has `LEAF_FLAG` set, bits 28..=30 carry the
/// leaf kind and bits 0..=27 the primitive count; `a` is the page id (top
/// level) or the first primitive of the leaf's contiguous run (per page).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct FusionBvhNode {
    pub aabb_min: [f32; 3],
    pub a: u32,
    pub aabb_max: [f32; 3],
    pub b: u32,
}

impl FusionBvhNode {
    pub fn is_leaf(&self) -> bool {
        self.b & LEAF_FLAG != 0
    }

    pub fn leaf_kind(&self) -> u32 {
        (self.b >> KIND_SHIFT) & 0x7
    }

    pub fn leaf_count(&self) -> u32 {
        self.b & COUNT_MASK
    }

    pub fn aabb(&self) -> Aabb {
        Aabb::new(self.aabb_min, self.aabb_max)
    }
}

fn leaf_word(kind: u32, count: u32) -> u32 {
    LEAF_FLAG | ((kind & 0x7) << KIND_SHIFT) | (count & COUNT_MASK)
}

/// A built tree plus the primitive permutation that makes every leaf a
/// contiguous run: `order[k]` is the input index stored at position `k`.
#[derive(Clone, Debug)]
pub struct Bvh {
    pub nodes: Vec<FusionBvhNode>,
    pub order: Vec<u32>,
}

impl Bvh {
    /// Root bounds, or `Aabb::EMPTY` for an empty tree.
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::EMPTY, |node| node.aabb())
    }

    /// Maximum root-to-leaf depth (root alone = 1).
    pub fn depth(&self) -> u32 {
        fn walk(nodes: &[FusionBvhNode], index: usize) -> u32 {
            let node = &nodes[index];
            if node.is_leaf() {
                1
            } else {
                1 + walk(nodes, node.a as usize).max(walk(nodes, node.b as usize))
            }
        }
        if self.nodes.is_empty() {
            0
        } else {
            walk(&self.nodes, 0)
        }
    }
}

struct Builder<'a> {
    boxes: &'a [Aabb],
    centroids: Vec<[f32; 3]>,
    order: Vec<u32>,
    nodes: Vec<FusionBvhNode>,
    leaf_size: usize,
}

impl Builder<'_> {
    /// Build the subtree over `order[start..end]`; returns its node index.
    fn build(&mut self, start: usize, end: usize) -> u32 {
        let bounds = self.order[start..end]
            .iter()
            .fold(Aabb::EMPTY, |acc, &i| acc.union(self.boxes[i as usize]));
        let index = self.nodes.len();
        self.nodes.push(FusionBvhNode {
            aabb_min: bounds.min,
            a: 0,
            aabb_max: bounds.max,
            b: 0,
        });
        let count = end - start;
        if count <= self.leaf_size {
            self.nodes[index].a = start as u32;
            self.nodes[index].b = leaf_word(0, count as u32);
            return index as u32;
        }
        // Median split along the widest centroid extent: balanced, so the
        // depth is logarithmic and the GPU traversal stack stays bounded.
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for &i in &self.order[start..end] {
            let c = self.centroids[i as usize];
            for axis in 0..3 {
                lo[axis] = lo[axis].min(c[axis]);
                hi[axis] = hi[axis].max(c[axis]);
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap();
        let mid = count / 2;
        let centroids = &self.centroids;
        self.order[start..end].select_nth_unstable_by(mid, |&a, &b| {
            centroids[a as usize][axis]
                .total_cmp(&centroids[b as usize][axis])
                .then(a.cmp(&b))
        });
        let left = self.build(start, start + mid);
        let right = self.build(start + mid, end);
        self.nodes[index].a = left;
        self.nodes[index].b = right;
        index as u32
    }
}

/// Build a balanced BVH over `boxes` with at most `leaf_size` primitives per
/// leaf. Leaves reference contiguous runs of the returned permutation.
pub fn build_bvh(boxes: &[Aabb], leaf_size: usize) -> Bvh {
    if boxes.is_empty() {
        return Bvh {
            nodes: Vec::new(),
            order: Vec::new(),
        };
    }
    let mut builder = Builder {
        boxes,
        centroids: boxes.iter().map(|b| b.centroid()).collect(),
        order: (0..boxes.len() as u32).collect(),
        nodes: Vec::with_capacity(2 * boxes.len() / leaf_size.max(1) + 1),
        leaf_size: leaf_size.max(1),
    };
    builder.build(0, boxes.len());
    Bvh {
        nodes: builder.nodes,
        order: builder.order,
    }
}

/// One leaf of the fused top-level tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasLeaf {
    pub aabb: Aabb,
    /// `KIND_SPLAT`, `KIND_LIDAR` or `KIND_TERRAIN`.
    pub kind: u32,
    /// Global page id (splat/LiDAR) or terrain tile index.
    pub id: u32,
}

/// Build the single top-level acceleration structure over splat page
/// proxies, COPC octree node boxes and terrain tiles. Every leaf holds
/// exactly one entry whose kind and id are baked into the node.
pub fn build_tlas(leaves: &[TlasLeaf]) -> Vec<FusionBvhNode> {
    let boxes: Vec<Aabb> = leaves.iter().map(|leaf| leaf.aabb).collect();
    let mut bvh = build_bvh(&boxes, 1);
    for node in &mut bvh.nodes {
        if node.is_leaf() {
            let leaf = leaves[bvh.order[node.a as usize] as usize];
            node.a = leaf.id;
            node.b = leaf_word(leaf.kind, 1);
        }
    }
    bvh.nodes
}

/// Upper bound on the node count of a per-page tree over `count` primitives.
/// The median split never produces a leaf smaller than two primitives once a
/// node holds more than `BLAS_LEAF_SIZE`, so the tree has at most
/// `ceil(count / 2)` leaves.
pub fn blas_node_bound(count: usize) -> usize {
    (2 * count.div_ceil(2)).max(1)
}

#[cfg(test)]
mod tests {
    use super::super::num::f32_from_usize;
    use super::*;

    fn lattice(n: usize) -> Vec<Aabb> {
        (0..n)
            .map(|i| {
                // Deterministic low-discrepancy scatter.
                let x = (f32_from_usize(i) * 0.618_034).fract() * 100.0;
                let y = (f32_from_usize(i) * 0.414_214).fract() * 10.0;
                let z = (f32_from_usize(i) * 0.732_051).fract() * 100.0;
                Aabb::new([x, y, z], [x + 0.5, y + 0.25, z + 0.5])
            })
            .collect()
    }

    fn check_containment(nodes: &[FusionBvhNode], index: usize) {
        let node = nodes[index];
        if node.is_leaf() {
            return;
        }
        for child in [node.a as usize, node.b as usize] {
            let c = nodes[child];
            for axis in 0..3 {
                assert!(c.aabb_min[axis] >= node.aabb_min[axis]);
                assert!(c.aabb_max[axis] <= node.aabb_max[axis]);
            }
            check_containment(nodes, child);
        }
    }

    #[test]
    fn blas_leaves_partition_the_primitives_and_parents_contain_children() {
        for n in [1usize, 2, 3, 5, 17, 4096] {
            let boxes = lattice(n);
            let bvh = build_bvh(&boxes, BLAS_LEAF_SIZE);
            assert!(bvh.nodes.len() <= blas_node_bound(n), "{n}: {}", bvh.nodes.len());
            let mut seen = vec![false; n];
            for node in &bvh.nodes {
                if node.is_leaf() {
                    assert!(node.leaf_count() as usize <= BLAS_LEAF_SIZE);
                    for k in 0..node.leaf_count() as usize {
                        let prim = bvh.order[node.a as usize + k] as usize;
                        assert!(!seen[prim], "primitive {prim} appears twice");
                        seen[prim] = true;
                        for axis in 0..3 {
                            assert!(boxes[prim].min[axis] >= node.aabb_min[axis]);
                            assert!(boxes[prim].max[axis] <= node.aabb_max[axis]);
                        }
                    }
                }
            }
            assert!(seen.iter().all(|s| *s));
            check_containment(&bvh.nodes, 0);
        }
    }

    #[test]
    fn tlas_is_balanced_and_carries_kind_and_id() {
        let boxes = lattice(1000);
        let leaves: Vec<TlasLeaf> = boxes
            .iter()
            .enumerate()
            .map(|(i, aabb)| TlasLeaf {
                aabb: *aabb,
                kind: 1 + (i as u32 % 3),
                id: 7 * i as u32,
            })
            .collect();
        let nodes = build_tlas(&leaves);
        assert_eq!(nodes.len(), 2 * leaves.len() - 1);
        let bvh = Bvh {
            nodes: nodes.clone(),
            order: Vec::new(),
        };
        assert!(bvh.depth() <= 11, "depth {}", bvh.depth());
        let mut ids: Vec<u32> = nodes
            .iter()
            .filter(|node| node.is_leaf())
            .map(|node| {
                assert_eq!(node.leaf_kind(), 1 + (node.a / 7) % 3);
                node.a
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..1000).map(|i| 7 * i).collect::<Vec<u32>>());
        check_containment(&nodes, 0);
    }

    #[test]
    fn ray_span_matches_slab_geometry() {
        let aabb = Aabb::new([1.0, -1.0, -1.0], [3.0, 1.0, 1.0]);
        let span = aabb.ray_span([0.0; 3], [1.0, 0.0, 0.0], 0.0, 100.0).unwrap();
        assert_eq!(span, (1.0, 3.0));
        assert!(aabb.ray_span([0.0, 2.0, 0.0], [1.0, 0.0, 0.0], 0.0, 100.0).is_none());
        assert!(aabb.ray_span([0.0; 3], [1.0, 0.0, 0.0], 0.0, 0.5).is_none());
        assert!(build_bvh(&[], 4).nodes.is_empty());
    }
}
