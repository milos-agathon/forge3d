//! H12,H13: Graph rendering with nodes and edges
//! Separate pipelines for node points and edge lines

use crate::error::RenderError;
use crate::vector::api::GraphDef;
use crate::vector::data::PointInstance;
use crate::vector::layer::Layer;
use crate::vector::line::{LineCap, LineInstance, LineJoin, LineRenderer};
use crate::vector::point::PointRenderer;
use glam::Vec2;

/// Graph renderer with separate node and edge pipelines
pub struct GraphRenderer {
    node_renderer: PointRenderer,
    edge_renderer: LineRenderer,
}

/// Packed graph data for GPU rendering
#[derive(Debug, Clone)]
pub struct PackedGraph {
    pub node_instances: Vec<PointInstance>,
    pub edge_instances: Vec<LineInstance>,
    pub node_count: u32,
    pub edge_count: u32,
}

impl GraphRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let node_renderer = PointRenderer::new(device, target_format)?;
        let edge_renderer = LineRenderer::new(device, target_format)?;

        Ok(Self {
            node_renderer,
            edge_renderer,
        })
    }

    /// Pack graphs into contiguous arrays for GPU rendering
    pub fn pack_graphs(&self, graphs: &[GraphDef]) -> Result<PackedGraph, RenderError> {
        let mut node_instances = Vec::new();
        let mut edge_instances = Vec::new();

        for graph in graphs {
            // Validate graph structure
            if graph.nodes.is_empty() {
                return Err(RenderError::Upload(
                    "Graph must have at least one node".to_string(),
                ));
            }

            // Pack nodes as point instances
            for &node_pos in &graph.nodes {
                node_instances.push(PointInstance {
                    position: [node_pos.x, node_pos.y],
                    size: graph.node_style.point_size,
                    color: graph.node_style.fill_color,
                    rotation: 0.0,
                    uv_offset: [0.0, 0.0],
                    _pad: 0.0,
                });
            }

            // Pack edges as line instances
            let node_count = graph.nodes.len() as u32;
            for &(from_idx, to_idx) in &graph.edges {
                // Validate edge indices
                if from_idx >= node_count || to_idx >= node_count {
                    return Err(RenderError::Upload(format!(
                        "Edge ({}, {}) references invalid node indices (max: {})",
                        from_idx,
                        to_idx,
                        node_count - 1
                    )));
                }

                // Skip self-loops (edges to the same node)
                if from_idx == to_idx {
                    continue;
                }

                let start_pos = graph.nodes[from_idx as usize];
                let end_pos = graph.nodes[to_idx as usize];

                // Skip degenerate edges (duplicate positions)
                let edge_length = (end_pos - start_pos).length();
                if edge_length < 1e-6 {
                    continue;
                }

                edge_instances.push(LineInstance {
                    start_pos: [start_pos.x, start_pos.y],
                    end_pos: [end_pos.x, end_pos.y],
                    width: graph.edge_style.stroke_width,
                    color: graph.edge_style.stroke_color,
                    miter_limit: 4.0,
                    _pad: [0.0; 2],
                });
            }
        }

        let node_count = node_instances.len() as u32;
        let edge_count = edge_instances.len() as u32;

        Ok(PackedGraph {
            node_instances,
            edge_instances,
            node_count,
            edge_count,
        })
    }

    /// Upload graph data to GPU buffers
    pub fn upload_graph(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        packed_graph: &PackedGraph,
    ) -> Result<(), RenderError> {
        // Upload nodes via point renderer
        if !packed_graph.node_instances.is_empty() {
            self.node_renderer
                .upload_points(device, queue, &packed_graph.node_instances)?;
        }

        // Upload edges via line renderer
        if !packed_graph.edge_instances.is_empty() {
            self.edge_renderer
                .upload_lines(device, &packed_graph.edge_instances)?;
        }

        Ok(())
    }

    /// Render picking IDs for graph nodes and edges into R32Uint attachment.
    /// Mapping: [base_pick_id, base_pick_id + node_count) -> node indices,
    ///          [base_pick_id + node_count, base_pick_id + node_count + edge_count) -> edge indices
    pub fn render_pick<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
        packed_graph: &PackedGraph,
        base_pick_id: u32,
    ) -> Result<(), RenderError> {
        // Edges pick IDs start after nodes
        if packed_graph.node_count > 0 {
            self.node_renderer.render_pick(
                render_pass,
                queue,
                transform,
                viewport_size,
                pixel_scale,
                packed_graph.node_count,
                base_pick_id,
            )?;
        }

        if packed_graph.edge_count > 0 {
            let edge_base = base_pick_id + packed_graph.node_count;
            self.edge_renderer.render_pick(
                render_pass,
                queue,
                transform,
                viewport_size,
                packed_graph.edge_count,
                edge_base,
            )?;
        }

        Ok(())
    }

    /// Render nodes and edges into weighted OIT accumulation targets (MRT)
    pub fn render_oit<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        packed_graph: &PackedGraph,
    ) -> Result<(), RenderError> {
        // Render edges first into OIT buffers
        if packed_graph.edge_count > 0 {
            self.edge_renderer.render_oit(
                render_pass,
                queue,
                transform,
                viewport_size,
                packed_graph.edge_count,
                LineCap::Round,
                LineJoin::Round,
                2.0,
            )?;
        }

        // Render nodes into OIT buffers (use pixel_scale=1.0)
        if packed_graph.node_count > 0 {
            self.node_renderer.render_oit(
                render_pass,
                queue,
                transform,
                viewport_size,
                1.0,
                packed_graph.node_count,
            )?;
        }

        Ok(())
    }

    /// Render graph with nodes and edges
    /// Edges are rendered first (behind nodes) for proper layering
    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        packed_graph: &PackedGraph,
    ) -> Result<(), RenderError> {
        // Render edges first (background layer)
        if packed_graph.edge_count > 0 {
            self.edge_renderer.render(
                render_pass,
                queue,
                transform,
                viewport_size,
                packed_graph.edge_count,
                LineCap::Round,  // Use round caps for graph edges
                LineJoin::Round, // Use round joins for smooth connections
                2.0,             // Miter limit
            )?;
        }

        // Render nodes on top (foreground layer)
        if packed_graph.node_count > 0 {
            self.node_renderer.render(
                render_pass,
                queue,
                transform,
                viewport_size,
                1.0, // pixel_scale for point sizing
                packed_graph.node_count,
            )?;
        }

        Ok(())
    }

    /// Get layer for graph rendering (nodes over edges)
    pub fn layer() -> Layer {
        Layer::Vector
    }
}

/// Calculate graph layout using simple force-directed positioning
pub fn layout_force_directed(
    nodes: &mut [Vec2],
    edges: &[(u32, u32)],
    iterations: u32,
    damping: f32,
) -> Result<(), RenderError> {
    if nodes.is_empty() {
        return Ok(());
    }

    let node_count = nodes.len();
    let mut forces = vec![Vec2::ZERO; node_count];

    for _ in 0..iterations {
        // Clear forces
        forces.fill(Vec2::ZERO);

        // Repulsive forces between all node pairs
        for i in 0..node_count {
            for j in (i + 1)..node_count {
                let diff = nodes[i] - nodes[j];
                let dist = diff.length().max(0.1);
                let force = diff.normalize_or_zero() * (1.0 / (dist * dist));

                forces[i] += force;
                forces[j] -= force;
            }
        }

        // Attractive forces along edges
        for &(from_idx, to_idx) in edges {
            if from_idx as usize >= node_count || to_idx as usize >= node_count {
                continue;
            }

            let from = from_idx as usize;
            let to = to_idx as usize;

            let diff = nodes[to] - nodes[from];
            let dist = diff.length();
            if dist > 0.01 {
                let force = diff.normalize() * (dist * 0.1);
                forces[from] += force;
                forces[to] -= force;
            }
        }

        // Apply forces with damping
        for i in 0..node_count {
            nodes[i] += forces[i] * damping;
        }
    }

    Ok(())
}

/// Calculate bounding box for a graph
pub fn calculate_graph_bounds(graph: &GraphDef) -> Option<(Vec2, Vec2)> {
    if graph.nodes.is_empty() {
        return None;
    }

    let mut min_pos = graph.nodes[0];
    let mut max_pos = graph.nodes[0];

    for &node_pos in &graph.nodes[1..] {
        min_pos = min_pos.min(node_pos);
        max_pos = max_pos.max(node_pos);
    }

    // Expand bounds slightly for node sizes
    let node_radius = graph.node_style.point_size * 0.5;
    let expansion = Vec2::splat(node_radius);

    Some((min_pos - expansion, max_pos + expansion))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::VectorStyle;
    use glam::Vec2;

    #[test]
    fn test_pack_simple_graph() {
        let device = crate::gpu::create_device_for_test();
        let renderer = GraphRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let graph = GraphDef {
            nodes: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.5, 1.0),
            ],
            edges: vec![(0, 1), (1, 2), (2, 0)],
            node_style: VectorStyle {
                point_size: 5.0,
                fill_color: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
            edge_style: VectorStyle {
                stroke_width: 2.0,
                stroke_color: [0.0, 1.0, 0.0, 1.0],
                ..Default::default()
            },
        };

        let packed = renderer.pack_graphs(&[graph]).unwrap();

        assert_eq!(packed.node_count, 3);
        assert_eq!(packed.edge_count, 3);
        assert_eq!(packed.node_instances.len(), 3);
        assert_eq!(packed.edge_instances.len(), 3);

        // Check node instance data
        assert_eq!(packed.node_instances[0].size, 5.0);
        assert_eq!(packed.node_instances[0].color, [1.0, 0.0, 0.0, 1.0]);

        // Check edge instance data
        assert_eq!(packed.edge_instances[0].width, 2.0);
        assert_eq!(packed.edge_instances[0].color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_skip_invalid_edges() {
        let device = crate::gpu::create_device_for_test();
        let renderer = GraphRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let graph = GraphDef {
            nodes: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)],
            edges: vec![
                (0, 0), // Self-loop - should be skipped
                (0, 1), // Valid edge
            ],
            node_style: VectorStyle::default(),
            edge_style: VectorStyle::default(),
        };

        let packed = renderer.pack_graphs(&[graph]).unwrap();

        assert_eq!(packed.node_count, 2);
        assert_eq!(packed.edge_count, 1); // Self-loop was skipped
    }

    #[test]
    fn test_reject_invalid_edge_indices() {
        let device = crate::gpu::create_device_for_test();
        let renderer = GraphRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let invalid_graph = GraphDef {
            nodes: vec![Vec2::new(0.0, 0.0)],
            edges: vec![(0, 5)], // Invalid node index 5
            node_style: VectorStyle::default(),
            edge_style: VectorStyle::default(),
        };

        let result = renderer.pack_graphs(&[invalid_graph]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid node indices"));
    }

    #[test]
    fn test_force_directed_layout() {
        let mut nodes = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 0.0), // Start with overlapping nodes
            Vec2::new(0.0, 0.0),
        ];
        let edges = vec![(0, 1), (1, 2)];

        layout_force_directed(&mut nodes, &edges, 10, 0.1).unwrap();

        // Nodes should have spread out due to repulsive forces
        let dist_01 = (nodes[1] - nodes[0]).length();
        let dist_12 = (nodes[2] - nodes[1]).length();

        assert!(dist_01 > 0.01, "Nodes should repel each other");
        assert!(dist_12 > 0.01, "Nodes should repel each other");
    }

    #[test]
    fn test_graph_bounds() {
        let graph = GraphDef {
            nodes: vec![
                Vec2::new(-1.0, -2.0),
                Vec2::new(3.0, 1.0),
                Vec2::new(0.0, 4.0),
            ],
            edges: vec![],
            node_style: VectorStyle {
                point_size: 2.0,
                ..Default::default()
            },
            edge_style: VectorStyle::default(),
        };

        let (min_bound, max_bound) = calculate_graph_bounds(&graph).unwrap();

        // Should include node radius expansion
        assert!(min_bound.x <= -1.0 - 1.0); // -1.0 - radius
        assert!(min_bound.y <= -2.0 - 1.0); // -2.0 - radius
        assert!(max_bound.x >= 3.0 + 1.0); // 3.0 + radius
        assert!(max_bound.y >= 4.0 + 1.0); // 4.0 + radius
    }
}
