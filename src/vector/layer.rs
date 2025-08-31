//! H2: Layer base & render order
//! Deterministic draw ordering for vector/graph primitives

use std::cmp::Ordering;

/// Layer types for deterministic render ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layer {
    /// Background elements (filled polygons, terrain overlays)
    Background = 0,
    /// Main vector content (lines, strokes, polygon outlines) 
    Vector = 1,
    /// Point features and symbols
    Points = 2,
    /// Graph nodes (rendered above edges)
    GraphNodes = 3,
    /// Graph edges (rendered below nodes)
    GraphEdges = 4,
    /// Text and labels (highest priority)
    Labels = 5,
}

impl Layer {
    /// Get render order priority (lower values render first)
    pub fn render_order(&self) -> u32 {
        *self as u32
    }
    
    /// Get debug label for GPU debugging
    pub fn debug_label(&self) -> &'static str {
        match self {
            Layer::Background => "vf.Vector.Background",
            Layer::Vector => "vf.Vector.Vector", 
            Layer::Points => "vf.Vector.Points",
            Layer::GraphNodes => "vf.Vector.GraphNodes",
            Layer::GraphEdges => "vf.Vector.GraphEdges", 
            Layer::Labels => "vf.Vector.Labels",
        }
    }
}

impl PartialOrd for Layer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Layer {
    fn cmp(&self, other: &Self) -> Ordering {
        self.render_order().cmp(&other.render_order())
    }
}

/// Draw command with layer information for sorting
#[derive(Debug)]
pub struct LayeredDrawCmd {
    pub layer: Layer,
    pub draw_id: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
}

impl LayeredDrawCmd {
    pub fn new(layer: Layer, draw_id: u32, vertex_count: u32, instance_count: u32) -> Self {
        Self {
            layer,
            draw_id,
            vertex_count,
            instance_count,
        }
    }
}

/// Stable sort for layered draw commands
/// Ensures deterministic order across frames for the same input
pub fn sort_draw_commands(commands: &mut [LayeredDrawCmd]) {
    commands.sort_by(|a, b| {
        // Primary: layer order
        match a.layer.cmp(&b.layer) {
            Ordering::Equal => {
                // Secondary: draw_id for deterministic ordering within layer
                a.draw_id.cmp(&b.draw_id)
            }
            other => other,
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_ordering() {
        let mut layers = vec![
            Layer::Labels,
            Layer::Background, 
            Layer::GraphNodes,
            Layer::Vector,
            Layer::Points,
            Layer::GraphEdges,
        ];
        
        layers.sort();
        
        assert_eq!(layers, vec![
            Layer::Background,
            Layer::Vector, 
            Layer::Points,
            Layer::GraphNodes,
            Layer::GraphEdges,
            Layer::Labels,
        ]);
    }
    
    #[test]
    fn test_draw_command_sorting() {
        let mut commands = vec![
            LayeredDrawCmd::new(Layer::Points, 2, 100, 1),
            LayeredDrawCmd::new(Layer::Background, 1, 200, 1),
            LayeredDrawCmd::new(Layer::Points, 1, 150, 1), // Same layer, lower draw_id
            LayeredDrawCmd::new(Layer::Vector, 3, 300, 1),
        ];
        
        sort_draw_commands(&mut commands);
        
        // Should be sorted by layer first, then draw_id within layer
        assert_eq!(commands[0].layer, Layer::Background);
        assert_eq!(commands[0].draw_id, 1);
        
        assert_eq!(commands[1].layer, Layer::Vector);
        assert_eq!(commands[1].draw_id, 3);
        
        assert_eq!(commands[2].layer, Layer::Points);
        assert_eq!(commands[2].draw_id, 1); // Lower draw_id first
        
        assert_eq!(commands[3].layer, Layer::Points);
        assert_eq!(commands[3].draw_id, 2);
    }
    
    #[test]
    fn test_debug_labels() {
        assert_eq!(Layer::Background.debug_label(), "vf.Vector.Background");
        assert_eq!(Layer::GraphNodes.debug_label(), "vf.Vector.GraphNodes");
    }
}