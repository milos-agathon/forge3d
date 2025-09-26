//! Hierarchical scene graph system
//!
//! Provides a tree-based scene representation where nodes can have parent-child
//! relationships with automatic transform inheritance.

use crate::core::matrix_stack::MatrixStack;
use crate::error::{RenderError, RenderResult};
use glam::{Mat4, Quat, Vec3};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a scene node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

/// Local transformation data for a scene node
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    /// Local position relative to parent
    pub translation: Vec3,
    /// Local rotation relative to parent
    pub rotation: Quat,
    /// Local scale relative to parent
    pub scale: Vec3,
    /// Whether transforms are dirty and need recalculation
    pub dirty: bool,
}

impl Transform {
    /// Create a new identity transform
    pub fn new() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            dirty: true,
        }
    }

    /// Create transform with specific values
    pub fn new_with(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation,
            rotation,
            scale,
            dirty: true,
        }
    }

    /// Convert to a 4x4 transformation matrix
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Mark transform as dirty
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Set translation and mark dirty
    pub fn set_translation(&mut self, translation: Vec3) {
        self.translation = translation;
        self.mark_dirty();
    }

    /// Set rotation and mark dirty
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.rotation = rotation;
        self.mark_dirty();
    }

    /// Set scale and mark dirty
    pub fn set_scale(&mut self, scale: Vec3) {
        self.scale = scale;
        self.mark_dirty();
    }

    /// Translate by offset
    pub fn translate(&mut self, offset: Vec3) {
        self.translation += offset;
        self.mark_dirty();
    }

    /// Rotate by additional rotation
    pub fn rotate(&mut self, rotation: Quat) {
        self.rotation = self.rotation * rotation;
        self.mark_dirty();
    }

    /// Scale by additional factor
    pub fn scale_by(&mut self, factor: Vec3) {
        self.scale *= factor;
        self.mark_dirty();
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

/// Scene node with transform and hierarchy information
#[derive(Debug)]
pub struct SceneNode {
    /// Unique identifier
    pub id: NodeId,
    /// Human-readable name for debugging
    pub name: String,
    /// Local transformation relative to parent
    pub local_transform: Transform,
    /// Cached world transformation matrix
    pub world_matrix: Mat4,
    /// Whether world matrix is up to date
    pub world_dirty: bool,
    /// Parent node ID (None for root nodes)
    pub parent: Option<NodeId>,
    /// Child node IDs
    pub children: Vec<NodeId>,
    /// Whether this node is visible
    pub visible: bool,
}

impl Clone for SceneNode {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            name: self.name.clone(),
            local_transform: self.local_transform.clone(),
            world_matrix: self.world_matrix,
            world_dirty: self.world_dirty,
            parent: self.parent,
            children: self.children.clone(),
            visible: self.visible,
        }
    }
}

impl SceneNode {
    /// Create a new scene node
    pub fn new(id: NodeId, name: String) -> Self {
        Self {
            id,
            name,
            local_transform: Transform::new(),
            world_matrix: Mat4::IDENTITY,
            world_dirty: true,
            parent: None,
            children: Vec::new(),
            visible: true,
        }
    }

    /// Mark this node's world transform as dirty
    pub fn mark_world_dirty(&mut self) {
        self.world_dirty = true;
    }

    /// Check if the node has children
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Get number of children
    pub fn child_count(&self) -> usize {
        self.children.len()
    }
}

/// Traversal visitor trait for scene graph operations
pub trait SceneVisitor {
    /// Error type for visitor operations
    type Error: std::error::Error + 'static;

    /// Called when entering a node during traversal
    fn enter_node(&mut self, node: &SceneNode, world_matrix: &Mat4) -> Result<(), Self::Error>;

    /// Called when exiting a node during traversal  
    fn exit_node(&mut self, node: &SceneNode) -> Result<(), Self::Error>;
}

/// Hierarchical scene graph container
#[derive(Debug)]
pub struct SceneGraph {
    /// All nodes in the scene indexed by ID
    nodes: HashMap<NodeId, SceneNode>,
    /// Root node IDs (nodes without parents)
    roots: HashSet<NodeId>,
    /// Next node ID to assign
    next_id: usize,
    /// Matrix stack for transform calculations
    matrix_stack: MatrixStack,
}

impl SceneGraph {
    /// Create a new empty scene graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: HashSet::new(),
            next_id: 0,
            matrix_stack: MatrixStack::new(),
        }
    }

    /// Create a new scene node
    pub fn create_node(&mut self, name: String) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        let node = SceneNode::new(id, name);
        self.nodes.insert(id, node);
        self.roots.insert(id); // Initially a root node

        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(&id)
    }

    /// Remove a node and all its children
    pub fn remove_node(&mut self, id: NodeId) -> RenderResult<()> {
        // First collect all descendants to remove
        let mut to_remove = Vec::new();
        self.collect_descendants(id, &mut to_remove)?;
        to_remove.push(id);

        // Remove from parent's children list
        if let Some(node) = self.nodes.get(&id) {
            if let Some(parent_id) = node.parent {
                if let Some(parent) = self.nodes.get_mut(&parent_id) {
                    parent.children.retain(|&child_id| child_id != id);
                }
            } else {
                // Remove from roots if it was a root node
                self.roots.remove(&id);
            }
        }

        // Remove all nodes
        for &remove_id in &to_remove {
            self.nodes.remove(&remove_id);
            self.roots.remove(&remove_id);
        }

        Ok(())
    }

    /// Add a child to a parent node
    pub fn add_child(&mut self, parent_id: NodeId, child_id: NodeId) -> RenderResult<()> {
        // Check if nodes exist
        if !self.nodes.contains_key(&parent_id) {
            return Err(RenderError::render(&format!(
                "Parent node {:?} not found",
                parent_id
            )));
        }
        if !self.nodes.contains_key(&child_id) {
            return Err(RenderError::render(&format!(
                "Child node {:?} not found",
                child_id
            )));
        }

        // Check for circular dependency
        if self.would_create_cycle(parent_id, child_id)? {
            return Err(RenderError::render(
                "Adding child would create circular dependency",
            ));
        }

        // Remove child from current parent if it has one
        let old_parent = self.nodes.get(&child_id).unwrap().parent;
        if let Some(old_parent_id) = old_parent {
            if let Some(old_parent_node) = self.nodes.get_mut(&old_parent_id) {
                old_parent_node.children.retain(|&id| id != child_id);
            }
        } else {
            // Remove from roots since it's getting a parent
            self.roots.remove(&child_id);
        }

        // Update parent to add child
        if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
            parent_node.children.push(child_id);
        }

        // Update child to set parent
        if let Some(child_node) = self.nodes.get_mut(&child_id) {
            child_node.parent = Some(parent_id);
            child_node.mark_world_dirty();
        }

        // Mark all descendants as dirty
        self.mark_descendants_dirty(child_id)?;

        Ok(())
    }

    /// Remove a child from its parent
    pub fn remove_child(&mut self, parent_id: NodeId, child_id: NodeId) -> RenderResult<()> {
        // Remove from parent's children list
        if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
            parent_node.children.retain(|&id| id != child_id);
        }

        // Update child to remove parent
        if let Some(child_node) = self.nodes.get_mut(&child_id) {
            child_node.parent = None;
            child_node.mark_world_dirty();
        }

        // Add to roots since it no longer has a parent
        self.roots.insert(child_id);

        // Mark all descendants as dirty
        self.mark_descendants_dirty(child_id)?;

        Ok(())
    }

    /// Get all root nodes
    pub fn get_roots(&self) -> Vec<NodeId> {
        self.roots.iter().cloned().collect()
    }

    /// Update world matrices for all dirty nodes
    pub fn update_transforms(&mut self) -> RenderResult<()> {
        // Process each root and its subtree
        for &root_id in &self.roots.clone() {
            self.matrix_stack.reset();
            self.update_node_transforms(root_id)?;
        }
        Ok(())
    }

    /// Traverse the scene graph with a visitor
    pub fn traverse<V: SceneVisitor>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Update transforms first
        self.update_transforms()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        // Process each root and its subtree
        for &root_id in &self.roots.clone() {
            self.matrix_stack.reset();
            self.traverse_node(root_id, visitor)?;
        }

        Ok(())
    }

    /// Get the total number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of root nodes
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }

    /// Clear all nodes
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.roots.clear();
        self.next_id = 0;
        self.matrix_stack.reset();
    }

    /// Helper: Collect all descendant IDs
    fn collect_descendants(
        &self,
        node_id: NodeId,
        descendants: &mut Vec<NodeId>,
    ) -> RenderResult<()> {
        if let Some(node) = self.nodes.get(&node_id) {
            for &child_id in &node.children {
                descendants.push(child_id);
                self.collect_descendants(child_id, descendants)?;
            }
        }
        Ok(())
    }

    /// Helper: Check if adding child to parent would create a cycle
    fn would_create_cycle(&self, parent_id: NodeId, child_id: NodeId) -> RenderResult<bool> {
        // Walk up from parent to see if we encounter child
        let mut current = Some(parent_id);
        while let Some(current_id) = current {
            if current_id == child_id {
                return Ok(true);
            }
            current = self.nodes.get(&current_id).and_then(|node| node.parent);
        }
        Ok(false)
    }

    /// Helper: Mark a node and all descendants as dirty
    fn mark_descendants_dirty(&mut self, node_id: NodeId) -> RenderResult<()> {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.mark_world_dirty();
            let children = node.children.clone();
            for child_id in children {
                self.mark_descendants_dirty(child_id)?;
            }
        }
        Ok(())
    }

    /// Helper: Update transforms for a node and its subtree
    fn update_node_transforms(&mut self, node_id: NodeId) -> RenderResult<()> {
        // Get the node (we need to split borrows)
        let (local_transform, world_dirty, children) = {
            if let Some(node) = self.nodes.get(&node_id) {
                (
                    node.local_transform.clone(),
                    node.world_dirty,
                    node.children.clone(),
                )
            } else {
                return Ok(()); // Node doesn't exist
            }
        };

        // Update world matrix if dirty
        if world_dirty || local_transform.dirty {
            // Apply local transform to current matrix
            self.matrix_stack.mult(local_transform.to_matrix());
            let world_matrix = self.matrix_stack.top();

            // Update the node
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.world_matrix = world_matrix;
                node.world_dirty = false;
                node.local_transform.dirty = false;
            }
        } else {
            // Still need to apply transform to matrix stack for children
            self.matrix_stack.mult(local_transform.to_matrix());
        }

        // Process children
        for child_id in children {
            self.matrix_stack.push()?;
            self.update_node_transforms(child_id)?;
            self.matrix_stack.pop()?;
        }

        Ok(())
    }

    /// Helper: Traverse a node and its subtree with visitor
    fn traverse_node<V: SceneVisitor>(
        &mut self,
        node_id: NodeId,
        visitor: &mut V,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get node data (split borrows)
        let (world_matrix, visible, children) = {
            if let Some(node) = self.nodes.get(&node_id) {
                (node.world_matrix, node.visible, node.children.clone())
            } else {
                return Ok(()); // Node doesn't exist
            }
        };

        if !visible {
            return Ok(()); // Skip invisible nodes and their children
        }

        // Enter the node
        if let Some(node) = self.nodes.get(&node_id) {
            visitor
                .enter_node(node, &world_matrix)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        }

        // Visit children
        for child_id in children {
            self.traverse_node(child_id, visitor)?;
        }

        // Exit the node
        if let Some(node) = self.nodes.get(&node_id) {
            visitor
                .exit_node(node)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        }

        Ok(())
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_graph_basic_operations() {
        let mut graph = SceneGraph::new();

        // Create nodes
        let root = graph.create_node("root".to_string());
        let child1 = graph.create_node("child1".to_string());
        let child2 = graph.create_node("child2".to_string());

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.root_count(), 3); // All nodes are initially roots

        // Add parent-child relationships
        graph.add_child(root, child1).unwrap();
        graph.add_child(root, child2).unwrap();

        assert_eq!(graph.root_count(), 1); // Only root should be a root now

        let root_node = graph.get_node(root).unwrap();
        assert_eq!(root_node.children.len(), 2);

        let child1_node = graph.get_node(child1).unwrap();
        assert_eq!(child1_node.parent, Some(root));
    }

    #[test]
    fn test_scene_graph_transform_inheritance() {
        let mut graph = SceneGraph::new();

        let root = graph.create_node("root".to_string());
        let child = graph.create_node("child".to_string());

        // Set transforms
        graph
            .get_node_mut(root)
            .unwrap()
            .local_transform
            .set_translation(Vec3::new(1.0, 0.0, 0.0));
        graph
            .get_node_mut(child)
            .unwrap()
            .local_transform
            .set_translation(Vec3::new(0.0, 1.0, 0.0));

        // Add hierarchy
        graph.add_child(root, child).unwrap();

        // Update transforms
        graph.update_transforms().unwrap();

        // Check world matrices
        let root_node = graph.get_node(root).unwrap();
        let child_node = graph.get_node(child).unwrap();

        // Root should have its local transform
        assert_eq!(
            root_node.world_matrix.w_axis.xyz(),
            Vec3::new(1.0, 0.0, 0.0)
        );

        // Child should have combined transform (parent + local)
        assert_eq!(
            child_node.world_matrix.w_axis.xyz(),
            Vec3::new(1.0, 1.0, 0.0)
        );
    }

    #[test]
    fn test_scene_graph_cycle_detection() {
        let mut graph = SceneGraph::new();

        let node1 = graph.create_node("node1".to_string());
        let node2 = graph.create_node("node2".to_string());
        let node3 = graph.create_node("node3".to_string());

        // Create a chain: node1 -> node2 -> node3
        graph.add_child(node1, node2).unwrap();
        graph.add_child(node2, node3).unwrap();

        // Attempting to make node1 a child of node3 should fail (cycle)
        assert!(graph.add_child(node3, node1).is_err());
    }

    #[test]
    fn test_scene_graph_removal() {
        let mut graph = SceneGraph::new();

        let root = graph.create_node("root".to_string());
        let child1 = graph.create_node("child1".to_string());
        let grandchild = graph.create_node("grandchild".to_string());

        graph.add_child(root, child1).unwrap();
        graph.add_child(child1, grandchild).unwrap();

        assert_eq!(graph.node_count(), 3);

        // Remove child1 (should also remove grandchild)
        graph.remove_node(child1).unwrap();

        assert_eq!(graph.node_count(), 1); // Only root should remain
        assert!(!graph.nodes.contains_key(&child1));
        assert!(!graph.nodes.contains_key(&grandchild));
    }
}
