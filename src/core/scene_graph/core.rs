use super::{NodeId, SceneNode};
use crate::core::error::{RenderError, RenderResult};
use crate::core::matrix_stack::MatrixStack;
use std::collections::{HashMap, HashSet};

/// Hierarchical scene graph container
#[derive(Debug)]
pub struct SceneGraph {
    pub(super) nodes: HashMap<NodeId, SceneNode>,
    pub(super) roots: HashSet<NodeId>,
    pub(super) next_id: usize,
    pub(super) matrix_stack: MatrixStack,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: HashSet::new(),
            next_id: 0,
            matrix_stack: MatrixStack::new(),
        }
    }

    pub fn create_node(&mut self, name: String) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        let node = SceneNode::new(id, name);
        self.nodes.insert(id, node);
        self.roots.insert(id);

        id
    }

    pub fn get_node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(&id)
    }

    pub fn remove_node(&mut self, id: NodeId) -> RenderResult<()> {
        let mut to_remove = Vec::new();
        self.collect_descendants(id, &mut to_remove)?;
        to_remove.push(id);

        if let Some(node) = self.nodes.get(&id) {
            if let Some(parent_id) = node.parent {
                if let Some(parent) = self.nodes.get_mut(&parent_id) {
                    parent.children.retain(|&child_id| child_id != id);
                }
            } else {
                self.roots.remove(&id);
            }
        }

        for &remove_id in &to_remove {
            self.nodes.remove(&remove_id);
            self.roots.remove(&remove_id);
        }

        Ok(())
    }

    pub fn add_child(&mut self, parent_id: NodeId, child_id: NodeId) -> RenderResult<()> {
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

        if self.would_create_cycle(parent_id, child_id)? {
            return Err(RenderError::render(
                "Adding child would create circular dependency",
            ));
        }

        let old_parent = self.nodes.get(&child_id).unwrap().parent;
        if let Some(old_parent_id) = old_parent {
            if let Some(old_parent_node) = self.nodes.get_mut(&old_parent_id) {
                old_parent_node.children.retain(|&id| id != child_id);
            }
        } else {
            self.roots.remove(&child_id);
        }

        if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
            parent_node.children.push(child_id);
        }

        if let Some(child_node) = self.nodes.get_mut(&child_id) {
            child_node.parent = Some(parent_id);
            child_node.mark_world_dirty();
        }

        self.mark_descendants_dirty(child_id)?;

        Ok(())
    }

    pub fn remove_child(&mut self, parent_id: NodeId, child_id: NodeId) -> RenderResult<()> {
        if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
            parent_node.children.retain(|&id| id != child_id);
        }

        if let Some(child_node) = self.nodes.get_mut(&child_id) {
            child_node.parent = None;
            child_node.mark_world_dirty();
        }

        self.roots.insert(child_id);
        self.mark_descendants_dirty(child_id)?;

        Ok(())
    }

    pub fn get_roots(&self) -> Vec<NodeId> {
        self.roots.iter().cloned().collect()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_count(&self) -> usize {
        self.roots.len()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.roots.clear();
        self.next_id = 0;
        self.matrix_stack.reset();
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}
