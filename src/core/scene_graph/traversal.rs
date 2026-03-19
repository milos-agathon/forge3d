use super::{NodeId, SceneGraph, SceneVisitor};
use crate::core::error::RenderResult;

impl SceneGraph {
    pub fn update_transforms(&mut self) -> RenderResult<()> {
        for &root_id in &self.roots.clone() {
            self.matrix_stack.reset();
            self.update_node_transforms(root_id)?;
        }
        Ok(())
    }

    pub fn traverse<V: SceneVisitor>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.update_transforms()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        for &root_id in &self.roots.clone() {
            self.matrix_stack.reset();
            self.traverse_node(root_id, visitor)?;
        }

        Ok(())
    }

    pub(super) fn collect_descendants(
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

    fn would_create_cycle(&self, parent_id: NodeId, child_id: NodeId) -> RenderResult<bool> {
        let mut current = Some(parent_id);
        while let Some(current_id) = current {
            if current_id == child_id {
                return Ok(true);
            }
            current = self.nodes.get(&current_id).and_then(|node| node.parent);
        }
        Ok(false)
    }

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

    fn update_node_transforms(&mut self, node_id: NodeId) -> RenderResult<()> {
        let (local_transform, world_dirty, children) = {
            if let Some(node) = self.nodes.get(&node_id) {
                (
                    node.local_transform.clone(),
                    node.world_dirty,
                    node.children.clone(),
                )
            } else {
                return Ok(());
            }
        };

        if world_dirty || local_transform.dirty {
            self.matrix_stack.mult(local_transform.to_matrix());
            let world_matrix = self.matrix_stack.top();

            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.world_matrix = world_matrix;
                node.world_dirty = false;
                node.local_transform.dirty = false;
            }
        } else {
            self.matrix_stack.mult(local_transform.to_matrix());
        }

        for child_id in children {
            self.matrix_stack.push()?;
            self.update_node_transforms(child_id)?;
            self.matrix_stack.pop()?;
        }

        Ok(())
    }

    fn traverse_node<V: SceneVisitor>(
        &mut self,
        node_id: NodeId,
        visitor: &mut V,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (world_matrix, visible, children) = {
            if let Some(node) = self.nodes.get(&node_id) {
                (node.world_matrix, node.visible, node.children.clone())
            } else {
                return Ok(());
            }
        };

        if !visible {
            return Ok(());
        }

        if let Some(node) = self.nodes.get(&node_id) {
            visitor
                .enter_node(node, &world_matrix)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        }

        for child_id in children {
            self.traverse_node(child_id, visitor)?;
        }

        if let Some(node) = self.nodes.get(&node_id) {
            visitor
                .exit_node(node)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        }

        Ok(())
    }
}
