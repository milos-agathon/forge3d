use glam::{Mat4, Quat, Vec3};

/// Unique identifier for a scene node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(super) usize);

/// Local transformation data for a scene node
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub dirty: bool,
}

impl Transform {
    pub fn new() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            dirty: true,
        }
    }

    pub fn new_with(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation,
            rotation,
            scale,
            dirty: true,
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn set_translation(&mut self, translation: Vec3) {
        self.translation = translation;
        self.mark_dirty();
    }

    pub fn set_rotation(&mut self, rotation: Quat) {
        self.rotation = rotation;
        self.mark_dirty();
    }

    pub fn set_scale(&mut self, scale: Vec3) {
        self.scale = scale;
        self.mark_dirty();
    }

    pub fn translate(&mut self, offset: Vec3) {
        self.translation += offset;
        self.mark_dirty();
    }

    pub fn rotate(&mut self, rotation: Quat) {
        self.rotation *= rotation;
        self.mark_dirty();
    }

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
    pub id: NodeId,
    pub name: String,
    pub local_transform: Transform,
    pub world_matrix: Mat4,
    pub world_dirty: bool,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
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

    pub fn mark_world_dirty(&mut self) {
        self.world_dirty = true;
    }

    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }
}

/// Traversal visitor trait for scene graph operations
pub trait SceneVisitor {
    type Error: std::error::Error + 'static;

    fn enter_node(&mut self, node: &SceneNode, world_matrix: &Mat4) -> Result<(), Self::Error>;

    fn exit_node(&mut self, node: &SceneNode) -> Result<(), Self::Error>;
}
