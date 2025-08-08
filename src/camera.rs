// Simple camera module placeholder for T2.2 implementation
// This provides basic camera utilities that might be needed

pub struct Camera {
    pub position: glam::Vec3,
    pub target: glam::Vec3,
    pub up: glam::Vec3,
}

impl Camera {
    pub fn new(position: glam::Vec3, target: glam::Vec3, up: glam::Vec3) -> Self {
        Self { position, target, up }
    }
    
    pub fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.position, self.target, self.up)
    }
}