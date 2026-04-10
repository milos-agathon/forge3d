// B16-BEGIN: Light data structures
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLight {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    pub _pad1: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpotLight {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub direction: [f32; 3],
    pub _pad1: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    pub inner_cone: f32,
    pub outer_cone: f32,
    pub _pad2: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            radius: 10.0,
            _pad1: [0.0; 3],
        }
    }
}

// [methods moved into impl TerrainSpike]

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            direction: [0.0, -1.0, 0.0], // Point downward
            _pad1: 0.0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            radius: 10.0,
            inner_cone: 0.2, // ~11.5 degrees
            outer_cone: 0.4, // ~23 degrees
            _pad2: 0.0,
        }
    }
}
// B16-END: Light data structures
