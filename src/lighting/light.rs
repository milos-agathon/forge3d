// src/lighting/light.rs
// Light type definitions with GPU-aligned layouts
// Split from types.rs for single-responsibility

use bytemuck::{Pod, Zeroable};

/// Light type enumeration (P0/P1)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional = 0,
    Point = 1,
    Spot = 2,
    Environment = 3,
    AreaRect = 4,
    AreaDisk = 5,
    AreaSphere = 6,
}

impl LightType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Light configuration (P1 extended)
/// GPU-aligned struct for SSBO upload (std430 layout)
/// Size: 80 bytes (5 vec4s), Alignment: 16 bytes
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Light {
    pub kind: u32,
    pub intensity: f32,
    pub range: f32,
    pub env_texture_index: u32,
    pub color: [f32; 3],
    pub _pad1: f32,
    pub pos_ws: [f32; 3],
    pub _pad2: f32,
    pub dir_ws: [f32; 3],
    pub _pad3: f32,
    pub cone_cos: [f32; 2],
    pub area_half: [f32; 2],
}

impl Default for Light {
    fn default() -> Self {
        Self {
            kind: LightType::Directional.as_u32(),
            intensity: 3.0,
            range: 1000.0,
            env_texture_index: 0,
            color: [1.0, 1.0, 1.0],
            _pad1: 0.0,
            pos_ws: [0.0, 0.0, 0.0],
            _pad2: 0.0,
            dir_ws: [0.0, -1.0, 0.0],
            _pad3: 0.0,
            cone_cos: [0.939, 0.819],
            area_half: [1.0, 1.0],
        }
    }
}

impl Light {
    pub fn directional(azimuth_deg: f32, elevation_deg: f32, intensity: f32, color: [f32; 3]) -> Self {
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();
        let x = el_rad.cos() * az_rad.sin();
        let y = -el_rad.sin();
        let z = -el_rad.cos() * az_rad.cos();
        Self {
            kind: LightType::Directional.as_u32(),
            intensity,
            range: 0.0,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: [0.0; 3],
            _pad2: 0.0,
            dir_ws: [x, y, z],
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [0.0, 0.0],
        }
    }

    pub fn point(position: [f32; 3], range: f32, intensity: f32, color: [f32; 3]) -> Self {
        Self {
            kind: LightType::Point.as_u32(),
            intensity,
            range,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: [0.0; 3],
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [0.0, 0.0],
        }
    }

    pub fn spot(
        position: [f32; 3],
        direction: [f32; 3],
        range: f32,
        inner_angle_deg: f32,
        outer_angle_deg: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        Self {
            kind: LightType::Spot.as_u32(),
            intensity,
            range,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: direction,
            _pad3: 0.0,
            cone_cos: [inner_angle_deg.to_radians().cos(), outer_angle_deg.to_radians().cos()],
            area_half: [0.0, 0.0],
        }
    }

    pub fn environment(intensity: f32, env_texture_index: u32) -> Self {
        Self {
            kind: LightType::Environment.as_u32(),
            intensity,
            range: 0.0,
            env_texture_index,
            color: [1.0, 1.0, 1.0],
            _pad1: 0.0,
            pos_ws: [0.0; 3],
            _pad2: 0.0,
            dir_ws: [0.0; 3],
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [0.0, 0.0],
        }
    }

    pub fn area_rect(
        position: [f32; 3],
        normal: [f32; 3],
        half_width: f32,
        half_height: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        Self {
            kind: LightType::AreaRect.as_u32(),
            intensity,
            range: 100.0,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: normal,
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [half_width, half_height],
        }
    }

    pub fn area_disk(
        position: [f32; 3],
        normal: [f32; 3],
        radius: f32,
        intensity: f32,
        color: [f32; 3],
    ) -> Self {
        Self {
            kind: LightType::AreaDisk.as_u32(),
            intensity,
            range: 100.0,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: normal,
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [radius, 0.0],
        }
    }

    pub fn area_sphere(position: [f32; 3], radius: f32, intensity: f32, color: [f32; 3]) -> Self {
        Self {
            kind: LightType::AreaSphere.as_u32(),
            intensity,
            range: 100.0,
            env_texture_index: 0,
            color,
            _pad1: 0.0,
            pos_ws: position,
            _pad2: 0.0,
            dir_ws: [0.0; 3],
            _pad3: 0.0,
            cone_cos: [1.0, 1.0],
            area_half: [radius, 0.0],
        }
    }
}
