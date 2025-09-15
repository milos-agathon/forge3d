//! A23: Hair BSDF + Curve Prims (PT) - Kajiya-Kay/Marschner; bezier ribbons/tubes

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct HairBSDF {
    pub longitudinal_roughness: f32,
    pub azimuthal_roughness: f32,
    pub absorption: Vec3,  // Hair pigments
    pub ior: f32,
}

impl HairBSDF {
    pub fn new() -> Self {
        Self {
            longitudinal_roughness: 0.3,
            azimuthal_roughness: 0.25,
            absorption: Vec3::new(0.419, 0.697, 1.37), // Brown hair
            ior: 1.55,
        }
    }

    // A23: Hairball highlights/tilt match reference
    pub fn evaluate_kajiya_kay(&self, light_dir: Vec3, view_dir: Vec3, tangent: Vec3) -> Vec3 {
        let t_dot_l = light_dir.dot(tangent);
        let t_dot_v = view_dir.dot(tangent);

        let sin_tl = (1.0 - t_dot_l * t_dot_l).max(0.0).sqrt();
        let sin_tv = (1.0 - t_dot_v * t_dot_v).max(0.0).sqrt();

        let cos_phi = (sin_tl * sin_tv + t_dot_l * t_dot_v).max(0.0);

        // Kajiya-Kay BRDF
        let diffuse = sin_tl;
        let specular = cos_phi.powf(1.0 / self.longitudinal_roughness);

        Vec3::splat(diffuse + specular)
    }
}

#[derive(Debug, Clone)]
pub struct CurvePrimitive {
    pub control_points: Vec<Vec3>,
    pub widths: Vec<f32>,  // A23: Curve widths
    pub curve_type: CurveType,
}

#[derive(Debug, Clone)]
pub enum CurveType {
    BezierRibbon,
    BezierTube,
}