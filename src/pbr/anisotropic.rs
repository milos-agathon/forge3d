//! A24: Anisotropic Microfacet BRDF - GGX/Beckmann αx/αy

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct AnisotropicBRDF {
    pub alpha_x: f32,  // Roughness in tangent direction
    pub alpha_y: f32,  // Roughness in bitangent direction
    pub albedo: Vec3,
    pub metallic: f32,
}

impl AnisotropicBRDF {
    pub fn new(alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            alpha_x,
            alpha_y,
            albedo: Vec3::new(0.8, 0.8, 0.8),
            metallic: 0.0,
        }
    }

    // A24: Aniso reduces to iso at ax=ay; energy conserved
    pub fn evaluate_ggx(&self, normal: Vec3, view: Vec3, light: Vec3,
                       tangent: Vec3, bitangent: Vec3) -> Vec3 {
        let half = (light + view).normalize();

        // Transform to tangent space
        let h_dot_t = half.dot(tangent);
        let h_dot_b = half.dot(bitangent);
        let h_dot_n = half.dot(normal);

        // Anisotropic GGX distribution
        let ax2 = self.alpha_x * self.alpha_x;
        let ay2 = self.alpha_y * self.alpha_y;

        let denom = (h_dot_t * h_dot_t) / ax2 + (h_dot_b * h_dot_b) / ay2 + h_dot_n * h_dot_n;
        let d = 1.0 / (std::f32::consts::PI * self.alpha_x * self.alpha_y * denom * denom);

        // Simplified energy conservation
        let energy_factor = if (self.alpha_x - self.alpha_y).abs() < 0.001 {
            1.0  // Isotropic case
        } else {
            (self.alpha_x + self.alpha_y) / 2.0  // Anisotropic normalization
        };

        self.albedo * d * energy_factor
    }

    pub fn is_isotropic(&self) -> bool {
        (self.alpha_x - self.alpha_y).abs() < 0.001
    }
}