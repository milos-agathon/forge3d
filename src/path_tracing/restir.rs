// src/path_tracing/restir.rs
// ReSTIR DI (Reservoir-based Spatio-Temporal Importance Resampling for Direct Illumination) implementation

use crate::path_tracing::alias_table::AliasTable;
use bytemuck::{Pod, Zeroable};

/// Light sample for ReSTIR
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightSample {
    /// Light position
    pub position: [f32; 3],
    /// Light index
    pub light_index: u32,
    /// Light direction (for directional lights)
    pub direction: [f32; 3],
    /// Light intensity/radiance
    pub intensity: f32,
    /// Light type (0=point, 1=directional, 2=area)
    pub light_type: u32,
    /// Additional light parameters
    pub params: [f32; 3],
}

/// Reservoir for weighted sampling
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Reservoir {
    /// Current sample
    pub sample: LightSample,
    /// Sum of weights (w_sum)
    pub w_sum: f32,
    /// Number of samples seen (M)
    pub m: u32,
    /// Weight of current sample (W = w_sum / (M * p_hat))
    pub weight: f32,
    /// Target PDF for the current sample
    pub target_pdf: f32,
}

impl Default for Reservoir {
    fn default() -> Self {
        Self {
            sample: LightSample {
                position: [0.0; 3],
                light_index: 0,
                direction: [0.0; 3],
                intensity: 0.0,
                light_type: 0,
                params: [0.0; 3],
            },
            w_sum: 0.0,
            m: 0,
            weight: 0.0,
            target_pdf: 0.0,
        }
    }
}

impl Reservoir {
    /// Create a new empty reservoir
    pub fn new() -> Self {
        Self::default()
    }

    /// Update reservoir with a new sample using reservoir sampling
    /// Returns true if the sample was accepted
    pub fn update(&mut self, sample: LightSample, weight: f32, random: f32) -> bool {
        self.w_sum += weight;
        self.m += 1;

        // Reservoir sampling: accept with probability weight / w_sum
        if random * self.w_sum <= weight {
            self.sample = sample;
            self.target_pdf = weight;
            true
        } else {
            false
        }
    }

    /// Finalize the reservoir by computing the final weight
    pub fn finalize(&mut self) {
        if self.w_sum > 0.0 && self.target_pdf > 0.0 {
            self.weight = self.w_sum / (self.m as f32 * self.target_pdf);
        } else {
            self.weight = 0.0;
        }
    }

    /// Combine this reservoir with another (for spatial/temporal reuse)
    pub fn combine(&mut self, other: &Reservoir, other_jacobian: f32, random: f32) {
        if other.m == 0 || other.weight == 0.0 {
            return;
        }

        // Calculate the weight for the other reservoir's sample in our context
        let other_contribution = other.target_pdf * other_jacobian * other.m as f32;

        self.w_sum += other_contribution;
        self.m += other.m;

        // Reservoir sampling: accept other's sample with probability other_contribution / w_sum
        if random * self.w_sum <= other_contribution {
            self.sample = other.sample;
            self.target_pdf = other.target_pdf * other_jacobian;
        }
    }

    /// Get the effective weight for shading
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    /// Check if the reservoir has a valid sample
    pub fn is_valid(&self) -> bool {
        self.m > 0 && self.weight > 0.0 && self.target_pdf > 0.0
    }

    /// Reset the reservoir
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// ReSTIR configuration parameters
#[derive(Clone, Debug)]
pub struct RestirConfig {
    /// Number of initial light candidates to consider
    pub initial_candidates: u32,
    /// Number of temporal neighbors to consider
    pub temporal_neighbors: u32,
    /// Number of spatial neighbors to consider
    pub spatial_neighbors: u32,
    /// Spatial radius for neighbor search
    pub spatial_radius: f32,
    /// Maximum temporal reuse age (in frames)
    pub max_temporal_age: u32,
    /// Bias correction mode
    pub bias_correction: bool,
}

impl Default for RestirConfig {
    fn default() -> Self {
        Self {
            initial_candidates: 32,
            temporal_neighbors: 1,
            spatial_neighbors: 4,
            spatial_radius: 16.0,
            max_temporal_age: 20,
            bias_correction: true,
        }
    }
}

/// ReSTIR DI implementation
pub struct RestirDI {
    /// Configuration parameters
    config: RestirConfig,
    /// Alias table for light sampling
    light_alias_table: Option<AliasTable>,
    /// Light data
    lights: Vec<LightSample>,
}

impl RestirDI {
    /// Create a new ReSTIR DI instance
    pub fn new(config: RestirConfig) -> Self {
        Self {
            config,
            light_alias_table: None,
            lights: Vec::new(),
        }
    }

    /// Set the lights and build alias table
    pub fn set_lights(&mut self, lights: Vec<LightSample>, light_weights: &[f32]) {
        self.lights = lights;

        if !light_weights.is_empty() {
            self.light_alias_table = Some(AliasTable::new(light_weights));
        }
    }

    /// Sample a light using the alias table
    pub fn sample_light(&self, u1: f32, u2: f32) -> Option<(usize, f32)> {
        self.light_alias_table
            .as_ref()
            .map(|table| table.sample(u1, u2))
    }

    /// Get the lights array
    pub fn lights(&self) -> &[LightSample] {
        &self.lights
    }

    /// Get the configuration
    pub fn config(&self) -> &RestirConfig {
        &self.config
    }

    /// Get the alias table entries for GPU upload
    pub fn alias_table_entries(&self) -> Option<&[crate::path_tracing::alias_table::AliasEntry]> {
        self.light_alias_table.as_ref().map(|table| table.entries())
    }

    /// Calculate target PDF for a light sample at a shading point
    pub fn target_pdf(
        &self,
        sample: &LightSample,
        shading_point: [f32; 3],
        normal: [f32; 3],
    ) -> f32 {
        // Simple geometric term calculation
        let light_dir = [
            sample.position[0] - shading_point[0],
            sample.position[1] - shading_point[1],
            sample.position[2] - shading_point[2],
        ];

        let dist_sq =
            light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2];
        if dist_sq <= 0.0 {
            return 0.0;
        }

        let dist = dist_sq.sqrt();
        let light_dir_norm = [
            light_dir[0] / dist,
            light_dir[1] / dist,
            light_dir[2] / dist,
        ];

        // Cosine term (N · L)
        let cos_theta = normal[0] * light_dir_norm[0]
            + normal[1] * light_dir_norm[1]
            + normal[2] * light_dir_norm[2];
        if cos_theta <= 0.0 {
            return 0.0;
        }

        // Simplified BRDF * G * Le / distance^2
        let geometric_term = cos_theta / dist_sq;
        sample.intensity * geometric_term
    }

    /// Perform initial sampling to fill a reservoir
    pub fn initial_sampling(
        &self,
        shading_point: [f32; 3],
        normal: [f32; 3],
        randoms: &[f32],
    ) -> Reservoir {
        let mut reservoir = Reservoir::new();

        if self.lights.is_empty() || randoms.len() < self.config.initial_candidates as usize * 3 {
            return reservoir;
        }

        for i in 0..self.config.initial_candidates {
            let idx = i as usize;
            if idx * 3 + 2 >= randoms.len() {
                break;
            }

            // Sample a light using alias table
            if let Some((light_idx, _pdf)) =
                self.sample_light(randoms[idx * 3], randoms[idx * 3 + 1])
            {
                if light_idx < self.lights.len() {
                    let light_sample = self.lights[light_idx];
                    let target_pdf = self.target_pdf(&light_sample, shading_point, normal);

                    if target_pdf > 0.0 {
                        reservoir.update(light_sample, target_pdf, randoms[idx * 3 + 2]);
                    }
                }
            }
        }

        reservoir.finalize();
        reservoir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_update() {
        let mut reservoir = Reservoir::new();

        let sample = LightSample {
            position: [1.0, 2.0, 3.0],
            light_index: 0,
            direction: [0.0, 0.0, 1.0],
            intensity: 1.0,
            light_type: 0,
            params: [0.0; 3],
        };

        let accepted = reservoir.update(sample, 1.0, 0.5);
        assert!(accepted);
        assert_eq!(reservoir.m, 1);
        assert_eq!(reservoir.w_sum, 1.0);
    }

    #[test]
    fn test_reservoir_combine() {
        let mut reservoir1 = Reservoir::new();
        let mut reservoir2 = Reservoir::new();

        let sample1 = LightSample {
            position: [1.0, 0.0, 0.0],
            light_index: 0,
            direction: [0.0, 0.0, 1.0],
            intensity: 1.0,
            light_type: 0,
            params: [0.0; 3],
        };

        let sample2 = LightSample {
            position: [2.0, 0.0, 0.0],
            light_index: 1,
            direction: [0.0, 0.0, 1.0],
            intensity: 2.0,
            light_type: 0,
            params: [0.0; 3],
        };

        reservoir1.update(sample1, 1.0, 0.5);
        reservoir1.finalize();

        reservoir2.update(sample2, 2.0, 0.5);
        reservoir2.finalize();

        reservoir1.combine(&reservoir2, 1.0, 0.5);
        assert!(reservoir1.m > 1);
    }

    #[test]
    fn test_restir_di_creation() {
        let config = RestirConfig::default();
        let restir = RestirDI::new(config.clone());

        assert_eq!(
            restir.config().initial_candidates,
            config.initial_candidates
        );
        assert!(restir.lights().is_empty());
    }
}
