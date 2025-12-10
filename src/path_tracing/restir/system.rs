use super::types::*;
use crate::path_tracing::alias_table::AliasTable;

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
