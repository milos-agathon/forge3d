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
    /// Padding to 64 bytes to match WGSL std430 layout
    pub _pad: [u32; 4],
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
                _pad: [0; 4],
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
