// src/shadows/state.rs
// CPU fallback state for cascaded shadow maps configuration and validation
// Exists to keep Python bindings and tests working without GPU dependencies
// RELEVANT FILES: src/lib.rs, python/forge3d/lighting.py, tests/test_b4_csm.py, src/shadows/csm.rs

use glam::Vec3;

const MIN_CASCADE_COUNT: u32 = 2;
const MAX_CASCADE_COUNT: u32 = 4;
const MIN_SHADOW_MAP_SIZE: u32 = 512;
const MAX_SHADOW_MAP_SIZE: u32 = 8192;
const PRACTICAL_SPLIT_LAMBDA: f32 = 0.75;
const MIN_NEAR_PLANE: f32 = 0.05;
const MIN_DISTANCE_EPS: f32 = 0.001;
const TEXEL_EPS: f32 = 1.0e-6;

/// CPU representation of the CSM configuration used by the bindings layer
#[derive(Debug, Clone)]
pub struct CpuCsmConfig {
    pub cascade_count: u32,
    pub shadow_map_size: u32,
    pub max_shadow_distance: f32,
    pub pcf_kernel_size: u32,
    pub depth_bias: f32,
    pub slope_bias: f32,
    pub peter_panning_offset: f32,
    pub enable_evsm: bool,
    pub debug_mode: u32,
}

impl CpuCsmConfig {
    pub fn new(
        cascade_count: u32,
        shadow_map_size: u32,
        max_shadow_distance: f32,
        pcf_kernel_size: u32,
        depth_bias: f32,
        slope_bias: f32,
        peter_panning_offset: f32,
        enable_evsm: bool,
        debug_mode: u32,
    ) -> Result<Self, String> {
        let config = Self {
            cascade_count,
            shadow_map_size,
            max_shadow_distance,
            pcf_kernel_size,
            depth_bias,
            slope_bias,
            peter_panning_offset,
            enable_evsm,
            debug_mode,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.cascade_count < MIN_CASCADE_COUNT || self.cascade_count > MAX_CASCADE_COUNT {
            return Err(format!(
                "cascade_count must be between {} and {}",
                MIN_CASCADE_COUNT, MAX_CASCADE_COUNT
            ));
        }

        if self.shadow_map_size < MIN_SHADOW_MAP_SIZE || self.shadow_map_size > MAX_SHADOW_MAP_SIZE
        {
            return Err(format!(
                "shadow_map_size must be between {} and {}",
                MIN_SHADOW_MAP_SIZE, MAX_SHADOW_MAP_SIZE
            ));
        }

        if !matches!(self.pcf_kernel_size, 1 | 3 | 5 | 7) {
            return Err("pcf_kernel_size must be 1, 3, 5, or 7".to_string());
        }

        if self.max_shadow_distance <= 0.0 {
            return Err("max_shadow_distance must be positive".to_string());
        }

        if self.depth_bias < 0.0 {
            return Err("depth_bias must be non-negative".to_string());
        }

        if self.slope_bias < 0.0 {
            return Err("slope_bias must be non-negative".to_string());
        }

        if self.peter_panning_offset < 0.0 {
            return Err("peter_panning_offset must be non-negative".to_string());
        }

        if self.debug_mode > 2 {
            return Err("debug_mode must be 0, 1, or 2".to_string());
        }

        Ok(())
    }
}

impl Default for CpuCsmConfig {
    fn default() -> Self {
        Self {
            cascade_count: 3,
            shadow_map_size: 2048,
            max_shadow_distance: 200.0,
            pcf_kernel_size: 3,
            depth_bias: 0.005,
            slope_bias: 0.01,
            peter_panning_offset: 0.001,
            enable_evsm: false,
            debug_mode: 0,
        }
    }
}

/// Simple cascade metadata used by debug queries
#[derive(Debug, Clone, Copy)]
pub struct CascadeInfo {
    pub near: f32,
    pub far: f32,
    pub texel_size: f32,
}

/// CPU-resident state of the cascaded shadow map system
#[derive(Debug, Clone)]
pub struct CpuCsmState {
    enabled: bool,
    config: CpuCsmConfig,
    light_direction: Vec3,
    camera_near: f32,
    camera_far: f32,
    cascades: Vec<CascadeInfo>,
}

impl Default for CpuCsmState {
    fn default() -> Self {
        let config = CpuCsmConfig::default();
        let mut state = Self {
            enabled: false,
            light_direction: Vec3::new(0.0, -1.0, 0.0),
            camera_near: 0.1,
            camera_far: config.max_shadow_distance.max(0.1 + MIN_DISTANCE_EPS),
            cascades: Vec::new(),
            config,
        };
        state.rebuild_cascades();
        state
    }
}

impl CpuCsmState {
    pub fn config(&self) -> &CpuCsmConfig {
        &self.config
    }

    pub fn apply_config(&mut self, config: CpuCsmConfig) -> Result<(), String> {
        config.validate()?;
        self.config = config;
        self.camera_far = self
            .config
            .max_shadow_distance
            .max(self.camera_near + MIN_DISTANCE_EPS);
        self.rebuild_cascades();
        Ok(())
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_light_direction(&mut self, direction: [f32; 3]) {
        let v = Vec3::from(direction);
        self.light_direction = if v.length_squared() <= 1.0e-6 {
            Vec3::new(0.0, -1.0, 0.0)
        } else {
            v.normalize()
        };
    }

    pub fn set_pcf_kernel(&mut self, kernel_size: u32) -> Result<(), String> {
        if !matches!(kernel_size, 1 | 3 | 5 | 7) {
            return Err("pcf_kernel_size must be 1, 3, 5, or 7".to_string());
        }
        self.config.pcf_kernel_size = kernel_size;
        Ok(())
    }

    pub fn set_bias_params(
        &mut self,
        depth_bias: f32,
        slope_bias: f32,
        peter_panning_offset: f32,
    ) -> Result<(), String> {
        if depth_bias < 0.0 {
            return Err("depth_bias must be non-negative".to_string());
        }
        if slope_bias < 0.0 {
            return Err("slope_bias must be non-negative".to_string());
        }
        if peter_panning_offset < 0.0 {
            return Err("peter_panning_offset must be non-negative".to_string());
        }
        self.config.depth_bias = depth_bias;
        self.config.slope_bias = slope_bias;
        self.config.peter_panning_offset = peter_panning_offset;
        Ok(())
    }

    pub fn set_debug_mode(&mut self, mode: u32) -> Result<(), String> {
        if mode > 2 {
            return Err("debug_mode must be 0, 1, or 2".to_string());
        }
        self.config.debug_mode = mode;
        Ok(())
    }

    pub fn set_evsm_enabled(&mut self, enable: bool) {
        self.config.enable_evsm = enable;
    }

    pub fn cascade_info(&self) -> Vec<(f32, f32, f32)> {
        self.cascades
            .iter()
            .map(|c| (c.near, c.far, c.texel_size))
            .collect()
    }

    pub fn validate_peter_panning(&self) -> bool {
        self.config.depth_bias > 1.0e-4 && self.config.peter_panning_offset > 1.0e-4
    }

    fn rebuild_cascades(&mut self) {
        let near = self.camera_near.max(MIN_NEAR_PLANE);
        let far = self.config.max_shadow_distance.max(near + MIN_DISTANCE_EPS);
        let splits = practical_split_scheme(near, far, self.config.cascade_count);
        let inv_map_size = 1.0 / self.config.shadow_map_size as f32;

        self.cascades = splits
            .windows(2)
            .map(|pair| {
                let near = pair[0];
                let far = pair[1];
                let extent = (far - near).abs().max(MIN_DISTANCE_EPS);
                let texel_size = (extent * inv_map_size).max(TEXEL_EPS);
                CascadeInfo {
                    near,
                    far,
                    texel_size,
                }
            })
            .collect();
    }
}

fn practical_split_scheme(near: f32, far: f32, cascade_count: u32) -> Vec<f32> {
    let count = cascade_count.max(MIN_CASCADE_COUNT);
    let mut splits = Vec::with_capacity(count as usize + 1);

    let safe_near = near.max(MIN_NEAR_PLANE);
    let safe_far = far.max(safe_near + MIN_DISTANCE_EPS);
    let range = safe_far - safe_near;
    let ratio = (safe_far / safe_near).max(1.0001);

    splits.push(safe_near);

    for i in 1..count {
        let t = i as f32 / count as f32;
        let uniform_split = safe_near + range * t;
        let log_split = safe_near * ratio.powf(t);
        let blended =
            PRACTICAL_SPLIT_LAMBDA * log_split + (1.0 - PRACTICAL_SPLIT_LAMBDA) * uniform_split;
        splits.push(blended);
    }

    splits.push(safe_far);
    splits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_monotonic() {
        let splits = practical_split_scheme(0.1, 100.0, 4);
        assert_eq!(splits.len(), 5);
        for i in 1..splits.len() {
            assert!(splits[i] > splits[i - 1]);
        }
    }

    #[test]
    fn state_rebuilds_on_config_change() {
        let mut state = CpuCsmState::default();
        let original = state.cascade_info();
        assert_eq!(original.len(), state.config().cascade_count as usize);

        let new_config =
            CpuCsmConfig::new(4, 4096, 300.0, 5, 0.003, 0.007, 0.0005, true, 1).unwrap();
        state.apply_config(new_config).unwrap();
        let updated = state.cascade_info();
        assert_eq!(updated.len(), 4);
        assert!(updated[0].0 >= 0.05);
    }

    #[test]
    fn bias_validation_behaves() {
        let mut state = CpuCsmState::default();
        assert!(state.validate_peter_panning());
        state.set_bias_params(0.0, 0.01, 0.00001).unwrap();
        assert!(!state.validate_peter_panning());
    }
}
