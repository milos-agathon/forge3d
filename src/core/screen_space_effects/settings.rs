/// SSAO/GTAO settings
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoSettings {
    pub radius: f32,
    pub intensity: f32,
    pub bias: f32,
    pub num_samples: u32,
    pub technique: u32,   // 0=SSAO, 1=GTAO
    pub frame_index: u32, // frame counter for noise
    pub inv_resolution: [f32; 2],
    pub proj_scale: f32, // 0.5 * height / tan(fov/2) = 0.5 * height * P[1][1]
    pub ao_min: f32,     // minimum AO value to prevent full black (default 0.35)
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            radius: 0.5,
            intensity: 1.5, // Higher default intensity for stronger AO effect
            bias: 0.025,
            num_samples: 16,
            technique: 0,
            frame_index: 0,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            proj_scale: 0.5 * 1080.0 * (1.0 / (45.0_f32.to_radians() * 0.5).tan()),
            ao_min: 0.05, // Allow stronger crease darkening while keeping a floor
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SsaoTemporalParamsUniform {
    pub(crate) temporal_alpha: f32,
    pub(crate) _pad: [f32; 7],
}

/// SSGI settings
/// Note: Size must be 80 bytes to match WGSL std140 layout where vec3<u32> is aligned to 16 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsgiSettings {
    pub radius: f32,
    pub intensity: f32,
    pub num_steps: u32,
    pub step_size: f32,
    pub inv_resolution: [f32; 2],
    pub temporal_alpha: f32,
    pub temporal_enabled: u32,
    pub use_half_res: u32,
    pub upsample_depth_sigma: f32,
    pub upsample_normal_sigma: f32,
    pub use_edge_aware: u32,
    pub _pad1: u32,
    pub frame_index: u32,
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
    pub _pad6: [u32; 4],
    pub _pad7: [u32; 3],
    pub _pad8: [u32; 4],
    pub _pad9: [u32; 8],
}

impl Default for SsgiSettings {
    fn default() -> Self {
        Self {
            radius: 1.0,
            intensity: 0.5,
            num_steps: 16,
            step_size: 0.1,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            temporal_alpha: 0.1,
            temporal_enabled: 1,
            use_half_res: 0,
            upsample_depth_sigma: 0.02,
            // Normal sigma controls bilateral falloff (radians)
            upsample_normal_sigma: 0.25,
            use_edge_aware: 1,
            _pad1: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            frame_index: 0,
            _pad6: [0; 4],
            _pad7: [0; 3],
            _pad8: [0; 4],
            _pad9: [0; 8],
        }
    }
}

/// SSR settings
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsrSettings {
    pub max_steps: u32,
    pub thickness: f32,
    pub max_distance: f32,
    pub intensity: f32,
    pub inv_resolution: [f32; 2],
    pub _pad0: [f32; 2],
}

impl Default for SsrSettings {
    fn default() -> Self {
        Self {
            max_steps: 96,
            thickness: 0.1,
            max_distance: 32.0,
            intensity: 5.0,
            inv_resolution: [1.0 / 1920.0, 1.0 / 1080.0],
            _pad0: [0.0; 2],
        }
    }
}

/// Aggregated statistics emitted by the SSR pipeline.
#[derive(Debug, Clone, Default)]
pub struct SsrStats {
    pub num_rays: u32,
    pub num_hits: u32,
    pub total_steps: u32,
    pub num_misses: u32,
    pub miss_ibl_samples: u32,
    pub trace_ms: f32,
    pub shade_ms: f32,
    pub fallback_ms: f32,
}

impl SsrStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn hit_rate(&self) -> f32 {
        if self.num_rays == 0 {
            0.0
        } else {
            self.num_hits as f32 / self.num_rays as f32
        }
    }

    pub fn avg_steps(&self) -> f32 {
        if self.num_rays == 0 {
            0.0
        } else {
            self.total_steps as f32 / self.num_rays as f32
        }
    }

    pub fn miss_ibl_ratio(&self) -> f32 {
        if self.num_misses == 0 {
            0.0
        } else {
            self.miss_ibl_samples as f32 / self.num_misses as f32
        }
    }

    pub fn perf_ms(&self) -> f32 {
        self.trace_ms + self.shade_ms + self.fallback_ms
    }
}

/// Camera parameters for screen-space effects
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraParams {
    pub view_matrix: [[f32; 4]; 4],
    pub inv_view_matrix: [[f32; 4]; 4],
    pub proj_matrix: [[f32; 4]; 4],
    pub inv_proj_matrix: [[f32; 4]; 4],
    /// P1.1: Previous frame's view_proj matrix for motion vector computation
    pub prev_view_proj_matrix: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    /// Frame index for temporal effects
    pub frame_index: u32,
    /// P1.2: Sub-pixel jitter offset for TAA (in pixel units, [-0.5, 0.5])
    pub jitter_offset: [f32; 2],
    /// Padding to maintain 16-byte alignment
    pub _pad_jitter: [f32; 2],
}

impl Default for CameraParams {
    fn default() -> Self {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self {
            view_matrix: identity,
            inv_view_matrix: identity,
            proj_matrix: identity,
            inv_proj_matrix: identity,
            prev_view_proj_matrix: identity,
            camera_pos: [0.0, 0.0, 0.0],
            frame_index: 0,
            jitter_offset: [0.0, 0.0],
            _pad_jitter: [0.0, 0.0],
        }
    }
}
