// src/path_tracing/inverse/params.rs
// DIFFERENTIA: parameter block for the differentiable inverse path tracer.
// `InverseParams` is the host-side scalar optimization state (the per-texel
// albedo lives on the GPU texture inside the shared `TerrainPtScene`; sun
// direction/intensity and turbidity live here and are re-uploaded each
// iteration). `InverseParamsGpu` is the WGSL-facing uniform block shared by
// the inverse entry points that ride alongside the real forward kernels.
// RELEVANT FILES: src/shaders/pt_inverse_loss.wgsl, src/shaders/pt_inverse_shade.wgsl,
//                 src/path_tracing/inverse/mod.rs

use bytemuck::{Pod, Zeroable};

use crate::core::error::RenderError;

/// Number of scalar gradient slots in the f32-CAS accumulator
/// (sun_dir xyz, sun_intensity, turbidity, edge-mass diagnostic,
/// non-finite-adds counter, spare). Order must match pt_inverse_loss.wgsl.
pub const GRAD_SCALAR_SLOTS: usize = 8;
/// Clamp on the score-function (REINFORCE) log-probability gradient —
/// bounds the variance of the discrete-selection correction.
pub const SCORE_CLAMP: f32 = 4.0;
/// Weight of the luminance-structure term inside the per-pixel loss
/// (loss = MSE + w * luminance_error², a small SSIM-ish regularizer).
pub const LOSS_STRUCT_W: f32 = 0.15;

/// Physical clamps applied after every Adam step.
pub const TURBIDITY_MIN: f32 = 1.0;
pub const TURBIDITY_MAX: f32 = 10.0;
pub const SUN_INTENSITY_MIN: f32 = 0.0;
pub const SUN_INTENSITY_MAX: f32 = 64.0;
/// Albedo is optimized in log space: strictly positive, bounded below 1.
pub const ALBEDO_MIN: f32 = 1e-4;
pub const ALBEDO_MAX: f32 = 1.0;

/// Optimization state for the scene parameters that are NOT spatial
/// textures. The albedo map itself is the RGBA32F parameter texture inside
/// the shared `TerrainPtScene`; this struct carries the scalars.
#[derive(Clone, Copy, Debug)]
pub struct InverseParams {
    /// Direction from the surface TOWARD the sun (kernel convention), kept
    /// normalized on write.
    pub sun_dir: [f32; 3],
    pub sun_intensity: f32,
    /// Atmospheric turbidity in the physical range [1, 10]; enters the
    /// shared terrain uniform as `albedo_pad.w = turbidity - 1`.
    pub turbidity: f32,
}

impl InverseParams {
    /// Apply the physical clamps in place (albedo is clamped on its own map).
    pub fn clamp(&mut self) {
        let len = (self.sun_dir[0] * self.sun_dir[0]
            + self.sun_dir[1] * self.sun_dir[1]
            + self.sun_dir[2] * self.sun_dir[2])
            .sqrt();
        if len.is_finite() && len > 1e-6 {
            self.sun_dir = [
                self.sun_dir[0] / len,
                self.sun_dir[1] / len,
                self.sun_dir[2] / len,
            ];
        }
        self.sun_intensity = self
            .sun_intensity
            .clamp(SUN_INTENSITY_MIN, SUN_INTENSITY_MAX);
        self.turbidity = self.turbidity.clamp(TURBIDITY_MIN, TURBIDITY_MAX);
    }
}

/// GPU uniform block for the inverse state (group 0 binding 2). Six vec4
/// rows = 96 bytes, alignment-trivial like `TerrainPtUniforms`.
/// Mirrors `struct InvParams` in pt_inverse_loss.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InverseParamsGpu {
    /// x = albedo texture width (DEM texels), y = height, z = texel count,
    /// w = frames per eval (the primal accumulation-loop length).
    pub dims: [u32; 4],
    /// x = flags (bit0 = spatial reuse, bit1 = edge term, bit2 = score
    /// correction); y = whist slot count K = min(tile_size, frames) — the
    /// frames of reservoir history retained per adjoint checkpoint tile;
    /// z/w unused.
    pub ctrl: [u32; 4],
    /// rgb = unit-intensity sun color, a = sun intensity. Kept in sync with
    /// the lighting uniform the primal reads (`light_color = color*I`).
    pub sunv: [f32; 4],
    /// x = LOSS_STRUCT_W, y = SCORE_CLAMP, z = 1/pixel_count,
    /// w = 1/(frames*spp).
    pub fl: [f32; 4],
    /// x = 1/replicate_count (mean-image accumulation weight), y = replicate
    /// ordinal (0 = overwrite the mean region, >0 = accumulate), z/w unused.
    pub rsv0: [f32; 4],
    pub rsv1: [f32; 4],
}

impl InverseParamsGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sun_color: [f32; 3],
        sun_intensity: f32,
        alb_dims: (u32, u32),
        frames: u32,
        spp: u32,
        pixel_count: u64,
        tile_size: u32,
        spatial_reuse: bool,
        edge_term: bool,
        score_correction: bool,
    ) -> Self {
        let mut flags = 0u32;
        if spatial_reuse {
            flags |= 1;
        }
        if edge_term {
            flags |= 2;
        }
        if score_correction {
            flags |= 4;
        }
        Self {
            dims: [
                alb_dims.0,
                alb_dims.1,
                alb_dims.0 * alb_dims.1,
                frames.max(1),
            ],
            ctrl: [flags, frames.max(1).min(tile_size.max(1)), 0, 0],
            sunv: [sun_color[0], sun_color[1], sun_color[2], sun_intensity],
            fl: [
                LOSS_STRUCT_W,
                SCORE_CLAMP,
                1.0 / (pixel_count.max(1) as f32),
                1.0 / ((frames.max(1) as f32) * (spp.max(1) as f32)),
            ],
            rsv0: [0.0; 4],
            rsv1: [0.0; 4],
        }
    }
}

/// Uniforms for the standalone pt_restir_temporal / pt_restir_spatial passes.
/// Byte-compatible with the `Uniforms` struct those two WGSL modules declare
/// (field order differs from the hybrid kernel's Uniforms — spp replaces
/// aov_flags at offset 12).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RestirUniforms {
    pub width: u32,
    pub height: u32,
    pub frame_index: u32,
    pub spp: u32,
    pub cam_origin: [f32; 3],
    pub cam_fov_y: f32,
    pub cam_right: [f32; 3],
    pub cam_aspect: f32,
    pub cam_up: [f32; 3],
    pub cam_exposure: f32,
    pub cam_forward: [f32; 3],
    pub seed_hi: u32,
    pub seed_lo: u32,
    /// Uniform-address-space layout rounds the WGSL struct to a 16-byte
    /// multiple: fields end at 88, so the block must carry 96 bytes.
    pub _pad: [u32; 3],
}

/// Full solver input. `target_rgba` is the observed beauty image (the u8
/// beauty the forward path writes — Reinhard-space quantized, no sRGB
/// encode); the DEM is a known input (geometry is not recovered).
/// Everything is validated at the trust boundary before any GPU work.
#[derive(Clone)]
pub struct InverseSolveDesc {
    // --- scene (known) ---
    pub heights: Vec<f32>,
    pub dem_width: u32,
    pub dem_height: u32,
    pub spacing: (f32, f32),
    pub exaggeration: f32,
    // --- observation ---
    /// Forward beauty image, row-major RGBA u8, len = 4*width*height.
    pub target_rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    // --- camera ---
    pub cam_origin: [f32; 3],
    pub cam_look_at: [f32; 3],
    pub cam_up: [f32; 3],
    pub fov_y_deg: f32,
    pub exposure: f32,
    // --- initialization ---
    /// Initial per-texel albedo (linear), len = 3*dem_width*dem_height.
    pub init_albedo: Vec<f32>,
    pub init: InverseParams,
    pub sun_color: [f32; 3],
    pub env_map: Option<(Vec<f32>, u32, u32)>,
    pub env_intensity: f32,
    // --- optimizer ---
    pub iters: u32,
    pub spp: u32,
    /// Accumulation frames re-rendered per gradient eval (the ReSTIR loop
    /// length; the forward kernels are re-dispatched verbatim per frame).
    pub frames: u32,
    /// Frames of reservoir history retained per adjoint checkpoint tile
    /// (>= 8). The reservoir-gradient pass keeps only `min(tile_size,
    /// frames)` per-(frame, pixel) W/channel snapshots resident; when
    /// `frames > tile_size` the driver replays the seed-deterministic
    /// forward chain once per tile of frames instead of caching the full
    /// history — classic gradient checkpointing over the ReSTIR loop.
    pub tile_size: u32,
    pub seed: u32,
    /// Per-parameter-group learning rates.
    pub lr_albedo: f32,
    pub lr_sun: f32,
    pub lr_turbidity: f32,
    /// Early stop: halt when the best loss has not improved by more than
    /// `early_stop_tol` (relative — `tol * |best|`) for
    /// `early_stop_patience` iterations.
    pub early_stop_tol: f32,
    pub early_stop_patience: u32,
    /// Run the spatial reuse pass between primal and shading (default true).
    pub spatial_reuse: bool,
    /// Evaluate the reparameterized boundary term (default true).
    pub edge_term: bool,
    /// Score-function correction for the detached discrete reservoir
    /// selection (default true). This is an expectation-space term — the
    /// realized single path's selection is a.e. constant — so the finite-
    /// difference cross-check disables it to compare the continuous and
    /// reparameterized channels term-for-term.
    pub score_correction: bool,
}

/// Solver output. `albedo` is the recovered linear per-texel albedo map
/// (len 3*dem_width*dem_height); `rgba` is the final forward render in the
/// forward display space (the "after" image for the recovery artifact pair).
pub struct InverseSolveOutput {
    pub albedo: Vec<f32>,
    pub sun_dir: [f32; 3],
    pub sun_intensity: f32,
    pub turbidity: f32,
    pub loss_history: Vec<f32>,
    pub peak_host_visible_bytes: u64,
    pub iterations_run: u32,
    /// Final forward beauty (len = 4*width*height) — the recovered render.
    pub rgba: Vec<u8>,
}

fn finite3(v: [f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Trust-boundary validation; mirrors render_terrain::validate_desc.
pub fn validate_solve_desc(desc: &InverseSolveDesc) -> Result<(), RenderError> {
    let err = |msg: String| Err(RenderError::Render(msg));
    if desc.width == 0 || desc.height == 0 {
        return err("inverse solve requires non-zero image dims".into());
    }
    let px = (desc.width as usize) * (desc.height as usize);
    if desc.target_rgba.len() != px * 4 {
        return err(format!(
            "target_rgba length {} != 4*{}*{}",
            desc.target_rgba.len(),
            desc.width,
            desc.height
        ));
    }
    let texels = (desc.dem_width as usize) * (desc.dem_height as usize);
    if desc.dem_width < 2 || desc.dem_height < 2 {
        return err("terrain heightfield must be at least 2x2 texels".into());
    }
    if desc.heights.len() != texels || desc.heights.iter().any(|v| !v.is_finite()) {
        return err("heightfield length/dem dims mismatch or non-finite".into());
    }
    if desc.init_albedo.len() != texels * 3
        || desc.init_albedo.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return err(format!(
            "init_albedo must be 3*{}*{} finite non-negative values, got {}",
            desc.dem_width,
            desc.dem_height,
            desc.init_albedo.len()
        ));
    }
    if !(desc.spacing.0.is_finite()
        && desc.spacing.0 > 0.0
        && desc.spacing.1.is_finite()
        && desc.spacing.1 > 0.0)
    {
        return err(format!(
            "spacing must be finite and > 0, got {:?}",
            desc.spacing
        ));
    }
    if !(desc.exaggeration.is_finite() && desc.exaggeration > 0.0) {
        return err("exaggeration must be finite and > 0".into());
    }
    if !(finite3(desc.cam_origin) && finite3(desc.cam_look_at) && finite3(desc.cam_up)) {
        return err("camera origin/look_at/up must be finite".into());
    }
    if !(desc.fov_y_deg.is_finite() && desc.fov_y_deg > 0.0 && desc.fov_y_deg < 180.0) {
        return err(format!("fov_y must be in (0, 180), got {}", desc.fov_y_deg));
    }
    if !(desc.exposure.is_finite() && desc.exposure > 0.0) {
        return err("exposure must be finite and > 0".into());
    }
    if !(finite3(desc.init.sun_dir)
        && desc.init.sun_dir.iter().map(|v| v * v).sum::<f32>().sqrt() > 1e-6)
    {
        return err("init sun_dir must be finite and non-zero".into());
    }
    if !(desc.init.sun_intensity.is_finite() && desc.init.sun_intensity >= 0.0) {
        return err("init sun_intensity must be finite and >= 0".into());
    }
    if !(desc.init.turbidity.is_finite()
        && (TURBIDITY_MIN..=TURBIDITY_MAX).contains(&desc.init.turbidity))
    {
        return err(format!(
            "init turbidity must be in [{TURBIDITY_MIN}, {TURBIDITY_MAX}], got {}",
            desc.init.turbidity
        ));
    }
    if !finite3(desc.sun_color) || desc.sun_color.iter().any(|c| *c < 0.0) {
        return err("sun_color must be three finite non-negative values".into());
    }
    if !(desc.env_intensity.is_finite() && desc.env_intensity >= 0.0) {
        return err("env_intensity must be finite and >= 0".into());
    }
    if let Some((data, w, h)) = &desc.env_map {
        if *w == 0 || *h == 0 || data.len() != (*w as usize) * (*h as usize) * 3 {
            return err("env_map dims do not match data length".into());
        }
        if data.iter().any(|v| !v.is_finite()) {
            return err("env_map contains non-finite samples".into());
        }
    }
    if desc.iters == 0 || desc.spp == 0 || desc.spp > 64 {
        return err(format!(
            "iters must be >= 1 and spp in 1..=64, got iters={} spp={}",
            desc.iters, desc.spp
        ));
    }
    if desc.frames == 0 || desc.frames > 64 {
        return err(format!("frames must be in 1..=64, got {}", desc.frames));
    }
    if desc.tile_size < 8 {
        return err(format!("tile_size must be >= 8, got {}", desc.tile_size));
    }
    for (name, lr) in [
        ("lr_albedo", desc.lr_albedo),
        ("lr_sun", desc.lr_sun),
        ("lr_turbidity", desc.lr_turbidity),
    ] {
        if !(lr.is_finite() && lr >= 0.0) {
            return err(format!("{name} must be finite and >= 0, got {lr}"));
        }
    }
    if !(desc.early_stop_tol.is_finite() && desc.early_stop_tol >= 0.0) {
        return err("early_stop_tol must be finite and >= 0".into());
    }
    Ok(())
}

/// Adam state for one parameter group (per-coordinate first/second moments).
#[derive(Clone, Debug)]
pub struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AdamState {
    pub fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// One Adam step: `params -= lr * m_hat / (sqrt(v_hat) + eps)` elementwise.
    /// Non-finite gradients are dropped (the GPU CAS accumulators already
    /// guard; this is the last-line defense).
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.m.len());
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = if grads[i].is_finite() { grads[i] } else { 0.0 };
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_converges_on_quadratic() {
        // Minimize f(x) = (x-2)^2: grad = 2(x-2). Adam should approach 2.
        let mut p = [0.0f32];
        let mut adam = AdamState::new(1);
        for _ in 0..400 {
            let g = 2.0 * (p[0] - 2.0);
            adam.step(&mut p, &[g], 0.05);
        }
        assert!(
            (p[0] - 2.0).abs() < 1e-2,
            "adam failed to converge: {}",
            p[0]
        );
    }

    #[test]
    fn params_clamp_is_physical() {
        let mut p = InverseParams {
            sun_dir: [2.0, 0.0, 0.0],
            sun_intensity: -1.0,
            turbidity: 42.0,
        };
        p.clamp();
        assert!((p.sun_dir[0] - 1.0).abs() < 1e-6);
        assert_eq!(p.sun_intensity, 0.0);
        assert_eq!(p.turbidity, TURBIDITY_MAX);
    }

    #[test]
    fn gpu_block_sizes() {
        assert_eq!(std::mem::size_of::<InverseParamsGpu>(), 96);
        assert_eq!(std::mem::size_of::<RestirUniforms>(), 96);
    }
}
