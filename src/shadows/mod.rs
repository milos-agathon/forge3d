// src/shadows/mod.rs
// Shadow mapping implementations for Workstream B
// Exists to centralize GPU/CPU shadow utilities shared across bindings and pipelines
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

mod cascade_math;
mod csm_depth_control;
mod csm_renderer;
mod csm_types;

pub mod blur_pass;
pub mod manager;
pub mod moment_pass;
pub mod state;

// Re-export CSM types from split modules
pub use cascade_math::detect_peter_panning;
pub use csm_renderer::CsmRenderer;
pub use csm_types::{CascadeStatistics, CsmConfig, CsmUniforms, ShadowCascade};

pub use blur_pass::{ShadowBlurPass, DEFAULT_MOMENT_BLUR_RADIUS};
pub use manager::{ShadowManager, ShadowManagerConfig};
pub(crate) use manager::{
    DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS, DEFAULT_PCSS_FILTER_RADIUS_TEXELS, DEFAULT_PCSS_LIGHT_SIZE,
};
pub use moment_pass::{create_moment_storage_view, MomentGenerationPass};

// Re-export common shadow types and utilities
pub use csm_renderer::CsmRenderer as CascadedShadowMaps;

pub(crate) const CSM_SHADER_SOURCE: &str = concat!(
    include_str!("../shaders/includes/determinism.wgsl"),
    "\n",
    include_str!("../shaders/includes/shadow_moments.wgsl"),
    "\n",
    include_str!("../shaders/shadows.wgsl")
);

/// Largest EVSM exponent an `Rgba16Float` moment atlas can carry.
///
/// Both EVSM lobes are normalized into `[-1, 1]`. The remaining fp16 constraint is
/// keeping their squared moments above the normal range near the middle of the
/// depth interval; 9 leaves useful precision while preserving a strong warp.
pub const EVSM_MAX_EXPONENT_RGBA16F: f32 = 9.0;
pub(crate) const MIN_SHADOW_MAP_SIZE: u32 = 512;
pub(crate) const MAX_SHADOW_MAP_SIZE: u32 = 8192;
pub(crate) const MAX_SHADOW_CASCADES: u32 = 4;
pub(crate) const MAX_SHADOW_ALLOCATION_BYTES: u64 = 512 * 1024 * 1024;

pub(crate) fn validate_shadow_dimensions(
    resolution: u32,
    cascades: u32,
) -> crate::core::error::RenderResult<()> {
    if !(MIN_SHADOW_MAP_SIZE..=MAX_SHADOW_MAP_SIZE).contains(&resolution)
        || !resolution.is_power_of_two()
    {
        return Err(crate::core::error::RenderError::render(format!(
            "shadow resolution must be a power of two between {MIN_SHADOW_MAP_SIZE} and {MAX_SHADOW_MAP_SIZE}, got {resolution}"
        )));
    }
    if !(1..=MAX_SHADOW_CASCADES).contains(&cascades) {
        return Err(crate::core::error::RenderError::render(format!(
            "shadow cascade count must be within 1..={MAX_SHADOW_CASCADES}, got {cascades}"
        )));
    }
    Ok(())
}

pub(crate) fn shadow_allocation_bytes(
    resolution: u32,
    cascades: u32,
    requires_moments: bool,
) -> crate::core::error::RenderResult<u64> {
    let bytes_per_texel = if requires_moments { 20_u64 } else { 4_u64 };
    u64::from(resolution)
        .checked_mul(u64::from(resolution))
        .and_then(|pixels| pixels.checked_mul(u64::from(cascades)))
        .and_then(|pixels| pixels.checked_mul(bytes_per_texel))
        .ok_or_else(|| crate::core::error::RenderError::render("shadow allocation size overflow"))
}

pub(crate) fn validate_shadow_device_limits(
    device: &wgpu::Device,
    resolution: u32,
    cascades: u32,
) -> crate::core::error::RenderResult<()> {
    validate_shadow_dimensions(resolution, cascades)?;
    let limits = device.limits();
    if resolution > limits.max_texture_dimension_2d {
        return Err(crate::core::error::RenderError::device(format!(
            "shadow resolution {resolution} exceeds device max_texture_dimension_2d {}",
            limits.max_texture_dimension_2d
        )));
    }
    if cascades > limits.max_texture_array_layers {
        return Err(crate::core::error::RenderError::device(format!(
            "shadow cascade count {cascades} exceeds device max_texture_array_layers {}",
            limits.max_texture_array_layers
        )));
    }
    Ok(())
}

/// Clamp an EVSM exponent to the range the moment atlas can actually represent.
///
/// Must be applied identically to the moment-generation pass and to the shader
/// uniforms that sample it: producer and consumer have to warp by the same constant.
pub fn clamp_evsm_exponent(exponent: f32) -> f32 {
    if exponent.is_finite() {
        exponent.clamp(0.0, EVSM_MAX_EXPONENT_RGBA16F)
    } else {
        0.0
    }
}

pub(crate) fn requires_moment_blur(technique: crate::lighting::types::ShadowTechnique) -> bool {
    matches!(
        technique,
        crate::lighting::types::ShadowTechnique::VSM
            | crate::lighting::types::ShadowTechnique::EVSM
            | crate::lighting::types::ShadowTechnique::MSM
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_dimensions_reject_invalid_requests_before_allocation() {
        for resolution in [0, 511, 513, 16_384, u32::MAX] {
            assert!(validate_shadow_dimensions(resolution, 1).is_err());
        }
        for cascades in [0, 5, u32::MAX] {
            assert!(validate_shadow_dimensions(512, cascades).is_err());
        }
        assert!(validate_shadow_dimensions(512, 1).is_ok());
        assert!(validate_shadow_dimensions(8192, 4).is_ok());
    }

    #[test]
    fn moment_shadow_allocation_counts_depth_atlas_and_intermediate() {
        assert_eq!(
            shadow_allocation_bytes(4096, 2, true).expect("valid allocation"),
            640 * 1024 * 1024
        );
    }

    fn execute_shader_probe(source: &str, label: &str, probe_body: &str) -> [f32; 5] {
        let context = crate::core::gpu::try_ctx().expect("GPU context");
        let device = &context.device;
        let queue = &context.queue;
        let source = format!(
            "{source}
@group(0) @binding(31) var<storage, read_write> visibility_output: array<f32, 5>;

@compute @workgroup_size(1)
fn test_visibility_entry() {{
    {probe_body}
}}"
        );
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("evsm-visibility-contract"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evsm-visibility-contract"),
            layout: None,
            module: &module,
            entry_point: "test_visibility_entry",
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("{label} shader probe failed: {error}");
        }
        let output = crate::core::resource_tracker::tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("evsm-visibility-output"),
                size: 20,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .expect("output buffer");
        let readback = crate::core::resource_tracker::tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("evsm-visibility-readback"),
                size: 20,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .expect("readback buffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("evsm-visibility-contract"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 31,
                resource: output.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evsm-visibility-contract"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("evsm-visibility-contract"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 20);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("map callback receiver");
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().expect("map callback").expect("map result");
        let mapped = slice.get_mapped_range();
        let result = *bytemuck::from_bytes::<[f32; 5]>(&mapped);
        drop(mapped);
        readback.unmap();
        result
    }

    #[test]
    fn evsm_exponent_clamp_keeps_normalized_moments_representable_in_rgba16f() {
        assert_eq!(clamp_evsm_exponent(-1.0), 0.0);
        assert_eq!(clamp_evsm_exponent(40.0), EVSM_MAX_EXPONENT_RGBA16F);
        assert_eq!(clamp_evsm_exponent(f32::NAN), 0.0);
        assert_eq!(clamp_evsm_exponent(f32::INFINITY), 0.0);

        for depth in [0.0_f32, 0.5, 1.0] {
            let positive = (EVSM_MAX_EXPONENT_RGBA16F * (depth - 1.0)).exp();
            let negative = (-EVSM_MAX_EXPONENT_RGBA16F * depth).exp();
            assert!(positive <= 1.0 && negative <= 1.0);
            assert!(positive.is_finite() && negative.is_finite());
        }
        let midpoint_second_moment = (-EVSM_MAX_EXPONENT_RGBA16F).exp();
        assert!(midpoint_second_moment >= 0.00006103515625);
    }

    #[test]
    fn shared_evsm_visibility_helpers_execute_lit_front_of_mean() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
visibility_output[0] = chebyshev_upper_bound_visibility(0.5, 0.01, 0.25);
visibility_output[1] = chebyshev_upper_bound_visibility(0.5, 0.01, 0.6);
let moments = vec4<f32>(0.5, 0.26, -0.5, 0.26);
visibility_output[2] =
    evsm_visibility_from_moments(moments, 0.25, -0.4, vec2<f32>(0.0001));
visibility_output[3] =
    evsm_visibility_from_moments(moments, 0.6, -0.75, vec2<f32>(0.0001));
visibility_output[4] =
    evsm_visibility_from_moments(moments, 0.25, -0.75, vec2<f32>(0.0001));";
        let [front, behind, positive_front, negative_front, both_front] =
            execute_shader_probe(source, "EVSM visibility", probe);
        assert_eq!(front, 1.0, "front-of-mean receiver must be lit");
        assert!(
            (behind - 0.5).abs() < 1.0e-5,
            "behind-mean Chebyshev result was {behind}"
        );
        assert!(
            (positive_front - 0.5).abs() < 1.0e-5,
            "the positive lobe's lit shortcut masked the negative lobe"
        );
        assert!(
            (negative_front - 0.5).abs() < 1.0e-5,
            "the negative lobe's lit shortcut masked the positive lobe"
        );
        assert_eq!(both_front, 1.0, "both front-of-mean lobes must be lit");
    }

    #[test]
    fn shared_evsm_minimum_variance_scales_each_warp_derivative() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let minimum = evsm_minimum_variance(
    vec2<f32>(2.0, -0.5),
    vec2<f32>(4.0, 4.0)
);
visibility_output[0] = minimum.x;
visibility_output[1] = minimum.y;";
        let [positive, negative, ..] = execute_shader_probe(source, "EVSM minimum variance", probe);
        assert!((positive - 0.000009).abs() < 1.0e-8);
        assert!((negative - 0.0000005625).abs() < 1.0e-9);
    }

    #[test]
    fn shared_evsm_moment_leak_control_preserves_a_soft_penumbra() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let exponent = 9.0;
let near_depth = exp(exponent * (0.45 - 1.0));
let far_depth = exp(exponent * (0.55 - 1.0));
let mean = 0.5 * (near_depth + far_depth);
let mean_squared = 0.5 * (near_depth * near_depth + far_depth * far_depth);
let moments = vec2<f32>(mean, mean_squared);
visibility_output[0] = evsm_moment_leak_control(
    moments, exp(exponent * (0.4 - 1.0)), exponent, 0.000001
);
visibility_output[1] = evsm_moment_leak_control(
    moments, exp(exponent * (0.5 - 1.0)), exponent, 0.000001
);
visibility_output[2] = evsm_moment_leak_control(
    moments, exp(exponent * (0.75 - 1.0)), exponent, 0.000001
);
let uniform_depth = exp(exponent * (0.5 - 1.0));
visibility_output[3] = evsm_moment_leak_control(
    vec2<f32>(uniform_depth, uniform_depth * uniform_depth),
    exp(exponent * (0.499 - 1.0)),
    exponent,
    0.000001
);
visibility_output[4] = evsm_moment_leak_control(
    vec2<f32>(uniform_depth, uniform_depth * uniform_depth),
    exp(exponent * (0.55 - 1.0)),
    exponent,
    0.000001
);";
        let [front, penumbra, shadow, uniform_front, uniform_shadow] =
            execute_shader_probe(source, "EVSM moment leak control", probe);
        assert_eq!(front, 1.0);
        assert_eq!(
            uniform_front, 1.0,
            "variance floor widened a uniform distribution"
        );
        assert!(
            uniform_shadow < 0.05,
            "uniform moments leaked behind their occluder: {uniform_shadow}"
        );
        assert!(
            penumbra > 0.05 && penumbra < 0.95,
            "mixed moments did not retain a soft penumbra: {penumbra}"
        );
        assert!(
            shadow < penumbra && shadow < 0.05,
            "visibility did not converge to shadow: penumbra={penumbra}, shadow={shadow}"
        );
    }

    #[test]
    fn spatial_moment_blur_covers_every_moment_technique() {
        use crate::lighting::types::ShadowTechnique;

        assert!(requires_moment_blur(ShadowTechnique::VSM));
        assert!(requires_moment_blur(ShadowTechnique::MSM));
        assert!(requires_moment_blur(ShadowTechnique::EVSM));
        assert!(!requires_moment_blur(ShadowTechnique::PCF));
    }

    #[test]
    fn shared_pcss_penumbra_grows_with_light_size_and_receiver_separation() {
        let probe = "
visibility_output[0] = pcss_penumbra_size(0.55, 0.5, 1.0);
visibility_output[1] = pcss_penumbra_size(0.55, 0.5, 12.0);
visibility_output[2] = pcss_penumbra_size(0.8, 0.5, 1.0);
visibility_output[3] = pcss_penumbra_size(0.8, 0.5, 12.0);
visibility_output[4] = 0.0;";
        let [near_small, near_large, far_small, far_large, _] =
            execute_shader_probe(CSM_SHADER_SOURCE, "PCSS penumbra", probe);
        let near_growth = near_large - near_small;
        let far_growth = far_large - far_small;

        assert!(near_growth > 1.0, "light-size response was {near_growth}");
        assert!(
            far_growth >= near_growth + 4.0,
            "receiver separation did not widen PCSS: near={near_growth}, far={far_growth}"
        );
    }

    #[test]
    fn shared_msm_visibility_is_bounded_for_degenerate_and_non_finite_inputs() {
        let source = include_str!("../shaders/includes/shadow_moments.wgsl");
        let probe = "
let nan_value = visibility_output[4] / visibility_output[4];
visibility_output[0] =
    msm_visibility_from_moments(vec4<f32>(0.5, 0.33333334, 0.25, 0.2), 0.6, 0.0005);
visibility_output[1] =
    msm_visibility_from_moments(vec4<f32>(0.5, 0.25, 0.125, 0.0625), 0.7, 0.0);
visibility_output[2] =
    msm_visibility_from_moments(vec4<f32>(nan_value), 0.7, 0.0005);
visibility_output[3] =
    msm_visibility_from_moments(vec4<f32>(0.5, 0.25, 0.125, 0.0625), nan_value, 0.0005);
visibility_output[4] =
    msm_visibility_from_moments(vec4<f32>(0.9, 0.1, 0.9, 0.1), 0.95, 0.0005);";
        let values = execute_shader_probe(source, "MSM boundary behavior", probe);
        for value in values {
            assert!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "MSM returned invalid visibility {value}"
            );
        }
        assert_eq!(values[2], 1.0, "invalid stored moments must fail open");
        assert_eq!(values[3], 1.0, "invalid receiver depth must fail open");
    }
}

#[cfg(test)]
mod msm_tests;
