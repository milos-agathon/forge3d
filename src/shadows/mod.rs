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
/// The moment atlas stores `exp(c * d)` AND its square for `d` in `[0, 1]`, so the
/// binding constraint is `exp(2 * c) <= 65504` => `c <= 5.545`. Above that the second
/// moment saturates to `+Inf`, `E[x^2] - E[x]^2` becomes `NaN`, and every Chebyshev
/// bound downstream collapses - which renders the whole scene as if fully shadowed.
pub const EVSM_MAX_EXPONENT_RGBA16F: f32 = 5.54;

/// Clamp an EVSM exponent to the range the moment atlas can actually represent.
///
/// Must be applied identically to the moment-generation pass and to the shader
/// uniforms that sample it: producer and consumer have to warp by the same constant.
pub fn clamp_evsm_exponent(exponent: f32) -> f32 {
    exponent.clamp(0.0, EVSM_MAX_EXPONENT_RGBA16F)
}

pub(crate) fn requires_moment_blur(technique: crate::lighting::types::ShadowTechnique) -> bool {
    matches!(
        technique,
        crate::lighting::types::ShadowTechnique::VSM | crate::lighting::types::ShadowTechnique::MSM
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn evsm_exponent_clamp_keeps_squared_moment_finite_in_rgba16f() {
        assert_eq!(clamp_evsm_exponent(-1.0), 0.0);
        assert_eq!(clamp_evsm_exponent(40.0), EVSM_MAX_EXPONENT_RGBA16F);

        let largest_squared_moment = (2.0 * EVSM_MAX_EXPONENT_RGBA16F).exp();
        assert!(largest_squared_moment.is_finite());
        assert!(largest_squared_moment <= 65_504.0);
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
    fn spatial_moment_blur_excludes_evsm_light_bleeding() {
        use crate::lighting::types::ShadowTechnique;

        assert!(requires_moment_blur(ShadowTechnique::VSM));
        assert!(requires_moment_blur(ShadowTechnique::MSM));
        assert!(!requires_moment_blur(ShadowTechnique::EVSM));
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
