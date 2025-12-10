//! Blend state configurations for Order Independent Transparency.
//!
//! Provides blend states for accumulation and reveal buffers used in weighted OIT.

/// Get MRT color target states for OIT accumulation pass.
pub fn get_mrt_color_targets() -> [Option<wgpu::ColorTargetState>; 2] {
    [
        // Color accumulation target (Rgba16Float)
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(get_accum_blend_state()),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        // Reveal accumulation target (R16Float)
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R16Float,
            blend: Some(get_reveal_blend_state()),
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ]
}

/// Get blend state for color accumulation buffer.
///
/// Uses additive blending for weighted color accumulation.
pub fn get_accum_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// Get blend state for reveal accumulation buffer.
///
/// Uses multiplicative blending for alpha reveal tracking.
pub fn get_reveal_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// Get ColorTargetState for accumulation buffer (Rgba16Float format).
pub fn accum_target_state() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba16Float,
        blend: Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// Get ColorTargetState for reveal buffer (R16Float format).
pub fn reveal_target_state() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: wgpu::TextureFormat::R16Float,
        blend: Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// Calculate OIT weight for a fragment based on depth and alpha.
///
/// Uses the McGuire & Mara weight function:
/// `w = alpha * clamp(0.03 / (1e-5 + pow(z/200, 4)), 1e-2, 3e3)`
pub fn calculate_weight(depth: f32, alpha: f32) -> f32 {
    let z_norm = depth / 200.0;
    alpha * (0.03 / (1e-5 + z_norm.powi(4))).clamp(1e-2, 3e3)
}
