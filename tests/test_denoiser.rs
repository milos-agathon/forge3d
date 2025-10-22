//! Tests for denoiser module (Workstream FOLLOW-UP 3)
//!
//! Validates:
//! - OIDN reduces variance vs none (PSNR +2dB or variance −30% on RT output)
//! - Switching to bilateral runs without OIDN installed
//! - Memory & time within budget (no unbounded copies)

use forge3d::post::{denoise_rgba, compute_patch_variance, DenoiseConfig, DenoiserType};

/// Generate noisy test image with Perlin-like noise
fn generate_noisy_image(width: u32, height: u32, base_color: [u8; 3], noise_amplitude: u8) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 4) as usize];
    
    // Simple pseudo-random noise generator
    let mut rng_state = 12345u64;
    let mut simple_rand = || {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state / 65536) % 256) as u8
    };
    
    for i in 0..(width * height) as usize {
        let noise = (simple_rand() as i16 - 128) * noise_amplitude as i16 / 128;
        data[i * 4] = (base_color[0] as i16 + noise).clamp(0, 255) as u8;
        data[i * 4 + 1] = (base_color[1] as i16 + noise).clamp(0, 255) as u8;
        data[i * 4 + 2] = (base_color[2] as i16 + noise).clamp(0, 255) as u8;
        data[i * 4 + 3] = 255;
    }
    
    data
}

/// Compute PSNR (Peak Signal-to-Noise Ratio) between two images
fn compute_psnr(img1: &[u8], img2: &[u8]) -> f32 {
    assert_eq!(img1.len(), img2.len());
    let n = img1.len() / 4;
    
    let mut mse = 0.0f32;
    for i in 0..n {
        let r1 = img1[i * 4] as f32;
        let g1 = img1[i * 4 + 1] as f32;
        let b1 = img1[i * 4 + 2] as f32;
        
        let r2 = img2[i * 4] as f32;
        let g2 = img2[i * 4 + 1] as f32;
        let b2 = img2[i * 4 + 2] as f32;
        
        mse += (r1 - r2).powi(2) + (g1 - g2).powi(2) + (b1 - b2).powi(2);
    }
    mse /= (n * 3) as f32;
    
    if mse < 1e-10 {
        return 100.0; // Effectively identical
    }
    
    let max_val = 255.0;
    20.0 * (max_val / mse.sqrt()).log10()
}

#[test]
fn test_denoiser_none_passthrough() {
    let width = 64;
    let height = 64;
    let data = vec![128u8; (width * height * 4) as usize];
    
    let config = DenoiseConfig {
        denoiser: DenoiserType::None,
        strength: 1.0,
    };
    
    let result = denoise_rgba(&data, width, height, &config).unwrap();
    assert_eq!(result, data, "None denoiser should pass through unchanged");
}

#[test]
fn test_bilateral_reduces_variance() {
    let width = 256;
    let height = 256;
    
    // Create noisy image
    let noisy = generate_noisy_image(width, height, [100, 150, 200], 80);
    let original_variance = compute_patch_variance(&noisy, width, height);
    
    println!("Original variance: {:.6}", original_variance);
    
    // Apply bilateral denoising
    let config = DenoiseConfig {
        denoiser: DenoiserType::Bilateral,
        strength: 0.8,
    };
    
    let denoised = denoise_rgba(&noisy, width, height, &config).unwrap();
    let denoised_variance = compute_patch_variance(&denoised, width, height);
    
    println!("Denoised variance: {:.6}", denoised_variance);
    println!("Variance reduction: {:.1}%", (1.0 - denoised_variance / original_variance) * 100.0);
    
    // Acceptance criterion: variance should be reduced by at least 30%
    assert!(
        denoised_variance < original_variance * 0.7,
        "Bilateral filter should reduce variance by at least 30%: original={:.6}, denoised={:.6}",
        original_variance,
        denoised_variance
    );
}

#[test]
fn test_bilateral_preserves_edges() {
    let width = 128;
    let height = 128;
    
    // Create step edge image (sharp transition)
    let mut data = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize * 4;
            let val = if x < width / 2 { 50 } else { 200 };
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
            data[idx + 3] = 255;
        }
    }
    
    // Apply bilateral (should preserve edge)
    let config = DenoiseConfig {
        denoiser: DenoiserType::Bilateral,
        strength: 0.8,
    };
    
    let result = denoise_rgba(&data, width, height, &config).unwrap();
    
    // Check that edge is still present (sample pixels near edge)
    let left_idx = ((height / 2) * width + width / 2 - 10) as usize * 4;
    let right_idx = ((height / 2) * width + width / 2 + 10) as usize * 4;
    
    let left_val = result[left_idx] as i32;
    let right_val = result[right_idx] as i32;
    
    println!("Left value: {}, Right value: {}", left_val, right_val);
    
    // Edge should still be strong (difference > 100)
    assert!(
        (right_val - left_val).abs() > 100,
        "Bilateral filter should preserve edges: left={}, right={}",
        left_val,
        right_val
    );
}

#[test]
fn test_denoise_strength_scaling() {
    let width = 128;
    let height = 128;
    let noisy = generate_noisy_image(width, height, [128, 128, 128], 60);
    
    let original_variance = compute_patch_variance(&noisy, width, height);
    
    // Test different strength values
    let strengths = [0.3, 0.5, 0.8];
    let mut variances = Vec::new();
    
    for &strength in &strengths {
        let config = DenoiseConfig {
            denoiser: DenoiserType::Bilateral,
            strength,
        };
        let denoised = denoise_rgba(&noisy, width, height, &config).unwrap();
        let variance = compute_patch_variance(&denoised, width, height);
        variances.push(variance);
        println!("Strength {:.1}: variance = {:.6}", strength, variance);
    }
    
    // Higher strength should produce lower variance
    assert!(
        variances[0] > variances[1] && variances[1] > variances[2],
        "Higher strength should produce more denoising (lower variance)"
    );
    
    // All should be less than original
    for &var in &variances {
        assert!(var < original_variance, "All denoised should have less variance than original");
    }
}

#[test]
fn test_bilateral_memory_efficiency() {
    // Test that bilateral filter doesn't create unbounded copies
    let width = 512;
    let height = 512;
    let data = generate_noisy_image(width, height, [128, 128, 128], 40);
    
    let config = DenoiseConfig {
        denoiser: DenoiserType::Bilateral,
        strength: 0.8,
    };
    
    // This should complete without OOM
    let result = denoise_rgba(&data, width, height, &config).unwrap();
    
    assert_eq!(result.len(), data.len(), "Output should match input size");
}

#[test]
fn test_bilateral_timing_budget() {
    // Test that bilateral filter completes in reasonable time
    use std::time::Instant;
    
    let width = 512;
    let height = 512;
    let data = generate_noisy_image(width, height, [128, 128, 128], 40);
    
    let config = DenoiseConfig {
        denoiser: DenoiserType::Bilateral,
        strength: 0.8,
    };
    
    let start = Instant::now();
    let _result = denoise_rgba(&data, width, height, &config).unwrap();
    let elapsed = start.elapsed();
    
    println!("Bilateral filter (512x512) took: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    
    // Should complete in under 5 seconds for 512x512
    assert!(
        elapsed.as_secs() < 5,
        "Bilateral filter should complete in under 5s for 512x512, took: {:.2}s",
        elapsed.as_secs_f64()
    );
}

#[test]
fn test_compute_patch_variance_uniform() {
    let width = 64;
    let height = 64;
    let uniform = vec![128u8; (width * height * 4) as usize];
    
    let variance = compute_patch_variance(&uniform, width, height);
    println!("Uniform image variance: {:.6}", variance);
    
    // Uniform image should have near-zero variance
    assert!(variance < 0.001, "Uniform image should have near-zero variance");
}

#[test]
fn test_compute_patch_variance_noisy() {
    let width = 64;
    let height = 64;
    let noisy = generate_noisy_image(width, height, [128, 128, 128], 80);
    
    let variance = compute_patch_variance(&noisy, width, height);
    println!("Noisy image variance: {:.6}", variance);
    
    // Noisy image should have high variance
    assert!(variance > 0.01, "Noisy image should have high variance");
}

#[test]
fn test_denoiser_type_from_str() {
    assert_eq!(DenoiserType::from_str("none"), Some(DenoiserType::None));
    assert_eq!(DenoiserType::from_str("NONE"), Some(DenoiserType::None));
    assert_eq!(DenoiserType::from_str("oidn"), Some(DenoiserType::Oidn));
    assert_eq!(DenoiserType::from_str("OIDN"), Some(DenoiserType::Oidn));
    assert_eq!(DenoiserType::from_str("bilateral"), Some(DenoiserType::Bilateral));
    assert_eq!(DenoiserType::from_str("Bilateral"), Some(DenoiserType::Bilateral));
    assert_eq!(DenoiserType::from_str("invalid"), None);
}

#[test]
fn test_bilateral_alpha_preservation() {
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; (width * height * 4) as usize];
    
    // Set varying alpha values
    for i in 0..(width * height) as usize {
        data[i * 4] = 128;
        data[i * 4 + 1] = 128;
        data[i * 4 + 2] = 128;
        data[i * 4 + 3] = ((i % 256) as u8); // Varying alpha
    }
    
    let config = DenoiseConfig {
        denoiser: DenoiserType::Bilateral,
        strength: 0.8,
    };
    
    let result = denoise_rgba(&data, width, height, &config).unwrap();
    
    // Check that alpha is preserved
    for i in 0..(width * height) as usize {
        assert_eq!(
            result[i * 4 + 3],
            data[i * 4 + 3],
            "Alpha channel should be preserved at pixel {}",
            i
        );
    }
}

// OIDN tests are conditional on feature flag
#[cfg(feature = "oidn")]
mod oidn_tests {
    use super::*;
    
    #[test]
    fn test_oidn_reduces_variance_more_than_bilateral() {
        let width = 256;
        let height = 256;
        let noisy = generate_noisy_image(width, height, [100, 150, 200], 80);
        
        let original_variance = compute_patch_variance(&noisy, width, height);
        
        // Test bilateral
        let bilateral_config = DenoiseConfig {
            denoiser: DenoiserType::Bilateral,
            strength: 0.8,
        };
        let bilateral_result = denoise_rgba(&noisy, width, height, &bilateral_config).unwrap();
        let bilateral_variance = compute_patch_variance(&bilateral_result, width, height);
        
        // Test OIDN
        let oidn_config = DenoiseConfig {
            denoiser: DenoiserType::Oidn,
            strength: 0.8,
        };
        let oidn_result = denoise_rgba(&noisy, width, height, &oidn_config).unwrap();
        let oidn_variance = compute_patch_variance(&oidn_result, width, height);
        
        println!("Original variance: {:.6}", original_variance);
        println!("Bilateral variance: {:.6}", bilateral_variance);
        println!("OIDN variance: {:.6}", oidn_variance);
        
        // OIDN should reduce variance by at least 30% vs original
        assert!(
            oidn_variance < original_variance * 0.7,
            "OIDN should reduce variance by at least 30%"
        );
        
        // OIDN should outperform bilateral (or be comparable)
        assert!(
            oidn_variance <= bilateral_variance * 1.1,
            "OIDN should be at least as good as bilateral"
        );
    }
    
    #[test]
    fn test_oidn_psnr_improvement() {
        let width = 256;
        let height = 256;
        
        // Create clean reference
        let clean = generate_noisy_image(width, height, [100, 150, 200], 0);
        
        // Create noisy version
        let noisy = generate_noisy_image(width, height, [100, 150, 200], 60);
        
        let noisy_psnr = compute_psnr(&clean, &noisy);
        
        // Denoise with OIDN
        let config = DenoiseConfig {
            denoiser: DenoiserType::Oidn,
            strength: 0.9,
        };
        let denoised = denoise_rgba(&noisy, width, height, &config).unwrap();
        let denoised_psnr = compute_psnr(&clean, &denoised);
        
        println!("Noisy PSNR: {:.2} dB", noisy_psnr);
        println!("Denoised PSNR: {:.2} dB", denoised_psnr);
        println!("Improvement: {:.2} dB", denoised_psnr - noisy_psnr);
        
        // Acceptance criterion: PSNR improvement of at least 2dB
        assert!(
            denoised_psnr > noisy_psnr + 2.0,
            "OIDN should improve PSNR by at least 2dB: {} -> {} (+{:.2}dB)",
            noisy_psnr,
            denoised_psnr,
            denoised_psnr - noisy_psnr
        );
    }
}

// Test that OIDN falls back to bilateral when unavailable
#[cfg(not(feature = "oidn"))]
#[test]
fn test_oidn_fallback_to_bilateral() {
    let width = 128;
    let height = 128;
    let noisy = generate_noisy_image(width, height, [128, 128, 128], 60);
    
    // Request OIDN (should fallback to bilateral)
    let config = DenoiseConfig {
        denoiser: DenoiserType::Oidn,
        strength: 0.8,
    };
    
    // Should succeed with fallback
    let result = denoise_rgba(&noisy, width, height, &config);
    assert!(
        result.is_ok(),
        "OIDN should fallback to bilateral when unavailable"
    );
    
    let denoised = result.unwrap();
    let original_variance = compute_patch_variance(&noisy, width, height);
    let denoised_variance = compute_patch_variance(&denoised, width, height);
    
    // Should still denoise (via bilateral fallback)
    assert!(
        denoised_variance < original_variance * 0.8,
        "Fallback bilateral should still reduce noise"
    );
}
