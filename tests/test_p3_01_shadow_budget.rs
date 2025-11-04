// tests/test_p3_01_shadow_budget.rs
// P3-01: Shadow manager memory budget enforcement tests
// Exit criteria: Given technique, cascade count, and requested map size,
// the atlas resolution is clamped so total ≤256 MiB. Unit test covers edge cases.

use forge3d::lighting::types::ShadowTechnique;
use forge3d::shadows::{CsmConfig, ShadowManager, ShadowManagerConfig};

/// Helper to calculate expected memory for validation
/// Note: All moment techniques (VSM/EVSM/MSM) use Rgba32Float = 16 bytes per pixel
fn calculate_expected_memory(resolution: u32, cascades: u32, technique: ShadowTechnique) -> u64 {
    let res = resolution as u64;
    let casc = cascades as u64;
    let depth_bytes = res * res * casc * 4;
    
    let moment_bytes = if technique.requires_moments() {
        // All moment techniques use Rgba32Float: 4 channels × 4 bytes = 16 bytes per pixel
        res * res * casc * 16
    } else {
        0
    };
    
    depth_bytes + moment_bytes
}

#[test]
fn test_budget_enforcement_pcf_within_budget() {
    // PCF at 2048px with 3 cascades should fit in 256 MiB budget
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 2048,
            cascade_count: 3,
            ..Default::default()
        },
        technique: ShadowTechnique::PCF,
        max_memory_bytes: 256 * 1024 * 1024,
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    // Should not downscale
    assert_eq!(manager.config().csm.shadow_map_size, 2048);
    
    // Verify memory calculation
    let expected_mem = calculate_expected_memory(2048, 3, ShadowTechnique::PCF);
    assert_eq!(manager.memory_bytes(), expected_mem);
    assert!(manager.memory_bytes() <= 256 * 1024 * 1024);
}

#[test]
fn test_budget_enforcement_evsm_downscale_required() {
    // EVSM at 4096px with 4 cascades exceeds 256 MiB budget
    // Expected: 4096² × 4 × (4 + 16) = 1,342,177,280 bytes (~1.25 GiB)
    // Should downscale to fit budget
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 4096,
            cascade_count: 4,
            ..Default::default()
        },
        technique: ShadowTechnique::EVSM,
        max_memory_bytes: 256 * 1024 * 1024,
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    // Should have downscaled
    assert!(manager.config().csm.shadow_map_size < 4096);
    
    // Verify memory is under budget
    assert!(manager.memory_bytes() <= 256 * 1024 * 1024);
    
    // Verify single-step downscaling stability (should be power of 2)
    let resolution = manager.config().csm.shadow_map_size;
    assert!(resolution.is_power_of_two());
}

#[test]
fn test_budget_enforcement_vsm_edge_case() {
    // VSM at 2048px with 4 cascades exceeds budget (uses Rgba32Float)
    // Expected: 2048² × 4 × (4 + 16) = 335,544,320 bytes (~320 MiB)
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 2048,
            cascade_count: 4,
            ..Default::default()
        },
        technique: ShadowTechnique::VSM,
        max_memory_bytes: 256 * 1024 * 1024,
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    // Should downscale to fit budget (2048px exceeds 256 MiB)
    assert!(manager.config().csm.shadow_map_size < 2048);
    assert!(manager.memory_bytes() <= 256 * 1024 * 1024);
    
    // Verify downscaled resolution is power of 2
    assert!(manager.config().csm.shadow_map_size.is_power_of_two());
}

#[test]
fn test_budget_enforcement_msm_high_resolution() {
    // MSM at 4096px with 3 cascades far exceeds budget (uses Rgba32Float)
    // Expected: 4096² × 3 × (4 + 16) = 1,006,632,960 bytes (~960 MiB)
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 4096,
            cascade_count: 3,
            ..Default::default()
        },
        technique: ShadowTechnique::MSM,
        max_memory_bytes: 256 * 1024 * 1024,
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    // Should downscale to fit budget (likely 512px or 1024px)
    assert!(manager.config().csm.shadow_map_size < 4096);
    assert!(manager.memory_bytes() <= 256 * 1024 * 1024);
    
    // Verify it's a valid power-of-2 resolution
    assert!(manager.config().csm.shadow_map_size.is_power_of_two());
}

#[test]
fn test_budget_enforcement_minimum_resolution_clamp() {
    // Test that downscaling respects minimum resolution (256px)
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 8192,
            cascade_count: 4,
            ..Default::default()
        },
        technique: ShadowTechnique::EVSM,
        max_memory_bytes: 1024 * 1024, // Very small budget: 1 MiB
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    // Should clamp to minimum resolution (256px)
    assert_eq!(manager.config().csm.shadow_map_size, 256);
    
    // May exceed budget if minimum resolution still too large
    // This is expected behavior - we warn but don't fail
}

#[test]
fn test_budget_enforcement_hard_shadows_no_moments() {
    // Hard shadows don't allocate moment textures
    // Expected: 2048² × 3 × 4 = 50,331,648 bytes (~48 MiB)
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 2048,
            cascade_count: 3,
            ..Default::default()
        },
        technique: ShadowTechnique::Hard,
        max_memory_bytes: 256 * 1024 * 1024,
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    assert_eq!(manager.config().csm.shadow_map_size, 2048);
    assert!(!manager.uses_moments());
    
    let expected_mem = calculate_expected_memory(2048, 3, ShadowTechnique::Hard);
    assert_eq!(manager.memory_bytes(), expected_mem);
}

#[test]
fn test_budget_enforcement_different_cascade_counts() {
    // Test with different cascade counts (1, 2, 3, 4)
    for cascades in 1..=4 {
        let config = ShadowManagerConfig {
            csm: CsmConfig {
                shadow_map_size: 2048,
                cascade_count: cascades,
                ..Default::default()
            },
            technique: ShadowTechnique::PCF,
            max_memory_bytes: 256 * 1024 * 1024,
            ..Default::default()
        };
        
        let device = forge3d::gpu::create_device_for_test();
        let manager = ShadowManager::new(&device, config.clone());
        
        // All should fit in budget
        assert!(manager.memory_bytes() <= 256 * 1024 * 1024);
        
        // Verify memory scales linearly with cascade count
        let expected_mem = calculate_expected_memory(2048, cascades, ShadowTechnique::PCF);
        assert_eq!(manager.memory_bytes(), expected_mem);
    }
}

#[test]
fn test_budget_enforcement_custom_budget() {
    // Test with custom budget (512 MiB)
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 4096,
            cascade_count: 3,
            ..Default::default()
        },
        technique: ShadowTechnique::EVSM,
        max_memory_bytes: 512 * 1024 * 1024,
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    // Should fit in 512 MiB budget
    assert!(manager.memory_bytes() <= 512 * 1024 * 1024);
    
    // Expected: 4096² × 3 × (4 + 16) = 1,006,632,960 bytes (~960 MiB) > 512 MiB
    // Should downscale to 2048
    assert_eq!(manager.config().csm.shadow_map_size, 2048);
}

#[test]
fn test_budget_enforcement_stability_no_thrashing() {
    // Verify that running budget enforcement multiple times produces same result
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 4096,
            cascade_count: 4,
            ..Default::default()
        },
        technique: ShadowTechnique::EVSM,
        max_memory_bytes: 256 * 1024 * 1024,
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager1 = ShadowManager::new(&device, config.clone());
    let manager2 = ShadowManager::new(&device, config.clone());
    
    // Both should produce identical results (deterministic)
    assert_eq!(
        manager1.config().csm.shadow_map_size,
        manager2.config().csm.shadow_map_size
    );
    assert_eq!(manager1.memory_bytes(), manager2.memory_bytes());
}

#[test]
fn test_budget_enforcement_all_techniques() {
    // Test all shadow techniques with same configuration
    let techniques = [
        ShadowTechnique::Hard,
        ShadowTechnique::PCF,
        ShadowTechnique::PCSS,
        ShadowTechnique::VSM,
        ShadowTechnique::EVSM,
        ShadowTechnique::MSM,
    ];
    
    for technique in techniques {
        let config = ShadowManagerConfig {
            csm: CsmConfig {
                shadow_map_size: 2048,
                cascade_count: 3,
                ..Default::default()
            },
            technique,
            max_memory_bytes: 256 * 1024 * 1024,
            ..Default::default()
        };
        
        let device = forge3d::gpu::create_device_for_test();
        let manager = ShadowManager::new(&device, config.clone());
        
        // All should respect budget
        assert!(
            manager.memory_bytes() <= 256 * 1024 * 1024,
            "Technique {:?} exceeds budget: {} bytes",
            technique,
            manager.memory_bytes()
        );
        
        // Verify moment texture usage
        match technique {
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM => {
                assert!(manager.uses_moments(), "Technique {:?} should use moments", technique);
            }
            _ => {
                assert!(!manager.uses_moments(), "Technique {:?} should not use moments", technique);
            }
        }
    }
}

#[test]
fn test_memory_calculation_accuracy() {
    // Verify memory calculation matches actual texture allocation
    let test_cases = vec![
        (512, 2, ShadowTechnique::Hard),
        (1024, 3, ShadowTechnique::PCF),
        (2048, 4, ShadowTechnique::VSM),
        (1024, 3, ShadowTechnique::EVSM),
        (512, 4, ShadowTechnique::MSM),
    ];
    
    for (resolution, cascades, technique) in test_cases {
        let config = ShadowManagerConfig {
            csm: CsmConfig {
                shadow_map_size: resolution,
                cascade_count: cascades,
                ..Default::default()
            },
            technique,
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GiB (generous)
            ..Default::default()
        };
        
        let device = forge3d::gpu::create_device_for_test();
        let manager = ShadowManager::new(&device, config.clone());
        
        let expected_mem = calculate_expected_memory(resolution, cascades, technique);
        assert_eq!(
            manager.memory_bytes(),
            expected_mem,
            "Memory mismatch for {}px × {} cascades × {:?}",
            resolution,
            cascades,
            technique
        );
    }
}

#[test]
fn test_downscale_power_of_two() {
    // Verify downscaling maintains power-of-two resolutions
    let config = ShadowManagerConfig {
        csm: CsmConfig {
            shadow_map_size: 8192,
            cascade_count: 4,
            ..Default::default()
        },
        technique: ShadowTechnique::EVSM,
        max_memory_bytes: 128 * 1024 * 1024, // Force downscaling
        ..Default::default()
    };
    
    let device = forge3d::gpu::create_device_for_test();
    let manager = ShadowManager::new(&device, config.clone());
    
    let resolution = manager.config().csm.shadow_map_size;
    assert!(resolution.is_power_of_two(), "Resolution {} is not power of two", resolution);
    assert!(resolution >= 256, "Resolution should not go below minimum");
}
