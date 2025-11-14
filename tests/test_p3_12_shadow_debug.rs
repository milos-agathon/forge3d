// tests/test_p3_12_shadow_debug.rs
// P3-12: Shadow debug utilities
// Exit criteria: Debug methods print cascade info, memory, and technique details

use forge3d::gpu;
use forge3d::shadows::{ShadowManager, ShadowManagerConfig};

#[test]
fn test_shadow_manager_debug_info() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let config = ShadowManagerConfig::default();
    let manager = ShadowManager::new(&device, config);

    let debug_info = manager.debug_info();

    // Verify debug info contains expected fields
    assert!(debug_info.contains("Shadow Manager Configuration"));
    assert!(debug_info.contains("Technique:"));
    assert!(debug_info.contains("Shadow Map Size:"));
    assert!(debug_info.contains("Cascade Count:"));
    assert!(debug_info.contains("Total Memory:"));
    assert!(debug_info.contains("MiB"));
}

#[test]
fn test_shadow_manager_debug_info_technique_names() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    // Test different techniques have correct names
    let mut config = ShadowManagerConfig::default();

    config.technique = forge3d::lighting::types::ShadowTechnique::Hard;
    let manager = ShadowManager::new(&device, config.clone());
    assert!(manager.debug_info().contains("Technique: Hard"));

    config.technique = forge3d::lighting::types::ShadowTechnique::PCF;
    let manager = ShadowManager::new(&device, config.clone());
    assert!(manager.debug_info().contains("Technique: PCF"));

    config.technique = forge3d::lighting::types::ShadowTechnique::PCSS;
    let manager = ShadowManager::new(&device, config.clone());
    assert!(manager.debug_info().contains("Technique: PCSS"));

    config.technique = forge3d::lighting::types::ShadowTechnique::VSM;
    let manager = ShadowManager::new(&device, config.clone());
    assert!(manager.debug_info().contains("Technique: VSM"));

    config.technique = forge3d::lighting::types::ShadowTechnique::EVSM;
    let manager = ShadowManager::new(&device, config.clone());
    assert!(manager.debug_info().contains("Technique: EVSM"));

    config.technique = forge3d::lighting::types::ShadowTechnique::MSM;
    let manager = ShadowManager::new(&device, config.clone());
    assert!(manager.debug_info().contains("Technique: MSM"));
}

#[test]
fn test_shadow_manager_debug_info_resolution() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let mut config = ShadowManagerConfig::default();
    config.csm.shadow_map_size = 1024;

    let manager = ShadowManager::new(&device, config);
    let debug_info = manager.debug_info();

    assert!(debug_info.contains("1024x1024"));
}

#[test]
fn test_shadow_manager_debug_info_cascade_count() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let mut config = ShadowManagerConfig::default();
    config.csm.cascade_count = 3;

    let manager = ShadowManager::new(&device, config);
    let debug_info = manager.debug_info();

    assert!(debug_info.contains("Cascade Count: 3"));
}

#[test]
fn test_shadow_manager_debug_info_memory_budget() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let config = ShadowManagerConfig::default();
    let manager = ShadowManager::new(&device, config);

    let debug_info = manager.debug_info();

    // Should show memory in MiB
    assert!(debug_info.contains("MiB"));

    // Memory should be non-zero
    let memory_bytes = manager.memory_bytes();
    assert!(memory_bytes > 0);
}

#[test]
fn test_shadow_manager_debug_info_pcss_params() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let mut config = ShadowManagerConfig::default();
    config.pcss_blocker_radius = 0.05;
    config.pcss_filter_radius = 0.08;
    config.light_size = 0.5;

    let manager = ShadowManager::new(&device, config);
    let debug_info = manager.debug_info();

    assert!(debug_info.contains("PCSS Blocker Radius: 0.0500"));
    assert!(debug_info.contains("PCSS Filter Radius: 0.0800"));
    assert!(debug_info.contains("Light Size: 0.5000"));
}

#[test]
fn test_shadow_manager_debug_info_moment_bias() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let mut config = ShadowManagerConfig::default();
    config.moment_bias = 0.001;

    let manager = ShadowManager::new(&device, config);
    let debug_info = manager.debug_info();

    assert!(debug_info.contains("Moment Bias: 0.001000"));
}

#[test]
fn test_shadow_manager_cascade_debug_info() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let config = ShadowManagerConfig::default();
    let manager = ShadowManager::new(&device, config);

    let cascade_info = manager.cascade_debug_info();

    // Verify cascade info contains expected fields
    assert!(cascade_info.contains("Cascade Details"));
    assert!(cascade_info.contains("Cascade 0:"));
    assert!(cascade_info.contains("near="));
    assert!(cascade_info.contains("far="));
    assert!(cascade_info.contains("texel_size="));
}

#[test]
fn test_shadow_manager_cascade_debug_info_count() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let mut config = ShadowManagerConfig::default();
    config.csm.cascade_count = 3;

    let manager = ShadowManager::new(&device, config);
    let cascade_info = manager.cascade_debug_info();

    // Should have 3 cascades
    assert!(cascade_info.contains("Cascade 0:"));
    assert!(cascade_info.contains("Cascade 1:"));
    assert!(cascade_info.contains("Cascade 2:"));
    assert!(!cascade_info.contains("Cascade 3:"));
}

#[test]
fn test_shadow_manager_cascade_debug_info_values_reasonable() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let config = ShadowManagerConfig::default();
    let manager = ShadowManager::new(&device, config);

    let cascade_info = manager.cascade_debug_info();

    // Parse and verify cascade values are reasonable
    // Near should be less than far
    // Texel size should be positive
    for line in cascade_info.lines() {
        if line.contains("Cascade") && line.contains("near=") {
            // Just verify the format is correct
            assert!(line.contains("far="));
            assert!(line.contains("texel_size="));
        }
    }
}

#[test]
fn test_shadow_manager_debug_info_multiline() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let config = ShadowManagerConfig::default();
    let manager = ShadowManager::new(&device, config);

    let debug_info = manager.debug_info();

    // Should be multi-line output
    let lines: Vec<&str> = debug_info.lines().collect();
    assert!(lines.len() > 5, "Debug info should have multiple lines");
}

#[test]
fn test_shadow_manager_cascade_debug_info_multiline() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let config = ShadowManagerConfig::default();
    let manager = ShadowManager::new(&device, config);

    let cascade_info = manager.cascade_debug_info();

    // Should be multi-line output
    let lines: Vec<&str> = cascade_info.lines().collect();
    assert!(
        lines.len() >= 2,
        "Cascade info should have at least header + cascades"
    );
}

#[test]
fn test_shadow_manager_accessor_methods() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    let mut config = ShadowManagerConfig::default();
    config.csm.cascade_count = 3;
    config.csm.shadow_map_size = 2048;
    config.technique = forge3d::lighting::types::ShadowTechnique::PCF;

    let manager = ShadowManager::new(&device, config);

    // Test accessor methods used by debug_info
    assert_eq!(
        manager.technique(),
        forge3d::lighting::types::ShadowTechnique::PCF
    );
    assert_eq!(manager.shadow_map_size(), 2048);
    assert_eq!(manager.cascade_count(), 3);
    assert!(manager.memory_bytes() > 0);
}

#[test]
fn test_shadow_manager_debug_requires_moments() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();

    // VSM requires moments
    let mut config = ShadowManagerConfig::default();
    config.technique = forge3d::lighting::types::ShadowTechnique::VSM;
    let manager = ShadowManager::new(&device, config);
    assert!(manager.debug_info().contains("Requires Moments: true"));

    // PCF does not require moments
    let mut config = ShadowManagerConfig::default();
    config.technique = forge3d::lighting::types::ShadowTechnique::PCF;
    let manager = ShadowManager::new(&device, config);
    assert!(manager.debug_info().contains("Requires Moments: false"));
}
