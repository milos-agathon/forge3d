// tests/restir_integration.rs
// Integration tests for ReSTIR DI implementation

use forge3d::path_tracing::{alias_table::*, restir::*};

#[test]
fn test_alias_table_construction() {
    let weights = [0.1, 0.3, 0.6];
    let table = AliasTable::new(&weights);

    assert_eq!(table.len(), 3);
    assert_eq!(table.total_weight(), 1.0);
    assert!(!table.is_empty());
}

#[test]
fn test_alias_table_uniform_weights() {
    let weights = [1.0, 1.0, 1.0, 1.0];
    let table = AliasTable::new(&weights);

    assert_eq!(table.len(), 4);
    assert_eq!(table.total_weight(), 4.0);

    // Test sampling multiple times
    let mut counts = vec![0; 4];
    for i in 0..1000 {
        let u1 = (i as f32) / 1000.0;
        let u2 = ((i * 17) % 1000) as f32 / 1000.0;
        let (idx, _pdf) = table.sample(u1, u2);
        counts[idx] += 1;
    }

    // With uniform weights, each bin should get roughly equal samples
    for count in counts {
        assert!(
            count > 200 && count < 300,
            "Sample count {} outside expected range",
            count
        );
    }
}

#[test]
fn test_alias_table_empty() {
    let weights: [f32; 0] = [];
    let table = AliasTable::new(&weights);

    assert!(table.is_empty());
    assert_eq!(table.total_weight(), 0.0);

    let (idx, pdf) = table.sample(0.5, 0.5);
    assert_eq!(idx, 0);
    assert_eq!(pdf, 0.0);
}

#[test]
fn test_alias_table_zero_weights() {
    let weights = [0.0, 0.0, 0.0];
    let table = AliasTable::new(&weights);

    assert_eq!(table.len(), 3);
    assert_eq!(table.total_weight(), 0.0);

    // Should handle gracefully
    let (idx, _pdf) = table.sample(0.5, 0.5);
    assert!(idx < 3);
}

#[test]
fn test_reservoir_creation() {
    let reservoir = Reservoir::new();

    assert_eq!(reservoir.m, 0);
    assert_eq!(reservoir.w_sum, 0.0);
    assert_eq!(reservoir.weight, 0.0);
    assert!(!reservoir.is_valid());
}

#[test]
fn test_reservoir_update() {
    let mut reservoir = Reservoir::new();

    let sample = LightSample {
        position: [1.0, 2.0, 3.0],
        light_index: 0,
        direction: [0.0, 0.0, 1.0],
        intensity: 1.0,
        light_type: 0,
        params: [0.0; 3],
    };

    // Update with weight 1.0, always accept (random = 0.0)
    let accepted = reservoir.update(sample, 1.0, 0.0);
    assert!(accepted);
    assert_eq!(reservoir.m, 1);
    assert_eq!(reservoir.w_sum, 1.0);
    assert_eq!(reservoir.target_pdf, 1.0);

    // Finalize
    reservoir.finalize();
    assert_eq!(reservoir.weight, 1.0); // w_sum / (M * target_pdf) = 1.0 / (1 * 1.0)
    assert!(reservoir.is_valid());
}

#[test]
fn test_reservoir_multiple_updates() {
    let mut reservoir = Reservoir::new();

    let sample1 = LightSample {
        position: [1.0, 0.0, 0.0],
        light_index: 0,
        direction: [0.0, 0.0, 1.0],
        intensity: 1.0,
        light_type: 0,
        params: [0.0; 3],
    };

    let sample2 = LightSample {
        position: [2.0, 0.0, 0.0],
        light_index: 1,
        direction: [0.0, 0.0, 1.0],
        intensity: 2.0,
        light_type: 0,
        params: [0.0; 3],
    };

    // First update (always accept)
    reservoir.update(sample1, 1.0, 0.0);
    assert_eq!(reservoir.sample.light_index, 0);

    // Second update (never accept due to random = 1.0)
    reservoir.update(sample2, 2.0, 1.0);
    assert_eq!(reservoir.sample.light_index, 0); // Should still be first sample
    assert_eq!(reservoir.m, 2);
    assert_eq!(reservoir.w_sum, 3.0); // 1.0 + 2.0

    reservoir.finalize();
    assert!(reservoir.is_valid());
}

#[test]
fn test_reservoir_combine() {
    let mut reservoir1 = Reservoir::new();
    let mut reservoir2 = Reservoir::new();

    let sample1 = LightSample {
        position: [1.0, 0.0, 0.0],
        light_index: 0,
        direction: [0.0, 0.0, 1.0],
        intensity: 1.0,
        light_type: 0,
        params: [0.0; 3],
    };

    let sample2 = LightSample {
        position: [2.0, 0.0, 0.0],
        light_index: 1,
        direction: [0.0, 0.0, 1.0],
        intensity: 2.0,
        light_type: 0,
        params: [0.0; 3],
    };

    // Setup both reservoirs
    reservoir1.update(sample1, 1.0, 0.0);
    reservoir1.finalize();

    reservoir2.update(sample2, 2.0, 0.0);
    reservoir2.finalize();

    let initial_m1 = reservoir1.m;
    let initial_w_sum1 = reservoir1.w_sum;

    // Combine reservoir2 into reservoir1 with Jacobian = 1.0
    reservoir1.combine(&reservoir2, 1.0, 0.5);

    assert!(reservoir1.m > initial_m1);
    assert!(reservoir1.w_sum > initial_w_sum1);
}

#[test]
fn test_reservoir_combine_empty() {
    let mut reservoir1 = Reservoir::new();
    let reservoir2 = Reservoir::new(); // Empty reservoir

    let sample = LightSample {
        position: [1.0, 0.0, 0.0],
        light_index: 0,
        direction: [0.0, 0.0, 1.0],
        intensity: 1.0,
        light_type: 0,
        params: [0.0; 3],
    };

    reservoir1.update(sample, 1.0, 0.0);
    reservoir1.finalize();

    let initial_state = reservoir1;

    // Combining with empty reservoir should not change anything
    reservoir1.combine(&reservoir2, 1.0, 0.5);

    assert_eq!(reservoir1.m, initial_state.m);
    assert_eq!(reservoir1.w_sum, initial_state.w_sum);
    assert_eq!(
        reservoir1.sample.light_index,
        initial_state.sample.light_index
    );
}

#[test]
fn test_restir_config_default() {
    let config = RestirConfig::default();

    assert_eq!(config.initial_candidates, 32);
    assert_eq!(config.temporal_neighbors, 1);
    assert_eq!(config.spatial_neighbors, 4);
    assert_eq!(config.spatial_radius, 16.0);
    assert_eq!(config.max_temporal_age, 20);
    assert!(config.bias_correction);
}

#[test]
fn test_restir_di_creation() {
    let config = RestirConfig::default();
    let restir = RestirDI::new(config.clone());

    assert_eq!(
        restir.config().initial_candidates,
        config.initial_candidates
    );
    assert!(restir.lights().is_empty());
    assert!(restir.alias_table_entries().is_none());
}

#[test]
fn test_restir_di_set_lights() {
    let config = RestirConfig::default();
    let mut restir = RestirDI::new(config);

    let lights = vec![
        LightSample {
            position: [0.0, 1.0, 0.0],
            light_index: 0,
            direction: [0.0, 0.0, 1.0],
            intensity: 1.0,
            light_type: 0,
            params: [0.0; 3],
        },
        LightSample {
            position: [1.0, 1.0, 0.0],
            light_index: 1,
            direction: [0.0, -1.0, 0.0],
            intensity: 2.0,
            light_type: 1,
            params: [0.0; 3],
        },
    ];

    let weights = [1.0, 3.0];

    restir.set_lights(lights.clone(), &weights);

    assert_eq!(restir.lights().len(), 2);
    assert_eq!(restir.lights()[0].intensity, 1.0);
    assert_eq!(restir.lights()[1].intensity, 2.0);
    assert_eq!(restir.lights()[1].light_type, 1);

    // Alias table should be available
    assert!(restir.alias_table_entries().is_some());
}

#[test]
fn test_restir_di_sample_light() {
    let config = RestirConfig::default();
    let mut restir = RestirDI::new(config);

    let lights = vec![LightSample {
        position: [0.0, 1.0, 0.0],
        light_index: 0,
        direction: [0.0, 0.0, 1.0],
        intensity: 1.0,
        light_type: 0,
        params: [0.0; 3],
    }];

    let weights = [1.0];
    restir.set_lights(lights, &weights);

    // Sample a light
    let result = restir.sample_light(0.5, 0.5);
    assert!(result.is_some());

    let (light_idx, pdf) = result.unwrap();
    assert_eq!(light_idx, 0);
    assert!(pdf > 0.0);
}

#[test]
fn test_restir_di_sample_empty() {
    let config = RestirConfig::default();
    let restir = RestirDI::new(config);

    // No lights set
    let result = restir.sample_light(0.5, 0.5);
    assert!(result.is_none());
}

#[test]
fn test_target_pdf_calculation() {
    let config = RestirConfig::default();
    let restir = RestirDI::new(config);

    let light_sample = LightSample {
        position: [0.0, 2.0, 0.0], // 2 units above shading point
        light_index: 0,
        direction: [0.0, 0.0, 1.0],
        intensity: 1.0,
        light_type: 0,
        params: [0.0; 3],
    };

    let shading_point = [0.0, 0.0, 0.0];
    let normal = [0.0, 1.0, 0.0]; // Pointing up

    let target_pdf = restir.target_pdf(&light_sample, shading_point, normal);
    assert!(
        target_pdf > 0.0,
        "Target PDF should be positive for visible light"
    );

    // Test with light behind surface
    let normal_down = [0.0, -1.0, 0.0]; // Pointing down
    let target_pdf_behind = restir.target_pdf(&light_sample, shading_point, normal_down);
    assert_eq!(
        target_pdf_behind, 0.0,
        "Target PDF should be zero for backfacing light"
    );
}

#[test]
fn test_initial_sampling() {
    let mut config = RestirConfig::default();
    config.initial_candidates = 4; // Small number for test

    let mut restir = RestirDI::new(config);

    let lights = vec![
        LightSample {
            position: [0.0, 2.0, 0.0],
            light_index: 0,
            direction: [0.0, 0.0, 1.0],
            intensity: 1.0,
            light_type: 0,
            params: [0.0; 3],
        },
        LightSample {
            position: [1.0, 2.0, 0.0],
            light_index: 1,
            direction: [0.0, 0.0, 1.0],
            intensity: 2.0,
            light_type: 0,
            params: [0.0; 3],
        },
    ];

    let weights = [1.0, 2.0];
    restir.set_lights(lights, &weights);

    let shading_point = [0.0, 0.0, 0.0];
    let normal = [0.0, 1.0, 0.0];

    // Generate random numbers for sampling
    let randoms = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3];

    let reservoir = restir.initial_sampling(shading_point, normal, &randoms);

    // Should have a valid reservoir with some samples
    assert!(
        reservoir.m > 0,
        "Reservoir should have processed some samples"
    );
    assert!(
        reservoir.is_valid() || reservoir.m == 0,
        "Reservoir should be valid if it has samples"
    );
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Ignore by default, run with --ignored for performance testing
    fn test_alias_table_performance() {
        let num_lights = 10000;
        let weights: Vec<f32> = (0..num_lights).map(|i| (i + 1) as f32).collect();

        let start = Instant::now();
        let table = AliasTable::new(&weights);
        let construction_time = start.elapsed();

        println!(
            "Alias table construction for {} lights: {:?}",
            num_lights, construction_time
        );

        // Test sampling performance
        let start = Instant::now();
        let num_samples = 100000;
        for i in 0..num_samples {
            let u1 = (i as f32) / num_samples as f32;
            let u2 = ((i * 17) % num_samples) as f32 / num_samples as f32;
            let (_idx, _pdf) = table.sample(u1, u2);
        }
        let sampling_time = start.elapsed();

        println!(
            "Alias table sampling {} samples: {:?}",
            num_samples, sampling_time
        );
        println!("Average time per sample: {:?}", sampling_time / num_samples);
    }

    #[test]
    #[ignore]
    fn test_reservoir_performance() {
        let num_updates = 100000;
        let mut reservoir = Reservoir::new();

        let sample = LightSample {
            position: [1.0, 2.0, 3.0],
            light_index: 0,
            direction: [0.0, 0.0, 1.0],
            intensity: 1.0,
            light_type: 0,
            params: [0.0; 3],
        };

        let start = Instant::now();
        for i in 0..num_updates {
            let random = (i as f32) / num_updates as f32;
            reservoir.update(sample, 1.0, random);
        }
        reservoir.finalize();
        let update_time = start.elapsed();

        println!("Reservoir updates {} times: {:?}", num_updates, update_time);
        println!("Average time per update: {:?}", update_time / num_updates);
        println!(
            "Final reservoir M: {}, weight: {}",
            reservoir.m, reservoir.weight
        );
    }
}
