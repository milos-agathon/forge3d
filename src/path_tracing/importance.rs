//! A25: Object Importance Sampling - Per-object importance hints

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ImportanceSample {
    pub object_id: u32,
    pub importance_weight: f32,
    pub mis_weight: f32,
}

pub struct ObjectImportanceSampler {
    object_weights: HashMap<u32, f32>,
    total_weight: f32,
}

impl ObjectImportanceSampler {
    pub fn new() -> Self {
        Self {
            object_weights: HashMap::new(),
            total_weight: 0.0,
        }
    }

    // A25: MIS weighting tweaks; tags
    pub fn set_object_importance(&mut self, object_id: u32, weight: f32) {
        let old_weight = self.object_weights.get(&object_id).unwrap_or(&0.0);
        self.total_weight = self.total_weight - old_weight + weight;
        self.object_weights.insert(object_id, weight);
    }

    pub fn sample_object(&self, u: f32) -> Option<ImportanceSample> {
        if self.total_weight <= 0.0 {
            return None;
        }

        let target = u * self.total_weight;
        let mut cumulative = 0.0;

        for (&object_id, &weight) in &self.object_weights {
            cumulative += weight;
            if cumulative >= target {
                return Some(ImportanceSample {
                    object_id,
                    importance_weight: weight,
                    mis_weight: weight / self.total_weight,
                });
            }
        }

        None
    }

    // A25: ≥15% MSE ↓ on tagged objects w/o bias
    pub fn calculate_variance_reduction(&self, baseline_mse: f32, optimized_mse: f32) -> f32 {
        if baseline_mse <= 0.0 {
            return 0.0;
        }
        (baseline_mse - optimized_mse) / baseline_mse
    }
}