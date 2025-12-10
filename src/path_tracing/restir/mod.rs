// src/path_tracing/restir.rs
// ReSTIR DI (Reservoir-based Spatio-Temporal Importance Resampling for Direct Illumination) implementation

mod buffers;
mod system;
mod types;

pub use buffers::*;
pub use system::RestirDI;
pub use types::{LightSample, Reservoir, RestirConfig};

#[cfg(test)]
mod tests {
    use super::*;

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
            _pad: [0; 4],
        };

        let accepted = reservoir.update(sample, 1.0, 0.5);
        assert!(accepted);
        assert_eq!(reservoir.m, 1);
        assert_eq!(reservoir.w_sum, 1.0);
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
            _pad: [0; 4],
        };

        let sample2 = LightSample {
            position: [2.0, 0.0, 0.0],
            light_index: 1,
            direction: [0.0, 0.0, 1.0],
            intensity: 2.0,
            light_type: 0,
            params: [0.0; 3],
            _pad: [0; 4],
        };

        reservoir1.update(sample1, 1.0, 0.5);
        reservoir1.finalize();

        reservoir2.update(sample2, 2.0, 0.5);
        reservoir2.finalize();

        reservoir1.combine(&reservoir2, 1.0, 0.5);
        assert!(reservoir1.m > 1);
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
    }
}
