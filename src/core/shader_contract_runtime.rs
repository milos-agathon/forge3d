//! Runtime shader-contract observations for render captures.
//!
//! Static WGSL proofs show that a contract is sufficient. This module records
//! the values a real render supplied for selected contract inputs so
//! `shader_report()` can distinguish "proved but never observed" from a
//! feature-gated runtime lane that actually ran.

use serde::Serialize;
use std::cell::RefCell;
use std::sync::Mutex;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeBindingCheck {
    pub name: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binding: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_max: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_max: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length: Option<u64>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alarm: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeContractObservation {
    pub scene: String,
    pub pipeline: String,
    pub module: String,
    pub path: String,
    pub entry_point: String,
    pub contract: String,
    pub checked_bindings: Vec<RuntimeBindingCheck>,
    pub status: String,
}

impl RuntimeContractObservation {
    pub(crate) fn new(
        scene: &str,
        pipeline: &str,
        module: &str,
        path: &str,
        entry_point: &str,
        contract: &str,
    ) -> Self {
        Self {
            scene: scene.to_string(),
            pipeline: pipeline.to_string(),
            module: module.to_string(),
            path: path.to_string(),
            entry_point: entry_point.to_string(),
            contract: contract.to_string(),
            checked_bindings: Vec::new(),
            status: "passed".to_string(),
        }
    }

    pub(crate) fn check_range(
        &mut self,
        kind: &str,
        name: &str,
        binding: Option<u32>,
        observed_min: f32,
        observed_max: f32,
        allowed_min: f32,
        allowed_max: f32,
    ) {
        let mut status = "passed".to_string();
        let mut alarm = None;
        if !observed_min.is_finite() || !observed_max.is_finite() {
            status = "failed".to_string();
            alarm = Some(format!(
                "{name} observed non-finite range [{observed_min}, {observed_max}]"
            ));
        } else if observed_min < allowed_min || observed_max > allowed_max {
            status = "failed".to_string();
            alarm = Some(format!(
                "{name} observed range [{observed_min}, {observed_max}] outside [{allowed_min}, {allowed_max}]"
            ));
        }
        if status != "passed" {
            self.status = "failed".to_string();
        }
        self.checked_bindings.push(RuntimeBindingCheck {
            name: name.to_string(),
            kind: kind.to_string(),
            binding,
            observed_min: Some(observed_min),
            observed_max: Some(observed_max),
            allowed_min: Some(allowed_min),
            allowed_max: Some(allowed_max),
            length: None,
            status,
            alarm,
        });
    }

    pub(crate) fn check_length(
        &mut self,
        kind: &str,
        name: &str,
        binding: Option<u32>,
        observed: u64,
        minimum: u64,
    ) {
        let mut status = "passed".to_string();
        let mut alarm = None;
        if observed < minimum {
            status = "failed".to_string();
            alarm = Some(format!(
                "{name} length {observed} is below minimum {minimum}"
            ));
            self.status = "failed".to_string();
        }
        self.checked_bindings.push(RuntimeBindingCheck {
            name: name.to_string(),
            kind: kind.to_string(),
            binding,
            observed_min: None,
            observed_max: None,
            allowed_min: None,
            allowed_max: None,
            length: Some(observed),
            status,
            alarm,
        });
    }

    fn validate(&self) -> Result<(), String> {
        if self.status == "passed" {
            return Ok(());
        }
        Err(self
            .checked_bindings
            .iter()
            .filter_map(|binding| binding.alarm.as_deref())
            .collect::<Vec<_>>()
            .join("; "))
    }
}

static LAST: Mutex<Vec<RuntimeContractObservation>> = Mutex::new(Vec::new());

thread_local! {
    static CURRENT: RefCell<Option<Vec<RuntimeContractObservation>>> = const { RefCell::new(None) };
}

pub(crate) fn begin_runtime_contract_capture() {
    CURRENT.with(|slot| {
        *slot.borrow_mut() = cfg!(feature = "shader-contract-asserts").then(Vec::new);
    })
}

pub(crate) fn finish_runtime_contract_capture() {
    let observations = CURRENT.with(|slot| slot.borrow_mut().take().unwrap_or_default());
    *LAST.lock().unwrap_or_else(|p| p.into_inner()) = observations;
}

pub(crate) fn abort_runtime_contract_capture() {
    CURRENT.with(|slot| {
        slot.borrow_mut().take();
    });
}

#[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
pub(crate) fn capture_active() -> bool {
    CURRENT.with(|slot| slot.borrow().is_some())
}

pub(crate) fn record_observation(observation: RuntimeContractObservation) -> Result<(), String> {
    CURRENT.with(|slot| {
        if let Some(observations) = slot.borrow_mut().as_mut() {
            if let Err(alarm) = observation.validate() {
                LAST.lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(observation);
                return Err(alarm);
            }
            observations.push(observation);
        }
        Ok(())
    })
}

pub(crate) fn last_observations() -> Vec<RuntimeContractObservation> {
    LAST.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_range_marks_observation_failed() {
        let mut observation = RuntimeContractObservation::new(
            "scene", "pipeline", "module", "path", "entry", "contract",
        );
        observation.check_range("uniform", "u.value", Some(0), -1.0, 2.0, 0.0, 1.0);
        assert_eq!(observation.status, "failed");
        assert_eq!(observation.checked_bindings[0].status, "failed");
    }

    #[test]
    fn failed_runtime_observation_is_rejected() {
        let mut observation = RuntimeContractObservation::new(
            "scene", "pipeline", "module", "path", "entry", "contract",
        );
        observation.check_range("uniform", "u.value", Some(0), -1.0, -1.0, 0.0, 1.0);
        begin_runtime_contract_capture();
        if cfg!(feature = "shader-contract-asserts") {
            assert!(record_observation(observation).is_err());
            assert_eq!(last_observations().last().unwrap().status, "failed");
        } else {
            assert!(record_observation(observation).is_ok());
        }
        abort_runtime_contract_capture();
    }
}
