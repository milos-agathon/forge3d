use crate::terrain::probes::types::{ProbeError, ProbeIrradianceSet, ProbePlacement};

/// Abstract probe baker backend.
pub trait ProbeBaker {
    fn bake(&self, placement: &ProbePlacement) -> Result<ProbeIrradianceSet, ProbeError>;
}
