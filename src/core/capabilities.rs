//! Negotiated GPU capability set: what forge3d wants, requires, and got.
use crate::core::degradation::record_degradation;
use wgpu::Features;

/// (stable name, feature bit, consequence when absent)
pub const WANTED: &[(&str, Features, &str)] = &[
    (
        "timestamp_query",
        Features::TIMESTAMP_QUERY,
        "per-pass GPU timings unavailable; certificate passes[].gpu_ms will be 0",
    ),
    (
        "pipeline_statistics_query",
        Features::PIPELINE_STATISTICS_QUERY,
        "pipeline_stats absent from certificate passes[]",
    ),
    (
        "texture_binding_array",
        Features::TEXTURE_BINDING_ARRAY,
        "terrain LUT texture-array bind path disabled; single-LUT rebind path used",
    ),
    (
        "sampled_texture_and_storage_buffer_array_non_uniform_indexing",
        Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        "non-uniform descriptor indexing unavailable; bindless paths disabled",
    ),
    (
        "texture_compression_bc",
        Features::TEXTURE_COMPRESSION_BC,
        "BC-compressed textures cannot upload natively; loader reports unsupported",
    ),
    (
        "float32_filterable",
        Features::FLOAT32_FILTERABLE,
        "float32 textures sample without filtering (nearest)",
    ),
    (
        "indirect_first_instance",
        Features::INDIRECT_FIRST_INSTANCE,
        "indirect draws with non-zero first_instance unavailable",
    ),
];

#[derive(Clone, Copy, Debug)]
pub struct CapabilitySet {
    pub wanted: Features,
    pub required: Features,
    pub granted: Features,
}

impl CapabilitySet {
    pub fn wants() -> Features {
        WANTED
            .iter()
            .fold(Features::empty(), |acc, (_, f, _)| acc | *f)
    }

    /// Intersect wants with adapter features. Nothing is hard-required, so
    /// this never fails; every want not granted records a degradation.
    pub fn negotiate(adapter_features: Features) -> Self {
        let wanted = Self::wants();
        let granted = wanted & adapter_features;
        for (name, feature, consequence) in WANTED {
            if !granted.contains(*feature) {
                record_degradation("capability_absent", name, consequence);
            }
        }
        CapabilitySet {
            wanted,
            required: Features::empty(),
            granted,
        }
    }

    pub fn wanted_names(&self) -> Vec<&'static str> {
        WANTED
            .iter()
            .filter(|(_, f, _)| self.wanted.contains(*f))
            .map(|(n, _, _)| *n)
            .collect()
    }
    pub fn granted_names(&self) -> Vec<&'static str> {
        WANTED
            .iter()
            .filter(|(_, f, _)| self.granted.contains(*f))
            .map(|(n, _, _)| *n)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intersection_only_grants_adapter_features() {
        let caps =
            CapabilitySet::negotiate(Features::TIMESTAMP_QUERY | Features::DEPTH_CLIP_CONTROL);
        assert!(caps.granted.contains(Features::TIMESTAMP_QUERY));
        assert!(!caps.granted.contains(Features::TEXTURE_BINDING_ARRAY));
        assert!(!caps.granted.contains(Features::DEPTH_CLIP_CONTROL)); // never request unwanted
        assert_eq!(caps.wanted, CapabilitySet::wants());
    }
    #[test]
    fn empty_adapter_grants_nothing_and_degrades() {
        crate::core::degradation::clear_degradations();
        let caps = CapabilitySet::negotiate(Features::empty());
        assert!(caps.granted.is_empty());
        let degs = crate::core::degradation::degradations_snapshot();
        assert_eq!(degs.len(), WANTED.len());
        assert!(degs.iter().all(|d| d.kind == "capability_absent"));
        crate::core::degradation::clear_degradations();
    }
}
