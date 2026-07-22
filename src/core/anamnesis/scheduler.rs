//! Bottom-up incremental scheduler over byte-addressable pass outputs.
//!
//! GPU integrations restore a cached blob into the declared output resource
//! before dependents execute. Barrier planning remains a framegraph concern:
//! cached producers are still present in the compiled graph and their output
//! resources therefore retain the same declared state transitions.

use super::key::{pass_key, PassKey};
use super::report::CacheReport;
use super::store::ContentStore;
use std::io::Result;
use std::time::Instant;

pub struct PassRequest<'a> {
    pub label: &'a str,
    pub pipeline_descriptor_bytes: &'a [u8],
    pub uniform_bytes: &'a [u8],
    pub input_keys: &'a [PassKey],
    pub capability_fingerprint_bytes: &'a [u8],
    pub engine_fingerprint_bytes: &'a [u8],
    pub estimated_wall_ms: f64,
}

pub struct Scheduler {
    store: ContentStore,
    report: CacheReport,
}

impl Scheduler {
    pub fn new(store: ContentStore) -> Self {
        Self {
            store,
            report: CacheReport::default(),
        }
    }

    pub fn execute<F>(&mut self, request: PassRequest<'_>, encode: F) -> Result<(PassKey, Vec<u8>)>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        let (key, derivation) = pass_key(
            request.label,
            request.pipeline_descriptor_bytes,
            request.uniform_bytes,
            request.input_keys,
            request.capability_fingerprint_bytes,
            request.engine_fingerprint_bytes,
        );
        if let Some((blob, _)) = self.store.get(key)? {
            self.report.hits.push(request.label.to_string());
            self.report.bytes_read += blob.len() as u64;
            self.report.wall_ms_saved += request.estimated_wall_ms.max(0.0);
            return Ok((key, blob));
        }
        let started = Instant::now();
        let blob = encode()?;
        self.store.put(key, &blob, derivation)?;
        self.report.misses.push(request.label.to_string());
        self.report.bytes_written += blob.len() as u64;
        let _elapsed = started.elapsed();
        Ok((key, blob))
    }

    pub fn report(&self) -> &CacheReport {
        &self.report
    }
    pub fn into_report(self) -> CacheReport {
        self.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::anamnesis::EngineFingerprint;
    use std::cell::Cell;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn verified_hit_skips_the_encoder() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("forge3d-anamnesis-scheduler-{nonce}"));
        let store = ContentStore::new(&root, 1024, true).unwrap();
        let engine = EngineFingerprint::current().canonical_bytes();
        let encoded = Cell::new(0u32);
        let execute = |scheduler: &mut Scheduler| {
            scheduler
                .execute(
                    PassRequest {
                        label: "test.output",
                        pipeline_descriptor_bytes: b"pipeline",
                        uniform_bytes: b"uniform",
                        input_keys: &[],
                        capability_fingerprint_bytes: b"caps",
                        engine_fingerprint_bytes: &engine,
                        estimated_wall_ms: 4.5,
                    },
                    || {
                        encoded.set(encoded.get() + 1);
                        Ok(b"pixels".to_vec())
                    },
                )
                .unwrap()
        };

        let mut cold = Scheduler::new(store.clone());
        let (_, cold_blob) = execute(&mut cold);
        assert_eq!(encoded.get(), 1);
        assert_eq!(cold.report().misses, ["test.output"]);

        let mut warm = Scheduler::new(store);
        let (_, warm_blob) = execute(&mut warm);
        assert_eq!(encoded.get(), 1, "warm hit must not invoke the encoder");
        assert_eq!(warm.report().hits, ["test.output"]);
        assert_eq!(warm.report().bytes_read, 6);
        assert_eq!(warm.report().wall_ms_saved, 4.5);
        assert_eq!(cold_blob, warm_blob);
        fs::remove_dir_all(root).unwrap();
    }
}
