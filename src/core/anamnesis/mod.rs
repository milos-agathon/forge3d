//! ANAMNESIS: content-addressed execution for deterministic offline renders.
//!
//! This module intentionally does not affect the interactive viewer. Cache
//! participation is opt-in; callers that pass no store execute exactly the
//! pre-ANAMNESIS renderer path.

pub mod key;
pub mod report;
pub mod scheduler;
pub mod store;

pub use key::{
    leaf_key, pass_key, reconstruct_pass_key, CapabilityFingerprint, EngineFingerprint, InputKey,
    PassKey, PassKeyMaterial,
};
pub use report::CacheReport;
pub use scheduler::{GraphScheduler, PassRequest, Scheduler};
pub use store::{ContentStore, StoreMetadata, VerifyReport};
