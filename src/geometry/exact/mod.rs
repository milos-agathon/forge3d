//! Exact planar predicates and their dependency-free test oracle.
//!
//! The production predicates use adaptive floating-point expansions.  The
//! oracle is deliberately separate and slow: it evaluates the same
//! determinants as exact dyadic integers and exists to adjudicate tests.

pub mod oracle;
pub mod predicates;

#[cfg(feature = "extension-module")]
mod py;

pub use predicates::{
    incircle, incircle_with_stage, orient2d, orient2d_with_stage, PredicateStage,
};
#[cfg(feature = "extension-module")]
pub use py::euclidea_predicate_report_py;
