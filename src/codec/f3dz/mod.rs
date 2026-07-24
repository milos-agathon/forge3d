//! F3DZ v1: deterministic, error-bounded DEM compression.

pub mod decode;
pub mod encode;
pub mod format;
pub mod predict;
pub mod rans;

pub use decode::{decode_dem, DecodedDem};
pub use encode::{encode_dem, EncodeOptions};

use std::fmt;

/// Structured F3DZ failures. Decode paths fail closed: corrupt or
/// contract-incompatible data never produces a stale or interpolated height.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum F3dzError {
    InvalidArgument(String),
    InvalidHeader(String),
    CorruptPage {
        page: usize,
        reason: String,
    },
    EpsilonMismatch {
        requested_bits: u32,
        stream_bits: u32,
    },
    UnsupportedVersion(u16),
    Truncated {
        needed: usize,
        available: usize,
    },
    GpuUnavailable(String),
}

impl fmt::Display for F3dzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "invalid_argument: {message}"),
            Self::InvalidHeader(message) => write!(f, "invalid_header: {message}"),
            Self::CorruptPage { page, reason } => {
                write!(f, "corrupt_page: page={page} reason={reason}")
            }
            Self::EpsilonMismatch {
                requested_bits,
                stream_bits,
            } => write!(
                f,
                "epsilon_mismatch: requested_bits=0x{requested_bits:08x} stream_bits=0x{stream_bits:08x}"
            ),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported_version: f3dz/{version}")
            }
            Self::Truncated { needed, available } => {
                write!(f, "truncated_stream: needed={needed} available={available}")
            }
            Self::GpuUnavailable(message) => write!(f, "gpu_unavailable: {message}"),
        }
    }
}

impl std::error::Error for F3dzError {}

pub type F3dzResult<T> = Result<T, F3dzError>;
