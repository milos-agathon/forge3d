//! P3: COG streaming error types.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CogError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    /// A ranged HTTP response was returned but failed validation (missing,
    /// malformed, mismatched, truncated, or oversized `Content-Range`/body).
    /// Distinct from `HttpError` so callers can classify it as a
    /// `invalid_range_response` fallback reason rather than a transport failure.
    #[error("Invalid range response: {0}")]
    InvalidRangeResponse(String),

    #[error("Invalid byte range: offset={offset}, length={length}, file_size={file_size}")]
    InvalidRange {
        offset: u64,
        length: u64,
        file_size: u64,
    },

    #[error("Invalid TIFF header: {0}")]
    InvalidTiffHeader(String),

    #[error("Invalid IFD: {0}")]
    InvalidIfd(String),

    #[error("Unsupported compression: {0}")]
    UnsupportedCompression(u16),

    #[error("Unsupported sample format: bits={bits}, format={format}")]
    UnsupportedSampleFormat { bits: u16, format: u16 },

    #[error("Tile not found: x={x}, y={y}, lod={lod}")]
    TileNotFound { x: u32, y: u32, lod: u32 },

    #[error("Decompression failed: {0}")]
    DecompressionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("URL parse error: {0}")]
    UrlError(String),
}

#[cfg(feature = "cog_streaming")]
impl From<reqwest::Error> for CogError {
    fn from(e: reqwest::Error) -> Self {
        CogError::HttpError(e.to_string())
    }
}
