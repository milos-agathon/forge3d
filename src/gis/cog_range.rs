//! Range-backed windowed reads for remote COGs (feature `cog_streaming`).
//!
//! Fetches only the strips or tiles overlapping the requested window via HTTP
//! range requests instead of downloading the whole object. Chunky-planar striped
//! and tiled GeoTIFFs are streamed; planar-separate or otherwise unstreamable
//! inputs, transport failures, and invalid range responses are reported as a
//! classified fallback so the caller can decide between a full fetch and
//! propagating the error (request-validation and fatal decode errors propagate).

use std::io::{self, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tiff::decoder::Decoder;

use crate::gis::affine::PixelWindow;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_info::{read_window_from_decoder, RasterReadResult};
use crate::terrain::cog::{CogError, RangeReader};

/// Transport granularity of the range cursor. Every network fetch is one
/// BLOCK-ALIGNED block of at most this size, so the byte ranges requested for a
/// given file region are canonical: revisiting a region re-requests the exact
/// same `(offset, length)` pair and hits `RangeReader`'s byte cache instead of
/// re-downloading an overlapping window. Total transfer for a windowed read is
/// therefore bounded by the intersecting chunk payloads rounded up to block
/// granularity (+ header/IFD blocks), never an unaligned refetch per seek.
const CURSOR_BLOCK_SIZE: u64 = 16 * 1024;

/// Synchronous `Read + Seek` cursor that fetches bytes on demand from a
/// `RangeReader` (HTTP range requests), bridging the async reader to the sync
/// `tiff` decoder via `block_on` on a dedicated current-thread runtime.
struct RangeCursor {
    reader: Arc<RangeReader>,
    runtime: Arc<tokio::runtime::Runtime>,
    pos: u64,
    len: u64,
    /// The most recently fetched aligned block (`start`, bytes), serving the
    /// decoder's many small sequential reads without further `block_on` calls.
    block: Option<(u64, Vec<u8>)>,
    // Records the last transport/range error surfaced through `read`, so a later
    // decode failure can be told apart from a genuine (transport-free) decode
    // error when classifying whether to fall back or propagate.
    last_error: Arc<Mutex<Option<CogError>>>,
}

impl RangeCursor {
    /// Fetch the aligned block containing `pos` (cached locally and, by virtue of
    /// the canonical aligned range, in the shared `RangeReader` byte cache).
    fn load_block(&mut self, pos: u64) -> io::Result<()> {
        let start = pos - pos % CURSOR_BLOCK_SIZE;
        let length = CURSOR_BLOCK_SIZE.min(self.len - start);
        let reader = self.reader.clone();
        match self
            .runtime
            .block_on(async move { reader.read_range(start, length).await })
        {
            Ok(data) => {
                self.block = Some((start, data));
                Ok(())
            }
            Err(err) => {
                let message = err.to_string();
                if let Ok(mut slot) = self.last_error.lock() {
                    *slot = Some(err);
                }
                Err(io::Error::new(io::ErrorKind::Other, message))
            }
        }
    }
}

impl Read for RangeCursor {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let in_block = matches!(
            &self.block,
            Some((start, data)) if self.pos >= *start && self.pos < *start + data.len() as u64
        );
        if !in_block {
            self.load_block(self.pos)?;
        }
        let (start, data) = self.block.as_ref().expect("block loaded above");
        let offset = (self.pos - start) as usize;
        let n = buf.len().min(data.len() - offset);
        buf[..n].copy_from_slice(&data[offset..offset + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for RangeCursor {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let target = match pos {
            SeekFrom::Start(offset) => offset as i128,
            SeekFrom::End(offset) => self.len as i128 + offset as i128,
            SeekFrom::Current(offset) => self.pos as i128 + offset as i128,
        };
        if target < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before start of file",
            ));
        }
        self.pos = target as u64;
        Ok(self.pos)
    }
}

/// The result of attempting a range-backed windowed read of a remote COG.
pub(crate) enum RangeReadOutcome {
    /// The window was streamed successfully via HTTP range requests.
    Streamed(RasterReadResult),
    /// The range attempt could not proceed; the caller should fall back to a full
    /// fetch, recording the given (stable, non-sensitive) reason.
    Fallback(RangeFallback),
}

/// Why a range-backed read fell back to a full fetch. Each maps to a stable,
/// deterministic diagnostic category — never a raw transport/IO message.
#[derive(Clone, Copy)]
pub(crate) enum RangeFallback {
    /// The server does not usefully support range requests, or the range reader
    /// could not be initialized over the network.
    RangeNotSupported,
    /// A 206 response was returned but failed validation (missing/malformed/
    /// mismatched Content-Range, or a truncated/oversized body).
    InvalidRangeResponse,
    /// The TIFF layout is recognized but not streamable here (planar-separate, or
    /// a photometric/band combination the chunk decoder cannot assemble).
    UnsupportedLayout,
}

impl RangeFallback {
    /// Stable diagnostic reason string exposed to callers/warnings.
    pub(crate) fn reason(self) -> &'static str {
        match self {
            RangeFallback::RangeNotSupported => "range_not_supported",
            RangeFallback::InvalidRangeResponse => "invalid_range_response",
            RangeFallback::UnsupportedLayout => "unsupported_layout",
        }
    }
}

/// Attempt to read only the window's overlapping strips/tiles from a remote COG
/// via HTTP range requests.
///
/// Returns `Ok(Streamed)` on success, `Ok(Fallback)` when the caller should retry
/// with a full fetch (transport failure, invalid range response, or a recognized
/// unstreamable layout), and `Err` for errors that must propagate WITHOUT any
/// download: request validation (`InvalidArgument`/`InvalidBounds`) and fatal
/// decode/corruption errors a full fetch could not repair.
pub(crate) fn read_remote_window(url: &str, window: PixelWindow) -> GisResult<RangeReadOutcome> {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => Arc::new(runtime),
        Err(_) => return Ok(RangeReadOutcome::Fallback(RangeFallback::RangeNotSupported)),
    };
    let reader = match runtime.block_on(RangeReader::new(url)) {
        Ok(reader) => Arc::new(reader),
        Err(_) => return Ok(RangeReadOutcome::Fallback(RangeFallback::RangeNotSupported)),
    };
    let len = reader.file_size();
    let last_error: Arc<Mutex<Option<CogError>>> = Arc::new(Mutex::new(None));
    let cursor = RangeCursor {
        reader,
        runtime,
        pos: 0,
        len,
        block: None,
        last_error: last_error.clone(),
    };
    let mut decoder = match Decoder::new(cursor) {
        Ok(decoder) => decoder,
        // Classified like any later decode failure: a transport/range error under
        // the header read is fallback-eligible, but a genuinely corrupt header
        // (bytes arrived, TIFF parse failed) is fatal and propagates — a full
        // fetch of the same bytes would fail identically.
        Err(err) => return classify_window_error(GisError::from(err), &last_error),
    };
    match read_window_from_decoder(&mut decoder, PathBuf::from(url), window) {
        Ok(result) => Ok(RangeReadOutcome::Streamed(result)),
        Err(err) => classify_window_error(err, &last_error),
    }
}

/// Map a windowed-read error to either a propagated error or a classified
/// fallback, using the range cursor's recorded transport error to tell a
/// transport failure from a genuine decode failure.
fn classify_window_error(
    err: GisError,
    last_error: &Mutex<Option<CogError>>,
) -> GisResult<RangeReadOutcome> {
    match err {
        // Request-validation errors must propagate — never a silent full download.
        GisError::InvalidArgument(_) | GisError::InvalidBounds(_) => Err(err),
        // A recognized-but-unstreamable layout falls back to the full decoder.
        GisError::BackendUnavailable(_) => {
            Ok(RangeReadOutcome::Fallback(RangeFallback::UnsupportedLayout))
        }
        // A transport failure over the range cursor is fallback-eligible; a decode
        // error with no transport failure is fatal and propagates (a full fetch of
        // the same bytes would fail identically).
        other => match cursor_fallback(last_error) {
            Some(reason) => Ok(RangeReadOutcome::Fallback(reason)),
            None => Err(other),
        },
    }
}

/// If the range cursor recorded a transport/range error, classify it into a
/// fallback reason; `None` means no transport error occurred.
fn cursor_fallback(last_error: &Mutex<Option<CogError>>) -> Option<RangeFallback> {
    let slot = last_error.lock().ok()?;
    slot.as_ref().map(|err| match err {
        CogError::InvalidRangeResponse(_) => RangeFallback::InvalidRangeResponse,
        _ => RangeFallback::RangeNotSupported,
    })
}
