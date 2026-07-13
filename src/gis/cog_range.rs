//! Range-backed windowed reads for remote COGs (feature `cog_streaming`).
//!
//! Fetches only the strips overlapping the requested window via HTTP range
//! requests instead of downloading the whole object. Striped GeoTIFFs only;
//! tiled or otherwise unsupported inputs return an error so the caller falls
//! back to a full fetch-and-slice.

use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

use tiff::decoder::Decoder;

use crate::gis::affine::PixelWindow;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_info::{read_striped_window_from_decoder, RasterReadResult};
use crate::terrain::cog::RangeReader;

/// Synchronous `Read + Seek` cursor that fetches bytes on demand from a
/// `RangeReader` (HTTP range requests), bridging the async reader to the sync
/// `tiff` decoder via `block_on` on a dedicated current-thread runtime.
struct RangeCursor {
    reader: Arc<RangeReader>,
    runtime: Arc<tokio::runtime::Runtime>,
    pos: u64,
    len: u64,
}

impl Read for RangeCursor {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(self.len - self.pos);
        let reader = self.reader.clone();
        let pos = self.pos;
        let data = self
            .runtime
            .block_on(async move { reader.read_range(pos, want).await })
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
        let n = data.len().min(buf.len());
        buf[..n].copy_from_slice(&data[..n]);
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

/// Read only the window's overlapping strips from a remote COG via HTTP range
/// requests. Returns an error (the caller should fall back to a full fetch) for
/// tiled/unsupported inputs or transport failures.
pub(crate) fn read_remote_window(url: &str, window: PixelWindow) -> GisResult<RasterReadResult> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| GisError::Io(format!("io: tokio runtime: {err}")))?;
    let runtime = Arc::new(runtime);
    let reader = runtime
        .block_on(RangeReader::new(url))
        .map_err(|err| GisError::InvalidRaster(format!("network_error: {err}")))?;
    let reader = Arc::new(reader);
    let len = reader.file_size();
    let cursor = RangeCursor {
        reader,
        runtime,
        pos: 0,
        len,
    };
    // 64 KiB buffering coalesces the decoder's many small reads into fewer,
    // larger range requests.
    let mut decoder = Decoder::new(BufReader::with_capacity(64 * 1024, cursor))
        .map_err(|err| GisError::InvalidRaster(format!("invalid_raster: {err}")))?;
    read_striped_window_from_decoder(&mut decoder, PathBuf::from(url), window)
}
