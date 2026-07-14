//! P3.1: HTTP range request primitives for COG streaming.

use super::content_range::parse_content_range;
use super::error::CogError;
use super::range_cache::{ByteCache, DiskCache};
use super::range_stats::RangeReaderStats;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const DEFAULT_BYTE_CACHE_BUDGET: u64 = 64 * 1024 * 1024;

/// HTTP range reader for fetching byte ranges from remote files.
pub struct RangeReader {
    client: reqwest::Client,
    url: String,
    file_size: u64,
    byte_cache: Arc<Mutex<ByteCache>>,
    disk_cache: Option<Arc<Mutex<DiskCache>>>,
    stats: Arc<RangeReaderStats>,
}

impl RangeReader {
    /// Create a new range reader for the given URL.
    /// Performs a HEAD request to determine file size.
    pub async fn new(url: &str) -> Result<Self, CogError> {
        Self::new_with_cache_budget(url, DEFAULT_BYTE_CACHE_BUDGET).await
    }

    /// Create a new range reader with an explicit byte-cache budget.
    pub async fn new_with_cache_budget(
        url: &str,
        cache_budget_bytes: u64,
    ) -> Result<Self, CogError> {
        Self::new_with_cache_options(url, cache_budget_bytes, None, 0).await
    }

    /// Create a new range reader with explicit memory and optional disk-cache budgets.
    pub async fn new_with_cache_options(
        url: &str,
        cache_budget_bytes: u64,
        cache_dir: Option<PathBuf>,
        disk_cache_budget_bytes: u64,
    ) -> Result<Self, CogError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let response = client.head(url).send().await?;

        if !response.status().is_success() {
            return Err(CogError::HttpError(format!(
                "HEAD request failed with status: {}",
                response.status()
            )));
        }

        let file_size = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| CogError::HttpError("Missing Content-Length header".into()))?;

        Ok(Self {
            client,
            url: url.to_string(),
            file_size,
            byte_cache: Arc::new(Mutex::new(ByteCache::new(cache_budget_bytes))),
            disk_cache: cache_dir
                .map(|dir| DiskCache::new(dir, url, disk_cache_budget_bytes))
                .transpose()?
                .map(|cache| Arc::new(Mutex::new(cache))),
            stats: Arc::new(RangeReaderStats::default()),
        })
    }

    /// Create a range reader for a local file (file:// URL).
    pub fn new_local(path: &str) -> Result<Self, CogError> {
        Self::new_local_with_cache_budget(path, DEFAULT_BYTE_CACHE_BUDGET)
    }

    /// Create a local range reader with an explicit byte-cache budget.
    pub fn new_local_with_cache_budget(
        path: &str,
        cache_budget_bytes: u64,
    ) -> Result<Self, CogError> {
        Self::new_local_with_cache_options(path, cache_budget_bytes, None, 0)
    }

    /// Create a local range reader with explicit memory and optional disk-cache budgets.
    pub fn new_local_with_cache_options(
        path: &str,
        cache_budget_bytes: u64,
        cache_dir: Option<PathBuf>,
        disk_cache_budget_bytes: u64,
    ) -> Result<Self, CogError> {
        use std::fs;
        let metadata = fs::metadata(path)?;
        let file_size = metadata.len();
        let url = format!("file://{}", path);

        Ok(Self {
            client: reqwest::Client::new(),
            url: url.clone(),
            file_size,
            byte_cache: Arc::new(Mutex::new(ByteCache::new(cache_budget_bytes))),
            disk_cache: cache_dir
                .map(|dir| DiskCache::new(dir, &url, disk_cache_budget_bytes))
                .transpose()?
                .map(|cache| Arc::new(Mutex::new(cache))),
            stats: Arc::new(RangeReaderStats::default()),
        })
    }

    /// Get the total file size.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Get the URL being read.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Read a byte range from the file.
    pub async fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CogError> {
        // A zero-length read is the empty range by definition: return it without
        // touching the cache or the network.
        if length == 0 {
            return Ok(Vec::new());
        }
        // Checked arithmetic: a range whose end overflows u64 or exceeds the
        // known file size is invalid before any fetch is issued.
        let end_exclusive = offset.checked_add(length).ok_or(CogError::InvalidRange {
            offset,
            length,
            file_size: self.file_size,
        })?;
        if end_exclusive > self.file_size {
            return Err(CogError::InvalidRange {
                offset,
                length,
                file_size: self.file_size,
            });
        }

        let cache_key = (offset, length);
        if let Ok(mut cache) = self.byte_cache.lock() {
            if let Some(data) = cache.get(&cache_key) {
                self.stats
                    .cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(data.clone());
            }
        }
        if let Some(disk_cache) = &self.disk_cache {
            if let Ok(cache) = disk_cache.lock() {
                if let Some(data) = cache.get(cache_key) {
                    self.stats
                        .cache_hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(data);
                }
            }
        }

        let start_time = std::time::Instant::now();

        let data = if self.url.starts_with("file://") {
            self.read_local_range(offset, length)?
        } else {
            self.read_http_range(offset, length).await?
        };

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        self.stats
            .requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .bytes_fetched
            .fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .total_latency_ms
            .fetch_add(elapsed_ms, std::sync::atomic::Ordering::Relaxed);

        if let Ok(mut cache) = self.byte_cache.lock() {
            cache.insert(cache_key, data.clone());
            self.stats
                .cached_bytes
                .store(cache.used_bytes(), std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(disk_cache) = &self.disk_cache {
            if let Ok(mut cache) = disk_cache.lock() {
                cache.insert(cache_key, &data)?;
                self.stats
                    .disk_cached_bytes
                    .store(cache.used_bytes(), std::sync::atomic::Ordering::Relaxed);
            }
        }

        Ok(data)
    }

    async fn read_http_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CogError> {
        // `read_range` guarantees length >= 1 and offset + length <= file_size,
        // so the inclusive end never underflows or exceeds the object.
        let end = offset + length - 1;
        let range_header = format!("bytes={}-{}", offset, end);

        let response = self
            .client
            .get(&self.url)
            .header(reqwest::header::RANGE, range_header)
            .send()
            .await?;

        // Require 206 Partial Content: a 200 OK means the server ignored the Range
        // header and returned the WHOLE body, which would silently corrupt a
        // mid-file read (returning file-start bytes for the requested offset).
        // Erroring here lets read_cog fall back to a correct full fetch.
        if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(CogError::HttpError(format!(
                "server did not honor the HTTP range request (status {}); \
                 range streaming requires 206 Partial Content",
                response.status()
            )));
        }

        // Validate the partial response against exactly what we asked for. A
        // proxy or misbehaving origin can return 206 with a different range, an
        // unparseable/absent Content-Range, or a truncated/oversized body;
        // accepting any of those would place the wrong bytes at the requested
        // offset. Reject before the bytes can enter the cache or a decoder.
        let content_range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                CogError::InvalidRangeResponse(
                    "206 Partial Content response is missing a Content-Range header".to_string(),
                )
            })?;
        let (start, resp_end, total) = parse_content_range(&content_range).ok_or_else(|| {
            CogError::InvalidRangeResponse(format!(
                "206 response has an unparseable Content-Range header: {content_range:?}"
            ))
        })?;
        if start != offset || resp_end != end || total != self.file_size {
            return Err(CogError::InvalidRangeResponse(format!(
                "206 Content-Range bytes {start}-{resp_end}/{total} does not match the \
                 requested bytes {offset}-{end}/{}",
                self.file_size
            )));
        }

        let bytes = response.bytes().await?;
        if bytes.len() as u64 != length {
            return Err(CogError::InvalidRangeResponse(format!(
                "206 response body is {} bytes, expected {length}",
                bytes.len()
            )));
        }
        Ok(bytes.to_vec())
    }

    fn read_local_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CogError> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let path = self.url.strip_prefix("file://").unwrap_or(&self.url);
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;

        let mut buffer = vec![0u8; length as usize];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Read multiple byte ranges in a single request (if server supports).
    pub async fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, CogError> {
        let mut results = Vec::with_capacity(ranges.len());
        for &(offset, length) in ranges {
            results.push(self.read_range(offset, length).await?);
        }
        Ok(results)
    }

    /// Get statistics for this reader.
    pub fn stats(&self) -> &RangeReaderStats {
        &self.stats
    }

    pub fn byte_cache_budget_bytes(&self) -> u64 {
        self.byte_cache
            .lock()
            .map(|cache| cache.budget_bytes())
            .unwrap_or(0)
    }

    pub fn disk_cache_budget_bytes(&self) -> u64 {
        self.disk_cache
            .as_ref()
            .and_then(|cache| cache.lock().ok().map(|cache| cache.budget_bytes()))
            .unwrap_or(0)
    }

    /// Clear the byte cache.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.byte_cache.lock() {
            cache.clear();
            self.stats
                .cached_bytes
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn byte_cache_respects_byte_budget() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "forge3d-range-reader-budget-{}.bin",
            std::process::id()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&vec![7u8; 256]).unwrap();

        let reader = RangeReader::new_local_with_cache_budget(path.to_str().unwrap(), 64).unwrap();
        for offset in (0..256).step_by(32) {
            let data = reader.read_range(offset, 32).await.unwrap();
            assert_eq!(data.len(), 32);
        }

        assert!(reader.stats().cached_bytes() <= 64);
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn disk_cache_respects_byte_budget() {
        let mut data_path = std::env::temp_dir();
        data_path.push(format!(
            "forge3d-range-reader-disk-data-{}.bin",
            std::process::id()
        ));
        let mut file = std::fs::File::create(&data_path).unwrap();
        file.write_all(&vec![9u8; 512]).unwrap();

        let mut cache_dir = std::env::temp_dir();
        cache_dir.push(format!(
            "forge3d-range-reader-disk-cache-{}",
            std::process::id()
        ));

        let reader = RangeReader::new_local_with_cache_options(
            data_path.to_str().unwrap(),
            0,
            Some(cache_dir.clone()),
            96,
        )
        .unwrap();
        for offset in (0..512).step_by(64) {
            let data = reader.read_range(offset, 64).await.unwrap();
            assert_eq!(data.len(), 64);
        }

        assert!(reader.stats().disk_cached_bytes() <= 96);
        std::fs::remove_file(data_path).ok();
        std::fs::remove_dir_all(cache_dir).ok();
    }

    #[tokio::test]
    async fn zero_length_read_returns_empty_without_reading_the_file() {
        // A zero-length range is empty by definition and must not fault, even at an
        // offset past the end of the file.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "forge3d-range-reader-zero-{}.bin",
            std::process::id()
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&[1u8; 16])
            .unwrap();
        let reader = RangeReader::new_local(path.to_str().unwrap()).unwrap();
        assert!(reader.read_range(0, 0).await.unwrap().is_empty());
        assert!(reader.read_range(9999, 0).await.unwrap().is_empty());
        std::fs::remove_file(path).ok();
    }
}
