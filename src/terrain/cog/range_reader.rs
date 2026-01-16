//! P3.1: HTTP range request primitives for COG streaming.

use super::error::CogError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// HTTP range reader for fetching byte ranges from remote files.
pub struct RangeReader {
    client: reqwest::Client,
    url: String,
    file_size: u64,
    byte_cache: Arc<Mutex<HashMap<(u64, u64), Vec<u8>>>>,
    stats: Arc<RangeReaderStats>,
}

/// Statistics for range reader operations.
#[derive(Debug, Default)]
pub struct RangeReaderStats {
    pub requests: std::sync::atomic::AtomicU64,
    pub bytes_fetched: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub total_latency_ms: std::sync::atomic::AtomicU64,
}

impl RangeReader {
    /// Create a new range reader for the given URL.
    /// Performs a HEAD request to determine file size.
    pub async fn new(url: &str) -> Result<Self, CogError> {
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
            byte_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RangeReaderStats::default()),
        })
    }

    /// Create a range reader for a local file (file:// URL).
    pub fn new_local(path: &str) -> Result<Self, CogError> {
        use std::fs;
        let metadata = fs::metadata(path)?;
        let file_size = metadata.len();

        Ok(Self {
            client: reqwest::Client::new(),
            url: format!("file://{}", path),
            file_size,
            byte_cache: Arc::new(Mutex::new(HashMap::new())),
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
        if offset + length > self.file_size {
            return Err(CogError::InvalidRange {
                offset,
                length,
                file_size: self.file_size,
            });
        }

        let cache_key = (offset, length);
        if let Ok(cache) = self.byte_cache.lock() {
            if let Some(data) = cache.get(&cache_key) {
                self.stats
                    .cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(data.clone());
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
            if cache.len() < 1000 {
                cache.insert(cache_key, data.clone());
            }
        }

        Ok(data)
    }

    async fn read_http_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CogError> {
        let range_header = format!("bytes={}-{}", offset, offset + length - 1);

        let response = self
            .client
            .get(&self.url)
            .header(reqwest::header::RANGE, range_header)
            .send()
            .await?;

        if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT
        {
            return Err(CogError::HttpError(format!(
                "Range request failed with status: {}",
                response.status()
            )));
        }

        let bytes = response.bytes().await?;
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

    /// Clear the byte cache.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.byte_cache.lock() {
            cache.clear();
        }
    }
}

impl RangeReaderStats {
    pub fn requests(&self) -> u64 {
        self.requests.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn bytes_fetched(&self) -> u64 {
        self.bytes_fetched
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn avg_latency_ms(&self) -> f64 {
        let reqs = self.requests();
        if reqs == 0 {
            0.0
        } else {
            self.total_latency_ms
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / reqs as f64
        }
    }
}
