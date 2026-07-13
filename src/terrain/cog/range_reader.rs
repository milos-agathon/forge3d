//! P3.1: HTTP range request primitives for COG streaming.

use super::error::CogError;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
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

/// Statistics for range reader operations.
#[derive(Debug, Default)]
pub struct RangeReaderStats {
    pub requests: std::sync::atomic::AtomicU64,
    pub bytes_fetched: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    cached_bytes: std::sync::atomic::AtomicU64,
    disk_cached_bytes: std::sync::atomic::AtomicU64,
    pub total_latency_ms: std::sync::atomic::AtomicU64,
}

struct ByteCache {
    entries: HashMap<(u64, u64), Vec<u8>>,
    lru: VecDeque<(u64, u64)>,
    used_bytes: u64,
    budget_bytes: u64,
}

struct DiskCache {
    dir: PathBuf,
    source_hash: u64,
    budget_bytes: u64,
    used_bytes: u64,
}

impl DiskCache {
    fn new(dir: PathBuf, source: &str, budget_bytes: u64) -> Result<Self, CogError> {
        std::fs::create_dir_all(&dir)?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let mut cache = Self {
            dir,
            source_hash: hasher.finish(),
            budget_bytes,
            used_bytes: 0,
        };
        cache.used_bytes = cache.scan_used_bytes()?;
        cache.evict_to_budget(0)?;
        Ok(cache)
    }

    fn get(&self, key: (u64, u64)) -> Option<Vec<u8>> {
        std::fs::read(self.path_for(key)).ok()
    }

    fn insert(&mut self, key: (u64, u64), data: &[u8]) -> Result<(), CogError> {
        let len = data.len() as u64;
        if len > self.budget_bytes {
            return Ok(());
        }
        self.evict_to_budget(len)?;
        let path = self.path_for(key);
        if let Ok(metadata) = std::fs::metadata(&path) {
            self.used_bytes = self.used_bytes.saturating_sub(metadata.len());
        }
        std::fs::write(&path, data)?;
        self.used_bytes += len;
        Ok(())
    }

    fn path_for(&self, key: (u64, u64)) -> PathBuf {
        self.dir.join(format!(
            "{:016x}_{:016x}_{:016x}.bin",
            self.source_hash, key.0, key.1
        ))
    }

    fn scan_used_bytes(&self) -> Result<u64, CogError> {
        let mut total = 0;
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(&format!("{:016x}_", self.source_hash))
            {
                total += entry.metadata()?.len();
            }
        }
        Ok(total)
    }

    fn evict_to_budget(&mut self, incoming_bytes: u64) -> Result<(), CogError> {
        if self.used_bytes + incoming_bytes <= self.budget_bytes {
            return Ok(());
        }
        let prefix = format!("{:016x}_", self.source_hash);
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if !entry.file_name().to_string_lossy().starts_with(&prefix) {
                continue;
            }
            let metadata = entry.metadata()?;
            let modified = metadata.modified().ok();
            files.push((entry.path(), metadata.len(), modified));
        }
        files.sort_by_key(|(_, _, modified)| *modified);

        for (path, len, _) in files {
            if self.used_bytes + incoming_bytes <= self.budget_bytes {
                break;
            }
            if std::fs::remove_file(path).is_ok() {
                self.used_bytes = self.used_bytes.saturating_sub(len);
            }
        }
        Ok(())
    }

    fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

impl ByteCache {
    fn new(budget_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            used_bytes: 0,
            budget_bytes,
        }
    }

    fn get(&mut self, key: &(u64, u64)) -> Option<Vec<u8>> {
        let data = self.entries.get(key)?.clone();
        self.lru.retain(|k| k != key);
        self.lru.push_back(*key);
        Some(data)
    }

    fn insert(&mut self, key: (u64, u64), data: Vec<u8>) {
        let len = data.len() as u64;
        if len > self.budget_bytes {
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.len() as u64);
            self.lru.retain(|k| k != &key);
        }
        while self.used_bytes + len > self.budget_bytes {
            let Some(old_key) = self.lru.pop_front() else {
                break;
            };
            if let Some(old) = self.entries.remove(&old_key) {
                self.used_bytes = self.used_bytes.saturating_sub(old.len() as u64);
            }
        }
        self.entries.insert(key, data);
        self.lru.push_back(key);
        self.used_bytes += len;
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used_bytes = 0;
    }

    fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
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
        if offset + length > self.file_size {
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
                .store(cache.used_bytes, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(disk_cache) = &self.disk_cache {
            if let Ok(mut cache) = disk_cache.lock() {
                cache.insert(cache_key, &data)?;
                self.stats
                    .disk_cached_bytes
                    .store(cache.used_bytes, std::sync::atomic::Ordering::Relaxed);
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

    pub fn cached_bytes(&self) -> u64 {
        self.cached_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn disk_cached_bytes(&self) -> u64 {
        self.disk_cached_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
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
}
