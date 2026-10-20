//! Byte-range caches backing `RangeReader`: an in-memory LRU (`ByteCache`) and
//! an optional on-disk cache (`DiskCache`), both keyed by exact
//! `(offset, length)` and bounded by an explicit byte budget.

use super::error::CogError;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

pub(crate) struct ByteCache {
    entries: HashMap<(u64, u64), Vec<u8>>,
    lru: VecDeque<(u64, u64)>,
    used_bytes: u64,
    budget_bytes: u64,
}

pub(crate) struct DiskCache {
    dir: PathBuf,
    source_prefix: String,
    budget_bytes: u64,
    used_bytes: u64,
}

pub(crate) fn source_cache_prefix(source: &str) -> String {
    let digest = Sha256::digest(source.as_bytes());
    let mut prefix: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
    prefix.push('_');
    prefix
}

impl DiskCache {
    pub(crate) fn new(dir: PathBuf, source: &str, budget_bytes: u64) -> Result<Self, CogError> {
        std::fs::create_dir_all(&dir)?;
        let mut cache = Self {
            dir,
            source_prefix: source_cache_prefix(source),
            budget_bytes,
            used_bytes: 0,
        };
        cache.used_bytes = cache.scan_used_bytes()?;
        cache.evict_to_budget(0)?;
        Ok(cache)
    }

    pub(crate) fn get(&self, key: (u64, u64)) -> Option<Vec<u8>> {
        let data = std::fs::read(self.path_for(key)).ok()?;
        (data.len() as u64 == key.1).then_some(data)
    }

    pub(crate) fn insert(&mut self, key: (u64, u64), data: &[u8]) -> Result<(), CogError> {
        let len = data.len() as u64;
        let path = self.path_for(key);
        if len > self.budget_bytes {
            return Ok(());
        }
        let existing_len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        self.used_bytes = self.used_bytes.saturating_sub(existing_len);
        let _ = std::fs::remove_file(&path);
        self.evict_to_budget(len)?;
        std::fs::write(&path, data)?;
        self.used_bytes = self.used_bytes.saturating_add(len);
        Ok(())
    }

    fn path_for(&self, key: (u64, u64)) -> PathBuf {
        self.dir.join(format!(
            "{}{:016x}_{:016x}.bin",
            self.source_prefix, key.0, key.1
        ))
    }

    fn scan_used_bytes(&self) -> Result<u64, CogError> {
        let mut total = 0u64;
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(&self.source_prefix)
            {
                total = total.saturating_add(entry.metadata()?.len());
            }
        }
        Ok(total)
    }

    fn evict_to_budget(&mut self, incoming_bytes: u64) -> Result<(), CogError> {
        if self.used_bytes.saturating_add(incoming_bytes) <= self.budget_bytes {
            return Ok(());
        }
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if !entry
                .file_name()
                .to_string_lossy()
                .starts_with(&self.source_prefix)
            {
                continue;
            }
            let metadata = entry.metadata()?;
            let modified = metadata.modified().ok();
            files.push((entry.path(), metadata.len(), modified));
        }
        files.sort_by_key(|(_, _, modified)| *modified);

        for (path, len, _) in files {
            if self.used_bytes.saturating_add(incoming_bytes) <= self.budget_bytes {
                break;
            }
            if std::fs::remove_file(path).is_ok() {
                self.used_bytes = self.used_bytes.saturating_sub(len);
            }
        }
        Ok(())
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub(crate) fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

impl ByteCache {
    pub(crate) fn new(budget_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            used_bytes: 0,
            budget_bytes,
        }
    }

    pub(crate) fn get(&mut self, key: &(u64, u64)) -> Option<Vec<u8>> {
        let data = self.entries.get(key)?.clone();
        self.lru.retain(|k| k != key);
        self.lru.push_back(*key);
        Some(data)
    }

    pub(crate) fn insert(&mut self, key: (u64, u64), data: Vec<u8>) {
        let len = data.len() as u64;
        if len > self.budget_bytes {
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.len() as u64);
            self.lru.retain(|k| k != &key);
        }
        while self.used_bytes.saturating_add(len) > self.budget_bytes {
            let Some(old_key) = self.lru.pop_front() else {
                break;
            };
            if let Some(old) = self.entries.remove(&old_key) {
                self.used_bytes = self.used_bytes.saturating_sub(old.len() as u64);
            }
        }
        self.entries.insert(key, data);
        self.lru.push_back(key);
        self.used_bytes = self.used_bytes.saturating_add(len);
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used_bytes = 0;
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub(crate) fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{source_cache_prefix, DiskCache};

    #[test]
    fn disk_cache_uses_deterministic_sha256_prefix_and_preserves_entries_on_overwrite() {
        // SHA-256("https://example.invalid/dem.tif") starts with:
        // 1103868b1a5d9982f23d661e71b47ad4...
        assert_eq!(
            source_cache_prefix("https://example.invalid/dem.tif"),
            "1103868b1a5d9982f23d661e71b47ad4_"
        );

        let dir =
            std::env::temp_dir().join(format!("forge3d-disk-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut cache = DiskCache::new(dir.clone(), "https://example.invalid/dem.tif", 64).unwrap();
        cache.insert((0, 32), &[1u8; 32]).unwrap();
        cache.insert((32, 32), &[2u8; 32]).unwrap();
        assert_eq!(cache.used_bytes(), 64);

        // Overwriting an existing key at full budget must replace in place
        // without evicting the other resident key.
        cache.insert((0, 32), &[3u8; 32]).unwrap();
        assert_eq!(cache.used_bytes(), 64);
        assert_eq!(cache.get((0, 32)).as_deref(), Some(&[3u8; 32][..]));
        assert_eq!(cache.get((32, 32)).as_deref(), Some(&[2u8; 32][..]));

        // Truncated/corrupted on-disk file for a key must be rejected on read.
        std::fs::write(cache.path_for((0, 32)), &[9u8; 12]).unwrap();
        assert_eq!(cache.get((0, 32)), None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
