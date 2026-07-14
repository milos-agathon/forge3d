//! Byte-range caches backing `RangeReader`: an in-memory LRU (`ByteCache`) and
//! an optional on-disk cache (`DiskCache`), both keyed by exact
//! `(offset, length)` and bounded by an explicit byte budget.

use super::error::CogError;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

pub(crate) struct ByteCache {
    entries: HashMap<(u64, u64), Vec<u8>>,
    lru: VecDeque<(u64, u64)>,
    used_bytes: u64,
    budget_bytes: u64,
}

pub(crate) struct DiskCache {
    dir: PathBuf,
    source_hash: u64,
    budget_bytes: u64,
    used_bytes: u64,
}

impl DiskCache {
    pub(crate) fn new(dir: PathBuf, source: &str, budget_bytes: u64) -> Result<Self, CogError> {
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

    pub(crate) fn get(&self, key: (u64, u64)) -> Option<Vec<u8>> {
        std::fs::read(self.path_for(key)).ok()
    }

    pub(crate) fn insert(&mut self, key: (u64, u64), data: &[u8]) -> Result<(), CogError> {
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
