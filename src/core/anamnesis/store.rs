//! Bounded, self-verifying filesystem content store.

use super::key::{sha256, EngineFingerprint, PassKey, PassKeyMaterial};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const BLOB_NAME: &str = "blob";
const META_NAME: &str = "meta.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreMetadata {
    pub schema: String,
    pub key: PassKey,
    pub pass_label: String,
    pub input_keys: Vec<PassKey>,
    pub byte_length: u64,
    pub creation_engine_fingerprint: EngineFingerprint,
    pub self_hash: PassKey,
    pub created_unix_ms: u64,
    pub last_access_unix_ms: u64,
    pub derivation: PassKeyMaterial,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifyReport {
    pub valid: usize,
    pub quarantined: usize,
    pub bytes_checked: u64,
}

#[derive(Clone, Debug)]
pub struct ContentStore {
    root: PathBuf,
    max_bytes: u64,
    verify_reads: bool,
}

impl ContentStore {
    pub fn new(root: impl Into<PathBuf>, max_bytes: u64, verify_reads: bool) -> Result<Self> {
        if max_bytes == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "ANAMNESIS max_bytes must be positive",
            ));
        }
        let store = Self {
            root: root.into(),
            max_bytes,
            verify_reads,
        };
        fs::create_dir_all(&store.root)?;
        fs::create_dir_all(store.root.join("quarantine"))?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn entry_dir(&self, key: PassKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(hex)
    }

    pub fn put(
        &self,
        key: PassKey,
        blob: &[u8],
        derivation: PassKeyMaterial,
    ) -> Result<StoreMetadata> {
        if blob.len() as u64 > self.max_bytes {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "ANAMNESIS blob is larger than the entire store budget",
            ));
        }
        let entry = self.entry_dir(key);
        if entry.is_dir() {
            if let Some((_, meta)) = self.get(key)? {
                return Ok(meta);
            }
        }
        self.gc_to(self.max_bytes.saturating_sub(blob.len() as u64))?;
        fs::create_dir_all(entry.parent().expect("entry has prefix"))?;
        let temp = entry.with_extension(format!("tmp-{}", now_ms()));
        fs::create_dir_all(&temp)?;
        let now = now_ms();
        let metadata = StoreMetadata {
            schema: "forge3d.anamnesis.store/1".into(),
            key,
            pass_label: derivation.label.clone(),
            input_keys: derivation.input_keys.clone(),
            byte_length: blob.len() as u64,
            creation_engine_fingerprint: EngineFingerprint::current(),
            self_hash: sha256(blob),
            created_unix_ms: now,
            last_access_unix_ms: now,
            derivation,
        };
        fs::write(temp.join(BLOB_NAME), blob)?;
        fs::write(temp.join(META_NAME), json_bytes(&metadata)?)?;
        match fs::rename(&temp, &entry) {
            Ok(()) => {}
            Err(error) if entry.is_dir() => {
                fs::remove_dir_all(&temp)?;
                let _ = error;
            }
            Err(error) => {
                let _ = fs::remove_dir_all(&temp);
                return Err(error);
            }
        }
        Ok(metadata)
    }

    pub fn get(&self, key: PassKey) -> Result<Option<(Vec<u8>, StoreMetadata)>> {
        let entry = self.entry_dir(key);
        if !entry.is_dir() {
            return Ok(None);
        }
        let blob = match fs::read(entry.join(BLOB_NAME)) {
            Ok(blob) => blob,
            Err(_) => {
                self.quarantine(&entry, key)?;
                return Ok(None);
            }
        };
        let mut meta = match read_meta(&entry) {
            Ok(meta) => meta,
            Err(_) => {
                self.quarantine(&entry, key)?;
                return Ok(None);
            }
        };
        let valid = meta.key == key
            && meta.byte_length == blob.len() as u64
            && meta.self_hash == sha256(&blob);
        if !valid {
            self.quarantine(&entry, key)?;
            return Ok(None);
        }
        if self.verify_reads && meta.self_hash != sha256(&blob) {
            self.quarantine(&entry, key)?;
            return Ok(None);
        }
        meta.last_access_unix_ms = now_ms();
        fs::write(entry.join(META_NAME), json_bytes(&meta)?)?;
        Ok(Some((blob, meta)))
    }

    pub fn explain(&self, key: PassKey) -> Result<Option<StoreMetadata>> {
        let entry = self.entry_dir(key);
        if !entry.is_dir() {
            return Ok(None);
        }
        Ok(Some(read_meta(&entry)?))
    }

    pub fn verify(&self) -> Result<VerifyReport> {
        let mut report = VerifyReport::default();
        for entry in self.entries()? {
            let key = match entry
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(|s| PassKey::from_hex(s).ok())
            {
                Some(key) => key,
                None => continue,
            };
            let blob = fs::read(entry.join(BLOB_NAME)).unwrap_or_default();
            report.bytes_checked += blob.len() as u64;
            let valid = read_meta(&entry)
                .map(|m| {
                    m.key == key
                        && m.byte_length == blob.len() as u64
                        && m.self_hash == sha256(&blob)
                })
                .unwrap_or(false);
            if valid {
                report.valid += 1;
            } else {
                self.quarantine(&entry, key)?;
                report.quarantined += 1;
            }
        }
        Ok(report)
    }

    pub fn gc(&self, max_bytes: u64) -> Result<u64> {
        self.gc_to(max_bytes)
    }

    fn gc_to(&self, target_bytes: u64) -> Result<u64> {
        let mut entries = Vec::new();
        let mut total = 0u64;
        for path in self.entries()? {
            if let Ok(meta) = read_meta(&path) {
                total = total.saturating_add(meta.byte_length);
                entries.push((meta.last_access_unix_ms, meta.byte_length, path));
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
        let mut removed = 0u64;
        for (_, bytes, path) in entries {
            if total <= target_bytes {
                break;
            }
            fs::remove_dir_all(path)?;
            total = total.saturating_sub(bytes);
            removed = removed.saturating_add(bytes);
        }
        Ok(removed)
    }

    fn entries(&self) -> Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        for prefix in fs::read_dir(&self.root)? {
            let prefix = prefix?;
            let name = prefix.file_name();
            if name == "quarantine" || !prefix.path().is_dir() {
                continue;
            }
            for entry in fs::read_dir(prefix.path())? {
                let entry = entry?;
                if entry.path().is_dir() {
                    out.push(entry.path());
                }
            }
        }
        out.sort();
        Ok(out)
    }

    fn quarantine(&self, entry: &Path, key: PassKey) -> Result<()> {
        let target = self
            .root
            .join("quarantine")
            .join(format!("{}-{}", key, now_ms()));
        fs::rename(entry, target)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(value).map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

fn read_meta(entry: &Path) -> Result<StoreMetadata> {
    serde_json::from_slice(&fs::read(entry.join(META_NAME))?)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("forge3d-anamnesis-{name}-{}", now_ms()))
    }

    fn derivation(label: &str, blob: &[u8]) -> (PassKey, PassKeyMaterial) {
        super::super::key::pass_key(
            label,
            b"pipeline",
            b"uniform",
            &[super::super::key::leaf_key(blob)],
            b"caps",
            b"engine",
        )
    }

    #[test]
    fn corruption_is_quarantined_and_becomes_a_miss() {
        let root = scratch("corrupt");
        let store = ContentStore::new(&root, 1024, true).unwrap();
        let (key, material) = derivation("pass", b"good");
        store.put(key, b"good", material).unwrap();
        fs::write(store.entry_dir(key).join(BLOB_NAME), b"hood").unwrap();
        let report = store.verify().unwrap();
        assert_eq!(report.quarantined, 1);
        assert!(store.get(key).unwrap().is_none());
        assert_eq!(fs::read_dir(root.join("quarantine")).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lru_eviction_respects_bound() {
        let root = scratch("lru");
        let store = ContentStore::new(&root, 8, false).unwrap();
        let (a, da) = derivation("a", b"aaaa");
        store.put(a, b"aaaa", da).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let (b, db) = derivation("b", b"bbbb");
        store.put(b, b"bbbb", db).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let (c, dc) = derivation("c", b"cccc");
        store.put(c, b"cccc", dc).unwrap();
        assert!(store.get(a).unwrap().is_none());
        assert!(store.get(b).unwrap().is_some());
        assert!(store.get(c).unwrap().is_some());
        fs::remove_dir_all(root).unwrap();
    }
}
