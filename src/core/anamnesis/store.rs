//! Bounded, self-verifying filesystem content store.

use super::key::{
    reconstruct_pass_key, sha256, EngineFingerprint, InputKey, PassKey, PassKeyMaterial,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const BLOB_NAME: &str = "blob";
const META_NAME: &str = "meta.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KeyDerivation {
    Pass { material: PassKeyMaterial },
    Leaf { content_sha256: PassKey },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreMetadata {
    pub schema: String,
    pub key: PassKey,
    pub pass_label: String,
    pub input_keys: Vec<InputKey>,
    pub byte_length: u64,
    pub creation_engine_fingerprint: EngineFingerprint,
    pub self_hash: PassKey,
    pub created_unix_ms: u64,
    pub last_access_unix_ms: u64,
    pub derivation: KeyDerivation,
    #[serde(default)]
    pub frame: Option<i64>,
    #[serde(default)]
    pub measured_wall_ms: f64,
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
        self.put_measured(key, blob, derivation, None, 0.0)
    }

    pub fn put_measured(
        &self,
        key: PassKey,
        blob: &[u8],
        derivation: PassKeyMaterial,
        frame: Option<i64>,
        measured_wall_ms: f64,
    ) -> Result<StoreMetadata> {
        self.put_derivation(
            key,
            blob,
            derivation.label.clone(),
            derivation.input_keys.clone(),
            KeyDerivation::Pass {
                material: derivation,
            },
            frame,
            measured_wall_ms,
        )
    }

    pub fn put_leaf(
        &self,
        key: PassKey,
        blob: &[u8],
        label: impl Into<String>,
    ) -> Result<StoreMetadata> {
        if super::key::leaf_key(blob) != key {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "ANAMNESIS leaf key does not match leaf content",
            ));
        }
        self.put_derivation(
            key,
            blob,
            label.into(),
            Vec::new(),
            KeyDerivation::Leaf {
                content_sha256: sha256(blob),
            },
            None,
            0.0,
        )
    }

    fn put_derivation(
        &self,
        key: PassKey,
        blob: &[u8],
        pass_label: String,
        input_keys: Vec<InputKey>,
        derivation: KeyDerivation,
        frame: Option<i64>,
        measured_wall_ms: f64,
    ) -> Result<StoreMetadata> {
        let entry = self.entry_dir(key);
        if entry.is_dir() {
            if let Some((_, meta)) = self.get(key)? {
                return Ok(meta);
            }
        }
        fs::create_dir_all(entry.parent().expect("entry has prefix"))?;
        let temp = entry.with_extension(format!("tmp-{}", now_ms()));
        fs::create_dir_all(&temp)?;
        let now = now_ms();
        let metadata = StoreMetadata {
            schema: "forge3d.anamnesis.store/1".into(),
            key,
            pass_label,
            input_keys,
            byte_length: blob.len() as u64,
            creation_engine_fingerprint: EngineFingerprint::current(),
            self_hash: sha256(blob),
            created_unix_ms: now,
            last_access_unix_ms: now,
            derivation,
            frame,
            measured_wall_ms: measured_wall_ms.max(0.0),
        };
        fs::write(temp.join(BLOB_NAME), blob)?;
        fs::write(temp.join(META_NAME), json_bytes(&metadata)?)?;
        let staged_bytes = tree_bytes(&temp)?;
        if staged_bytes > self.max_bytes {
            fs::remove_dir_all(&temp)?;
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "ANAMNESIS entry requires {staged_bytes} bytes, exceeding max_bytes={}",
                    self.max_bytes
                ),
            ));
        }
        self.gc_for_staged(&temp, staged_bytes)?;
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
        self.enforce_bound(Some(entry.as_path()))?;
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
        let valid = metadata_is_valid(&meta, key, &blob);
        if !valid {
            self.quarantine(&entry, key)?;
            return Ok(None);
        }
        if self.verify_reads && !metadata_is_valid(&meta, key, &blob) {
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
                .map(|m| metadata_is_valid(&m, key, &blob))
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
        let mut total = tree_bytes(&self.root)?;
        for path in self.entries()? {
            if let Ok(meta) = read_meta(&path) {
                entries.push((meta.last_access_unix_ms, tree_bytes(&path)?, path));
            }
        }
        entries.extend(self.quarantine_entries()?);
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
        let mut removed = 0u64;
        for (_, bytes, path) in entries {
            if total <= target_bytes {
                break;
            }
            fs::remove_dir_all(path)?;
            total = tree_bytes(&self.root)?;
            removed = removed.saturating_add(bytes);
        }
        Ok(removed)
    }

    fn gc_for_staged(&self, temp: &Path, staged_bytes: u64) -> Result<()> {
        let mut candidates = Vec::new();
        for path in self.entries()? {
            if path != temp {
                let accessed = read_meta(&path)
                    .map(|meta| meta.last_access_unix_ms)
                    .unwrap_or_default();
                candidates.push((accessed, path));
            }
        }
        candidates.extend(
            self.quarantine_entries()?
                .into_iter()
                .map(|(accessed, _, path)| (accessed, path)),
        );
        candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (_, path) in candidates {
            let current = tree_bytes_excluding(&self.root, temp)?;
            if current.saturating_add(staged_bytes) <= self.max_bytes {
                return Ok(());
            }
            fs::remove_dir_all(path)?;
        }
        let projected = tree_bytes_excluding(&self.root, temp)?.saturating_add(staged_bytes);
        if projected > self.max_bytes {
            fs::remove_dir_all(temp)?;
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "ANAMNESIS store overhead plus entry requires {projected} bytes, exceeding max_bytes={}",
                    self.max_bytes
                ),
            ));
        }
        Ok(())
    }

    fn enforce_bound(&self, protected: Option<&Path>) -> Result<()> {
        if tree_bytes(&self.root)? <= self.max_bytes {
            return Ok(());
        }
        let mut candidates = self
            .entries()?
            .into_iter()
            .filter(|path| Some(path.as_path()) != protected)
            .collect::<Vec<_>>();
        candidates.sort_by_key(|path| {
            read_meta(path)
                .map(|meta| meta.last_access_unix_ms)
                .unwrap_or_default()
        });
        for path in candidates {
            if tree_bytes(&self.root)? <= self.max_bytes {
                return Ok(());
            }
            fs::remove_dir_all(path)?;
        }
        if tree_bytes(&self.root)? > self.max_bytes {
            if let Some(path) = protected {
                if path.exists() {
                    fs::remove_dir_all(path)?;
                }
            }
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "ANAMNESIS complete on-disk footprint exceeds max_bytes",
            ));
        }
        Ok(())
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

    fn quarantine_entries(&self) -> Result<Vec<(u64, u64, PathBuf)>> {
        let root = self.root.join("quarantine");
        let mut out = Vec::new();
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            if path.is_dir() {
                let accessed = path
                    .metadata()?
                    .modified()
                    .ok()
                    .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                    .map(|value| value.as_millis() as u64)
                    .unwrap_or_default();
                out.push((accessed, tree_bytes(&path)?, path));
            }
        }
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
    serde_json::to_vec(value).map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

fn read_meta(entry: &Path) -> Result<StoreMetadata> {
    serde_json::from_slice(&fs::read(entry.join(META_NAME))?)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

fn metadata_is_valid(meta: &StoreMetadata, key: PassKey, blob: &[u8]) -> bool {
    let derivation_key = match &meta.derivation {
        KeyDerivation::Pass { material } => reconstruct_pass_key(material).ok(),
        KeyDerivation::Leaf { content_sha256 } => {
            if *content_sha256 == sha256(blob) {
                Some(super::key::leaf_key(blob))
            } else {
                None
            }
        }
    };
    meta.schema == "forge3d.anamnesis.store/1"
        && meta.key == key
        && meta.byte_length == blob.len() as u64
        && meta.self_hash == sha256(blob)
        && derivation_key == Some(key)
}

fn tree_bytes(path: &Path) -> Result<u64> {
    let mut total = path.metadata()?.len();
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            total = total.saturating_add(tree_bytes(&entry?.path())?);
        }
    }
    Ok(total)
}

fn tree_bytes_excluding(path: &Path, excluded: &Path) -> Result<u64> {
    if path == excluded {
        return Ok(0);
    }
    let mut total = path.metadata()?.len();
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            total = total.saturating_add(tree_bytes_excluding(&entry?.path(), excluded)?);
        }
    }
    Ok(total)
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
            &[InputKey::new(
                "fixture@0",
                super::super::key::leaf_key(blob),
            )],
            b"caps",
            b"engine",
        )
    }

    #[test]
    fn corruption_is_quarantined_and_becomes_a_miss() {
        let root = scratch("corrupt");
        let store = ContentStore::new(&root, 16 * 1024, true).unwrap();
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
        let store = ContentStore::new(&root, 16 * 1024, false).unwrap();
        let (a, da) = derivation("a", b"aaaa");
        store.put(a, b"aaaa", da).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let (b, db) = derivation("b", b"bbbb");
        store.put(b, b"bbbb", db).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let (c, dc) = derivation("c", b"cccc");
        store.put(c, b"cccc", dc).unwrap();
        store
            .gc(tree_bytes(&root).unwrap() - tree_bytes(&store.entry_dir(a)).unwrap())
            .unwrap();
        assert!(store.get(a).unwrap().is_none());
        assert!(store.get(b).unwrap().is_some());
        assert!(store.get(c).unwrap().is_some());
        assert!(tree_bytes(&root).unwrap() <= store.max_bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn complete_footprint_never_exceeds_bound() {
        let root = scratch("hard-bound");
        let store = ContentStore::new(&root, 16 * 1024, false).unwrap();
        for index in 0..8 {
            let blob = vec![index; 256];
            let (key, material) = derivation(&format!("pass-{index}"), &blob);
            store.put(key, &blob, material).unwrap();
            assert!(tree_bytes(&root).unwrap() <= 16 * 1024);
        }
        fs::remove_dir_all(root).unwrap();
    }
}
