//! Bounded, self-verifying filesystem content store.

use super::key::{
    reconstruct_pass_key, sha256, EngineFingerprint, InputKey, PassKey, PassKeyMaterial,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

const ACCESS_LOG: &str = "access.log";

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
    admission_cleanup_bytes: Arc<AtomicU64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PackedRecord {
    offset: u64,
    length: u64,
    metadata: StoreMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PackedIndex {
    schema: String,
    entries: BTreeMap<String, PackedRecord>,
}

impl Default for PackedIndex {
    fn default() -> Self {
        Self {
            schema: "forge3d.anamnesis.packed-index/2".into(),
            entries: BTreeMap::new(),
        }
    }
}

/// Immutable packed bytes parsed ahead of a frozen read set. Construction
/// performs no entry verification; required keys are verified on demand.
#[derive(Debug)]
pub(crate) struct PackedReadSnapshot {
    index: PackedIndex,
}

impl PackedReadSnapshot {
    pub(crate) fn from_index(index: &[u8]) -> Result<Self> {
        let index: PackedIndex = serde_json::from_slice(index)
            .map_err(|error| Error::new(ErrorKind::InvalidData, error))?;
        if index.schema != "forge3d.anamnesis.packed-index/2" {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "unsupported packed index",
            ));
        }
        Ok(Self { index })
    }

    pub(crate) fn verify<'a>(
        &self,
        keys: &[PassKey],
        data: &'a [u8],
    ) -> Vec<(String, Option<&'a [u8]>)> {
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(4);
        if keys.len() < 256 || workers == 1 {
            return self.verify_chunk(keys, data);
        }
        // The caller has frozen this read set. Every chunk uses the same
        // integrity predicate over immutable owned bytes, and all workers
        // join before returning. Keep small reads and thread-spawn failures
        // on the serial path, without dropping or trusting any requested key.
        std::thread::scope(|scope| {
            let chunks: Vec<_> = keys.chunks(keys.len().div_ceil(workers)).collect();
            let pending: Vec<_> = chunks[..chunks.len() - 1]
                .iter()
                .map(|chunk| {
                    let chunk = *chunk;
                    let handle = std::thread::Builder::new()
                        .spawn_scoped(scope, move || self.verify_chunk(chunk, data))
                        .ok();
                    (handle, chunk)
                })
                .collect();
            let last = self.verify_chunk(chunks[chunks.len() - 1], data);
            let mut verified = Vec::with_capacity(keys.len());
            for (handle, chunk) in pending {
                verified.extend(match handle {
                    Some(handle) => handle.join().expect("packed verification worker panicked"),
                    None => self.verify_chunk(chunk, data),
                });
            }
            verified.extend(last);
            verified
        })
    }

    fn verify_chunk<'a>(
        &self,
        keys: &[PassKey],
        data: &'a [u8],
    ) -> Vec<(String, Option<&'a [u8]>)> {
        keys.iter()
            .filter_map(|key| {
                let name = key.to_hex();
                self.index.entries.get(&name).map(|record| {
                    let blob = ContentStore::slice(record, data)
                        .filter(|blob| metadata_is_valid(&record.metadata, *key, blob));
                    (name, blob)
                })
            })
            .collect()
    }
}

impl ContentStore {
    const DATA: &'static str = "data.v2.bin";
    const INDEX: &'static str = "index.v2.json";

    pub fn new(root: impl Into<PathBuf>, max_bytes: u64, verify_reads: bool) -> Result<Self> {
        if max_bytes == 0 {
            return Err(bound_error());
        }
        let store = Self {
            root: root.into(),
            max_bytes,
            verify_reads,
            admission_cleanup_bytes: Arc::new(AtomicU64::new(0)),
        };
        if store.root.try_exists()? {
            store.admit()?;
        } else {
            if directory_bytes(store.root.parent().unwrap_or(Path::new(".")))?.saturating_mul(2)
                > max_bytes
            {
                return Err(bound_error());
            }
            fs::create_dir_all(&store.root)?;
        }
        let quarantine = store.root.join("quarantine");
        if !quarantine.try_exists()? {
            let growth = if cfg!(windows) && tree_bytes(&store.root)? == 0 {
                0
            } else {
                directory_bytes(&store.root)? + index_bytes(&store.root)?
            };
            store.reserve(growth, None)?;
            store.checkpoint("quarantine directory", true, 0)?;
            fs::create_dir(quarantine)?;
            store.checkpoint("quarantine directory", false, 0)?;
        }
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn admission_cleanup_bytes(&self) -> u64 {
        self.admission_cleanup_bytes.load(Ordering::Relaxed)
    }

    fn invalid_index(&self) -> Result<Option<Vec<u8>>> {
        let bytes = match fs::read(self.root.join(Self::INDEX)) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(self.root.join(Self::DATA).try_exists()?.then(Vec::new));
            }
            Err(error) => return Err(error),
        };
        let valid = serde_json::from_slice::<PackedIndex>(&bytes)
            .map(|index| index.schema == PackedIndex::default().schema)
            .unwrap_or(false)
            && self.root.join(Self::DATA).try_exists()?;
        Ok((!valid).then_some(bytes))
    }

    fn quarantine_index(&self, raw: &[u8]) -> Result<()> {
        self.invalidate_manifest()?;
        for name in [Self::INDEX, Self::DATA] {
            match fs::remove_file(self.root.join(name)) {
                Ok(()) => (),
                Err(error) if error.kind() == ErrorKind::NotFound => (),
                Err(error) => return Err(error),
            }
        }
        let mut packet = b"ANAMNESIS invalid packed index\n".to_vec();
        packet.extend_from_slice(raw);
        let quarantine = self.root.join("quarantine");
        let growth = if cfg!(windows)
            && quarantine.metadata()?.len() == 0
            && fs::read_dir(&quarantine)?.next().is_none()
        {
            0
        } else {
            2 * index_bytes(&self.root)?
        };
        self.reserve(packet.len() as u64 + growth, None)?;
        self.write_file(
            &quarantine.join(format!("index-{}.invalid", now_ms())),
            &packet,
            "quarantine index",
        )
    }

    fn load(&self) -> Result<(PackedIndex, Vec<u8>)> {
        // Version 1 directories are deliberately never read or migrated.
        let index = fs::read(self.root.join(Self::INDEX))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PackedIndex>(&bytes).ok())
            .filter(|index| index.schema == PackedIndex::default().schema)
            .unwrap_or_default();
        let data = fs::read(self.root.join(Self::DATA)).unwrap_or_default();
        Ok((index, data))
    }

    fn slice<'a>(record: &PackedRecord, data: &'a [u8]) -> Option<&'a [u8]> {
        let start = usize::try_from(record.offset).ok()?;
        let end = usize::try_from(record.offset.checked_add(record.length)?).ok()?;
        data.get(start..end)
    }

    fn compact(index: &PackedIndex, data: &[u8]) -> (PackedIndex, Vec<u8>) {
        let mut packed = Vec::new();
        let mut rewritten = index.clone();
        for record in rewritten.entries.values_mut() {
            let blob = Self::slice(record, data).unwrap_or_default();
            record.offset = packed.len() as u64;
            record.length = blob.len() as u64;
            packed.extend_from_slice(blob);
        }
        (rewritten, packed)
    }

    fn publish(&self, index: &PackedIndex, data: &[u8]) -> Result<()> {
        let encoded = json_bytes(index)?;
        let mut base = tree_bytes(&self.root)?;
        let mut creation = 0u64;
        for name in [Self::DATA, Self::INDEX] {
            match self.root.join(name).metadata() {
                Ok(meta) => base = base.saturating_sub(meta.len()),
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    creation += index_bytes(&self.root)?
                }
                Err(error) => return Err(error),
            }
        }
        if base
            .saturating_add(creation)
            .saturating_add(data.len() as u64)
            .saturating_add(encoded.len() as u64)
            > self.max_bytes
        {
            return Err(bound_error());
        }
        // Invalidate the index before relocating bytes. A torn publication is
        // a miss, never an entry whose unverified bytes can be served. Shrink
        // first so admission cleanup has removed its excess before any write.
        for name in [Self::INDEX, Self::DATA] {
            let path = self.root.join(name);
            if path.try_exists()? {
                fs::OpenOptions::new().write(true).open(path)?.set_len(0)?;
            }
        }
        self.write_file(&self.root.join(Self::DATA), data, "payload")?;
        self.write_file(&self.root.join(Self::INDEX), &encoded, "packed index")
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
        self.admit()?;
        if let Some((_, meta)) = self.get(key)? {
            return Ok(meta);
        }
        let now = now_ms();
        let metadata = StoreMetadata {
            schema: "forge3d.anamnesis.store/2".into(),
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
        loop {
            let (mut index, mut data) = self.load()?;
            index.entries.insert(
                key.to_hex(),
                PackedRecord {
                    offset: data.len() as u64,
                    length: blob.len() as u64,
                    metadata: metadata.clone(),
                },
            );
            let encoded = json_bytes(&index)?;
            let previous_index = self
                .root
                .join(Self::INDEX)
                .metadata()
                .map(|m| m.len())
                .unwrap_or(0);
            let mut growth = (blob.len() as u64)
                .saturating_add((encoded.len() as u64).saturating_sub(previous_index));
            for name in [Self::DATA, Self::INDEX] {
                if !self.root.join(name).try_exists()? {
                    growth += index_bytes(&self.root)?;
                }
            }
            let before = tree_bytes(&self.root)?;
            if before.saturating_add(growth) <= self.max_bytes {
                data.extend_from_slice(blob);
                self.publish(&index, &data)?;
                return Ok(metadata);
            }
            self.gc_packed(self.max_bytes.saturating_sub(growth), None, true)?;
            if tree_bytes(&self.root)? >= before {
                return Err(bound_error());
            }
        }
    }

    pub fn get(&self, key: PassKey) -> Result<Option<(Vec<u8>, StoreMetadata)>> {
        self.admit()?;
        if let Some(raw) = self.invalid_index()? {
            self.quarantine_index(&raw)?;
            return Ok(None);
        }
        let (mut index, data) = self.load()?;
        let Some(record) = index.entries.get(&key.to_hex()) else {
            return Ok(None);
        };
        let valid = Self::slice(record, &data)
            .filter(|blob| metadata_is_valid(&record.metadata, key, blob));
        let Some(blob) = valid else {
            self.quarantine(key, &index, &data)?;
            return Ok(None);
        };
        // Integrity is unconditional, including when verify_reads=false.
        let _ = self.verify_reads;
        let blob = blob.to_vec();
        let mut meta = record.metadata.clone();
        meta.last_access_unix_ms = now_ms();
        if let Some(record) = index.entries.get_mut(&key.to_hex()) {
            record.metadata = meta.clone();
        }
        let encoded = json_bytes(&index)?;
        let old_len = self.root.join(Self::INDEX).metadata()?.len();
        self.reserve((encoded.len() as u64).saturating_sub(old_len), Some(key))?;
        // GC may have compacted offsets while reserving access-clock growth.
        let (mut latest, _) = self.load()?;
        if let Some(record) = latest.entries.get_mut(&key.to_hex()) {
            record.metadata = meta.clone();
        }
        self.write_file(
            &self.root.join(Self::INDEX),
            &json_bytes(&latest)?,
            "access metadata",
        )?;
        Ok(Some((blob, meta)))
    }

    pub fn explain(&self, key: PassKey) -> Result<Option<StoreMetadata>> {
        let (index, data) = self.load()?;
        Ok(index.entries.get(&key.to_hex()).and_then(|record| {
            Self::slice(record, &data)
                .filter(|blob| metadata_is_valid(&record.metadata, key, blob))
                .map(|_| record.metadata.clone())
        }))
    }

    pub fn verify(&self) -> Result<VerifyReport> {
        self.admit()?;
        if let Some(raw) = self.invalid_index()? {
            self.quarantine_index(&raw)?;
            return Ok(VerifyReport {
                quarantined: 1,
                ..VerifyReport::default()
            });
        }
        let (index, data) = self.load()?;
        let mut report = VerifyReport::default();
        for (name, record) in &index.entries {
            let Ok(key) = PassKey::from_hex(name) else {
                continue;
            };
            let blob = Self::slice(record, &data);
            report.bytes_checked += blob.map_or(0, |bytes| bytes.len()) as u64;
            if blob
                .map(|bytes| metadata_is_valid(&record.metadata, key, bytes))
                .unwrap_or(false)
            {
                report.valid += 1;
            } else {
                let (current, bytes) = self.load()?;
                self.quarantine(key, &current, &bytes)?;
                report.quarantined += 1;
            }
        }
        Ok(report)
    }

    pub fn gc(&self, max_bytes: u64) -> Result<u64> {
        self.admit()?;
        self.gc_packed(max_bytes, None, false)
    }

    fn gc_packed(&self, target: u64, protected: Option<PassKey>, keep_files: bool) -> Result<u64> {
        let before = tree_bytes(&self.root)?;
        let (mut index, data) = self.load()?;
        let clocks = self.access_times()?;
        let mut order: Vec<_> = index
            .entries
            .iter()
            .map(|(name, record)| {
                (
                    record.metadata.last_access_unix_ms.max(
                        clocks
                            .get(&record.metadata.key)
                            .copied()
                            .unwrap_or_default(),
                    ),
                    name.clone(),
                )
            })
            .collect();
        order.sort();
        let file_size = |name| {
            self.root
                .join(name)
                .metadata()
                .map(|m| m.len())
                .unwrap_or(0)
        };
        let mut base = before
            .saturating_sub(file_size(Self::DATA))
            .saturating_sub(file_size(Self::INDEX));
        for (_, name) in order {
            let (compacted, packed) = Self::compact(&index, &data);
            if base + packed.len() as u64 + json_bytes(&compacted)?.len() as u64 <= target {
                break;
            }
            if protected.map(|key| key.to_hex() == name).unwrap_or(false) {
                continue;
            }
            index.entries.remove(&name);
        }
        // Old schemas, quarantine and staging remain part of the complete bound.
        let mut extras = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.path().is_dir() && entry.file_name().len() == 2 {
                extras.push(entry.path());
            }
        }
        if self.root.join("quarantine").try_exists()? {
            extras.extend(
                fs::read_dir(self.root.join("quarantine"))?
                    .map(|e| e.map(|e| e.path()))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        extras.sort_by_key(|path| path.metadata().and_then(|m| m.modified()).ok());
        let (compacted, packed) = Self::compact(&index, &data);
        for path in extras {
            if base + packed.len() as u64 + json_bytes(&compacted)?.len() as u64 <= target {
                break;
            }
            let size = tree_bytes(&path)?;
            self.invalidate_manifest()?;
            if path.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
            base = base.saturating_sub(size);
        }
        if before > target {
            self.invalidate_manifest()?;
            if index.entries.is_empty() && !keep_files {
                for name in [Self::INDEX, Self::DATA] {
                    match fs::remove_file(self.root.join(name)) {
                        Ok(()) => (),
                        Err(e) if e.kind() == ErrorKind::NotFound => (),
                        Err(e) => return Err(e),
                    }
                }
            } else {
                self.publish(&compacted, &packed)?;
            }
        }
        Ok(before.saturating_sub(tree_bytes(&self.root)?))
    }

    fn admit(&self) -> Result<()> {
        let actual = tree_bytes(&self.root)?;
        if actual > self.max_bytes {
            let excess = actual - self.max_bytes;
            self.admission_cleanup_bytes
                .fetch_add(excess, Ordering::Relaxed);
            eprintln!("ANAMNESIS admission_cleanup starting_excess_bytes={excess}");
            self.gc_packed(self.max_bytes, None, false)?;
        }
        if tree_bytes(&self.root)? > self.max_bytes {
            return Err(bound_error());
        }
        Ok(())
    }

    fn reserve(&self, additional: u64, protected: Option<PassKey>) -> Result<()> {
        self.admit()?;
        if tree_bytes(&self.root)?.saturating_add(additional) > self.max_bytes {
            self.gc_packed(self.max_bytes.saturating_sub(additional), protected, true)?;
        }
        if tree_bytes(&self.root)?.saturating_add(additional) > self.max_bytes {
            return Err(bound_error());
        }
        Ok(())
    }

    fn quarantine(&self, key: PassKey, index: &PackedIndex, data: &[u8]) -> Result<()> {
        let Some(record) = index.entries.get(&key.to_hex()) else {
            return Ok(());
        };
        let mut packet = json_bytes(record)?;
        packet.push(b'\n');
        packet.extend_from_slice(Self::slice(record, data).unwrap_or_default());
        let mut retained = index.clone();
        retained.entries.remove(&key.to_hex());
        let (retained, bytes) = Self::compact(&retained, data);
        self.invalidate_manifest()?;
        self.publish(&retained, &bytes)?;
        let quarantine = self.root.join("quarantine");
        let growth = if cfg!(windows)
            && quarantine.metadata()?.len() == 0
            && fs::read_dir(&quarantine)?.next().is_none()
        {
            0
        } else {
            2 * index_bytes(&self.root)?
        };
        self.reserve(packet.len() as u64 + growth, None)?;
        self.write_file(
            &quarantine.join(format!("{}-{}", key, now_ms())),
            &packet,
            "quarantine",
        )
    }

    fn invalidate_manifest(&self) -> Result<()> {
        match fs::remove_file(self.root.join("last_manifest.json")) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn checkpoint(&self, operation: &'static str, before: bool, growth: u64) -> Result<()> {
        let actual = tree_bytes(&self.root)?;
        #[cfg(test)]
        WRITE_OBSERVATIONS.with(|rows| {
            rows.borrow_mut().push((
                self.root.clone(),
                operation,
                before,
                actual,
                growth,
                self.max_bytes,
            ))
        });
        #[cfg(not(test))]
        let _ = (operation, before);
        if actual.saturating_add(growth) > self.max_bytes {
            return Err(bound_error());
        }
        Ok(())
    }

    fn write_file(&self, path: &Path, bytes: &[u8], operation: &'static str) -> Result<()> {
        let old = path.metadata().map(|m| m.len()).unwrap_or(0);
        self.checkpoint(operation, true, (bytes.len() as u64).saturating_sub(old))?;
        fs::write(path, bytes)?;
        self.checkpoint(operation, false, 0)
    }

    fn access_times(&self) -> Result<BTreeMap<PassKey, u64>> {
        let mut result = BTreeMap::new();
        let value = fs::read_to_string(self.root.join(ACCESS_LOG)).unwrap_or_default();
        for line in value.lines() {
            let Some((timestamp, key)) = line.split_once('\t') else {
                continue;
            };
            if let (Ok(timestamp), Ok(key)) = (timestamp.parse::<u64>(), PassKey::from_hex(key)) {
                result
                    .entry(key)
                    .and_modify(|current: &mut u64| *current = (*current).max(timestamp))
                    .or_insert(timestamp);
            }
        }
        Ok(result)
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
    meta.schema == "forge3d.anamnesis.store/2"
        && meta.key == key
        && meta.byte_length == blob.len() as u64
        && meta.self_hash == sha256(blob)
        && derivation_key == Some(key)
}

fn tree_bytes(path: &Path) -> Result<u64> {
    let metadata = path.metadata()?;
    let mut total = metadata.len();
    if metadata.is_dir() {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Query a directory handle, as Python os.stat does: Windows
            // enumeration attributes can omit allocated NTFS index bytes.
            total = fs::OpenOptions::new()
                .read(true)
                .custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS)
                .open(path)?
                .metadata()?
                .len();
        }
        for entry in fs::read_dir(path)? {
            total = total.saturating_add(tree_bytes(&entry?.path())?);
        }
    }
    Ok(total)
}

fn bound_error() -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "ANAMNESIS complete on-disk footprint exceeds max_bytes",
    )
}

fn index_bytes(path: &Path) -> Result<u64> {
    #[cfg(windows)]
    {
        let _ = path;
        // Measured NTFS directory-index allocation in the reviewed store.
        Ok(4096)
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let ancestor = path
            .ancestors()
            .find(|path| path.is_dir())
            .unwrap_or(Path::new("."));
        Ok(ancestor.metadata()?.blksize())
    }
    #[cfg(not(any(windows, unix)))]
    {
        Ok(path.metadata()?.len())
    }
}

fn directory_bytes(path: &Path) -> Result<u64> {
    if cfg!(windows) {
        Ok(0)
    } else {
        index_bytes(path)
    }
}

#[cfg(test)]
type WriteObservation = (PathBuf, &'static str, bool, u64, u64, u64);

#[cfg(test)]
thread_local! {
    static WRITE_OBSERVATIONS: std::cell::RefCell<Vec<WriteObservation>> = const { std::cell::RefCell::new(Vec::new()) };
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
        let store = ContentStore::new(&root, 64 * 1024, true).unwrap();
        let (key, material) = derivation("pass", b"good");
        store.put(key, b"good", material).unwrap();
        fs::write(root.join(ContentStore::DATA), b"hood").unwrap();
        let report = store.verify().unwrap();
        assert_eq!(report.quarantined, 1);
        assert!(store.get(key).unwrap().is_none());
        assert_eq!(fs::read_dir(root.join("quarantine")).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lru_eviction_respects_bound() {
        let root = scratch("lru");
        let store = ContentStore::new(&root, 64 * 1024, false).unwrap();
        let (a, da) = derivation("a", b"aaaa");
        store.put(a, b"aaaa", da).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let (b, db) = derivation("b", b"bbbb");
        store.put(b, b"bbbb", db).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let (c, dc) = derivation("c", b"cccc");
        store.put(c, b"cccc", dc).unwrap();
        fs::write(
            root.join("last_manifest.json"),
            format!("{{\"entries\":{{\"0:a\":\"{a}\"}}}}"),
        )
        .unwrap();
        store
            .gc({
                let (mut index, data) = store.load().unwrap();
                index.entries.remove(&a.to_hex());
                let (index, data) = ContentStore::compact(&index, &data);
                tree_bytes(&root).unwrap()
                    - root.join(ContentStore::DATA).metadata().unwrap().len()
                    - root.join(ContentStore::INDEX).metadata().unwrap().len()
                    + data.len() as u64
                    + json_bytes(&index).unwrap().len() as u64
            })
            .unwrap();
        assert!(store.get(a).unwrap().is_none());
        assert!(!root.join("last_manifest.json").exists());
        assert!(store.get(b).unwrap().is_some());
        assert!(store.get(c).unwrap().is_some());
        assert!(tree_bytes(&root).unwrap() <= store.max_bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_gc_honors_python_batched_access_journal() {
        let root = scratch("shared-lru-journal");
        let store = ContentStore::new(&root, 64 * 1024, false).unwrap();
        let (a, da) = derivation("a", b"aaaa");
        store.put(a, b"aaaa", da).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let (b, db) = derivation("b", b"bbbb");
        store.put(b, b"bbbb", db).unwrap();
        fs::write(
            root.join(ACCESS_LOG),
            format!("{}\t{}\n", now_ms() + 1_000, a),
        )
        .unwrap();

        store
            .gc({
                let (mut index, data) = store.load().unwrap();
                index.entries.remove(&b.to_hex());
                let (index, data) = ContentStore::compact(&index, &data);
                tree_bytes(&root).unwrap()
                    - root.join(ContentStore::DATA).metadata().unwrap().len()
                    - root.join(ContentStore::INDEX).metadata().unwrap().len()
                    + data.len() as u64
                    + json_bytes(&index).unwrap().len() as u64
            })
            .unwrap();

        assert!(store.get(a).unwrap().is_some());
        assert!(store.get(b).unwrap().is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn complete_footprint_never_exceeds_bound() {
        let root = scratch("hard-bound");
        let max_bytes = 64 * 1024;
        let store = ContentStore::new(&root, max_bytes, false).unwrap();
        for index in 0..8 {
            let blob = vec![index; 256];
            let (key, material) = derivation(&format!("pass-{index}"), &blob);
            store.put(key, &blob, material).unwrap();
            assert!(tree_bytes(&root).unwrap() <= max_bytes);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn writes_reserve_before_payload_at_exact_capacity_and_after_admission() {
        for mode in ["exact-capacity", "reopened", "first-use"] {
            let root = scratch(mode);
            let mut store = ContentStore::new(&root, u64::MAX, false).unwrap();
            for byte in [1, 2] {
                let blob = vec![byte; 4096];
                store
                    .put_leaf(
                        super::super::key::leaf_key(&blob),
                        &blob,
                        "admission-fixture",
                    )
                    .unwrap();
            }
            let filled = tree_bytes(&root).unwrap();
            let lowered = if mode == "exact-capacity" {
                filled
            } else {
                filled - 4096
            };
            WRITE_OBSERVATIONS.with(|rows| rows.borrow_mut().clear());
            if mode == "reopened" {
                store = ContentStore::new(&root, lowered, false).unwrap();
                assert!(tree_bytes(&root).unwrap() <= lowered);
                assert_eq!(store.admission_cleanup_bytes(), 4096);
            } else {
                store.max_bytes = lowered;
                assert_eq!(tree_bytes(&root).unwrap(), filled);
            }
            let free = lowered.saturating_sub(tree_bytes(&root).unwrap());
            let blob = vec![3; 4096usize.max(usize::try_from(free + 1).unwrap())];
            assert!(blob.len() as u64 > free);
            store
                .put_leaf(
                    super::super::key::leaf_key(&blob),
                    &blob,
                    "admission-fixture",
                )
                .unwrap();
            assert!(tree_bytes(&root).unwrap() <= lowered);
            assert_eq!(
                store.admission_cleanup_bytes(),
                if mode == "exact-capacity" { 0 } else { 4096 }
            );
            WRITE_OBSERVATIONS.with(|rows| {
                let rows = rows.borrow();
                let observed: Vec<_> = rows.iter().filter(|row| row.0 == root).collect();
                assert!(!observed.is_empty());
                assert!(observed.iter().all(|row| row.3 + row.4 <= row.5), "{observed:?}");
                let before_payload = observed.iter().find(|row| row.1 == "payload" && row.2).unwrap();
                assert_eq!(before_payload.4, blob.len() as u64);
                eprintln!("ANAMNESIS {mode}: starting_tree={filled} max_bytes={lowered} free_before_put={free} admission_excess={} before_payload={} payload_growth={} peak_write_tree={}",
                    store.admission_cleanup_bytes(), before_payload.3, before_payload.4,
                    observed.iter().map(|row| row.3).max().unwrap());
            });
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn admission_raises_before_writing_when_control_footprint_cannot_fit() {
        let root = scratch("unreclaimable-admission");
        fs::create_dir_all(root.join("quarantine")).unwrap();
        fs::write(root.join("unmanaged-control"), vec![0; 4096 + 1]).unwrap();
        let bound = tree_bytes(&root).unwrap() - 4096;
        WRITE_OBSERVATIONS.with(|rows| rows.borrow_mut().clear());
        assert_eq!(
            ContentStore::new(&root, bound, false).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
        WRITE_OBSERVATIONS.with(|rows| assert!(rows.borrow().iter().all(|row| row.0 != root)));
        assert_eq!(
            fs::read(root.join("unmanaged-control")).unwrap(),
            vec![0; 4096 + 1]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packed_snapshot_defers_integrity_until_the_requested_read_set() {
        let root = scratch("snapshot-reads");
        let store = ContentStore::new(&root, u64::MAX, true).unwrap();
        let key = super::super::key::leaf_key(b"snapshot");
        store.put_leaf(key, b"snapshot", "snapshot.test").unwrap();
        let index = fs::read(root.join(ContentStore::INDEX)).unwrap();
        let data = fs::read(root.join(ContentStore::DATA)).unwrap();
        let snapshot = PackedReadSnapshot::from_index(&index).unwrap();
        let absent = super::super::key::leaf_key(b"absent");
        assert_eq!(
            snapshot.verify(&[key, absent], &data),
            vec![(key.to_hex(), Some(b"snapshot".as_slice()))]
        );
        let mut damaged = data;
        damaged[0] ^= 1;
        // Parsing succeeds without hashing or serving any entry.
        let snapshot = PackedReadSnapshot::from_index(&index).unwrap();
        assert_eq!(
            snapshot.verify(&[key], &damaged),
            vec![(key.to_hex(), None)]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packed_snapshot_rejects_old_schema_and_invalid_empty_ranges() {
        let root = scratch("snapshot-bounds");
        let store = ContentStore::new(&root, u64::MAX, true).unwrap();
        let key = super::super::key::leaf_key(b"");
        store.put_leaf(key, b"", "snapshot.empty").unwrap();
        let raw = fs::read(root.join(ContentStore::INDEX)).unwrap();
        let mut index: PackedIndex = serde_json::from_slice(&raw).unwrap();
        index.entries.get_mut(&key.to_hex()).unwrap().length = 1;
        let snapshot =
            PackedReadSnapshot::from_index(&serde_json::to_vec(&index).unwrap()).unwrap();
        assert_eq!(snapshot.verify(&[key], b""), vec![(key.to_hex(), None)]);
        index.schema = "forge3d.anamnesis.packed-index/1".into();
        assert!(PackedReadSnapshot::from_index(&serde_json::to_vec(&index).unwrap()).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packed_large_read_sets_preserve_order_and_reject_corrupt_entries() {
        let root = scratch("snapshot-large-reads");
        let store = ContentStore::new(&root, u64::MAX, true).unwrap();
        let (valid, valid_material) = derivation("valid", b"valid bytes");
        let (damaged_blob, blob_material) = derivation("blob", b"bad blob");
        let (damaged_derivation, bad_material) = derivation("derivation", b"bad derivation");
        store.put(valid, b"valid bytes", valid_material).unwrap();
        store.put(damaged_blob, b"bad blob", blob_material).unwrap();
        store
            .put(damaged_derivation, b"bad derivation", bad_material)
            .unwrap();
        let (mut index, mut data) = store.load().unwrap();
        data[index.entries[&damaged_blob.to_hex()].offset as usize] ^= 1;
        let KeyDerivation::Pass { material } = &mut index
            .entries
            .get_mut(&damaged_derivation.to_hex())
            .unwrap()
            .metadata
            .derivation
        else {
            panic!("expected a pass derivation");
        };
        material.uniform_sha256.0[0] ^= 1;
        let snapshot =
            PackedReadSnapshot::from_index(&serde_json::to_vec(&index).unwrap()).unwrap();
        let absent = super::super::key::leaf_key(b"absent");
        let pattern = [valid, damaged_blob, absent, damaged_derivation, valid];
        for length in [0, 1, 255, 256, 600, 601] {
            let keys: Vec<_> = pattern.into_iter().cycle().take(length).collect();
            let expected: Vec<_> = keys
                .iter()
                .filter(|key| **key != absent)
                .map(|key| {
                    (
                        key.to_hex(),
                        (*key == valid).then_some(b"valid bytes".as_slice()),
                    )
                })
                .collect();
            assert_eq!(snapshot.verify(&keys, &data), expected);
        }
        fs::remove_dir_all(root).unwrap();
    }
}
