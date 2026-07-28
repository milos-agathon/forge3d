//! Tile-indexed virtual-texture backing store.
//!
//! `MmapPageStore` keeps the public name from the TESSELLA contract. The
//! implementation deliberately uses positional `read_at`/`seek_read` calls
//! plus a bounded page cache instead of a new mmap dependency. Pages remain
//! out of the process address space until requested, work on Unix and Windows,
//! and never require copying the complete virtual image into a `Vec<u8>`.

use crate::core::provenance::sha256;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
#[cfg(feature = "cog_streaming")]
use std::sync::Arc;
use std::sync::Mutex;

const MAGIC: &[u8; 8] = b"F3DVT1\0\0";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 96;
const DIRECTORY_ENTRY_SIZE: usize = 64;
const FLAG_PROCEDURAL: u32 = 1;
const DEFAULT_CACHE_BYTES: u64 = 64 * 1024 * 1024;
pub const HEIGHT_FAMILY: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PageKey {
    pub family: u8,
    pub mip: u8,
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageFormat {
    Bc7Srgb,
    Bc5Unorm,
    Bc7Unorm,
    Rgba8Srgb,
    R32Float,
}

impl PageFormat {
    pub fn block_bytes(self, width: u32, height: u32) -> u64 {
        match self {
            Self::Bc7Srgb | Self::Bc5Unorm | Self::Bc7Unorm => {
                u64::from(width.div_ceil(4)) * u64::from(height.div_ceil(4)) * 16
            }
            Self::Rgba8Srgb | Self::R32Float => u64::from(width) * u64::from(height) * 4,
        }
    }

    pub fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Bc7Srgb => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
            Self::Bc5Unorm => wgpu::TextureFormat::Bc5RgUnorm,
            Self::Bc7Unorm => wgpu::TextureFormat::Bc7RgbaUnorm,
            Self::Rgba8Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::R32Float => wgpu::TextureFormat::R32Float,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Bc7Srgb => 1,
            Self::Bc5Unorm => 2,
            Self::Bc7Unorm => 3,
            Self::Rgba8Srgb => 4,
            Self::R32Float => 5,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, String> {
        match tag {
            1 => Ok(Self::Bc7Srgb),
            2 => Ok(Self::Bc5Unorm),
            3 => Ok(Self::Bc7Unorm),
            4 => Ok(Self::Rgba8Srgb),
            5 => Ok(Self::R32Float),
            _ => Err(format!("unknown VT page format tag {tag}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PageBytes {
    pub format: PageFormat,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub sha256: [u8; 32],
}

impl PageBytes {
    pub fn new(format: PageFormat, width: u32, height: u32, data: Vec<u8>) -> Result<Self, String> {
        let expected = format.block_bytes(width, height);
        if data.len() as u64 != expected {
            return Err(format!(
                "{format:?} page payload mismatch: got {} bytes, expected {expected}",
                data.len()
            ));
        }
        Ok(Self {
            format,
            width,
            height,
            sha256: sha256(&data),
            data,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PackedPage {
    pub key: PageKey,
    pub bytes: PageBytes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreMetadata {
    pub virtual_width: u64,
    pub virtual_height: u64,
    pub tile_size: u32,
    pub tile_border: u32,
    pub family_count: u32,
    pub procedural: bool,
    pub procedural_seed: u64,
}

impl StoreMetadata {
    pub fn slot_size(&self) -> u32 {
        self.tile_size + self.tile_border.saturating_mul(2)
    }

    pub fn logical_texel_bytes(&self) -> u128 {
        u128::from(self.virtual_width)
            * u128::from(self.virtual_height)
            * u128::from(self.family_count)
            * 4
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreManifest {
    pub format: String,
    pub version: u32,
    pub path: String,
    pub virtual_width: u64,
    pub virtual_height: u64,
    pub logical_texel_bytes: String,
    pub tile_size: u32,
    pub tile_border: u32,
    pub family_count: u32,
    pub page_count: u64,
    pub procedural: bool,
    pub page_order: String,
    pub encodings: Vec<PageFormat>,
    pub directory_sha256: String,
    /// Integrity digest for every physically packed page.
    pub pages: Vec<ManifestPageDigest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestPageDigest {
    pub key: PageKey,
    pub sha256: String,
}

pub trait VirtualTextureStore: Send + Sync {
    fn page(&self, key: PageKey) -> Result<PageBytes, String>;
    fn metadata(&self) -> &StoreMetadata;
    fn content_hash(&self) -> [u8; 32];
    fn page_count(&self) -> u64;
}

#[derive(Clone, Debug)]
struct DirectoryEntry {
    key: PageKey,
    format: PageFormat,
    width: u32,
    height: u32,
    offset: u64,
    length: u32,
    raw_length: u32,
    digest: [u8; 32],
}

struct PageCache {
    entries: HashMap<PageKey, PageBytes>,
    order: VecDeque<PageKey>,
    used_bytes: u64,
    budget_bytes: u64,
}

impl PageCache {
    fn new(budget_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            used_bytes: 0,
            budget_bytes: budget_bytes.max(1),
        }
    }

    fn get(&mut self, key: PageKey) -> Option<PageBytes> {
        let page = self.entries.get(&key)?.clone();
        if let Some(position) = self.order.iter().position(|candidate| *candidate == key) {
            self.order.remove(position);
        }
        self.order.push_back(key);
        Some(page)
    }

    fn insert(&mut self, key: PageKey, page: PageBytes) {
        let bytes = page.data.len() as u64;
        if bytes > self.budget_bytes {
            return;
        }
        while self.used_bytes + bytes > self.budget_bytes {
            let Some(victim) = self.order.pop_front() else {
                break;
            };
            if let Some(removed) = self.entries.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(removed.data.len() as u64);
            }
        }
        if let Some(previous) = self.entries.insert(key, page) {
            self.used_bytes = self.used_bytes.saturating_sub(previous.data.len() as u64);
        }
        self.order.push_back(key);
        self.used_bytes += bytes;
    }
}

pub struct MmapPageStore {
    path: PathBuf,
    file: File,
    metadata: StoreMetadata,
    directory: HashMap<PageKey, DirectoryEntry>,
    directory_hash: [u8; 32],
    cache: Mutex<PageCache>,
}

impl MmapPageStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::open_with_cache(path, DEFAULT_CACHE_BYTES)
    }

    pub fn open_with_cache(
        path: impl AsRef<Path>,
        cache_budget_bytes: u64,
    ) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .map_err(|error| format!("failed to open VT store {}: {error}", path.display()))?;
        let mut header = [0u8; HEADER_SIZE];
        read_exact_at(&file, &mut header, 0)?;
        let decoded = decode_header(&header)?;
        let directory_bytes_len = decoded
            .page_count
            .checked_mul(DIRECTORY_ENTRY_SIZE as u64)
            .ok_or_else(|| "VT directory length overflow".to_string())?;
        let mut directory_bytes = vec![
            0u8;
            usize::try_from(directory_bytes_len).map_err(|_| {
                "VT directory does not fit address space".to_string()
            })?
        ];
        read_exact_at(&file, &mut directory_bytes, decoded.directory_offset)?;
        let mut directory = HashMap::with_capacity(decoded.page_count as usize);
        for chunk in directory_bytes.chunks_exact(DIRECTORY_ENTRY_SIZE) {
            let entry = decode_directory_entry(chunk)?;
            if entry.offset < decoded.data_offset {
                return Err(format!(
                    "VT page {:?} points inside the header/directory",
                    entry.key
                ));
            }
            if directory.insert(entry.key, entry).is_some() {
                return Err("duplicate VT page-directory key".to_string());
            }
        }
        Ok(Self {
            path,
            file,
            metadata: decoded.metadata,
            directory,
            directory_hash: sha256(&directory_bytes),
            cache: Mutex::new(PageCache::new(cache_budget_bytes)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn verify(&self) -> Result<(), String> {
        let mut keys = self.directory.keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        for key in keys {
            self.read_page_uncached(key)?;
        }
        Ok(())
    }

    fn read_page_uncached(&self, key: PageKey) -> Result<PageBytes, String> {
        let Some(entry) = self.directory.get(&key) else {
            if self.metadata.procedural {
                if u32::from(key.family) >= self.metadata.family_count {
                    return Err(format!(
                        "procedural VT family {} is out of range",
                        key.family
                    ));
                }
                // Sparse procedural stores pack one immutable canonical page
                // per family. Virtual coordinates repeat that physical page;
                // no request is synthesized in host memory.
                let canonical = PageKey {
                    family: key.family,
                    mip: 0,
                    x: 0,
                    y: 0,
                };
                return self.read_page_uncached(canonical);
            }
            return Err(format!("VT page not found: {key:?}"));
        };
        let mut data = vec![0u8; entry.length as usize];
        read_exact_at(&self.file, &mut data, entry.offset)?;
        if entry.length != entry.raw_length {
            return Err(format!(
                "VT page {:?} uses unsupported secondary compression (stored={}, raw={})",
                key, entry.length, entry.raw_length
            ));
        }
        let actual = sha256(&data);
        if actual != entry.digest {
            return Err(format!(
                "VT page SHA-256 mismatch for {:?}: expected {}, got {}",
                key,
                crate::core::provenance::to_hex(&entry.digest),
                crate::core::provenance::to_hex(&actual)
            ));
        }
        PageBytes::new(entry.format, entry.width, entry.height, data)
    }
}

impl VirtualTextureStore for MmapPageStore {
    fn page(&self, key: PageKey) -> Result<PageBytes, String> {
        if let Some(page) = self
            .cache
            .lock()
            .map_err(|_| "VT page cache mutex poisoned".to_string())?
            .get(key)
        {
            return Ok(page);
        }
        let page = self.read_page_uncached(key)?;
        self.cache
            .lock()
            .map_err(|_| "VT page cache mutex poisoned".to_string())?
            .insert(key, page.clone());
        Ok(page)
    }

    fn metadata(&self) -> &StoreMetadata {
        &self.metadata
    }

    fn content_hash(&self) -> [u8; 32] {
        self.directory_hash
    }

    fn page_count(&self) -> u64 {
        self.directory.len() as u64
    }
}

#[cfg(feature = "cog_streaming")]
pub struct CogPageStore {
    reader: Arc<crate::terrain::cog::CogHeightReader>,
    metadata: StoreMetadata,
    digest: [u8; 32],
}

#[cfg(feature = "cog_streaming")]
impl CogPageStore {
    pub fn from_reader(reader: Arc<crate::terrain::cog::CogHeightReader>, tile_size: u32) -> Self {
        let (width, height) = reader
            .header()
            .full_resolution()
            .map(|ifd| (u64::from(ifd.width), u64::from(ifd.height)))
            .unwrap_or((1, 1));
        let mut identity = Vec::new();
        identity.extend_from_slice(&width.to_le_bytes());
        identity.extend_from_slice(&height.to_le_bytes());
        identity.extend_from_slice(&tile_size.to_le_bytes());
        Self {
            reader,
            metadata: StoreMetadata {
                virtual_width: width,
                virtual_height: height,
                tile_size,
                tile_border: 0,
                family_count: 1,
                procedural: false,
                procedural_seed: 0,
            },
            digest: sha256(&identity),
        }
    }
}

#[cfg(feature = "cog_streaming")]
impl VirtualTextureStore for CogPageStore {
    fn page(&self, key: PageKey) -> Result<PageBytes, String> {
        if key.family != HEIGHT_FAMILY {
            return Err(format!(
                "CogPageStore serves height family {HEIGHT_FAMILY}, got {}",
                key.family
            ));
        }
        let heights = self
            .reader
            .read_tile(key.x, key.y, u32::from(key.mip))
            .map_err(|error| error.to_string())?;
        let side = (heights.len() as f64).sqrt() as u32;
        if u64::from(side) * u64::from(side) != heights.len() as u64 {
            return Err("COG height tile is not square".to_string());
        }
        PageBytes::new(
            PageFormat::R32Float,
            side,
            side,
            bytemuck::cast_slice(&heights).to_vec(),
        )
    }

    fn metadata(&self) -> &StoreMetadata {
        &self.metadata
    }

    fn content_hash(&self) -> [u8; 32] {
        self.digest
    }

    fn page_count(&self) -> u64 {
        0
    }
}

pub fn write_packed_store(
    path: impl AsRef<Path>,
    metadata: &StoreMetadata,
    pages: impl IntoIterator<Item = PackedPage>,
) -> Result<StoreManifest, String> {
    validate_metadata(metadata)?;
    let path = path.as_ref();
    let mut pages = pages.into_iter().collect::<Vec<_>>();
    pages.sort_by_key(|page| {
        (
            page.key.family,
            page.key.mip,
            morton2(page.key.x, page.key.y),
            page.key.y,
            page.key.x,
        )
    });
    let page_count = pages.len() as u64;
    let directory_offset = HEADER_SIZE as u64;
    let data_offset = directory_offset
        .checked_add(page_count * DIRECTORY_ENTRY_SIZE as u64)
        .ok_or_else(|| "VT store data offset overflow".to_string())?;
    let mut directory = Vec::with_capacity(pages.len() * DIRECTORY_ENTRY_SIZE);
    let mut payload = Vec::new();
    let mut encodings = Vec::new();
    for page in &pages {
        if page.bytes.width != metadata.slot_size() || page.bytes.height != metadata.slot_size() {
            return Err(format!(
                "VT page {:?} is {}x{}, expected slot size {}x{}",
                page.key,
                page.bytes.width,
                page.bytes.height,
                metadata.slot_size(),
                metadata.slot_size()
            ));
        }
        if !encodings.contains(&page.bytes.format) {
            encodings.push(page.bytes.format);
        }
        let entry = DirectoryEntry {
            key: page.key,
            format: page.bytes.format,
            width: page.bytes.width,
            height: page.bytes.height,
            offset: data_offset + payload.len() as u64,
            length: page.bytes.data.len() as u32,
            raw_length: page.bytes.data.len() as u32,
            digest: page.bytes.sha256,
        };
        directory.extend_from_slice(&encode_directory_entry(&entry)?);
        payload.extend_from_slice(&page.bytes.data);
    }
    let header = encode_header(metadata, page_count, directory_offset, data_offset)?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to create VT store {}: {error}", path.display()))?;
    file.write_all(&header)
        .and_then(|_| file.write_all(&directory))
        .and_then(|_| file.write_all(&payload))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("failed to write VT store {}: {error}", path.display()))?;
    Ok(StoreManifest {
        format: "forge3d-vtpack".to_string(),
        version: VERSION,
        path: path.display().to_string(),
        virtual_width: metadata.virtual_width,
        virtual_height: metadata.virtual_height,
        logical_texel_bytes: metadata.logical_texel_bytes().to_string(),
        tile_size: metadata.tile_size,
        tile_border: metadata.tile_border,
        family_count: metadata.family_count,
        page_count,
        procedural: metadata.procedural,
        page_order: "family,mip,morton2".to_string(),
        encodings,
        directory_sha256: crate::core::provenance::to_hex(&sha256(&directory)),
        pages: pages
            .iter()
            .map(|page| ManifestPageDigest {
                key: page.key,
                sha256: crate::core::provenance::to_hex(&page.bytes.sha256),
            })
            .collect(),
    })
}

struct DecodedHeader {
    metadata: StoreMetadata,
    page_count: u64,
    directory_offset: u64,
    data_offset: u64,
}

fn encode_header(
    metadata: &StoreMetadata,
    page_count: u64,
    directory_offset: u64,
    data_offset: u64,
) -> Result<[u8; HEADER_SIZE], String> {
    validate_metadata(metadata)?;
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[0..8].copy_from_slice(MAGIC);
    put_u32(&mut bytes, 8, VERSION);
    put_u32(&mut bytes, 12, HEADER_SIZE as u32);
    put_u32(&mut bytes, 16, metadata.tile_size);
    put_u32(&mut bytes, 20, metadata.tile_border);
    put_u32(&mut bytes, 24, metadata.family_count);
    put_u32(
        &mut bytes,
        28,
        if metadata.procedural {
            FLAG_PROCEDURAL
        } else {
            0
        },
    );
    put_u64(&mut bytes, 32, metadata.virtual_width);
    put_u64(&mut bytes, 40, metadata.virtual_height);
    put_u64(&mut bytes, 48, page_count);
    put_u64(&mut bytes, 56, directory_offset);
    put_u64(&mut bytes, 64, data_offset);
    put_u64(&mut bytes, 72, metadata.procedural_seed);
    Ok(bytes)
}

fn decode_header(bytes: &[u8; HEADER_SIZE]) -> Result<DecodedHeader, String> {
    if &bytes[0..8] != MAGIC {
        return Err("invalid forge3d VT store magic".to_string());
    }
    if get_u32(bytes, 8) != VERSION {
        return Err(format!(
            "unsupported forge3d VT store version {}",
            get_u32(bytes, 8)
        ));
    }
    if get_u32(bytes, 12) as usize != HEADER_SIZE {
        return Err("invalid forge3d VT header size".to_string());
    }
    let metadata = StoreMetadata {
        tile_size: get_u32(bytes, 16),
        tile_border: get_u32(bytes, 20),
        family_count: get_u32(bytes, 24),
        procedural: get_u32(bytes, 28) & FLAG_PROCEDURAL != 0,
        virtual_width: get_u64(bytes, 32),
        virtual_height: get_u64(bytes, 40),
        procedural_seed: get_u64(bytes, 72),
    };
    validate_metadata(&metadata)?;
    let decoded = DecodedHeader {
        metadata,
        page_count: get_u64(bytes, 48),
        directory_offset: get_u64(bytes, 56),
        data_offset: get_u64(bytes, 64),
    };
    if decoded.directory_offset < HEADER_SIZE as u64
        || decoded.data_offset < decoded.directory_offset
    {
        return Err("invalid forge3d VT store offsets".to_string());
    }
    Ok(decoded)
}

fn encode_directory_entry(entry: &DirectoryEntry) -> Result<[u8; DIRECTORY_ENTRY_SIZE], String> {
    let mut bytes = [0u8; DIRECTORY_ENTRY_SIZE];
    bytes[0] = entry.key.family;
    bytes[1] = entry.key.mip;
    bytes[2] = entry.format.tag();
    put_u32(&mut bytes, 4, entry.key.x);
    put_u32(&mut bytes, 8, entry.key.y);
    let width = u16::try_from(entry.width).map_err(|_| "VT page width exceeds u16".to_string())?;
    let height =
        u16::try_from(entry.height).map_err(|_| "VT page height exceeds u16".to_string())?;
    bytes[12..14].copy_from_slice(&width.to_le_bytes());
    bytes[14..16].copy_from_slice(&height.to_le_bytes());
    put_u64(&mut bytes, 16, entry.offset);
    put_u32(&mut bytes, 24, entry.length);
    put_u32(&mut bytes, 28, entry.raw_length);
    bytes[32..64].copy_from_slice(&entry.digest);
    Ok(bytes)
}

fn decode_directory_entry(bytes: &[u8]) -> Result<DirectoryEntry, String> {
    if bytes.len() != DIRECTORY_ENTRY_SIZE {
        return Err("invalid VT directory entry length".to_string());
    }
    Ok(DirectoryEntry {
        key: PageKey {
            family: bytes[0],
            mip: bytes[1],
            x: get_u32(bytes, 4),
            y: get_u32(bytes, 8),
        },
        format: PageFormat::from_tag(bytes[2])?,
        width: u32::from(u16::from_le_bytes(bytes[12..14].try_into().unwrap())),
        height: u32::from(u16::from_le_bytes(bytes[14..16].try_into().unwrap())),
        offset: get_u64(bytes, 16),
        length: get_u32(bytes, 24),
        raw_length: get_u32(bytes, 28),
        digest: bytes[32..64].try_into().unwrap(),
    })
}

fn validate_metadata(metadata: &StoreMetadata) -> Result<(), String> {
    if metadata.virtual_width == 0 || metadata.virtual_height == 0 {
        return Err("VT virtual dimensions must be non-zero".to_string());
    }
    if metadata.tile_size == 0 || !metadata.tile_size.is_power_of_two() {
        return Err("VT tile_size must be a non-zero power of two".to_string());
    }
    if !metadata.slot_size().is_multiple_of(4) {
        return Err("VT tile_size + 2*tile_border must be BC block aligned".to_string());
    }
    if metadata.family_count == 0 || metadata.family_count > 4 {
        return Err("VT family_count must be in 1..=4".to_string());
    }
    Ok(())
}

fn morton2(x: u32, y: u32) -> u64 {
    fn spread(value: u32) -> u64 {
        let mut value = u64::from(value & 0x0000_ffff);
        value = (value | (value << 16)) & 0x0000_ffff_0000_ffff;
        value = (value | (value << 8)) & 0x00ff_00ff_00ff_00ff;
        value = (value | (value << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        value = (value | (value << 2)) & 0x3333_3333_3333_3333;
        (value | (value << 1)) & 0x5555_5555_5555_5555
    }
    spread(x) | (spread(y) << 1)
}

#[cfg(unix)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    read_exact_with(bytes, offset, |chunk, at| file.read_at(chunk, at))
}

#[cfg(windows)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::windows::fs::FileExt;
    read_exact_with(bytes, offset, |chunk, at| file.seek_read(chunk, at))
}

fn read_exact_with(
    mut bytes: &mut [u8],
    mut offset: u64,
    mut read: impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
) -> Result<(), String> {
    while !bytes.is_empty() {
        match read(bytes, offset) {
            Ok(0) => return Err("unexpected EOF in VT store".to_string()),
            Ok(count) => {
                offset += count as u64;
                bytes = &mut bytes[count..];
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(format!("VT store positional read failed: {error}")),
        }
    }
    Ok(())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("forge3d-{name}-{nonce}.f3dvt"))
    }

    #[test]
    fn page_directory_round_trip_and_sha_validation() {
        let path = scratch("vt-roundtrip");
        let metadata = StoreMetadata {
            virtual_width: 4096,
            virtual_height: 4096,
            tile_size: 8,
            tile_border: 2,
            family_count: 3,
            procedural: false,
            procedural_seed: 0,
        };
        let side = metadata.slot_size();
        let raw = vec![77u8; side as usize * side as usize * 4];
        let page = PageBytes::new(
            PageFormat::Bc7Srgb,
            side,
            side,
            crate::core::compressed_textures::encode_bc7_rgba8(&raw, side, side).unwrap(),
        )
        .unwrap();
        write_packed_store(
            &path,
            &metadata,
            [PackedPage {
                key: PageKey {
                    family: 0,
                    mip: 0,
                    x: 2,
                    y: 3,
                },
                bytes: page.clone(),
            }],
        )
        .unwrap();
        let store = MmapPageStore::open_with_cache(&path, 1024).unwrap();
        assert_eq!(store.metadata().virtual_width, 4096);
        assert_eq!(
            store
                .page(PageKey {
                    family: 0,
                    mip: 0,
                    x: 2,
                    y: 3
                })
                .unwrap()
                .data,
            page.data
        );
        store.verify().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn sparse_procedural_store_declares_256_gib_without_allocating_it() {
        let path = scratch("vt-procedural");
        let metadata = StoreMetadata {
            virtual_width: 1 << 18,
            virtual_height: 1 << 18,
            tile_size: 128,
            tile_border: 4,
            family_count: 3,
            procedural: true,
            procedural_seed: 19,
        };
        assert!(metadata.logical_texel_bytes() >= 256u128 * 1024 * 1024 * 1024);
        let side = metadata.slot_size();
        let pages = (0..3u8)
            .map(|family| {
                let (format, data) = if family == 1 {
                    (
                        PageFormat::Bc5Unorm,
                        crate::core::compressed_textures::encode_bc5_rg8(
                            &vec![128; side as usize * side as usize * 2],
                            side,
                            side,
                        )
                        .unwrap(),
                    )
                } else {
                    (
                        if family == 0 {
                            PageFormat::Bc7Srgb
                        } else {
                            PageFormat::Bc7Unorm
                        },
                        crate::core::compressed_textures::encode_bc7_rgba8(
                            &vec![128; side as usize * side as usize * 4],
                            side,
                            side,
                        )
                        .unwrap(),
                    )
                };
                PackedPage {
                    key: PageKey {
                        family,
                        mip: 0,
                        x: 0,
                        y: 0,
                    },
                    bytes: PageBytes::new(format, side, side, data).unwrap(),
                }
            })
            .collect::<Vec<_>>();
        let manifest = write_packed_store(&path, &metadata, pages).unwrap();
        assert_eq!(manifest.pages.len(), 3);
        let store = MmapPageStore::open(&path).unwrap();
        let page = store
            .page(PageKey {
                family: 1,
                mip: 4,
                x: 17,
                y: 9,
            })
            .unwrap();
        assert_eq!(page.format, PageFormat::Bc5Unorm);
        assert_eq!(page.data.len() as u64, page.format.block_bytes(136, 136));
        std::fs::remove_file(path).unwrap();
    }
}
