//! On-disk state at the boundary after terrain_publish and ReSTIR spatial reuse.
use super::render_terrain::{TerrainReferenceDesc, TerrainStatistics, WELFORD_WINDOW};
use super::*;
use crate::core::resource_tracker::tracked_host_allocation;
use crate::path_tracing::restir::Reservoir;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"F3DCTN\x01\0";
const DIGEST_BYTES: u64 = 32;
const PROGRESS_BYTES: u64 = 16;
const STATE_BYTES_PER_PIXEL: [u64; 8] = [
    16, // RGBA accumulated radiance/count
    std::mem::size_of::<TerrainStatistics>() as u64,
    std::mem::size_of::<Reservoir>() as u64,
    8, // Beauty, required for a checkpoint already at convergence
    8, // Frame-zero albedo
    8, // Frame-zero normal
    4, // Frame-zero depth
    4, // Frame-zero visibility (AETHER)
];

pub(crate) struct TerrainCheckpointOptions {
    pub path: PathBuf,
    pub every: Option<u32>,
    pub resume: bool,
    pub runtime: CheckpointIdentity,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct FrameProgress {
    pub frames: u32,
    pub variance_at: u32,
    pub variance: f32,
    pub converged: bool,
}

impl FrameProgress {
    fn encode(self) -> [u8; PROGRESS_BYTES as usize] {
        let mut bytes = [0; PROGRESS_BYTES as usize];
        for (chunk, value) in bytes.chunks_exact_mut(4).zip([
            self.frames,
            self.variance_at,
            self.variance.to_bits(),
            u32::from(self.converged),
        ]) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode(
        bytes: [u8; PROGRESS_BYTES as usize],
        desc: &TerrainReferenceDesc,
    ) -> Result<Self, RenderError> {
        let mut words = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        let frames = words.next().unwrap();
        let variance_at = words.next().unwrap();
        let variance = f32::from_bits(words.next().unwrap());
        let converged_word = words.next().unwrap();
        let converged = converged_word == 1;
        let last_check = if frames == desc.max_frames && frames >= 2 {
            frames
        } else {
            frames / WELFORD_WINDOW * WELFORD_WINDOW
        };
        if frames == 0
            || frames > desc.max_frames
            || variance_at != last_check
            || converged_word > 1
            || (variance_at == 0 && variance.to_bits() != 0)
            || (variance_at > 0 && (!variance.is_finite() || variance < 0.0))
            || converged
                != (variance_at >= desc.min_frames
                    && variance_at >= 2
                    && variance < desc.variance_threshold)
            || (converged && frames != variance_at)
        {
            return Err(invalid("invalid frame/convergence state"));
        }
        Ok(Self {
            frames,
            variance_at,
            variance,
            converged,
        })
    }
}

pub(crate) type CheckpointIdentity = Vec<(&'static str, [u8; 32])>;

fn invalid(reason: impl std::fmt::Display) -> RenderError {
    RenderError::render(format!("terrain checkpoint: {reason}"))
}

fn hash(mut reader: impl BufRead) -> std::io::Result<[u8; 32]> {
    let mut digest = Sha256::new();
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            return Ok(digest.finalize().into());
        }
        digest.update(bytes);
        let len = bytes.len();
        reader.consume(len);
    }
}

pub(crate) fn checkpoint_runtime() -> Result<CheckpointIdentity, RenderError> {
    let build = super::native_build_id::checkpoint_build_id()
        .ok_or_else(|| invalid("loaded native image has no linker build identifier"))?;
    let ctx = try_ctx()?;
    let info = ctx.adapter.get_info();
    if ctx.software_fallback || info.device_type == wgpu::DeviceType::Cpu {
        return Err(invalid("adapter is a software renderer"));
    }
    if info.driver_info.is_empty() {
        return Err(invalid("driver version is unavailable on this backend"));
    }
    let (device_uuid, driver_uuid) = vulkan_device_identity(&ctx.adapter).ok_or_else(|| {
        invalid("physical adapter and driver UUIDs are unavailable; use a qualified Vulkan adapter")
    })?;

    let mut shader = Sha256::new();
    for source in [
        crate::shader_sources::hybrid_kernel().as_bytes(),
        include_bytes!("../../shaders/pt_restir_temporal.wgsl"),
        include_bytes!("../../shaders/pt_restir_spatial.wgsl"),
        crate::shader_sources::prometheus_aerial().as_bytes(),
    ] {
        shader.update((source.len() as u64).to_le_bytes());
        shader.update(source);
    }
    Ok(vec![
        ("native_build", build),
        ("shader", shader.finalize().into()),
        (
            "adapter",
            digest_debug((
                device_uuid,
                info.vendor,
                info.device,
                info.name,
                info.device_type,
            )),
        ),
        ("backend", digest_debug((info.backend, ctx.dx12_compiler))),
        (
            "driver",
            digest_debug((driver_uuid, info.driver, info.driver_info)),
        ),
        (
            "device_features_and_limits",
            digest_debug((ctx.device.features(), ctx.device.limits())),
        ),
    ])
}

// wgpu 0.19 reports model IDs, which can match across two distinct cards.
// Use the Vulkan 1.1 device ID query through wgpu's existing HAL. The small
// pNext record mirrors VkPhysicalDeviceIDProperties from the Vulkan ABI.
#[cfg(any(
    windows,
    all(unix, not(target_vendor = "apple"), not(target_os = "emscripten"))
))]
fn vulkan_device_identity(adapter: &wgpu::Adapter) -> Option<([u8; 16], [u8; 16])> {
    #[repr(C)]
    struct DeviceIdProperties {
        s_type: i32,
        p_next: *mut std::ffi::c_void,
        device_uuid: [u8; 16],
        driver_uuid: [u8; 16],
        device_luid: [u8; 8],
        device_node_mask: u32,
        device_luid_valid: u32,
    }

    fn default_properties<Device, Properties: Default>(
        _: unsafe extern "system" fn(Device, *mut Properties),
    ) -> Properties {
        Properties::default()
    }

    // SAFETY: the HAL borrow remains live inside the callback. The query
    // writes only the Vulkan properties struct and its aligned pNext record.
    // Neither their pointers nor any HAL handles are retained or destroyed.
    unsafe {
        adapter.as_hal::<wgpu::hal::api::Vulkan, _, _>(|raw| {
            let raw = raw?;
            let instance = raw.shared_instance();
            const VK_API_VERSION_1_1: u32 = (1 << 22) | (1 << 12);
            if instance.instance_api_version() < VK_API_VERSION_1_1 {
                return None;
            }
            let query = instance
                .raw_instance()
                .fp_v1_1()
                .get_physical_device_properties2;
            let mut id = DeviceIdProperties {
                s_type: 1_000_071_004, // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
                p_next: std::ptr::null_mut(),
                device_uuid: [0; 16],
                driver_uuid: [0; 16],
                device_luid: [0; 8],
                device_node_mask: 0,
                device_luid_valid: 0,
            };
            let mut properties = default_properties(query);
            properties.p_next = std::ptr::from_mut(&mut id).cast();
            query(raw.raw_physical_device(), &mut properties);
            (id.device_uuid != [0; 16] && id.driver_uuid != [0; 16])
                .then_some((id.device_uuid, id.driver_uuid))
        })
    }
}

#[cfg(not(any(
    windows,
    all(unix, not(target_vendor = "apple"), not(target_os = "emscripten"))
)))]
fn vulkan_device_identity(_adapter: &wgpu::Adapter) -> Option<([u8; 16], [u8; 16])> {
    None
}

fn digest_debug(value: impl std::fmt::Debug) -> [u8; 32] {
    Sha256::digest(format!("{value:?}")).into()
}

fn digest_array<T: Pod>(values: &[T]) -> [u8; 32] {
    Sha256::digest(bytemuck::cast_slice(values)).into()
}

fn append_scene_identity(
    identity: &mut CheckpointIdentity,
    desc: &TerrainReferenceDesc,
    source: &str,
) {
    // Exhaustive so a new shader/certificate input must be assigned a field.
    let TerrainReferenceDesc {
        heights,
        dem_width,
        dem_height,
        spacing,
        exaggeration,
        albedo,
        albedo_map,
        albedo_sampling,
        turbidity,
        cam_origin,
        cam_look_at,
        cam_up,
        camera_model,
        seamless_camera,
        fov_y_deg,
        ortho_half_height,
        sensor_rect,
        full_width,
        full_height,
        pixel_offset_x,
        pixel_offset_y,
        exposure,
        sun_azimuth_deg,
        sun_elevation_deg,
        sun_intensity,
        sun_color,
        observer_geodetic_deg,
        earth_model,
        refraction_model,
        env_map,
        env_intensity,
        atmosphere,
        mesh,
        sdf_scene,
        width,
        height,
        seed,
        spp,
        max_frames,
        min_frames,
        variance_threshold,
    } = desc;
    identity.extend([
        ("heightmap", digest_array(heights)),
        (
            "terrain_grid",
            digest_debug((dem_width, dem_height, spacing, exaggeration)),
        ),
        (
            "material",
            digest_debug((
                albedo,
                *albedo_sampling as u32,
                albedo_map.as_deref().map(digest_array),
            )),
        ),
        (
            "camera",
            digest_debug((
                cam_origin,
                cam_look_at,
                cam_up,
                camera_model,
                fov_y_deg,
                ortho_half_height,
                exposure,
            )),
        ),
        (
            "sensor_window",
            digest_debug((
                sensor_rect,
                full_width,
                full_height,
                pixel_offset_x,
                pixel_offset_y,
                width,
                height,
                seamless_camera,
            )),
        ),
        (
            "lighting",
            digest_debug((
                sun_azimuth_deg,
                sun_elevation_deg,
                sun_intensity,
                sun_color,
                env_intensity,
                turbidity,
            )),
        ),
        (
            "curvature",
            digest_debug((observer_geodetic_deg, earth_model, refraction_model)),
        ),
        (
            "environment",
            digest_debug(
                env_map
                    .as_ref()
                    .map(|(rgb, w, h)| (w, h, digest_array(rgb))),
            ),
        ),
        (
            "atmosphere",
            digest_debug(atmosphere.as_ref().map(|lut| lut.deterministic_sha256())),
        ),
        (
            "mesh",
            digest_debug(
                mesh.as_ref()
                    .map(|(vertices, indices)| (digest_array(vertices), digest_array(indices))),
            ),
        ),
        ("sdf_scene", digest_debug(sdf_scene)),
        (
            "settings",
            digest_debug((
                seed,
                spp,
                max_frames,
                min_frames,
                variance_threshold,
                WELFORD_WINDOW,
            )),
        ),
        ("terrain_shader", Sha256::digest(source.as_bytes()).into()),
    ]);
}

pub(crate) struct TerrainCheckpoint {
    path: PathBuf,
    identity: CheckpointIdentity,
    every: Option<u32>,
    width: u32,
    height: u32,
    restore: Option<File>,
    pub progress: FrameProgress,
    _lock: File,
}

impl TerrainCheckpoint {
    pub fn open(
        options: TerrainCheckpointOptions,
        desc: &TerrainReferenceDesc,
        source: &str,
    ) -> Result<Self, RenderError> {
        let TerrainCheckpointOptions {
            path,
            every,
            resume,
            mut runtime,
        } = options;
        if every == Some(0) || (!resume && every.is_none()) {
            return Err(invalid(
                "a positive checkpoint_every is required when starting a terrain checkpoint",
            ));
        }
        let lock_path = path.with_extension(match path.extension().and_then(|v| v.to_str()) {
            Some(ext) => format!("{ext}.lock"),
            None => "lock".into(),
        });
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;
        lock.try_lock()
            .map_err(|error| invalid(format!("could not lock {}: {error}", path.display())))?;
        append_scene_identity(&mut runtime, desc, source);
        let mut checkpoint = Self {
            path,
            identity: runtime,
            every,
            width: desc.width,
            height: desc.height,
            restore: None,
            progress: FrameProgress::default(),
            _lock: lock,
        };
        if resume {
            checkpoint.load(desc)?;
        } else if checkpoint.path.try_exists()? {
            return Err(invalid(format!(
                "{} already exists; use resume=True or a new path",
                checkpoint.path.display()
            )));
        }
        Ok(checkpoint)
    }

    fn body_size(&self) -> Result<u64, RenderError> {
        let header =
            MAGIC.len() as u64 + self.identity.len() as u64 * DIGEST_BYTES + PROGRESS_BYTES;
        (self.width as u64 * self.height as u64)
            .checked_mul(STATE_BYTES_PER_PIXEL.iter().sum())
            .and_then(|payload| payload.checked_add(header))
            .filter(|body| body.checked_add(DIGEST_BYTES).is_some())
            .ok_or_else(|| invalid("declared dimensions overflow the checkpoint length"))
    }

    fn load(&mut self, desc: &TerrainReferenceDesc) -> Result<(), RenderError> {
        let mut file = File::open(&self.path)?;
        let mut magic = [0; MAGIC.len()];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("unsupported version or invalid magic"));
        }
        for (field, expected) in &self.identity {
            let mut recorded = [0; DIGEST_BYTES as usize];
            file.read_exact(&mut recorded)?;
            if &recorded != expected {
                return Err(invalid(format!("{field} mismatch")));
            }
        }
        let body = self.body_size()?;
        if file.metadata()?.len() != body + DIGEST_BYTES {
            return Err(invalid("length does not match the declared dimensions"));
        }
        let mut progress = [0; PROGRESS_BYTES as usize];
        file.read_exact(&mut progress)?;
        let payload = file.stream_position()?;
        file.seek(SeekFrom::Start(body))?;
        let mut expected = [0; DIGEST_BYTES as usize];
        file.read_exact(&mut expected)?;
        if hash_body(&mut file, body)? != expected {
            return Err(invalid("checksum mismatch"));
        }
        self.progress = FrameProgress::decode(progress, desc)?;
        file.seek(SeekFrom::Start(payload))?;
        self.restore = Some(file);
        Ok(())
    }

    pub fn is_due(&self, frames: u32) -> bool {
        self.every.is_some_and(|every| frames.is_multiple_of(every))
    }

    pub fn restore(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &[StateResource<'_>],
    ) -> Result<(), RenderError> {
        if let Some(mut file) = self.restore.take() {
            for resource in resources {
                resource.upload(device, queue, &mut file, self.width, self.height)?;
            }
        }
        Ok(())
    }

    pub fn save(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        progress: FrameProgress,
        resources: &[StateResource<'_>],
    ) -> Result<(), RenderError> {
        let body = self.body_size()?;
        atomic_write(&self.path, |file| {
            file.write_all(MAGIC)?;
            for (_, digest) in &self.identity {
                file.write_all(digest)?;
            }
            file.write_all(&progress.encode())?;
            for resource in resources {
                resource.download(device, queue, file, self.width, self.height)?;
            }
            if file.stream_position()? != body {
                return Err(invalid(
                    "renderer state did not match the declared dimensions",
                ));
            }
            let digest = hash_body(file, body)?;
            file.seek(SeekFrom::Start(body))?;
            file.write_all(&digest)?;
            Ok(())
        })
    }
}

fn hash_body(file: &mut File, body: u64) -> Result<[u8; 32], RenderError> {
    file.rewind()?;
    let buffer = BufReader::new(file.take(body));
    let _memory = tracked_host_allocation(buffer.capacity() as u64, "continua-checksum")?;
    Ok(hash(buffer)?)
}

fn atomic_write(
    path: &Path,
    write: impl FnOnce(&mut File) -> Result<(), RenderError>,
) -> Result<(), RenderError> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let directory = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (temp, mut file) = loop {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let temp = directory.join(format!(".continua-{}-{id}.tmp", std::process::id()));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temp)
        {
            Ok(file) => break (temp, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    let result = write(&mut file).and_then(|()| Ok(file.sync_all()?));
    drop(file);
    let result = result.and_then(|()| Ok(fs::rename(&temp, path)?));
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result?;
    #[cfg(unix)]
    File::open(directory)?.sync_all()?;
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) enum StateResource<'a> {
    Buffer(&'a wgpu::Buffer),
    Texture(&'a wgpu::Texture, u32),
}

impl StateResource<'_> {
    fn rows(self, width: u32, height: u32) -> (usize, usize, usize) {
        match self {
            Self::Buffer(buffer) => (buffer.size() as usize, buffer.size() as usize, 1),
            Self::Texture(_, bytes_per_pixel) => {
                let row = width * bytes_per_pixel;
                (row as usize, align_copy_bpr(row) as usize, height as usize)
            }
        }
    }

    fn copy(
        self,
        encoder: &mut wgpu::CommandEncoder,
        staging: &wgpu::Buffer,
        width: u32,
        height: u32,
        upload: bool,
    ) {
        match self {
            Self::Buffer(buffer) => {
                let (src, dst) = if upload {
                    (staging, buffer)
                } else {
                    (buffer, staging)
                };
                encoder.copy_buffer_to_buffer(src, 0, dst, 0, buffer.size());
            }
            Self::Texture(texture, bytes_per_pixel) => {
                let linear = wgpu::ImageCopyBuffer {
                    buffer: staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(align_copy_bpr(width * bytes_per_pixel)),
                        rows_per_image: Some(height),
                    },
                };
                let image = texture.as_image_copy();
                let extent = wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                };
                if upload {
                    encoder.copy_buffer_to_texture(linear, image, extent);
                } else {
                    encoder.copy_texture_to_buffer(image, linear, extent);
                }
            }
        }
    }

    fn upload(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        file: &mut File,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        let (row, stride, rows) = self.rows(width, height);
        let staging = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("continua-checkpoint-upload"),
                size: (stride * rows) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: true,
            },
        )?;
        {
            let mut bytes = staging.slice(..).get_mapped_range_mut();
            for data in bytes.chunks_exact_mut(stride) {
                file.read_exact(&mut data[..row])?;
            }
        }
        staging.unmap();
        let mut encoder = device.create_command_encoder(&Default::default());
        self.copy(&mut encoder, &staging, width, height, true);
        let submission = queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        Ok(())
    }

    fn download(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        file: &mut File,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        let (row, stride, rows) = self.rows(width, height);
        let staging = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("continua-checkpoint-readback"),
                size: (stride * rows) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let mut encoder = device.create_command_encoder(&Default::default());
        self.copy(&mut encoder, &staging, width, height, false);
        let submission = queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        receive
            .recv()
            .map_err(RenderError::readback)?
            .map_err(RenderError::readback)?;
        {
            let bytes = slice.get_mapped_range();
            for data in bytes.chunks_exact(stride) {
                file.write_all(&data[..row])?;
            }
        }
        staging.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continua_interrupted_replacement_preserves_published_file() {
        let path =
            std::env::temp_dir().join(format!("forge3d-continua-atomic-{}", std::process::id()));
        atomic_write(&path, |file| Ok(file.write_all(b"previous")?)).unwrap();
        assert!(atomic_write(&path, |file| {
            file.write_all(b"partial")?;
            Err(invalid("injected write failure"))
        })
        .is_err());
        assert_eq!(fs::read(&path).unwrap(), b"previous");
        atomic_write(&path, |file| Ok(file.write_all(b"replacement")?)).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"replacement");
        fs::remove_file(path).unwrap();
    }
}
