//! Texture-to-host transfers. A mapping callback retains the staging lease so
//! cleanup can finish even after its future or result receiver is discarded.

use super::error::RenderError;
use super::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use super::texture_layout::TransferLayout;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{mpsc, Arc};
use wgpu::{Device, Extent3d, Queue, Texture, TextureAspect, TextureFormat};

#[derive(Clone, Copy)]
pub(crate) struct ReadbackRequest {
    pub layout: TransferLayout,
    extent: Extent3d,
}

impl ReadbackRequest {
    pub fn new(
        texture: &Texture,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Self, RenderError> {
        let extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let layout = TransferLayout::buffer_copy(format, TextureAspect::All, extent)
            .map_err(RenderError::readback)?;
        // The byte-oriented APIs have always copied four-byte texels without
        // interpreting them (including BGRA, sRGB and integer/scalar formats).
        let compatible = texture.format() == format
            || (format == TextureFormat::Rgba8Unorm
                && texture.format().block_dimensions() == (1, 1)
                && texture.format().block_copy_size(Some(TextureAspect::All)) == Some(4));
        if !compatible {
            return Err(RenderError::readback("Texture readback format mismatch"));
        }
        if width > texture.width() || height > texture.height() {
            return Err(RenderError::readback("Readback exceeds texture dimensions"));
        }
        if texture.sample_count() != 1 {
            return Err(RenderError::readback(
                "Readback requires a resolved texture",
            ));
        }
        if !texture.usage().contains(wgpu::TextureUsages::COPY_SRC) {
            return Err(RenderError::readback(
                "Texture requires COPY_SRC for readback",
            ));
        }
        Ok(Self { layout, extent })
    }
}

pub(crate) struct StagingLease {
    resources: Arc<LeaseResources>,
}

#[repr(u8)]
enum MappingState {
    NotRequested,
    Pending,
    Mapped,
    Failed,
}

struct LeaseResources {
    buffer: Arc<TrackedBuffer>,
    mapping: AtomicU8,
    #[cfg(feature = "async_readback")]
    pool_slot: Option<pool::PoolSlot>,
}

impl StagingLease {
    pub fn allocate(device: &Device, size: u64, label: &str) -> Result<Self, RenderError> {
        if size > device.limits().max_buffer_size {
            return Err(RenderError::readback(
                "Readback exceeds device buffer limit",
            ));
        }
        let buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        Ok(Self {
            resources: Arc::new(LeaseResources {
                buffer: Arc::new(buffer),
                mapping: AtomicU8::new(MappingState::NotRequested as u8),
                #[cfg(feature = "async_readback")]
                pool_slot: None,
            }),
        })
    }

    pub fn submit(
        self,
        device: &Device,
        queue: &Queue,
        texture: &Texture,
        request: ReadbackRequest,
    ) -> TextureReadback {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("texture-readback-copy"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.resources.buffer,
                layout: request.layout.image_data_layout(),
            },
            request.extent,
        );
        queue.submit(Some(encoder.finish()));
        TextureReadback {
            staging: self,
            layout: request.layout,
        }
    }
}

impl Drop for LeaseResources {
    fn drop(&mut self) {
        // wgpu 0.19 rejects unmapping an idle buffer. Pending mappings keep
        // this owner alive in their callback until success/failure is known:
        // even Pending -> Failed can race with unmap on another poll thread.
        if self.mapping.load(Ordering::Acquire) == MappingState::Mapped as u8 {
            self.buffer.unmap();
        }
        // A successfully mapped buffer is unmapped before it can be reused;
        // failed and never-requested mappings already have idle buffers.
        #[cfg(feature = "async_readback")]
        if let Some(slot) = &self.pool_slot {
            slot.release();
        }
    }
}

pub(crate) struct TextureReadback {
    staging: StagingLease,
    layout: TransferLayout,
}

impl TextureReadback {
    fn start_mapping(&self) -> mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let (sender, receiver) = mpsc::channel();
        let resources = self.staging.resources.clone();
        resources
            .mapping
            .store(MappingState::Pending as u8, Ordering::Release);
        self.staging
            .resources
            .buffer
            .slice(..self.layout.transfer_bytes)
            .map_async(wgpu::MapMode::Read, move |result| {
                let state = if result.is_ok() {
                    MappingState::Mapped
                } else {
                    MappingState::Failed
                };
                resources.mapping.store(state as u8, Ordering::Release);
                // Drop callback ownership before waking the reader. If the
                // transfer was cancelled, this also retires its staging lease.
                drop(resources);
                // Cancellation may discard the receiver before the GPU calls back.
                let _ = sender.send(result);
            });
        #[cfg(all(test, not(target_arch = "wasm32")))]
        tests::mapping_started(&self.staging.resources.buffer);
        receiver
    }

    fn mapped_bytes(
        &self,
        result: Result<(), wgpu::BufferAsyncError>,
    ) -> Result<Vec<u8>, RenderError> {
        result.map_err(|e| RenderError::readback(format!("Buffer mapping failed: {e:?}")))?;
        let data = self
            .staging
            .resources
            .buffer
            .slice(..self.layout.transfer_bytes)
            .get_mapped_range();
        let capacity = usize::try_from(self.layout.payload_bytes)
            .map_err(|_| RenderError::readback("Readback payload exceeds host address space"))?;
        let mut bytes = Vec::with_capacity(capacity);
        for row in data.chunks_exact(self.layout.bytes_per_row as usize) {
            bytes.extend_from_slice(&row[..self.layout.row_bytes as usize]);
        }
        Ok(bytes)
    }

    pub fn read_sync(self, device: &Device) -> Result<Vec<u8>, RenderError> {
        let receiver = self.start_mapping();
        loop {
            // wgpu 0.19's blocking wait discards the HAL timeout result and
            // can retire unfinished Metal work. Poll the actual fence until
            // the mapping callback proves this transfer has completed.
            device.poll(wgpu::Maintain::Poll);
            match receiver.try_recv() {
                Ok(result) => return self.mapped_bytes(result),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(RenderError::readback("Buffer mapping callback lost"));
                }
                Err(mpsc::TryRecvError::Empty) => std::thread::yield_now(),
            }
        }
    }

    #[cfg(feature = "async_readback")]
    pub async fn read_async(self, device: &Device) -> Result<Vec<u8>, RenderError> {
        let receiver = self.start_mapping();
        #[cfg(all(test, not(target_arch = "wasm32")))]
        tests::wait_before_poll().await;
        loop {
            device.poll(wgpu::Maintain::Poll);
            match receiver.try_recv() {
                Ok(result) => return self.mapped_bytes(result),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(RenderError::readback("Buffer mapping callback lost"));
                }
                Err(mpsc::TryRecvError::Empty) => tokio::task::yield_now().await,
            }
        }
    }
}

pub(crate) fn read_texture_sync(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
    format: TextureFormat,
    label: &str,
) -> Result<Vec<u8>, RenderError> {
    let request = ReadbackRequest::new(texture, width, height, format)?;
    StagingLease::allocate(device, request.layout.transfer_bytes, label)?
        .submit(device, queue, texture, request)
        .read_sync(device)
}

#[cfg(feature = "async_readback")]
pub(crate) mod pool {
    use super::*;
    use std::sync::Mutex;

    struct Slot {
        buffer: Arc<TrackedBuffer>,
        size: u64,
        in_use: bool,
    }

    pub(super) struct PoolSlot {
        slots: Arc<Mutex<Vec<Slot>>>,
        index: usize,
    }

    impl PoolSlot {
        pub fn release(&self) {
            self.slots.lock().unwrap()[self.index].in_use = false;
        }
    }

    #[derive(Default)]
    pub(crate) struct StagingPool {
        slots: Arc<Mutex<Vec<Slot>>>,
    }

    impl StagingPool {
        pub fn acquire(
            &self,
            device: &Device,
            size: u64,
            pre_allocate: bool,
        ) -> Result<StagingLease, RenderError> {
            let mut slots = self.slots.lock().unwrap();
            if let Some(index) = slots
                .iter()
                .position(|slot| !slot.in_use && slot.size >= size)
            {
                slots[index].in_use = true;
                return Ok(StagingLease {
                    resources: Arc::new(LeaseResources {
                        buffer: slots[index].buffer.clone(),
                        mapping: AtomicU8::new(MappingState::NotRequested as u8),
                        pool_slot: Some(PoolSlot {
                            slots: self.slots.clone(),
                            index,
                        }),
                    }),
                });
            }
            // Preserve the existing allocation policy: retain newly allocated
            // buffers when requested, or retain the first buffer otherwise.
            if pre_allocate || slots.is_empty() {
                let mut lease = StagingLease::allocate(
                    device,
                    size,
                    &format!("async-readback-{}", slots.len()),
                )?;
                let index = slots.len();
                slots.push(Slot {
                    buffer: lease.resources.buffer.clone(),
                    size,
                    in_use: true,
                });
                Arc::get_mut(&mut lease.resources)
                    .expect("new lease has one owner")
                    .pool_slot = Some(PoolSlot {
                    slots: self.slots.clone(),
                    index,
                });
                Ok(lease)
            } else {
                StagingLease::allocate(device, size, "async_readback.temp_readback")
            }
        }

        pub fn stats(&self) -> (usize, usize, u64) {
            let slots = self.slots.lock().unwrap();
            (
                slots.len(),
                slots.iter().filter(|slot| slot.in_use).count(),
                slots.iter().map(|slot| slot.size).sum(),
            )
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "texture_readback_tests.rs"]
pub(crate) mod tests;
