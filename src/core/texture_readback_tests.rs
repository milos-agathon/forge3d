use super::*;
use crate::core::degradation::{degradations_snapshot, Degradation};
use crate::core::hdr_readback::{
    read_hdr_texture, read_ldr_texture, read_r32_texture, read_r32uint_texture,
};
use crate::core::resource_tracker::{ledger_snapshot, tracked_create_texture, TrackedTexture};
use std::cell::RefCell;
use std::sync::Mutex;

pub(crate) struct GpuFixture {
    pub queue: Arc<Queue>,
    pub device: Arc<Device>,
    errors: Arc<Mutex<Vec<String>>>,
    degradations_before: Vec<Degradation>,
    _instance: wgpu::Instance,
}

impl GpuFixture {
    pub fn assert_clean(&self) {
        let errors = self.errors.lock().unwrap().clone();
        assert!(errors.is_empty(), "GPU errors: {errors:?}");
        assert_eq!(
            degradations_snapshot(),
            self.degradations_before,
            "readback changed the degradation log"
        );
    }
}

impl Drop for GpuFixture {
    fn drop(&mut self) {
        self.device.poll(wgpu::Maintain::Wait);
        if !std::thread::panicking() {
            self.assert_clean();
        }
    }
}

pub(crate) fn gpu() -> Option<GpuFixture> {
    match crate::core::gpu::try_ctx() {
        Ok(context) => {
            eprintln!(
                "Readback GPU: {:?}; physical={}",
                context.adapter.get_info(),
                crate::core::gpu::is_physical_proof_adapter(
                    &context.adapter.get_info(),
                    context.software_fallback,
                ),
            );
            // A separate instance/device keeps the capturing handler and wgpu
            // object IDs local to this test, without touching the shared device.
            // Record degradations just as the production handler does, while
            // also retaining every error so sink deduplication cannot hide it.
            let degradations_before = degradations_snapshot();
            let info = context.adapter.get_info();
            let backends = wgpu::Backends::from(info.backend);
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            });
            let adapter = instance
                .enumerate_adapters(backends)
                .into_iter()
                .find(|adapter| {
                    let candidate = adapter.get_info();
                    candidate.vendor == info.vendor
                        && candidate.device == info.device
                        && candidate.name == info.name
                        && candidate.backend == info.backend
                })
                .expect("readback fixture uses the reported adapter");
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("readback-error-capture-device"),
                    required_features: context.device.features(),
                    required_limits: context.device.limits(),
                },
                None,
            ))
            .expect("readback fixture device");
            let errors = Arc::new(Mutex::new(Vec::new()));
            let captured = errors.clone();
            device.on_uncaptured_error(Box::new(move |error| {
                let message = error.to_string();
                crate::core::degradation::record_degradation(
                    "uncaptured_gpu_error",
                    "wgpu",
                    &message,
                );
                captured.lock().unwrap().push(message);
            }));
            Some(GpuFixture {
                device: Arc::new(device),
                queue: Arc::new(queue),
                errors,
                degradations_before,
                _instance: instance,
            })
        }
        Err(error) => {
            eprintln!("Readback GPU unavailable; round trip NOT exercised: {error}");
            None
        }
    }
}

enum MappingHook {
    Abort,
    #[cfg(feature = "async_readback")]
    Pause {
        started: tokio::sync::oneshot::Sender<()>,
        resume: tokio::sync::oneshot::Receiver<()>,
    },
}

thread_local! {
    static MAPPING_HOOK: RefCell<Option<MappingHook>> = const { RefCell::new(None) };
    #[cfg(feature = "async_readback")]
    static BEFORE_POLL: RefCell<Option<tokio::sync::oneshot::Receiver<()>>> = const { RefCell::new(None) };
}

pub(crate) struct MappingHookGuard;

impl Drop for MappingHookGuard {
    fn drop(&mut self) {
        MAPPING_HOOK.with(|hook| {
            hook.borrow_mut().take();
        });
        #[cfg(feature = "async_readback")]
        BEFORE_POLL.with(|gate| {
            gate.borrow_mut().take();
        });
    }
}

pub(crate) fn abort_next_mapping() -> MappingHookGuard {
    MAPPING_HOOK.with(|hook| {
        assert!(hook.borrow_mut().replace(MappingHook::Abort).is_none());
    });
    MappingHookGuard
}

#[cfg(feature = "async_readback")]
pub(crate) fn pause_next_mapping() -> (
    MappingHookGuard,
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
) {
    let (started, observed) = tokio::sync::oneshot::channel();
    let (resume, paused) = tokio::sync::oneshot::channel();
    MAPPING_HOOK.with(|hook| {
        assert!(hook
            .borrow_mut()
            .replace(MappingHook::Pause {
                started,
                resume: paused
            })
            .is_none());
    });
    (MappingHookGuard, observed, resume)
}

pub(crate) fn mapping_started(buffer: &TrackedBuffer) {
    let hook = MAPPING_HOOK.with(|hook| hook.borrow_mut().take());
    match hook {
        Some(MappingHook::Abort) => buffer.unmap(),
        #[cfg(feature = "async_readback")]
        Some(MappingHook::Pause { started, resume }) => {
            BEFORE_POLL.with(|gate| *gate.borrow_mut() = Some(resume));
            started.send(()).expect("observe mapping before polling");
        }
        None => {}
    }
}

#[cfg(feature = "async_readback")]
pub(crate) async fn wait_before_poll() {
    let gate = BEFORE_POLL.with(|gate| gate.borrow_mut().take());
    if let Some(gate) = gate {
        let _ = gate.await;
    }
}

pub(crate) fn upload(
    device: &Device,
    queue: &Queue,
    width: u32,
    height: u32,
    format: TextureFormat,
    bytes: &[u8],
) -> TrackedTexture {
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("readback-round-trip-fixture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )
    .expect("tracked readback fixture texture");
    let layout = TransferLayout::queue_write(format, TextureAspect::All, size).unwrap();
    queue.write_texture(
        texture.as_image_copy(),
        bytes,
        layout.image_data_layout(),
        size,
    );
    texture
}

#[test]
fn supported_formats_round_trip_padded_and_aligned_rows() {
    let Some(context) = gpu() else { return };
    let (device, queue) = (&context.device, &context.queue);
    let before = ledger_snapshot();
    for width in [7, 64] {
        let height = 3;
        let pixels = (width * height) as usize;
        let ldr: Vec<_> = (0..pixels * 4).map(|index| index as u8).collect();
        for format in [
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba8Uint,
            TextureFormat::Rgba8Sint,
            TextureFormat::Rgba8Snorm,
            TextureFormat::R32Uint,
        ] {
            let texture = upload(device, queue, width, height, format, &ldr);
            assert_eq!(
                read_ldr_texture(device, queue, &texture, width, height).unwrap(),
                ldr
            );
        }

        let floats: Vec<_> = (0..pixels * 4)
            .map(|index| {
                // Include signed zero, infinities and a NaN payload; compare bits.
                [0x8000_0000, 0x3f00_0000, 0xff80_0000, 0x7fc0_0123][index % 4]
            })
            .collect();
        let bytes: Vec<_> = floats
            .iter()
            .flat_map(|bits: &u32| bits.to_le_bytes())
            .collect();
        let texture = upload(
            device,
            queue,
            width,
            height,
            TextureFormat::Rgba32Float,
            &bytes,
        );
        let actual = read_hdr_texture(
            device,
            queue,
            &texture,
            width,
            height,
            TextureFormat::Rgba32Float,
        )
        .unwrap();
        assert_eq!(
            actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            floats
        );

        let halves: Vec<_> = (0..pixels * 4)
            .map(|index| half::f16::from_f32([-0.0, 0.5, -2.0, 1024.0][index % 4]))
            .collect();
        let bytes: Vec<_> = halves
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let texture = upload(
            device,
            queue,
            width,
            height,
            TextureFormat::Rgba16Float,
            &bytes,
        );
        let actual = read_hdr_texture(
            device,
            queue,
            &texture,
            width,
            height,
            TextureFormat::Rgba16Float,
        )
        .unwrap();
        assert_eq!(
            actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            halves
                .iter()
                .map(|value| value.to_f32().to_bits())
                .collect::<Vec<_>>()
        );

        let scalar_bits: Vec<_> = floats[..pixels].to_vec();
        let bytes: Vec<_> = scalar_bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let texture = upload(
            device,
            queue,
            width,
            height,
            TextureFormat::R32Float,
            &bytes,
        );
        let actual = read_r32_texture(device, queue, &texture, width, height).unwrap();
        assert_eq!(
            actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            scalar_bits
        );

        let ids: Vec<_> = (0..pixels)
            .map(|index| [0, 0x0100_0001, u32::MAX, 0x1234_5678][index % 4])
            .collect();
        let bytes: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let texture = upload(device, queue, width, height, TextureFormat::R32Uint, &bytes);
        assert_eq!(
            read_r32uint_texture(device, queue, &texture, width, height).unwrap(),
            ids
        );
    }
    let after = ledger_snapshot();
    for label in [
        "hdr_readback.hdr_staging",
        "hdr_readback.ldr_staging",
        "hdr_readback.r32_staging",
        "hdr_readback.r32uint_staging",
    ] {
        assert_eq!(
            after.by_label.get(label),
            before.by_label.get(label),
            "staging leak: {label}"
        );
    }
}

#[test]
fn mapping_failure_and_discarded_callback_release_tracked_staging() {
    let Some(context) = gpu() else { return };
    let (device, queue) = (&context.device, &context.queue);
    let texture = upload(device, queue, 7, 3, TextureFormat::Rgba8Unorm, &[37; 84]);
    let request = ReadbackRequest::new(&texture, 7, 3, TextureFormat::Rgba8Unorm).unwrap();
    let label = "readback-mapping-failure-fixture";
    let before = ledger_snapshot().by_label.get(label).copied();
    {
        // Abort a real wgpu map after it starts, exercising the entire public
        // reader -> read_sync -> mapping-error branch rather than a decoder.
        let _abort = abort_next_mapping();
        let error = read_ldr_texture(device, queue, &texture, 7, 3).unwrap_err();
        assert!(error.contains("Buffer mapping failed"), "{error}");
    }
    assert!(!ledger_snapshot()
        .by_label
        .contains_key("hdr_readback.ldr_staging"));
    context.assert_clean();
    assert_eq!(ledger_snapshot().by_label.get(label).copied(), before);

    let transfer = StagingLease::allocate(device, request.layout.transfer_bytes, label)
        .unwrap()
        .submit(device, queue, &texture, request);
    drop(transfer.start_mapping());
    drop(transfer);
    device.poll(wgpu::Maintain::Wait);
    assert_eq!(ledger_snapshot().by_label.get(label).copied(), before);
    // A later transfer on the same device remains usable after failure.
    assert_eq!(
        read_ldr_texture(device, queue, &texture, 7, 3).unwrap(),
        vec![37; 84]
    );
}

#[test]
fn invalid_requests_fail_before_allocating_or_submitting() {
    let Some(context) = gpu() else { return };
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        TextureFormat::Rgba8Unorm,
        &[0; 84],
    );
    let before = ledger_snapshot().by_label;
    for (width, height) in [(0, 3), (7, 0), (8, 3), (7, 4)] {
        assert!(
            read_ldr_texture(&context.device, &context.queue, &texture, width, height).is_err()
        );
    }
    assert!(read_hdr_texture(
        &context.device,
        &context.queue,
        &texture,
        7,
        3,
        TextureFormat::Rgba32Float
    )
    .is_err());
    assert!(read_hdr_texture(
        &context.device,
        &context.queue,
        &texture,
        7,
        3,
        TextureFormat::Rgba8Unorm
    )
    .is_err());
    assert!(read_r32_texture(&context.device, &context.queue, &texture, 7, 3).is_err());
    assert!(read_r32uint_texture(&context.device, &context.queue, &texture, 7, 3).is_err());
    assert_eq!(ledger_snapshot().by_label, before);
}

#[cfg(feature = "async_readback")]
#[tokio::test]
async fn pooled_mapping_failure_and_cancellation_return_an_unmapped_lease() {
    let Some(context) = gpu() else { return };
    let texture = upload(
        &context.device,
        &context.queue,
        7,
        3,
        TextureFormat::Rgba8Unorm,
        &[59; 84],
    );
    let request = ReadbackRequest::new(&texture, 7, 3, TextureFormat::Rgba8Unorm).unwrap();
    let pool = pool::StagingPool::default();
    let transfer = pool
        .acquire(&context.device, request.layout.transfer_bytes, true)
        .unwrap()
        .submit(&context.device, &context.queue, &texture, request);
    let receiver = transfer.start_mapping();
    transfer.staging.resources.buffer.unmap();
    context.device.poll(wgpu::Maintain::Wait);
    assert!(transfer.mapped_bytes(receiver.recv().unwrap()).is_err());
    drop(transfer);
    assert_eq!(pool.stats(), (1, 0, request.layout.transfer_bytes));

    let transfer = pool
        .acquire(&context.device, request.layout.transfer_bytes, true)
        .unwrap()
        .submit(&context.device, &context.queue, &texture, request);
    let receiver = transfer.start_mapping();
    // Discard an in-flight mapping owner and its receiver, as cancelling its
    // future does. The real callback still retires the lease without a runtime.
    drop(receiver);
    drop(transfer);
    context.device.poll(wgpu::Maintain::Wait);
    assert_eq!(pool.stats(), (1, 0, request.layout.transfer_bytes));
    context.assert_clean();
    let transfer = pool
        .acquire(&context.device, request.layout.transfer_bytes, true)
        .unwrap()
        .submit(&context.device, &context.queue, &texture, request);
    assert_eq!(
        transfer.read_async(&context.device).await.unwrap(),
        vec![59; 84]
    );
    assert_eq!(pool.stats(), (1, 0, request.layout.transfer_bytes));
}
