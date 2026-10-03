// tests/test_splat_fusion_occlusion.rs
// SPLAT-FUSED acceptance: cross-representation shadow correctness at
// billion-primitive scale inside the 512 MiB budget.
//
// The scene is the committed three-representation fixture (mini DEM with a
// ridge, a Gaussian splat cloud, a COPC LiDAR swath) embedded in a synthetic
// billion-point page index. It is rendered twice:
//
//   * by the fused ReSTIR integrator (HybridPathTracer::render_fused), whose
//     shadow mask is the unified transmittance T_total < 1/2, and
//   * by the AEQUITAS wavefront path tracer over hard proxies of the same
//     primitives (path_tracing::fused_reference), whose shadow mask is read
//     from the path-traced radiance.
//
// Gate: IoU(fused_shadow, reference_shadow) > 0.9 for "splat shadow on
// terrain" and "terrain shadow on splat/LiDAR"; residency peak <= 512 MiB;
// logical primitive count > 1e9.
//
// The remaining tests pin the pieces that gate rests on: the GPU traversal
// against a brute-force CPU evaluation of the same analytic model, the two
// paging policies, the diagnostics, and a deterministic golden image.
//
// GPU tests follow the repo convention: with no adapter they report and
// return (hosted Linux has none); FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 turns a
// missing adapter into a failure on the hardware lane.
#![cfg(feature = "splat-fusion")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use forge3d::path_tracing::fused_reference::{
    render_fused_reference, shadow_iou, FusedReferenceDesc, FusedReferenceOutput, ReceiverClass,
    ShadowIou,
};
use forge3d::path_tracing::hybrid_compute::{
    AlbedoSampling, FusedRenderDesc, FusedRenderOutput, FusedTerrainDesc, HybridPathTracer,
    MinMaxPrecision, TerrainAlbedoMap, TerrainPtScene,
};
use forge3d::splat::bvh::Aabb;
use forge3d::splat::fixture::{FixtureManifest, FixtureScene};
use forge3d::splat::fusion::{
    FusedScene, FusionParams, PagingPolicy, REQUIRED_STORAGE_BUFFERS_PER_STAGE,
};
use forge3d::splat::kernel::{
    disc_coverage, ray_gaussian, sphelet_coverage, splat_transmittance, terrain_smooth_normal,
};
use forge3d::splat::load_gaussian_splats;
use forge3d::splat::stream::{
    write_synthetic_point_field, CopcPageSource, DecodedPage, PageKind, PageSource, PageStoreFile,
    PageStoreWriter, PointCloudFrame, SplatCloudSource, SyntheticFieldDesc,
};
use forge3d::splat::surfel::{oct_decode, SURFEL_SPHERE};
use forge3d::splat::GaussianSplatCloud;

const SIZE: u32 = 256;
const IOU_GATE: f32 = 0.9;
const BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const HIT_TERRAIN: u8 = 3;
const HIT_SPLAT: u8 = 4;
const HIT_LIDAR: u8 = 5;
/// Receivers must face the sun by at least this cosine to be classified.
const MIN_SUN_COSINE: f32 = 0.15;
/// Project golden tolerance (tests/test_hybrid_terrain_pt.py).
const GOLDEN_SSIM_MIN: f64 = 0.995;
const GOLDEN_MEAN_ABS_MAX: f64 = 2.0;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("splat_fusion")
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("splat_fusion")
        .join("fused_fixture.png")
}

fn artifact_dir() -> Option<PathBuf> {
    std::env::var_os("FORGE3D_SPLAT_FUSION_ARTIFACT_DIR").map(|dir| {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    })
}

fn save_rgba(name: &str, width: u32, height: u32, rgba: &[u8]) {
    if let Some(dir) = artifact_dir() {
        image::save_buffer(dir.join(name), rgba, width, height, image::ColorType::Rgba8).unwrap();
    }
}

fn save_gray(name: &str, width: u32, height: u32, value: impl Fn(usize) -> f32) {
    let pixels = (width * height) as usize;
    let mut rgba = Vec::with_capacity(pixels * 4);
    for i in 0..pixels {
        let v = (value(i).clamp(0.0, 1.0) * 255.0) as u8;
        rgba.extend_from_slice(&[v, v, v, 255]);
    }
    save_rgba(name, width, height, &rgba);
}

fn gpu_available() -> bool {
    let unavailable = match forge3d::core::gpu::try_ctx() {
        Err(error) => Some(format!("no usable GPU adapter ({error})")),
        Ok(ctx) => {
            let info = ctx.adapter.get_info();
            let name = info.name.to_lowercase();
            let virtualized = ["paravirtual", "virtio", "vmware", "virtualbox", "qxl"]
                .iter()
                .any(|marker| name.contains(marker));
            let physical = !ctx.software_fallback
                && !virtualized
                && matches!(
                    info.device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                );
            let storage = ctx.device.limits().max_storage_buffers_per_shader_stage;
            if !physical {
                // Software rasterizers (WARP, lavapipe) and hypervisor GPUs are
                // the hosted matrices; the gate runs on the hardware lane.
                Some(format!(
                    "'{}' ({:?}) is not a physical adapter",
                    info.name, info.device_type
                ))
            } else if storage < REQUIRED_STORAGE_BUFFERS_PER_STAGE {
                Some(format!(
                    "'{}' exposes {storage} storage buffers per stage, the fused kernel needs {REQUIRED_STORAGE_BUFFERS_PER_STAGE}",
                    info.name
                ))
            } else {
                None
            }
        }
    };
    match unavailable {
        None => true,
        Some(reason) => {
            assert!(
                std::env::var_os("FORGE3D_SPLAT_FUSION_REQUIRED_GPU").is_none(),
                "FORGE3D_SPLAT_FUSION_REQUIRED_GPU is set but the fused GPU path is unavailable: {reason}"
            );
            eprintln!("SPLAT-FUSED GPU acceptance skipped: {reason}");
            false
        }
    }
}

fn toward_sun(m: &FixtureManifest) -> [f32; 3] {
    let (az, el) = (
        m.sun_azimuth_deg.to_radians(),
        m.sun_elevation_deg.to_radians(),
    );
    [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()]
}

fn terrain_desc(fixture: &FixtureScene) -> FusedTerrainDesc {
    let m = &fixture.manifest;
    FusedTerrainDesc {
        heights: fixture.heights.clone(),
        width: m.dem_width,
        height: m.dem_height,
        spacing: (m.dem_spacing[0], m.dem_spacing[1]),
        exaggeration: 1.0,
        albedo: m.terrain_albedo,
        // The shipped setting, so the acceptance suite exercises it.
        minmax_precision: MinMaxPrecision::F16Conservative,
        albedo_map: None,
        albedo_sampling: AlbedoSampling::Bilinear,
    }
}

fn fixture_params(fixture: &FixtureScene) -> FusionParams {
    FusionParams {
        lidar_radius: fixture.manifest.lidar_radius,
        policy: PagingPolicy::Exact,
        ..FusionParams::default()
    }
}

fn load_cloud(fixture: &FixtureScene) -> Arc<GaussianSplatCloud> {
    let cloud = Arc::new(load_gaussian_splats(fixture.splat_path()).unwrap());
    assert_eq!(cloud.len() as u32, fixture.manifest.splat_count);
    cloud
}

fn open_copc(fixture: &FixtureScene, params: &FusionParams) -> CopcPageSource {
    let source = CopcPageSource::open(
        fixture.copc_path(),
        PointCloudFrame {
            origin: fixture.manifest.copc_origin,
            z_up: true,
        },
        params.page_capacity,
    )
    .unwrap();
    assert_eq!(
        source.logical_primitives(),
        u64::from(fixture.manifest.copc_point_count)
    );
    source
}

/// The fixture's splat cloud and COPC swath as page sources.
fn fixture_sources(fixture: &FixtureScene, params: &FusionParams) -> Vec<Arc<dyn PageSource>> {
    let splats: Arc<dyn PageSource> = Arc::new(SplatCloudSource::new(load_cloud(fixture), 512));
    let lidar: Arc<dyn PageSource> = Arc::new(open_copc(fixture, params));
    vec![splats, lidar]
}

fn fused_desc<'a>(
    scene: &'a FusedScene,
    fixture: &FixtureScene,
    size: u32,
    spp: u32,
    frames: u32,
) -> FusedRenderDesc<'a> {
    let m = &fixture.manifest;
    FusedRenderDesc {
        scene,
        terrain: Some(Arc::new(terrain_desc(fixture))),
        cam_origin: m.cam_origin,
        cam_look_at: m.cam_look_at,
        cam_up: m.cam_up,
        fov_y_deg: m.fov_y_deg,
        exposure: 1.0,
        sun_azimuth_deg: m.sun_azimuth_deg,
        sun_elevation_deg: m.sun_elevation_deg,
        sun_intensity: 2.5,
        sun_color: [1.0, 0.97, 0.92],
        sky_turbidity: 2.5,
        sky_ground_albedo: 0.2,
        sky_intensity: 0.35,
        width: size,
        height: size,
        seed: 7,
        spp,
        frames,
        tile: None,
        aovs: true,
    }
}

/// A synthetic billion-point index around the fixture core: a tiled LiDAR
/// field whose ~246k index entries alias a handful of on-disk payload pages.
/// Nothing of it is resident until a ray asks for it.
fn billion_point_field(work: &Path, fixture: &FixtureScene, params: &FusionParams) -> PathBuf {
    let m = &fixture.manifest;
    let path = work.join("billion_point_field.f3dpages");
    let half = 0.5 * (m.dem_width - 1) as f32 * m.dem_spacing[0];
    let field = write_synthetic_point_field(
        &path,
        &SyntheticFieldDesc {
            page_capacity: params.page_capacity,
            template_pages: 16,
            grid: (496, 496),
            cell_size: 16.0,
            origin: [-3968.0, 0.6, -3968.0],
            thickness: 1.5,
            // Keep the DEM footprint clear: the fixture core is the hole.
            hole: [-half, -half, half, half],
            seed: 0x5eed_f00d,
        },
    )
    .unwrap();
    eprintln!(
        "synthetic index: {} pages, {} points, {} payload bytes on disk ({} file bytes)",
        field.page_count, field.logical_primitives, field.payload_bytes, field.file_bytes
    );
    assert!(field.logical_primitives > 1_000_000_000);
    path
}

fn work_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("forge3d-splat-fusion-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// World-space xz rectangle test on the fused G-buffer position.
fn in_rect(out: &FusedRenderOutput, i: usize, rect: [f32; 4]) -> bool {
    let (x, z) = (out.position[3 * i], out.position[3 * i + 2]);
    x >= rect[0] && x <= rect[2] && z >= rect[1] && z <= rect[3]
}

struct Case {
    name: &'static str,
    /// Fused hit kinds of the receivers.
    receivers: &'static [u8],
    /// Reference classes of the receivers (parallel to `receivers`).
    classes: &'static [ReceiverClass],
    /// World xz rectangle (min_x, min_z, max_x, max_z) the case covers.
    rect: [f32; 4],
}

/// The cross-representation cases. Rectangles are fixed in world space from
/// the fixture layout (occluder footprint displaced along the sun direction),
/// independent of either render.
const CASES: &[Case] = &[
    Case {
        name: "splat shadow on terrain",
        receivers: &[HIT_TERRAIN],
        classes: &[ReceiverClass::Terrain],
        rect: [10.0, -16.0, 27.0, -3.0],
    },
    Case {
        name: "LiDAR shadow on terrain",
        receivers: &[HIT_TERRAIN],
        classes: &[ReceiverClass::Terrain],
        rect: [8.0, 6.0, 21.0, 18.0],
    },
    Case {
        name: "terrain shadow on splat",
        receivers: &[HIT_SPLAT],
        classes: &[ReceiverClass::Splat],
        rect: [-17.0, -12.0, -3.0, 0.0],
    },
    Case {
        name: "terrain shadow on LiDAR",
        receivers: &[HIT_LIDAR],
        classes: &[ReceiverClass::Lidar],
        rect: [-17.0, 3.0, -3.0, 15.0],
    },
    Case {
        name: "terrain shadow on splat/LiDAR",
        receivers: &[HIT_SPLAT, HIT_LIDAR],
        classes: &[ReceiverClass::Splat, ReceiverClass::Lidar],
        rect: [-17.0, -12.0, -3.0, 15.0],
    },
];

/// Binary shadow masks of both renders and their IoU per case.
fn score(
    fused: &FusedRenderOutput,
    reference: &FusedReferenceOutput,
    sun: [f32; 3],
) -> Vec<(&'static Case, ShadowIou)> {
    let pixels = fused.hit_kind.len();
    let (w, h) = (fused.width, fused.height);
    // Fused mask: the unified occlusion model toward the sun.
    let fused_shadow: Vec<bool> = (0..pixels)
        .map(|i| fused.transmittance[4 * i] < 0.5)
        .collect();
    // Reference mask: read from the path-traced radiance.
    let reference_mask = reference.shadow_mask(sun, MIN_SUN_COSINE);
    let reference_shadow: Vec<bool> = reference_mask.iter().map(|m| *m == Some(true)).collect();
    let mut out = Vec::new();
    for case in CASES {
        let valid: Vec<bool> = (0..pixels)
            .map(|i| {
                let Some(slot) = case.receivers.iter().position(|k| *k == fused.hit_kind[i]) else {
                    return false;
                };
                reference.class(i) == case.classes[slot]
                    && reference_mask[i].is_some()
                    && fused.sun_cosine[i] > MIN_SUN_COSINE
                    && in_rect(fused, i, case.rect)
            })
            .collect();
        let iou = shadow_iou(&fused_shadow, &reference_shadow, &valid);
        eprintln!(
            "  {:<30} IoU {:.4}  (valid {} px, fused shadow {}, reference shadow {}, \
             intersection {}, union {})",
            case.name,
            iou.iou,
            iou.valid_pixels,
            iou.fused_shadow,
            iou.reference_shadow,
            iou.intersection,
            iou.union
        );
        let tag = case.name.replace([' ', '/'], "_");
        save_gray(&format!("mask_{tag}.png"), w, h, |i| {
            if !valid[i] {
                0.25
            } else {
                match (fused_shadow[i], reference_shadow[i]) {
                    (true, true) => 0.0,
                    (false, false) => 1.0,
                    (true, false) => 0.6,
                    (false, true) => 0.45,
                }
            }
        });
        out.push((case, iou));
    }
    save_gray("fused_shadow.png", w, h, |i| {
        if fused.hit_kind[i] == 0 {
            0.25
        } else if fused_shadow[i] {
            0.0
        } else {
            1.0
        }
    });
    save_gray("reference_shadow.png", w, h, |i| match reference_mask[i] {
        None => 0.25,
        Some(true) => 0.0,
        Some(false) => 1.0,
    });
    out
}

fn render_reference(
    scene: &FusedScene,
    fixture: &FixtureScene,
    size: u32,
    frames: u32,
) -> FusedReferenceOutput {
    let ctx = forge3d::core::gpu::try_ctx().unwrap();
    let m = &fixture.manifest;
    let terrain = terrain_desc(fixture);
    let half = 0.5 * (m.dem_width - 1) as f32 * m.dem_spacing[0];
    let reference = render_fused_reference(
        &ctx.device,
        &ctx.queue,
        &FusedReferenceDesc {
            scene,
            terrain: Some(&terrain),
            // The fixture core: everything strictly inside the DEM
            // footprint (the far-field tiles only touch its edge).
            region: Aabb::new(
                [-half + 0.5, -1000.0, -half + 0.5],
                [half - 0.5, 1000.0, half - 0.5],
            ),
            cam_origin: m.cam_origin,
            cam_look_at: m.cam_look_at,
            cam_up: m.cam_up,
            fov_y_deg: m.fov_y_deg,
            sun_azimuth_deg: m.sun_azimuth_deg,
            sun_elevation_deg: m.sun_elevation_deg,
            sun_intensity: 2.5,
            sun_color: [1.0, 0.97, 0.92],
            width: size,
            height: size,
            spp_frames: frames,
            seed: 11,
        },
    )
    .unwrap();
    let pixels = reference.depth.len();
    let mut sorted: Vec<f32> = reference.radiance.clone();
    sorted.sort_by(f32::total_cmp);
    let white = sorted[sorted.len() * 9 / 10].max(1e-6) * 1.5;
    let mut rgba = Vec::with_capacity(pixels * 4);
    for i in 0..pixels {
        for c in 0..3 {
            let v = (reference.radiance[3 * i + c] / white)
                .clamp(0.0, 1.0)
                .powf(1.0 / 2.2);
            rgba.push((v * 255.0) as u8);
        }
        rgba.push(255);
    }
    save_rgba("reference_radiance.png", size, size, &rgba);
    reference
}

fn dump_fused(prefix: &str, out: &FusedRenderOutput) {
    let (w, h) = (out.width, out.height);
    save_rgba(&format!("{prefix}_beauty.png"), w, h, &out.rgba);
    for (channel, name) in ["total", "splat", "lidar", "terrain"].iter().enumerate() {
        save_gray(&format!("{prefix}_t_{name}.png"), w, h, |i| {
            out.transmittance[4 * i + channel]
        });
    }
    save_gray(&format!("{prefix}_hit_kind.png"), w, h, |i| {
        f32::from(out.hit_kind[i]) / 5.0
    });
    save_gray(&format!("{prefix}_reservoir_visibility.png"), w, h, |i| {
        out.reservoir_visibility[i]
    });
}

fn hit_counts(out: &FusedRenderOutput) -> [usize; 6] {
    let mut kinds = [0usize; 6];
    for kind in &out.hit_kind {
        kinds[(*kind as usize).min(5)] += 1;
    }
    kinds
}

/// THE definition-of-done gate.
#[test]
fn test_splat_fusion_occlusion() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let m = &fixture.manifest;
    let params = fixture_params(&fixture);
    let work = work_dir("dod");

    let mut sources = fixture_sources(&fixture, &params);
    let fixture_pages: u32 = sources.iter().map(|s| s.page_count()).sum();
    let field_path = billion_point_field(&work, &fixture, &params);
    let far_field: Arc<dyn PageSource> = Arc::new(PageStoreFile::open(&field_path).unwrap());
    sources.push(far_field);
    let scene = FusedScene::new(sources, params.clone()).unwrap();
    eprintln!(
        "fused scene: {} pages, {} logical primitives ({})",
        scene.page_count(),
        scene.logical_primitive_count(),
        scene.describe()
    );

    // --- Fused ReSTIR render ---
    let tracer = HybridPathTracer::new_fused().unwrap();
    let started = std::time::Instant::now();
    let fused = tracer
        .render_fused(&fused_desc(&scene, &fixture, SIZE, 2, 48))
        .unwrap();
    eprintln!(
        "fused render: {:.2}s, {} frames, {} restarts, variance {:.3e}, TLAS {} nodes, hits \
         (miss,-,-,terrain,splat,lidar) {:?}",
        started.elapsed().as_secs_f32(),
        fused.frames,
        fused.restarts,
        fused.variance,
        fused.tlas_node_count,
        hit_counts(&fused)
    );
    eprintln!("paging: {:?}", fused.paging);
    dump_fused("fused", &fused);

    // --- AEQUITAS path-traced occlusion reference over hard proxies ---
    let started = std::time::Instant::now();
    let reference = render_reference(&scene, &fixture, SIZE, 96);
    eprintln!(
        "reference render: {:.2}s, {} terrain triangles, {} splat proxies, {} LiDAR proxies",
        started.elapsed().as_secs_f32(),
        reference.terrain_triangles,
        reference.splat_proxies,
        reference.lidar_proxies
    );
    assert_eq!(reference.splat_proxies, m.splat_count);
    assert_eq!(reference.lidar_proxies, m.copc_point_count);

    // --- Shadow IoU: the hard gate ---
    eprintln!("shadow IoU, fused vs path-traced reference (gate > {IOU_GATE}):");
    for (case, iou) in score(&fused, &reference, toward_sun(m)) {
        let name = case.name;
        assert!(
            iou.valid_pixels > 300 && iou.fused_shadow > 60 && iou.reference_shadow > 60,
            "{name}: the case is not exercised by enough pixels: {iou:?}"
        );
        assert!(
            iou.fused_shadow < iou.valid_pixels && iou.reference_shadow < iou.valid_pixels,
            "{name}: every receiver pixel is shadowed, the mask has no lit side: {iou:?}"
        );
        assert!(
            iou.iou > IOU_GATE,
            "{name}: shadow IoU {:.4} <= {IOU_GATE} ({iou:?})",
            iou.iou
        );
    }

    // --- Billion-primitive scale, paged ---
    assert!(
        fused.logical_primitives > 1_000_000_000,
        "logical scene holds {} primitives",
        fused.logical_primitives
    );
    let resident =
        u64::from(fused.paging.residency.peak_resident_pages) * u64::from(params.page_capacity);
    eprintln!(
        "logical primitives: {} | peak resident pages: {} of {} ({} primitive slots, {:.6} % of \
         the scene)",
        fused.logical_primitives,
        fused.paging.residency.peak_resident_pages,
        fused.page_count,
        resident,
        100.0 * resident as f64 / fused.logical_primitives as f64
    );
    assert!(
        fused.paging.residency.peak_resident_pages < fused.page_count / 100,
        "residency must be paged, not resident: {:?}",
        fused.paging
    );
    assert!(fused.paging.miss_events > 0, "the paging path never ran");
    // Pages of the billion-point index were streamed from disk on demand,
    // on top of the fixture's own pages.
    assert!(
        fused.paging.residency.loads > u64::from(fixture_pages),
        "no far-field page was ever requested: {:?}",
        fused.paging
    );

    // --- 512 MiB budget through the memory tracker ---
    eprintln!(
        "memory: render peak total {} B ({:.1} MiB), render peak host-visible {} B ({:.1} MiB), \
         residency pool {} B ({:.1} MiB), process host-visible peak {} B, limit {} B",
        fused.peak_total_bytes,
        fused.peak_total_bytes as f64 / 1048576.0,
        fused.peak_host_visible_bytes,
        fused.peak_host_visible_bytes as f64 / 1048576.0,
        fused.paging.pool_bytes,
        fused.paging.pool_bytes as f64 / 1048576.0,
        fused.tracker_peak_host_visible_bytes,
        fused.tracker_limit_bytes
    );
    assert_eq!(fused.tracker_limit_bytes, BUDGET_BYTES);
    assert!(fused.peak_host_visible_bytes <= BUDGET_BYTES);
    assert!(fused.peak_total_bytes <= BUDGET_BYTES);
    assert!(fused.tracker_peak_host_visible_bytes <= BUDGET_BYTES);
    assert!(fused.paging.pool_bytes <= BUDGET_BYTES / 2);

    // --- Single integrator: the ReSTIR chain carried valid reservoirs ---
    assert!(fused.reservoir_valid_count > 0);
    assert_eq!(fused.frames, 48);

    drop(scene);
    std::fs::remove_dir_all(&work).ok();
}

/// The GPU traversal (top-level BVH -> page table -> per-page BVH ->
/// analytic kernels) must return exactly what a brute-force CPU evaluation of
/// the same model returns over every fixture primitive.
#[test]
fn fused_transmittance_matches_brute_force_cpu_oracle() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let m = &fixture.manifest;
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params.clone()).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let out = tracer
        .render_fused(&fused_desc(&scene, &fixture, SIZE, 1, 2))
        .unwrap();

    // Every primitive, resident on the CPU, no acceleration structure.
    let cloud = load_cloud(&fixture);
    let copc = open_copc(&fixture, &params);
    // The exact records the GPU pool holds: positions plus the per-page
    // surfel normal (or SURFEL_SPHERE) of `DecodedPage::from_payload`.
    let mut points = Vec::new();
    for page in 0..copc.page_count() {
        let payload = copc.load(page).unwrap();
        let decoded =
            DecodedPage::from_payload(page, &payload, params.lidar_radius, params.lidar_surfels)
                .unwrap();
        points.extend(decoded.points.iter().copied());
    }
    assert_eq!(points.len() as u32, m.copc_point_count);
    let surfels = points
        .iter()
        .filter(|p| p.normal_oct != SURFEL_SPHERE)
        .count();

    let sun = toward_sun(m);
    let epsilon = params.transmittance_epsilon;
    let roi = m.roi_half_extent;
    let (mut compared, mut splat_shadowed, mut lidar_shadowed, mut early_outs) = (0, 0, 0, 0);
    let mut worst = 0.0f32;
    for i in (0..out.hit_kind.len()).step_by(3) {
        // Terrain receivers inside the fixture core that the terrain itself
        // does not shadow (an occluded ray stops before visiting pages).
        if out.hit_kind[i] != HIT_TERRAIN
            || out.transmittance[4 * i + 3] < 0.5
            || !in_rect(&out, i, [-roi, -roi, roi, roi])
        {
            continue;
        }
        let n = &out.normal[3 * i..3 * i + 3];
        let p = &out.position[3 * i..3 * i + 3];
        let origin = [p[0] + n[0] * 1e-3, p[1] + n[1] * 1e-3, p[2] + n[2] * 1e-3];
        let mut t_splat = 1.0f32;
        for k in 0..cloud.len() {
            let hit = ray_gaussian(
                origin,
                sun,
                1e-3,
                1e30,
                cloud.positions[k],
                cloud.inv_cov()[k],
                cloud.opacities[k],
            );
            t_splat *= splat_transmittance(hit.response, params.kappa);
        }
        let mut t_lidar = 1.0f32;
        for point in &points {
            let centre = [
                point.pos_radius[0],
                point.pos_radius[1],
                point.pos_radius[2],
            ];
            let coverage = if point.normal_oct == SURFEL_SPHERE {
                sphelet_coverage(
                    origin,
                    sun,
                    1e-3,
                    1e30,
                    centre,
                    params.lidar_radius,
                    params.lidar_opacity,
                )
            } else {
                disc_coverage(
                    origin,
                    sun,
                    1e-3,
                    1e30,
                    centre,
                    oct_decode(point.normal_oct),
                    params.lidar_radius,
                    params.lidar_opacity,
                )
            }
            .coverage;
            t_lidar *= 1.0 - coverage.clamp(0.0, 1.0);
        }
        let (gpu_splat, gpu_lidar) = (out.transmittance[4 * i + 1], out.transmittance[4 * i + 2]);
        compared += 1;
        splat_shadowed += usize::from(t_splat < 0.9);
        lidar_shadowed += usize::from(t_lidar < 0.9);
        if gpu_splat * gpu_lidar < epsilon {
            // The GPU stopped early below epsilon: its factors are partial
            // products, which can only be larger than the full ones.
            early_outs += 1;
            assert!(
                t_splat * t_lidar < epsilon * 1.5,
                "pixel {i}: GPU early-out at {} but the full product is {}",
                gpu_splat * gpu_lidar,
                t_splat * t_lidar
            );
            continue;
        }
        let error = (gpu_splat - t_splat).abs().max((gpu_lidar - t_lidar).abs());
        worst = worst.max(error);
        assert!(
            error < 4e-3,
            "pixel {i} at {p:?}: GPU (T_splat {gpu_splat}, T_lidar {gpu_lidar}) vs CPU \
             ({t_splat}, {t_lidar})"
        );
    }
    eprintln!(
        "CPU oracle: {compared} shadow rays compared, {splat_shadowed} splat-shadowed, \
         {lidar_shadowed} LiDAR-shadowed, {early_outs} early-outs, worst |GPU - CPU| = {worst:.2e}"
    );
    eprintln!(
        "CPU oracle LiDAR records: {surfels} of {} are surfels",
        points.len()
    );
    assert!(compared > 4000, "{compared}");
    assert!(splat_shadowed > 100 && lidar_shadowed > 100);
}

/// Out-of-core policies: a pool smaller than the working set is an error
/// under the exact policy and a progressively refined image under the
/// progressive one, and neither ever exceeds the pool.
#[test]
fn paging_policies_respect_the_residency_pool() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let work = work_dir("paging");
    let tracer = HybridPathTracer::new_fused().unwrap();

    // Reference: everything fits.
    let roomy = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &roomy), roomy.clone()).unwrap();
    let exact = tracer
        .render_fused(&fused_desc(&scene, &fixture, 128, 2, 16))
        .unwrap();
    assert_eq!(exact.paging.deferred_pages, 0);
    assert_eq!(exact.stale_frames, 0);
    assert_eq!(
        exact.paging.residency.resident_pages,
        scene.page_count(),
        "every fixture page is reached by this view"
    );

    // A pool that cannot hold the working set.
    let tight = FusionParams {
        splat_slots: 2,
        point_slots: 1,
        ..roomy.clone()
    };
    let scene = FusedScene::new(fixture_sources(&fixture, &tight), tight.clone()).unwrap();
    let error = tracer
        .render_fused(&fused_desc(&scene, &fixture, 128, 2, 16))
        .map(|_| ())
        .expect_err("the exact policy must refuse a working set larger than the pool");
    let message = format!("{error}");
    assert!(
        message.contains("does not fit the residency pool"),
        "{message}"
    );

    let progressive = FusionParams {
        policy: PagingPolicy::Progressive,
        ..tight
    };
    let scene =
        FusedScene::new(fixture_sources(&fixture, &progressive), progressive.clone()).unwrap();
    let out = tracer
        .render_fused(&fused_desc(&scene, &fixture, 128, 2, 16))
        .unwrap();
    eprintln!("progressive, tight pool: {:?}", out.paging);
    assert_eq!(out.frames, 16);
    assert_eq!(out.restarts, 0);
    assert!(out.stale_frames > 0, "misses must be reported, not hidden");
    assert!(out.paging.deferred_pages > 0);
    assert!(out.paging.residency.peak_resident_pages <= 3);
    assert!(out.radiance.iter().all(|v| v.is_finite()));

    // Progressive paging over the billion-point index with a small point
    // pool: far-field pages rotate through the pool (LRU eviction) while the
    // render keeps going.
    let field_path = billion_point_field(&work, &fixture, &roomy);
    let rotating = FusionParams {
        policy: PagingPolicy::Progressive,
        point_slots: 12,
        ..roomy.clone()
    };
    let mut sources = fixture_sources(&fixture, &rotating);
    sources.push(Arc::new(PageStoreFile::open(&field_path).unwrap()));
    let scene = FusedScene::new(sources, rotating.clone()).unwrap();
    let out = tracer
        .render_fused(&fused_desc(&scene, &fixture, 128, 2, 24))
        .unwrap();
    eprintln!(
        "progressive, billion-point index, 12 point slots: {:?}",
        out.paging
    );
    assert!(out.logical_primitives > 1_000_000_000);
    assert!(out.paging.residency.peak_resident_pages <= rotating.splat_slots + 12);
    assert!(
        out.paging.residency.evictions > 0 || out.paging.deferred_pages > 0,
        "{:?}",
        out.paging
    );
    assert!(out.peak_total_bytes <= BUDGET_BYTES);
    // After the first frame the merged reservoirs carry a last-known
    // visibility for the kernel to fall back on.
    let carried = out
        .reservoir_visibility
        .iter()
        .filter(|v| (0.0..=1.0).contains(*v))
        .count();
    assert!(carried as u64 == out.reservoir_valid_count && carried > 1000);

    drop(scene);
    std::fs::remove_dir_all(&work).ok();
}

/// A fly-through shares one residency pool: the second view evicts the
/// first view's far-field pages least-recently-used, stays exact, and
/// produces the same image as rendering that view on its own.
#[test]
fn sequence_shares_one_residency_pool_and_evicts_lru() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let work = work_dir("sequence");
    // Room for one view's far-field pages, not for both views' union.
    let params = FusionParams {
        point_slots: 64,
        ..fixture_params(&fixture)
    };
    let field_path = billion_point_field(&work, &fixture, &params);
    let mut sources = fixture_sources(&fixture, &params);
    sources.push(Arc::new(PageStoreFile::open(&field_path).unwrap()));
    let scene = FusedScene::new(sources, params.clone()).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();

    let front = fused_desc(&scene, &fixture, 128, 2, 12);
    let mut back = fused_desc(&scene, &fixture, 128, 2, 12);
    // The opposite side of the scene: a different set of far-field tiles.
    back.cam_origin = [
        -fixture.manifest.cam_origin[0],
        fixture.manifest.cam_origin[1],
        -fixture.manifest.cam_origin[2],
    ];
    let outputs = tracer
        .render_fused_sequence(&[front.clone(), back.clone()])
        .unwrap();
    let (first, second) = (&outputs[0], &outputs[1]);
    eprintln!("sequence view 1: {:?}", first.paging);
    eprintln!("sequence view 2: {:?}", second.paging);
    assert_eq!(first.paging.residency.evictions, 0);
    assert!(
        second.paging.residency.evictions > 0,
        "the second view must reclaim slots from the first: {:?}",
        second.paging
    );
    assert!(second.paging.residency.loads > first.paging.residency.loads);
    assert!(second.paging.residency.peak_resident_pages <= params.splat_slots + 64);
    assert_eq!(second.paging.deferred_pages, 0);
    assert!(second.peak_total_bytes <= BUDGET_BYTES);

    // Exact paging is history-free: each view equals its standalone render.
    let alone_front = tracer.render_fused(&front).unwrap();
    let alone_back = tracer.render_fused(&back).unwrap();
    assert!(
        first.rgba == alone_front.rgba,
        "view 1 differs from its standalone render"
    );
    assert!(
        second.rgba == alone_back.rgba,
        "view 2 differs from its standalone render"
    );
    assert!(first.rgba != second.rgba);
    assert_eq!(alone_back.paging.residency.evictions, 0);

    // Views of different scenes cannot share a pool.
    let other = FusedScene::new(fixture_sources(&fixture, &params), params.clone()).unwrap();
    let error = tracer
        .render_fused_sequence(&[
            fused_desc(&scene, &fixture, 32, 1, 2),
            fused_desc(&other, &fixture, 32, 1, 2),
        ])
        .err()
        .unwrap();
    assert!(format!("{error}").contains("share one scene"), "{error}");
    assert!(tracer.render_fused_sequence(&[]).is_err());

    drop(scene);
    std::fs::remove_dir_all(&work).ok();
}

/// Unsupported or missing inputs raise diagnostics; nothing renders a
/// quietly degraded scene.
#[test]
fn missing_or_unsupported_inputs_raise_diagnostics() {
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    // A page source whose pages do not fit a residency slot.
    let small_slots = FusionParams {
        page_capacity: 256,
        ..params.clone()
    };
    let error = FusedScene::new(fixture_sources(&fixture, &small_slots), small_slots)
        .err()
        .unwrap();
    assert!(format!("{error}").contains("residency slot"), "{error}");
    // No slots for a kind that is present.
    let no_points = FusionParams {
        point_slots: 0,
        ..params.clone()
    };
    let error = FusedScene::new(fixture_sources(&fixture, &no_points), no_points)
        .err()
        .unwrap();
    assert!(format!("{error}").contains("no point slots"), "{error}");
    assert!(load_gaussian_splats(fixture_dir().join("scene.json")).is_err());
    assert!(PageStoreFile::open(fixture.splat_path()).is_err());

    if !gpu_available() {
        return;
    }
    // Nothing to trace.
    let empty = FusedScene::new(Vec::new(), params.clone()).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let mut desc = fused_desc(&empty, &fixture, 32, 1, 2);
    desc.terrain = None;
    let error = tracer.render_fused(&desc).err().unwrap();
    assert!(format!("{error}").contains("nothing to trace"), "{error}");
    // The default (non-fused) kernel has no fused bindings.
    let plain = HybridPathTracer::new().unwrap();
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params.clone()).unwrap();
    let error = plain
        .render_fused(&fused_desc(&scene, &fixture, 32, 1, 2))
        .err()
        .unwrap();
    assert!(format!("{error}").contains("new_fused"), "{error}");
    // Each representation renders on its own, and says what it traced.
    let mut splat_only = fused_desc(&scene, &fixture, 64, 1, 4);
    splat_only.terrain = None;
    let out = tracer.render_fused(&splat_only).unwrap();
    let kinds = hit_counts(&out);
    assert_eq!(kinds[3], 0, "no terrain was supplied: {kinds:?}");
    assert!(kinds[4] > 0 && kinds[5] > 0, "{kinds:?}");
    let terrain_only = fused_desc(&empty, &fixture, 64, 1, 4);
    let out = tracer.render_fused(&terrain_only).unwrap();
    let kinds = hit_counts(&out);
    assert!(kinds[3] > 0 && kinds[4] == 0 && kinds[5] == 0, "{kinds:?}");
    assert_eq!(out.paging.residency.loads, 0);
}

/// Mean SSIM over 8x8 windows (stride 4) of the luma of two RGBA8 images.
fn ssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    let luma = |img: &[u8], x: usize, y: usize| {
        let p = &img[4 * (y * width + x)..];
        0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])
    };
    let (c1, c2) = ((0.01f64 * 255.0).powi(2), (0.03f64 * 255.0).powi(2));
    let (mut total, mut windows) = (0.0f64, 0u32);
    for y0 in (0..=height - 8).step_by(4) {
        for x0 in (0..=width - 8).step_by(4) {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for y in y0..y0 + 8 {
                for x in x0..x0 + 8 {
                    let (va, vb) = (luma(a, x, y), luma(b, x, y));
                    sa += va;
                    sb += vb;
                    saa += va * va;
                    sbb += vb * vb;
                    sab += va * vb;
                }
            }
            let n = 64.0;
            let (ma, mb) = (sa / n, sb / n);
            let (va, vb, cov) = (saa / n - ma * ma, sbb / n - mb * mb, sab / n - ma * mb);
            total += ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            windows += 1;
        }
    }
    total / f64::from(windows)
}

/// Deterministic small-scale golden of the fused fixture scene.
#[test]
fn fused_fixture_matches_golden_image() {
    if !gpu_available() {
        return;
    }
    const GOLDEN_SIZE: u32 = 96;
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let desc = fused_desc(&scene, &fixture, GOLDEN_SIZE, 8, 256);
    let out = tracer.render_fused(&desc).unwrap();
    // Same inputs, same adapter -> identical bytes.
    let again = tracer.render_fused(&desc).unwrap();
    assert!(
        out.rgba == again.rgba,
        "the fused render is not deterministic"
    );
    save_rgba("golden_candidate.png", GOLDEN_SIZE, GOLDEN_SIZE, &out.rgba);

    let path = golden_path();
    if std::env::var_os("FORGE3D_UPDATE_SPLAT_FUSION_GOLDENS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        image::save_buffer(
            &path,
            &out.rgba,
            GOLDEN_SIZE,
            GOLDEN_SIZE,
            image::ColorType::Rgba8,
        )
        .unwrap();
        eprintln!("updated {}", path.display());
        return;
    }
    let golden = image::open(&path)
        .unwrap_or_else(|e| {
            panic!(
                "missing fused golden {} ({e}); regenerate with \
                 FORGE3D_UPDATE_SPLAT_FUSION_GOLDENS=1",
                path.display()
            )
        })
        .to_rgba8();
    assert_eq!(golden.dimensions(), (GOLDEN_SIZE, GOLDEN_SIZE));
    let expected = golden.as_raw();
    let mean_abs = out
        .rgba
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .flat_map(|(a, b)| (0..3).map(move |c| (f64::from(a[c]) - f64::from(b[c])).abs()))
        .sum::<f64>()
        / f64::from(GOLDEN_SIZE * GOLDEN_SIZE * 3);
    let score = ssim(
        &out.rgba,
        expected,
        GOLDEN_SIZE as usize,
        GOLDEN_SIZE as usize,
    );
    eprintln!("fused golden drift: SSIM {score:.6}, mean abs {mean_abs:.4}");
    assert!(
        score >= GOLDEN_SSIM_MIN,
        "fused golden drift: SSIM {score:.6}"
    );
    assert!(
        mean_abs <= GOLDEN_MEAN_ABS_MAX,
        "fused golden drift: mean abs {mean_abs:.4}"
    );
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn terrain_f16_minmax_changes_no_output_bit() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let mut desc = fused_desc(&scene, &fixture, 160, 2, 4);
    Arc::make_mut(desc.terrain.as_mut().unwrap()).minmax_precision = MinMaxPrecision::F32;
    let a = tracer.render_fused(&desc).unwrap();
    Arc::make_mut(desc.terrain.as_mut().unwrap()).minmax_precision =
        MinMaxPrecision::F16Conservative;
    let b = tracer.render_fused(&desc).unwrap();
    assert!(!a.terrain_minmax_f16 && b.terrain_minmax_f16);
    assert_eq!(a.rgba, b.rgba);
    assert_eq!(a.hit_kind, b.hit_kind);
    for (name, x, y) in [
        ("radiance", &a.radiance, &b.radiance),
        ("depth", &a.depth, &b.depth),
        ("transmittance", &a.transmittance, &b.transmittance),
        ("normal", &a.normal, &b.normal),
    ] {
        assert_eq!(bits(x), bits(y), "{name} differs");
    }
    println!(
        "fixture terrain scene bytes: with f32 pyramid {} B, with f16 pyramid {} B",
        a.terrain_bytes, b.terrain_bytes
    );
}

#[test]
fn terrain_f16_minmax_halves_the_pyramid_bytes() {
    if !gpu_available() {
        return;
    }
    // Lauterbrunnen-shaped DEM: 1250 x 2500 (pow2-padded to 2048 x 4096 cells).
    let (w, h) = (1250u32, 2500u32);
    let heights: Vec<f32> = (0..w * h)
        .map(|i| 700.0 + ((i % w) as f32 * 0.37).sin() * 900.0 + (i / w) as f32 * 0.5)
        .collect();
    let ctx = forge3d::core::gpu::try_ctx().unwrap();
    let mk = |p| {
        TerrainPtScene::new_with_options(
            &ctx.device,
            &ctx.queue,
            &heights,
            w,
            h,
            (6.0, 6.0),
            1.0,
            [0.4; 3],
            None,
            1.0,
            TerrainAlbedoMap::None,
            AlbedoSampling::Nearest,
            1.0,
            p,
        )
        .unwrap()
    };
    let full = mk(MinMaxPrecision::F32);
    let half = mk(MinMaxPrecision::F16Conservative);
    assert_eq!(
        half.minmax_bytes() * 2,
        full.minmax_bytes(),
        "f16 pyramid must be exactly half"
    );
    assert_eq!(
        full.byte_size() - half.byte_size(),
        half.minmax_bytes(),
        "nothing else may change"
    );
    // pow2-padded 2048 x 4096 level 0 -> 8 B/texel (f32) over the full mip chain.
    assert!(full.minmax_bytes() >= 2048 * 4096 * 8);
    assert!(half.minmax_is_f16() && !full.minmax_is_f16());
    println!(
        "1250x2500 DEM: min-max pyramid f32 {} B, f16 {} B; terrain total f32 {} B, f16 {} B",
        full.minmax_bytes(),
        half.minmax_bytes(),
        full.byte_size(),
        half.byte_size()
    );
}
fn srgb_to_linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(v: f32) -> i32 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as i32
}

#[test]
fn terrain_albedo_map_drives_the_albedo_aov() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let m = &fixture.manifest;
    let (w, h) = (m.dem_width as usize, m.dem_height as usize);
    let stripes = [[200u8, 40, 40], [40, 60, 200]];
    let mut map = vec![0u8; w * h * 4];
    for r in 0..h {
        for c in 0..w {
            let s = stripes[(c / 8) % 2];
            let alpha = if r < h / 2 { 255 } else { 0 }; // southern half: masked -> uniform albedo
            map[(r * w + c) * 4..(r * w + c) * 4 + 4].copy_from_slice(&[s[0], s[1], s[2], alpha]);
        }
    }
    let mut desc = fused_desc(&scene, &fixture, 192, 1, 1);
    let t = Arc::make_mut(desc.terrain.as_mut().unwrap());
    t.albedo_map = Some(map);
    t.albedo_sampling = AlbedoSampling::Nearest;
    let out = HybridPathTracer::new_fused()
        .unwrap()
        .render_fused(&desc)
        .unwrap();
    let (sx, sz) = (m.dem_spacing[0], m.dem_spacing[1]);
    let (ox, oz) = (
        -0.5 * (m.dem_width - 1) as f32 * sx,
        -0.5 * (m.dem_height - 1) as f32 * sz,
    );
    let (mut mapped, mut masked) = (0, 0);
    for i in 0..out.hit_kind.len() {
        if out.hit_kind[i] != HIT_TERRAIN {
            continue;
        }
        let col = (out.position[3 * i] - ox) / sx;
        let row = (out.position[3 * i + 2] - oz) / sz;
        let (cn, rn) = (col.round() as usize, row.round() as usize);
        let in_stripe = (col.round() - col).abs() < 0.4 && (1..7).contains(&(cn % 8));
        if !in_stripe || rn == h / 2 || rn + 1 == h / 2 {
            continue;
        }
        let rgb = [
            out.albedo[3 * i],
            out.albedo[3 * i + 1],
            out.albedo[3 * i + 2],
        ];
        let expect: [u8; 3] = if rn < h / 2 {
            mapped += 1;
            stripes[(cn / 8) % 2]
        } else {
            masked += 1;
            m.terrain_albedo.map(|v| linear_to_srgb_u8(v) as u8)
        };
        for ch in 0..3 {
            assert!(
                (linear_to_srgb_u8(rgb[ch]) - i32::from(expect[ch])).abs() <= 1,
                "pixel {i} ch {ch}: albedo {} (sRGB {}) vs expected sRGB {}",
                rgb[ch],
                linear_to_srgb_u8(rgb[ch]),
                expect[ch]
            );
        }
    }
    println!(
        "albedo map: {mapped} mapped and {masked} masked terrain pixels within +-1 sRGB code; \
         sRGB 128 decodes to {}",
        srgb_to_linear(128)
    );
    assert!(
        mapped >= 500 && masked >= 300,
        "mapped {mapped} masked {masked}"
    );
}
/// DEM cell `(cx, cz)` and in-cell `(u, v)` of a world-space terrain point.
fn terrain_cell_of(m: &FixtureManifest, x: f32, z: f32) -> (usize, usize, f32, f32) {
    let (sx, sz) = (m.dem_spacing[0], m.dem_spacing[1]);
    let ox = -0.5 * (m.dem_width - 1) as f32 * sx;
    let oz = -0.5 * (m.dem_height - 1) as f32 * sz;
    let fx = ((x - ox) / sx).clamp(0.0, (m.dem_width - 1) as f32);
    let fz = ((z - oz) / sz).clamp(0.0, (m.dem_height - 1) as f32);
    let cx = (fx.floor() as usize).min(m.dem_width as usize - 2);
    let cz = (fz.floor() as usize).min(m.dem_height as usize - 2);
    (cx, cz, fx - cx as f32, fz - cz as f32)
}

#[test]
fn gpu_smooth_terrain_normal_matches_cpu_mirror() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let desc = fused_desc(&scene, &fixture, 160, 1, 1);
    let out = HybridPathTracer::new_fused()
        .unwrap()
        .render_fused(&desc)
        .unwrap();
    let m = &fixture.manifest;
    let (w, h) = (m.dem_width as usize, m.dem_height as usize);
    let (mut checked, mut worst) = (0usize, 0.0f32);
    for i in 0..out.hit_kind.len() {
        if out.hit_kind[i] != HIT_TERRAIN {
            continue;
        }
        let (cx, cz, u, v) = terrain_cell_of(m, out.position[3 * i], out.position[3 * i + 2]);
        let cpu = terrain_smooth_normal(
            &fixture.heights,
            w,
            h,
            (m.dem_spacing[0], m.dem_spacing[1]),
            1.0,
            u,
            v,
            cx,
            cz,
        );
        let gpu = [
            out.normal[3 * i],
            out.normal[3 * i + 1],
            out.normal[3 * i + 2],
        ];
        let cos = (cpu[0] * gpu[0] + cpu[1] * gpu[1] + cpu[2] * gpu[2]).clamp(-1.0, 1.0);
        let err = cos.acos();
        worst = worst.max(err);
        assert!(
            err <= 1e-3,
            "pixel {i}: {err} rad (gpu {gpu:?}, cpu {cpu:?})"
        );
        checked += 1;
    }
    println!("smooth terrain normal: {checked} terrain pixels, worst GPU-CPU angle {worst:e} rad");
    assert!(checked >= 2000, "only {checked} terrain pixels");
}

#[test]
fn smooth_terrain_normals_add_no_self_shadow_acne() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let render = |smooth: bool| {
        let params = FusionParams {
            terrain_smooth_normals: smooth,
            ..fixture_params(&fixture)
        };
        let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
        tracer
            .render_fused(&fused_desc(&scene, &fixture, 160, 1, 1))
            .unwrap()
    };
    let off = render(false);
    let on = render(true);
    let (mut terrain, mut newly) = (0usize, 0usize);
    for i in 0..off.hit_kind.len() {
        if off.hit_kind[i] != HIT_TERRAIN || on.hit_kind[i] != HIT_TERRAIN {
            continue;
        }
        terrain += 1;
        if off.transmittance[4 * i + 3] == 1.0 && on.transmittance[4 * i + 3] < 1.0 {
            newly += 1;
        }
    }
    let share = newly as f64 / terrain as f64;
    println!(
        "smooth normals: {newly} of {terrain} terrain pixels newly terrain-shadowed ({:.4}%)",
        100.0 * share
    );
    assert!(terrain >= 2000);
    assert!(share <= 0.001, "{newly} of {terrain} newly shadowed");
}
/// A point page store at `path` holding `points` (pages of 4096).
fn point_store(path: &Path, points: &[[f32; 3]]) -> PageStoreFile {
    let mut writer = PageStoreWriter::create(path, PageKind::Points, 4096, 0).unwrap();
    for chunk in points.chunks(4096) {
        let rgba = vec![[120u8, 120, 120, 2]; chunk.len()];
        writer.push_point_page(chunk, &rgba).unwrap();
    }
    writer.finish().unwrap();
    PageStoreFile::open(path).unwrap()
}

/// A terrain-free fused render of a point store.
fn points_render(
    store: PageStoreFile,
    surfels: bool,
    lidar_self_bias_radii: f32,
    cam_origin: [f32; 3],
    cam_look_at: [f32; 3],
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
) -> FusedRenderOutput {
    let params = FusionParams {
        lidar_radius: 0.85,
        lidar_opacity: 1.0,
        lidar_surfels: surfels,
        lidar_self_bias_radii,
        policy: PagingPolicy::Exact,
        ..FusionParams::default()
    };
    let sources: Vec<Arc<dyn PageSource>> = vec![Arc::new(store)];
    let scene = FusedScene::new(sources, params).unwrap();
    HybridPathTracer::new_fused()
        .unwrap()
        .render_fused(&FusedRenderDesc {
            scene: &scene,
            terrain: None,
            cam_origin,
            cam_look_at,
            cam_up: [0.0, 1.0, 0.0],
            fov_y_deg: 50.0,
            exposure: 1.0,
            sun_azimuth_deg,
            sun_elevation_deg,
            sun_intensity: 2.5,
            sun_color: [1.0, 0.97, 0.92],
            sky_turbidity: 2.5,
            sky_ground_albedo: 0.2,
            sky_intensity: 0.35,
            width: 192,
            height: 192,
            seed: 7,
            spp: 1,
            frames: 1,
            tile: None,
            aovs: true,
        })
        .unwrap()
}

fn dense_plane() -> Vec<[f32; 3]> {
    let mut points = Vec::new();
    for i in 0..120i32 {
        for k in 0..120i32 {
            points.push([(i - 60) as f32, 0.0, (k - 60) as f32]);
        }
    }
    points
}

#[test]
fn dense_lidar_plane_does_not_shadow_itself() {
    if !gpu_available() {
        return;
    }
    let dir = work_dir("dense-plane");
    let plane = dense_plane();
    let mean_t_lidar = |surfels: bool, elevation: f32| {
        let path = dir.join(format!("plane-{surfels}-{elevation}.f3dpages"));
        // No self-shadow bias: on an exactly flat grid the default one-radius
        // skip already lifts sphelet shadow rays over their neighbours' tops
        // (measured T_lidar 1.0 for spheres at 10 and 22 deg), which would
        // hide the difference this test is about. Surfels need no bias.
        let out = points_render(
            point_store(&path, &plane),
            surfels,
            0.0,
            [0.0, 60.0, -70.0],
            [0.0, 0.0, 0.0],
            45.0,
            elevation,
        );
        let (mut sum, mut n) = (0.0f64, 0usize);
        for i in 0..out.hit_kind.len() {
            let (x, z) = (out.position[3 * i], out.position[3 * i + 2]);
            if out.hit_kind[i] == HIT_LIDAR && x.abs() <= 50.0 && z.abs() <= 50.0 {
                sum += f64::from(out.transmittance[4 * i + 2]);
                n += 1;
            }
        }
        assert!(n >= 2000, "only {n} LiDAR pixels");
        (sum / n as f64, n)
    };
    for elevation in [5.0f32, 10.0, 22.0, 40.0] {
        let (t, n) = mean_t_lidar(true, elevation);
        println!("dense plane, surfels, sun {elevation} deg: mean T_lidar {t:.4} over {n} px");
        assert!(t >= 0.98, "surfels at {elevation} deg: mean T_lidar {t}");
    }
    // Sensitivity: the same plane as sphelets shadows itself at grazing sun
    // (unbiased: 0.97 at 22 deg, 0.90 at 10 deg, 0.78 at 5 deg).
    let (t, n) = mean_t_lidar(false, 5.0);
    println!("dense plane, spheres, sun 5 deg: mean T_lidar {t:.4} over {n} px");
    assert!(
        t <= 0.90,
        "sphelets at 5 deg should self-shadow: mean T_lidar {t}"
    );
}

#[test]
fn lidar_wall_still_casts_its_shadow() {
    if !gpu_available() {
        return;
    }
    let dir = work_dir("lidar-wall");
    let mut points = dense_plane();
    for iy in 0..=16i32 {
        for iz in -40..=40i32 {
            points.push([0.0, iy as f32 * 0.5, iz as f32 * 0.5]);
        }
    }
    let out = points_render(
        point_store(&dir.join("wall.f3dpages"), &points),
        true,
        FusionParams::default().lidar_self_bias_radii,
        [-7.0, 60.0, -50.0],
        [-7.0, 0.0, 0.0],
        0.0,
        30.0,
    );
    let (mut total, mut dark) = (0usize, 0usize);
    for i in 0..out.hit_kind.len() {
        let (x, y, z) = (
            out.position[3 * i],
            out.position[3 * i + 1],
            out.position[3 * i + 2],
        );
        if out.hit_kind[i] == HIT_LIDAR
            && (-12.0..=-2.0).contains(&x)
            && z.abs() <= 15.0
            && y.abs() < 0.1
        {
            total += 1;
            if out.transmittance[4 * i + 2] <= 0.05 {
                dark += 1;
            }
        }
    }
    println!("LiDAR wall shadow: {dark} of {total} ground pixels have T_lidar <= 0.05");
    assert!(total >= 200, "only {total} shadow pixels");
    assert!(dark * 100 >= total * 95, "{dark} of {total}");
}
#[test]
fn tiled_fused_render_equals_the_monolithic_render_bit_for_bit() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let mut desc = fused_desc(&scene, &fixture, 0, 2, 4);
    desc.width = 300;
    desc.height = 200;
    let whole = tracer.render_fused(&desc).unwrap();
    for tile in [(150, 100), (128, 96)] {
        desc.tile = Some(tile);
        let tiled = tracer.render_fused(&desc).unwrap();
        assert_eq!(whole.rgba, tiled.rgba, "{tile:?} rgba");
        assert_eq!(whole.hit_kind, tiled.hit_kind, "{tile:?} hit_kind");
        for (name, a, b) in [
            ("radiance", &whole.radiance, &tiled.radiance),
            ("albedo", &whole.albedo, &tiled.albedo),
            ("normal", &whole.normal, &tiled.normal),
            ("position", &whole.position, &tiled.position),
            ("direct", &whole.direct, &tiled.direct),
            ("transmittance", &whole.transmittance, &tiled.transmittance),
            ("depth", &whole.depth, &tiled.depth),
            ("sun_cosine", &whole.sun_cosine, &tiled.sun_cosine),
            ("self_bias", &whole.self_bias, &tiled.self_bias),
            (
                "reservoir_visibility",
                &whole.reservoir_visibility,
                &tiled.reservoir_visibility,
            ),
        ] {
            assert_eq!(bits(a), bits(b), "{tile:?} {name}");
        }
        assert_eq!(whole.variance.to_bits(), tiled.variance.to_bits());
        assert_eq!(whole.reservoir_valid_count, tiled.reservoir_valid_count);
        println!(
            "tile {tile:?}: {} tiles, bit-identical to the monolithic 300x200 render",
            tiled.tiles
        );
    }
}

#[test]
fn fused_peak_memory_does_not_grow_with_output_resolution() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let run = |w: u32, h: u32, tile: Option<(u32, u32)>, aovs: bool| {
        let mut d = fused_desc(&scene, &fixture, 0, 1, 1);
        d.width = w;
        d.height = h;
        d.tile = tile;
        d.aovs = aovs;
        let out = tracer.render_fused(&d).unwrap();
        assert_eq!(out.rgba.len(), (w * h * 4) as usize);
        println!(
            "{w}x{h} tile {tile:?} aovs {aovs}: {} tiles, render peak {} B ({:.1} MiB)",
            out.tiles,
            out.peak_total_bytes,
            out.peak_total_bytes as f64 / 1048576.0
        );
        out.peak_total_bytes
    };
    const MIB: u64 = 1 << 20;
    let base = run(960, 540, None, true);
    assert!(run(1920, 1080, Some((960, 540)), true) <= base + MIB);
    let base_lean = run(960, 540, None, false);
    assert!(run(3840, 2160, Some((960, 540)), false) <= base_lean + MIB);
    // The output layout still binds seven 1x1 placeholder AOV targets
    // (48 bytes), so the saving is 48 B for every pixel but one.
    assert!(
        base_lean + 48 * (960 * 540 - 1) <= base,
        "AOV textures must be freed when aovs = false: saved {} B",
        base - base_lean
    );
    assert!(base <= BUDGET_BYTES);
}

#[test]
fn aov_opt_out_leaves_the_beauty_untouched() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let mut d = fused_desc(&scene, &fixture, 192, 2, 4);
    let with = tracer.render_fused(&d).unwrap();
    d.aovs = false;
    let without = tracer.render_fused(&d).unwrap();
    assert_eq!(with.rgba, without.rgba);
    assert!(without.hit_kind.is_empty() && without.albedo.is_empty());
}

/// A sequence uploads the terrain once and swaps only each view's sky: a sun
/// sweep sharing one `Arc` terrain must equal the standalone renders bit for
/// bit, and a view whose terrain differs in any field is rejected.
#[test]
fn sequence_shares_one_terrain_and_swaps_the_sky_per_view() {
    if !gpu_available() {
        return;
    }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let base = fused_desc(&scene, &fixture, 96, 2, 4);
    let views: Vec<_> = [
        (150.0f32, 12.0f32, 0.2f32),
        (172.0, 36.0, 0.35),
        (200.0, 60.0, 0.5),
    ]
    .iter()
    .map(|&(az, el, sky)| {
        let mut d = base.clone();
        d.sun_azimuth_deg = az;
        d.sun_elevation_deg = el;
        d.sky_intensity = sky;
        d
    })
    .collect();
    // Every view shares the one terrain allocation.
    for view in &views {
        assert!(Arc::ptr_eq(
            view.terrain.as_ref().unwrap(),
            base.terrain.as_ref().unwrap()
        ));
    }
    let sequence = tracer.render_fused_sequence(&views).unwrap();
    for (i, (view, got)) in views.iter().zip(&sequence).enumerate() {
        let alone = tracer.render_fused(view).unwrap();
        assert!(
            got.rgba == alone.rgba,
            "view {i}: rgba differs from its standalone render"
        );
        assert_eq!(
            bits(&got.radiance),
            bits(&alone.radiance),
            "view {i} radiance"
        );
        assert_eq!(got.terrain_bytes, alone.terrain_bytes);
    }
    assert!(
        sequence[0].rgba != sequence[2].rgba,
        "the sun sweep must change the image"
    );

    // The terrain scene is built from the first view: a different colour map
    // (or any other terrain field) cannot join the sequence.
    let mut mapped = base.clone();
    let m = &fixture.manifest;
    Arc::make_mut(mapped.terrain.as_mut().unwrap()).albedo_map =
        Some(vec![255u8; (m.dem_width * m.dem_height * 4) as usize]);
    let error = tracer
        .render_fused_sequence(&[base.clone(), mapped])
        .err()
        .unwrap();
    assert!(format!("{error}").contains("one terrain"), "{error}");
}
