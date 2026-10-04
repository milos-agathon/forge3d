// src/splat/fixture.rs
// Deterministic three-representation demo scene for the fused integrator:
// a mini DEM with a shadow-casting ridge, a Gaussian splat cloud (one blob
// floating above the terrain, one mound in the ridge's shadow), and a COPC
// LiDAR swath (a canopy disc above the terrain, a ground patch in the
// ridge's shadow). The generator writes the files under
// tests/fixtures/splat_fusion/ and the loader reads them back for the Rust
// and Python acceptance tests, so both languages consume identical bytes.
// RELEVANT FILES: tests/test_splat_fusion_occlusion.rs, tests/test_splat_api.py,
//                 src/splat/fusion.rs

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::num::f32_from_u32;
use super::stream::hash_unit;
use super::{load::save_gaussian_splats, GaussianSplatCloud, ShRest};
use crate::core::error::RenderError;

/// Camera, sun and file layout of the fixture (`scene.json`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FixtureManifest {
    pub dem_file: String,
    pub dem_width: u32,
    pub dem_height: u32,
    pub dem_spacing: [f32; 2],
    pub terrain_albedo: [f32; 3],
    pub splat_file: String,
    pub splat_count: u32,
    pub copc_file: String,
    pub copc_point_count: u32,
    /// LAS coordinate that maps to the scene origin (z-up, metres).
    pub copc_origin: [f64; 3],
    pub lidar_radius: f32,
    pub cam_origin: [f32; 3],
    pub cam_look_at: [f32; 3],
    pub cam_up: [f32; 3],
    pub fov_y_deg: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    /// World-space half extent of the region the acceptance masks cover.
    pub roi_half_extent: f32,
}

/// The fixture as loaded from disk.
#[derive(Clone, Debug)]
pub struct FixtureScene {
    pub dir: PathBuf,
    pub manifest: FixtureManifest,
    pub heights: Vec<f32>,
}

impl FixtureScene {
    /// Load `scene.json` and the DEM from a fixture directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, RenderError> {
        let dir = dir.as_ref().to_path_buf();
        let describe = |what: &str, e: &dyn std::fmt::Display| {
            RenderError::Upload(format!("fusion fixture {}: {what}: {e}", dir.display()))
        };
        let text = std::fs::read_to_string(dir.join("scene.json"))
            .map_err(|e| describe("scene.json", &e))?;
        let manifest: FixtureManifest =
            serde_json::from_str(&text).map_err(|e| describe("scene.json", &e))?;
        let raw = std::fs::read(dir.join(&manifest.dem_file))
            .map_err(|e| describe(&manifest.dem_file, &e))?;
        let expected = manifest.dem_width as usize * manifest.dem_height as usize * 4;
        if raw.len() != expected {
            return Err(describe(
                &manifest.dem_file,
                &format!("{} bytes, expected {expected}", raw.len()),
            ));
        }
        let heights = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(Self {
            dir,
            manifest,
            heights,
        })
    }

    pub fn splat_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.splat_file)
    }

    pub fn copc_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.copc_file)
    }
}

const DEM_SIZE: u32 = 65;
const DEM_SPACING: f32 = 1.0;
const RIDGE_X: f32 = -20.0;
const RIDGE_HEIGHT: f32 = 9.0;
const RIDGE_WIDTH: f32 = 3.5;
const LIDAR_RADIUS: f32 = 0.22;
pub const COPC_ORIGIN: [f64; 3] = [500_000.0, 4_000_000.0, 1_200.0];
pub const COPC_HALFSIZE: f64 = 40.0;

/// Terrain height at world (x, z): gentle undulation plus a ridge running
/// along z that shadows everything immediately down-sun of it.
pub fn terrain_height(x: f32, z: f32) -> f32 {
    let ridge = {
        let u = (x - RIDGE_X) / RIDGE_WIDTH;
        RIDGE_HEIGHT * (-u * u).exp()
    };
    // Smooth, trig-free undulation (products of bounded polynomials).
    let a = x / 32.0;
    let b = z / 32.0;
    let bumps = 0.6 * (a * a - 0.5) * (1.0 - b * b) + 0.4 * a * b;
    ridge + bumps + 1.0
}

fn dem() -> Vec<f32> {
    let half = 0.5 * f32_from_u32(DEM_SIZE - 1) * DEM_SPACING;
    let mut heights = Vec::with_capacity((DEM_SIZE * DEM_SIZE) as usize);
    for row in 0..DEM_SIZE {
        for col in 0..DEM_SIZE {
            let x = f32_from_u32(col) * DEM_SPACING - half;
            let z = f32_from_u32(row) * DEM_SPACING - half;
            heights.push(terrain_height(x, z));
        }
    }
    heights
}

/// Deterministic point inside the unit ball (rejection-free: cube root radius
/// times a hash-derived direction built without trigonometry).
fn unit_ball(seed: u32) -> [f32; 3] {
    let mut salt = 0u32;
    loop {
        let p = [
            2.0 * hash_unit(seed ^ salt.wrapping_mul(0x9e37_79b9)) - 1.0,
            2.0 * hash_unit(seed ^ salt.wrapping_mul(0x85eb_ca6b) ^ 0x1234_5678) - 1.0,
            2.0 * hash_unit(seed ^ salt.wrapping_mul(0xc2b2_ae35) ^ 0x9abc_def0) - 1.0,
        ];
        if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= 1.0 {
            return p;
        }
        salt += 1;
    }
}

struct SplatParts {
    positions: Vec<[f32; 3]>,
    scales: Vec<[f32; 3]>,
    rotations: Vec<[f32; 4]>,
    opacities: Vec<f32>,
    sh0: Vec<[f32; 3]>,
    sh1: Vec<[f32; 3]>,
}

/// Append a blob of splats. A `carpet` blob hugs the terrain as a thin,
/// nearly flat layer (so it receives shadows without shadowing itself);
/// otherwise the splats fill an ellipsoid around `center`.
fn push_blob(
    parts: &mut SplatParts,
    count: u32,
    seed: u32,
    center: [f32; 3],
    radii: [f32; 3],
    carpet: bool,
    rgb: [f32; 3],
) {
    for i in 0..count {
        let s = seed.wrapping_add(i.wrapping_mul(0x2545_f491));
        let p = unit_ball(s);
        let (x, z) = (center[0] + p[0] * radii[0], center[2] + p[2] * radii[2]);
        let y = if carpet {
            terrain_height(x, z) + center[1] + radii[1] * p[1].abs()
        } else {
            center[1] + p[1] * radii[1]
        };
        let position = [x, y, z];
        let sigma = |k: u32| 0.18 + 0.2 * hash_unit(s ^ k);
        let tilt = if carpet { 0.15 } else { 1.0 };
        let q = [
            1.0,
            tilt * (hash_unit(s ^ 0x51) - 0.5),
            tilt * (hash_unit(s ^ 0x52) - 0.5),
            tilt * (hash_unit(s ^ 0x53) - 0.5),
        ];
        parts.positions.push(position);
        let flatten = if carpet { 0.35 } else { 0.6 };
        parts
            .scales
            .push([sigma(0x41), flatten * sigma(0x42), sigma(0x43)]);
        parts.rotations.push(q);
        parts.opacities.push(0.65 + 0.3 * hash_unit(s ^ 0x61));
        // DC coefficient such that 0.5 + C0 * dc == rgb (plus mild variation).
        let tint = 0.9 + 0.2 * hash_unit(s ^ 0x71);
        parts
            .sh0
            .push(rgb.map(|c| (c * tint - 0.5) / super::SH_C0));
        for k in 0..3u32 {
            let v = 0.06 * (hash_unit(s ^ (0x81 + k)) - 0.5);
            parts.sh1.push([v, v, v]);
        }
    }
}

/// The fixture's splat cloud: an occluder blob above the open terrain and a
/// receiver mound inside the ridge's shadow band.
pub fn demo_splat_cloud() -> Result<GaussianSplatCloud, RenderError> {
    let mut parts = SplatParts {
        positions: Vec::new(),
        scales: Vec::new(),
        rotations: Vec::new(),
        opacities: Vec::new(),
        sh0: Vec::new(),
        sh1: Vec::new(),
    };
    push_blob(
        &mut parts,
        2600,
        0x0a11_ce01,
        [8.0, 8.5, -8.0],
        [4.0, 2.5, 3.5],
        false,
        [0.80, 0.34, 0.26],
    );
    // Receiver carpet 0.3..0.6 m above the terrain inside the ridge's
    // shadow band.
    push_blob(
        &mut parts,
        900,
        0x0b0b_0b02,
        [-10.0, 0.3, -6.0],
        [5.0, 0.3, 4.0],
        true,
        [0.28, 0.42, 0.80],
    );
    GaussianSplatCloud::from_parts(
        parts.positions,
        parts.scales,
        parts.rotations,
        parts.opacities,
        parts.sh0,
        Some(ShRest {
            degree: 1,
            coeffs: parts.sh1,
        }),
    )
}

/// One LiDAR return in LAS coordinates (z-up) with colour and class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LasPoint {
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub classification: u8,
}

fn las_from_scene(p: [f32; 3]) -> [f64; 3] {
    [
        COPC_ORIGIN[0] + f64::from(p[0]),
        COPC_ORIGIN[1] - f64::from(p[2]),
        COPC_ORIGIN[2] + f64::from(p[1]),
    ]
}

/// The fixture's LiDAR swath: a canopy disc that shadows the open terrain and
/// a ground patch that the ridge shadows.
pub fn demo_lidar_points() -> Vec<LasPoint> {
    let mut points = Vec::new();
    // Canopy: thick disc floating above the terrain.
    for i in 0..3000u32 {
        let s = 0x0c0f_fee3u32.wrapping_add(i.wrapping_mul(0x2545_f491));
        let p = unit_ball(s);
        let scene = [6.0 + 3.6 * p[0], 7.0 + 0.6 * p[1], 13.0 + 3.6 * p[2]];
        let g = 110 + (hash_unit(s ^ 0x91) * 70.0) as u8;
        points.push(LasPoint {
            xyz: las_from_scene(scene),
            rgb: [g / 3, g, g / 4],
            classification: 5,
        });
    }
    // Ground patch hugging the terrain inside the ridge's shadow band.
    for i in 0..2500u32 {
        let s = 0x0d15_c004u32.wrapping_add(i.wrapping_mul(0x2545_f491));
        let p = unit_ball(s);
        let (x, z) = (-10.0 + 5.0 * p[0], 9.0 + 4.0 * p[2]);
        // A single thin layer, like ground returns over a real surface.
        let scene = [x, terrain_height(x, z) + 0.35 + 0.05 * p[1].abs(), z];
        let t = 150 + (hash_unit(s ^ 0x92) * 60.0) as u8;
        points.push(LasPoint {
            xyz: las_from_scene(scene),
            rgb: [t, t - 30, t / 2],
            classification: 2,
        });
    }
    points
}

/// Write an uncompressed COPC file (LAS 1.4, point format 7): the returns are
/// distributed over a depth-2 octree with a sparse root subsample, the layout
/// the repo's `CopcDataset` reader consumes.
pub fn write_copc(path: &Path, points: &[LasPoint]) -> Result<(), RenderError> {
    const HEADER: usize = 375;
    const VLR_HEADER: usize = 54;
    const COPC_INFO: usize = 160;
    const RECORD: usize = 36;
    const SCALE: f64 = 0.001;
    let center = COPC_ORIGIN;

    // Assign points to octree nodes: every 16th return to the root, the rest
    // to the depth-2 cell that contains them.
    let mut nodes: std::collections::BTreeMap<(u32, u32, u32, u32), Vec<&LasPoint>> =
        std::collections::BTreeMap::new();
    for (index, point) in points.iter().enumerate() {
        let key = if index % 16 == 0 {
            (0, 0, 0, 0)
        } else {
            let cell = |axis: usize| {
                let unit = (point.xyz[axis] - (center[axis] - COPC_HALFSIZE))
                    / (2.0 * COPC_HALFSIZE);
                ((unit * 4.0).floor().clamp(0.0, 3.0)) as u32
            };
            (2, cell(0), cell(1), cell(2))
        };
        nodes.entry(key).or_default().push(point);
    }

    let point_offset = HEADER + VLR_HEADER + COPC_INFO;
    let mut chunks = Vec::new();
    let mut hierarchy = Vec::new();
    let mut cursor = point_offset as u64;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for (key, members) in &nodes {
        let mut chunk = Vec::with_capacity(members.len() * RECORD);
        for point in members {
            let mut record = [0u8; RECORD];
            for axis in 0..3 {
                lo[axis] = lo[axis].min(point.xyz[axis]);
                hi[axis] = hi[axis].max(point.xyz[axis]);
                let quantised = ((point.xyz[axis] - center[axis]) / SCALE).round() as i32;
                record[axis * 4..axis * 4 + 4].copy_from_slice(&quantised.to_le_bytes());
            }
            record[12..14].copy_from_slice(&1000u16.to_le_bytes()); // intensity
            record[14] = 0x11; // return 1 of 1
            record[16] = point.classification;
            for (channel, value) in point.rgb.iter().enumerate() {
                let wide = u16::from(*value) * 257;
                record[30 + channel * 2..32 + channel * 2].copy_from_slice(&wide.to_le_bytes());
            }
            chunk.extend_from_slice(&record);
        }
        let mut entry = [0u8; 32];
        for (slot, value) in [key.0, key.1, key.2, key.3].into_iter().enumerate() {
            entry[slot * 4..slot * 4 + 4].copy_from_slice(&(value as i32).to_le_bytes());
        }
        entry[16..24].copy_from_slice(&cursor.to_le_bytes());
        entry[24..28].copy_from_slice(&(chunk.len() as i32).to_le_bytes());
        entry[28..32].copy_from_slice(&(members.len() as i32).to_le_bytes());
        hierarchy.extend_from_slice(&entry);
        cursor += chunk.len() as u64;
        chunks.push(chunk);
    }
    let evlr_offset = cursor;
    let hierarchy_offset = evlr_offset + 60;

    let mut header = [0u8; HEADER];
    header[0..4].copy_from_slice(b"LASF");
    header[24] = 1;
    header[25] = 4;
    header[26..33].copy_from_slice(b"forge3d");
    header[58..73].copy_from_slice(b"forge3d fixture");
    header[94..96].copy_from_slice(&(HEADER as u16).to_le_bytes());
    header[96..100].copy_from_slice(&(point_offset as u32).to_le_bytes());
    header[100..104].copy_from_slice(&1u32.to_le_bytes());
    header[104] = 7;
    header[105..107].copy_from_slice(&(RECORD as u16).to_le_bytes());
    for axis in 0..3 {
        header[131 + axis * 8..139 + axis * 8].copy_from_slice(&SCALE.to_le_bytes());
        header[155 + axis * 8..163 + axis * 8].copy_from_slice(&center[axis].to_le_bytes());
        // LAS 1.4: max x, min x, max y, min y, max z, min z.
        header[179 + axis * 16..187 + axis * 16].copy_from_slice(&hi[axis].to_le_bytes());
        header[187 + axis * 16..195 + axis * 16].copy_from_slice(&lo[axis].to_le_bytes());
    }
    header[235..243].copy_from_slice(&evlr_offset.to_le_bytes());
    header[243..247].copy_from_slice(&1u32.to_le_bytes());
    header[247..255].copy_from_slice(&(points.len() as u64).to_le_bytes());
    header[255..263].copy_from_slice(&(points.len() as u64).to_le_bytes());

    let mut vlr = [0u8; VLR_HEADER];
    vlr[2..6].copy_from_slice(b"copc");
    vlr[18..20].copy_from_slice(&1u16.to_le_bytes());
    vlr[20..22].copy_from_slice(&(COPC_INFO as u16).to_le_bytes());
    let mut info = [0u8; COPC_INFO];
    for axis in 0..3 {
        info[axis * 8..axis * 8 + 8].copy_from_slice(&center[axis].to_le_bytes());
    }
    info[24..32].copy_from_slice(&COPC_HALFSIZE.to_le_bytes());
    info[32..40].copy_from_slice(&(2.0 * COPC_HALFSIZE / 128.0).to_le_bytes());
    info[40..48].copy_from_slice(&hierarchy_offset.to_le_bytes());
    info[48..56].copy_from_slice(&(hierarchy.len() as u64).to_le_bytes());

    let mut evlr = [0u8; 60];
    evlr[2..6].copy_from_slice(b"copc");
    evlr[18..20].copy_from_slice(&1000u16.to_le_bytes());
    evlr[20..28].copy_from_slice(&(hierarchy.len() as u64).to_le_bytes());

    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(&header)?;
    out.write_all(&vlr)?;
    out.write_all(&info)?;
    for chunk in &chunks {
        out.write_all(chunk)?;
    }
    out.write_all(&evlr)?;
    out.write_all(&hierarchy)?;
    out.flush()?;
    Ok(())
}

/// Manifest of the generated fixture.
pub fn demo_manifest(splat_count: u32, copc_point_count: u32) -> FixtureManifest {
    FixtureManifest {
        dem_file: "dem_65x65.f32".into(),
        dem_width: DEM_SIZE,
        dem_height: DEM_SIZE,
        dem_spacing: [DEM_SPACING, DEM_SPACING],
        terrain_albedo: [0.55, 0.52, 0.47],
        splat_file: "cloud.ply".into(),
        splat_count,
        copc_file: "swath.copc.laz".into(),
        copc_point_count,
        copc_origin: COPC_ORIGIN,
        lidar_radius: LIDAR_RADIUS,
        cam_origin: [6.0, 52.0, 46.0],
        cam_look_at: [0.0, 1.0, 1.0],
        cam_up: [0.0, 1.0, 0.0],
        fov_y_deg: 52.0,
        sun_azimuth_deg: 172.0,
        sun_elevation_deg: 36.0,
        roi_half_extent: 27.0,
    }
}

/// Generate every fixture file into `dir`.
pub fn write_demo_fixture(dir: impl AsRef<Path>) -> Result<FixtureManifest, RenderError> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let cloud = demo_splat_cloud()?;
    let points = demo_lidar_points();
    let manifest = demo_manifest(cloud.len() as u32, points.len() as u32);
    let heights = dem();
    let mut raw = Vec::with_capacity(heights.len() * 4);
    for h in &heights {
        raw.extend_from_slice(&h.to_le_bytes());
    }
    std::fs::write(dir.join(&manifest.dem_file), raw)?;
    save_gaussian_splats(dir.join(&manifest.splat_file), &cloud)?;
    write_copc(&dir.join(&manifest.copc_file), &points)?;
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| RenderError::Upload(format!("fixture manifest: {e}")))?;
    std::fs::write(dir.join("scene.json"), json + "\n")?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::super::stream::{CopcPageSource, PagePayload, PageSource, PointCloudFrame};
    use super::*;

    #[test]
    fn fixture_round_trips_through_the_real_readers() {
        let dir = std::env::temp_dir().join(format!("forge3d-fusion-fixture-{}", std::process::id()));
        let manifest = write_demo_fixture(&dir).unwrap();
        let scene = FixtureScene::load(&dir).unwrap();
        assert_eq!(scene.manifest, manifest);
        assert_eq!(scene.heights.len(), 65 * 65);
        let cloud = super::super::load_gaussian_splats(scene.splat_path()).unwrap();
        assert_eq!(cloud.len(), 3500);
        assert_eq!(cloud.sh_degree(), 1);

        // The COPC swath decodes back to the scene frame through the f64
        // origin: canopy points sit above the terrain, ground points on it.
        let frame = PointCloudFrame {
            origin: manifest.copc_origin,
            z_up: true,
        };
        let source = CopcPageSource::open(scene.copc_path(), frame, 4096).unwrap();
        assert_eq!(source.logical_primitives(), 5500);
        assert!(source.page_count() >= 3, "expected a multi-node octree");
        let mut total = 0usize;
        let mut canopy = 0usize;
        for page in 0..source.page_count() {
            let meta = source.meta(page);
            let PagePayload::Points { positions, colors } = source.load(page).unwrap() else {
                panic!("expected points");
            };
            assert_eq!(positions.len(), meta.count as usize);
            for (p, c) in positions.iter().zip(&colors) {
                for axis in 0..3 {
                    assert!(p[axis] >= meta.aabb.min[axis] - 1e-3);
                    assert!(p[axis] <= meta.aabb.max[axis] + 1e-3);
                }
                assert!(p[0].abs() < 32.0 && p[2].abs() < 32.0);
                let ground = terrain_height(p[0], p[2]);
                assert!(p[1] > ground, "point below terrain: {p:?} vs {ground}");
                if p[1] > ground + 3.0 {
                    canopy += 1;
                    assert!(c[1] > c[0] && c[1] > c[2], "canopy returns are green");
                }
            }
            total += positions.len();
        }
        assert_eq!(total, 5500);
        assert_eq!(canopy, 3000);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regenerates the committed fixture under tests/fixtures/splat_fusion.
    /// Run explicitly: `cargo test --features splat-fusion --lib
    /// regenerate_committed_fixture -- --ignored`.
    #[test]
    #[ignore = "writes into the source tree; run on purpose to regenerate the fixture"]
    fn regenerate_committed_fixture() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("splat_fusion");
        write_demo_fixture(&dir).unwrap();
    }

    #[test]
    fn ridge_casts_the_shadow_band_the_receivers_straddle() {
        // Sun elevation 36 degrees: a 9 m ridge at x = -20 shadows ground
        // out to roughly x = -20 + 9 / tan(36 deg) = -7.6, so the receiver
        // clusters centred at x = -10 straddle the shadow boundary.
        let reach = RIDGE_X + (terrain_height(RIDGE_X, 0.0) - terrain_height(-8.0, 0.0)) / 0.7265;
        assert!(reach > -12.0 && reach < -5.0, "{reach}");
    }
}
