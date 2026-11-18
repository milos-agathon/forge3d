//! Writes the committed sample projects under `studio/samples/`.
//!
//! Asset paths are written **relative to the sample file** (the app resolves
//! them on load), so the folder is portable: an `.tif` lives in the repository's
//! `assets/`, the HDRI ships with the samples. That also means this tool writes
//! the JSON directly instead of `save_atomic`, whose relativisation only covers
//! assets inside the project's own directory.
//!
//! Run from `studio/`:
//!     cargo run -p studio-core --example make_samples
//!
//! Regenerate after changing the starter scene or the sample layout, then
//! commit `samples/`. Ids and timestamps are freshly generated on each run, so
//! regenerating rewrites them — that churn is expected and harmless.
//! `crates/studio-core/tests/samples.rs` loads every file.

use std::fs;
use std::path::{Path, PathBuf};

use studio_core::ids;
use studio_core::persist::to_json_pretty;
use studio_core::project::{Asset, AssetKind, Project, TerrainSeed, TerrainSeedOverlay};
use studio_core::scene::{MeshObject, ObjectKind, Primitive};
use studio_core::timeline::{Channel, ChannelKeys, Interpolation, Keyframe, ObjectTrack};

fn samples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples")
}

fn key(frame: u32, value: f32, interpolation: Interpolation) -> Keyframe {
    Keyframe {
        frame,
        value,
        interpolation,
    }
}

fn studio_sky_asset(dir: &Path) -> Asset {
    let path = "media/studio-sky.hdr";
    let byte_size = fs::metadata(dir.join(path))
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    Asset {
        id: ids::new_id(),
        name: "studio-sky.hdr".to_owned(),
        path: path.to_owned(),
        kind: AssetKind::Hdr,
        byte_size,
    }
}

fn write(dir: &Path, name: &str, project: &Project) {
    project
        .validate()
        .unwrap_or_else(|reason| panic!("{name} is invalid: {reason}"));
    let path = dir.join(name);
    fs::write(&path, to_json_pretty(project).expect("serialize project")).expect("write sample");
    eprintln!("wrote {}", path.display());
}

/// Bryce Canyon: the terrain-first starter, drape attached, lit by the sample
/// HDRI. This is the scene the README walks through.
fn bryce_terrain(dir: &Path) -> Project {
    let mut project = Project::starter(
        "Bryce Canyon",
        Some(TerrainSeed {
            title: Some("Bryce Canyon".to_owned()),
            name: "Bryce_Canyon_1024.tif".to_owned(),
            path: "../../assets/tif/Bryce_Canyon_1024.tif".to_owned(),
            byte_size: 3_321_356,
            dimensions: Some([1024, 1024]),
            elevation_range: Some([2099.23, 2553.58]),
            world_origin_xz: Some([395_998.5, -4_165_574.5]),
            world_span_xz: Some([4000.0, 3800.0]),
            overlay: Some(TerrainSeedOverlay {
                name: "Bryce_Canyon_naip.tif".to_owned(),
                path: "../../assets/tif/Bryce_Canyon_naip.tif".to_owned(),
                byte_size: 11_775_186,
                opacity: 0.82,
            }),
        }),
    );
    let sky = studio_sky_asset(dir);
    project.render.environment = Some(sky.id);
    project.render.ibl_intensity = 1.0;
    project.assets.push(sky);
    project
}

/// A keyed cube: translation and rotation over 96 frames at 24 fps. Playback
/// here is the Phase 6 feature in its smallest honest form.
fn cube_animation() -> Project {
    let mut project = Project::new("Cube animation");
    project.timeline.length = 96;
    let cube = project
        .scene
        .objects
        .iter()
        .find(|object| object.is_mesh())
        .expect("starter scene has a cube")
        .id;
    project.timeline.tracks.push(ObjectTrack {
        object_id: cube,
        channels: vec![
            ChannelKeys {
                channel: Channel::TranslationX,
                keys: vec![
                    key(0, 0.0, Interpolation::Smooth),
                    key(24, 3.0, Interpolation::Smooth),
                    key(48, 0.0, Interpolation::Smooth),
                    key(72, -3.0, Interpolation::Smooth),
                    key(96, 0.0, Interpolation::Smooth),
                ],
            },
            ChannelKeys {
                channel: Channel::RotationZ,
                keys: vec![
                    key(0, 0.0, Interpolation::Smooth),
                    key(96, 180.0, Interpolation::Smooth),
                ],
            },
        ],
    });
    project
}

/// A sphere lit only by the HDRI (the sun is turned down), so the Material
/// preview's environment setting is visible on its own.
fn hdri_studio(dir: &Path) -> Project {
    let mut project = Project::new("HDRI studio");
    if let Some(object) = project
        .scene
        .objects
        .iter_mut()
        .find(|object| object.is_mesh())
    {
        object.name = "Sphere".to_owned();
        object.kind = ObjectKind::Mesh(MeshObject::primitive(Primitive::Sphere { radius: 1.2 }));
    }
    for object in project.scene.objects.iter_mut() {
        if let ObjectKind::Light(light) = &mut object.kind {
            light.intensity = 0.6;
        }
    }
    let sky = studio_sky_asset(dir);
    project.render.environment = Some(sky.id);
    project.render.ibl_intensity = 1.2;
    project.render.sky = true;
    project.assets.push(sky);
    project
}

fn main() {
    let dir = samples_dir();
    fs::create_dir_all(dir.join("media")).expect("create samples directory");
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
    if !assets.join("tif/Bryce_Canyon_1024.tif").is_file() {
        eprintln!(
            "note: the Bryce GeoTIFF is not in this checkout (Git LFS?); the sample still writes \
             its reference"
        );
    }

    write(&dir, "bryce-terrain.studio3d", &bryce_terrain(&dir));
    write(&dir, "cube-animation.studio3d", &cube_animation());
    write(&dir, "hdri-studio.studio3d", &hdri_studio(&dir));
}
