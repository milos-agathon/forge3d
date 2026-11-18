//! The versioned project document.
//!
//! `Project` is the whole persisted truth: identity, scene graph, selection,
//! active workspace, viewport mode, and render configuration. It contains no
//! GPU resources, process handles, or file-system absolute paths, so a project
//! file is portable and diffable.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::ids::{new_id, AssetId, ObjectId, ProjectId};
use crate::math::{Transform, Vec3};
use crate::now_unix_ms;
use crate::scene::{
    validate_name, CameraObject, LightKind, LightObject, MeshObject, ObjectKind, Primitive, Scene,
    SceneObject, TerrainObject,
};

/// Width of the terrain in the Studio viewport's scene units.
///
/// The viewport draws a DEM across this many units regardless of its real size,
/// and the renderer works in metres; every conversion between the two uses it.
pub const SCENE_TERRAIN_SIZE: f32 = 12.0;

/// Format tag written into every project file.
pub const FORMAT: &str = "studio3d";

/// Schema version written by this build.
///
/// v1 carried `kind.primitive` on meshes; v2 adds imported glTF/GLB meshes
/// (`kind.source`) and the project asset registry; v3 adds DEM-backed terrain
/// objects. Earlier files are migrated on load — see [`crate::persist::migrate`].
pub const SCHEMA_VERSION: u32 = 5;

/// Default project name for `File → New`.
pub const UNTITLED: &str = "Untitled";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    /// Must equal [`FORMAT`]; a mismatched tag means the user opened the wrong
    /// kind of file, which deserves a specific error, not a JSON trace.
    pub format: String,
    pub schema_version: u32,
    pub id: ProjectId,
    pub name: String,
    pub created_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub scene: Scene,
    /// Imported files referenced by mesh objects. Paths are stored relative to
    /// the project file where possible (see [`crate::persist`]).
    #[serde(default)]
    pub assets: Vec<Asset>,
    /// Selected object IDs; view state, persisted so a reopened project looks
    /// the way the user left it.
    #[serde(default)]
    pub selection: Vec<ObjectId>,
    pub workspace: Workspace,
    pub viewport: ViewportState,
    pub render: RenderSettings,
    /// Animation timeline (Phase 6). `default` so v3 documents load; the
    /// v3→v4 migration stamps the version so older builds reject them
    /// instead of silently dropping keyframes.
    #[serde(default)]
    pub timeline: crate::timeline::Timeline,
    /// The map's own layer stack (Phase 15). `default` so v4 documents load;
    /// the v4→v5 migration turns the old per-object fields (the draped
    /// raster, the vector layer) and the render settings (the colormap, the
    /// plate) into layers, so a reopened map looks the way it was left.
    #[serde(default)]
    pub cartography: crate::cartography::CartographyStack,
}

impl Project {
    /// A new project with the Phase 1 starter scene: one cube, one camera
    /// looking at it, and one directional light.
    pub fn new(name: impl Into<String>) -> Self {
        let now = now_unix_ms();
        let mut scene = Scene::new("Scene");

        let cube = SceneObject {
            id: new_id(),
            name: "Cube".to_owned(),
            parent: None,
            transform: Transform::default(),
            visible: true,
            locked: false,
            kind: ObjectKind::Mesh(MeshObject::primitive(Primitive::Cube { size: 2.0 })),
        };

        let camera = SceneObject {
            id: new_id(),
            name: "Camera".to_owned(),
            parent: None,
            // Roughly (6, 4, 8) looking at the origin under the model's
            // Rz*Ry*Rx Euler convention; see `math` docs.
            transform: Transform {
                translation: Vec3::new(6.0, 4.0, 8.0),
                rotation_deg: Vec3::new(-22.0, 37.0, 0.0),
                scale: Vec3::ONE,
            },
            visible: true,
            locked: false,
            kind: ObjectKind::Camera(CameraObject {
                fov_deg: 45.0,
                near: 0.1,
                far: 1000.0,
            }),
        };

        let light = SceneObject {
            id: new_id(),
            name: "Sun".to_owned(),
            parent: None,
            transform: Transform {
                translation: Vec3::new(4.0, 6.0, 3.0),
                rotation_deg: Vec3::new(-50.0, 30.0, 0.0),
                scale: Vec3::ONE,
            },
            visible: true,
            locked: false,
            kind: ObjectKind::Light(LightObject {
                kind: LightKind::Directional,
                color: [1.0, 0.97, 0.92],
                intensity: 3.0,
            }),
        };

        scene.active_camera = Some(camera.id);
        scene.objects = vec![cube, camera, light];

        Self {
            format: FORMAT.to_owned(),
            schema_version: SCHEMA_VERSION,
            id: new_id(),
            name: name.into(),
            created_unix_ms: now,
            updated_unix_ms: now,
            scene,
            assets: Vec::new(),
            selection: Vec::new(),
            workspace: Workspace::Studio,
            viewport: ViewportState::default(),
            render: RenderSettings::default(),
            timeline: crate::timeline::Timeline::default(),
            cartography: crate::cartography::default_stack(),
        }
    }

    /// A starter scene for a real project: a terrain from `seed` when one is
    /// available, otherwise the primitive starter.
    ///
    /// The shell resolves the DEM (a bundled Bryce Canyon sample by default) and
    /// hands it in, so this crate stays free of file-system discovery.
    pub fn starter(name: impl Into<String>, seed: Option<TerrainSeed>) -> Self {
        let mut project = Self::new(name);
        let Some(seed) = seed else {
            return project;
        };
        let asset = Asset {
            id: new_id(),
            name: seed.name.clone(),
            path: seed.path.clone(),
            kind: AssetKind::Dem,
            byte_size: seed.byte_size,
        };
        let title = seed.title.clone().unwrap_or_else(|| humanize(&seed.name));
        let mut terrain = TerrainObject::new(asset.id);
        if let (Some(dimensions), Some(range)) = (seed.dimensions, seed.elevation_range) {
            terrain =
                terrain.with_dem_info(dimensions, range, seed.world_origin_xz, seed.world_span_xz);
        }
        // The starter albedo (the Bryce NAIP image) becomes the map's imagery
        // layer: it is how the engine drapes imagery on terrain.
        if let Some(raster) = seed.overlay.clone() {
            let id = new_id();
            project.assets.push(Asset {
                id,
                name: raster.name,
                path: raster.path,
                kind: AssetKind::Orthophoto,
                byte_size: raster.byte_size,
            });
            project
                .cartography
                .push(crate::cartography::CartographyLayer::new(
                    crate::cartography::CartographyLayerKind::Imagery {
                        asset: id,
                        opacity: raster.opacity.clamp(0.0, 1.0),
                        preserve_colors: true,
                        despeckle: false,
                    },
                ));
        }
        project.assets.push(asset);
        project.scene.objects.retain(|object| !object.is_mesh());
        project.scene.objects.insert(
            0,
            SceneObject {
                id: new_id(),
                name: title,
                parent: None,
                transform: Transform::default(),
                visible: true,
                locked: false,
                kind: ObjectKind::Terrain(terrain),
            },
        );
        // Frame the terrain: the default camera sits close to a 2-unit cube,
        // which would be inside a kilometre-scale DEM.
        if let Some(camera) = project
            .scene
            .objects
            .iter_mut()
            .find(|object| object.is_camera())
        {
            camera.transform.translation = Vec3::new(11.0, 7.5, 13.0);
            camera.transform.rotation_deg = Vec3::new(-24.0, 40.0, 0.0);
        }
        project.touch();
        project
    }

    /// Bump the modification timestamp. Called by the editor after every
    /// applied command, so "modified" is meaningful in a saved file.
    pub fn touch(&mut self) {
        self.updated_unix_ms = now_unix_ms();
    }

    /// The scene's terrain, if any (the viewer renders one).
    pub fn terrain_object(&self) -> Option<&SceneObject> {
        self.scene.objects.iter().find(|object| object.is_terrain())
    }

    /// Assets referenced by at least one object, in registry order.
    pub fn used_assets(&self) -> Vec<AssetId> {
        let mut used: Vec<AssetId> = Vec::new();
        for object in &self.scene.objects {
            if let Some(asset) = object.mesh().and_then(MeshObject::asset_id) {
                if !used.contains(&asset) {
                    used.push(asset);
                }
            }
            if let Some(mesh) = object.mesh() {
                if let Some(texture) = mesh.material.texture {
                    if !used.contains(&texture) {
                        used.push(texture);
                    }
                }
            }
            if let Some(terrain) = object.terrain() {
                if !used.contains(&terrain.asset) {
                    used.push(terrain.asset);
                }
                if let Some(albedo) = terrain.albedo {
                    if !used.contains(&albedo) {
                        used.push(albedo);
                    }
                }
            }
        }
        for layer in &self.cartography {
            if let Some(asset) = layer.kind.asset() {
                if !used.contains(&asset) {
                    used.push(asset);
                }
            }
        }
        if let Some(environment) = self.render.environment {
            if !used.contains(&environment) {
                used.push(environment);
            }
        }
        used
    }

    /// How many objects reference an asset (dependency count for the UI).
    pub fn asset_usage(&self, id: AssetId) -> usize {
        self.scene
            .objects
            .iter()
            .filter(|object| {
                object.mesh().and_then(MeshObject::asset_id) == Some(id)
                    || object.terrain().map(|terrain| terrain.asset) == Some(id)
                    || object.mesh().and_then(|mesh| mesh.material.texture) == Some(id)
            })
            .count()
            + self
                .cartography
                .iter()
                .filter(|layer| layer.kind.asset() == Some(id))
                .count()
            + usize::from(self.render.environment == Some(id))
    }

    /// The HDRI environment's path, when the project sets one.
    pub fn environment_path(&self) -> Option<&str> {
        let asset = self.asset(self.render.environment?)?;
        Some(asset.path.as_str())
    }

    pub fn asset(&self, id: AssetId) -> Option<&Asset> {
        self.assets.iter().find(|asset| asset.id == id)
    }

    pub fn validate(&self) -> Result<(), String> {
        // The map's layers carry the ranges the renderer takes: a project file
        // edited by hand must not smuggle a width the rasterizer refuses.
        for layer in &self.cartography {
            layer
                .validate()
                .map_err(|reason| format!("cartography layer: {reason}"))?;
            if let Some(asset) = layer.kind.asset() {
                if self.asset(asset).is_none() {
                    return Err(format!(
                        "the cartography layer {:?} references missing asset {}",
                        layer.name, asset
                    ));
                }
            }
        }
        if self.format != FORMAT {
            return Err(format!(
                "format tag is {:?}, expected {FORMAT:?}",
                self.format
            ));
        }
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "schema version {} is not the current version {SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        validate_name(&self.name, "project name")?;
        self.scene.validate()?;

        let mut asset_ids: Vec<AssetId> = Vec::with_capacity(self.assets.len());
        for asset in &self.assets {
            asset
                .validate()
                .map_err(|reason| format!("asset {:?}: {reason}", asset.name))?;
            if asset_ids.contains(&asset.id) {
                return Err(format!("duplicate asset id {}", asset.id));
            }
            asset_ids.push(asset.id);
        }
        // Material textures must exist and be images.
        for object in &self.scene.objects {
            if let Some(mesh) = object.mesh() {
                if let Some(texture) = mesh.material.texture {
                    let Some(asset) = self.asset(texture) else {
                        return Err(format!(
                            "object {:?} uses a missing material texture {texture}",
                            object.name
                        ));
                    };
                    if !asset.kind.is_raster() {
                        return Err(format!(
                            "object {:?} uses {:?} as a texture, but it is {:?}",
                            object.name, asset.name, asset.kind
                        ));
                    }
                }
            }
        }

        // A mesh or terrain may only reference an asset the project carries;
        // dangling references are what "missing asset" panics are made of.
        let background_intensity = self.render.environment_background_intensity;
        if !background_intensity.is_finite() || !(0.0..=1.0).contains(&background_intensity) {
            return Err(
                "render background intensity must be a finite number between 0 and 1".to_owned(),
            );
        }
        if let Some(environment) = self.render.environment {
            let Some(asset) = self.asset(environment) else {
                return Err(format!(
                    "render environment references missing asset {environment}"
                ));
            };
            if asset.kind != AssetKind::Hdr {
                return Err(format!(
                    "render environment uses {:?}, but it is {:?} (expected an .hdr environment)",
                    asset.name, asset.kind
                ));
            }
        }
        for object in &self.scene.objects {
            if let Some(asset) = object.mesh().and_then(MeshObject::asset_id) {
                if !asset_ids.contains(&asset) {
                    return Err(format!(
                        "object {:?} references missing asset {asset}",
                        object.name
                    ));
                }
            }
            if let Some(terrain) = object.terrain() {
                let Some(asset) = self.asset(terrain.asset) else {
                    return Err(format!(
                        "object {:?} references missing terrain asset {}",
                        object.name, terrain.asset
                    ));
                };
                if asset.kind != AssetKind::Dem {
                    return Err(format!(
                        "object {:?} uses asset {:?} as terrain, but it is {:?}",
                        object.name, asset.name, asset.kind
                    ));
                }
                if let Some(albedo) = terrain.albedo {
                    if !asset_ids.contains(&albedo) {
                        return Err(format!(
                            "object {:?} references missing albedo asset {albedo}",
                            object.name
                        ));
                    }
                }
            }
        }

        let mut seen: Vec<ObjectId> = Vec::with_capacity(self.selection.len());
        for id in &self.selection {
            if seen.contains(id) {
                return Err(format!("selection contains {id} twice"));
            }
            seen.push(*id);
            if self.scene.object(*id).is_none() {
                return Err(format!("selection references missing object {id}"));
            }
        }

        if self.render.width == 0 || self.render.height == 0 {
            return Err("render resolution must be greater than zero".to_owned());
        }
        if self.render.width > 16_384 || self.render.height > 16_384 {
            return Err("render resolution must not exceed 16384 pixels".to_owned());
        }
        if !self.render.exposure.is_finite() || !(0.05..=8.0).contains(&self.render.exposure) {
            return Err("render exposure must be between 0.05 and 8".to_owned());
        }
        if !self.render.ibl_intensity.is_finite()
            || !(0.0..=4.0).contains(&self.render.ibl_intensity)
        {
            return Err("environment/IBL intensity must be between 0 and 4".to_owned());
        }
        if !(256..=8192).contains(&self.render.shadow_resolution) {
            return Err("shadow resolution must be between 256 and 8192 texels".to_owned());
        }
        if !self.render.sky_turbidity.is_finite()
            || !(1.0..=10.0).contains(&self.render.sky_turbidity)
        {
            return Err("sky turbidity must be between 1 and 10".to_owned());
        }
        if !self.render.sky_ground_albedo.is_finite()
            || !(0.0..=1.0).contains(&self.render.sky_ground_albedo)
        {
            return Err("sky ground albedo must be between 0 and 1".to_owned());
        }
        if !self.render.sky_exposure.is_finite() || !(0.0..=4.0).contains(&self.render.sky_exposure)
        {
            return Err("sky exposure must be between 0 and 4".to_owned());
        }
        if !self.render.water.level.is_finite() {
            return Err("water level must be a finite elevation".to_owned());
        }
        if !self.render.water.roughness.is_finite()
            || !(0.0..=1.0).contains(&self.render.water.roughness)
        {
            return Err("water roughness must be between 0 and 1".to_owned());
        }
        if !self.render.water.slope_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.render.water.slope_threshold)
        {
            return Err("water slope threshold must be between 0 and 1".to_owned());
        }
        if !self.render.sun_visibility_distance.is_finite()
            || !(100.0..=20_000.0).contains(&self.render.sun_visibility_distance)
        {
            return Err("sun visibility distance must be between 100 and 20000 m".to_owned());
        }
        // The engine refuses more than 64 steps per sun ray.
        if !(8..=64).contains(&self.render.sun_visibility_steps) {
            return Err("sun visibility steps must be between 8 and 64".to_owned());
        }
        self.render
            .clouds
            .validate()
            .map_err(|reason| reason.to_string())?;
        if !self.render.fog.density.is_finite() || !(0.0..=100.0).contains(&self.render.fog.density)
        {
            return Err("fog density must be between 0 and 100".to_owned());
        }
        if !self.render.fog.reach.is_finite() || !(0.01..=1.0).contains(&self.render.fog.reach) {
            return Err("fog reach must be between 0.01 and 1".to_owned());
        }
        if self
            .render
            .fog
            .inscatter
            .iter()
            .any(|c| !c.is_finite() || !(0.0..=1.0).contains(c))
        {
            return Err("fog inscatter colour must be three values between 0 and 1".to_owned());
        }
        if !self.render.bloom.threshold.is_finite()
            || !(0.0..=1.0).contains(&self.render.bloom.threshold)
        {
            return Err("bloom threshold must be between 0 and 1".to_owned());
        }
        if !self.render.bloom.softness.is_finite()
            || !(0.0..=1.0).contains(&self.render.bloom.softness)
        {
            return Err("bloom softness must be between 0 and 1".to_owned());
        }
        if !self.render.bloom.intensity.is_finite()
            || !(0.0..=2.0).contains(&self.render.bloom.intensity)
        {
            return Err("bloom intensity must be between 0 and 2".to_owned());
        }
        if !self.render.vignette.is_finite() || !(0.0..=1.0).contains(&self.render.vignette) {
            return Err("vignette must be between 0 and 1".to_owned());
        }
        self.timeline.validate()?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Workspace {
    #[default]
    Studio,
    Forge3d,
}

impl Workspace {
    pub fn label(self) -> &'static str {
        match self {
            Workspace::Studio => "Studio",
            Workspace::Forge3d => "Forge3D",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewportMode {
    /// Fast editing mode: grid, outlines, no production shading.
    #[default]
    Solid,
    /// Forge3D-backed rendering.
    Rendered,
    /// Intermediate: lit terrain with the overlay, no shadows or POM.
    Material,
}

impl ViewportMode {
    pub fn label(self) -> &'static str {
        match self {
            ViewportMode::Solid => "Solid",
            ViewportMode::Rendered => "Rendered",
            ViewportMode::Material => "Material",
        }
    }
}

/// An orthophoto the shell resolved on disk, draped on the starter terrain
/// (the Bryce Canyon sample ships its NAIP image).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainSeedOverlay {
    pub name: String,
    pub path: String,
    pub byte_size: u64,
    /// 0..=1; defaults to the gallery's 0.82 when absent.
    #[serde(default = "default_overlay_opacity")]
    pub opacity: f32,
}

fn default_overlay_opacity() -> f32 {
    0.82
}

/// A DEM the shell resolved on disk, used to seed a new project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainSeed {
    /// Display title for the terrain object ("Swiss Alps"); defaults to the file stem.
    #[serde(default)]
    pub title: Option<String>,
    pub name: String,
    pub path: String,
    pub byte_size: u64,
    /// Optional albedo raster draped on the terrain, attached as an overlay.
    #[serde(default)]
    pub overlay: Option<TerrainSeedOverlay>,
    /// Measured DEM size, when the shell sampled it.
    #[serde(default)]
    pub dimensions: Option<[u32; 2]>,
    #[serde(default)]
    pub elevation_range: Option<[f32; 2]>,
    /// Viewer-basis `[map X, -map Y]` origin in metres.
    #[serde(default)]
    pub world_origin_xz: Option<[f64; 2]>,
    /// Viewer-basis span in metres.
    #[serde(default)]
    pub world_span_xz: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ViewportState {
    pub mode: ViewportMode,
    pub grid_visible: bool,
    /// Slope-angle hillshade painted onto the terrain in the viewport
    /// (a diffuse multiplier; the still always renders with the engine's own
    /// shading regardless of this flag).
    #[serde(default = "default_true")]
    pub hillshade: bool,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            mode: ViewportMode::Solid,
            grid_visible: true,
            hillshade: true,
        }
    }
}

/// Auxiliary render outputs for the final still ("Arbitrary Output
/// Variables"). Written next to the still as `<name>_albedo.png`,
/// `<name>_normal.png`, `<name>_depth.png` — the engine's own naming, so
/// texture-synthesis tools pick them up without configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StillAov {
    /// Unlit base color before lighting.
    #[serde(default)]
    pub albedo: bool,
    /// World-space normals remapped to [0, 1].
    #[serde(default)]
    pub normal: bool,
    /// Linear depth normalized to [near, far].
    #[serde(default)]
    pub depth: bool,
}

impl StillAov {
    /// True when at least one channel was requested.
    pub fn any_enabled(&self) -> bool {
        self.albedo || self.normal || self.depth
    }
}

/// Water on the terrain.
///
/// The terrain path has no "water level" knob: it shades water where a *mask*
/// says so, and the mask comes from the DEM (low, flat ground). The project
/// therefore stores the level and the slope threshold, and the render service
/// derives the mask with the engine's own `forge3d.gis.derive_water_mask`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WaterSettings {
    /// Flood the low, flat ground.
    #[serde(default)]
    pub enabled: bool,
    /// Water surface elevation in the DEM's own metres. 0 means "auto": the
    /// engine's own default, the lowest 15% of the terrain.
    #[serde(default)]
    pub level: f32,
    /// Ground steeper than this (metres of rise per DEM cell) stays dry.
    #[serde(default = "default_water_slope_threshold")]
    pub slope_threshold: f32,
    /// The water surface's roughness. Higher is a softer, wider sun highlight;
    /// a mirror-calm lake is near 0, a choppy one higher.
    #[serde(default = "default_water_roughness")]
    pub roughness: f32,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            level: 0.0,
            slope_threshold: default_water_slope_threshold(),
            roughness: default_water_roughness(),
        }
    }
}

/// The cloud shadows on the terrain: the engine's own cloud field, projected
/// along the sun onto the ground.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CloudSettings {
    /// Draw the clouds.
    #[serde(default)]
    pub enabled: bool,
    /// Cast the clouds' shadows onto the terrain.
    #[serde(default = "default_true")]
    pub shadows_enabled: bool,
    /// How much of the sky is covered, 0..=1.
    #[serde(default = "default_cloud_coverage")]
    pub coverage: f32,
    /// How dense the covered part is, 0..=1.
    #[serde(default = "default_cloud_density")]
    pub density: f32,
    /// How dark the shadows are, 0..=1.
    #[serde(default = "default_cloud_shadow_strength")]
    pub shadow_strength: f32,
    /// The engine's field resolution: `low` | `medium` | `high` | `ultra`.
    #[serde(default = "default_cloud_quality")]
    pub quality: String,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            shadows_enabled: true,
            coverage: default_cloud_coverage(),
            density: default_cloud_density(),
            shadow_strength: default_cloud_shadow_strength(),
            quality: default_cloud_quality(),
        }
    }
}

impl CloudSettings {
    /// The values the engine accepts.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("coverage", self.coverage),
            ("density", self.density),
            ("shadow strength", self.shadow_strength),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("the clouds' {name} must be between 0 and 1"));
            }
        }
        if !matches!(self.quality.as_str(), "low" | "medium" | "high" | "ultra") {
            return Err("the clouds' quality must be low, medium, high or ultra".to_owned());
        }
        Ok(())
    }
}

/// Atmospheric fog on the terrain.
///
/// The engine applies height-based exponential fog after the PBR pass and
/// before the tonemap: `density` is its strength and `height_falloff` how
/// fast it thins with altitude.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FogSettings {
    /// Apply the fog. Density 0 is the engine's own no-op.
    #[serde(default)]
    pub enabled: bool,
    /// Fog density; 0 is no fog. 0.05 is a haze, 0.5 is a fog bank.
    #[serde(default = "default_fog_density")]
    pub density: f32,
    /// How far up the terrain the fog climbs, as a fraction of its height
    /// (0.5 = halfway up the mountain). The engine wants a falloff per
    /// metre; the render service converts.
    #[serde(default = "default_fog_reach")]
    pub reach: f32,
    /// The colour the fog scatters into the view, linear RGB 0..=1. White is
    /// the neutral default; a warm tint reads as dust, a cool one as mist.
    #[serde(default = "default_fog_inscatter")]
    pub inscatter: [f32; 3],
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            density: default_fog_density(),
            reach: default_fog_reach(),
            inscatter: default_fog_inscatter(),
        }
    }
}

/// Bloom on the finished frame.
///
/// The engine's GPU bloom (the terrain bloom processor) and its vignette (in
/// the viewer's terrain shader) live in render paths Studio does not use: the
/// terrain path Studio renders through has neither, and both were measured to
/// change nothing there. These are the same two effects applied to the finished
/// frame instead — display-referred, so the threshold is a *display* brightness
/// rather than an HDR radiance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BloomSettings {
    /// Glow the bright pixels.
    #[serde(default)]
    pub enabled: bool,
    /// Display brightness above which pixels glow, 0..=1.
    #[serde(default = "default_bloom_threshold")]
    pub threshold: f32,
    /// Width of the threshold transition, 0..=1.
    #[serde(default = "default_bloom_softness")]
    pub softness: f32,
    /// How much glow is added, 0..=2.
    #[serde(default = "default_bloom_intensity")]
    pub intensity: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: default_bloom_threshold(),
            softness: default_bloom_softness(),
            intensity: default_bloom_intensity(),
        }
    }
}

/// Render output configuration — the Forge3D controls the pipeline exposes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    /// Tone-mapping exposure. The pipeline uses ACES.
    #[serde(default = "default_exposure")]
    pub exposure: f32,
    /// PCF sun shadows.
    #[serde(default = "default_true")]
    pub shadows: bool,
    /// Render the sky background in Rendered frames.
    #[serde(default = "default_true")]
    pub sky: bool,
    /// Environment/IBL intensity multiplier: the environment light's
    /// brightness relative to the sun (0 = environment light off).
    #[serde(default = "default_ibl_intensity")]
    pub ibl_intensity: f32,
    /// HDRI environment (Radiance `.hdr`) used for image-based lighting in
    /// Material previews and Rendered frames. `None` means the built-in
    /// studio sky. Stored as an asset id so relinking and dependency counts
    /// work like every other file reference.
    #[serde(default)]
    pub environment: Option<AssetId>,
    /// Brightness of the HDRI background, 0..=1: the Material preview's
    /// backdrop and the Rendered sky exposure (the engine samples the same
    /// environment map as its sky). 0 hides the Material backdrop and renders
    /// a black Rendered sky; turn the project sky off for no sky at all.
    #[serde(default = "default_one")]
    pub environment_background_intensity: f32,
    /// AOV export for the final still. Final stills and sequence frames only;
    /// viewport preview frames skip it entirely (they are 3 × the work of one
    /// frame and nobody watches a preview's albedo pass).
    #[serde(default)]
    pub still_aov: StillAov,
    /// Denoise the final beauty pass with Open Image Denoise (OIDN). Needs an
    /// OIDN runtime installed beside the engine; the render fails loudly with
    /// an explicit message when it is missing. Same scope as `still_aov`.
    #[serde(default)]
    pub still_denoise: bool,
    /// Shadow filter. PCF is the cheap inset filter; PCSS samples the blocker
    /// search, so contact shadows soften with distance at roughly twice the
    /// cost. Applies to previews and finals alike.
    #[serde(default)]
    pub shadow_technique: ShadowTechnique,
    /// Shadow-map edge length in texels. The engine default is 4096, which is
    /// generous for a 960×600 preview; 1024–2048 trades crispness for cost.
    #[serde(default = "default_shadow_resolution")]
    pub shadow_resolution: u32,
    /// Sun visibility: the terrain ray-marches the heightfield toward the sun,
    /// so a ridge shadows the ground behind it — terrain self-shadowing the
    /// shadow map does not resolve. Off by default; it costs a ray march.
    #[serde(default)]
    pub sun_visibility: bool,
    /// How far (metres) each sun ray searches across the heightfield. A short
    /// reach only catches nearby ridges; a long one catches the far skyline.
    #[serde(default = "default_sun_visibility_distance")]
    pub sun_visibility_distance: f64,
    /// Steps per sun ray. More steps follow a ridge line further before it is
    /// missed; the engine caps this at 64.
    #[serde(default = "default_sun_visibility_steps")]
    pub sun_visibility_steps: u32,
    /// Which sky model draws the backdrop when no HDRI is set. The backdrop
    /// only: the environment map (HDRI, or the built-in studio sky) is what
    /// lights the terrain.
    #[serde(default)]
    pub sky_model: SkyModel,
    /// Hosek/Preetham turbidity (atmospheric haze), 1..=10.
    #[serde(default = "default_sky_turbidity")]
    pub sky_turbidity: f32,
    /// Ground reflectance the sky model bounces back into the lower hemisphere.
    #[serde(default = "default_sky_ground_albedo")]
    pub sky_ground_albedo: f32,
    /// Camera-style exposure for the *analytic* sky pass (the engine applies
    /// it to the sky texture only; terrain radiance and IBL are untouched).
    /// HDRI skies keep using `environment_background_intensity` instead.
    #[serde(default = "default_sky_exposure")]
    pub sky_exposure: f32,
    /// Water on the terrain: flood the low, flat ground. The engine derives
    /// the water mask from the DEM, so the project stores the level and the
    /// slope threshold rather than a mask.
    #[serde(default)]
    pub water: WaterSettings,
    /// Atmospheric fog on the terrain.
    #[serde(default)]
    pub fog: FogSettings,
    /// The cloud shadows on the terrain (the engine's own cloud field).
    #[serde(default)]
    pub clouds: CloudSettings,
    /// Bloom on the finished frame: the bright pixels glow.
    #[serde(default)]
    pub bloom: BloomSettings,
    /// Vignette: corner darkening, 0..=1. 0 is no vignette.
    #[serde(default)]
    pub vignette: f32,
}

/// The analytic sky models the engine can draw as a backdrop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SkyModel {
    /// Hosek–Wilkie: the engine's default daylight fit.
    #[default]
    Hosek,
    /// Preetham: the older analytic fit, still useful for a flatter sky.
    Preetham,
    /// AETHER: spectral sky from a baked LUT (shipped for 2.0 / 0.3).
    Aether,
}

impl SkyModel {
    /// The token the render service maps onto `SkySettings.model`.
    pub fn token(self) -> &'static str {
        match self {
            SkyModel::Hosek => "hosek",
            SkyModel::Preetham => "preetham",
            SkyModel::Aether => "aether",
        }
    }
}

/// Shadow filter the renderer applies to the sun's shadow map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadowTechnique {
    /// Percentage-closer filtering: fixed-width, cheap, crisp.
    #[default]
    Pcf,
    /// Percentage-closer soft shadows: blocker search, contact-hardening.
    Pcss,
}

impl ShadowTechnique {
    /// The engine's technique name (`ShadowSettings.technique`).
    pub fn engine_name(self) -> &'static str {
        match self {
            ShadowTechnique::Pcf => "PCF",
            ShadowTechnique::Pcss => "PCSS",
        }
    }
}

fn default_exposure() -> f32 {
    1.35
}

fn default_true() -> bool {
    true
}

fn default_ibl_intensity() -> f32 {
    1.0
}

fn default_one() -> f32 {
    1.0
}

fn default_shadow_resolution() -> u32 {
    4096
}

fn default_sky_turbidity() -> f32 {
    2.0
}

fn default_sky_ground_albedo() -> f32 {
    0.3
}

fn default_sky_exposure() -> f32 {
    0.08
}

fn default_sun_visibility_distance() -> f64 {
    2000.0
}

fn default_sun_visibility_steps() -> u32 {
    32
}

fn default_cloud_coverage() -> f32 {
    0.5
}

fn default_cloud_density() -> f32 {
    0.5
}

fn default_cloud_shadow_strength() -> f32 {
    0.35
}

fn default_cloud_quality() -> String {
    "medium".to_owned()
}

fn default_fog_density() -> f32 {
    // 0.5 veiled 72% of a 922x795 frame; 0.05 is a haze you can see through.
    0.05
}

fn default_fog_inscatter() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_fog_reach() -> f32 {
    0.5
}

fn default_bloom_threshold() -> f32 {
    0.8
}

fn default_bloom_softness() -> f32 {
    0.2
}

fn default_bloom_intensity() -> f32 {
    0.3
}

fn default_water_roughness() -> f32 {
    0.35
}

fn default_water_slope_threshold() -> f32 {
    // The engine's own water-and-clouds example uses 0.18; the library default
    // (0.02) is flat-only and leaves a puddle.
    0.18
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            exposure: default_exposure(),
            shadows: true,
            sky: true,
            ibl_intensity: default_ibl_intensity(),
            environment: None,
            environment_background_intensity: default_one(),
            still_aov: StillAov::default(),
            still_denoise: false,
            shadow_technique: ShadowTechnique::default(),
            shadow_resolution: default_shadow_resolution(),
            sun_visibility: false,
            sun_visibility_distance: default_sun_visibility_distance(),
            sun_visibility_steps: default_sun_visibility_steps(),
            sky_model: SkyModel::default(),
            sky_turbidity: default_sky_turbidity(),
            sky_ground_albedo: default_sky_ground_albedo(),
            sky_exposure: default_sky_exposure(),
            water: WaterSettings::default(),
            fog: FogSettings::default(),
            clouds: CloudSettings::default(),
            bloom: BloomSettings::default(),
            vignette: 0.0,
        }
    }
}

/// An imported source file. The project stores a reference, never the bytes:
/// the viewport reads the file when it needs it, and the Assets panel can
/// relink it when the file moves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    pub id: AssetId,
    /// Display name, normally the imported file name.
    pub name: String,
    /// Path to the source file. Relative paths resolve against the project
    /// file's directory; absolute paths survive only when the file cannot be
    /// expressed relatively.
    pub path: String,
    pub kind: AssetKind,
    pub byte_size: u64,
}

impl Asset {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_nil() {
            return Err("asset id must not be nil".to_owned());
        }
        validate_name(&self.name, "asset name")?;
        if self.path.trim().is_empty() {
            return Err("asset path must not be empty".to_owned());
        }
        if self.path.chars().any(|ch| ch.is_control()) {
            return Err("asset path must not contain control characters".to_owned());
        }
        if self.byte_size == 0 {
            return Err("asset file is empty".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Gltf,
    Glb,
    /// A GeoTIFF elevation model (single band, float or integer).
    Dem,
    /// An orthophoto / albedo raster draped over terrain.
    Orthophoto,
    /// A Radiance `.hdr` environment map for image-based lighting.
    Hdr,
    /// A vector layer: a GeoPackage or GeoJSON of lines (rail, river, boundary).
    Vector,
}

impl AssetKind {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("gltf") => Some(AssetKind::Gltf),
            Some("glb") => Some(AssetKind::Glb),
            Some("tif" | "tiff") => Some(AssetKind::Dem),
            Some("hdr") => Some(AssetKind::Hdr),
            Some("gpkg" | "geojson" | "json") => Some(AssetKind::Vector),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            AssetKind::Gltf => "gltf",
            AssetKind::Glb => "glb",
            AssetKind::Dem => "tif",
            AssetKind::Orthophoto => "tif",
            AssetKind::Hdr => "hdr",
            AssetKind::Vector => "gpkg",
        }
    }

    pub fn is_raster(self) -> bool {
        matches!(self, AssetKind::Dem | AssetKind::Orthophoto)
    }
}

/// `switzerland_dem.tif` -> "Switzerland Dem" when no title was supplied.
fn humanize(file_name: &str) -> String {
    let stem = file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name);
    let words = stem
        .split(['_', '-', ' '])
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>();
    if words.is_empty() {
        return file_name.to_owned();
    }
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_project_has_the_starter_scene() {
        let project = Project::new(UNTITLED);
        assert_eq!(project.scene.objects.len(), 3);
        assert_eq!(
            project.scene.objects.iter().filter(|o| o.is_mesh()).count(),
            1
        );
        assert_eq!(
            project
                .scene
                .objects
                .iter()
                .filter(|o| o.is_camera())
                .count(),
            1
        );
        assert_eq!(
            project
                .scene
                .objects
                .iter()
                .filter(|o| o.is_light())
                .count(),
            1
        );
        assert!(project.scene.camera().is_some());
        assert_eq!(project.scene.camera().unwrap().name, "Camera");
        project.validate().expect("starter project is valid");
    }

    #[test]
    fn new_projects_get_distinct_ids() {
        let a = Project::new(UNTITLED);
        let b = Project::new(UNTITLED);
        assert_ne!(a.id, b.id);
        assert_ne!(a.scene.id, b.scene.id);
        for (left, right) in a.scene.objects.iter().zip(&b.scene.objects) {
            assert_ne!(left.id, right.id);
        }
    }

    #[test]
    fn validation_rejects_dangling_selection() {
        let mut project = Project::new(UNTITLED);
        project.selection = vec![new_id()];
        assert!(project.validate().unwrap_err().contains("selection"));
    }

    #[test]
    fn validation_rejects_zero_resolution() {
        let mut project = Project::new(UNTITLED);
        project.render.height = 0;
        assert!(project
            .validate()
            .unwrap_err()
            .contains("greater than zero"));
    }

    #[test]
    fn assets_are_validated() {
        let good = Asset {
            id: new_id(),
            name: "duck.glb".to_owned(),
            path: "assets/duck.glb".to_owned(),
            kind: AssetKind::Glb,
            byte_size: 1024,
        };
        good.validate().unwrap();

        let mut empty = good.clone();
        empty.byte_size = 0;
        assert!(empty.validate().is_err());

        let mut nameless = good.clone();
        nameless.name = "  ".to_owned();
        assert!(nameless.validate().is_err());

        let mut pathless = good.clone();
        pathless.path = String::new();
        assert!(pathless.validate().is_err());

        let mut nil = good;
        nil.id = uuid::Uuid::nil();
        assert!(nil.validate().is_err());
    }

    #[test]
    fn asset_kind_is_read_from_the_extension() {
        assert_eq!(
            AssetKind::from_path(Path::new("a.glb")),
            Some(AssetKind::Glb)
        );
        assert_eq!(
            AssetKind::from_path(Path::new("DIR/A.GLTF")),
            Some(AssetKind::Gltf)
        );
        assert_eq!(AssetKind::from_path(Path::new("a.obj")), None);
    }

    #[test]
    fn meshes_may_not_reference_assets_the_project_lacks() {
        let mut project = Project::new(UNTITLED);
        project.scene.objects[0].kind = ObjectKind::Mesh(MeshObject::gltf(new_id(), None));
        let error = project.validate().unwrap_err();
        assert!(error.contains("missing asset"), "{error}");
    }

    #[test]
    fn validation_rejects_wrong_format_tag() {
        let mut project = Project::new(UNTITLED);
        project.format = "forge3d".to_owned();
        assert!(project.validate().unwrap_err().contains("format tag"));
    }
}
