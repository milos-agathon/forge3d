//! The scene graph: objects, their transforms, and their typed payloads.
//!
//! Phase 2 extends meshes from primitives-only to primitives **or** an imported
//! glTF/GLB asset. That is a schema version bump: v1 objects carried
//! `kind.primitive`, v2 objects carry `kind.source` (`persist::migrate` rewrites
//! v1 files on load).

use serde::{Deserialize, Serialize};

use crate::ids::{new_id, AssetId, ObjectId, SceneId};
use crate::math::Transform;

/// Maximum display-name length; long enough for real names, short enough that
/// a corrupted file cannot smuggle megabytes into a panel header.
pub const MAX_NAME_LEN: usize = 64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Scene {
    pub id: SceneId,
    pub name: String,
    /// Ordered list; order is the outliner order. Hierarchy is expressed by
    /// [`SceneObject::parent`], never by list position.
    pub objects: Vec<SceneObject>,
    /// Camera rendered by the viewport and by Forge3D handoff.
    pub active_camera: Option<ObjectId>,
}

impl Scene {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: new_id(),
            name: name.into(),
            objects: Vec::new(),
            active_camera: None,
        }
    }

    pub fn object(&self, id: ObjectId) -> Option<&SceneObject> {
        self.objects.iter().find(|object| object.id == id)
    }

    pub fn object_mut(&mut self, id: ObjectId) -> Option<&mut SceneObject> {
        self.objects.iter_mut().find(|object| object.id == id)
    }

    pub fn index_of(&self, id: ObjectId) -> Option<usize> {
        self.objects.iter().position(|object| object.id == id)
    }

    /// The camera Forge3D and the viewport should use: the active camera when
    /// it exists and is a camera, otherwise the first camera in the scene.
    pub fn camera(&self) -> Option<&SceneObject> {
        let active = self
            .active_camera
            .and_then(|id| self.object(id))
            .filter(|object| object.is_camera());
        active.or_else(|| self.objects.iter().find(|object| object.is_camera()))
    }

    /// Children of `id`, in outliner order.
    pub fn children_of(&self, id: ObjectId) -> Vec<ObjectId> {
        self.objects
            .iter()
            .filter(|object| object.parent == Some(id))
            .map(|object| object.id)
            .collect()
    }

    /// A display name that is not already used, Blender-style (`Cube.001`).
    pub fn unique_name(&self, base: &str) -> String {
        let trimmed: String = base.chars().take(MAX_NAME_LEN.saturating_sub(4)).collect();
        let taken = |candidate: &str| {
            self.objects
                .iter()
                .any(|object| object.name.eq_ignore_ascii_case(candidate))
        };
        if !taken(&trimmed) {
            return trimmed;
        }
        for suffix in 1..10_000 {
            let candidate = format!("{trimmed}.{suffix:03}");
            if !taken(&candidate) {
                return candidate;
            }
        }
        format!("{trimmed}.copy")
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_name(&self.name, "scene name")?;

        let mut seen: Vec<ObjectId> = Vec::with_capacity(self.objects.len());
        for object in &self.objects {
            if seen.contains(&object.id) {
                return Err(format!("duplicate object id {}", object.id));
            }
            seen.push(object.id);
        }

        for object in &self.objects {
            object.validate()?;
            if let Some(parent) = object.parent {
                if parent == object.id {
                    return Err(format!("object {:?} is its own parent", object.name));
                }
                if self.object(parent).is_none() {
                    return Err(format!(
                        "object {:?} references missing parent {parent}",
                        object.name
                    ));
                }
                // Walk up; a cycle would otherwise hang every recursive walk.
                let mut cursor = Some(parent);
                let mut depth = 0;
                while let Some(current) = cursor {
                    depth += 1;
                    if depth > self.objects.len() {
                        return Err(format!(
                            "object {:?} is part of a parent cycle",
                            object.name
                        ));
                    }
                    if current == object.id {
                        return Err(format!(
                            "object {:?} is part of a parent cycle",
                            object.name
                        ));
                    }
                    cursor = self.object(current).and_then(|next| next.parent);
                }
            }
        }

        if let Some(active) = self.active_camera {
            match self.object(active) {
                Some(object) if object.is_camera() => {}
                Some(_) => return Err(format!("active camera {active} is not a camera object")),
                None => {
                    return Err(format!("active camera {active} does not exist"));
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneObject {
    pub id: ObjectId,
    pub name: String,
    /// Parent-relative hierarchy; `None` means the object is at scene root.
    #[serde(default)]
    pub parent: Option<ObjectId>,
    pub transform: Transform,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default)]
    pub locked: bool,
    pub kind: ObjectKind,
}

impl SceneObject {
    pub fn is_camera(&self) -> bool {
        matches!(self.kind, ObjectKind::Camera(_))
    }

    pub fn is_light(&self) -> bool {
        matches!(self.kind, ObjectKind::Light(_))
    }

    pub fn is_mesh(&self) -> bool {
        matches!(self.kind, ObjectKind::Mesh(_))
    }

    pub fn is_terrain(&self) -> bool {
        matches!(self.kind, ObjectKind::Terrain(_))
    }

    pub fn terrain(&self) -> Option<&TerrainObject> {
        match &self.kind {
            ObjectKind::Terrain(terrain) => Some(terrain),
            _ => None,
        }
    }

    /// Short label used by the outliner type column.
    pub fn kind_label(&self) -> &'static str {
        match self.kind {
            ObjectKind::Mesh(_) => "mesh",
            ObjectKind::Terrain(_) => "terrain",
            ObjectKind::Camera(_) => "camera",
            ObjectKind::Light(_) => "light",
        }
    }

    pub fn mesh(&self) -> Option<&MeshObject> {
        match &self.kind {
            ObjectKind::Mesh(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_name(&self.name, "object name")?;
        self.transform
            .validate()
            .map_err(|reason| format!("object {:?}: {reason}", self.name))?;
        match &self.kind {
            ObjectKind::Mesh(mesh) => mesh
                .validate()
                .map_err(|reason| format!("object {:?}: {reason}", self.name)),
            ObjectKind::Terrain(terrain) => terrain
                .validate()
                .map_err(|reason| format!("object {:?}: {reason}", self.name)),
            ObjectKind::Camera(camera) => camera
                .validate()
                .map_err(|reason| format!("object {:?}: {reason}", self.name)),
            ObjectKind::Light(light) => light
                .validate()
                .map_err(|reason| format!("object {:?}: {reason}", self.name)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObjectKind {
    Mesh(MeshObject),
    /// A DEM-backed terrain surface. Forge3D renders it natively; Studio shows
    /// a sampled heightfield preview.
    Terrain(TerrainObject),
    Camera(CameraObject),
    Light(LightObject),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshObject {
    pub source: MeshSource,
    /// Studio-side material. The terrain renderer does not consume it, which the
    /// capability report states explicitly rather than pretending otherwise.
    #[serde(default)]
    pub material: Material,
}

/// A simple, capability-taggable material. No node graph: Studio is not a
/// universal shader editor, and Forge3D's terrain path has no material override.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Material {
    /// Linear RGB, 0..=1.
    #[serde(default = "default_base_color")]
    pub base_color: [f32; 3],
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default = "default_metallic")]
    pub metallic: f32,
    /// Optional albedo texture (PNG/JPEG). The Solid viewport samples it.
    #[serde(default)]
    pub texture: Option<AssetId>,
}

fn default_base_color() -> [f32; 3] {
    [0.62, 0.66, 0.72]
}

fn default_roughness() -> f32 {
    0.6
}

fn default_metallic() -> f32 {
    0.04
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: default_base_color(),
            roughness: default_roughness(),
            metallic: default_metallic(),
            texture: None,
        }
    }
}

impl Material {
    pub fn validate(&self) -> Result<(), String> {
        for channel in self.base_color {
            if !channel.is_finite() || !(0.0..=1.0).contains(&channel) {
                return Err("material colour channels must be between 0 and 1".to_owned());
            }
        }
        if !self.roughness.is_finite() || !(0.0..=1.0).contains(&self.roughness) {
            return Err("material roughness must be between 0 and 1".to_owned());
        }
        if !self.metallic.is_finite() || !(0.0..=1.0).contains(&self.metallic) {
            return Err("material metallic must be between 0 and 1".to_owned());
        }
        Ok(())
    }
}

impl MeshObject {
    pub fn primitive(primitive: Primitive) -> Self {
        Self {
            source: MeshSource::Primitive { primitive },
            material: Material::default(),
        }
    }

    pub fn gltf(asset: AssetId, node: Option<String>) -> Self {
        Self {
            source: MeshSource::Gltf { asset, node },
            material: Material::default(),
        }
    }

    pub fn primitive_source(&self) -> Option<&Primitive> {
        match &self.source {
            MeshSource::Primitive { primitive } => Some(primitive),
            MeshSource::Gltf { .. } => None,
        }
    }

    pub fn asset_id(&self) -> Option<AssetId> {
        match self.source {
            MeshSource::Gltf { asset, .. } => Some(asset),
            MeshSource::Primitive { .. } => None,
        }
    }

    fn validate(&self) -> Result<(), String> {
        self.material.validate()?;
        match &self.source {
            MeshSource::Primitive { primitive } => primitive.validate(),
            MeshSource::Gltf { asset, node } => {
                if asset.is_nil() {
                    return Err("imported mesh has no asset id".to_owned());
                }
                if let Some(node) = node {
                    if node.trim().is_empty() {
                        return Err("imported mesh node must not be empty".to_owned());
                    }
                }
                Ok(())
            }
        }
    }
}

/// A DEM-backed terrain object.
///
/// The DEM itself lives in the asset registry; this holds the render-time knobs
/// the viewer understands. One terrain per scene until the viewer can load more
/// than one (see the capability report's degradations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainObject {
    pub asset: AssetId,
    /// Vertical exaggeration. 1.0 renders the DEM at its true scale.
    #[serde(default = "default_z_scale")]
    pub z_scale: f32,
    /// Optional orthophoto/albedo raster.
    #[serde(default)]
    pub albedo: Option<AssetId>,
    /// DEM pixel dimensions (display + diagnostics).
    #[serde(default)]
    pub dimensions: Option<[u32; 2]>,
    /// Viewer-basis world origin, `[map X, -map Y]`, metres. Georeferenced DEMs
    /// live far from the origin, so this is not optional in practice.
    #[serde(default)]
    pub world_origin_xz: Option<[f64; 2]>,
    /// Viewer-basis world span, metres (the viewer's terrain world units).
    #[serde(default)]
    pub world_span_xz: Option<[f64; 2]>,
    /// Elevation range in metres, `[min, max]`.
    #[serde(default)]
    pub elevation_range: Option<[f32; 2]>,
    // The draped raster, the vector layer and their labels are the map's
    // layers now (`Project::cartography`): one stack, one order, one place.
}

fn default_z_scale() -> f32 {
    1.0
}

/// What `TerrainRenderParams` accepts; it raises `z_scale must be 0.1-50.0`.
pub const MIN_Z_SCALE: f32 = 0.1;
pub const MAX_Z_SCALE: f32 = 50.0;

/// How much relief makes a DEM read as terrain rather than as a printed sheet:
/// 6% of the longer footprint edge.
///
/// The rule only ever exaggerates. Bryce Canyon, the bundled sample, is 11.3%
/// relief at true scale, so it keeps `1.0`; the Swiss DEM is 1.1% (4,031 m of
/// relief across 352 km) and comes out at 5.25, where its valleys and ridges
/// are visible instead of a plate. A DEM whose footprint is unknown stays at
/// true scale rather than guessing.
pub fn dem_default_z_scale(
    elevation_range: Option<[f32; 2]>,
    world_span_xz: Option<[f64; 2]>,
) -> f32 {
    const TARGET_RELIEF_RATIO: f64 = 0.06;
    let (Some([low, high]), Some(span)) = (elevation_range, world_span_xz) else {
        return 1.0;
    };
    let relief = f64::from(high) - f64::from(low);
    let footprint = span[0].max(span[1]);
    if !relief.is_finite() || !footprint.is_finite() || relief <= 0.0 || footprint <= 0.0 {
        return 1.0;
    }
    (TARGET_RELIEF_RATIO * footprint / relief).clamp(1.0, f64::from(MAX_Z_SCALE)) as f32
}

impl TerrainObject {
    pub fn new(asset: AssetId) -> Self {
        Self {
            asset,
            z_scale: 1.0,
            albedo: None,
            dimensions: None,
            world_origin_xz: None,
            world_span_xz: None,
            elevation_range: None,
        }
    }

    /// Attach the DEM's measured size, footprint, and elevation range.
    ///
    /// The vertical scale follows the DEM: a new elevation model's relief has
    /// nothing to do with the previous one's, so the default is re-derived
    /// here (and the Inspector shows the value, so it is still the user's).
    pub fn with_dem_info(
        mut self,
        dimensions: [u32; 2],
        elevation_range: [f32; 2],
        world_origin_xz: Option<[f64; 2]>,
        world_span_xz: Option<[f64; 2]>,
    ) -> Self {
        self.dimensions = Some(dimensions);
        self.elevation_range = Some(elevation_range);
        self.world_origin_xz = world_origin_xz;
        self.world_span_xz = world_span_xz;
        self.z_scale = dem_default_z_scale(self.elevation_range, self.world_span_xz);
        self
    }

    /// The viewer frames terrain as `radius = 1.5 * |(w, 0, h)|` around
    /// `(w/2, elevation_range * z_scale / 2, h/2)` — see
    /// `src/viewer/terrain/scene/terrain_load.rs`. Reproducing it here keeps the
    /// two workspaces looking at the same thing.
    pub fn viewer_framing(&self) -> Option<(f64, [f64; 3])> {
        // Georeferenced: origin + span in metres. Local fallback: the viewer uses
        // DEM pixels as world units, origin at zero.
        let (origin, span) = match (self.world_origin_xz, self.world_span_xz) {
            (Some(origin), Some(span)) => (origin, span),
            _ => {
                let [width, height] = self.dimensions?;
                ([0.0, 0.0], [f64::from(width), f64::from(height)])
            }
        };
        if !span[0].is_finite() || !span[1].is_finite() || span[0] <= 0.0 || span[1] <= 0.0 {
            return None;
        }
        let [min_elevation, max_elevation] = self.elevation_range.unwrap_or([0.0, 0.0]);
        let radius = span[0].max(span[1]) * 1.5;
        let target = [
            origin[0] + span[0] * 0.5,
            f64::from(max_elevation - min_elevation) * f64::from(self.z_scale) * 0.5,
            origin[1] + span[1] * 0.5,
        ];
        Some((radius, target))
    }

    fn validate(&self) -> Result<(), String> {
        if self.asset.is_nil() {
            return Err("terrain has no DEM asset".to_owned());
        }
        // The engine refuses anything outside its range (it raises instead of
        // clamping), so a value in here must be one `TerrainRenderParams`
        // accepts. The map's own layers are validated with the stack.
        if !self.z_scale.is_finite() || !(MIN_Z_SCALE..=MAX_Z_SCALE).contains(&self.z_scale) {
            return Err(format!(
                "terrain vertical scale must be between {MIN_Z_SCALE} and {MAX_Z_SCALE}"
            ));
        }
        Ok(())
    }
}

/// Where a mesh's geometry comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MeshSource {
    Primitive {
        primitive: Primitive,
    },
    /// A node inside an imported glTF/GLB asset. `node` is the glTF node name;
    /// `None` means "everything the asset contains".
    Gltf {
        asset: AssetId,
        #[serde(default)]
        node: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Primitive {
    Cube { size: f32 },
    Sphere { radius: f32 },
    Plane { size: f32 },
}

impl Primitive {
    /// Edge length / diameter used by the solid viewport gizmo.
    pub fn extent(&self) -> f32 {
        match *self {
            Primitive::Cube { size } => size,
            Primitive::Sphere { radius } => radius * 2.0,
            Primitive::Plane { size } => size,
        }
    }

    fn validate(&self) -> Result<(), String> {
        let (label, value) = match *self {
            Primitive::Cube { size } => ("cube size", size),
            Primitive::Sphere { radius } => ("sphere radius", radius),
            Primitive::Plane { size } => ("plane size", size),
        };
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{label} must be a positive finite number"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CameraObject {
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl CameraObject {
    pub fn validate(&self) -> Result<(), String> {
        if !self.fov_deg.is_finite() || !(1.0..179.0).contains(&self.fov_deg) {
            return Err("camera field of view must be between 1 and 179 degrees".to_owned());
        }
        if !self.near.is_finite() || self.near <= 0.0 {
            return Err("camera near plane must be a positive number".to_owned());
        }
        if !self.far.is_finite() || self.far <= self.near {
            return Err("camera far plane must be greater than the near plane".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightKind {
    Directional,
    Point,
}

impl LightKind {
    pub fn label(self) -> &'static str {
        match self {
            LightKind::Directional => "Directional",
            LightKind::Point => "Point",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LightObject {
    pub kind: LightKind,
    /// Linear RGB in 0..=1.
    pub color: [f32; 3],
    pub intensity: f32,
}

impl LightObject {
    pub fn validate(&self) -> Result<(), String> {
        for channel in self.color {
            if !channel.is_finite() || !(0.0..=1.0).contains(&channel) {
                return Err("light colour channels must be between 0 and 1".to_owned());
            }
        }
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return Err("light intensity must be a non-negative number".to_owned());
        }
        Ok(())
    }
}

fn default_true() -> bool {
    true
}

pub(crate) fn validate_name(name: &str, field: &str) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if trimmed.chars().count() > MAX_NAME_LEN {
        return Err(format!("{field} must be at most {MAX_NAME_LEN} characters"));
    }
    if name.chars().any(|ch| ch.is_control()) {
        return Err(format!("{field} must not contain control characters"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    fn cube(name: &str) -> SceneObject {
        SceneObject {
            id: new_id(),
            name: name.to_owned(),
            parent: None,
            transform: Transform::default(),
            visible: true,
            locked: false,
            kind: ObjectKind::Mesh(MeshObject::primitive(Primitive::Cube { size: 2.0 })),
        }
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let mut scene = Scene::new("Scene");
        let object = cube("Cube");
        let mut clone = object.clone();
        clone.name = "Cube 2".to_owned();
        scene.objects.push(object);
        scene.objects.push(clone);
        assert!(scene
            .validate()
            .unwrap_err()
            .contains("duplicate object id"));
    }

    #[test]
    fn parent_cycles_are_rejected() {
        let mut scene = Scene::new("Scene");
        let mut a = cube("A");
        let mut b = cube("B");
        a.parent = Some(b.id);
        b.parent = Some(a.id);
        scene.objects.push(a);
        scene.objects.push(b);
        assert!(scene.validate().unwrap_err().contains("parent cycle"));
    }

    #[test]
    fn missing_parent_is_rejected() {
        let mut scene = Scene::new("Scene");
        let mut a = cube("A");
        a.parent = Some(new_id());
        scene.objects.push(a);
        assert!(scene.validate().unwrap_err().contains("missing parent"));
    }

    #[test]
    fn active_camera_must_be_a_camera() {
        let mut scene = Scene::new("Scene");
        let change = cube("Cube");
        scene.active_camera = Some(change.id);
        scene.objects.push(change);
        assert!(scene.validate().unwrap_err().contains("not a camera"));
    }

    #[test]
    fn camera_falls_back_to_first_camera_object() {
        let mut scene = Scene::new("Scene");
        let mut camera = cube("Camera");
        camera.kind = ObjectKind::Camera(CameraObject {
            fov_deg: 45.0,
            near: 0.1,
            far: 100.0,
        });
        let camera_id = camera.id;
        scene.objects.push(camera);
        assert_eq!(scene.camera().map(|object| object.id), Some(camera_id));
        assert_eq!(scene.objects[0].transform.translation, Vec3::ZERO);
    }

    #[test]
    fn name_validation_rejects_control_characters() {
        assert!(validate_name("ok", "name").is_ok());
        assert!(validate_name("  ", "name").is_err());
        assert!(validate_name("bad\u{7}", "name").is_err());
        assert!(validate_name(&"x".repeat(MAX_NAME_LEN + 1), "name").is_err());
    }

    #[test]
    fn unique_names_follow_the_blender_suffix_pattern() {
        let mut scene = Scene::new("Scene");
        scene.objects.push(cube("Cube"));
        assert_eq!(scene.unique_name("Cube"), "Cube.001");
        let mut second = cube("Cube.001");
        second.id = new_id();
        scene.objects.push(second);
        assert_eq!(scene.unique_name("Cube"), "Cube.002");
        assert_eq!(scene.unique_name("Other"), "Other");
    }

    #[test]
    fn gltf_meshes_need_a_non_nil_asset() {
        let mesh = MeshObject::gltf(uuid::Uuid::nil(), None);
        assert!(mesh.validate().is_err());
        let mesh = MeshObject::gltf(new_id(), Some(" ".to_owned()));
        assert!(mesh.validate().is_err());
        let mesh = MeshObject::gltf(new_id(), Some("Sphere".to_owned()));
        assert!(mesh.validate().is_ok());
    }
}
