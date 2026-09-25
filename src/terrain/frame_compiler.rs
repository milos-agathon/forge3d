use std::collections::BTreeSet;

use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

use crate::core::resource_tracker::{register_buffer_explicit, ResourceHandle};
use crate::terrain::accumulation::frame_seed;

/// Fixed label-fade interval in frame counts: 256 fade steps keep the largest
/// possible alpha quantization error strictly below one 8-bit step (< 1/255).
pub const LABEL_FADE_FRAMES: u64 = 256;

pub const COMPILED_FRAME_SCHEMA: &str = "forge3d.chronos.compiled_frame/1";

fn engine_revision() -> String {
    format!("{}+{}", env!("CARGO_PKG_VERSION"), env!("FORGE3D_GIT_SHA"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn tracked_payload_handle(bytes: u64) -> Result<ResourceHandle, String> {
    crate::core::memory_tracker::global_tracker().check_budget(bytes)?;
    Ok(register_buffer_explicit(bytes, true))
}

pub(crate) fn normalize_value(value: &Value) -> Result<Value, String> {
    match value {
        Value::Null | Value::Bool(_) | Value::String(_) => Ok(value.clone()),
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                return Ok(value.clone());
            }
            let Some(raw) = number.as_f64() else {
                return Err("unrepresentable JSON number".to_string());
            };
            if !raw.is_finite() {
                return Err("non-finite number in canonical JSON".to_string());
            }
            let fixed = if raw == 0.0 { 0.0 } else { raw };
            Number::from_f64(fixed)
                .map(Value::Number)
                .ok_or_else(|| "unrepresentable JSON number".to_string())
        }
        Value::Array(items) => items
            .iter()
            .map(normalize_value)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, item) in map.iter() {
                out.insert(key.clone(), normalize_value(item)?);
            }
            Ok(Value::Object(out))
        }
    }
}

pub(crate) fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    let normalized = normalize_value(value)?;
    serde_json::to_vec(&normalized).map_err(|e| format!("canonical JSON serialization failed: {e}"))
}

fn canonical_json_string(value: &Value) -> Result<String, String> {
    let bytes = canonical_json_bytes(value)?;
    String::from_utf8(bytes).map_err(|e| format!("canonical JSON is not UTF-8: {e}"))
}

fn as_u64(value: &Value, field: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("compiled frame field '{field}' must be a non-negative integer"))
}

fn as_f64(value: &Value, field: &str) -> Result<f64, String> {
    value
        .as_f64()
        .ok_or_else(|| format!("compiled frame field '{field}' must be a number"))
}

fn get_field<'a>(obj: &'a Map<String, Value>, field: &str) -> Result<&'a Value, String> {
    obj.get(field)
        .ok_or_else(|| format!("compiled frame missing field '{field}'"))
}

#[derive(Debug, Clone)]
pub struct FrozenLabel {
    pub layer_id: String,
    pub label_id: String,
    pub visible: bool,
    pub alpha: f64,
    pub appearance_frame: Option<u64>,
    pub disappearance_frame: Option<u64>,
    pub payload: Value,
}

impl FrozenLabel {
    fn to_value(&self) -> Value {
        let mut map = Map::new();
        map.insert("alpha".to_string(), Value::from(self.alpha));
        map.insert(
            "appearance_frame".to_string(),
            self.appearance_frame
                .map(Value::from)
                .unwrap_or(Value::Null),
        );
        map.insert(
            "disappearance_frame".to_string(),
            self.disappearance_frame
                .map(Value::from)
                .unwrap_or(Value::Null),
        );
        map.insert("label_id".to_string(), Value::from(self.label_id.clone()));
        map.insert("layer_id".to_string(), Value::from(self.layer_id.clone()));
        map.insert("payload".to_string(), self.payload.clone());
        map.insert("visible".to_string(), Value::from(self.visible));
        Value::Object(map)
    }
}

#[derive(Debug, Clone)]
pub struct FrozenLabelSet {
    pub labels: Vec<FrozenLabel>,
    pub hash_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ChronosResidencyRecord {
    pub(crate) family_slot: u32,
    pub(crate) material_index: u32,
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) mip_level: u32,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct ChronosRingMorph {
    pub(crate) ring: u32,
    pub(crate) morph: f32,
}

#[derive(Debug)]
pub(crate) struct ChronosFrameState {
    pub(crate) frame_index: u64,
    pub(crate) base_seed: u64,
    pub(crate) frame_seed: u64,
    pub(crate) samples: u32,
    pub(crate) required_residency: Vec<ChronosResidencyRecord>,
    #[allow(dead_code)]
    pub(crate) ring_morphs: Vec<ChronosRingMorph>,
}

fn label_alpha(
    visible: bool,
    frame_index: u64,
    appearance_frame: Option<u64>,
    disappearance_frame: Option<u64>,
) -> f64 {
    if !visible
        || appearance_frame.is_some_and(|appearance| frame_index < appearance)
        || disappearance_frame.is_some_and(|disappearance| frame_index >= disappearance)
    {
        return 0.0;
    }
    let fade = LABEL_FADE_FRAMES as f64;
    let mut alpha = 1.0f64;
    if let Some(appearance) = appearance_frame {
        alpha = alpha.min((frame_index as f64 - appearance as f64 + 1.0) / fade);
    }
    if let Some(disappearance) = disappearance_frame {
        alpha = alpha.min((disappearance - frame_index) as f64 / fade);
    }
    alpha.clamp(0.0, 1.0)
}

fn parse_labels(
    scene: &Map<String, Value>,
    frame_index: u64,
) -> Result<(Vec<FrozenLabel>, Value), String> {
    let mut labels = Vec::new();
    if let Some(list) = scene.get("labels") {
        let items = list
            .as_array()
            .ok_or_else(|| "scene.labels must be an array".to_string())?;
        for (index, item) in items.iter().enumerate() {
            let obj = item
                .as_object()
                .ok_or_else(|| format!("scene.labels[{index}] must be an object"))?;
            let layer_id = get_field(obj, "layer_id")?
                .as_str()
                .ok_or_else(|| format!("scene.labels[{index}].layer_id must be a string"))?
                .to_string();
            let label_id = get_field(obj, "label_id")?
                .as_str()
                .ok_or_else(|| format!("scene.labels[{index}].label_id must be a string"))?
                .to_string();
            let visible = get_field(obj, "visible")?
                .as_bool()
                .ok_or_else(|| format!("scene.labels[{index}].visible must be a bool"))?;
            let appearance_frame = match obj.get("appearance_frame") {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.as_u64().ok_or_else(|| {
                    format!("scene.labels[{index}].appearance_frame must be null or integer")
                })?),
            };
            let disappearance_frame = match obj.get("disappearance_frame") {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.as_u64().ok_or_else(|| {
                    format!("scene.labels[{index}].disappearance_frame must be null or integer")
                })?),
            };
            if let (Some(appearance), Some(disappearance)) = (appearance_frame, disappearance_frame)
            {
                if appearance >= disappearance {
                    return Err(format!(
                        "scene.labels[{index}] appearance_frame {appearance} >= disappearance_frame {disappearance}"
                    ));
                }
            }
            let payload = normalize_value(obj.get("payload").unwrap_or(&Value::Null))?;
            let alpha = label_alpha(visible, frame_index, appearance_frame, disappearance_frame);
            labels.push(FrozenLabel {
                layer_id,
                label_id,
                visible,
                alpha,
                appearance_frame,
                disappearance_frame,
                payload,
            });
        }
    }
    labels.sort_by(|a, b| (&a.layer_id, &a.label_id).cmp(&(&b.layer_id, &b.label_id)));
    if labels
        .windows(2)
        .any(|pair| pair[0].layer_id == pair[1].layer_id && pair[0].label_id == pair[1].label_id)
    {
        return Err("scene.labels contains duplicate (layer_id, label_id) entries".to_string());
    }
    let value = Value::Array(labels.iter().map(FrozenLabel::to_value).collect());
    Ok((labels, normalize_value(&value)?))
}

fn parse_clipmap(scene: &Map<String, Value>) -> Result<(Vec<ChronosRingMorph>, Value), String> {
    let mut morphs = Vec::new();
    let mut records = Vec::new();
    let Some(descriptor) = scene.get("clipmap") else {
        return Ok((morphs, Value::Array(records)));
    };
    if descriptor.is_null() {
        return Ok((morphs, Value::Array(records)));
    }
    let obj = descriptor
        .as_object()
        .ok_or_else(|| "scene.clipmap must be an object or null".to_string())?;
    let ring_count =
        as_u64(get_field(obj, "ring_count")?, "clipmap.ring_count")?.clamp(1, 8) as u32;
    let morph_range =
        as_f64(get_field(obj, "morph_range")?, "clipmap.morph_range")?.clamp(0.0, 1.0);
    let terrain_span = as_f64(get_field(obj, "terrain_span")?, "clipmap.terrain_span")?;
    if terrain_span <= 0.0 {
        return Err("clipmap.terrain_span must be positive".to_string());
    }
    let camera_distance = as_f64(
        get_field(obj, "camera_distance")?,
        "clipmap.camera_distance",
    )?;
    if camera_distance < 0.0 {
        return Err("clipmap.camera_distance must be non-negative".to_string());
    }
    for ring in 0..ring_count {
        let outer = terrain_span / 2f64.powi(ring as i32);
        let morph = if morph_range == 0.0 {
            0.0
        } else {
            let width = (outer * morph_range).max(f64::EPSILON);
            let start = outer - width;
            ((camera_distance - start) / width).clamp(0.0, 1.0)
        };
        morphs.push(ChronosRingMorph {
            ring,
            morph: morph as f32,
        });
        let mut record = Map::new();
        record.insert("lod".to_string(), Value::from(ring));
        record.insert("morph".to_string(), Value::from(morph));
        record.insert(
            "tile_id".to_string(),
            Value::from(format!("clipmap:ring:{ring}")),
        );
        records.push(Value::Object(record));
    }
    Ok((morphs, normalize_value(&Value::Array(records))?))
}

fn parse_virtual_texture(
    scene: &Map<String, Value>,
) -> Result<(Vec<ChronosResidencyRecord>, Value), String> {
    let mut residency: BTreeSet<ChronosResidencyRecord> = BTreeSet::new();
    let Some(descriptor) = scene.get("virtual_texture") else {
        return Ok((Vec::new(), Value::Array(Vec::new())));
    };
    if descriptor.is_null() {
        return Ok((Vec::new(), Value::Array(Vec::new())));
    }
    let obj = descriptor
        .as_object()
        .ok_or_else(|| "scene.virtual_texture must be an object or null".to_string())?;
    let virtual_size = get_field(obj, "virtual_size_px")?
        .as_array()
        .ok_or_else(|| "virtual_texture.virtual_size_px must be an array".to_string())?;
    if virtual_size.len() != 2 {
        return Err("virtual_texture.virtual_size_px must have 2 elements".to_string());
    }
    let virtual_width = as_u64(&virtual_size[0], "virtual_texture.virtual_size_px[0]")? as u32;
    let tile_size = as_u64(get_field(obj, "tile_size")?, "virtual_texture.tile_size")? as u32;
    let requested_max_mip_levels = as_u64(
        get_field(obj, "max_mip_levels")?,
        "virtual_texture.max_mip_levels",
    )? as u32;
    let render_size = get_field(obj, "render_size")?
        .as_array()
        .ok_or_else(|| "virtual_texture.render_size must be an array".to_string())?;
    if render_size.len() != 2 {
        return Err("virtual_texture.render_size must have 2 elements".to_string());
    }
    let render_w = as_u64(&render_size[0], "virtual_texture.render_size[0]")? as u32;
    let render_h = as_u64(&render_size[1], "virtual_texture.render_size[1]")? as u32;
    let terrain_span = as_f64(
        get_field(obj, "terrain_span")?,
        "virtual_texture.terrain_span",
    )? as f32;
    let camera_mode = get_field(obj, "camera_mode")?
        .as_str()
        .ok_or_else(|| "virtual_texture.camera_mode must be a string".to_string())?;
    let camera_target = get_field(obj, "camera_target")?
        .as_array()
        .ok_or_else(|| "virtual_texture.camera_target must be an array".to_string())?;
    if camera_target.len() != 3 {
        return Err("virtual_texture.camera_target must have 3 elements".to_string());
    }
    let target = [
        as_f64(&camera_target[0], "virtual_texture.camera_target[0]")? as f32,
        as_f64(&camera_target[1], "virtual_texture.camera_target[1]")? as f32,
        as_f64(&camera_target[2], "virtual_texture.camera_target[2]")? as f32,
    ];
    let camera_distance = as_f64(
        get_field(obj, "camera_distance")?,
        "virtual_texture.camera_distance",
    )? as f32;
    let fov_deg = as_f64(get_field(obj, "fov_deg")?, "virtual_texture.fov_deg")? as f32;
    let virtual_height = as_u64(&virtual_size[1], "virtual_texture.virtual_size_px[1]")? as u32;
    let sources = get_field(obj, "sources")?
        .as_array()
        .ok_or_else(|| "virtual_texture.sources must be an array".to_string())?;

    if tile_size == 0 || virtual_width == 0 || virtual_height == 0 {
        return Err("virtual_texture tile_size and virtual_size_px must be non-zero".to_string());
    }
    if requested_max_mip_levels == 0 {
        return Err("virtual_texture.max_mip_levels must be non-zero".to_string());
    }
    let pages_x0 = ((virtual_width as u64 + tile_size as u64 - 1) / tile_size as u64)
        .min(u32::MAX as u64) as u32;
    let pages_y0 = ((virtual_height as u64 + tile_size as u64 - 1) / tile_size as u64)
        .min(u32::MAX as u64) as u32;
    let page_table_levels = u32::BITS - pages_x0.max(pages_y0).max(1).leading_zeros();
    let max_mip_levels = requested_max_mip_levels.min(page_table_levels).max(1);

    let is_mesh = camera_mode
        .split(':')
        .next()
        .unwrap_or(camera_mode)
        .trim()
        .eq_ignore_ascii_case("mesh");
    let (uv_min, uv_max) = if is_mesh {
        let aspect = render_w as f32 / render_h.max(1) as f32;
        let center = [
            (target[0] / terrain_span.max(1e-3)) + 0.5,
            (target[1] / terrain_span.max(1e-3)) + 0.5,
        ];
        let half_height = camera_distance.max(1.0) * (fov_deg.to_radians() * 0.5).tan();
        let half_width = half_height * aspect;
        let span_u = ((half_width * 2.5) / terrain_span.max(1e-3)).clamp(0.05, 1.0);
        let span_v = ((half_height * 2.5) / terrain_span.max(1e-3)).clamp(0.05, 1.0);
        (
            [
                (center[0] - span_u * 0.5).clamp(0.0, 1.0),
                (center[1] - span_v * 0.5).clamp(0.0, 1.0),
            ],
            [
                (center[0] + span_u * 0.5).clamp(0.0, 1.0),
                (center[1] + span_v * 0.5).clamp(0.0, 1.0),
            ],
        )
    } else {
        ([0.0f32, 0.0], [1.0f32, 1.0])
    };
    let uv_span_x = (uv_max[0] - uv_min[0]).max(1.0 / render_w.max(1) as f32);
    let uv_span_y = (uv_max[1] - uv_min[1]).max(1.0 / render_h.max(1) as f32);
    let texels_per_pixel_x = virtual_width as f32 * uv_span_x / render_w.max(1) as f32;
    let texels_per_pixel_y = virtual_height as f32 * uv_span_y / render_h.max(1) as f32;
    let texels_per_pixel = texels_per_pixel_x.max(texels_per_pixel_y).max(1.0);
    let desired_mip =
        (texels_per_pixel.log2().floor().max(0.0) as u32).min(max_mip_levels.saturating_sub(1));

    let div = 1u64.checked_shl(desired_mip).unwrap_or(u64::MAX).max(1);
    let pages_x = (((pages_x0.max(1) as u64) + div - 1) / div).max(1) as u32;
    let pages_y = (((pages_y0.max(1) as u64) + div - 1) / div).max(1) as u32;
    let start_x = ((uv_min[0] * pages_x as f32).floor() as i32).clamp(0, pages_x as i32 - 1);
    let start_y = ((uv_min[1] * pages_y as f32).floor() as i32).clamp(0, pages_y as i32 - 1);
    let end_x = ((uv_max[0] * pages_x as f32).ceil() as i32 - 1).clamp(0, pages_x as i32 - 1);
    let end_y = ((uv_max[1] * pages_y as f32).ceil() as i32 - 1).clamp(0, pages_y as i32 - 1);

    for (index, source) in sources.iter().enumerate() {
        let source_obj = source
            .as_object()
            .ok_or_else(|| format!("virtual_texture.sources[{index}] must be an object"))?;
        let family_slot = as_u64(
            get_field(source_obj, "family_slot")?,
            "virtual_texture.sources.family_slot",
        )? as u32;
        let material_index = as_u64(
            get_field(source_obj, "material_index")?,
            "virtual_texture.sources.material_index",
        )? as u32;
        for y in start_y..=end_y {
            for x in start_x..=end_x {
                let mut key = ChronosResidencyRecord {
                    family_slot,
                    material_index,
                    x: x as u32,
                    y: y as u32,
                    mip_level: desired_mip,
                };
                loop {
                    residency.insert(key);
                    if key.mip_level + 1 >= max_mip_levels {
                        break;
                    }
                    key = ChronosResidencyRecord {
                        x: key.x / 2,
                        y: key.y / 2,
                        mip_level: key.mip_level + 1,
                        ..key
                    };
                }
            }
        }
    }
    let mut records = Vec::with_capacity(residency.len());
    for key in &residency {
        let mut record = Map::new();
        record.insert("family_slot".to_string(), Value::from(key.family_slot));
        record.insert(
            "material_index".to_string(),
            Value::from(key.material_index),
        );
        record.insert("mip_level".to_string(), Value::from(key.mip_level));
        record.insert("x".to_string(), Value::from(key.x));
        record.insert("y".to_string(), Value::from(key.y));
        records.push(Value::Object(record));
    }
    Ok((
        residency.into_iter().collect(),
        normalize_value(&Value::Array(records))?,
    ))
}

fn parse_frozen_label_set(payload: &Map<String, Value>) -> Result<FrozenLabelSet, String> {
    let labels_value = get_field(payload, "labels")?;
    let items = labels_value
        .as_array()
        .ok_or_else(|| "labels must be an array".to_string())?;
    let mut labels = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let obj = item
            .as_object()
            .ok_or_else(|| format!("labels[{index}] must be an object"))?;
        let appearance_frame = match obj.get("appearance_frame") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_u64().ok_or_else(|| {
                format!("labels[{index}].appearance_frame must be null or integer")
            })?),
        };
        let disappearance_frame = match obj.get("disappearance_frame") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_u64().ok_or_else(|| {
                format!("labels[{index}].disappearance_frame must be null or integer")
            })?),
        };
        labels.push(FrozenLabel {
            layer_id: get_field(obj, "layer_id")?
                .as_str()
                .ok_or_else(|| format!("labels[{index}].layer_id must be a string"))?
                .to_string(),
            label_id: get_field(obj, "label_id")?
                .as_str()
                .ok_or_else(|| format!("labels[{index}].label_id must be a string"))?
                .to_string(),
            visible: get_field(obj, "visible")?
                .as_bool()
                .ok_or_else(|| format!("labels[{index}].visible must be a bool"))?,
            alpha: as_f64(get_field(obj, "alpha")?, "labels.alpha")?,
            appearance_frame,
            disappearance_frame,
            payload: obj.get("payload").cloned().unwrap_or(Value::Null),
        });
    }
    let labels_bytes = canonical_json_bytes(labels_value)?;
    let hash_hex = sha256_hex(&labels_bytes);
    let declared = get_field(payload, "label_set_hash")?
        .as_str()
        .ok_or_else(|| "label_set_hash must be a string".to_string())?;
    if declared != hash_hex {
        return Err("label_set_hash does not match the canonical label set".to_string());
    }
    Ok(FrozenLabelSet { labels, hash_hex })
}

fn parse_state(payload: &Map<String, Value>) -> Result<ChronosFrameState, String> {
    let frame_index = as_u64(get_field(payload, "frame_index")?, "frame_index")?;
    let base_seed = as_u64(get_field(payload, "base_seed")?, "base_seed")?;
    let derived = as_u64(get_field(payload, "frame_seed")?, "frame_seed")?;
    if derived != frame_seed(base_seed, frame_index) {
        return Err("compiled frame_seed does not match base_seed/frame_index".to_string());
    }
    let samples = as_u64(get_field(payload, "samples")?, "samples")? as u32;
    let mut required_residency = Vec::new();
    if let Some(list) = payload.get("residency") {
        let items = list
            .as_array()
            .ok_or_else(|| "residency must be an array".to_string())?;
        for (index, item) in items.iter().enumerate() {
            let obj = item
                .as_object()
                .ok_or_else(|| format!("residency[{index}] must be an object"))?;
            required_residency.push(ChronosResidencyRecord {
                family_slot: as_u64(get_field(obj, "family_slot")?, "family_slot")? as u32,
                material_index: as_u64(get_field(obj, "material_index")?, "material_index")? as u32,
                x: as_u64(get_field(obj, "x")?, "x")? as u32,
                y: as_u64(get_field(obj, "y")?, "y")? as u32,
                mip_level: as_u64(get_field(obj, "mip_level")?, "mip_level")? as u32,
            });
        }
    }
    let mut ring_morphs = Vec::new();
    if let Some(list) = payload.get("lod") {
        let items = list
            .as_array()
            .ok_or_else(|| "lod must be an array".to_string())?;
        for (index, item) in items.iter().enumerate() {
            let obj = item
                .as_object()
                .ok_or_else(|| format!("lod[{index}] must be an object"))?;
            ring_morphs.push(ChronosRingMorph {
                ring: as_u64(get_field(obj, "lod")?, "lod.lod")? as u32,
                morph: as_f64(get_field(obj, "morph")?, "lod.morph")? as f32,
            });
        }
    }
    Ok(ChronosFrameState {
        frame_index,
        base_seed,
        frame_seed: derived,
        samples,
        required_residency,
        ring_morphs,
    })
}

pub struct CompiledFrame {
    canonical: String,
    payload: Value,
    state: ChronosFrameState,
    frozen_labels: FrozenLabelSet,
    _handle: ResourceHandle,
}

impl CompiledFrame {
    pub fn canonical_json(&self) -> &str {
        &self.canonical
    }

    pub fn payload(&self) -> &Value {
        &self.payload
    }

    pub(crate) fn state(&self) -> &ChronosFrameState {
        &self.state
    }

    pub fn frozen_labels(&self) -> &FrozenLabelSet {
        &self.frozen_labels
    }

    pub fn frame_index(&self) -> u64 {
        self.state.frame_index
    }

    pub fn base_seed(&self) -> u64 {
        self.state.base_seed
    }

    pub fn frame_seed(&self) -> u64 {
        self.state.frame_seed
    }

    pub fn samples(&self) -> u32 {
        self.state.samples
    }

    pub fn from_canonical(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| format!("compiled frame JSON parse failed: {e}"))?;
        let payload = normalize_value(&value)?;
        let obj = payload
            .as_object()
            .ok_or_else(|| "compiled frame payload must be an object".to_string())?;
        let schema = get_field(obj, "schema")?
            .as_str()
            .ok_or_else(|| "compiled frame schema must be a string".to_string())?;
        if schema != COMPILED_FRAME_SCHEMA {
            return Err(format!("unsupported compiled frame schema '{schema}'"));
        }
        let state = parse_state(obj)?;
        let expected = Self::build_payload(
            state.frame_index,
            state.base_seed,
            state.samples,
            get_field(obj, "camera")?,
            get_field(obj, "scene")?,
        )?;
        let expected_canonical = canonical_json_string(&expected)?;
        if expected_canonical != json {
            return Err("compiled frame payload failed canonical/hash revalidation".to_string());
        }
        let frozen_labels = parse_frozen_label_set(obj)?;
        let handle = tracked_payload_handle(json.len() as u64)?;
        Ok(Self {
            canonical: json.to_string(),
            payload,
            state,
            frozen_labels,
            _handle: handle,
        })
    }

    fn build_payload(
        frame_index: u64,
        base_seed: u64,
        samples: u32,
        camera: &Value,
        scene_json: &Value,
    ) -> Result<Value, String> {
        if samples == 0 {
            return Err("samples must be at least 1".to_string());
        }
        if !camera.is_object() {
            return Err("camera_json must be an object".to_string());
        }
        let camera = normalize_value(camera)?;
        let scene_value = normalize_value(scene_json)?;
        let scene_obj = scene_value
            .as_object()
            .ok_or_else(|| "scene_json must be an object".to_string())?;
        let (_frozen_labels, labels_value) = parse_labels(scene_obj, frame_index)?;
        let (_ring_morphs, lod_value) = parse_clipmap(scene_obj)?;
        let (_residency, residency_value) = parse_virtual_texture(scene_obj)?;

        let camera_bytes = canonical_json_bytes(&camera)?;
        let scene_bytes = canonical_json_bytes(&scene_value)?;
        let labels_bytes = canonical_json_bytes(&labels_value)?;
        let residency_bytes = canonical_json_bytes(&residency_value)?;
        let lod_bytes = canonical_json_bytes(&lod_value)?;

        let derived_seed = frame_seed(base_seed, frame_index);
        let mut payload = Map::new();
        payload.insert("schema".to_string(), Value::from(COMPILED_FRAME_SCHEMA));
        payload.insert("frame_index".to_string(), Value::from(frame_index));
        payload.insert("base_seed".to_string(), Value::from(base_seed));
        payload.insert("frame_seed".to_string(), Value::from(derived_seed));
        payload.insert("samples".to_string(), Value::from(samples));
        payload.insert("camera".to_string(), camera);
        payload.insert("scene".to_string(), scene_value);
        payload.insert("labels".to_string(), labels_value);
        payload.insert("lod".to_string(), lod_value);
        payload.insert("residency".to_string(), residency_value);
        payload.insert(
            "camera_hash".to_string(),
            Value::from(sha256_hex(&camera_bytes)),
        );
        payload.insert(
            "scene_hash".to_string(),
            Value::from(sha256_hex(&scene_bytes)),
        );
        payload.insert(
            "label_set_hash".to_string(),
            Value::from(sha256_hex(&labels_bytes)),
        );
        payload.insert(
            "residency_hash".to_string(),
            Value::from(sha256_hex(&residency_bytes)),
        );
        payload.insert("lod_hash".to_string(), Value::from(sha256_hex(&lod_bytes)));
        payload.insert(
            "engine_revision".to_string(),
            Value::from(engine_revision()),
        );

        Ok(Value::Object(payload))
    }
}

pub struct FrameCompiler;

impl FrameCompiler {
    pub fn compile(
        frame_index: u64,
        base_seed: u64,
        samples: u32,
        camera_json: &str,
        scene_json: &str,
    ) -> Result<CompiledFrame, String> {
        let camera: Value = serde_json::from_str(camera_json)
            .map_err(|e| format!("camera_json parse failed: {e}"))?;
        let scene: Value = serde_json::from_str(scene_json)
            .map_err(|e| format!("scene_json parse failed: {e}"))?;
        Self::compile_values(frame_index, base_seed, samples, &camera, &scene)
    }

    pub fn compile_values(
        frame_index: u64,
        base_seed: u64,
        samples: u32,
        camera: &Value,
        scene: &Value,
    ) -> Result<CompiledFrame, String> {
        let payload = CompiledFrame::build_payload(frame_index, base_seed, samples, camera, scene)?;
        let canonical = canonical_json_string(&payload)?;
        let obj = payload
            .as_object()
            .ok_or_else(|| "compiled payload must be an object".to_string())?;
        let state = parse_state(obj)?;
        let frozen_labels = parse_frozen_label_set(obj)?;
        let handle = tracked_payload_handle(canonical.len() as u64)?;
        Ok(CompiledFrame {
            canonical,
            payload,
            state,
            frozen_labels,
            _handle: handle,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn camera() -> Value {
        json!({"kind": "orbit", "azimuth": 30.0, "distance": 900.0})
    }

    fn scene_with_labels(labels: Value) -> Value {
        json!({"labels": labels, "clipmap": null, "virtual_texture": null, "scene": {}})
    }

    #[test]
    fn test_compile_canonical_deterministic() {
        let scene_a = json!({
            "labels": [{"layer_id": "a", "label_id": "x", "visible": true,
                        "appearance_frame": null, "disappearance_frame": null,
                        "payload": {"z": 1, "a": 2}}],
            "clipmap": null,
            "virtual_texture": null,
            "scene": {"b": 1, "a": 2},
        });
        let scene_b: Value =
            serde_json::from_str(&serde_json::to_string(&scene_a).unwrap()).unwrap();
        let frame_a = FrameCompiler::compile_values(3, 11, 4, &camera(), &scene_a).unwrap();
        let frame_b = FrameCompiler::compile_values(3, 11, 4, &camera(), &scene_b).unwrap();
        assert_eq!(frame_a.canonical_json(), frame_b.canonical_json());
        assert!(frame_a.canonical_json().contains("\"a\":2"));
    }

    #[test]
    fn test_compile_rejects_nonfinite() {
        let err = FrameCompiler::compile(0, 0, 1, "{\"d\": 1e999}", "{}");
        match err {
            Err(message) => assert!(
                message.contains("non-finite") || message.contains("parse"),
                "{message}"
            ),
            Ok(_) => panic!("non-finite camera must be rejected"),
        }
    }

    #[test]
    fn test_compile_rejects_zero_samples_and_nonobject_camera() {
        let err = FrameCompiler::compile(0, 0, 0, "{}", "{}").err().unwrap();
        assert!(err.contains("samples"), "{err}");
        let scene = scene_with_labels(json!([])).to_string();
        let err = FrameCompiler::compile(0, 0, 1, "[1,2]", &scene)
            .err()
            .unwrap();
        assert!(err.contains("camera_json must be an object"), "{err}");
    }

    #[test]
    fn test_label_alpha_trajectory() {
        let labels = json!([{
            "layer_id": "l", "label_id": "p", "visible": true,
            "appearance_frame": 10, "disappearance_frame": 20, "payload": {}
        }]);
        let alpha_at = |frame: u64| -> f64 {
            let compiled = FrameCompiler::compile_values(
                frame,
                0,
                1,
                &camera(),
                &scene_with_labels(labels.clone()),
            )
            .unwrap();
            compiled.payload()["labels"][0]["alpha"].as_f64().unwrap()
        };
        assert!((alpha_at(10) - 1.0 / 256.0).abs() < 1e-12);
        assert_eq!(alpha_at(9), 0.0);
        assert_eq!(alpha_at(20), 0.0);
        assert_eq!(alpha_at(265), 0.0);
        assert!((alpha_at(15) - 5.0 / 256.0).abs() < 1e-12);
        let persistent = json!([{
            "layer_id": "l", "label_id": "p", "visible": true,
            "appearance_frame": 10, "disappearance_frame": null, "payload": {}
        }]);
        let compiled =
            FrameCompiler::compile_values(400, 0, 1, &camera(), &scene_with_labels(persistent))
                .unwrap();
        assert_eq!(compiled.payload()["labels"][0]["alpha"].as_f64(), Some(1.0));
        let hidden = json!([{
            "layer_id": "l", "label_id": "p", "visible": false,
            "appearance_frame": 10, "disappearance_frame": 20, "payload": {}
        }]);
        let compiled =
            FrameCompiler::compile_values(12, 0, 1, &camera(), &scene_with_labels(hidden)).unwrap();
        assert_eq!(compiled.payload()["labels"][0]["alpha"].as_f64(), Some(0.0));
        assert_eq!(
            compiled.payload()["labels"][0]["visible"].as_bool(),
            Some(false)
        );
    }

    #[test]
    fn test_labels_reject_invalid_interval_and_duplicates() {
        let bad_interval = json!([{
            "layer_id": "l", "label_id": "p", "visible": true,
            "appearance_frame": 9, "disappearance_frame": 9, "payload": {}
        }]);
        assert!(FrameCompiler::compile_values(
            0,
            0,
            1,
            &camera(),
            &scene_with_labels(bad_interval)
        )
        .is_err());
        let bad_interval = json!([{
            "layer_id": "l", "label_id": "p", "visible": true,
            "appearance_frame": 12, "disappearance_frame": 9, "payload": {}
        }]);
        assert!(FrameCompiler::compile_values(
            0,
            0,
            1,
            &camera(),
            &scene_with_labels(bad_interval)
        )
        .is_err());
        let duplicates = json!([
            {"layer_id": "l", "label_id": "p", "visible": true,
             "appearance_frame": null, "disappearance_frame": null, "payload": {}},
            {"layer_id": "l", "label_id": "p", "visible": false,
             "appearance_frame": null, "disappearance_frame": null, "payload": {}}
        ]);
        assert!(
            FrameCompiler::compile_values(0, 0, 1, &camera(), &scene_with_labels(duplicates))
                .is_err()
        );
    }

    #[test]
    fn test_frozen_label_set_matches_payload() {
        let compiled = FrameCompiler::compile_values(
            3,
            42,
            2,
            &camera(),
            &scene_with_labels(json!([{
                "layer_id": "l", "label_id": "p", "visible": true,
                "appearance_frame": 3, "disappearance_frame": 9,
                "payload": {"text": "x"}
            }])),
        )
        .unwrap();
        let set = compiled.frozen_labels();
        assert_eq!(set.labels.len(), 1);
        assert_eq!(set.labels[0].layer_id, "l");
        assert_eq!(set.labels[0].label_id, "p");
        assert_eq!(set.labels[0].alpha, 1.0 / 256.0);
        let labels_bytes = canonical_json_bytes(&compiled.payload()["labels"]).unwrap();
        assert_eq!(set.hash_hex, sha256_hex(&labels_bytes));
        assert_eq!(
            set.hash_hex,
            compiled.payload()["label_set_hash"].as_str().unwrap()
        );
        let restored = CompiledFrame::from_canonical(compiled.canonical_json()).unwrap();
        assert_eq!(restored.frozen_labels().hash_hex, set.hash_hex);
        assert_eq!(restored.frozen_labels().labels.len(), 1);
    }

    #[test]
    fn test_label_alpha_no_events() {
        let labels = json!([{
            "layer_id": "l", "label_id": "p", "visible": true,
            "appearance_frame": null, "disappearance_frame": null, "payload": {}
        }]);
        let compiled =
            FrameCompiler::compile_values(999, 0, 1, &camera(), &scene_with_labels(labels))
                .unwrap();
        assert_eq!(compiled.payload()["labels"][0]["alpha"].as_f64(), Some(1.0));
    }

    #[test]
    fn test_ring_morph_c0_threshold() {
        let descriptor = |distance: f64| {
            json!({
                "labels": [],
                "clipmap": {"ring_count": 2, "morph_range": 0.25,
                            "terrain_span": 1024.0, "camera_distance": distance},
                "virtual_texture": null,
                "scene": {},
            })
        };
        let morph_at = |distance: f64, ring: usize| -> f64 {
            let compiled =
                FrameCompiler::compile_values(0, 0, 1, &camera(), &descriptor(distance)).unwrap();
            compiled.payload()["lod"][ring]["morph"].as_f64().unwrap()
        };
        assert_eq!(morph_at(500.0, 0), 0.0);
        assert_eq!(morph_at(768.0, 0), 0.0);
        assert!((morph_at(896.0, 0) - 0.5).abs() < 1e-9);
        assert_eq!(morph_at(2000.0, 0), 1.0);
        assert_eq!(morph_at(384.0, 1), 0.0);
        assert!((morph_at(448.0, 1) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_clipmap_range_contract() {
        let descriptor = |ring_count: u64, morph_range: f64, span: f64, distance: f64| {
            json!({
                "labels": [],
                "clipmap": {"ring_count": ring_count, "morph_range": morph_range,
                            "terrain_span": span, "camera_distance": distance},
                "virtual_texture": null,
                "scene": {},
            })
        };
        let compiled =
            FrameCompiler::compile_values(0, 0, 1, &camera(), &descriptor(12, 0.25, 1024.0, 100.0))
                .unwrap();
        assert_eq!(compiled.payload()["lod"].as_array().unwrap().len(), 8);
        let compiled =
            FrameCompiler::compile_values(0, 0, 1, &camera(), &descriptor(0, 0.25, 1024.0, 100.0))
                .unwrap();
        assert_eq!(compiled.payload()["lod"].as_array().unwrap().len(), 1);
        assert!(FrameCompiler::compile_values(
            0,
            0,
            1,
            &camera(),
            &descriptor(2, 0.25, 0.0, 100.0)
        )
        .is_err());
        assert!(FrameCompiler::compile_values(
            0,
            0,
            1,
            &camera(),
            &descriptor(2, 0.25, -5.0, 100.0)
        )
        .is_err());
        assert!(FrameCompiler::compile_values(
            0,
            0,
            1,
            &camera(),
            &descriptor(2, 0.25, 1024.0, -1.0)
        )
        .is_err());
    }

    #[test]
    fn test_clipmap_zero_morph_range_hard_snap() {
        let scene = json!({
            "labels": [],
            "clipmap": {"ring_count": 3, "morph_range": 0.0,
                        "terrain_span": 1024.0, "camera_distance": 5000.0},
            "virtual_texture": null,
            "scene": {},
        });
        let compiled = FrameCompiler::compile_values(0, 0, 1, &camera(), &scene).unwrap();
        for record in compiled.payload()["lod"].as_array().unwrap() {
            assert_eq!(record["morph"].as_f64(), Some(0.0));
        }
        assert!(compiled
            .state()
            .ring_morphs
            .iter()
            .all(|morph| morph.morph == 0.0));
        let over = json!({
            "labels": [],
            "clipmap": {"ring_count": 2, "morph_range": 4.0,
                        "terrain_span": 1024.0, "camera_distance": 512.0},
            "virtual_texture": null,
            "scene": {},
        });
        let compiled = FrameCompiler::compile_values(0, 0, 1, &camera(), &over).unwrap();
        let morph = compiled.payload()["lod"][0]["morph"].as_f64().unwrap();
        let clamped = FrameCompiler::compile_values(
            0,
            0,
            1,
            &camera(),
            &json!({
                "labels": [],
                "clipmap": {"ring_count": 2, "morph_range": 1.0,
                            "terrain_span": 1024.0, "camera_distance": 512.0},
                "virtual_texture": null,
                "scene": {},
            }),
        )
        .unwrap();
        assert_eq!(
            morph,
            clamped.payload()["lod"][0]["morph"].as_f64().unwrap()
        );
    }

    #[test]
    fn test_residency_sorted_and_ancestors() {
        let scene = json!({
            "labels": [],
            "clipmap": null,
            "virtual_texture": {
                "virtual_size_px": [1024, 1024],
                "tile_size": 128,
                "max_mip_levels": 4,
                "sources": [
                    {"family_slot": 0, "material_index": 0},
                    {"family_slot": 1, "material_index": 2}
                ],
                "render_size": [256, 256],
                "terrain_span": 1000.0,
                "camera_mode": "map",
                "camera_target": [0.0, 0.0, 0.0],
                "camera_distance": 100.0,
                "fov_deg": 45.0
            },
            "scene": {},
        });
        let compiled = FrameCompiler::compile_values(0, 0, 1, &camera(), &scene).unwrap();
        let residency = compiled.payload()["residency"].as_array().unwrap();
        assert!(!residency.is_empty());
        let keys: Vec<(u64, u64, u64, u64, u64)> = residency
            .iter()
            .map(|r| {
                (
                    r["family_slot"].as_u64().unwrap(),
                    r["material_index"].as_u64().unwrap(),
                    r["x"].as_u64().unwrap(),
                    r["y"].as_u64().unwrap(),
                    r["mip_level"].as_u64().unwrap(),
                )
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        assert_eq!(compiled.state().required_residency.len(), residency.len());
    }

    #[test]
    fn test_residency_mip_count_matches_page_table() {
        let scene = json!({
            "labels": [],
            "clipmap": null,
            "virtual_texture": {
                "virtual_size_px": [512, 512],
                "tile_size": 248,
                "max_mip_levels": 8,
                "sources": [{"family_slot": 0, "material_index": 0}],
                "render_size": [256, 256],
                "terrain_span": 1000.0,
                "camera_mode": "map",
                "camera_target": [0.0, 0.0, 0.0],
                "camera_distance": 100.0,
                "fov_deg": 45.0
            },
            "scene": {},
        });
        let compiled = FrameCompiler::compile_values(0, 0, 1, &camera(), &scene).unwrap();
        let residency = compiled.payload()["residency"].as_array().unwrap();
        assert!(!residency.is_empty());
        let mips: Vec<u64> = residency
            .iter()
            .map(|record| record["mip_level"].as_u64().unwrap())
            .collect();
        assert!(mips.iter().all(|mip| *mip < 2));
        assert_eq!(mips.iter().max().copied(), Some(1));
    }

    #[test]
    fn test_from_canonical_round_trip_and_tamper() {
        let compiled = FrameCompiler::compile_values(
            7,
            42,
            2,
            &camera(),
            &scene_with_labels(json!([{
                "layer_id": "l", "label_id": "p", "visible": true,
                "appearance_frame": 3, "disappearance_frame": 9,
                "payload": {"text": "x"}
            }])),
        )
        .unwrap();
        let canonical = compiled.canonical_json().to_string();
        let restored = CompiledFrame::from_canonical(&canonical).unwrap();
        assert_eq!(restored.canonical_json(), canonical);
        assert_eq!(restored.frame_seed(), frame_seed(42, 7));

        let mut value: Value = serde_json::from_str(&canonical).unwrap();
        value["labels"][0]["alpha"] = json!(0.5);
        assert!(CompiledFrame::from_canonical(&serde_json::to_string(&value).unwrap()).is_err());
        let mut value: Value = serde_json::from_str(&canonical).unwrap();
        value["frame_seed"] = json!(12345u64);
        assert!(CompiledFrame::from_canonical(&serde_json::to_string(&value).unwrap()).is_err());
        let mut value: Value = serde_json::from_str(&canonical).unwrap();
        value["residency_hash"] = json!("deadbeef");
        assert!(CompiledFrame::from_canonical(&serde_json::to_string(&value).unwrap()).is_err());
        let mut value: Value = serde_json::from_str(&canonical).unwrap();
        value["schema"] = json!("bogus");
        assert!(CompiledFrame::from_canonical(&serde_json::to_string(&value).unwrap()).is_err());
    }

    #[test]
    fn test_known_seed_in_payload() {
        let compiled =
            FrameCompiler::compile_values(0, 0, 1, &camera(), &scene_with_labels(json!([])))
                .unwrap();
        assert_eq!(compiled.frame_seed(), 0xe220a8397b1dcdaf);
        assert_eq!(
            compiled.payload()["frame_seed"].as_u64(),
            Some(0xe220a8397b1dcdaf)
        );
    }
}
