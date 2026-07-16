use crate::labels::font::{FontCollection, FontRequest, TextError as FontError};
use crate::labels::msdf::{bake_msdf_atlas, bake_msdf_atlas_from_shaped, BakedMsdfAtlas};
use crate::labels::positioned::{positioned_glyphs, positioned_outlines, svg_path_data};
use crate::labels::raster::rasterize;
use crate::labels::shape::{self, Direction, FeatureSetting, ShapedText, TextError};
use numpy::{PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;
use std::ops::Range;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::UNIX_EPOCH;
use ttf_parser::Tag;

#[pyclass(name = "ShapedText", module = "forge3d._forge3d", frozen)]
#[derive(Clone)]
pub struct PyShapedText {
    pub(crate) inner: Arc<ShapedText>,
}

fn tag_string(tag: Tag) -> String {
    String::from_utf8_lossy(&tag.to_bytes()).into_owned()
}

fn diagnostic<'py>(py: Python<'py>, reason: &str) -> Bound<'py, PyDict> {
    let value = PyDict::new_bound(py);
    value.set_item("status", "diagnostic_block").unwrap();
    value.set_item("reason", reason).unwrap();
    value
}

fn exception_with_diagnostic<E>(py: Python<'_>, message: String, value: Bound<'_, PyDict>) -> PyErr
where
    E: pyo3::PyTypeInfo,
{
    let error = PyErr::new::<E, _>(message);
    let diagnostics = PyList::new_bound(py, [value]);
    let _ = error.value_bound(py).setattr("diagnostics", diagnostics);
    error
}

fn font_error_diagnostic<'py>(py: Python<'py>, error: &FontError) -> Bound<'py, PyDict> {
    match error {
        FontError::MissingGlyph { codepoint, sources } => {
            let value = diagnostic(py, "missing_glyph");
            value
                .set_item("codepoint", format!("U+{codepoint:04X}"))
                .unwrap();
            value.set_item("font_chain", sources).unwrap();
            value
        }
        FontError::InvalidFont { source, face_index } => {
            let value = diagnostic(py, "malformed_font");
            value.set_item("font", source).unwrap();
            value.set_item("face_index", face_index).unwrap();
            value
        }
        _ => diagnostic(py, "malformed_font"),
    }
}

fn text_error_diagnostic<'py>(py: Python<'py>, error: &TextError) -> Bound<'py, PyDict> {
    match error {
        TextError::Font(error) => font_error_diagnostic(py, error),
        TextError::UnsupportedLookup {
            table,
            lookup_type,
            script,
        } => {
            let value = diagnostic(py, "unsupported_lookup");
            value.set_item("table", table).unwrap();
            value.set_item("lookup_type", lookup_type).unwrap();
            value.set_item("script", tag_string(*script)).unwrap();
            value
        }
        TextError::InvalidSize => diagnostic(py, "invalid_size"),
        TextError::UnsupportedScript(script) => {
            let value = diagnostic(py, "unsupported_script");
            value.set_item("script", script).unwrap();
            value
        }
        TextError::Bidi(message) => {
            let value = diagnostic(py, "bidi_error");
            value.set_item("message", message).unwrap();
            value
        }
        TextError::MalformedOpenType(kind) if kind.contains("tag") => {
            let value = diagnostic(py, "malformed_tag");
            value.set_item("kind", kind).unwrap();
            value
        }
        TextError::OutOfBounds { .. } | TextError::MalformedOpenType(_) => {
            diagnostic(py, "malformed_font")
        }
    }
}

#[pymethods]
impl PyShapedText {
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    #[getter]
    fn size(&self) -> f32 {
        self.inner.size
    }

    #[pyo3(signature = (line_ranges = None))]
    fn to_dict(
        &self,
        py: Python<'_>,
        line_ranges: Option<Vec<(usize, usize)>>,
    ) -> PyResult<PyObject> {
        let output = PyDict::new_bound(py);
        output.set_item("text", &self.inner.text)?;
        output.set_item("size", self.inner.size)?;
        output.set_item("levels", &self.inner.levels)?;
        output.set_item("legal_breaks", &self.inner.legal_breaks)?;
        output.set_item("font_sources", self.inner.fonts.sources())?;
        output.set_item(
            "font_sha256",
            self.inner
                .face_descriptors
                .iter()
                .map(|descriptor| {
                    descriptor
                        .sha256
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>()
                })
                .collect::<Vec<_>>(),
        )?;
        let runs = PyList::empty_bound(py);
        for run in &self.inner.runs {
            let item = PyDict::new_bound(py);
            item.set_item("text_range", [run.text_range.start, run.text_range.end])?;
            item.set_item(
                "direction",
                match run.direction {
                    Direction::LeftToRight => "ltr",
                    Direction::RightToLeft => "rtl",
                },
            )?;
            item.set_item("script", tag_string(run.script))?;
            item.set_item("language", run.language.map(tag_string))?;
            item.set_item("bidi_levels", &run.bidi_levels)?;
            let glyphs = PyList::empty_bound(py);
            for glyph in &run.glyphs {
                let value = PyDict::new_bound(py);
                value.set_item("glyph_id", glyph.glyph_id)?;
                value.set_item("font_index", glyph.font_index)?;
                value.set_item("cluster", glyph.cluster)?;
                value.set_item("x_advance", glyph.x_advance)?;
                value.set_item("y_advance", glyph.y_advance)?;
                value.set_item("x_offset", glyph.x_offset)?;
                value.set_item("y_offset", glyph.y_offset)?;
                value.set_item("attached_to", glyph.attached_to)?;
                glyphs.append(value)?;
            }
            item.set_item("glyphs", glyphs)?;
            runs.append(item)?;
        }
        output.set_item("runs", runs)?;
        let ranges = concrete_line_ranges(&self.inner, line_ranges)?;
        let line_ranges = PyList::empty_bound(py);
        for range in &ranges {
            line_ranges.append([range.start, range.end])?;
        }
        output.set_item("line_ranges", line_ranges)?;
        let positioned = positioned_glyphs(&self.inner, &ranges)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let positioned_output = PyList::empty_bound(py);
        for glyph in positioned {
            let value = PyDict::new_bound(py);
            value.set_item("glyph_id", glyph.glyph_id)?;
            value.set_item("font_index", glyph.font_index)?;
            value.set_item("cluster", glyph.cluster)?;
            value.set_item("line_index", glyph.line_index)?;
            value.set_item("origin", glyph.origin)?;
            value.set_item("advance", glyph.advance)?;
            value.set_item("has_outline", glyph.path.is_some())?;
            positioned_output.append(value)?;
        }
        output.set_item("positioned_glyphs", positioned_output)?;
        Ok(output.into())
    }

    #[pyo3(signature = (line_ranges = None, precision = 4))]
    fn svg_path(
        &self,
        line_ranges: Option<Vec<(usize, usize)>>,
        precision: u8,
    ) -> PyResult<String> {
        let ranges = concrete_line_ranges(&self.inner, line_ranges)?;
        let outlines = positioned_outlines(&self.inner, &ranges)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(svg_path_data(&outlines, precision))
    }

    #[pyo3(signature = (line_ranges = None))]
    fn outline_bounds(
        &self,
        line_ranges: Option<Vec<(usize, usize)>>,
    ) -> PyResult<Option<(f32, f32, f32, f32)>> {
        let ranges = concrete_line_ranges(&self.inner, line_ranges)?;
        let outlines = positioned_outlines(&self.inner, &ranges)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(crate::labels::positioned::outline_bounds(&outlines)
            .map(|bounds| (bounds[0], bounds[1], bounds[2], bounds[3])))
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapedText(text={:?}, runs={}, size={})",
            self.inner.text,
            self.inner.runs.len(),
            self.inner.size
        )
    }
}

fn concrete_line_ranges(
    shaped: &ShapedText,
    line_ranges: Option<Vec<(usize, usize)>>,
) -> PyResult<Vec<Range<usize>>> {
    let ranges = line_ranges
        .unwrap_or_else(|| {
            default_line_ranges(&shaped.text)
                .into_iter()
                .map(|range| (range.start, range.end))
                .collect()
        })
        .into_iter()
        .map(|(start, end)| start..end)
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return Err(PyValueError::new_err("line_ranges must not be empty"));
    }
    Ok(ranges)
}

fn is_mandatory_break(character: char) -> bool {
    matches!(
        character,
        '\n' | '\r' | '\u{0085}' | '\u{2028}' | '\u{2029}'
    )
}

fn default_line_ranges(text: &str) -> Vec<Range<usize>> {
    let characters = text.chars().collect::<Vec<_>>();
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index < characters.len() {
        if !is_mandatory_break(characters[index]) {
            index += 1;
            continue;
        }
        ranges.push(start..index);
        if characters[index] == '\r' && characters.get(index + 1) == Some(&'\n') {
            index += 2;
        } else {
            index += 1;
        }
        start = index;
    }
    ranges.push(start..characters.len());
    ranges
}

fn feature_settings(
    py: Python<'_>,
    features: Option<HashMap<String, bool>>,
) -> PyResult<Vec<FeatureSetting>> {
    let mut values: Vec<_> = features.unwrap_or_default().into_iter().collect();
    values.sort_by(|left, right| left.0.cmp(&right.0));
    values
        .into_iter()
        .map(|(tag, enabled)| {
            let bytes: [u8; 4] = tag.as_bytes().try_into().map_err(|_| {
                let value = diagnostic(py, "malformed_tag");
                value.set_item("kind", "feature").unwrap();
                value.set_item("tag", &tag).unwrap();
                exception_with_diagnostic::<PyValueError>(
                    py,
                    format!("feature tag must be 4 bytes: {tag:?}"),
                    value,
                )
            })?;
            Ok(FeatureSetting::new(Tag::from_bytes(&bytes), enabled))
        })
        .collect()
}

fn load_fonts(py: Python<'_>, font_chain: Vec<String>) -> PyResult<Arc<FontCollection>> {
    if font_chain.is_empty() {
        return Err(exception_with_diagnostic::<PyValueError>(
            py,
            "font_chain must not be empty".to_owned(),
            diagnostic(py, "font_chain_required"),
        ));
    }
    type FontCacheKey = Vec<(String, u64, u128)>;
    static FONT_CACHE: LazyLock<Mutex<HashMap<FontCacheKey, Arc<FontCollection>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let mut key = Vec::with_capacity(font_chain.len());
    for path in &font_chain {
        let metadata = fs::metadata(path).map_err(|error| {
            let reason = if error.kind() == std::io::ErrorKind::NotFound {
                "font_not_found"
            } else {
                "font_io_error"
            };
            let value = diagnostic(py, reason);
            value.set_item("font", path).unwrap();
            exception_with_diagnostic::<PyValueError>(
                py,
                format!("failed to inspect font {path}: {error}"),
                value,
            )
        })?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |value| value.as_nanos());
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.into());
        key.push((canonical.display().to_string(), metadata.len(), modified));
    }
    if let Some(fonts) = FONT_CACHE.lock().unwrap().get(&key).cloned() {
        return Ok(fonts);
    }

    let requests = font_chain
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).map_err(|error| {
                let reason = if error.kind() == std::io::ErrorKind::NotFound {
                    "font_not_found"
                } else {
                    "font_io_error"
                };
                let value = diagnostic(py, reason);
                value.set_item("font", &path).unwrap();
                exception_with_diagnostic::<PyValueError>(
                    py,
                    format!("failed to read font {path}: {error}"),
                    value,
                )
            })?;
            Ok(FontRequest::from_bytes(path, bytes))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let fonts = FontCollection::load(&requests)
        .map(Arc::new)
        .map_err(|error| {
            exception_with_diagnostic::<PyValueError>(
                py,
                error.to_string(),
                font_error_diagnostic(py, &error),
            )
        })?;
    FONT_CACHE.lock().unwrap().insert(key, Arc::clone(&fonts));
    Ok(fonts)
}

#[pyfunction]
#[pyo3(name = "text_shape")]
#[pyo3(signature = (text, font_chain, size, script = None, language = None, features = None))]
pub(crate) fn text_shape_py(
    py: Python<'_>,
    text: &str,
    font_chain: Vec<String>,
    size: f32,
    script: Option<&str>,
    language: Option<&str>,
    features: Option<HashMap<String, bool>>,
) -> PyResult<PyShapedText> {
    let fonts = load_fonts(py, font_chain)?;
    let shaped = shape::shape(
        text,
        fonts,
        size,
        script,
        language,
        &feature_settings(py, features)?,
    )
    .map_err(|error| {
        exception_with_diagnostic::<PyValueError>(
            py,
            error.to_string(),
            text_error_diagnostic(py, &error),
        )
    })?;
    Ok(PyShapedText {
        inner: Arc::new(shaped),
    })
}

#[pyfunction]
#[pyo3(name = "rasterize_shaped_run")]
#[pyo3(signature = (shaped, width, height, origin = (0.0, 0.0), line_ranges = None))]
pub(crate) fn rasterize_shaped_run_py<'py>(
    py: Python<'py>,
    shaped: PyRef<'_, PyShapedText>,
    width: usize,
    height: usize,
    origin: (f32, f32),
    line_ranges: Option<Vec<(usize, usize)>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ranges = concrete_line_ranges(&shaped.inner, line_ranges)?;
    let outlines = positioned_outlines(&shaped.inner, &ranges)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array_bound(
        py,
        rasterize(&outlines, width, height, origin),
    ))
}

#[pyfunction]
#[pyo3(name = "bake_msdf_atlas")]
#[pyo3(signature = (font_chain, charset, font_size, px_range = 8.0, padding = 4))]
pub(crate) fn bake_msdf_atlas_py(
    py: Python<'_>,
    font_chain: Vec<String>,
    charset: &str,
    font_size: f32,
    px_range: f32,
    padding: u32,
) -> PyResult<PyObject> {
    if !font_size.is_finite() || font_size <= 0.0 || !px_range.is_finite() || px_range <= 0.0 {
        return Err(PyValueError::new_err(
            "font_size and px_range must be finite and positive",
        ));
    }
    let characters: Vec<_> = charset
        .chars()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if characters.is_empty() {
        return Err(PyValueError::new_err("charset must not be empty"));
    }
    let fonts = load_fonts(py, font_chain)?;
    let baked =
        bake_msdf_atlas(&fonts, &characters, font_size, px_range, padding).map_err(|error| {
            exception_with_diagnostic::<PyValueError>(
                py,
                error.to_string(),
                font_error_diagnostic(py, &error),
            )
        })?;
    baked_msdf_to_py(py, &baked)
}

#[pyfunction]
#[pyo3(name = "bake_msdf_atlas_shaped")]
#[pyo3(signature = (shaped, font_size = None, px_range = 8.0, padding = 4))]
pub(crate) fn bake_msdf_atlas_shaped_py(
    py: Python<'_>,
    shaped: PyRef<'_, PyShapedText>,
    font_size: Option<f32>,
    px_range: f32,
    padding: u32,
) -> PyResult<PyObject> {
    let font_size = font_size.unwrap_or(shaped.inner.size);
    if !font_size.is_finite() || font_size <= 0.0 || !px_range.is_finite() || px_range <= 0.0 {
        return Err(PyValueError::new_err(
            "font_size and px_range must be finite and positive",
        ));
    }
    let baked = bake_msdf_atlas_from_shaped(&shaped.inner, font_size, px_range, padding).map_err(
        |error| {
            exception_with_diagnostic::<PyValueError>(
                py,
                error.to_string(),
                font_error_diagnostic(py, &error),
            )
        },
    )?;
    baked_msdf_to_py(py, &baked)
}

fn baked_msdf_to_py(py: Python<'_>, baked: &BakedMsdfAtlas) -> PyResult<PyObject> {
    let image = ndarray::Array3::from_shape_vec(
        (baked.height as usize, baked.width as usize, 3),
        baked.image.clone(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let sdf_image = ndarray::Array2::from_shape_vec(
        (baked.height as usize, baked.width as usize),
        baked.sdf_image.clone(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let metrics = PyDict::new_bound(py);
    metrics.set_item("kind", "msdf_font_atlas")?;
    metrics.set_item("font_size", baked.font_size)?;
    metrics.set_item("line_height", baked.line_height)?;
    metrics.set_item("baseline", baked.baseline)?;
    metrics.set_item("px_range", baked.px_range)?;
    metrics.set_item("padding", baked.padding)?;
    metrics.set_item("channels", 3)?;
    metrics.set_item("width", baked.width)?;
    metrics.set_item("height", baked.height)?;
    metrics.set_item("bake_ms", baked.bake_ms)?;
    metrics.set_item("byte_count", baked.byte_count())?;
    metrics.set_item("sdf_byte_count", baked.sdf_image.len())?;
    let glyphs = PyDict::new_bound(py);
    let glyphs_by_id = PyDict::new_bound(py);
    let unicode_map = PyDict::new_bound(py);
    for glyph in &baked.glyphs {
        let value = PyDict::new_bound(py);
        value.set_item("x", glyph.x)?;
        value.set_item("y", glyph.y)?;
        value.set_item("w", glyph.width)?;
        value.set_item("h", glyph.height)?;
        value.set_item("ox", glyph.offset_x)?;
        value.set_item("oy", glyph.offset_y)?;
        value.set_item("adv", glyph.advance)?;
        value.set_item("font_index", glyph.font_index)?;
        value.set_item("glyph_id", glyph.glyph_id)?;
        let identity = format!("{}:{}", glyph.font_index, glyph.glyph_id);
        glyphs_by_id.set_item(&identity, &value)?;
        if glyph.codepoint != 0 {
            glyphs.set_item(glyph.codepoint.to_string(), &value)?;
            unicode_map.set_item(glyph.codepoint.to_string(), identity)?;
        }
    }
    metrics.set_item("glyphs", glyphs)?;
    metrics.set_item("glyphs_by_id", glyphs_by_id)?;
    metrics.set_item("unicode_map", unicode_map)?;
    let output = PyDict::new_bound(py);
    output.set_item("image", PyArray3::from_owned_array_bound(py, image))?;
    output.set_item("sdf_image", PyArray2::from_owned_array_bound(py, sdf_image))?;
    output.set_item("metrics", metrics)?;
    Ok(output.into())
}

#[cfg(test)]
mod tests {
    use super::{default_line_ranges, text_error_diagnostic};
    use crate::labels::font::{FontCollection, FontRequest};
    use crate::labels::positioned::positioned_glyphs;
    use crate::labels::shape;
    use crate::labels::shape::TextError;
    use pyo3::prelude::*;
    use std::sync::Arc;
    use ttf_parser::Tag;

    #[test]
    fn default_line_ranges_split_on_mandatory_break_controls() {
        assert_eq!(default_line_ranges("בד\nשב"), vec![0..2, 3..5]);
        assert_eq!(default_line_ranges("A\r\nB"), vec![0..1, 3..4]);
        assert_eq!(default_line_ranges("A"), vec![0..1]);
    }

    #[test]
    fn mandatory_break_default_ranges_produce_two_positioned_lines() {
        let fonts = Arc::new(
            FontCollection::load(&[FontRequest::from_bytes(
                "NotoSansHebrew-subset.ttf",
                include_bytes!("../../assets/fonts/NotoSansHebrew-subset.ttf").to_vec(),
            )])
            .unwrap(),
        );
        let shaped = shape::shape("בד\nשב", fonts, 16.0, None, None, &[]).unwrap();
        let ranges = default_line_ranges(&shaped.text);
        let positioned = positioned_glyphs(&shaped, &ranges).unwrap();

        assert_eq!(ranges, vec![0..2, 3..5]);
        assert!(positioned.iter().all(|glyph| glyph.cluster != 4));
        assert_eq!(
            positioned
                .iter()
                .map(|glyph| glyph.line_index)
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 1]
        );
        assert_ne!(positioned[0].origin[1], positioned[2].origin[1]);
    }

    #[test]
    fn all_non_font_text_error_families_have_stable_reasons() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let cases = [
                (
                    TextError::UnsupportedLookup {
                        table: "GSUB",
                        lookup_type: 8,
                        script: Tag::from_bytes(b"arab"),
                    },
                    "unsupported_lookup",
                ),
                (TextError::Bidi("bad levels".to_owned()), "bidi_error"),
                (
                    TextError::MalformedOpenType("language tag"),
                    "malformed_tag",
                ),
                (
                    TextError::OutOfBounds {
                        offset: 4,
                        length: 8,
                    },
                    "malformed_font",
                ),
            ];
            for (error, expected) in cases {
                let value = text_error_diagnostic(py, &error);
                let reason: String = value
                    .get_item("reason")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                assert_eq!(reason, expected);
            }
        });
    }
}
