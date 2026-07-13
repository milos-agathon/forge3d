#![cfg(feature = "extension-module")]

use crate::labels::font::{FontCollection, FontRequest, TextError as FontError};
use crate::labels::shape::{self, Direction, FeatureSetting, ShapedText, TextError};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
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

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let output = PyDict::new_bound(py);
        output.set_item("text", &self.inner.text)?;
        output.set_item("size", self.inner.size)?;
        output.set_item("levels", &self.inner.levels)?;
        output.set_item("legal_breaks", &self.inner.legal_breaks)?;
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
                value.set_item("x_offset", glyph.x_offset)?;
                glyphs.append(value)?;
            }
            item.set_item("glyphs", glyphs)?;
            runs.append(item)?;
        }
        output.set_item("runs", runs)?;
        Ok(output.into())
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
    if font_chain.is_empty() {
        return Err(exception_with_diagnostic::<PyValueError>(
            py,
            "font_chain must not be empty".to_owned(),
            diagnostic(py, "font_chain_required"),
        ));
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
    let fonts = Arc::new(FontCollection::load(&requests).map_err(|error| {
        exception_with_diagnostic::<PyValueError>(
            py,
            error.to_string(),
            font_error_diagnostic(py, &error),
        )
    })?);
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
pub(crate) fn rasterize_shaped_run_py(py: Python<'_>) -> PyResult<()> {
    let value = diagnostic(py, "littera_rendering_deferred");
    value.set_item("operation", "rasterize_shaped_run")?;
    Err(exception_with_diagnostic::<PyNotImplementedError>(
        py,
        "LITTERA analytic rasterization is implemented in Task 8".to_owned(),
        value,
    ))
}

#[pyfunction]
#[pyo3(name = "bake_msdf_atlas")]
pub(crate) fn bake_msdf_atlas_py(py: Python<'_>) -> PyResult<()> {
    let value = diagnostic(py, "littera_rendering_deferred");
    value.set_item("operation", "bake_msdf_atlas")?;
    Err(exception_with_diagnostic::<PyNotImplementedError>(
        py,
        "LITTERA MSDF atlas baking is implemented in Task 9".to_owned(),
        value,
    ))
}

#[cfg(test)]
mod tests {
    use super::text_error_diagnostic;
    use crate::labels::shape::TextError;
    use pyo3::prelude::*;
    use ttf_parser::Tag;

    #[test]
    fn all_non_font_text_error_families_have_stable_reasons() {
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
