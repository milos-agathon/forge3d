#![cfg(feature = "extension-module")]

use crate::labels::font::{FontCollection, FontRequest};
use crate::labels::shape::{self, Direction, FeatureSetting, ShapedText};
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

fn feature_settings(features: Option<HashMap<String, bool>>) -> PyResult<Vec<FeatureSetting>> {
    let mut values: Vec<_> = features.unwrap_or_default().into_iter().collect();
    values.sort_by(|left, right| left.0.cmp(&right.0));
    values
        .into_iter()
        .map(|(tag, enabled)| {
            let bytes: [u8; 4] = tag.as_bytes().try_into().map_err(|_| {
                PyValueError::new_err(format!("feature tag must be 4 bytes: {tag:?}"))
            })?;
            Ok(FeatureSetting::new(Tag::from_bytes(&bytes), enabled))
        })
        .collect()
}

#[pyfunction]
#[pyo3(name = "text_shape")]
#[pyo3(signature = (text, font_chain, size, script = None, language = None, features = None))]
pub(crate) fn text_shape_py(
    text: &str,
    font_chain: Vec<String>,
    size: f32,
    script: Option<&str>,
    language: Option<&str>,
    features: Option<HashMap<String, bool>>,
) -> PyResult<PyShapedText> {
    if font_chain.is_empty() {
        return Err(PyValueError::new_err("font_chain must not be empty"));
    }
    let requests = font_chain
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).map_err(|error| {
                PyValueError::new_err(format!("failed to read font {path}: {error}"))
            })?;
            Ok(FontRequest::from_bytes(path, bytes))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let fonts = Arc::new(
        FontCollection::load(&requests)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
    );
    let shaped = shape::shape(
        text,
        fonts,
        size,
        script,
        language,
        &feature_settings(features)?,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyShapedText {
        inner: Arc::new(shaped),
    })
}

#[pyfunction]
#[pyo3(name = "rasterize_shaped_run")]
pub(crate) fn rasterize_shaped_run_py() -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "LITTERA analytic rasterization is implemented in Task 8",
    ))
}

#[pyfunction]
#[pyo3(name = "bake_msdf_atlas")]
pub(crate) fn bake_msdf_atlas_py() -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "LITTERA MSDF atlas baking is implemented in Task 9",
    ))
}
