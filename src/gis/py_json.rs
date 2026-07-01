use serde_json::{Map, Number, Value};

use crate::gis::error::GisError;
use crate::gis::types::RasterWarning;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods, PyList, PyTuple};
use pyo3::IntoPy;

pub(crate) fn warnings_to_py(py: Python<'_>, warnings: &[RasterWarning]) -> PyResult<PyObject> {
    let mut items = Vec::with_capacity(warnings.len());
    for warning in warnings {
        let dict = PyDict::new_bound(py);
        dict.set_item("code", warning.code.clone())?;
        dict.set_item("message", warning.message.clone())?;
        dict.set_item("field", warning.field.clone())?;
        items.push(dict.into_py(py));
    }
    Ok(PyList::new_bound(py, items).into_py(py))
}

pub(crate) fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(Value::Bool(value));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = value.extract::<u64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = value.extract::<f64>() {
        return Number::from_f64(value).map(Value::Number).ok_or_else(|| {
            GisError::InvalidArgument("JSON numbers must be finite".to_string()).into()
        });
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(Value::String(value));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut out = Map::new();
        for (key, item) in dict.iter() {
            let key = key.extract::<String>().map_err(|_| {
                GisError::InvalidArgument("JSON object keys must be strings".to_string())
            })?;
            if key == "vector_info" {
                continue;
            }
            out.insert(key, py_to_json(&item)?);
        }
        return Ok(Value::Object(out));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        return list
            .iter()
            .map(|item| py_to_json(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        return tuple
            .iter()
            .map(|item| py_to_json(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    Err(GisError::InvalidArgument(
        "source must be a vector path, VectorInfo, or GeoJSON-like dict".to_string(),
    )
    .into())
}

pub(crate) fn py_to_json_strict(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(Value::Bool(value));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = value.extract::<u64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = value.extract::<f64>() {
        return Number::from_f64(value).map(Value::Number).ok_or_else(|| {
            GisError::InvalidGeometry("invalid_geometry: JSON numbers must be finite".to_string())
                .into()
        });
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(Value::String(value));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut out = Map::new();
        for (key, item) in dict.iter() {
            let key = key.extract::<String>().map_err(|_| {
                GisError::InvalidGeometry(
                    "invalid_geometry: JSON object keys must be strings".to_string(),
                )
            })?;
            if key == "vector_info" {
                continue;
            }
            out.insert(key, py_to_json_strict(&item)?);
        }
        return Ok(Value::Object(out));
    }
    if let Ok(list) = value.downcast::<PyList>() {
        return list
            .iter()
            .map(|item| py_to_json_strict(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        return tuple
            .iter()
            .map(|item| py_to_json_strict(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    Err(GisError::InvalidGeometry(
        "invalid_geometry: GeoJSON values must be JSON-compatible".to_string(),
    )
    .into())
}

pub(crate) fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => Ok(value.into_py(py)),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                Ok(value.into_py(py))
            } else if let Some(value) = number.as_u64() {
                Ok(value.into_py(py))
            } else if let Some(value) = number.as_f64() {
                Ok(value.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(value) => Ok(value.clone().into_py(py)),
        Value::Array(items) => {
            let items = items
                .iter()
                .map(|item| json_to_py(py, item))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new_bound(py, items).into_py(py))
        }
        Value::Object(items) => {
            let dict = PyDict::new_bound(py);
            for (key, item) in items {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            Ok(dict.into_py(py))
        }
    }
}
