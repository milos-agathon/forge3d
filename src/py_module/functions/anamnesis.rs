// Callback-free built-ins and approved plain recipes share one native memo.
// If callbacks may run, copy the entire batch with Python from the beginning:
// a hook can mutate later state fields, classes or the copy registry.
struct RecipeCopier<'py> {
    py: Python<'py>,
    deepcopy: Bound<'py, PyAny>,
    new_object: Bound<'py, PyAny>,
    atomics: Vec<Bound<'py, pyo3::types::PyType>>,
    ordinary: Vec<Bound<'py, PyAny>>,
    keep_alive: std::cell::RefCell<Vec<Bound<'py, PyAny>>>,
}

struct CopyRecursionGuard;

impl Drop for CopyRecursionGuard {
    fn drop(&mut self) {
        // SAFETY: constructed only after a successful EnterRecursiveCall, and
        // dropped before the GIL-bound copy invocation returns.
        unsafe { pyo3::ffi::Py_LeaveRecursiveCall() };
    }
}

impl<'py> RecipeCopier<'py> {
    fn can_clone_without_callbacks(
        &self,
        recipes: &Bound<'py, PyDict>,
        copy_module: &Bound<'py, pyo3::types::PyModule>,
    ) -> PyResult<bool> {
        use pyo3::types::{PyList, PyTuple, PyType};
        if !copy_module.is_exact_instance_of::<pyo3::types::PyModule>() {
            return Ok(false);
        }
        // PyFunction_Type is outside the abi3-py310 surface. Inspect the
        // runtime type without depending on CPython's private object layout.
        let types = self.py.import_bound("types")?;
        let Some(function_type) = types.dict().get_item("FunctionType")? else {
            return Ok(false);
        };
        if !self.deepcopy.get_type().is(&function_type) {
            return Ok(false);
        }
        let copy_state = copy_module.dict();
        let Some(dispatch) = copy_state.get_item("_deepcopy_dispatch")? else {
            return Ok(false);
        };
        let Some(reducers) = copy_state.get_item("dispatch_table")? else {
            return Ok(false);
        };
        if !dispatch.is_exact_instance_of::<PyDict>() || !reducers.is_exact_instance_of::<PyDict>()
        {
            return Ok(false);
        }
        let dispatch = dispatch.downcast::<PyDict>()?;
        let reducers = reducers.downcast::<PyDict>()?;
        // A custom metaclass can run Python during a registry lookup. Leave
        // those environments entirely to the original protocol.
        let type_type = self.py.get_type_bound::<PyType>();
        for (key, _) in dispatch.iter().chain(reducers.iter()) {
            if !key.get_type().is(&type_type) {
                return Ok(false);
            }
        }
        for (kind, name) in [
            (self.py.get_type_bound::<PyDict>(), "_deepcopy_dict"),
            (self.py.get_type_bound::<PyList>(), "_deepcopy_list"),
            (self.py.get_type_bound::<PyTuple>(), "_deepcopy_tuple"),
        ] {
            let Some(expected) = copy_state.get_item(name)? else {
                return Ok(false);
            };
            if !expected.get_type().is(&function_type)
                || !dispatch
                    .get_item(&kind)?
                    .is_some_and(|actual| actual.is(&expected))
            {
                return Ok(false);
            }
            if name == "_deepcopy_dict" {
                let defaults = expected.getattr("__defaults__")?;
                let Ok(defaults) = defaults.downcast::<PyTuple>() else {
                    return Ok(false);
                };
                if defaults.len() != 1 || !defaults.get_item(0)?.is(&self.deepcopy) {
                    return Ok(false);
                }
            }
        }
        let Some(atomic) = copy_state.get_item("_deepcopy_atomic")? else {
            // An unfamiliar standard-library implementation remains correct
            // through the Python path; its internals are not an API contract.
            return Ok(false);
        };
        for kind in &self.atomics {
            if !dispatch
                .get_item(kind)?
                .is_some_and(|actual| actual.is(&atomic))
            {
                return Ok(false);
            }
        }
        let numpy = self.py.import_bound("numpy")?;
        let Some(ndarray) = numpy.dict().get_item("ndarray")? else {
            return Ok(false);
        };
        if dispatch.contains(&ndarray)? || reducers.contains(&ndarray)? {
            return Ok(false);
        }
        let mut pending = vec![recipes.clone().into_any()];
        let mut visited = std::collections::HashSet::new();
        let mut checked_types = std::collections::HashSet::new();
        while let Some(value) = pending.pop() {
            let kind = value.get_type();
            if self.atomics.iter().any(|atomic| kind.is(atomic))
                || !visited.insert(value.as_ptr() as usize)
            {
                continue;
            }
            if kind.is(&self.py.get_type_bound::<PyDict>()) {
                for (key, child) in value.downcast::<PyDict>()?.iter() {
                    pending.push(key);
                    pending.push(child);
                }
            } else if kind.is(&self.py.get_type_bound::<PyList>()) {
                pending.extend(value.downcast::<PyList>()?.iter());
            } else if kind.is(&self.py.get_type_bound::<PyTuple>()) {
                pending.extend(value.downcast::<PyTuple>()?.iter());
            } else if kind.is(&ndarray) {
                let dtype = value.getattr("dtype")?;
                if dtype.getattr("hasobject")?.is_truthy()?
                    || dtype.getattr("isbuiltin")?.extract::<i32>()? != 1
                {
                    return Ok(false);
                }
            } else if self.ordinary.iter().any(|ordinary| kind.is(ordinary)) {
                if checked_types.insert(kind.as_ptr() as usize) {
                    if !kind.get_type().is(&type_type) {
                        return Ok(false);
                    }
                    let bases = kind.getattr("__bases__")?;
                    let bases = bases.downcast::<PyTuple>()?;
                    if bases.len() != 1
                        || !bases.get_item(0)?.is(&self.py.get_type_bound::<PyAny>())
                        || dispatch.contains(&kind)?
                        || reducers.contains(&kind)?
                    {
                        return Ok(false);
                    }
                    let attributes = kind.getattr("__dict__")?;
                    for hook in [
                        "__deepcopy__",
                        "__reduce__",
                        "__reduce_ex__",
                        "__getstate__",
                        "__setstate__",
                        "__slots__",
                        "__new__",
                        "__getattribute__",
                        "__getattr__",
                        "__setattr__",
                        "__delattr__",
                        "__getnewargs__",
                        "__getnewargs_ex__",
                    ] {
                        if attributes.contains(hook)? {
                            return Ok(false);
                        }
                    }
                }
                pending.push(value.getattr("__dict__")?);
            } else {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn clone_value(
        &self,
        value: &Bound<'py, PyAny>,
        memo: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::types::{PyList, PyTuple};

        let kind = value.get_type();
        if self.atomics.iter().any(|atomic| kind.is(atomic)) {
            return Ok(value.clone());
        }
        let identity = value.as_ptr() as usize;
        if let Some(owned) = memo.get_item(identity)? {
            return Ok(owned);
        }
        // Like copy._keep_alive, retain originals while custom hooks execute:
        // hooks can remove objects from their source graph and reuse addresses.
        self.keep_alive.borrow_mut().push(value.clone());
        // SAFETY: the GIL is held; the message is static and NUL terminated.
        // Respect Python's native recursion guard before entering Rust recursion.
        if unsafe {
            pyo3::ffi::Py_EnterRecursiveCall(c" while copying an ANAMNESIS recipe".as_ptr())
        } != 0
        {
            return Err(PyErr::fetch(self.py));
        }
        let _recursion = CopyRecursionGuard;
        if value.is_exact_instance_of::<PyDict>() {
            let result = PyDict::new_bound(self.py);
            memo.set_item(identity, &result)?;
            // A copy hook may mutate its containing dictionary. Python's
            // iterator raises RuntimeError; PyO3's direct dict iterator panics.
            for item in value.call_method0("items")?.iter()? {
                let item = item?;
                let key = item.get_item(0)?;
                let child = item.get_item(1)?;
                // Python deepcopy evaluates the value before the key.
                let owned = self.clone_value(&child, memo)?;
                result.set_item(self.clone_value(&key, memo)?, owned)?;
            }
            return Ok(result.into_any());
        }
        if value.is_exact_instance_of::<PyList>() {
            let result = PyList::empty_bound(self.py);
            memo.set_item(identity, &result)?;
            for child in value.iter()? {
                result.append(self.clone_value(&child?, memo)?)?;
            }
            return Ok(result.into_any());
        }
        if value.is_exact_instance_of::<PyTuple>() {
            let source = value.downcast::<PyTuple>()?;
            let children = source
                .iter()
                .map(|child| self.clone_value(&child, memo))
                .collect::<PyResult<Vec<_>>>()?;
            // A list inside the tuple may already have completed this cycle.
            if let Some(owned) = memo.get_item(identity)? {
                return Ok(owned);
            }
            let result = if source.iter().zip(&children).all(|(a, b)| a.is(b)) {
                value.clone()
            } else {
                PyTuple::new_bound(self.py, children).into_any()
            };
            memo.set_item(identity, &result)?;
            return Ok(result);
        }
        if self.ordinary.iter().any(|ordinary| kind.is(ordinary)) {
            let state = value.getattr("__dict__")?;
            if state
                .downcast::<PyDict>()?
                .keys()
                .iter()
                .all(|key| key.is_exact_instance_of::<pyo3::types::PyString>())
            {
                let result = self.new_object.call1((&kind,))?;
                memo.set_item(identity, &result)?;
                let state_identity = state.as_ptr() as usize;
                if let Some(owned) = memo.get_item(state_identity)? {
                    result.setattr("__dict__", owned)?;
                    return Ok(result);
                }
                // Copy atomic fields in one native dict copy, as the Python
                // recipe fast path does. Recurse only into mutable children.
                let owned = state.downcast::<PyDict>()?.copy()?;
                self.keep_alive.borrow_mut().push(state);
                memo.set_item(state_identity, &owned)?;
                result.setattr("__dict__", &owned)?;
                // This is our private dict copy and preflight excluded all
                // callbacks. Replacing its values cannot invalidate iteration.
                for (key, child) in owned.iter() {
                    let child_type = child.get_type();
                    if !self.atomics.iter().any(|atomic| child_type.is(atomic)) {
                        owned.set_item(key, self.clone_value(&child, memo)?)?;
                    }
                }
                return Ok(result);
            }
        }
        self.deepcopy.call1((value, memo))
    }
}

/// Internal adapter helper, deliberately absent from forge3d's public exports.
#[pyfunction]
#[pyo3(signature = (recipes, ordinary_types, *, return_preflight=false))]
pub(crate) fn _anamnesis_snapshot_recipes(
    py: Python<'_>,
    recipes: &Bound<'_, PyDict>,
    ordinary_types: &Bound<'_, pyo3::types::PyTuple>,
    return_preflight: bool,
) -> PyResult<PyObject> {
    use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyString};
    let copy_module = py.import_bound("copy")?;
    let copier = RecipeCopier {
        py,
        deepcopy: copy_module
            .dict()
            .get_item("deepcopy")?
            .unwrap_or_else(|| py.None().into_bound(py)),
        new_object: py.get_type_bound::<PyAny>().getattr("__new__")?,
        atomics: vec![
            py.None().bind(py).get_type(),
            py.get_type_bound::<PyBool>(),
            py.get_type_bound::<PyInt>(),
            py.get_type_bound::<PyFloat>(),
            py.get_type_bound::<PyComplex>(),
            py.get_type_bound::<PyString>(),
            py.get_type_bound::<PyBytes>(),
        ],
        ordinary: ordinary_types.iter().collect(),
        keep_alive: std::cell::RefCell::new(Vec::new()),
    };
    let native = copier.can_clone_without_callbacks(recipes, &copy_module)?;
    let result = PyDict::new_bound(py);
    for item in recipes.call_method0("items")?.iter()? {
        let item = item?;
        let mut pair = item.iter()?;
        let frame = pair.next().transpose()?.ok_or_else(|| {
            PyValueError::new_err("not enough values to unpack (expected 2, got 0)")
        })?;
        let recipe = pair.next().transpose()?.ok_or_else(|| {
            PyValueError::new_err("not enough values to unpack (expected 2, got 1)")
        })?;
        if pair.next().transpose()?.is_some() {
            return Err(PyValueError::new_err(
                "too many values to unpack (expected 2)",
            ));
        }
        let owned = if native {
            copier.keep_alive.borrow_mut().clear();
            // A fresh memo for every frame is part of the snapshot contract.
            copier.clone_value(&recipe, &PyDict::new_bound(py))?
        } else {
            // Match the Python comprehension, including a hook replacing
            // copy.deepcopy for a later recipe. Preflight executes no hooks.
            copy_module.getattr("deepcopy")?.call1((&recipe,))?
        };
        result.set_item(frame, owned)?;
    }
    if return_preflight {
        Ok((result, native).into_py(py))
    } else {
        Ok(result.into())
    }
}

use super::*;

// Borrow Python bytes on the hot path; retain the previous Vec<u8> extraction
// for every other accepted input, including sequences of byte integers.
fn key_bytes<'a, 'py>(value: &'a Bound<'py, PyAny>) -> PyResult<std::borrow::Cow<'a, [u8]>> {
    if let Ok(bytes) = value.downcast::<PyBytes>() {
        Ok(std::borrow::Cow::Borrowed(bytes.as_bytes()))
    } else {
        Ok(std::borrow::Cow::Owned(value.extract::<Vec<u8>>()?))
    }
}

#[pyfunction]
pub(crate) fn anamnesis_leaf_key(content: Vec<u8>) -> String {
    crate::core::anamnesis::leaf_key(&content).to_hex()
}

#[pyfunction]
pub(crate) fn anamnesis_pass_key(
    label: &str,
    pipeline_descriptor: &Bound<'_, PyAny>,
    uniform_bytes: &Bound<'_, PyAny>,
    input_keys: Vec<(String, String)>,
    capability_fingerprint: &Bound<'_, PyAny>,
    engine_fingerprint: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let pipeline_descriptor = key_bytes(pipeline_descriptor)?;
    let uniform_bytes = key_bytes(uniform_bytes)?;
    let capability_fingerprint = key_bytes(capability_fingerprint)?;
    let engine_fingerprint = key_bytes(engine_fingerprint)?;
    let inputs = input_keys
        .iter()
        .map(|(binding, value)| {
            crate::core::anamnesis::PassKey::from_hex(value)
                .map(|key| crate::core::anamnesis::InputKey::new(binding, key))
                .map_err(PyValueError::new_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(crate::core::anamnesis::key::pass_key_only(
        label,
        &pipeline_descriptor,
        &uniform_bytes,
        &inputs,
        &capability_fingerprint,
        &engine_fingerprint,
    )
    .to_hex())
}

#[pyfunction]
pub(crate) fn anamnesis_engine_fingerprint() -> PyResult<String> {
    serde_json::to_string(&crate::core::anamnesis::EngineFingerprint::current())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
pub(crate) fn anamnesis_store_verify(
    py: Python<'_>,
    root: std::path::PathBuf,
    max_bytes: u64,
) -> PyResult<PyObject> {
    let store = crate::core::anamnesis::ContentStore::new(root, max_bytes.max(1), true)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let report = store
        .verify()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let result = PyDict::new_bound(py);
    result.set_item("valid", report.valid)?;
    result.set_item("quarantined", report.quarantined)?;
    result.set_item("bytes_checked", report.bytes_checked)?;
    Ok(result.into())
}

#[pyfunction]
pub(crate) fn anamnesis_store_gc(root: std::path::PathBuf, max_bytes: u64) -> PyResult<u64> {
    let store = crate::core::anamnesis::ContentStore::new(&root, u64::MAX, false)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    store
        .gc(max_bytes)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
pub(crate) fn anamnesis_store_put_leaf(
    root: std::path::PathBuf,
    blob: Vec<u8>,
    label: &str,
    max_bytes: u64,
) -> PyResult<String> {
    let store = crate::core::anamnesis::ContentStore::new(root, max_bytes.max(1), true)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let key = crate::core::anamnesis::leaf_key(&blob);
    store
        .put_leaf(key, &blob, label)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(key.to_hex())
}

#[pyfunction]
pub(crate) fn anamnesis_store_get(
    py: Python<'_>,
    root: std::path::PathBuf,
    key: &str,
    max_bytes: u64,
) -> PyResult<Option<PyObject>> {
    let key = crate::core::anamnesis::PassKey::from_hex(key).map_err(PyValueError::new_err)?;
    let store = crate::core::anamnesis::ContentStore::new(root, max_bytes.max(1), true)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(store
        .get(key)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .map(|(blob, _)| PyBytes::new_bound(py, &blob).into()))
}

#[pyfunction]
pub(crate) fn anamnesis_restore_rgba8(
    py: Python<'_>,
    blob: Vec<u8>,
    width: u32,
    height: u32,
) -> PyResult<PyObject> {
    if width == 0 || height == 0 || blob.len() != width as usize * height as usize * 4 {
        return Err(PyValueError::new_err(
            "ANAMNESIS RGBA8 restore requires width*height*4 tightly packed bytes",
        ));
    }
    let ctx = crate::core::gpu::try_ctx()?;
    let texture = crate::core::resource_tracker::tracked_create_texture(
        ctx.device.as_ref(),
        &wgpu::TextureDescriptor {
            label: Some("anamnesis.portable.restore.rgba8"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )?;
    ctx.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &blob,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let restored = crate::read_texture_tight(
        ctx.device.as_ref(),
        ctx.queue.as_ref(),
        &texture,
        (width, height),
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyBytes::new_bound(py, &restored).into())
}

// This digest is an in-process memo key, not a content-store format.
fn plain_digest(py: Python<'_>, root: &Bound<'_, PyAny>) -> PyResult<Option<[u8; 32]>> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString, PyTuple};
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    let mut pending = vec![root.clone()];
    let mut seen = std::collections::HashSet::new();
    while let Some(value) = pending.pop() {
        let kind = value.get_type();
        if value.is_none() {
            hash.update(b"n");
        } else if kind.is(&py.get_type_bound::<PyBool>()) {
            hash.update([b'b', u8::from(value.is_truthy()?)]);
        } else if kind.is(&py.get_type_bound::<PyFloat>()) {
            hash.update(b"f");
            hash.update(value.extract::<f64>()?.to_bits().to_le_bytes());
        } else if kind.is(&py.get_type_bound::<PyInt>())
            || kind.is(&py.get_type_bound::<PyString>())
        {
            let integer = kind.is(&py.get_type_bound::<PyInt>());
            let string = if integer {
                match value.str() {
                    Ok(string) => string,
                    Err(_) => return Ok(None),
                }
            } else {
                value.downcast::<PyString>()?.clone()
            };
            let Ok(text) = string.to_str() else {
                return Ok(None);
            };
            hash.update(if integer { b"i" } else { b"s" });
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        } else if kind.is(&py.get_type_bound::<PyDict>()) {
            let mapping = value.downcast::<PyDict>()?;
            if !mapping.is_empty() && !seen.insert(value.as_ptr() as usize) {
                return Ok(None);
            }
            hash.update(b"d");
            hash.update((mapping.len() as u64).to_le_bytes());
            let mut entries = Vec::new();
            for (key, child) in mapping.iter() {
                if !key.get_type().is(&py.get_type_bound::<PyString>()) {
                    return Ok(None);
                }
                let Ok(text) = key.extract::<String>() else {
                    return Ok(None);
                };
                entries.push((text, key, child));
            }
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (_, key, child) in entries.into_iter().rev() {
                pending.push(child);
                pending.push(key);
            }
        } else if kind.is(&py.get_type_bound::<PyList>())
            || kind.is(&py.get_type_bound::<PyTuple>())
        {
            let length = value.len()?;
            if length != 0 && !seen.insert(value.as_ptr() as usize) {
                return Ok(None);
            }
            hash.update(if kind.is(&py.get_type_bound::<PyList>()) {
                b"l"
            } else {
                b"t"
            });
            hash.update((length as u64).to_le_bytes());
            for index in (0..length).rev() {
                pending.push(value.get_item(index)?);
            }
        } else {
            return Ok(None); // No repr, comparison, conversion or user callbacks.
        }
    }
    Ok(Some(hash.finalize().into()))
}

#[pyfunction]
#[pyo3(signature = (value, guard=None))]
fn _anamnesis_plain_key(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    guard: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(key) = plain_digest(py, value)? else {
        return Ok(None);
    };
    if let Some(guard) = guard {
        if plain_digest(py, guard)?.is_none() {
            return Ok(None);
        }
    }
    Ok(Some(PyBytes::new_bound(py, &key).into()))
}

/// Batch the existing size/mtime checks without per-entry Python/GIL overhead.
#[pyfunction]
fn _anamnesis_entry_signatures(
    py: Python<'_>,
    paths: Vec<std::path::PathBuf>,
) -> Vec<Option<(u64, u128, u64, u128)>> {
    py.allow_threads(|| {
        paths
            .into_iter()
            .map(|path| {
                let mut blob = None;
                let mut meta = None;
                for entry in std::fs::read_dir(path).ok()? {
                    let entry = entry.ok()?;
                    let name = entry.file_name();
                    if name == "blob" || name == "meta.json" {
                        let info = if entry.file_type().ok()?.is_symlink() {
                            // Match Python's DirEntry.stat() link-following policy.
                            std::fs::metadata(entry.path()).ok()?
                        } else {
                            entry.metadata().ok()?
                        };
                        let stamp = info
                            .modified()
                            .ok()?
                            .duration_since(std::time::UNIX_EPOCH)
                            .ok()?
                            .as_nanos();
                        let signature = (info.len(), stamp);
                        if name == "blob" {
                            blob = Some(signature);
                        } else {
                            meta = Some(signature);
                        }
                    }
                }
                let (blob, meta) = (blob?, meta?);
                Some((blob.0, blob.1, meta.0, meta.1))
            })
            .collect()
    })
}

/// Initialize the ordinary device while Python prepares its owned scene.
/// The caller's sequence timer includes submission and joining this work.
#[pyfunction]
fn _anamnesis_prepare_context(py: Python<'_>) -> PyResult<()> {
    py.allow_threads(|| crate::core::gpu::try_ctx().map(|_| ()))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[pyclass(module = "forge3d._forge3d", name = "_AnamnesisPackedSnapshot")]
struct PyPackedReadSnapshot {
    inner: crate::core::anamnesis::store::PackedReadSnapshot,
    data: Py<PyBytes>,
}

#[pymethods]
impl PyPackedReadSnapshot {
    fn verify(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<Py<PyDict>> {
        let keys = keys
            .iter()
            .map(|key| {
                crate::core::anamnesis::PassKey::from_hex(key).map_err(PyValueError::new_err)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let data = self.data.bind(py).as_bytes();
        let verified = py.allow_threads(|| self.inner.verify(&keys, data));
        let result = PyDict::new_bound(py);
        for (key, blob) in verified {
            if let Some(blob) = blob {
                result.set_item(key, PyBytes::new_bound(py, blob))?;
            } else {
                result.set_item(key, py.None())?;
            }
        }
        Ok(result.unbind())
    }
}

#[pyfunction]
fn _anamnesis_packed_snapshot(
    py: Python<'_>,
    index: &Bound<'_, PyBytes>,
    data: &Bound<'_, PyBytes>,
) -> PyResult<PyPackedReadSnapshot> {
    let index = index.as_bytes();
    let inner = py
        .allow_threads(|| crate::core::anamnesis::store::PackedReadSnapshot::from_index(index))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyPackedReadSnapshot {
        inner,
        data: data.clone().unbind(),
    })
}

#[pyfunction]
fn _anamnesis_prepare_label_scene(
    py: Python<'_>,
    width: u32,
    height: u32,
) -> PyResult<crate::scene::Scene> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err(
            "label compositor dimensions must be non-zero",
        ));
    }
    py.allow_threads(|| crate::scene::Scene::new(width, height, None, None))
}

pub(super) fn register_anamnesis_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_anamnesis_prepare_context, m)?)?;
    m.add_function(wrap_pyfunction!(_anamnesis_prepare_label_scene, m)?)?;
    m.add_function(wrap_pyfunction!(_anamnesis_packed_snapshot, m)?)?;
    m.add_class::<PyPackedReadSnapshot>()?;
    m.add_function(wrap_pyfunction!(_anamnesis_plain_key, m)?)?;
    m.add_function(wrap_pyfunction!(_anamnesis_entry_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(_anamnesis_snapshot_recipes, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_leaf_key, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_pass_key, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_engine_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_verify, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_gc, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_put_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_store_get, m)?)?;
    m.add_function(wrap_pyfunction!(anamnesis_restore_rgba8, m)?)?;
    Ok(())
}
