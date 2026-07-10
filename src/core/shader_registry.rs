//! CENSOR: WGSL shader-hash registry + validation error scopes.
//!
//! Every WGSL module the engine hands to naga is created through
//! [`create_labeled_shader_module`], which records the SHA256 of the EXACT
//! (already preprocessed) source keyed by a stable label. The signed
//! Renderer construction and render captures retain only their own registered
//! hashes, so a RenderCertificate cannot inherit modules from earlier renders.
//!
//! Hashing is deliberately separated from module creation: [`register_shader_source`]
//! is a total function over `(label, source)` that needs no GPU device, so the
//! collision/suffix logic is unit-testable under the curated cargo feature set
//! (which excludes `extension-module`). [`create_labeled_shader_module`] registers
//! then builds the wgpu module.

use crate::core::degradation::record_degradation;
use crate::core::provenance::{sha256, to_hex};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Global label -> hex-sha256(preprocessed WGSL) registry.
static REGISTRY: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

thread_local! {
    static CONSTRUCTION_CAPTURE: RefCell<Option<BTreeMap<String, String>>> = const { RefCell::new(None) };
    static RENDER_CAPTURE: RefCell<Option<BTreeMap<String, String>>> = const { RefCell::new(None) };
}

/// RAII capture of shader modules created while one renderer is constructed.
pub struct ShaderConstructionCapture {
    active: bool,
}

impl ShaderConstructionCapture {
    pub fn finish(mut self) -> BTreeMap<String, String> {
        self.active = false;
        CONSTRUCTION_CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default())
    }
}

impl Drop for ShaderConstructionCapture {
    fn drop(&mut self) {
        if self.active {
            CONSTRUCTION_CAPTURE.with(|capture| {
                capture.borrow_mut().take();
            });
        }
    }
}

/// Start recording the exact modules created by the current renderer constructor.
pub fn begin_shader_construction_capture() -> ShaderConstructionCapture {
    CONSTRUCTION_CAPTURE.with(|capture| {
        *capture.borrow_mut() = Some(BTreeMap::new());
    });
    ShaderConstructionCapture { active: true }
}

/// Seed the shader set for one render from its renderer-owned modules.
pub fn begin_shader_render_capture(seed: &BTreeMap<String, String>) {
    RENDER_CAPTURE.with(|capture| {
        *capture.borrow_mut() = Some(seed.clone());
    });
}

/// Finish and return the exact shader set associated with the current render.
pub fn finish_shader_render_capture() -> BTreeMap<String, String> {
    RENDER_CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default())
}

fn record_captured_shader(label: &str, hash: &str) {
    CONSTRUCTION_CAPTURE.with(|capture| {
        if let Some(captured) = capture.borrow_mut().as_mut() {
            captured.insert(label.to_string(), hash.to_string());
        }
    });
    RENDER_CAPTURE.with(|capture| {
        if let Some(captured) = capture.borrow_mut().as_mut() {
            captured.insert(label.to_string(), hash.to_string());
        }
    });
}

/// Pure hash-insert step, factored out so it is testable without a GPU device
/// or global state. Inserts `hash` under `label`; on a same-label/different-hash
/// collision it appends `#2`, `#3`, ... until a free (or matching) slot is found.
/// Returns `(final_label, collided)` where `collided` is true iff a distinct
/// source forced a suffix.
fn insert_shader_hash(
    registry: &mut BTreeMap<String, String>,
    label: &str,
    hash: &str,
) -> (String, bool) {
    match registry.get(label) {
        None => {
            registry.insert(label.to_string(), hash.to_string());
            (label.to_string(), false)
        }
        // Idempotent: same label + same bytes = same certificate key.
        Some(existing) if existing == hash => (label.to_string(), false),
        // Distinct source under a shared label: suffix so the certificate key
        // stays unambiguous.
        Some(_) => {
            let mut n = 2u32;
            loop {
                let candidate = format!("{label}#{n}");
                match registry.get(&candidate) {
                    None => {
                        registry.insert(candidate.clone(), hash.to_string());
                        return (candidate, true);
                    }
                    Some(existing) if existing == hash => return (candidate, true),
                    Some(_) => n += 1,
                }
            }
        }
    }
}

/// Record `sha256(source)` under `label` and return the final (possibly
/// suffixed) label. On a same-label/different-source collision, a
/// `"duplicate_shader_label"` degradation is recorded. No GPU device required.
pub fn register_shader_source(label: &str, source: &str) -> String {
    let hash = to_hex(&sha256(source.as_bytes()));
    let (final_label, collided) = {
        let mut registry = REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
        insert_shader_hash(&mut registry, label, &hash)
    };
    if collided {
        record_degradation(
            "duplicate_shader_label",
            label,
            "two distinct WGSL sources share a label; certificate keys may be ambiguous",
        );
    }
    record_captured_shader(&final_label, &hash);
    final_label
}

/// Records sha256 of the exact (preprocessed) WGSL handed to naga, keyed by
/// label, then creates the module. The certificate's wgsl_module_hashes reads
/// this registry.
pub fn create_labeled_shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    let final_label = register_shader_source(label, source);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&final_label),
        source: wgpu::ShaderSource::Wgsl(source.to_string().into()),
    })
}

/// Snapshot of the process-wide collision cache, used only for diagnostics/tests.
pub fn shader_hashes_snapshot() -> BTreeMap<String, String> {
    REGISTRY.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Pushes a Validation error scope, runs `f`, pops; a captured error records a
/// `"validation_error"` degradation named after the `label` and logs it.
pub fn with_error_scope<T>(device: &wgpu::Device, label: &str, f: impl FnOnce() -> T) -> T {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = f();
    if let Some(err) = pollster::block_on(device.pop_error_scope()) {
        let message = err.to_string();
        record_degradation("validation_error", label, &message);
        log::error!("wgpu validation error in '{label}': {message}");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(source: &str) -> String {
        to_hex(&sha256(source.as_bytes()))
    }

    #[test]
    fn two_distinct_registrations_yield_two_entries() {
        let mut registry = BTreeMap::new();
        let (a, ca) = insert_shader_hash(&mut registry, "sub.alpha", &hash("source-a"));
        let (b, cb) = insert_shader_hash(&mut registry, "sub.beta", &hash("source-b"));
        assert_eq!(a, "sub.alpha");
        assert_eq!(b, "sub.beta");
        assert!(!ca && !cb);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn one_byte_source_change_changes_hash() {
        assert_ne!(hash("void main() {}"), hash("void main() {} "));
    }

    #[test]
    fn same_label_same_source_is_idempotent() {
        let mut registry = BTreeMap::new();
        let h = hash("identical");
        let (l1, c1) = insert_shader_hash(&mut registry, "sub.same", &h);
        let (l2, c2) = insert_shader_hash(&mut registry, "sub.same", &h);
        assert_eq!(l1, "sub.same");
        assert_eq!(l2, "sub.same");
        assert!(!c1 && !c2);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn duplicate_label_different_source_is_suffixed() {
        let mut registry = BTreeMap::new();
        let (l1, c1) = insert_shader_hash(&mut registry, "sub.dup", &hash("first"));
        let (l2, c2) = insert_shader_hash(&mut registry, "sub.dup", &hash("second"));
        let (l3, c3) = insert_shader_hash(&mut registry, "sub.dup", &hash("third"));
        assert_eq!(l1, "sub.dup");
        assert_eq!(l2, "sub.dup#2");
        assert_eq!(l3, "sub.dup#3");
        assert!(!c1);
        assert!(c2 && c3);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn register_shader_source_records_collision_degradation() {
        crate::core::degradation::clear_degradations();
        // Use labels unlikely to collide with other global registrations.
        let base = "test.shader_registry.collision";
        let l1 = register_shader_source(base, "collision-source-one");
        let l2 = register_shader_source(base, "collision-source-two");
        assert_eq!(l1, base);
        assert_eq!(l2, format!("{base}#2"));
        assert!(crate::core::degradation::degradations_snapshot()
            .iter()
            .any(|d| d.kind == "duplicate_shader_label" && d.name == base));
        assert_eq!(
            shader_hashes_snapshot().get(base),
            Some(&hash("collision-source-one"))
        );
        crate::core::degradation::clear_degradations();
    }

    #[test]
    fn construction_capture_excludes_prior_process_shaders() {
        register_shader_source("test.unrelated.shader", "unrelated");

        let capture = begin_shader_construction_capture();
        register_shader_source("test.renderer.alpha", "alpha");
        register_shader_source("test.renderer.beta", "beta");
        let captured = capture.finish();

        assert_eq!(captured.len(), 2);
        assert!(captured.contains_key("test.renderer.alpha"));
        assert!(captured.contains_key("test.renderer.beta"));
        assert!(!captured.contains_key("test.unrelated.shader"));
    }
}
