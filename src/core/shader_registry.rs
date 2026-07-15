//! CENSOR: WGSL shader-hash registry + validation error scopes.
//!
//! Every WGSL module the engine hands to naga is created through
//! [`create_labeled_shader_module`], which records the SHA256 of the EXACT
//! (already preprocessed) source keyed by a stable label. The signed
//! Render captures retain only modules whose pipelines were actually used, so
//! a RenderCertificate cannot inherit modules from earlier renders.
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
    static RENDER_CAPTURE: RefCell<Option<BTreeMap<String, String>>> = const { RefCell::new(None) };
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

pub fn abort_shader_render_capture() {
    RENDER_CAPTURE.with(|capture| {
        capture.borrow_mut().take();
    });
}

/// Record one shader module actually used by the active render.
pub fn record_shader_use(label: &str) {
    let active = RENDER_CAPTURE.with(|capture| capture.borrow().is_some());
    if !active {
        return;
    }
    let hash = REGISTRY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(label)
        .cloned();
    if let Some(hash) = hash {
        RENDER_CAPTURE.with(|capture| {
            if let Some(captured) = capture.borrow_mut().as_mut() {
                captured.insert(label.to_string(), hash);
            }
        });
    } else {
        record_degradation(
            "shader_provenance_missing",
            label,
            "a used pipeline referenced an unregistered shader module",
        );
    }
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
    // Normalize line endings BEFORE hashing and compiling: `include_str!`
    // embeds whatever the checkout produced, so a CRLF (Windows autocrlf)
    // build and an LF (CI) build of the byte-identical repository otherwise
    // hash the "same" shader differently, and committed certificates only
    // verify on the platform that generated them (found by the first
    // cross-platform run of the certificate WGSL gate). The normalized text
    // is also what naga compiles, keeping hash == compiled bytes.
    let normalized = normalize_shader_source(source);
    let final_label = register_shader_source(label, &normalized);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&final_label),
        source: wgpu::ShaderSource::Wgsl(normalized.into()),
    })
}

fn normalize_shader_source(source: &str) -> String {
    source.replace("\r\n", "\n")
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

/// Create a render pipeline under a validation error scope. Pipeline labels
/// are mandatory because they identify both degradations and certificate
/// provenance.
pub fn create_render_pipeline_scoped(
    device: &wgpu::Device,
    descriptor: &wgpu::RenderPipelineDescriptor<'_>,
) -> wgpu::RenderPipeline {
    let label = descriptor
        .label
        .expect("render pipelines require a stable label");
    with_error_scope(device, label, || device.create_render_pipeline(descriptor))
}

/// Create a compute pipeline under a validation error scope.
pub fn create_compute_pipeline_scoped(
    device: &wgpu::Device,
    descriptor: &wgpu::ComputePipelineDescriptor<'_>,
) -> wgpu::ComputePipeline {
    let label = descriptor
        .label
        .expect("compute pipelines require a stable label");
    with_error_scope(device, label, || device.create_compute_pipeline(descriptor))
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
    fn normalized_wgsl_hash_is_line_ending_independent_but_content_sensitive() {
        let lf = normalize_shader_source("@compute\nfn main() {}\n");
        let crlf = normalize_shader_source("@compute\r\nfn main() {}\r\n");
        assert_eq!(lf, crlf);
        assert_eq!(hash(&lf), hash(&crlf));
        assert_ne!(hash(&lf), hash("@compute\nfn main() { }\n"));
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
        let _lock = crate::core::degradation::TEST_SINK_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
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
    fn render_capture_contains_only_explicitly_used_shaders() {
        register_shader_source("test.render.alpha", "alpha");
        register_shader_source("test.render.beta", "beta");
        register_shader_source("test.render.unused", "unused");

        begin_shader_render_capture(&BTreeMap::new());
        record_shader_use("test.render.alpha");
        record_shader_use("test.render.beta");
        let captured = finish_shader_render_capture();

        assert_eq!(captured.len(), 2);
        assert!(captured.contains_key("test.render.alpha"));
        assert!(captured.contains_key("test.render.beta"));
        assert!(!captured.contains_key("test.render.unused"));

        begin_shader_render_capture(&BTreeMap::new());
        record_shader_use("test.render.alpha");
        let next = finish_shader_render_capture();
        assert_eq!(next.len(), 1);
        assert!(next.contains_key("test.render.alpha"));
        assert!(!next.contains_key("test.render.beta"));
    }
}
