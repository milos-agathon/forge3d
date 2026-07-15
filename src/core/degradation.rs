//! Global degradation sink: every fallback/placeholder/absent-capability path
//! records a structured entry here; the RenderCertificate drains it.
use serde::Serialize;
use std::cell::RefCell;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Degradation {
    pub kind: String,
    pub name: String,
    pub consequence: String,
}

static SINK: Mutex<Vec<Degradation>> = Mutex::new(Vec::new());

#[cfg(test)]
pub(crate) static TEST_SINK_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static CAPTURE: RefCell<Option<Vec<Degradation>>> = const { RefCell::new(None) };
}

/// Record a degradation. Deduplicated on (kind, name) so per-frame fallbacks
/// do not flood the certificate.
pub fn record_degradation(kind: &str, name: &str, consequence: &str) {
    let entry = Degradation {
        kind: kind.to_string(),
        name: name.to_string(),
        consequence: consequence.to_string(),
    };
    let mut sink = SINK.lock().unwrap_or_else(|p| p.into_inner());
    if !sink.iter().any(|d| d.kind == kind && d.name == name) {
        sink.push(entry.clone());
    }
    drop(sink);
    CAPTURE.with(|capture| {
        if let Some(captured) = capture.borrow_mut().as_mut() {
            if !captured.iter().any(|d| d.kind == kind && d.name == name) {
                captured.push(entry);
            }
        }
    });
}

pub fn degradations_snapshot() -> Vec<Degradation> {
    SINK.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

pub fn begin_degradation_capture() {
    CAPTURE.with(|capture| *capture.borrow_mut() = Some(Vec::new()));
}

pub fn finish_degradation_capture() -> Vec<Degradation> {
    CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default())
}

pub fn abort_degradation_capture() {
    CAPTURE.with(|capture| {
        capture.borrow_mut().take();
    });
}

/// Reset the sink. Exposed to Python so tests can isolate renders.
pub fn clear_degradations() {
    SINK.lock().unwrap_or_else(|p| p.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dedup_on_kind_and_name() {
        let _lock = TEST_SINK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_degradations();
        record_degradation("capability_absent", "timestamp_query", "a");
        record_degradation("capability_absent", "timestamp_query", "b");
        record_degradation("cpu_fallback", "timestamp_query", "c");
        assert_eq!(degradations_snapshot().len(), 2);
        clear_degradations();
    }

    #[test]
    fn degradation_capture_is_render_local() {
        let _lock = TEST_SINK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_degradations();
        record_degradation("pre_render", "excluded", "before");
        record_degradation("repeated", "included", "before");

        begin_degradation_capture();
        record_degradation("repeated", "included", "during");
        record_degradation("in_render", "included", "during");
        let captured = finish_degradation_capture();

        assert!(!captured.iter().any(|d| d.name == "excluded"));
        assert!(captured
            .iter()
            .any(|d| d.kind == "repeated" && d.name == "included"));
        assert!(captured.iter().any(|d| d.kind == "in_render"));
        clear_degradations();
    }
}
