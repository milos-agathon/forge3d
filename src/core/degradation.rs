//! Global degradation sink: every fallback/placeholder/absent-capability path
//! records a structured entry here; the RenderCertificate drains it.
use serde::Serialize;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Degradation {
    pub kind: String,
    pub name: String,
    pub consequence: String,
}

static SINK: Mutex<Vec<Degradation>> = Mutex::new(Vec::new());

/// Record a degradation. Deduplicated on (kind, name) so per-frame fallbacks
/// do not flood the certificate.
pub fn record_degradation(kind: &str, name: &str, consequence: &str) {
    let mut sink = SINK.lock().unwrap_or_else(|p| p.into_inner());
    if !sink.iter().any(|d| d.kind == kind && d.name == name) {
        sink.push(Degradation {
            kind: kind.to_string(),
            name: name.to_string(),
            consequence: consequence.to_string(),
        });
    }
}

pub fn degradations_snapshot() -> Vec<Degradation> {
    SINK.lock().unwrap_or_else(|p| p.into_inner()).clone()
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
        clear_degradations();
        record_degradation("capability_absent", "timestamp_query", "a");
        record_degradation("capability_absent", "timestamp_query", "b");
        record_degradation("cpu_fallback", "timestamp_query", "c");
        assert_eq!(degradations_snapshot().len(), 2);
        clear_degradations();
    }
}
