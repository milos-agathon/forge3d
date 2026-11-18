//! Stable identifiers.
//!
//! Every project, scene, and object carries a UUID that survives save, load,
//! undo, and recovery. IDs are generated once, at creation, and never derived
//! from display names or list positions.

use uuid::Uuid;

pub type ProjectId = Uuid;
pub type SceneId = Uuid;
pub type ObjectId = Uuid;
pub type AssetId = Uuid;

/// A fresh random (v4) identifier.
pub fn new_id() -> Uuid {
    Uuid::new_v4()
}

/// Parse an identifier that arrived as text (Tauri IPC, project file).
pub fn parse_id(text: &str) -> Result<Uuid, uuid::Error> {
    Uuid::parse_str(text)
}

/// First eight characters of an ID — for log lines and diagnostics only.
pub fn short_id(id: &Uuid) -> String {
    let text = id.to_string();
    text[..8].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_short_form_is_stable() {
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
        assert_eq!(short_id(&a).len(), 8);
        assert_eq!(parse_id(&a.to_string()).unwrap(), a);
    }
}
