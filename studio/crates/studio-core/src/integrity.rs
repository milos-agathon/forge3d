//! Asset integrity: is every file the project references still there, and does
//! it still look like the file the project recorded?
//!
//! The filesystem is not the core's business, so the caller supplies `stat`.
//! The shell passes `std::fs::metadata`; tests pass a map. That keeps the rules
//! — which assets matter, what counts as broken, what the UI shows — pure and
//! testable without touching disk.

use serde::{Deserialize, Serialize};

use crate::ids::AssetId;
use crate::project::{AssetKind, Project};

/// One asset's on-disk state, as checked at project open (and after relink).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetIntegrity {
    pub id: AssetId,
    pub name: String,
    pub path: String,
    pub kind: AssetKind,
    /// Size recorded when the asset was imported or relinked.
    pub expected_bytes: u64,
    /// Size on disk, when the file could be read.
    pub actual_bytes: Option<u64>,
    /// Why the file could not be read (missing, permissions, a directory),
    /// when it could not. Always present when `actual_bytes` is `None`.
    pub error: Option<String>,
}

impl AssetIntegrity {
    /// The file exists and is readable.
    pub fn exists(&self) -> bool {
        self.actual_bytes.is_some()
    }

    /// The file is the size the project recorded. A legitimate in-place
    /// replacement reports `false` — the UI treats that as "changed", not
    /// "missing", because the renderer will usually still read it.
    pub fn size_matches(&self) -> bool {
        self.actual_bytes == Some(self.expected_bytes)
    }

    /// Neither missing nor changed: nothing for the user to do.
    pub fn is_ok(&self) -> bool {
        self.exists() && self.size_matches()
    }
}

/// Check every asset in the project, in registry order.
///
/// `stat` returns the file's byte length, or a human-readable reason it could
/// not be read. A missing file is reported with its recorded path so the user
/// can relink it; nothing here mutates the project.
pub fn asset_integrity(
    project: &Project,
    mut stat: impl FnMut(&str) -> Result<u64, String>,
) -> Vec<AssetIntegrity> {
    project
        .assets
        .iter()
        .map(|asset| match stat(&asset.path) {
            Ok(actual) => AssetIntegrity {
                id: asset.id,
                name: asset.name.clone(),
                path: asset.path.clone(),
                kind: asset.kind,
                expected_bytes: asset.byte_size,
                actual_bytes: Some(actual),
                error: None,
            },
            Err(error) => AssetIntegrity {
                id: asset.id,
                name: asset.name.clone(),
                path: asset.path.clone(),
                kind: asset.kind,
                expected_bytes: asset.byte_size,
                actual_bytes: None,
                error: Some(error),
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids;
    use crate::project::{Asset, Project, UNTITLED};
    use std::collections::HashMap;

    fn project_with(assets: Vec<Asset>) -> Project {
        let mut project = Project::new(UNTITLED);
        project.assets.extend(assets);
        project
    }

    fn asset(name: &str, path: &str, byte_size: u64) -> Asset {
        Asset {
            id: ids::new_id(),
            name: name.to_owned(),
            path: path.to_owned(),
            kind: AssetKind::Dem,
            byte_size,
        }
    }

    #[test]
    fn reports_present_missing_and_changed_files_in_registry_order() {
        let dem = asset("dem.tif", "/data/dem.tif", 100);
        let texture = asset("ortho.tif", "/data/ortho.tif", 200);
        let hdri = asset("studio.hdr", "/hdri/studio.hdr", 300);
        let project = project_with(vec![dem.clone(), texture.clone(), hdri.clone()]);
        let sizes: HashMap<String, u64> = [
            (dem.path.clone(), 100), // unchanged
            (texture.path.clone(), 512), // replaced in place
                                     // hdri is absent
        ]
        .into_iter()
        .collect();

        let report = asset_integrity(&project, |path| {
            sizes
                .get(path)
                .copied()
                .ok_or_else(|| "file not found".to_owned())
        });
        assert_eq!(report.len(), 3);
        assert_eq!(report[0].id, dem.id);
        assert!(report[0].is_ok());
        assert_eq!(report[0].actual_bytes, Some(100));

        assert!(report[1].exists());
        assert!(!report[1].size_matches());
        assert!(!report[1].is_ok());
        assert_eq!(report[1].actual_bytes, Some(512));

        assert!(!report[2].exists());
        assert_eq!(report[2].error.as_deref(), Some("file not found"));

        // The report is read-only: the project still carries every asset.
        assert_eq!(project.assets.len(), 3);
    }

    #[test]
    fn an_unreadable_file_explains_itself() {
        let project = project_with(vec![asset("dem.tif", "/data/dem.tif", 1)]);
        let report = asset_integrity(&project, |_| Err("permission denied".to_owned()));
        assert!(!report[0].exists());
        assert_eq!(report[0].error.as_deref(), Some("permission denied"));
    }

    #[test]
    fn a_project_without_assets_reports_nothing() {
        let project = Project::new(UNTITLED);
        let report = asset_integrity(&project, |_| panic!("no asset should be checked"));
        assert!(report.is_empty());
    }
}
