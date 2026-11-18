//! Environment (HDRI) acceptance: a project-scoped Radiance `.hdr` that both
//! the Material preview and Rendered frames light with.
//!
//! The bytes never live in the project: it stores an asset reference, the
//! viewport reads them for its PMREM, and the engine reads the path for
//! `IBL.from_hdr`. These tests pin the model half — recognition, validation,
//! undo, persistence, and the handoff the shell turns into a render request.

use studio_core::command::Command;
use studio_core::editor::Editor;
use studio_core::ids;
use studio_core::persist::{load, save_atomic};
use studio_core::project::{Asset, AssetKind, Project, RenderSettings, UNTITLED};
use studio_core::workspace::{
    environment_handoff, handoff, parameter_capabilities, CapabilityLevel,
};
use studio_core::Workspace;

fn hdr_asset() -> Asset {
    Asset {
        id: ids::new_id(),
        name: "studio_sunrise.hdr".to_owned(),
        path: "/hdri/studio_sunrise.hdr".to_owned(),
        kind: AssetKind::Hdr,
        byte_size: 24_000_000,
    }
}

#[test]
fn only_radiance_hdr_files_are_environment_assets() {
    assert_eq!(
        AssetKind::from_path(std::path::Path::new("/hdri/a.hdr")),
        Some(AssetKind::Hdr)
    );
    assert_eq!(
        AssetKind::from_path(std::path::Path::new("/hdri/A.HDR")),
        Some(AssetKind::Hdr)
    );
    // The viewport loader and the engine's `IBL.from_hdr` both read RGBE;
    // claiming EXR support would be a silent failure, so it is not an asset
    // kind at all.
    assert_eq!(
        AssetKind::from_path(std::path::Path::new("/hdri/a.exr")),
        None
    );
    assert_eq!(AssetKind::Hdr.extension(), "hdr");
    assert!(!AssetKind::Hdr.is_raster());
}

#[test]
fn the_environment_is_a_reversible_persisted_setting() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("environment.studio3d");
    let mut editor = Editor::new(Project::new(UNTITLED));
    let asset = hdr_asset();
    editor
        .apply(Command::AddAsset {
            asset: asset.clone(),
        })
        .unwrap();

    let render = RenderSettings {
        environment: Some(asset.id),
        ..editor.project().render.clone()
    };
    editor.apply(Command::SetRenderSettings { render }).unwrap();
    assert_eq!(
        editor.project().environment_path(),
        Some("/hdri/studio_sunrise.hdr")
    );
    // Dependency accounting must see it, or the Assets panel would offer to
    // delete a file the renderer is using.
    assert_eq!(editor.project().asset_usage(asset.id), 1);
    assert!(editor.project().used_assets().contains(&asset.id));

    // Undo clears the reference; redo restores it.
    editor.undo().unwrap();
    assert_eq!(editor.project().render.environment, None);
    assert_eq!(editor.project().environment_path(), None);
    editor.redo().unwrap();
    assert_eq!(editor.project().render.environment, Some(asset.id));

    // Save → reload keeps both the asset and the reference.
    save_atomic(&path, editor.project()).unwrap();
    let reloaded = load(&path).unwrap();
    assert_eq!(reloaded.render.environment, Some(asset.id));
    assert_eq!(
        reloaded.environment_path(),
        Some("/hdri/studio_sunrise.hdr")
    );
}

#[test]
fn environment_validation_rejects_dangling_and_wrong_kinds() {
    let mut editor = Editor::new(Project::new(UNTITLED));

    // An id the project does not carry.
    let dangling = RenderSettings {
        environment: Some(ids::new_id()),
        ..editor.project().render.clone()
    };
    let error = editor
        .apply(Command::SetRenderSettings { render: dangling })
        .expect_err("a dangling environment must be rejected")
        .to_string();
    assert!(error.contains("missing asset"), "{error}");

    // A real asset that is not an HDRI.
    let texture = Asset {
        id: ids::new_id(),
        name: "ortho.tif".to_owned(),
        path: "/data/ortho.tif".to_owned(),
        kind: AssetKind::Orthophoto,
        byte_size: 1024,
    };
    editor
        .apply(Command::AddAsset {
            asset: texture.clone(),
        })
        .unwrap();
    let wrong_kind = RenderSettings {
        environment: Some(texture.id),
        ..editor.project().render.clone()
    };
    let error = editor
        .apply(Command::SetRenderSettings { render: wrong_kind })
        .expect_err("a texture is not an environment")
        .to_string();
    assert!(error.contains("expected an .hdr"), "{error}");

    // The rejected edits left the setting untouched.
    assert_eq!(editor.project().render.environment, None);
}

#[test]
fn the_environment_asset_cannot_be_removed_while_used() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    let asset = hdr_asset();
    editor
        .apply(Command::AddAsset {
            asset: asset.clone(),
        })
        .unwrap();
    let render = RenderSettings {
        environment: Some(asset.id),
        ..editor.project().render.clone()
    };
    editor.apply(Command::SetRenderSettings { render }).unwrap();

    let error = editor
        .apply(Command::RemoveAsset { id: asset.id })
        .expect_err("the environment still references this asset")
        .to_string();
    assert!(error.contains("still used"), "{error}");

    // Clearing the environment releases it.
    editor
        .apply(Command::SetRenderSettings {
            render: RenderSettings {
                environment: None,
                ..editor.project().render.clone()
            },
        })
        .unwrap();
    editor.apply(Command::RemoveAsset { id: asset.id }).unwrap();
    assert!(editor.project().asset(asset.id).is_none());
}

#[test]
fn the_handoff_carries_the_environment_path() {
    let mut project = Project::new(UNTITLED);
    let asset = hdr_asset();
    project.assets.push(asset.clone());
    project.render.environment = Some(asset.id);

    let forged = handoff(&project, 3, Workspace::Forge3d);
    let environment = forged.environment.expect("the handoff must carry the HDRI");
    assert_eq!(environment.id, asset.id);
    assert_eq!(environment.path, "/hdri/studio_sunrise.hdr");
    assert_eq!(environment.kind, AssetKind::Hdr);
    assert_eq!(
        environment_handoff(&project).map(|entry| entry.name),
        Some("studio_sunrise.hdr".to_owned())
    );

    // No environment → nothing to hand over (the engine keeps its studio sky).
    project.render.environment = None;
    assert!(handoff(&project, 4, Workspace::Forge3d)
        .environment
        .is_none());
}

#[test]
fn the_environment_is_a_fully_supported_parameter() {
    let table = parameter_capabilities();
    assert_eq!(table.environment.len(), 1);
    let row = &table.environment[0];
    assert_eq!(row.level, CapabilityLevel::Full, "{row:?}");
    assert!(row.detail.contains("Material preview"), "{row:?}");
    assert!(row.detail.contains("Rendered"), "{row:?}");
}

#[test]
fn final_still_extras_default_off_and_round_trip() {
    let render = RenderSettings::default();
    assert!(!render.still_denoise);
    assert!(!render.still_aov.any_enabled());

    // A project saved before these keys existed loads with them off, and the
    // serialized project carries the new keys.
    let mut editor = Editor::new(Project::new(UNTITLED));
    let render = RenderSettings {
        still_aov: studio_core::StillAov {
            albedo: true,
            normal: true,
            depth: false,
        },
        still_denoise: true,
        ..RenderSettings::default()
    };
    editor.apply(Command::SetRenderSettings { render }).unwrap();

    let json = serde_json::to_string(editor.project()).unwrap();
    let parsed: Project = serde_json::from_str(&json).unwrap();
    assert!(parsed.render.still_aov.albedo);
    assert!(parsed.render.still_aov.normal);
    assert!(!parsed.render.still_aov.depth);
    assert!(parsed.render.still_denoise);
}

#[test]
fn the_hdri_background_intensity_is_validated() {
    let mut editor = Editor::new(Project::new(UNTITLED));
    for value in [-0.1f32, 1.5, f32::NAN] {
        let render = RenderSettings {
            environment_background_intensity: value,
            ..editor.project().render.clone()
        };
        let error = editor
            .apply(Command::SetRenderSettings { render })
            .expect_err("out-of-range background intensity")
            .to_string();
        assert!(error.contains("background intensity"), "{value}: {error}");
    }

    let render = RenderSettings {
        environment_background_intensity: 0.35,
        ..editor.project().render.clone()
    };
    editor.apply(Command::SetRenderSettings { render }).unwrap();
    assert_eq!(
        editor.project().render.environment_background_intensity,
        0.35
    );
    editor.undo().unwrap();
    assert_eq!(
        editor.project().render.environment_background_intensity,
        1.0
    );
}
