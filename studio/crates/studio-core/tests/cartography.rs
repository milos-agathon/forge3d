//! The map's layer stack.
//!
//! The cartography is one ordered list of layers — the relief at the bottom,
//! the ramp and the imagery over it, the vector layers and their labels, the
//! plate at the front. These tests lock the four edits the panel offers (add,
//! remove, move, edit), their undo, and the ranges the engine takes.

use studio_core::cartography::{CartographyLayer, CartographyLayerKind};
use studio_core::command::Command;
use studio_core::editor::Editor;
use studio_core::ids::{new_id, AssetId};
use studio_core::project::{Asset, AssetKind, Project, TerrainSeed};

fn project_with_terrain() -> (Editor, AssetId) {
    let asset = Asset {
        id: new_id(),
        name: "luxembourg_rail.gpkg".to_owned(),
        path: "/data/luxembourg_rail.gpkg".to_owned(),
        kind: AssetKind::Vector,
        byte_size: 1_000_000,
    };
    let seed = TerrainSeed {
        title: Some("Luxembourg".to_owned()),
        name: "luxembourg_dem.tif".to_owned(),
        path: "/data/luxembourg_dem.tif".to_owned(),
        byte_size: 1,
        dimensions: Some([2929, 2705]),
        elevation_range: Some([200.0, 560.0]),
        world_origin_xz: Some([0.0, 0.0]),
        world_span_xz: Some([65_000.0, 60_000.0]),
        overlay: None,
    };
    let mut project = Project::starter("Luxembourg", Some(seed));
    project.assets.push(asset.clone());
    (Editor::new(project), asset.id)
}

fn vector_layer(asset: AssetId) -> CartographyLayer {
    CartographyLayer::new(CartographyLayerKind::Vector {
        asset,
        colour: [0.9, 0.29, 0.1, 1.0],
        width_px: 3.0,
        legend_field: None,
    })
}

#[test]
fn a_layer_is_added_removed_and_undone() {
    let (mut editor, asset) = project_with_terrain();
    let layer = vector_layer(asset);

    editor
        .apply(Command::AddCartographyLayer {
            layer: layer.clone(),
            index: None,
        })
        .expect("the layer is added");
    assert_eq!(editor.project().cartography.len(), 2);
    assert_eq!(editor.project().cartography[1].id, layer.id);

    editor
        .apply(Command::RemoveCartographyLayer { id: layer.id })
        .expect("the layer is removed");
    assert_eq!(editor.project().cartography.len(), 1);

    // Undo brings the layer back, in the place it was.
    editor.undo().expect("undo is available");
    assert_eq!(editor.project().cartography.len(), 2);
    assert_eq!(editor.project().cartography[1].id, layer.id);
}

#[test]
fn a_layer_moves_within_the_stack_and_undoes_its_place() {
    let (mut editor, asset) = project_with_terrain();
    let relief = editor.project().cartography[0].id;
    let layer = vector_layer(asset);
    editor
        .apply(Command::AddCartographyLayer {
            layer: layer.clone(),
            index: None,
        })
        .expect("the layer is added");
    let other = CartographyLayer::new(CartographyLayerKind::Plate {
        title: "Luxembourg".to_owned(),
        caption: "Luxembourg".to_owned(),
        north_arrow: true,
        scale_bar: true,
        legend: true,
    });
    editor
        .apply(Command::AddCartographyLayer {
            layer: other.clone(),
            index: None,
        })
        .expect("the plate is added");

    editor
        .apply(Command::MoveCartographyLayer {
            id: other.id,
            to: 0,
        })
        .expect("the plate moves to the front");
    assert_eq!(editor.project().cartography[0].id, other.id);
    assert_eq!(editor.project().cartography[1].id, relief);

    editor.undo().expect("undo is available");
    assert_eq!(editor.project().cartography[2].id, other.id);
}

#[test]
fn editing_a_layer_is_one_undoable_command() {
    let (mut editor, asset) = project_with_terrain();
    let layer = vector_layer(asset);
    editor
        .apply(Command::AddCartographyLayer {
            layer: layer.clone(),
            index: None,
        })
        .expect("the layer is added");

    let mut edited = layer.clone();
    edited.name = "Rail network".to_owned();
    edited.visible = false;
    if let CartographyLayerKind::Vector { colour, .. } = &mut edited.kind {
        *colour = [0.1, 0.4, 0.95, 1.0];
    }
    editor
        .apply(Command::UpdateCartographyLayer {
            id: layer.id,
            layer: edited,
        })
        .expect("the layer is edited");

    let stored = &editor.project().cartography[1];
    assert_eq!(stored.name, "Rail network");
    assert!(!stored.visible);

    editor.undo().expect("undo is available");
    let stored = &editor.project().cartography[1];
    assert_eq!(stored.name, layer.name);
    assert!(stored.visible);
}

#[test]
fn the_refusals_are_the_engine_s_own_ranges() {
    let (mut editor, asset) = project_with_terrain();
    let layer = vector_layer(asset);

    // A width the rasterizer cannot take.
    let mut wide = layer.clone();
    if let CartographyLayerKind::Vector { width_px, .. } = &mut wide.kind {
        *width_px = 50.0;
    }
    let error = editor
        .apply(Command::AddCartographyLayer {
            layer: wide,
            index: None,
        })
        .expect_err("the width must be refused")
        .to_string();
    assert!(error.contains("width"), "{error}");

    // A layer that draws an asset the project does not have.
    let mut orphan = layer.clone();
    if let CartographyLayerKind::Vector { asset, .. } = &mut orphan.kind {
        *asset = new_id();
    }
    let error = editor
        .apply(Command::AddCartographyLayer {
            layer: orphan,
            index: None,
        })
        .expect_err("the asset must be refused")
        .to_string();
    assert!(error.contains("does not exist"), "{error}");

    // A label layer with no attribute to name a feature.
    let labels = CartographyLayer::new(CartographyLayerKind::Labels {
        source: layer.id,
        field: "  ".to_owned(),
        colour: [0.08, 0.08, 0.1, 1.0],
        size_px: 14.0,
    });
    let error = editor
        .apply(Command::AddCartographyLayer {
            layer: labels,
            index: None,
        })
        .expect_err("the label field must be refused")
        .to_string();
    assert!(error.contains("label"), "{error}");
}

#[test]
fn the_stack_is_the_draw_order_and_survives_a_save() {
    let (mut editor, asset) = project_with_terrain();
    let layer = vector_layer(asset);
    editor
        .apply(Command::AddCartographyLayer {
            layer: layer.clone(),
            index: None,
        })
        .expect("the layer is added");
    let text = serde_json::to_string(&editor.project().cartography).expect("serialize");
    assert!(text.contains("\"type\":\"relief\""), "{text}");
    assert!(text.contains("\"type\":\"vector\""), "{text}");
    let back: studio_core::cartography::CartographyStack =
        serde_json::from_str(&text).expect("deserialize");
    assert_eq!(back, editor.project().cartography);
}
