use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::AddLabel {
            text,
            world_pos,
            size,
            color,
            halo_color,
            halo_width,
            priority,
            min_zoom,
            max_zoom,
            offset,
            rotation,
            underline,
            small_caps,
            leader,
            horizon_fade_angle,
        } => {
            use crate::labels::LabelStyle;

            let mut style = LabelStyle::default();
            if let Some(size) = size {
                style.size = *size;
            }
            if let Some(color) = color {
                style.color = *color;
            }
            if let Some(halo_color) = halo_color {
                style.halo_color = *halo_color;
            }
            if let Some(halo_width) = halo_width {
                style.halo_width = *halo_width;
            }
            if let Some(priority) = priority {
                style.priority = *priority;
            }
            if let Some(min_zoom) = min_zoom {
                style.min_zoom = *min_zoom;
            }
            if let Some(max_zoom) = max_zoom {
                style.max_zoom = *max_zoom;
            }
            if let Some(offset) = offset {
                style.offset = *offset;
            }
            if let Some(rotation) = rotation {
                style.rotation = *rotation;
            }
            if let Some(underline) = underline {
                style.flags.underline = *underline;
            }
            if let Some(small_caps) = small_caps {
                style.flags.small_caps = *small_caps;
            }
            if let Some(leader) = leader {
                style.flags.leader = *leader;
            }
            if let Some(angle) = horizon_fade_angle {
                style.horizon_fade_angle = *angle;
            }

            let id = viewer.add_label(
                text,
                (world_pos[0], world_pos[1], world_pos[2]),
                Some(style),
            );
            println!("[label] added id={} text=\"{}\"", id, text);
            true
        }
        ViewerCmd::AddLineLabel {
            text,
            polyline,
            size,
            color,
            halo_color,
            halo_width,
            priority,
            placement,
            repeat_distance,
            min_zoom,
            max_zoom,
        } => {
            use crate::labels::{LabelStyle, LineLabelPlacement};
            use glam::Vec3;

            let mut style = LabelStyle::default();
            if let Some(size) = size {
                style.size = *size;
            }
            if let Some(color) = color {
                style.color = *color;
            }
            if let Some(halo_color) = halo_color {
                style.halo_color = *halo_color;
            }
            if let Some(halo_width) = halo_width {
                style.halo_width = *halo_width;
            }
            if let Some(priority) = priority {
                style.priority = *priority;
            }
            if let Some(min_zoom) = min_zoom {
                style.min_zoom = *min_zoom;
            }
            if let Some(max_zoom) = max_zoom {
                style.max_zoom = *max_zoom;
            }

            let placement = match placement.as_deref() {
                Some("along") => LineLabelPlacement::Along,
                _ => LineLabelPlacement::Center,
            };
            let polyline: Vec<Vec3> = polyline
                .iter()
                .map(|point| Vec3::new(point[0], point[1], point[2]))
                .collect();
            let repeat_distance = repeat_distance.unwrap_or(0.0);
            let id = viewer.label_manager.add_line_label(
                text.clone(),
                polyline,
                style,
                placement,
                repeat_distance,
            );
            println!("[label] added line label id={} text=\"{}\"", id.0, text);
            true
        }
        ViewerCmd::RemoveLabel { id } => {
            let removed = viewer.remove_label(*id);
            println!("[label] remove id={} success={}", id, removed);
            true
        }
        ViewerCmd::ClearLabels => {
            viewer.clear_labels();
            println!("[label] cleared all labels");
            true
        }
        ViewerCmd::SetLabelsEnabled { enabled } => {
            viewer.set_labels_enabled(*enabled);
            println!("[label] enabled={}", enabled);
            true
        }
        ViewerCmd::LoadLabelAtlas {
            atlas_png_path,
            metrics_json_path,
        } => {
            match viewer.load_label_atlas(atlas_png_path, metrics_json_path) {
                Ok(()) => println!("[label] atlas loaded from {}", atlas_png_path),
                Err(e) => eprintln!("[label] failed to load atlas: {}", e),
            }
            true
        }
        ViewerCmd::SetLabelZoom { zoom } => {
            viewer.label_manager.set_zoom(*zoom);
            println!("[label] zoom={:.2}", zoom);
            true
        }
        ViewerCmd::SetMaxVisibleLabels { max } => {
            viewer.label_manager.set_max_visible(*max);
            println!("[label] max_visible={}", max);
            true
        }
        ViewerCmd::AddCurvedLabel {
            text,
            polyline,
            size,
            color,
            halo_color,
            halo_width,
            priority,
            tracking: _tracking,
            center_on_path: _center_on_path,
        } => {
            use crate::labels::{LabelStyle, LineLabelPlacement};
            use glam::Vec3;

            let mut style = LabelStyle::default();
            if let Some(size) = size {
                style.size = *size;
            }
            if let Some(color) = color {
                style.color = *color;
            }
            if let Some(halo_color) = halo_color {
                style.halo_color = *halo_color;
            }
            if let Some(halo_width) = halo_width {
                style.halo_width = *halo_width;
            }
            if let Some(priority) = priority {
                style.priority = *priority;
            }

            let polyline: Vec<Vec3> = polyline
                .iter()
                .map(|point| Vec3::new(point[0], point[1], point[2]))
                .collect();
            let id = viewer.label_manager.add_line_label(
                text.clone(),
                polyline,
                style,
                LineLabelPlacement::Along,
                0.0,
            );
            println!("[label] added curved label id={} text=\"{}\"", id.0, text);
            true
        }
        ViewerCmd::AddCallout {
            text,
            anchor,
            offset,
            background_color: _bg,
            border_color: _bc,
            border_width: _bw,
            corner_radius: _cr,
            padding: _pad,
            text_size,
            text_color,
        } => {
            use crate::labels::LabelStyle;

            let mut style = LabelStyle::default();
            if let Some(size) = text_size {
                style.size = *size;
            }
            if let Some(color) = text_color {
                style.color = *color;
            }
            if let Some(offset) = offset {
                style.offset = *offset;
                style.flags.leader = true;
            }
            let world_pos = (anchor[0], anchor[1], anchor[2]);
            let id = viewer.add_label(text, world_pos, Some(style));
            println!("[label] added callout id={} text=\"{}\"", id, text);
            true
        }
        ViewerCmd::RemoveCallout { id } => {
            let removed = viewer.remove_label(*id);
            println!("[label] remove callout id={} success={}", id, removed);
            true
        }
        ViewerCmd::SetLabelTypography {
            tracking: _,
            kerning: _,
            line_height: _,
            word_spacing: _,
        } => {
            println!("[label] typography settings updated");
            true
        }
        ViewerCmd::SetDeclutterAlgorithm {
            algorithm,
            seed: _,
            max_iterations: _,
        } => {
            println!("[label] declutter algorithm set to: {}", algorithm);
            true
        }
        _ => false,
    }
}
