use crate::viewer::viewer_enums::ViewerCmd;

pub(crate) fn command_priority(cmd: &ViewerCmd) -> u8 {
    match cmd {
        ViewerCmd::LoadTerrain(_) => 0,
        ViewerCmd::LoadPointCloud { .. } => 1,
        ViewerCmd::SetTerrainCamera { .. } | ViewerCmd::SetTerrain { .. } => 2,
        ViewerCmd::SetPointCloudParams { .. } => 3,
        ViewerCmd::SetCamLookAt { .. } => 4,
        _ => 5,
    }
}

/// Stable canonical order: resource/frame establishment precedes dependent
/// content while preserving order within each semantic class.
pub(crate) fn order_command_batch(mut commands: Vec<ViewerCmd>) -> Vec<ViewerCmd> {
    commands.sort_by_key(command_priority);
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_order_is_stable_and_loads_precede_frame_mutations_and_content() {
        let commands = vec![
            ViewerCmd::ClearLabels,
            ViewerCmd::SetCamLookAt {
                eye: [0.0; 3],
                target: [0.0; 3],
                up: [0.0, 1.0, 0.0],
            },
            ViewerCmd::LoadPointCloud {
                path: "p.laz".into(),
                point_size: 1.0,
                max_points: 1,
                color_mode: None,
            },
            ViewerCmd::LoadTerrain("t.tif".into()),
        ];
        let ordered = order_command_batch(commands);
        assert!(matches!(ordered[0], ViewerCmd::LoadTerrain(_)));
        assert!(matches!(ordered[1], ViewerCmd::LoadPointCloud { .. }));
        assert!(matches!(ordered[2], ViewerCmd::SetCamLookAt { .. }));
        assert!(matches!(ordered[3], ViewerCmd::ClearLabels));
    }

    #[test]
    fn every_frame_establisher_permutation_has_one_canonical_order() {
        let commands = [
            ViewerCmd::LoadTerrain("t.tif".into()),
            ViewerCmd::LoadPointCloud {
                path: "p.laz".into(),
                point_size: 1.0,
                max_points: 1,
                color_mode: None,
            },
            ViewerCmd::SetCamLookAt {
                eye: [0.0; 3],
                target: [0.0; 3],
                up: [0.0, 1.0, 0.0],
            },
            ViewerCmd::SetTransform {
                translation: Some([1.0, 2.0, 3.0]),
                rotation_quat: None,
                scale: None,
            },
        ];
        let permutations = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ];

        for permutation in permutations {
            let batch = permutation
                .into_iter()
                .map(|index| commands[index].clone())
                .collect();
            let ordered = order_command_batch(batch);
            assert!(matches!(ordered[0], ViewerCmd::LoadTerrain(_)));
            assert!(matches!(ordered[1], ViewerCmd::LoadPointCloud { .. }));
            assert!(matches!(ordered[2], ViewerCmd::SetCamLookAt { .. }));
            assert!(matches!(ordered[3], ViewerCmd::SetTransform { .. }));
        }
    }
}
