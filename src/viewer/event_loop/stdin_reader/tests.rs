use crate::cli::args::GiVizMode;
use crate::viewer::viewer_enums::ViewerCmd;

use super::parser::parse_stdin_command;

#[test]
fn parses_gi_seed_value() {
    let cmds = parse_stdin_command(":gi-seed 42").unwrap();
    assert!(matches!(cmds.as_slice(), [ViewerCmd::SetGiSeed(42)]));
}

#[test]
fn parses_gi_off_bundle() {
    let cmds = parse_stdin_command(":gi off").unwrap();
    assert_eq!(cmds.len(), 3);
    assert!(matches!(cmds[0], ViewerCmd::GiToggle("ssao", false)));
    assert!(matches!(cmds[1], ViewerCmd::GiToggle("ssgi", false)));
    assert!(matches!(cmds[2], ViewerCmd::GiToggle("ssr", false)));
}

#[test]
fn parses_snapshot_with_size() {
    let cmds = parse_stdin_command(":snapshot out.png 640x480").unwrap();
    assert!(matches!(
        cmds.as_slice(),
        [ViewerCmd::SnapshotWithSize {
            path,
            width: Some(640),
            height: Some(480),
        }] if path == "out.png"
    ));
}

#[test]
fn parses_gtao_alias() {
    let cmds = parse_stdin_command(":gi gtao on").unwrap();
    assert_eq!(cmds.len(), 2);
    assert!(matches!(cmds[0], ViewerCmd::GiToggle("ssao", true)));
    assert!(matches!(cmds[1], ViewerCmd::SetSsaoTechnique(1)));
}

#[test]
fn parses_gi_visualization_mode() {
    let cmds = parse_stdin_command(":viz gi ao").unwrap();
    assert!(matches!(
        cmds.as_slice(),
        [ViewerCmd::SetGiViz(mode)] if matches!(mode, GiVizMode::Ao)
    ));
}

#[test]
fn parses_oit_query_without_mode() {
    let cmds = parse_stdin_command(":oit").unwrap();
    assert!(matches!(cmds.as_slice(), [ViewerCmd::GetOitMode]));
}
