// src/viewer/ipc_split/mod.rs
// IPC module for non-blocking viewer control via TCP + NDJSON
// Split from ipc.rs for <= 300 LOC per file

mod protocol;
mod server;

pub use protocol::{
    ipc_request_to_viewer_cmd, parse_ipc_request, IpcRequest, IpcResponse,
    TerrainVolumetricsReport, TerrainVolumetricsVolumeReport, ViewerStats,
};
pub use server::{start_ipc_server, IpcServerConfig, IpcServerHandle};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_load_obj() {
        let json = r#"{"cmd":"load_obj","path":"model.obj"}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::LoadObj { path } => assert_eq!(path, "model.obj"),
            _ => panic!("Expected LoadObj"),
        }
    }

    #[test]
    fn test_parse_cam_lookat() {
        let json = r#"{"cmd":"cam_lookat","eye":[0,5,10],"target":[0,0,0],"up":[0,1,0]}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::CamLookat { eye, target, up } => {
                assert_eq!(eye, [0.0, 5.0, 10.0]);
                assert_eq!(target, [0.0, 0.0, 0.0]);
                assert_eq!(up, [0.0, 1.0, 0.0]);
            }
            _ => panic!("Expected CamLookat"),
        }
    }

    #[test]
    fn test_parse_set_transform() {
        let json = r#"{"cmd":"set_transform","translation":[1,2,3]}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::SetTransform { translation, .. } => {
                assert_eq!(translation, Some([1.0, 2.0, 3.0]));
            }
            _ => panic!("Expected SetTransform"),
        }
    }

    #[test]
    fn test_parse_snapshot() {
        let json = r#"{"cmd":"snapshot","path":"out.png","width":1920,"height":1080}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::Snapshot {
                path,
                width,
                height,
            } => {
                assert_eq!(path, "out.png");
                assert_eq!(width, Some(1920));
                assert_eq!(height, Some(1080));
            }
            _ => panic!("Expected Snapshot"),
        }
    }

    #[test]
    fn test_parse_set_terrain_scatter() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"trees",
                    "color":[0.2,0.6,0.3,1.0],
                    "max_draw_distance":180.0,
                    "transforms":[[1.0,0.0,0.0,3.0,0.0,1.0,0.0,4.0,0.0,0.0,1.0,5.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "normals":[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2],
                            "max_distance":90.0
                        }
                    ]
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                assert_eq!(batches[0].name.as_deref(), Some("trees"));
                assert_eq!(batches[0].transforms.len(), 1);
                assert_eq!(batches[0].levels.len(), 1);
                assert_eq!(batches[0].levels[0].indices, vec![0, 1, 2]);
            }
            _ => panic!("Expected SetTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                assert_eq!(batches[0].name.as_deref(), Some("trees"));
                assert_eq!(batches[0].transforms.len(), 1);
                assert_eq!(batches[0].levels.len(), 1);
                assert_eq!(batches[0].levels[0].mesh.indices, vec![0, 1, 2]);
            }
            _ => panic!("Expected ViewerCmd::SetTerrainScatter"),
        }
    }

    #[test]
    fn test_parse_clear_terrain_scatter() {
        let json = r#"{"cmd":"clear_terrain_scatter"}"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::ClearTerrainScatter => {}
            _ => panic!("Expected ClearTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::ClearTerrainScatter => {}
            _ => panic!("Expected ViewerCmd::ClearTerrainScatter"),
        }
    }

    #[test]
    fn test_parse_set_point_cloud_camera_params() {
        let json = r#"{"cmd":"set_point_cloud_params","phi":0.6,"theta":0.5,"radius":1.4}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::SetPointCloudParams {
                phi, theta, radius, ..
            } => {
                assert_eq!(phi, Some(0.6));
                assert_eq!(theta, Some(0.5));
                assert_eq!(radius, Some(1.4));
            }
            _ => panic!("Expected SetPointCloudParams"),
        }
    }

    #[test]
    fn test_response_serialization() {
        let resp = IpcResponse::success();
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":true"#));
        assert!(!json.contains("error"));
        assert!(!json.contains("stats"));

        let resp = IpcResponse::error("test error");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":false"#));
        assert!(json.contains("test error"));
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[test]
    fn test_parse_set_terrain_scatter_with_hlod() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"trees",
                    "color":[0.2,0.6,0.3,1.0],
                    "max_draw_distance":180.0,
                    "transforms":[[1.0,0.0,0.0,3.0,0.0,1.0,0.0,4.0,0.0,0.0,1.0,5.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "normals":[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2],
                            "max_distance":90.0
                        }
                    ],
                    "hlod":{
                        "hlod_distance":100.0,
                        "cluster_radius":25.0,
                        "simplify_ratio":0.5
                    }
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                let hlod = batches[0].hlod.as_ref().expect("hlod should be present");
                assert_eq!(hlod.hlod_distance, 100.0);
                assert_eq!(hlod.cluster_radius, 25.0);
                assert_eq!(hlod.simplify_ratio, 0.5);
            }
            _ => panic!("Expected SetTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                let hlod_config = batches[0]
                    .hlod_config
                    .as_ref()
                    .expect("hlod_config should be present");
                assert_eq!(hlod_config.hlod_distance, 100.0);
                assert_eq!(hlod_config.cluster_radius, 25.0);
                assert_eq!(hlod_config.simplify_ratio, 0.5);
            }
            _ => panic!("Expected ViewerCmd::SetTerrainScatter"),
        }
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[test]
    fn test_parse_set_terrain_scatter_without_hlod_backward_compat() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"rocks",
                    "transforms":[[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2]
                        }
                    ]
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                assert!(batches[0].hlod.is_none(), "hlod should be None when omitted");
            }
            _ => panic!("Expected SetTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetTerrainScatter { batches } => {
                assert!(
                    batches[0].hlod_config.is_none(),
                    "hlod_config should be None when omitted"
                );
            }
            _ => panic!("Expected ViewerCmd::SetTerrainScatter"),
        }
    }
}
