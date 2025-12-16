// src/viewer/ipc_split/mod.rs
// IPC module for non-blocking viewer control via TCP + NDJSON
// Split from ipc.rs for ≤300 LOC per file

mod protocol;
mod server;

pub use protocol::{
    ipc_request_to_viewer_cmd, parse_ipc_request, IpcRequest, IpcResponse, ViewerStats,
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
}
