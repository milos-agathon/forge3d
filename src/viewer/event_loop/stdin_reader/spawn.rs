use std::io::BufRead;

use winit::event_loop::EventLoopProxy;

use crate::viewer::viewer_enums::ViewerCmd;

use super::helpers::print_help;
use super::parser::parse_stdin_command;

/// Spawn a thread that reads stdin and sends ViewerCmd events via the proxy
pub fn spawn_stdin_reader(proxy: EventLoopProxy<ViewerCmd>) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut iter = stdin.lock().lines();
        while let Some(Ok(line)) = iter.next() {
            let line = line.trim().to_lowercase();
            if line.is_empty() {
                continue;
            }
            if let Some(cmds) = parse_stdin_command(&line) {
                for cmd in cmds {
                    let _ = proxy.send_event(cmd);
                }
            } else if matches!(line.as_str(), ":quit" | "quit" | ":exit" | "exit") {
                let _ = proxy.send_event(ViewerCmd::Quit);
                break;
            } else {
                print_help();
            }
        }
    });
}
