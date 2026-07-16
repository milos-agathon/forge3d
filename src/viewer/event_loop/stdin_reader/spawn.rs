use std::io::BufRead;

use winit::event_loop::EventLoopProxy;

use crate::viewer::viewer_enums::ViewerCmd;

use super::helpers::print_help;
use super::parser::{translate_text_command, TextCommandCapabilities};

/// Spawn a thread that reads stdin and sends ViewerCmd events via the proxy
pub fn spawn_stdin_reader(proxy: EventLoopProxy<ViewerCmd>) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut iter = stdin.lock().lines();
        while let Some(Ok(line)) = iter.next() {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            match translate_text_command(&line, TextCommandCapabilities::runtime()) {
                Ok(Some(cmds)) => {
                    for cmd in cmds {
                        let _ = proxy.send_event(cmd);
                    }
                }
                Ok(None)
                    if matches!(
                        line.to_lowercase().as_str(),
                        ":quit" | "quit" | ":exit" | "exit"
                    ) =>
                {
                    let _ = proxy.send_event(ViewerCmd::Quit);
                    break;
                }
                Ok(None) => print_help(),
                Err(err) => eprintln!("[viewer] command rejected before enqueue: {err}"),
            }
        }
    });
}
