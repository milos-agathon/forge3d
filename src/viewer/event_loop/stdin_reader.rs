mod helpers;
mod parser;
mod spawn;

pub(crate) use parser::{translate_text_command, TextCommandCapabilities};
pub use spawn::spawn_stdin_reader;

#[cfg(test)]
mod tests;
