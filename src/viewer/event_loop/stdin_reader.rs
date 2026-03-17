mod helpers;
mod parser;
mod spawn;

pub use spawn::spawn_stdin_reader;

#[cfg(test)]
mod tests;
