use std::error::Error;

use forge3d::cli::interactive_viewer::run_interactive_viewer_cli;

fn main() -> Result<(), Box<dyn Error>> {
    run_interactive_viewer_cli()
}
