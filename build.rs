//! Build script: embed the short git SHA so the RenderCertificate
//! (`src/core/certificate.rs`) can report the exact source revision it was
//! built from. Falls back to "unknown" when git is unavailable (e.g. building
//! from an unpacked source tarball).

use std::process::Command;

fn main() {
    let sha = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=FORGE3D_GIT_SHA={sha}");
    // Re-run when HEAD moves so the embedded SHA stays current. `.git/HEAD`
    // changes on branch switches; `.git/logs/HEAD` changes on every commit
    // (including same-branch commits), so watching it refreshes the embedded
    // SHA after an ordinary `git commit` that leaves HEAD pointing at the same
    // ref.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/logs/HEAD");
}
