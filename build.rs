//! Build script: embed the short git SHA so the RenderCertificate
//! (`src/core/certificate.rs`) can report the exact source revision it was
//! built from. Falls back to "unknown" when git is unavailable (e.g. building
//! from an unpacked source tarball).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
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
    write_registered_wgsl_modules();
    // Re-run when HEAD moves so the embedded SHA stays current. `.git/HEAD`
    // changes on branch switches; `.git/logs/HEAD` changes on every commit
    // (including same-branch commits), so watching it refreshes the embedded
    // SHA after an ordinary `git commit` that leaves HEAD pointing at the same
    // ref.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/logs/HEAD");
    println!("cargo:rerun-if-changed=src");
}

fn write_registered_wgsl_modules() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut modules = Vec::new();
    collect_registered_wgsl(&manifest_dir, &manifest_dir.join("src"), &mut modules);
    let mut index = 0;
    while index < modules.len() {
        let shader = manifest_dir.join(&modules[index]);
        collect_shader_includes(&manifest_dir, &shader, &mut modules);
        index += 1;
    }
    modules.sort();
    modules.dedup();

    let mut generated = String::from("&[\n");
    for module in modules {
        generated.push_str("    ");
        generated.push_str(&format!("{module:?}"));
        generated.push_str(",\n");
    }
    generated.push_str("]\n");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_dir.join("registered_wgsl.rs"), generated).unwrap();
}

fn collect_registered_wgsl(root: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_registered_wgsl(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let source = fs::read_to_string(&path).unwrap();
            let mut tail = source.as_str();
            while let Some(start) = tail.find("include_str!") {
                tail = &tail[start + "include_str!".len()..];
                let Some(open) = tail.find('"') else { break };
                tail = &tail[open + 1..];
                let Some(close) = tail.find('"') else { break };
                let include = &tail[..close];
                tail = &tail[close + 1..];
                if include.ends_with(".wgsl") {
                    add_existing_shader(root, &path.parent().unwrap().join(include), out);
                }
            }
        }
    }
}

fn collect_shader_includes(root: &Path, shader: &Path, out: &mut Vec<String>) {
    let Ok(source) = fs::read_to_string(shader) else {
        return;
    };
    for line in source.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("#include") else {
            continue;
        };
        let include = rest.trim().trim_matches(['"', '<', '>']);
        for candidate in [
            shader.parent().unwrap().join(include),
            root.join("src/shaders").join(include),
        ] {
            if candidate.is_file() {
                add_existing_shader(root, &candidate, out);
                break;
            }
        }
    }
}

fn add_existing_shader(root: &Path, path: &Path, out: &mut Vec<String>) {
    let Ok(path) = path.canonicalize() else {
        return;
    };
    let Ok(relative) = path.strip_prefix(root.canonicalize().unwrap()) else {
        return;
    };
    // Only production shaders under src/ are runtime registrations. Verifier
    // fixtures also use include_str!, but must never enter the coverage ledger.
    if !relative.starts_with("src") {
        return;
    }
    if relative.extension().is_some_and(|ext| ext == "wgsl") {
        let relative = relative.to_string_lossy().replace('\\', "/");
        if !out.contains(&relative) {
            out.push(relative);
        }
    }
}
