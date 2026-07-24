//! Build script: embed the short git SHA so the RenderCertificate
//! (`src/core/certificate.rs`) can report the exact source revision it was
//! built from. Falls back to "unknown" when git is unavailable (e.g. building
//! from an unpacked source tarball).

use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn shader_tree_hash(root: &Path) -> String {
    fn collect(path: &Path, files: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, files);
            } else if path.extension().and_then(|value| value.to_str()) == Some("wgsl") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        // Path display uses `\\` on Windows and `/` on Unix. Hash only the
        // shader-root-relative component sequence with an explicit `/`
        // separator so identical source trees have identical engine
        // fingerprints across the portability matrix.
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let normalized = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        hash.update(normalized.as_bytes());
        if let Ok(bytes) = fs::read(path) {
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
    }
    format!("{:x}", hash.finalize())
}

fn main() {
    let git_revision = |argument: &str| {
        Command::new("git")
            .args(["rev-parse", argument, "HEAD"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    };
    let sha = git_revision("--short=12");
    let full_sha = env::var("GITHUB_SHA")
        .ok()
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .unwrap_or_else(|| {
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown".to_string())
        });

    println!("cargo:rustc-env=FORGE3D_GIT_SHA={sha}");
    println!("cargo:rustc-env=FORGE3D_GIT_SHA_FULL={full_sha}");
    println!(
        "cargo:rustc-env=FORGE3D_NAGA_VERSION={}",
        locked_package_version("naga").unwrap_or_else(|| "unknown".into())
    );
    write_registered_wgsl_modules();
    println!(
        "cargo:rustc-env=FORGE3D_WGSL_TREE_SHA256={}",
        shader_tree_hash(Path::new("src/shaders"))
    );
    // Re-run when HEAD or its resolved ref moves. `git rev-parse --git-path`
    // also handles linked worktrees, where the checkout metadata is not stored
    // in a repository-local `.git` directory.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-env-changed=GITHUB_SHA");
    for git_path in ["HEAD".to_string()].into_iter().chain(
        Command::new("git")
            .args(["symbolic-ref", "-q", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|reference| vec![reference.trim().to_string()])
            .unwrap_or_default(),
    ) {
        if let Some(path) = Command::new("git")
            .args(["rev-parse", "--git-path", &git_path])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|path| path.trim().to_string())
            .filter(|path| !path.is_empty())
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}

fn locked_package_version(package: &str) -> Option<String> {
    let lock = fs::read_to_string("Cargo.lock").ok()?;
    for block in lock.split("[[package]]") {
        let mut name = None;
        let mut version = None;
        for line in block.lines() {
            let Some((field, value)) = line.split_once('=') else {
                continue;
            };
            match field.trim() {
                "name" => name = Some(value.trim().trim_matches('"')),
                "version" => version = Some(value.trim().trim_matches('"')),
                _ => {}
            }
        }
        if name == Some(package) {
            return version.map(ToString::to_string);
        }
    }
    None
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
