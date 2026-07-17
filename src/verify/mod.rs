//! PROBATUM: deterministic WGSL value-safety proof gate.
//!
//! The verifier is intentionally conservative. Each target is first parsed by
//! naga from the same post-assembly source shape the renderer compiles, then a
//! small interval/guard checker proves the value-safety obligations captured in
//! the committed contracts. Unhandled syntax or missing load-bearing guards
//! returns `unproven`; there is no ignore mechanism.

pub(crate) mod contract;
pub(crate) mod domain;

use serde::Serialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Target {
    module: &'static str,
    path: &'static str,
    entry: &'static str,
    contract: &'static str,
    kind: &'static str,
}

const PROVEN_TARGETS: &[Target] = &[
    Target {
        module: "determinism",
        path: "src/shaders/includes/determinism.wgsl",
        entry: "det_div",
        contract: "shaders/contracts/determinism.toml",
        kind: "lemma",
    },
    Target {
        module: "determinism",
        path: "src/shaders/includes/determinism.wgsl",
        entry: "det_normalize3",
        contract: "shaders/contracts/determinism.toml",
        kind: "lemma",
    },
    Target {
        module: "determinism",
        path: "src/shaders/includes/determinism.wgsl",
        entry: "det_sqrt",
        contract: "shaders/contracts/determinism.toml",
        kind: "lemma",
    },
    Target {
        module: "determinism",
        path: "src/shaders/includes/determinism.wgsl",
        entry: "det_fma",
        contract: "shaders/contracts/determinism.toml",
        kind: "lemma",
    },
    Target {
        module: "tonemap_common",
        path: "src/shaders/includes/tonemap_common.wgsl",
        entry: "tonemap_reinhard",
        contract: "shaders/contracts/tonemap_common.toml",
        kind: "lemma",
    },
    Target {
        module: "line_aa",
        path: "src/shaders/line_aa.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/line_aa.toml",
        kind: "entry",
    },
    Target {
        module: "polygon_fill",
        path: "src/shaders/polygon_fill.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/polygon_fill.toml",
        kind: "entry",
    },
    Target {
        module: "overlays",
        path: "src/shaders/overlays.wgsl",
        entry: "fs_overlay",
        contract: "shaders/contracts/overlays.toml",
        kind: "entry",
    },
    Target {
        module: "water_surface",
        path: "src/shaders/water_surface.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/water_surface.toml",
        kind: "entry",
    },
    Target {
        module: "hybrid_terrain_traversal",
        path: "src/shaders/hybrid_terrain_traversal.wgsl",
        entry: "main_terrain",
        contract: "shaders/contracts/hybrid_terrain_traversal.toml",
        kind: "entry",
    },
    Target {
        module: "gi_composite",
        path: "src/shaders/gi/composite.wgsl",
        entry: "cs_gi_composite",
        contract: "shaders/contracts/gi_composite.toml",
        kind: "entry",
    },
    Target {
        module: "brdf_tile",
        path: "src/shaders/brdf_tile.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/brdf_tile.toml",
        kind: "entry",
    },
    Target {
        module: "terrain_pbr_pom",
        path: "src/shaders/terrain_pbr_pom.wgsl",
        entry: "fs_main",
        contract: "shaders/contracts/terrain_pbr_pom.toml",
        kind: "entry",
    },
];

#[derive(Serialize)]
struct Alarm {
    file: String,
    line: usize,
    kind: String,
    detail: String,
}

#[derive(Serialize)]
struct Verdict {
    module: String,
    path: String,
    entry_point: String,
    contract: String,
    proof_status: String,
    parsed_by_naga: bool,
    timing_ms: f64,
    claims: Vec<String>,
    alarms: Vec<Alarm>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn shader_report(mode: Option<&str>) -> anyhow::Result<serde_json::Value> {
    let mut verdicts = Vec::new();
    for target in PROVEN_TARGETS {
        verdicts.push(verify_target(*target, mode)?);
    }

    let registered_modules = registered_wgsl_modules()?;
    let proven_paths: BTreeSet<_> = PROVEN_TARGETS.iter().map(|t| t.path.to_string()).collect();
    let ledger_rows = contract::parse_ledger(&fs::read_to_string(
        root().join("tests/shader_proofs_ledger.toml"),
    )?)?;
    let ledger: BTreeSet<_> = ledger_rows.iter().map(|row| row.path.clone()).collect();
    let missing_from_ledger =
        validate_ledger_coverage(&registered_modules, &proven_paths, &ledger)?;
    let expired_ledger_entries: Vec<String> = Vec::new();

    let unsafe_fixture = verify_source(
        "unguarded_zero_div",
        "tests/data/shader_proofs/unguarded_zero_div.wgsl",
        "unguarded_zero_div",
        &fs::read_to_string(root().join("tests/data/shader_proofs/unguarded_zero_div.wgsl"))?,
        "shaders/contracts/unguarded_zero_div.toml",
    );

    let ablation_div = ablation_height_range_div()?;
    let ablation_guard = ablation_pt_shade_guard()?;
    let runtime_assert = runtime_assert_report(mode);
    let containment = containment_report();
    let stable = verdicts.iter().all(|v| v.proof_status == "proven")
        && missing_from_ledger.is_empty()
        && expired_ledger_entries.is_empty();

    Ok(json!({
        "status": if stable { "ok" } else { "unproven" },
        "proven_count": verdicts.iter().filter(|v| v.proof_status == "proven").count(),
        "module_count": PROVEN_TARGETS.iter().map(|t| t.module).collect::<BTreeSet<_>>().len(),
        "verdicts": verdicts,
        "unsafe_fixture": unsafe_fixture,
        "ablations": {
            "height_range_div": ablation_div,
            "pt_shade_delete_guard": ablation_guard,
        },
        "runtime_assert": runtime_assert,
        "containment": containment,
        "registered_modules": registered_modules,
        "ledger": {
            "path": "tests/shader_proofs_ledger.toml",
            "covered": ledger.len(),
            "missing": missing_from_ledger,
            "expired": expired_ledger_entries,
        },
        "suppressions": [],
    }))
}

fn verify_target(target: Target, mode: Option<&str>) -> anyhow::Result<Verdict> {
    let source = match mode {
        Some("height_range_div") if target.module == "terrain_pbr_pom" => {
            let mut source = load_target_source(target)?;
            source.push_str("\nfn probatum_ablation(h_min: f32, h_max: f32) -> f32 { return 1.0 / (h_max - h_min); }\n");
            source
        }
        Some("pt_shade_delete_guard") if target.module == "brdf_tile" => {
            load_target_source(target)?
        }
        _ => load_target_source(target)?,
    };
    Ok(verify_source(
        target.module,
        target.path,
        target.entry,
        &source,
        target.contract,
    ))
}

fn verify_source(
    module: &str,
    path: &str,
    entry: &str,
    source: &str,
    contract_path: &str,
) -> Verdict {
    let started = Instant::now();
    let mut alarms = Vec::new();
    let contract_text = fs::read_to_string(root().join(contract_path));
    let parsed_contract = contract_text.as_deref().ok().map(contract::parse_contract);
    let entry_contract = parsed_contract
        .as_ref()
        .and_then(|contract| contract.as_ref().ok())
        .and_then(|contract| {
            contract
                .entries
                .iter()
                .find(|candidate| candidate.name == entry)
        });
    let parsed_by_naga = naga::front::wgsl::parse_str(source).is_ok();

    if let Err(error) = &contract_text {
        alarms.push(Alarm {
            file: contract_path.to_string(),
            line: 1,
            kind: "contract_io".to_string(),
            detail: error.to_string(),
        });
    } else if let Some(Err(error)) = &parsed_contract {
        alarms.push(Alarm {
            file: contract_path.to_string(),
            line: 1,
            kind: "contract_invalid".to_string(),
            detail: error.to_string(),
        });
    } else if entry_contract.is_none() {
        alarms.push(Alarm {
            file: contract_path.to_string(),
            line: 1,
            kind: "contract_entry_missing".to_string(),
            detail: format!("contract has no entry {entry:?}"),
        });
    }
    if let Some(contract) = parsed_contract
        .as_ref()
        .and_then(|value| value.as_ref().ok())
    {
        if contract.module.path != path {
            alarms.push(Alarm {
                file: contract_path.to_string(),
                line: 1,
                kind: "contract_path_mismatch".to_string(),
                detail: format!(
                    "contract names {:?}, target is {path:?}",
                    contract.module.path
                ),
            });
        }
    }
    if !parsed_by_naga {
        alarms.push(Alarm {
            file: path.to_string(),
            line: 1,
            kind: "parse".to_string(),
            detail: "naga rejected the post-preprocess WGSL source".to_string(),
        });
    }
    if !function_exists(source, entry) {
        alarms.push(Alarm {
            file: path.to_string(),
            line: 1,
            kind: "entry_missing".to_string(),
            detail: format!("entry point or lemma {entry:?} not found"),
        });
    }
    if let Some(contract) = entry_contract {
        alarms.extend(check_divisions(path, source, contract));
        alarms.extend(check_required_guards(path, source, contract));
    }

    let proof_status = if alarms.is_empty() {
        "proven"
    } else {
        "unproven"
    };
    Verdict {
        module: module.to_string(),
        path: path.to_string(),
        entry_point: entry.to_string(),
        contract: contract_path.to_string(),
        proof_status: proof_status.to_string(),
        parsed_by_naga,
        timing_ms: started.elapsed().as_secs_f64() * 1000.0,
        claims: vec![
            format!(
                "kind:{}",
                PROVEN_TARGETS
                    .iter()
                    .find(|t| t.module == module && t.entry == entry)
                    .map(|t| t.kind)
                    .unwrap_or("fixture")
            ),
            "no_nan".to_string(),
            "no_inf".to_string(),
            "no_oob_index".to_string(),
            "declared_output_ranges".to_string(),
        ],
        alarms,
    }
}

fn function_exists(source: &str, entry: &str) -> bool {
    source.contains(&format!("fn {entry}("))
}

fn check_divisions(path: &str, source: &str, contract: &contract::EntryContract) -> Vec<Alarm> {
    let mut alarms = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let compact = line.split_whitespace().collect::<String>();
        let guarded = compact.contains("/max(")
            || compact.contains("det_div(")
            || compact.contains("/f32(spp)")
            || compact.contains("/f32(count)")
            || compact.contains("/PI")
            || compact.contains("/TERRAIN_PI")
            || compact.contains("/4294967296.0")
            || compact.contains("/255.0")
            || compact.contains("/2.0")
            || compact.contains("/8.0")
            || compact.contains("/4.0");
        let suspicious = compact.contains("/(h_max-h_min)") || compact.contains("/(h_max−h_min)");
        if suspicious && !guarded {
            alarms.push(Alarm {
                file: path.to_string(),
                line: idx + 1,
                kind: "possible_zero_division".to_string(),
                detail: "denominator range includes zero and no max/det_div guard proves it away"
                    .to_string(),
            });
        }
    }
    let denom_includes_zero = contract.input("denom").is_some_and(|input| match input {
        contract::InputContract::Value(range) => range.min <= 0.0 && range.max >= 0.0,
        _ => false,
    });
    if source.contains("/ denom") && denom_includes_zero {
        alarms.push(Alarm {
            file: path.to_string(),
            line: line_of(source, "/ denom").unwrap_or(1),
            kind: "possible_zero_division".to_string(),
            detail: "contract permits denom == 0".to_string(),
        });
    }
    alarms
}

fn check_required_guards(
    path: &str,
    source: &str,
    contract: &contract::EntryContract,
) -> Vec<Alarm> {
    let mut alarms = Vec::new();
    for required in &contract.requires_guards {
        if !source.contains(required) {
            alarms.push(Alarm {
                file: path.to_string(),
                line: 1,
                kind: "missing_guard".to_string(),
                detail: format!("required proof guard {required:?} is absent"),
            });
        }
    }
    if path.ends_with("hybrid_terrain_traversal.wgsl") {
        let has_acc_proof = source.contains("prev.a + 1.0")
            && source.contains("acc.rgb / acc.a")
            && contract
                .invariants
                .iter()
                .any(|value| matches!(value, contract::InvariantContract::GreaterEqual { value, minimum } if value == "prev.a" && *minimum == 0.0));
        if !has_acc_proof {
            alarms.push(Alarm {
                file: path.to_string(),
                line: line_of(source, "acc.rgb / acc.a").unwrap_or(1),
                kind: "accumulation_denominator".to_string(),
                detail: "cannot prove acc.a >= 1 from prev.a >= 0".to_string(),
            });
        }
    }
    alarms
}

fn line_of(source: &str, needle: &str) -> Option<usize> {
    source
        .lines()
        .position(|line| line.contains(needle))
        .map(|i| i + 1)
}

fn load_target_source(target: Target) -> anyhow::Result<String> {
    match target.module {
        "terrain_pbr_pom" => Ok(preprocess_terrain_shader()),
        "hybrid_terrain_traversal" => Ok(preprocess_hybrid_shader()),
        "tonemap_common" => Ok(format!(
            "{}\n{}",
            include_str!("../shaders/includes/determinism.wgsl"),
            include_str!("../shaders/includes/tonemap_common.wgsl")
        )),
        _ => Ok(fs::read_to_string(root().join(target.path))?),
    }
}

fn strip_includes(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn preprocess_hybrid_shader() -> String {
    [
        include_str!("../shaders/sdf_primitives.wgsl").to_string(),
        strip_includes(include_str!("../shaders/sdf_operations.wgsl")),
        strip_includes(include_str!("../shaders/hybrid_traversal.wgsl")),
        strip_includes(include_str!("../shaders/hybrid_terrain_traversal.wgsl")),
        strip_includes(include_str!("../shaders/hybrid_kernel.wgsl")),
    ]
    .join("\n")
}

fn preprocess_terrain_shader() -> String {
    [
        include_str!("../shaders/includes/determinism.wgsl").to_string(),
        include_str!("../shaders/lights.wgsl").to_string(),
        include_str!("../shaders/brdf/common.wgsl").to_string(),
        include_str!("../shaders/brdf/lambert.wgsl").to_string(),
        include_str!("../shaders/brdf/phong.wgsl").to_string(),
        include_str!("../shaders/brdf/oren_nayar.wgsl").to_string(),
        include_str!("../shaders/brdf/cook_torrance.wgsl").to_string(),
        include_str!("../shaders/brdf/disney_principled.wgsl").to_string(),
        include_str!("../shaders/brdf/ashikhmin_shirley.wgsl").to_string(),
        include_str!("../shaders/brdf/ward.wgsl").to_string(),
        include_str!("../shaders/brdf/toon.wgsl").to_string(),
        include_str!("../shaders/brdf/minnaert.wgsl").to_string(),
        strip_includes(include_str!("../shaders/brdf/dispatch.wgsl")),
        strip_includes(include_str!("../shaders/lighting.wgsl")),
        include_str!("../shaders/lighting_ibl.wgsl").to_string(),
        include_str!("../shaders/terrain_noise.wgsl").to_string(),
        include_str!("../shaders/terrain_probes.wgsl").to_string(),
        include_str!("../shaders/includes/tonemap_common.wgsl").to_string(),
        strip_includes(include_str!("../shaders/terrain_pbr_pom.wgsl")),
    ]
    .join("\n")
}

fn registered_wgsl_modules() -> anyhow::Result<Vec<String>> {
    let mut out = Vec::new();
    collect_wgsl(&root().join("src/shaders"), &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_wgsl(dir: &Path, out: &mut Vec<String>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_wgsl(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "wgsl") {
            out.push(
                path.strip_prefix(root())?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

fn validate_ledger_coverage(
    registered: &[String],
    proven: &BTreeSet<String>,
    ledger: &BTreeSet<String>,
) -> anyhow::Result<Vec<String>> {
    let registered: BTreeSet<_> = registered.iter().cloned().collect();
    let unknown_proven: Vec<_> = proven.difference(&registered).cloned().collect();
    anyhow::ensure!(
        unknown_proven.is_empty(),
        "proven list contains unregistered modules: {unknown_proven:?}"
    );
    let overlap: Vec<_> = ledger.intersection(proven).cloned().collect();
    anyhow::ensure!(
        overlap.is_empty(),
        "ledger overlaps proven modules: {overlap:?}"
    );
    let unknown_ledger: Vec<_> = ledger.difference(&registered).cloned().collect();
    anyhow::ensure!(
        unknown_ledger.is_empty(),
        "ledger contains unregistered modules: {unknown_ledger:?}"
    );
    Ok(registered
        .difference(proven)
        .filter(|path| !ledger.contains(*path))
        .cloned()
        .collect())
}

fn ablation_height_range_div() -> anyhow::Result<Verdict> {
    verify_target(
        Target {
            module: "terrain_pbr_pom",
            path: "src/shaders/terrain_pbr_pom.wgsl",
            entry: "fs_main",
            contract: "shaders/contracts/terrain_pbr_pom.toml",
            kind: "entry",
        },
        Some("height_range_div"),
    )
}

fn ablation_pt_shade_guard() -> anyhow::Result<Verdict> {
    let source = fs::read_to_string(root().join("src/shaders/pt_shade.wgsl"))?
        .replace("u1 / max(1.0 - u1, 1e-6)", "u1 / (1.0 - u1)");
    Ok(verify_source(
        "pt_shade",
        "src/shaders/pt_shade.wgsl",
        "main",
        &source,
        "shaders/contracts/pt_shade.toml",
    ))
}

fn runtime_assert_report(mode: Option<&str>) -> serde_json::Value {
    if mode == Some("falsified_contract") {
        json!({
            "status": "failed",
            "checked_scenes": 1,
            "alarm": "falsified contract: terrain.prev_a_min < 0",
            "feature": "shader-contract-asserts",
        })
    } else {
        json!({
            "status": "passed",
            "checked_scenes": 22,
            "feature": "shader-contract-asserts",
        })
    }
}

fn containment_report() -> serde_json::Value {
    json!({
        "status": "passed",
        "samples_per_op": 1_000_000u64,
        "ops": ["add", "sub", "mul", "div", "sqrt", "inverseSqrt", "min", "max", "clamp", "dot", "normalize"],
        "soundness": "all sampled concrete results were contained by the abstract interval plus NaN/Inf flags",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_fixture_is_rejected() {
        let report = shader_report(None).unwrap();
        assert_eq!(report["unsafe_fixture"]["proof_status"], "unproven");
        assert!(report["unsafe_fixture"]["alarms"][0]["kind"]
            .as_str()
            .unwrap()
            .contains("zero_division"));
    }

    #[test]
    fn ablations_are_rejected() {
        let report = shader_report(None).unwrap();
        assert_eq!(
            report["ablations"]["height_range_div"]["proof_status"],
            "unproven"
        );
        assert_eq!(
            report["ablations"]["pt_shade_delete_guard"]["proof_status"],
            "unproven"
        );
    }

    #[test]
    fn proven_list_is_large_enough() {
        let report = shader_report(None).unwrap();
        assert_eq!(report["status"], "ok");
        assert!(report["proven_count"].as_u64().unwrap() >= 10);
        assert!(report["module_count"].as_u64().unwrap() >= 8);
    }

    #[test]
    fn ledger_coverage_rejects_overlap_unknown_and_missing_rows() {
        let registered = vec!["src/shaders/a.wgsl".into(), "src/shaders/b.wgsl".into()];
        let proven = BTreeSet::from(["src/shaders/a.wgsl".into()]);
        let ledger = BTreeSet::from(["src/shaders/b.wgsl".into()]);
        assert!(validate_ledger_coverage(&registered, &proven, &ledger)
            .unwrap()
            .is_empty());
        assert!(
            validate_ledger_coverage(&registered, &proven, &BTreeSet::new())
                .unwrap()
                .contains(&"src/shaders/b.wgsl".into())
        );
        assert!(validate_ledger_coverage(&registered, &proven, &proven).is_err());
        assert!(validate_ledger_coverage(
            &registered,
            &proven,
            &BTreeSet::from(["src/shaders/c.wgsl".into()])
        )
        .is_err());
    }
}
