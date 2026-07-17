use anyhow::{bail, Context};
use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Contract {
    pub module: ModuleContract,
    pub entries: Vec<EntryContract>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModuleContract {
    pub path: String,
    pub owner: String,
    pub expiry: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EntryContract {
    pub name: String,
    pub proof_status: String,
    pub inputs: Vec<InputContract>,
    pub outputs: Vec<RangeContract>,
    pub invariants: Vec<InvariantContract>,
    pub requires_guards: Vec<String>,
}

impl EntryContract {
    /// Missing facts are deliberately `None`; the interpreter must use its
    /// NaN/Inf-capable top value rather than inventing a safe range.
    pub(crate) fn input(&self, name: &str) -> Option<&InputContract> {
        self.inputs.iter().find(|input| input.name() == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum InputContract {
    Value(RangeContract),
    Uniform(RangeContract),
    Texture {
        range: RangeContract,
        dimensions: Vec<u32>,
    },
    Buffer {
        range: RangeContract,
        min_length: u32,
    },
    Sampler(String),
}

impl InputContract {
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Value(range) | Self::Uniform(range) => &range.name,
            Self::Texture { range, .. } | Self::Buffer { range, .. } => &range.name,
            Self::Sampler(name) => name,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RangeContract {
    pub name: String,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum InvariantContract {
    GreaterEqual {
        value: String,
        minimum: f32,
    },
    AbsGreaterEqual {
        value: String,
        minimum: f32,
    },
    SumAbsGreaterEqual {
        left: String,
        right: String,
        minimum: f32,
    },
    DifferenceGreaterEqual {
        left: String,
        right: String,
        minimum: f32,
    },
    LengthAtLeastProduct {
        buffer: String,
        width: String,
        height: String,
    },
    DimensionsCover {
        texture: String,
        width: String,
        height: String,
    },
    SameDimensions {
        left: String,
        right: String,
    },
    CountWithin {
        count: String,
        buffer: String,
    },
    IndicesWithin {
        indices: String,
        target: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LedgerRow {
    pub path: String,
    pub reason: String,
    pub owner: String,
    pub expiry: String,
}

#[derive(Default)]
struct ModuleBuilder {
    path: Option<String>,
    owner: Option<String>,
    expiry: Option<String>,
}

#[derive(Default)]
struct EntryBuilder {
    name: Option<String>,
    proof_status: Option<String>,
    inputs: Option<Vec<InputContract>>,
    outputs: Option<Vec<RangeContract>>,
    invariants: Option<Vec<InvariantContract>>,
    requires_guards: Option<Vec<String>>,
}

enum Section {
    None,
    Module,
    Entry(usize),
}

pub(crate) fn parse_contract(text: &str) -> anyhow::Result<Contract> {
    let mut module = ModuleBuilder::default();
    let mut entries = Vec::<EntryBuilder>::new();
    let mut section = Section::None;
    for (line_index, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "[module]" => {
                if !matches!(section, Section::None) {
                    bail!("line {}: duplicate or misplaced [module]", line_index + 1);
                }
                section = Section::Module;
                continue;
            }
            "[[entry]]" => {
                if matches!(section, Section::None) {
                    bail!("line {}: [module] must come first", line_index + 1);
                }
                entries.push(EntryBuilder::default());
                section = Section::Entry(entries.len() - 1);
                continue;
            }
            _ if line.starts_with('[') => bail!("line {}: unknown table {line}", line_index + 1),
            _ => {}
        }
        let (key, value) = key_value(line).with_context(|| format!("line {}", line_index + 1))?;
        match section {
            Section::None => bail!("line {}: value outside a table", line_index + 1),
            Section::Module => match key {
                "path" => set_once(&mut module.path, parse_string(value)?, key)?,
                "owner" => set_once(&mut module.owner, parse_string(value)?, key)?,
                "expiry" => set_once(&mut module.expiry, parse_string(value)?, key)?,
                _ => bail!("line {}: unknown module key {key}", line_index + 1),
            },
            Section::Entry(index) => {
                let entry = &mut entries[index];
                match key {
                    "name" => set_once(&mut entry.name, parse_string(value)?, key)?,
                    "proof_status" => set_once(&mut entry.proof_status, parse_string(value)?, key)?,
                    "inputs" => set_once(&mut entry.inputs, parse_inputs(value)?, key)?,
                    "outputs" => set_once(&mut entry.outputs, parse_outputs(value)?, key)?,
                    "invariants" => set_once(&mut entry.invariants, parse_invariants(value)?, key)?,
                    "requires_guards" => set_once(
                        &mut entry.requires_guards,
                        parse_unique_strings(value)?,
                        key,
                    )?,
                    _ => bail!("line {}: unknown entry key {key}", line_index + 1),
                }
            }
        }
    }

    let module = ModuleContract {
        path: required(module.path, "module.path")?,
        owner: nonempty(required(module.owner, "module.owner")?, "module.owner")?,
        expiry: required(module.expiry, "module.expiry")?,
    };
    validate_path(&module.path)?;
    validate_unexpired(&module.expiry)?;
    if entries.is_empty() {
        bail!("contract has no entries");
    }
    let mut names = BTreeSet::new();
    let entries = entries
        .into_iter()
        .map(|entry| {
            let name = nonempty(required(entry.name, "entry.name")?, "entry.name")?;
            if !names.insert(name.clone()) {
                bail!("duplicate entry {name}");
            }
            let proof_status = required(entry.proof_status, "entry.proof_status")?;
            if !matches!(proof_status.as_str(), "proven" | "must_reject") {
                bail!("invalid proof_status {proof_status}");
            }
            let inputs = required(entry.inputs, "entry.inputs")?;
            let outputs = required(entry.outputs, "entry.outputs")?;
            if inputs.is_empty() || outputs.is_empty() {
                bail!("entry {name} must declare inputs and outputs");
            }
            Ok(EntryContract {
                name,
                proof_status,
                inputs,
                outputs,
                invariants: entry.invariants.unwrap_or_default(),
                requires_guards: entry.requires_guards.unwrap_or_default(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Contract { module, entries })
}

pub(crate) fn parse_ledger(text: &str) -> anyhow::Result<Vec<LedgerRow>> {
    let mut rows = Vec::<ModuleBuilder>::new();
    let mut reasons = Vec::<Option<String>>::new();
    let mut current = None;
    for (line_index, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[unproven]]" {
            rows.push(ModuleBuilder::default());
            reasons.push(None);
            current = Some(rows.len() - 1);
            continue;
        }
        if line.starts_with('[') {
            bail!("line {}: unknown ledger table {line}", line_index + 1);
        }
        let index = current.context("ledger value before [[unproven]]")?;
        let (key, value) = key_value(line).with_context(|| format!("line {}", line_index + 1))?;
        let value = parse_string(value)?;
        match key {
            "path" => set_once(&mut rows[index].path, value, key)?,
            "reason" => set_once(&mut reasons[index], value, key)?,
            "owner" => set_once(&mut rows[index].owner, value, key)?,
            "expiry" => set_once(&mut rows[index].expiry, value, key)?,
            _ => bail!("line {}: unknown ledger key {key}", line_index + 1),
        }
    }
    let mut paths = BTreeSet::new();
    rows.into_iter()
        .zip(reasons)
        .map(|(row, reason)| {
            let path = required(row.path, "unproven.path")?;
            validate_path(&path)?;
            if !paths.insert(path.clone()) {
                bail!("duplicate ledger path {path}");
            }
            let reason = nonempty(required(reason, "unproven.reason")?, "unproven.reason")?;
            if reason.len() < 24
                || !["not yet modeled", "unsupported", "widening", "precision"]
                    .iter()
                    .any(|term| reason.contains(term))
            {
                bail!("ledger reason for {path} does not identify a verifier imprecision");
            }
            let owner = nonempty(required(row.owner, "unproven.owner")?, "unproven.owner")?;
            let expiry = required(row.expiry, "unproven.expiry")?;
            validate_unexpired(&expiry)?;
            Ok(LedgerRow {
                path,
                reason,
                owner,
                expiry,
            })
        })
        .collect()
}

fn parse_inputs(value: &str) -> anyhow::Result<Vec<InputContract>> {
    let mut names = BTreeSet::new();
    parse_string_array(value)?
        .into_iter()
        .map(|item| {
            let parts: Vec<_> = item.split(':').collect();
            let input = match parts.as_slice() {
                ["value", name, min, max] => InputContract::Value(range(name, min, max)?),
                ["uniform", name, min, max] => InputContract::Uniform(range(name, min, max)?),
                ["texture", name, min, max, rank, dimensions @ ..] => {
                    let rank = positive_u32(rank, "texture rank")? as usize;
                    if dimensions.len() != rank {
                        bail!("texture {name} rank does not match its dimensions");
                    }
                    InputContract::Texture {
                        range: range(name, min, max)?,
                        dimensions: dimensions
                            .iter()
                            .map(|value| positive_u32(value, "texture dimension"))
                            .collect::<anyhow::Result<_>>()?,
                    }
                }
                ["buffer", name, min, max, length] => InputContract::Buffer {
                    range: range(name, min, max)?,
                    min_length: positive_u32(length, "buffer length")?,
                },
                ["sampler", name] if !name.is_empty() => InputContract::Sampler((*name).into()),
                _ => bail!("malformed input {item:?}"),
            };
            if !names.insert(input.name().to_string()) {
                bail!("duplicate input {}", input.name());
            }
            Ok(input)
        })
        .collect()
}

fn parse_outputs(value: &str) -> anyhow::Result<Vec<RangeContract>> {
    let mut names = BTreeSet::new();
    parse_string_array(value)?
        .into_iter()
        .map(|item| {
            let parts: Vec<_> = item.split(':').collect();
            let [name, min, max] = parts.as_slice() else {
                bail!("malformed output {item:?}");
            };
            let range = range(name, min, max)?;
            if !names.insert(range.name.clone()) {
                bail!("duplicate output {}", range.name);
            }
            Ok(range)
        })
        .collect()
}

fn parse_invariants(value: &str) -> anyhow::Result<Vec<InvariantContract>> {
    let facts = parse_unique_strings(value)?;
    facts
        .into_iter()
        .map(|fact| {
            let parts: Vec<_> = fact.split(':').collect();
            let finite = |value: &str| -> anyhow::Result<f32> {
                let value = value.parse::<f32>().context("invalid invariant bound")?;
                if !value.is_finite() {
                    bail!("invariant bound must be finite");
                }
                Ok(value)
            };
            let nonempty = |value: &str| -> anyhow::Result<String> {
                if value.is_empty() {
                    bail!("invariant value name is empty");
                }
                Ok(value.into())
            };
            match parts.as_slice() {
                ["ge", value, minimum] => Ok(InvariantContract::GreaterEqual {
                    value: nonempty(value)?,
                    minimum: finite(minimum)?,
                }),
                ["abs_ge", value, minimum] => Ok(InvariantContract::AbsGreaterEqual {
                    value: nonempty(value)?,
                    minimum: finite(minimum)?,
                }),
                ["sum_abs_ge", left, right, minimum] => Ok(InvariantContract::SumAbsGreaterEqual {
                    left: nonempty(left)?,
                    right: nonempty(right)?,
                    minimum: finite(minimum)?,
                }),
                ["difference_ge", left, right, minimum] => {
                    Ok(InvariantContract::DifferenceGreaterEqual {
                        left: nonempty(left)?,
                        right: nonempty(right)?,
                        minimum: finite(minimum)?,
                    })
                }
                ["length_product", buffer, width, height] => {
                    Ok(InvariantContract::LengthAtLeastProduct {
                        buffer: nonempty(buffer)?,
                        width: nonempty(width)?,
                        height: nonempty(height)?,
                    })
                }
                ["dimensions_cover", texture, width, height] => {
                    Ok(InvariantContract::DimensionsCover {
                        texture: nonempty(texture)?,
                        width: nonempty(width)?,
                        height: nonempty(height)?,
                    })
                }
                ["same_dimensions", left, right] => Ok(InvariantContract::SameDimensions {
                    left: nonempty(left)?,
                    right: nonempty(right)?,
                }),
                ["count_within", count, buffer] => Ok(InvariantContract::CountWithin {
                    count: nonempty(count)?,
                    buffer: nonempty(buffer)?,
                }),
                ["indices_within", indices, target] => Ok(InvariantContract::IndicesWithin {
                    indices: nonempty(indices)?,
                    target: nonempty(target)?,
                }),
                _ => bail!("unsupported invariant {fact:?}"),
            }
        })
        .collect()
}

fn parse_unique_strings(value: &str) -> anyhow::Result<Vec<String>> {
    let values = parse_string_array(value)?;
    let unique: BTreeSet<_> = values.iter().collect();
    if unique.len() != values.len() || values.iter().any(|value| value.trim().is_empty()) {
        bail!("string facts must be nonempty and unique");
    }
    Ok(values)
}

fn range(name: &str, min: &str, max: &str) -> anyhow::Result<RangeContract> {
    if name.is_empty() {
        bail!("range name is empty");
    }
    let min = min.parse::<f32>().context("invalid range minimum")?;
    let max = max.parse::<f32>().context("invalid range maximum")?;
    if !min.is_finite() || !max.is_finite() || min > max {
        bail!("range {name} must have finite ordered bounds");
    }
    Ok(RangeContract {
        name: name.into(),
        min,
        max,
    })
}

fn parse_string_array(value: &str) -> anyhow::Result<Vec<String>> {
    let value = value.trim();
    let inner = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .context("expected a string array")?
        .trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let mut values = Vec::new();
    let mut quoted = false;
    let mut start = 0;
    for (index, ch) in inner.char_indices() {
        if ch == '"' {
            quoted = !quoted;
        } else if ch == ',' && !quoted {
            values.push(parse_string(inner[start..index].trim())?);
            start = index + 1;
        }
    }
    if quoted {
        bail!("unterminated string array item");
    }
    values.push(parse_string(inner[start..].trim())?);
    Ok(values)
}

fn parse_string(value: &str) -> anyhow::Result<String> {
    let value = value.trim();
    let inner = value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .context("expected a quoted string")?;
    if inner.contains('"') || inner.contains('\\') {
        bail!("escapes are not supported in proof contracts");
    }
    Ok(inner.into())
}

fn key_value(line: &str) -> anyhow::Result<(&str, &str)> {
    let (key, value) = line.split_once('=').context("expected key = value")?;
    let key = key.trim();
    if key.is_empty() || value.trim().is_empty() {
        bail!("empty key or value");
    }
    Ok((key, value.trim()))
}

fn set_once<T>(slot: &mut Option<T>, value: T, key: &str) -> anyhow::Result<()> {
    if slot.replace(value).is_some() {
        bail!("duplicate key {key}");
    }
    Ok(())
}

fn required<T>(value: Option<T>, name: &str) -> anyhow::Result<T> {
    value.with_context(|| format!("missing required {name}"))
}

fn nonempty(value: String, name: &str) -> anyhow::Result<String> {
    if value.trim().is_empty() {
        bail!("{name} is empty");
    }
    Ok(value)
}

fn positive_u32(value: &str, name: &str) -> anyhow::Result<u32> {
    let value = value
        .parse::<u32>()
        .with_context(|| format!("invalid {name}"))?;
    if value == 0 {
        bail!("{name} must be positive");
    }
    Ok(value)
}

fn validate_path(path: &str) -> anyhow::Result<()> {
    if !path.starts_with("src/shaders/") && !path.starts_with("tests/data/shader_proofs/") {
        bail!("invalid shader path {path}");
    }
    if !path.ends_with(".wgsl") || path.contains("..") || path.contains('\\') {
        bail!("invalid shader path {path}");
    }
    Ok(())
}

fn validate_unexpired(expiry: &str) -> anyhow::Result<()> {
    let expiry_days = parse_date(expiry)?;
    let today_days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_secs()
        / 86_400;
    if expiry_days < today_days as i64 {
        bail!("expired on {expiry}");
    }
    Ok(())
}

fn parse_date(value: &str) -> anyhow::Result<i64> {
    if value.len() != 10 || value.as_bytes()[4] != b'-' || value.as_bytes()[7] != b'-' {
        bail!("invalid ISO date {value}");
    }
    let mut parts = value.split('-');
    let year = parts.next().context("missing year")?.parse::<i64>()?;
    let month = parts.next().context("missing month")?.parse::<u32>()?;
    let day = parts.next().context("missing day")?.parse::<u32>()?;
    if parts.next().is_some() || !(1970..=9999).contains(&year) || !(1..=12).contains(&month) {
        bail!("invalid ISO date {value}");
    }
    let month_days = [
        31,
        28 + u32::from(is_leap(year)),
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day == 0 || day > month_days[month as usize - 1] {
        bail!("invalid ISO date {value}");
    }
    let adjusted_year = year - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let shifted_month = month as i64 + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i64 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Ok(era * 146_097 + day_of_era - 719_468)
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
[module]
path = "src/shaders/example.wgsl"
owner = "render-quality"
expiry = "2027-01-17"

[[entry]]
name = "main"
proof_status = "proven"
inputs = ["value:arg:-1:1", "uniform:params:0:8", "texture:image:0:1:2:1:1", "buffer:data:-2:2:4", "sampler:samp"]
outputs = ["color:0:1"]
invariants = ["ge:params.count:1"]
requires_guards = ["max(denom, 1e-6)"]
"#;

    #[test]
    fn parses_typed_contract() {
        let contract = parse_contract(VALID).unwrap();
        assert_eq!(contract.entries.len(), 1);
        assert_eq!(contract.entries[0].inputs.len(), 5);
        assert_eq!(contract.entries[0].outputs[0].name, "color");
        assert!(contract.entries[0].input("missing").is_none());
    }

    #[test]
    fn contract_rejects_unknown_missing_duplicate_and_bad_ranges() {
        assert!(parse_contract(&VALID.replace("outputs =", "surprise =")).is_err());
        assert!(parse_contract(&VALID.replace("owner = \"render-quality\"\n", "")).is_err());
        assert!(parse_contract(&VALID.replace(
            "proof_status = \"proven\"",
            "proof_status = \"proven\"\nproof_status = \"proven\""
        ))
        .is_err());
        assert!(parse_contract(&VALID.replace("value:arg:-1:1", "value:arg:2:1")).is_err());
        assert!(parse_contract(&VALID.replace("value:arg:-1:1", "value:arg:NaN:1")).is_err());
        assert!(parse_contract(
            &VALID.replace("texture:image:0:1:2:1:1", "texture:image:0:1:2:0:1")
        )
        .is_err());
        assert!(
            parse_contract(&VALID.replace("buffer:data:-2:2:4", "buffer:data:-2:2:0")).is_err()
        );
        assert!(parse_contract(&VALID.replace(
            "invariants = [\"ge:params.count:1\"]",
            "invariants = [\"ge:params.count:1\", \"ge:params.count:1\"]"
        ))
        .is_err());
        assert!(parse_contract(&VALID.replace(
            "invariants = [\"ge:params.count:1\"]",
            "invariants = [\"prose is not a proof fact\"]"
        ))
        .is_err());
    }

    #[test]
    fn contract_requires_explicit_inputs_and_outputs() {
        assert!(parse_contract(&VALID.replace(
            "inputs = [\"value:arg:-1:1\", \"uniform:params:0:8\", \"texture:image:0:1:2:1:1\", \"buffer:data:-2:2:4\", \"sampler:samp\"]\n",
            ""
        ))
        .is_err());
        assert!(parse_contract(&VALID.replace("outputs = [\"color:0:1\"]\n", "")).is_err());
    }

    const LEDGER: &str = r#"
[[unproven]]
path = "src/shaders/a.wgsl"
reason = "texture gather precision is not modeled"
owner = "render-quality"
expiry = "2027-01-17"
"#;

    #[test]
    fn ledger_is_structural_and_fail_closed() {
        let rows = parse_ledger(LEDGER).unwrap();
        assert_eq!(rows[0].path, "src/shaders/a.wgsl");
        assert!(parse_ledger(&LEDGER.replace("reason =", "unknown =")).is_err());
        assert!(parse_ledger(&LEDGER.replace("owner = \"render-quality\"\n", "")).is_err());
        assert!(parse_ledger(
            &LEDGER.replace("texture gather precision is not modeled", "too vague")
        )
        .is_err());
        assert!(parse_ledger(&format!("{LEDGER}\n{LEDGER}")).is_err());
        assert!(parse_ledger(&LEDGER.replace("2027-01-17", "2020-01-01")).is_err());
        assert!(parse_ledger(&LEDGER.replace(
            "texture gather precision is not modeled",
            "outside PROBATUM v1 proven ratchet; conservative abstract domain not yet precise enough for this module"
        ))
        .is_err());
    }

    #[test]
    fn every_committed_contract_uses_the_strict_schema() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for file in std::fs::read_dir(root.join("shaders/contracts")).unwrap() {
            let file = file.unwrap().path();
            if file
                .extension()
                .is_some_and(|extension| extension == "toml")
            {
                parse_contract(&std::fs::read_to_string(&file).unwrap())
                    .unwrap_or_else(|error| panic!("{}: {error:#}", file.display()));
            }
        }
    }

    #[test]
    fn determinism_contract_covers_every_det_lemma() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let source =
            std::fs::read_to_string(root.join("src/shaders/includes/determinism.wgsl")).unwrap();
        let contract = parse_contract(
            &std::fs::read_to_string(root.join("shaders/contracts/determinism.toml")).unwrap(),
        )
        .unwrap();
        let source_names: BTreeSet<_> = source
            .lines()
            .filter_map(|line| line.strip_prefix("fn det_"))
            .filter_map(|line| line.split_once('(').map(|(name, _)| format!("det_{name}")))
            .collect();
        let contract_names: BTreeSet<_> = contract
            .entries
            .iter()
            .map(|entry| entry.name.clone())
            .collect();
        assert_eq!(contract_names, source_names);
        assert_eq!(contract_names.len(), 31);
    }
}
