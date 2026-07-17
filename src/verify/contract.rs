use anyhow::{bail, Context};
use std::collections::{BTreeMap, BTreeSet};
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
        dimensions: Vec<DimensionContract>,
    },
    Buffer {
        range: RangeContract,
        length: BufferLength,
    },
    Sampler(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DimensionContract {
    Minimum(u32),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BufferLength {
    Fixed(u32),
    Dynamic,
    Symbol(String),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedKind {
    Value,
    Uniform,
    Texture,
    Buffer,
    Sampler,
}

pub(crate) fn validate_contract_semantics(
    module: &naga::Module,
    entry_name: &str,
    contract: &EntryContract,
) -> anyhow::Result<()> {
    let (root, is_entry) = module
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_name)
        .map(|entry| (&entry.function, true))
        .or_else(|| {
            module
                .functions
                .iter()
                .find(|(_, function)| function.name.as_deref() == Some(entry_name))
                .map(|(_, function)| (function, false))
        })
        .with_context(|| format!("Naga module has no function {entry_name:?}"))?;

    let mut functions = Vec::new();
    let mut pending = called_functions(&root.body);
    let mut seen = BTreeSet::new();
    while let Some(handle) = pending.pop() {
        if seen.insert(handle.index()) {
            let function = &module.functions[handle];
            pending.extend(called_functions(&function.body));
            functions.push(function);
        }
    }

    let mut expected = BTreeMap::new();
    collect_argument_inputs(module, root, &mut expected);
    for function in std::iter::once(root).chain(functions.iter().copied()) {
        collect_global_inputs(module, function, &mut expected)?;
    }

    let actual: BTreeMap<_, _> = contract
        .inputs
        .iter()
        .map(|input| (input.name().to_string(), input_kind(input)))
        .collect();
    let missing: Vec<_> = expected
        .keys()
        .filter(|name| !actual.contains_key(*name))
        .cloned()
        .collect();
    let extra: Vec<_> = actual
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .cloned()
        .collect();
    let wrong_kind: Vec<_> = expected
        .iter()
        .filter_map(|(name, kind)| {
            actual
                .get(name)
                .filter(|actual| *actual != kind)
                .map(|actual| format!("{name}: expected {kind:?}, got {actual:?}"))
        })
        .collect();
    anyhow::ensure!(
        missing.is_empty() && extra.is_empty() && wrong_kind.is_empty(),
        "contract input mismatch; missing={missing:?}, extra={extra:?}, wrong_kind={wrong_kind:?}"
    );
    let mut outputs = BTreeSet::new();
    if let Some(result) = &root.result {
        add_output_leaves(
            module,
            result.ty,
            result.binding.as_ref(),
            is_entry,
            &mut outputs,
        );
    }
    for function in std::iter::once(root).chain(functions.iter().copied()) {
        collect_written_globals(module, function, &mut outputs);
    }
    let actual_outputs: BTreeSet<_> = contract
        .outputs
        .iter()
        .map(|range| range.name.clone())
        .collect();
    anyhow::ensure!(
        outputs == actual_outputs,
        "contract output mismatch; expected={outputs:?}, actual={actual_outputs:?}"
    );
    let all_functions: Vec<_> = std::iter::once(root)
        .chain(functions.iter().copied())
        .collect();
    validate_contract_references(module, root, &expected, &outputs, contract)?;
    validate_resource_shapes(module, &all_functions, contract)?;
    Ok(())
}

fn validate_contract_references(
    module: &naga::Module,
    root: &naga::Function,
    inputs: &BTreeMap<String, ExpectedKind>,
    outputs: &BTreeSet<String>,
    contract: &EntryContract,
) -> anyhow::Result<()> {
    let known: BTreeSet<_> = inputs.keys().chain(outputs.iter()).cloned().collect();
    let valid = |reference: &str| reference_resolves(module, root, &known, reference);
    for input in &contract.inputs {
        match input {
            InputContract::Texture { dimensions, .. } => {
                for dimension in dimensions {
                    if let DimensionContract::Symbol(symbol) = dimension {
                        anyhow::ensure!(valid(symbol), "unknown texture dimension symbol {symbol}");
                    }
                }
            }
            InputContract::Buffer {
                length: BufferLength::Symbol(symbol),
                ..
            } => {
                anyhow::ensure!(valid(symbol), "unknown buffer length symbol {symbol}");
            }
            _ => {}
        }
    }
    for reference in contract.invariants.iter().flat_map(invariant_references) {
        anyhow::ensure!(valid(reference), "unknown invariant reference {reference}");
    }
    validate_invariant_directions(module, root, &known, contract)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum ResolvedReference {
    Type(naga::Handle<naga::Type>),
    Scalar(naga::Scalar),
}

fn reference_resolves(
    module: &naga::Module,
    root: &naga::Function,
    known: &BTreeSet<String>,
    reference: &str,
) -> bool {
    resolve_reference_type(module, root, known, reference).is_some()
}

fn resolve_reference_type(
    module: &naga::Module,
    root: &naga::Function,
    known: &BTreeSet<String>,
    reference: &str,
) -> Option<ResolvedReference> {
    if known.contains(reference) {
        return known_symbol_type(module, root, reference).map(ResolvedReference::Type);
    }
    let Some(base) = known
        .iter()
        .filter(|name| {
            reference.starts_with(&format!("{name}.")) || reference.starts_with(&format!("{name}["))
        })
        .max_by_key(|name| name.len())
    else {
        return None;
    };
    let ty = known_symbol_type(module, root, base)?;
    resolve_type_suffix(module, ty, &reference[base.len()..])
}

fn known_symbol_type(
    module: &naga::Module,
    root: &naga::Function,
    name: &str,
) -> Option<naga::Handle<naga::Type>> {
    let (head, tail) = name.split_once('.').map_or((name, ""), |parts| parts);
    if let Some(argument) = root
        .arguments
        .iter()
        .find(|argument| argument.name.as_deref() == Some(head))
    {
        return traverse_named_members(module, argument.ty, tail);
    }
    if let Some((_, global)) = module
        .global_variables
        .iter()
        .find(|(_, global)| global.name.as_deref() == Some(head))
    {
        return traverse_named_members(module, global.ty, tail);
    }
    let result = root.result.as_ref()?;
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        return None;
    };
    members
        .iter()
        .find(|member| member.name.as_deref() == Some(name))
        .map(|member| member.ty)
}

fn traverse_named_members(
    module: &naga::Module,
    mut ty: naga::Handle<naga::Type>,
    path: &str,
) -> Option<naga::Handle<naga::Type>> {
    if path.is_empty() {
        return Some(ty);
    }
    for name in path.split('.') {
        let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner else {
            return None;
        };
        ty = members
            .iter()
            .find(|member| member.name.as_deref() == Some(name))?
            .ty;
    }
    Some(ty)
}

fn resolve_type_suffix(
    module: &naga::Module,
    mut ty: naga::Handle<naga::Type>,
    mut suffix: &str,
) -> Option<ResolvedReference> {
    while !suffix.is_empty() {
        if let Some(rest) = suffix.strip_prefix("[]") {
            ty = match module.types[ty].inner {
                naga::TypeInner::Array { base, .. }
                | naga::TypeInner::BindingArray { base, .. } => base,
                _ => return None,
            };
            suffix = rest;
            continue;
        }
        let Some(rest) = suffix.strip_prefix('.') else {
            return None;
        };
        let end = rest.find(['.', '[']).unwrap_or(rest.len());
        let member = &rest[..end];
        suffix = &rest[end..];
        match &module.types[ty].inner {
            naga::TypeInner::Struct { members, .. } => {
                let Some(found) = members
                    .iter()
                    .find(|candidate| candidate.name.as_deref() == Some(member))
                else {
                    return None;
                };
                ty = found.ty;
            }
            naga::TypeInner::Vector { size, scalar } => {
                let component = match member {
                    "x" | "r" => 1,
                    "y" | "g" => 2,
                    "z" | "b" => 3,
                    "w" | "a" => 4,
                    _ => return None,
                };
                let width = match size {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                return (component <= width && suffix.is_empty())
                    .then_some(ResolvedReference::Scalar(*scalar));
            }
            naga::TypeInner::Image { dim, arrayed, .. } => {
                let valid = match member {
                    "width" => true,
                    "height" => !matches!(dim, naga::ImageDimension::D1),
                    "depth" => matches!(dim, naga::ImageDimension::D3),
                    "layers" => *arrayed,
                    _ => false,
                };
                return (valid && suffix.is_empty()).then_some(ResolvedReference::Scalar(
                    naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: 4,
                    },
                ));
            }
            naga::TypeInner::Array {
                size: naga::ArraySize::Dynamic,
                ..
            } => {
                return (member == "length" && suffix.is_empty()).then_some(
                    ResolvedReference::Scalar(naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: 4,
                    }),
                )
            }
            _ => return None,
        }
    }
    Some(ResolvedReference::Type(ty))
}

fn validate_invariant_directions(
    module: &naga::Module,
    root: &naga::Function,
    known: &BTreeSet<String>,
    contract: &EntryContract,
) -> anyhow::Result<()> {
    let kind = |name: &str| contract.inputs.iter().find(|input| input.name() == name);
    let resolved = |name: &str| {
        resolve_reference_type(module, root, known, name)
            .with_context(|| format!("invariant reference {name} has no Naga type"))
    };
    for invariant in &contract.invariants {
        match invariant {
            InvariantContract::GreaterEqual { value, .. } => {
                anyhow::ensure!(
                    numeric_shape(module, resolved(value)?).is_some_and(|shape| shape.lanes == 1),
                    "ge operand {value} is not a numeric scalar"
                );
            }
            InvariantContract::AbsGreaterEqual { value, .. } => {
                anyhow::ensure!(
                    numeric_shape(module, resolved(value)?).is_some_and(|shape| {
                        shape.lanes == 1
                            && matches!(
                                shape.kind,
                                naga::ScalarKind::Float | naga::ScalarKind::Sint
                            )
                    }),
                    "abs_ge operand {value} is not a signed numeric scalar"
                );
            }
            InvariantContract::NormGreaterEqual { value, .. } => {
                anyhow::ensure!(
                    numeric_shape(module, resolved(value)?).is_some_and(|shape| {
                        shape.lanes > 1 && shape.kind == naga::ScalarKind::Float
                    }),
                    "norm_ge operand {value} is not a floating-point vector"
                );
            }
            InvariantContract::DistanceGreaterEqual { left, right, .. } => {
                let left_shape = numeric_shape(module, resolved(left)?);
                let right_shape = numeric_shape(module, resolved(right)?);
                anyhow::ensure!(
                    left_shape == right_shape
                        && left_shape.is_some_and(|shape| {
                            shape.lanes > 1 && shape.kind == naga::ScalarKind::Float
                        }),
                    "distance_ge operands {left} and {right} are not compatible float vectors"
                );
            }
            InvariantContract::SumAbsGreaterEqual { left, right, .. }
            | InvariantContract::DifferenceGreaterEqual { left, right, .. } => {
                let left_shape = numeric_shape(module, resolved(left)?);
                let right_shape = numeric_shape(module, resolved(right)?);
                anyhow::ensure!(
                    left_shape == right_shape
                        && left_shape.is_some_and(|shape| {
                            matches!(shape.kind, naga::ScalarKind::Float | naga::ScalarKind::Sint)
                        }),
                    "relational operands {left} and {right} have incompatible dimensions or types"
                );
            }
            InvariantContract::DimensionsCover {
                texture,
                width,
                height,
            } => {
                anyhow::ensure!(
                    image_shape(module, resolved(texture)?).is_some(),
                    "dimensions_cover target {texture} is not a texture"
                );
                for dimension in [width, height] {
                    anyhow::ensure!(
                        is_unsigned_scalar(module, resolved(dimension)?),
                        "dimensions_cover value {dimension} is not an unsigned scalar"
                    );
                }
            }
            InvariantContract::SameDimensions { left, right } => {
                let left_shape = image_shape(module, resolved(left)?);
                let right_shape = image_shape(module, resolved(right)?);
                anyhow::ensure!(
                    left_shape.is_some() && left_shape == right_shape,
                    "same_dimensions operands {left} and {right} are not compatible textures"
                );
            }
            InvariantContract::CountWithin { count, buffer } => {
                anyhow::ensure!(
                    matches!(kind(buffer), Some(InputContract::Buffer { .. })),
                    "count_within target {buffer} is not a buffer"
                );
                anyhow::ensure!(
                    !matches!(
                        kind(count),
                        Some(
                            InputContract::Buffer { .. }
                                | InputContract::Texture { .. }
                                | InputContract::Sampler(_)
                        )
                    ),
                    "count_within count {count} is not a numeric value"
                );
                anyhow::ensure!(
                    is_unsigned_scalar(module, resolved(count)?),
                    "count_within count {count} is not an unsigned integer scalar"
                );
            }
            InvariantContract::IndicesWithin { indices, target } => {
                anyhow::ensure!(
                    matches!(kind(indices), Some(InputContract::Buffer { .. })),
                    "indices_within source {indices} is not a buffer"
                );
                anyhow::ensure!(
                    matches!(kind(target), Some(InputContract::Buffer { .. })),
                    "indices_within target {target} is not a buffer"
                );
                let element = storage_array_element(module, indices)
                    .with_context(|| format!("indices_within source {indices} is not an array"))?;
                anyhow::ensure!(
                    matches!(
                        module.types[element].inner,
                        naga::TypeInner::Scalar(naga::Scalar {
                            kind: naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                            ..
                        })
                    ),
                    "indices_within source {indices} does not contain integer indices"
                );
            }
            InvariantContract::LengthAtLeastProduct {
                buffer,
                width,
                height,
            } => {
                anyhow::ensure!(
                    matches!(kind(buffer), Some(InputContract::Buffer { .. })),
                    "length invariant target {buffer} is not a buffer"
                );
                for factor in [width, height] {
                    anyhow::ensure!(
                        is_unsigned_scalar(module, resolved(factor)?),
                        "length factor {factor} is not an unsigned scalar"
                    );
                }
            }
            InvariantContract::LengthAtLeastProductCastU32 {
                buffer,
                width,
                height,
            } => {
                anyhow::ensure!(
                    matches!(kind(buffer), Some(InputContract::Buffer { .. })),
                    "length invariant target {buffer} is not a buffer"
                );
                for factor in [width, height] {
                    anyhow::ensure!(
                        numeric_shape(module, resolved(factor)?).is_some_and(|shape| {
                            shape.lanes == 1 && shape.kind == naga::ScalarKind::Float
                        }),
                        "u32-cast length factor {factor} is not a float scalar"
                    );
                }
            }
            InvariantContract::LengthAtLeastProductMaxOne {
                buffer, factors, ..
            } => {
                anyhow::ensure!(
                    matches!(kind(buffer), Some(InputContract::Buffer { .. })),
                    "length invariant target {buffer} is not a buffer"
                );
                for factor in factors {
                    anyhow::ensure!(
                        is_unsigned_scalar(module, resolved(factor)?),
                        "length factor {factor} is not an unsigned scalar"
                    );
                }
            }
            InvariantContract::AdditiveTextureBudget {
                output,
                diffuse,
                indirect,
                specular,
                reflection,
                maximum,
            } => {
                let shapes = [output, diffuse, indirect, specular, reflection]
                    .into_iter()
                    .map(|name| resolved(name).map(|reference| image_shape(module, reference)))
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .into_iter()
                    .collect::<Option<Vec<_>>>();
                let Some(shapes) = shapes else {
                    bail!("additive_texture_budget operands must all be textures");
                };
                anyhow::ensure!(
                    shapes.iter().all(|shape| *shape == shapes[0]),
                    "additive_texture_budget textures have incompatible dimensions"
                );
                anyhow::ensure!(
                    matches!(
                        image_class(module, resolved(output)?),
                        Some(naga::ImageClass::Storage {
                            format: naga::StorageFormat::Rgba16Float,
                            ..
                        })
                    ),
                    "additive texture budget output is not rgba16float storage"
                );
                for source in [diffuse, indirect, specular, reflection] {
                    anyhow::ensure!(
                        matches!(
                            image_class(module, resolved(source)?),
                            Some(naga::ImageClass::Sampled {
                                kind: naga::ScalarKind::Float,
                                ..
                            })
                        ),
                        "additive texture budget source {source} is not float-sampled"
                    );
                }
                let output_max = contract
                    .outputs
                    .iter()
                    .find(|range| range.name == *output)
                    .map(|range| range.max)
                    .context("additive texture budget output has no output range")?;
                anyhow::ensure!(
                    *maximum >= 0.0 && *maximum <= 65_504.0 && output_max >= *maximum,
                    "additive texture budget exceeds the rgba16float output range"
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NumericShape {
    kind: naga::ScalarKind,
    width: u8,
    lanes: u8,
}

fn numeric_shape(module: &naga::Module, reference: ResolvedReference) -> Option<NumericShape> {
    let (scalar, lanes) = match reference {
        ResolvedReference::Scalar(scalar) => (scalar, 1),
        ResolvedReference::Type(ty) => match module.types[ty].inner {
            naga::TypeInner::Scalar(scalar) => (scalar, 1),
            naga::TypeInner::Vector { size, scalar } => {
                let lanes = match size {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                (scalar, lanes)
            }
            _ => return None,
        },
    };
    (!matches!(
        scalar.kind,
        naga::ScalarKind::Bool | naga::ScalarKind::AbstractInt | naga::ScalarKind::AbstractFloat
    ))
    .then_some(NumericShape {
        kind: scalar.kind,
        width: scalar.width,
        lanes,
    })
}

fn is_unsigned_scalar(module: &naga::Module, reference: ResolvedReference) -> bool {
    numeric_shape(module, reference)
        .is_some_and(|shape| shape.lanes == 1 && shape.kind == naga::ScalarKind::Uint)
}

fn image_shape(
    module: &naga::Module,
    reference: ResolvedReference,
) -> Option<(naga::ImageDimension, bool)> {
    let ResolvedReference::Type(ty) = reference else {
        return None;
    };
    match module.types[ty].inner {
        naga::TypeInner::Image { dim, arrayed, .. } => Some((dim, arrayed)),
        _ => None,
    }
}

fn image_class(module: &naga::Module, reference: ResolvedReference) -> Option<naga::ImageClass> {
    let ResolvedReference::Type(ty) = reference else {
        return None;
    };
    match module.types[ty].inner {
        naga::TypeInner::Image { class, .. } => Some(class),
        _ => None,
    }
}

fn storage_array_element(module: &naga::Module, name: &str) -> Option<naga::Handle<naga::Type>> {
    let (_, global) = module
        .global_variables
        .iter()
        .find(|(_, global)| global.name.as_deref() == Some(name))?;
    array_element(module, global.ty)
}

fn array_element(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Option<naga::Handle<naga::Type>> {
    match &module.types[ty].inner {
        naga::TypeInner::Array { base, .. } | naga::TypeInner::BindingArray { base, .. } => {
            Some(*base)
        }
        naga::TypeInner::Struct { members, .. } => array_element(module, members.last()?.ty),
        _ => None,
    }
}

fn invariant_references(invariant: &InvariantContract) -> Vec<&str> {
    match invariant {
        InvariantContract::GreaterEqual { value, .. }
        | InvariantContract::AbsGreaterEqual { value, .. }
        | InvariantContract::NormGreaterEqual { value, .. } => vec![value],
        InvariantContract::SumAbsGreaterEqual { left, right, .. }
        | InvariantContract::DifferenceGreaterEqual { left, right, .. }
        | InvariantContract::DistanceGreaterEqual { left, right, .. }
        | InvariantContract::SameDimensions { left, right }
        | InvariantContract::IndicesWithin {
            indices: left,
            target: right,
        } => vec![left, right],
        InvariantContract::LengthAtLeastProduct {
            buffer,
            width,
            height,
        }
        | InvariantContract::LengthAtLeastProductCastU32 {
            buffer,
            width,
            height,
        }
        | InvariantContract::DimensionsCover {
            texture: buffer,
            width,
            height,
        } => {
            vec![buffer, width, height]
        }
        InvariantContract::LengthAtLeastProductMaxOne {
            buffer, factors, ..
        } => std::iter::once(buffer.as_str())
            .chain(factors.iter().map(String::as_str))
            .collect(),
        InvariantContract::AdditiveTextureBudget {
            output,
            diffuse,
            indirect,
            specular,
            reflection,
            ..
        } => vec![output, diffuse, indirect, specular, reflection],
        InvariantContract::CountWithin { count, buffer } => vec![count, buffer],
    }
}

fn validate_resource_shapes(
    module: &naga::Module,
    functions: &[&naga::Function],
    contract: &EntryContract,
) -> anyhow::Result<()> {
    let mut image_loads = BTreeSet::new();
    let mut length_queries = BTreeSet::new();
    for function in functions {
        for (_, expression) in function.expressions.iter() {
            let root = match expression {
                naga::Expression::ImageLoad { image, .. } => global_access(function, *image),
                naga::Expression::ArrayLength(pointer) => global_access(function, *pointer),
                _ => None,
            };
            if let Some((handle, _)) = root {
                if let Some(name) = &module.global_variables[handle].name {
                    match expression {
                        naga::Expression::ImageLoad { .. } => {
                            image_loads.insert(name.clone());
                        }
                        naga::Expression::ArrayLength(_) => {
                            length_queries.insert(name.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    for input in &contract.inputs {
        let Some((_, global)) = module
            .global_variables
            .iter()
            .find(|(_, global)| global.name.as_deref() == Some(input.name()))
        else {
            continue;
        };
        match input {
            InputContract::Texture { dimensions, .. } => {
                let naga::TypeInner::Image { dim, arrayed, .. } = module.types[global.ty].inner
                else {
                    bail!(
                        "{} is contracted as a texture but Naga disagrees",
                        input.name()
                    );
                };
                let rank = match dim {
                    naga::ImageDimension::D1 => 1,
                    naga::ImageDimension::D2 | naga::ImageDimension::Cube => 2,
                    naga::ImageDimension::D3 => 3,
                } + usize::from(arrayed);
                anyhow::ensure!(
                    dimensions.len() == rank,
                    "{} texture rank mismatch",
                    input.name()
                );
                if image_loads.contains(input.name()) {
                    anyhow::ensure!(
                        dimensions
                            .iter()
                            .all(|dimension| matches!(dimension, DimensionContract::Symbol(_)))
                            || contract
                                .invariants
                                .iter()
                                .any(|invariant| invariant_bounds_texture(invariant, input.name())),
                        "textureLoad resource {} needs symbolic dimensions or a dimension relation",
                        input.name()
                    );
                }
            }
            InputContract::Buffer { length, .. } if has_runtime_array(module, global.ty) => {
                anyhow::ensure!(
                    !matches!(length, BufferLength::Fixed(_)),
                    "runtime array {} must use the explicit dynamic length",
                    input.name()
                );
                anyhow::ensure!(
                    matches!(length, BufferLength::Symbol(_))
                        || length_queries.contains(input.name())
                        || contract
                            .invariants
                            .iter()
                            .any(|invariant| invariant_bounds_buffer(invariant, input.name())),
                    "runtime array {} has no length/count/index relation",
                    input.name()
                );
            }
            InputContract::Buffer { length, .. } => {
                anyhow::ensure!(
                    !matches!(length, BufferLength::Dynamic | BufferLength::Symbol(_)),
                    "fixed buffer {} is marked dynamic",
                    input.name()
                );
            }
            _ => {}
        }
    }
    Ok(())
}

fn has_runtime_array(module: &naga::Module, ty: naga::Handle<naga::Type>) -> bool {
    match &module.types[ty].inner {
        naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            ..
        } => true,
        naga::TypeInner::Struct { members, .. } => members
            .last()
            .is_some_and(|member| has_runtime_array(module, member.ty)),
        _ => false,
    }
}

fn invariant_bounds_texture(invariant: &InvariantContract, name: &str) -> bool {
    match invariant {
        InvariantContract::DimensionsCover { texture, .. } => texture == name,
        InvariantContract::SameDimensions { left, right } => left == name || right == name,
        _ => false,
    }
}

fn invariant_bounds_buffer(invariant: &InvariantContract, name: &str) -> bool {
    match invariant {
        InvariantContract::LengthAtLeastProduct { buffer, .. } => buffer == name,
        InvariantContract::LengthAtLeastProductCastU32 { buffer, .. } => buffer == name,
        InvariantContract::LengthAtLeastProductMaxOne { buffer, .. } => buffer == name,
        InvariantContract::CountWithin { buffer, .. } => buffer == name,
        InvariantContract::IndicesWithin { target, .. } => target == name,
        _ => false,
    }
}

fn input_kind(input: &InputContract) -> ExpectedKind {
    match input {
        InputContract::Value(_) => ExpectedKind::Value,
        InputContract::Uniform(_) => ExpectedKind::Uniform,
        InputContract::Texture { .. } => ExpectedKind::Texture,
        InputContract::Buffer { .. } => ExpectedKind::Buffer,
        InputContract::Sampler(_) => ExpectedKind::Sampler,
    }
}

fn collect_argument_inputs(
    module: &naga::Module,
    function: &naga::Function,
    out: &mut BTreeMap<String, ExpectedKind>,
) {
    let mut paths = BTreeSet::new();
    let identity: Vec<_> = function
        .arguments
        .iter()
        .enumerate()
        .map(|(index, _)| Some((index as u32, Vec::new())))
        .collect();
    collect_mapped_argument_paths(module, function, function, &identity, &mut paths, 0);
    for path in &paths {
        if !paths
            .iter()
            .any(|other| other.starts_with(&format!("{path}.")))
        {
            out.insert(path.clone(), ExpectedKind::Value);
        }
    }
}

fn collect_mapped_argument_paths(
    module: &naga::Module,
    root: &naga::Function,
    function: &naga::Function,
    mapping: &[Option<(u32, Vec<u32>)>],
    paths: &mut BTreeSet<String>,
    depth: usize,
) {
    if depth > module.functions.len() {
        return;
    }
    for (handle, _) in function.expressions.iter() {
        let Some((local_argument, indices)) = argument_access(function, handle) else {
            continue;
        };
        let Some(Some((root_argument, prefix))) = mapping.get(local_argument as usize) else {
            continue;
        };
        let Some(argument) = root.arguments.get(*root_argument as usize) else {
            continue;
        };
        let mut full_path = prefix.clone();
        full_path.extend(indices);
        let name = argument
            .name
            .clone()
            .unwrap_or_else(|| format!("arg{root_argument}"));
        if matches!(
            module.types[argument.ty].inner,
            naga::TypeInner::Struct { .. }
        ) {
            if let Some(path) = named_member_path(module, argument.ty, &full_path) {
                paths.insert(format!("{name}.{path}"));
            }
        } else {
            paths.insert(name);
        }
    }

    let mut calls = Vec::new();
    walk_statements(&function.body, &mut |statement| {
        if let naga::Statement::Call {
            function,
            arguments,
            ..
        } = statement
        {
            calls.push((*function, arguments.clone()));
        }
    });
    for (callee_handle, actuals) in calls {
        let callee = &module.functions[callee_handle];
        let callee_mapping: Vec<_> = actuals
            .iter()
            .map(|actual| {
                let (local_argument, indices) = argument_access(function, *actual)?;
                let (root_argument, prefix) = mapping.get(local_argument as usize)?.as_ref()?;
                let mut full_path = prefix.clone();
                full_path.extend(indices);
                Some((*root_argument, full_path))
            })
            .collect();
        collect_mapped_argument_paths(module, root, callee, &callee_mapping, paths, depth + 1);
    }
}

fn argument_access(
    function: &naga::Function,
    handle: naga::Handle<naga::Expression>,
) -> Option<(u32, Vec<u32>)> {
    match function.expressions[handle] {
        naga::Expression::FunctionArgument(index) => Some((index, Vec::new())),
        naga::Expression::AccessIndex { base, index } => {
            let (argument, mut path) = argument_access(function, base)?;
            path.push(index);
            Some((argument, path))
        }
        naga::Expression::Access { base, .. } | naga::Expression::Load { pointer: base } => {
            argument_access(function, base)
        }
        _ => None,
    }
}

fn collect_global_inputs(
    module: &naga::Module,
    function: &naga::Function,
    out: &mut BTreeMap<String, ExpectedKind>,
) -> anyhow::Result<()> {
    let mut uniform_paths = BTreeMap::<naga::Handle<naga::GlobalVariable>, BTreeSet<String>>::new();
    for (handle, _) in function.expressions.iter() {
        let Some((global_handle, indices)) = global_access(function, handle) else {
            continue;
        };
        let global = &module.global_variables[global_handle];
        if global.binding.is_none() && !matches!(global.space, naga::AddressSpace::PushConstant) {
            continue;
        }
        if !global_is_readable(module, global) {
            continue;
        }
        let name = global
            .name
            .as_deref()
            .context("reachable global has no WGSL name")?;
        match global_kind(module, global) {
            ExpectedKind::Uniform => {
                if let Some(path) = named_member_path(module, global.ty, &indices) {
                    uniform_paths
                        .entry(global_handle)
                        .or_default()
                        .insert(format!("{name}.{path}"));
                } else if !matches!(
                    module.types[global.ty].inner,
                    naga::TypeInner::Struct { .. }
                ) {
                    out.insert(name.into(), ExpectedKind::Uniform);
                }
            }
            kind => {
                out.insert(name.into(), kind);
            }
        }
    }
    for paths in uniform_paths.into_values() {
        for path in &paths {
            if !paths
                .iter()
                .any(|other| other.starts_with(&format!("{path}.")))
            {
                out.insert(path.clone(), ExpectedKind::Uniform);
            }
        }
    }
    Ok(())
}

fn global_is_readable(module: &naga::Module, global: &naga::GlobalVariable) -> bool {
    match global.space {
        naga::AddressSpace::Storage { access } => access.contains(naga::StorageAccess::LOAD),
        naga::AddressSpace::Handle => match module.types[global.ty].inner {
            naga::TypeInner::Image {
                class: naga::ImageClass::Storage { access, .. },
                ..
            } => access.contains(naga::StorageAccess::LOAD),
            _ => true,
        },
        _ => true,
    }
}

fn global_kind(module: &naga::Module, global: &naga::GlobalVariable) -> ExpectedKind {
    match global.space {
        naga::AddressSpace::Uniform | naga::AddressSpace::PushConstant => ExpectedKind::Uniform,
        naga::AddressSpace::Storage { .. } => ExpectedKind::Buffer,
        naga::AddressSpace::Handle => match module.types[global.ty].inner {
            naga::TypeInner::Sampler { .. } => ExpectedKind::Sampler,
            naga::TypeInner::Image { .. } => ExpectedKind::Texture,
            _ => ExpectedKind::Value,
        },
        _ => ExpectedKind::Value,
    }
}

fn global_access(
    function: &naga::Function,
    handle: naga::Handle<naga::Expression>,
) -> Option<(naga::Handle<naga::GlobalVariable>, Vec<u32>)> {
    match function.expressions[handle] {
        naga::Expression::GlobalVariable(global) => Some((global, Vec::new())),
        naga::Expression::AccessIndex { base, index } => {
            let (global, mut path) = global_access(function, base)?;
            path.push(index);
            Some((global, path))
        }
        naga::Expression::Access { base, .. } | naga::Expression::Load { pointer: base } => {
            global_access(function, base)
        }
        _ => None,
    }
}

fn named_member_path(
    module: &naga::Module,
    mut ty: naga::Handle<naga::Type>,
    indices: &[u32],
) -> Option<String> {
    let mut names = Vec::new();
    for &index in indices {
        match &module.types[ty].inner {
            naga::TypeInner::Struct { members, .. } => {
                let member = members.get(index as usize)?;
                names.push(member.name.clone()?);
                ty = member.ty;
            }
            naga::TypeInner::Array { base, .. } => ty = *base,
            _ => break,
        }
    }
    (!names.is_empty()).then(|| names.join("."))
}

fn called_functions(block: &naga::Block) -> Vec<naga::Handle<naga::Function>> {
    let mut calls = Vec::new();
    walk_statements(block, &mut |statement| {
        if let naga::Statement::Call { function, .. } = statement {
            calls.push(*function);
        }
    });
    calls
}

fn collect_written_globals(
    module: &naga::Module,
    function: &naga::Function,
    outputs: &mut BTreeSet<String>,
) {
    walk_statements(&function.body, &mut |statement| {
        let root = match statement {
            naga::Statement::Store { pointer, .. } | naga::Statement::Atomic { pointer, .. } => {
                global_access(function, *pointer)
            }
            naga::Statement::ImageStore { image, .. } => global_access(function, *image),
            _ => None,
        };
        if let Some((handle, _)) = root {
            let global = &module.global_variables[handle];
            let externally_observable = matches!(
                global.space,
                naga::AddressSpace::Storage { .. } | naga::AddressSpace::Handle
            );
            if externally_observable {
                if let Some(name) = &global.name {
                    outputs.insert(name.clone());
                }
            }
        }
    });
}

fn walk_statements(block: &naga::Block, visit: &mut impl FnMut(&naga::Statement)) {
    for statement in block {
        visit(statement);
        match statement {
            naga::Statement::Block(block) => walk_statements(block, visit),
            naga::Statement::If { accept, reject, .. } => {
                walk_statements(accept, visit);
                walk_statements(reject, visit);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    walk_statements(&case.body, visit);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                walk_statements(body, visit);
                walk_statements(continuing, visit);
            }
            _ => {}
        }
    }
}

fn add_output_leaves(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
    is_entry: bool,
    outputs: &mut BTreeSet<String>,
) {
    match &module.types[ty].inner {
        naga::TypeInner::Struct { members, .. } => {
            for member in members {
                if let Some(name) = &member.name {
                    outputs.insert(name.clone());
                }
            }
        }
        _ if is_entry => {
            outputs.insert(binding_name(binding));
        }
        _ => {
            outputs.insert("return".into());
        }
    }
}

fn binding_name(binding: Option<&naga::Binding>) -> String {
    match binding {
        Some(naga::Binding::Location { location, .. }) => format!("location{location}"),
        Some(naga::Binding::BuiltIn(builtin)) => format!("builtin:{builtin:?}"),
        None => "return".into(),
    }
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
    NormGreaterEqual {
        value: String,
        minimum: f32,
    },
    DistanceGreaterEqual {
        left: String,
        right: String,
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
    LengthAtLeastProductCastU32 {
        buffer: String,
        width: String,
        height: String,
    },
    LengthAtLeastProductMaxOne {
        buffer: String,
        constant: u32,
        factors: Vec<String>,
    },
    AdditiveTextureBudget {
        output: String,
        diffuse: String,
        indirect: String,
        specular: String,
        reflection: String,
        maximum: f32,
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
    let mut unique_reasons = BTreeSet::new();
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
            let module = path
                .strip_prefix("src/shaders/")
                .context("ledger path is outside src/shaders")?;
            anyhow::ensure!(
                reason.starts_with(&format!("{module}: ")),
                "ledger reason for {path} is not module-specific"
            );
            anyhow::ensure!(
                unique_reasons.insert(reason.clone()),
                "duplicate ledger reason {reason}"
            );
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
                            .map(|value| parse_dimension(value))
                            .collect::<anyhow::Result<_>>()?,
                    }
                }
                ["buffer", name, min, max, length] => InputContract::Buffer {
                    range: range(name, min, max)?,
                    length: if *length == "dynamic" {
                        BufferLength::Dynamic
                    } else if length.parse::<u32>().is_err() {
                        BufferLength::Symbol(parse_symbol(length, "buffer length")?)
                    } else {
                        BufferLength::Fixed(positive_u32(length, "buffer length")?)
                    },
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
                ["norm_ge", value, minimum] => Ok(InvariantContract::NormGreaterEqual {
                    value: nonempty(value)?,
                    minimum: finite(minimum)?,
                }),
                ["distance_ge", left, right, minimum] => {
                    Ok(InvariantContract::DistanceGreaterEqual {
                        left: nonempty(left)?,
                        right: nonempty(right)?,
                        minimum: finite(minimum)?,
                    })
                }
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
                ["length_product_u32_cast", buffer, width, height] => {
                    Ok(InvariantContract::LengthAtLeastProductCastU32 {
                        buffer: nonempty(buffer)?,
                        width: nonempty(width)?,
                        height: nonempty(height)?,
                    })
                }
                ["length_product_max1", buffer, constant, factors @ ..] if !factors.is_empty() => {
                    Ok(InvariantContract::LengthAtLeastProductMaxOne {
                        buffer: nonempty(buffer)?,
                        constant: positive_u32(constant, "length product constant")?,
                        factors: factors
                            .iter()
                            .map(|factor| nonempty(factor))
                            .collect::<anyhow::Result<Vec<_>>>()?,
                    })
                }
                [
                    "additive_texture_budget",
                    output,
                    diffuse,
                    indirect,
                    specular,
                    reflection,
                    maximum,
                ] => Ok(InvariantContract::AdditiveTextureBudget {
                    output: nonempty(output)?,
                    diffuse: nonempty(diffuse)?,
                    indirect: nonempty(indirect)?,
                    specular: nonempty(specular)?,
                    reflection: nonempty(reflection)?,
                    maximum: finite(maximum)?,
                }),
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

fn parse_dimension(value: &str) -> anyhow::Result<DimensionContract> {
    if let Some(minimum) = value.strip_prefix("min") {
        return Ok(DimensionContract::Minimum(positive_u32(
            minimum,
            "texture minimum dimension",
        )?));
    }
    if value.parse::<u32>().is_ok() {
        bail!("texture dimensions are symbolic or explicit minima (for example min1)");
    }
    Ok(DimensionContract::Symbol(parse_symbol(
        value,
        "texture dimension",
    )?))
}

fn parse_symbol(value: &str, kind: &str) -> anyhow::Result<String> {
    if value.is_empty()
        || !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.'))
    {
        bail!("invalid symbolic {kind} {value:?}");
    }
    Ok(value.into())
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
inputs = ["value:arg:-1:1", "uniform:params:0:8", "texture:image:0:1:2:min1:min1", "buffer:data:-2:2:4", "sampler:samp"]
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
        assert!(parse_contract(&VALID.replace(
            "texture:image:0:1:2:min1:min1",
            "texture:image:0:1:2:min0:min1"
        ))
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
            "inputs = [\"value:arg:-1:1\", \"uniform:params:0:8\", \"texture:image:0:1:2:min1:min1\", \"buffer:data:-2:2:4\", \"sampler:samp\"]\n",
            ""
        ))
        .is_err());
        assert!(parse_contract(&VALID.replace("outputs = [\"color:0:1\"]\n", "")).is_err());
    }

    const LEDGER: &str = r#"
[[unproven]]
path = "src/shaders/a.wgsl"
reason = "a.wgsl: texture gather precision is not modeled"
owner = "render-quality"
expiry = "2027-01-17"
"#;

    #[test]
    fn ledger_is_structural_and_fail_closed() {
        let rows = parse_ledger(LEDGER).unwrap();
        assert_eq!(rows[0].path, "src/shaders/a.wgsl");
        assert!(parse_ledger(&LEDGER.replace("reason =", "unknown =")).is_err());
        assert!(parse_ledger(&LEDGER.replace("owner = \"render-quality\"\n", "")).is_err());
        assert!(parse_ledger(&LEDGER.replace(
            "a.wgsl: texture gather precision is not modeled",
            "too vague"
        ))
        .is_err());
        assert!(parse_ledger(&format!("{LEDGER}\n{LEDGER}")).is_err());
        assert!(parse_ledger(&LEDGER.replace("2027-01-17", "2020-01-01")).is_err());
        assert!(parse_ledger(&LEDGER.replace(
            "a.wgsl: texture gather precision is not modeled",
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

    #[test]
    fn real_sidecar_rejects_omitted_renamed_and_bogus_members() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = std::fs::read_to_string(root.join("src/shaders/line_aa.wgsl")).unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let text = std::fs::read_to_string(root.join("shaders/contracts/line_aa.toml")).unwrap();
        let contract = parse_contract(&text).unwrap();
        validate_contract_semantics(&module, "fs_main", &contract.entries[0]).unwrap();

        for mutation in [
            text.replace(", \"uniform:uniforms.cap_style:0:2\"", ""),
            text.replace("uniform:uniforms.cap_style", "uniform:uniforms.cap_stile"),
            text.replace(
                "inputs = [",
                "inputs = [\"uniform:not_a_shader_symbol:0:1\", ",
            ),
        ] {
            let mutated = parse_contract(&mutation);
            assert!(
                mutated.is_err()
                    || validate_contract_semantics(
                        &module,
                        "fs_main",
                        &mutated.unwrap().entries[0]
                    )
                    .is_err()
            );
        }
    }

    #[test]
    fn real_sidecar_rejects_omitted_and_renamed_resources() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = std::fs::read_to_string(root.join("src/shaders/overlays.wgsl")).unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let text = std::fs::read_to_string(root.join("shaders/contracts/overlays.toml")).unwrap();
        let contract = parse_contract(&text).unwrap();
        validate_contract_semantics(&module, "fs_overlay", &contract.entries[0]).unwrap();

        for mutation in [
            text.replace(", \"texture:overlay_tex:0:1:2:min1:min1\"", ""),
            text.replace("texture:overlay_tex:", "texture:overlay_texture:"),
        ] {
            let mutated = parse_contract(&mutation).unwrap();
            assert!(
                validate_contract_semantics(&module, "fs_overlay", &mutated.entries[0]).is_err()
            );
        }
    }

    #[test]
    fn semantic_references_reject_bogus_suffixes() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = std::fs::read_to_string(root.join("src/shaders/water_surface.wgsl")).unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let text = std::fs::read_to_string(root.join("shaders/contracts/water_surface.toml"))
            .unwrap()
            .replace("norm_ge:in.normal:", "norm_ge:in.normal.zebra:");
        let contract = parse_contract(&text).unwrap();
        assert!(validate_contract_semantics(&module, "fs_main", &contract.entries[0]).is_err());
    }

    #[test]
    fn every_invariant_operator_rejects_wrong_naga_operand_types() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let water_source =
            std::fs::read_to_string(root.join("src/shaders/water_surface.wgsl")).unwrap();
        let water_module = naga::front::wgsl::parse_str(&water_source).unwrap();
        let water =
            std::fs::read_to_string(root.join("shaders/contracts/water_surface.toml")).unwrap();
        for invalid in [
            "norm_ge:in.view_distance:0.000001",
            "ge:in.normal:0",
            "abs_ge:in.normal:0",
            "distance_ge:in.normal:in.view_distance:0.000001",
            "sum_abs_ge:in.normal:in.view_distance:0.000001",
            "difference_ge:in.normal:in.view_distance:0.000001",
        ] {
            let mutation =
                water.replace("invariants = [", &format!("invariants = [\"{invalid}\", "));
            let contract = parse_contract(&mutation).unwrap();
            assert!(
                validate_contract_semantics(&water_module, "fs_main", &contract.entries[0])
                    .is_err(),
                "accepted invalid invariant {invalid}"
            );
        }

        let hybrid_source = super::super::preprocess_hybrid_shader();
        let hybrid_module = naga::front::wgsl::parse_str(&hybrid_source).unwrap();
        let hybrid =
            std::fs::read_to_string(root.join("shaders/contracts/hybrid_terrain_traversal.toml"))
                .unwrap();
        for (from, to) in [
            (
                "dimensions_cover:terrain_height_tex:terrain.dims.x:terrain.dims.y",
                "dimensions_cover:terrain_height_tex:terrain.dims:terrain.dims.y",
            ),
            (
                "dimensions_cover:terrain_height_tex:terrain.dims.x:terrain.dims.y",
                "same_dimensions:terrain_height_tex:uniforms.width",
            ),
            (
                "count_within:hybrid_uniforms.mesh_index_count:mesh_indices",
                "count_within:lighting.light_dir:mesh_indices",
            ),
            (
                "length_product:accum_hdr:uniforms.width:uniforms.height",
                "length_product:accum_hdr:uniforms.cam_forward:uniforms.height",
            ),
            (
                "indices_within:mesh_indices:mesh_vertices",
                "indices_within:mesh_vertices:mesh_indices",
            ),
        ] {
            let contract = parse_contract(&hybrid.replace(from, to)).unwrap();
            assert!(
                validate_contract_semantics(&hybrid_module, "main_terrain", &contract.entries[0])
                    .is_err(),
                "accepted invalid invariant {to}"
            );
        }

        let terrain_source = super::super::preprocess_terrain_shader();
        let terrain_module = naga::front::wgsl::parse_str(&terrain_source).unwrap();
        let terrain =
            std::fs::read_to_string(root.join("shaders/contracts/terrain_pbr_pom.toml")).unwrap();
        let invalid_max_one = terrain.replace(
            "terrain_vt_uniforms.config2.y:terrain_vt_uniforms.config2.x",
            "input.world_normal:terrain_vt_uniforms.config2.x",
        );
        let contract = parse_contract(&invalid_max_one).unwrap();
        assert!(
            validate_contract_semantics(&terrain_module, "fs_main", &contract.entries[0]).is_err()
        );
        let invalid_cast_product = terrain.replace(
            "probe_grid.grid_params.z:probe_grid.grid_params.w",
            "input.world_normal:probe_grid.grid_params.w",
        );
        let contract = parse_contract(&invalid_cast_product).unwrap();
        assert!(
            validate_contract_semantics(&terrain_module, "fs_main", &contract.entries[0]).is_err()
        );

        let gi_source =
            std::fs::read_to_string(root.join("src/shaders/gi/composite.wgsl")).unwrap();
        let gi_module = naga::front::wgsl::parse_str(&gi_source).unwrap();
        let gi = std::fs::read_to_string(root.join("shaders/contracts/gi_composite.toml")).unwrap();
        let invalid_budget = gi.replace(
            "spec_base_texture:ssr_texture:65504",
            "spec_base_texture:gi_params.ssr_weight:65504",
        );
        let contract = parse_contract(&invalid_budget).unwrap();
        assert!(
            validate_contract_semantics(&gi_module, "cs_gi_composite", &contract.entries[0])
                .is_err()
        );
    }

    #[test]
    fn entry_contract_contains_only_reachable_argument_members() {
        let source = r#"
struct Input { used: f32, unused: f32 }
fn consume(input: Input) -> f32 { return input.used; }
@fragment fn main(input: Input) -> @location(0) f32 {
    return consume(input);
}
"#;
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let contract_text = VALID
            .replace("demo", "main")
            .replace("outputs = [\"color:0:1\"]", "outputs = [\"location0:0:1\"]")
            .replace("invariants = [\"ge:params.count:1\"]", "invariants = []")
            .replace(
                "\"value:arg:-1:1\", \"uniform:params:0:8\", \"texture:image:0:1:2:min1:min1\", \"buffer:data:-2:2:4\", \"sampler:samp\"",
                "\"value:input.used:-1:1\"",
            );
        let contract = parse_contract(&contract_text).unwrap();
        validate_contract_semantics(&module, "main", &contract.entries[0]).unwrap();

        let extra = parse_contract(&contract_text.replace(
            "\"value:input.used:-1:1\"",
            "\"value:input.used:-1:1\", \"value:input.unused:-1:1\"",
        ))
        .unwrap();
        assert!(validate_contract_semantics(&module, "main", &extra.entries[0]).is_err());

        let omitted = parse_contract(&contract_text.replace("input.used", "input.unused")).unwrap();
        assert!(validate_contract_semantics(&module, "main", &omitted.entries[0]).is_err());
    }

    #[test]
    fn indices_relation_does_not_bound_the_indices_buffer_length() {
        let source = r#"
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@fragment fn main(@location(0) index: u32) -> @location(0) f32 {
    return vertices[indices[index]];
}
"#;
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let contract_text = VALID
            .replace("demo", "main")
            .replace(
                "\"value:arg:-1:1\", \"uniform:params:0:8\", \"texture:image:0:1:2:min1:min1\", \"buffer:data:-2:2:4\", \"sampler:samp\"",
                "\"value:index:0:0\", \"buffer:indices:0:4294967000:dynamic\", \"buffer:vertices:-1:1:dynamic\"",
            )
            .replace(
                "invariants = [\"ge:params.count:1\"]",
                "invariants = [\"indices_within:indices:vertices\"]",
            );
        let contract = parse_contract(&contract_text).unwrap();
        assert!(validate_contract_semantics(&module, "main", &contract.entries[0]).is_err());

        let float_indices = source
            .replace("array<u32>", "array<f32>")
            .replace("vertices[indices[index]]", "vertices[u32(indices[index])]");
        let float_module = naga::front::wgsl::parse_str(&float_indices).unwrap();
        let typed_contract = parse_contract(&contract_text.replace(
            "indices_within:indices:vertices",
            "count_within:index:indices\", \"indices_within:indices:vertices",
        ))
        .unwrap();
        assert!(
            validate_contract_semantics(&float_module, "main", &typed_contract.entries[0]).is_err()
        );
    }

    #[test]
    fn hybrid_contract_bounds_index_and_target_arrays_directionally() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let contract = parse_contract(
            &std::fs::read_to_string(root.join("shaders/contracts/hybrid_terrain_traversal.toml"))
                .unwrap(),
        )
        .unwrap();
        let invariants = &contract.entries[0].invariants;
        assert!(invariants.iter().any(|invariant| matches!(
            invariant,
            InvariantContract::CountWithin { count, buffer }
                if count == "hybrid_uniforms.mesh_index_count" && buffer == "mesh_indices"
        )));
        assert!(invariants.iter().any(|invariant| matches!(
            invariant,
            InvariantContract::IndicesWithin { indices, target }
                if indices == "mesh_indices" && target == "mesh_vertices"
        )));
    }

    #[test]
    fn gi_output_range_covers_all_additive_sources() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let contract = parse_contract(
            &std::fs::read_to_string(root.join("shaders/contracts/gi_composite.toml")).unwrap(),
        )
        .unwrap();
        let entry = &contract.entries[0];
        assert_eq!(entry.outputs[0].max, 65_504.0);
        let budget = entry
            .invariants
            .iter()
            .find_map(|invariant| match invariant {
                InvariantContract::AdditiveTextureBudget { maximum, .. } => Some(*maximum),
                _ => None,
            });
        assert_eq!(budget, Some(65_504.0));

        let within_budget = |diffuse: f32, ssgi: f32, specular: f32, reflection: f32| {
            diffuse + ssgi + specular.max(reflection) <= budget.unwrap()
        };
        assert!(within_budget(65_504.0, 0.0, 0.0, 0.0));
        assert!(within_budget(0.0, 65_504.0, 0.0, 0.0));
        assert!(within_budget(32_752.0, 16_376.0, 16_376.0, 10_000.0));
        assert!(within_budget(0.0, 0.0, 65_504.0, 65_504.0));
        assert!(!within_budget(65_504.0, 65_504.0, 65_504.0, 65_504.0));
    }
}
