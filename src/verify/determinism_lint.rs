//! TERRA-DETERMINATA: naga-IR lint for the pinned-arithmetic discipline.
//!
//! Runs over the post-assembly WGSL of every deterministic-path module (any
//! assembly that includes `src/shaders/includes/determinism.wgsl`) and rejects
//! the constructs a backend compiler is free to lower differently:
//!
//! - an unbarriered float multiply feeding an add/subtract (fma contraction),
//!   an unbarriered add/subtract feeding an add/subtract (reassociation), and
//!   an unbarriered multiply feeding a multiply (product-tree reassociation).
//!   A `det_barrier*` call between producer and consumer breaks the edge.
//!   Edges are followed the way a backend sees them after inlining: through
//!   call results (a callee's tail op, or a parameter it returns unchanged),
//!   into callees (an argument the callee adds to or multiplies again), and
//!   through `var` locals by constant field/element path.
//! - raw float `/`, and the per-API intrinsics outside `det_*`: `sqrt`,
//!   `inverseSqrt`, `dot`, `mix`, `fma`, `pow`, `exp`, `exp2`, `log`, `log2`,
//!   all trig/hyperbolic forms, `length`, `distance`, `normalize`, `cross`,
//!   `reflect`, `refract`, `faceForward`, `smoothstep`, `outer`, matrix
//!   `inverse`/`transpose`/`determinant`, and float matrix products.
//! - any `textureSample*`/`textureGather*` is reported as a `hardware_sample`
//!   advisory. Automatic source rewriting cannot safely recover each sampling
//!   contract; deterministic mode pins sampler filtering to nearest, while the
//!   runtime canaries and cross-vendor hash matrix detect residual hardware
//!   texel-selection divergence.
//! - non-`Coarse` derivative expressions (`dpdx`/`dpdy`/`fwidth` granularity
//!   is a per-implementation choice; callers must spell `dpdxCoarse` etc.).
//! - loads or stores of the private `det_zu` global outside the `det_*`
//!   helper family.
//! - an entry point whose call graph reaches a `det_zu` reader (the
//!   `det_barrier*` identities) without a dominating `det_seed` call: the
//!   seed must appear in the entry point's own top-level statement stream
//!   before the first call that transitively reaches a barrier.
//!
//! `det_*` functions themselves are exempt from the arithmetic/intrinsic
//! rules: they ARE the pin layer, and their bodies are proven by the
//! symbolic verifier (`src/verify/ir`). They are still checked for stray
//! `det_zu` access.

use std::collections::{HashMap, HashSet};

use naga::{BinaryOperator, Expression, Handle, MathFunction, Statement, TypeInner};

/// One lint violation in an assembled module.
#[derive(Debug)]
pub struct Violation {
    pub module: String,
    pub function: String,
    pub line: usize,
    pub kind: &'static str,
    pub detail: String,
    /// IR coordinates of the offending site for the source rewriter.
    pub site: Option<Site>,
}

/// Which arena the violating function lives in.
#[derive(Clone, Copy, Debug)]
pub enum FuncSite {
    Regular(Handle<naga::Function>),
    Entry(usize),
}

/// IR coordinates of a violation site. `expr` is the violating expression,
/// `operand` the consumer-side operand handle (use site), and `related` the
/// producer expression for edge violations.
#[derive(Clone, Copy, Debug)]
pub struct Site {
    pub func: FuncSite,
    pub expr: Option<Handle<Expression>>,
    pub operand: Option<Handle<Expression>>,
    pub related: Option<Handle<Expression>>,
}

impl Site {
    fn at(func: FuncSite, expr: Handle<Expression>) -> Self {
        Site {
            func,
            expr: Some(expr),
            operand: None,
            related: None,
        }
    }

    fn edge(func: FuncSite, operand: Handle<Expression>, related: Handle<Expression>) -> Self {
        Site {
            func,
            expr: None,
            operand: Some(operand),
            related: Some(related),
        }
    }

    fn func(func: FuncSite) -> Self {
        Site {
            func,
            expr: None,
            operand: None,
            related: None,
        }
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}: {}",
            self.module, self.function, self.line, self.kind, self.detail
        )
    }
}

/// Math intrinsics that must not appear outside `det_*`: each is a
/// per-backend lowering choice (precision class, polynomial, or reduction
/// order the driver owns).
fn forbidden_math(fun: MathFunction) -> Option<&'static str> {
    Some(match fun {
        MathFunction::Dot => "dot",
        MathFunction::Mix => "mix",
        MathFunction::Fma => "fma",
        MathFunction::Pow => "pow",
        MathFunction::Exp => "exp",
        MathFunction::Exp2 => "exp2",
        MathFunction::Log => "log",
        MathFunction::Log2 => "log2",
        MathFunction::Sqrt => "sqrt",
        MathFunction::InverseSqrt => "inverseSqrt",
        MathFunction::Length => "length",
        MathFunction::Distance => "distance",
        MathFunction::Normalize => "normalize",
        MathFunction::Cross => "cross",
        MathFunction::Reflect => "reflect",
        MathFunction::Refract => "refract",
        MathFunction::FaceForward => "faceForward",
        MathFunction::SmoothStep => "smoothstep",
        MathFunction::Outer => "outer",
        MathFunction::Sin => "sin",
        MathFunction::Cos => "cos",
        MathFunction::Tan => "tan",
        MathFunction::Asin => "asin",
        MathFunction::Acos => "acos",
        MathFunction::Atan => "atan",
        MathFunction::Atan2 => "atan2",
        MathFunction::Sinh => "sinh",
        MathFunction::Cosh => "cosh",
        MathFunction::Tanh => "tanh",
        MathFunction::Asinh => "asinh",
        MathFunction::Acosh => "acosh",
        MathFunction::Atanh => "atanh",
        MathFunction::Inverse => "matrix inverse",
        MathFunction::Transpose => "transpose",
        MathFunction::Determinant => "determinant",
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProducerKind {
    Mul,
    AddSub,
    Div,
}

/// The fuseable producer -> consumer edges: mul->add/sub (fma contraction),
/// add/sub->add/sub (reassociation), mul->mul (product-tree reassociation).
fn is_bad_edge(producer: ProducerKind, consumer: ProducerKind) -> bool {
    matches!(
        (producer, consumer),
        (ProducerKind::Mul, ProducerKind::AddSub)
            | (ProducerKind::AddSub, ProducerKind::AddSub)
            | (ProducerKind::Mul, ProducerKind::Mul)
    )
}

/// Stores into a function's `var` locals: root local -> (constant access
/// path from the root, stored value). A dynamic `a[i]` index ends the path,
/// so that store conservatively covers every element below it.
type LocalStores = HashMap<usize, Vec<StoredValue>>;

/// One store into a local: constant access path, whether that path is exact
/// (no dynamic index was truncated away), the stored value, and whether the
/// store sits inside a loop (where it can feed a load that runs before it).
type StoredValue = (Vec<u32>, bool, Handle<Expression>, bool);

/// Root local, constant access path, and exactness of a pointer
/// (`s.x` -> (s, [0], true); `a[i].x` -> (a, [], false)).
fn pointer_path(
    function: &naga::Function,
    mut pointer: Handle<Expression>,
) -> Option<(usize, Vec<u32>, bool)> {
    let mut reversed = Vec::new();
    let mut exact = true;
    loop {
        match function.expressions[pointer] {
            Expression::LocalVariable(local) => {
                reversed.reverse();
                return Some((local.index(), reversed, exact));
            }
            Expression::AccessIndex { base, index } => {
                reversed.push(index);
                pointer = base;
            }
            Expression::Access { base, .. } => {
                // Dynamic index: everything collected so far is below an
                // unknown element; keep only the part above it.
                reversed.clear();
                exact = false;
                pointer = base;
            }
            _ => return None,
        }
    }
}

/// Stored values a load of `path` under `root` may observe, each with the
/// projection still to apply inside the stored value. Paths overlap when one
/// is a prefix of the other: a whole-struct store feeds a field load (with
/// the remaining projection), and a field store feeds a whole-struct load.
fn stores_for_load<'a>(
    stores: &'a LocalStores,
    root: usize,
    path: &'a [u32],
    load_exact: bool,
) -> impl Iterator<Item = (Handle<Expression>, Vec<u32>)> + 'a {
    stores
        .get(&root)
        .into_iter()
        .flatten()
        .filter_map(move |(stored, store_exact, value, _)| {
            if path.starts_with(stored) {
                // Only project into the stored value when both paths name
                // exact elements; otherwise take the whole value.
                let projection = if load_exact && *store_exact {
                    path[stored.len()..].to_vec()
                } else {
                    Vec::new()
                };
                Some((*value, projection))
            } else if stored.starts_with(path) {
                Some((*value, Vec::new()))
            } else {
                None
            }
        })
}

/// Does a `Compose` of this type take exactly one component per indexable
/// element, so `AccessIndex(k)` selects `components[k]`?
fn compose_is_elementwise(inner: &TypeInner, components: usize) -> bool {
    match inner {
        TypeInner::Vector { size, .. } => *size as usize == components,
        TypeInner::Matrix { columns, .. } => *columns as usize == components,
        TypeInner::Struct { members, .. } => members.len() == components,
        TypeInner::Array {
            size: naga::ArraySize::Constant(size),
            ..
        } => size.get() as usize == components,
        _ => false,
    }
}

/// Every call in a body with its argument handles.
fn call_sites(body: &naga::Block) -> Vec<(Handle<naga::Function>, Vec<Handle<Expression>>)> {
    let mut out = Vec::new();
    let mut stack: Vec<&Statement> = body.iter().collect();
    while let Some(statement) = stack.pop() {
        match statement {
            Statement::Call {
                function,
                arguments,
                ..
            } => out.push((*function, arguments.clone())),
            Statement::Block(block) => stack.extend(block.iter()),
            Statement::If { accept, reject, .. } => {
                stack.extend(accept.iter());
                stack.extend(reject.iter());
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    stack.extend(case.body.iter());
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                stack.extend(body.iter());
                stack.extend(continuing.iter());
            }
            _ => {}
        }
    }
    out
}

/// The call statement (callee and arguments) that produced a CallResult.
fn call_statement(
    body: &naga::Block,
    result: Handle<Expression>,
) -> Option<(Handle<naga::Function>, Vec<Handle<Expression>>)> {
    let callee = call_result_producer(body, result)?;
    let mut stack: Vec<&Statement> = body.iter().collect();
    while let Some(statement) = stack.pop() {
        match statement {
            Statement::Call {
                function,
                arguments,
                result: Some(r),
            } if *r == result && *function == callee => {
                return Some((*function, arguments.clone()))
            }
            Statement::Block(block) => stack.extend(block.iter()),
            Statement::If { accept, reject, .. } => {
                stack.extend(accept.iter());
                stack.extend(reject.iter());
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    stack.extend(case.body.iter());
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                stack.extend(body.iter());
                stack.extend(continuing.iter());
            }
            _ => {}
        }
    }
    None
}

/// Parameters that `handle` carries unchanged, traced backward through the
/// same transparent wrappers the edge resolver sees through.
fn params_reaching(
    function: &naga::Function,
    local_producers: &LocalStores,
    handle: Handle<Expression>,
) -> HashSet<u32> {
    let mut out = HashSet::new();
    let mut seen = HashSet::new();
    let mut stack = vec![handle];
    while let Some(current) = stack.pop() {
        if !seen.insert(current) {
            continue;
        }
        match &function.expressions[current] {
            Expression::FunctionArgument(index) => {
                out.insert(*index);
            }
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                stack.push(*base)
            }
            Expression::Swizzle { vector, .. } => stack.push(*vector),
            Expression::Splat { value, .. } => stack.push(*value),
            Expression::Compose { components, .. } => stack.extend(components.iter().copied()),
            Expression::Unary { expr, .. } => stack.push(*expr),
            Expression::As { expr, .. } => stack.push(*expr),
            Expression::Load { pointer } => {
                if let Some((local, path, exact)) = pointer_path(function, *pointer) {
                    stack.extend(
                        stores_for_load(local_producers, local, &path, exact)
                            .map(|(value, _)| value),
                    );
                }
            }
            _ => {}
        }
    }
    out
}

fn function_local_producers(function: &naga::Function) -> LocalStores {
    let mut local_producers = HashMap::new();
    for statement in function.body.iter() {
        collect_stores(function, statement, false, &mut local_producers);
    }
    local_producers
}

/// Parameters a callee returns unchanged (through transparent wrappers).
fn returned_params(module: &naga::Module, callee: Handle<naga::Function>) -> HashSet<u32> {
    let function = &module.functions[callee];
    let local_producers = function_local_producers(function);
    let mut returns = Vec::new();
    collect_return_values(&function.body, &mut returns);
    returns
        .into_iter()
        .flat_map(|value| params_reaching(function, &local_producers, value))
        .collect()
}

/// Float arithmetic kinds each parameter of `callee` feeds directly once the
/// call is inlined, following parameters into nested calls. `det_barrier*`
/// consume their parameter through bitcasts and report none; valid WGSL has
/// an acyclic call graph, so the recursion terminates.
fn param_consumer_kinds(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    callee: Handle<naga::Function>,
    memo: &mut HashMap<usize, Vec<Vec<ProducerKind>>>,
) -> Vec<Vec<ProducerKind>> {
    if let Some(found) = memo.get(&callee.index()) {
        return found.clone();
    }
    let function = &module.functions[callee];
    let function_info = &info[callee];
    let local_producers = function_local_producers(function);
    let mut kinds = vec![Vec::new(); function.arguments.len()];
    for (handle, expression) in function.expressions.iter() {
        let Expression::Binary { op, left, right } = expression else {
            continue;
        };
        let inner = function_info[handle].ty.inner_with(&module.types);
        let Some(kind) = producer_kind(inner, *op) else {
            continue;
        };
        for operand in [*left, *right] {
            for index in params_reaching(function, &local_producers, operand) {
                kinds[index as usize].push(kind);
            }
        }
    }
    for (nested_callee, arguments) in call_sites(&function.body) {
        let nested = param_consumer_kinds(module, info, nested_callee, memo);
        for (position, argument) in arguments.iter().enumerate() {
            for index in params_reaching(function, &local_producers, *argument) {
                if let Some(nested_kinds) = nested.get(position) {
                    kinds[index as usize].extend(nested_kinds.iter().copied());
                }
            }
        }
    }
    memo.insert(callee.index(), kinds.clone());
    kinds
}

/// Kind of arithmetic a function's return value carries when inlined: the
/// tail operation its result comes from. `det_fma`'s tail is an unbarriered
/// add, so `det_fma(...) + x` is really `(add) + x` after inlining — the
/// edge rule must see through CallResult to this. `None` means the result
/// is barriered, selected, or otherwise not a raw arithmetic tail.
fn call_tail_kind(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    callee: Handle<naga::Function>,
) -> Option<ProducerKind> {
    let function = &module.functions[callee];
    let mut found: Option<ProducerKind> = None;
    let mut stack: Vec<Handle<Expression>> = Vec::new();
    collect_return_values(&function.body, &mut stack);
    while let Some(handle) = stack.pop() {
        match &function.expressions[handle] {
            Expression::Binary { op, .. } => {
                let inner = info[callee][handle].ty.inner_with(&module.types);
                if let Some(kind) = producer_kind(inner, *op) {
                    found = Some(merge_tail_kinds(found, kind));
                }
            }
            Expression::Compose { components, .. } => stack.extend(components.iter().copied()),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                stack.push(*base)
            }
            Expression::Swizzle { vector, .. } => stack.push(*vector),
            Expression::Unary { expr, .. } => stack.push(*expr),
            Expression::CallResult(_) => {
                // The return tail is another call's result: recurse into that
                // callee (det_div -> det_barrier -> None, i.e. barriered).
                // Valid WGSL has an acyclic call graph, so validation above
                // guarantees this traversal terminates without a depth cap.
                if let Some(producer) = call_result_producer(&function.body, handle) {
                    if let Some(kind) = call_tail_kind(module, info, producer) {
                        found = Some(merge_tail_kinds(found, kind));
                    }
                }
            }
            _ => {}
        }
    }
    found
}

/// A function may return different tail operations on different paths;
/// merge conservatively toward the kind that can form the most hazardous
/// edges (Mul > AddSub > Div).
fn merge_tail_kinds(found: Option<ProducerKind>, kind: ProducerKind) -> ProducerKind {
    match (found, kind) {
        (Some(ProducerKind::Mul), _) | (_, ProducerKind::Mul) => ProducerKind::Mul,
        (Some(ProducerKind::AddSub), _) | (_, ProducerKind::AddSub) => ProducerKind::AddSub,
        (Some(other), _) => other,
        (None, kind) => kind,
    }
}

fn collect_return_values(body: &naga::Block, out: &mut Vec<Handle<Expression>>) {
    for statement in body.iter() {
        match statement {
            Statement::Return { value: Some(value) } => out.push(*value),
            Statement::Block(block) => collect_return_values(block, out),
            Statement::If { accept, reject, .. } => {
                collect_return_values(accept, out);
                collect_return_values(reject, out);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_return_values(&case.body, out);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_return_values(body, out);
                collect_return_values(continuing, out);
            }
            _ => {}
        }
    }
}

/// Which callee produced this CallResult? CallResult expressions pair with
/// `Statement::Call` results by handle.
fn call_result_producer(
    body: &naga::Block,
    result: Handle<Expression>,
) -> Option<Handle<naga::Function>> {
    for statement in body.iter() {
        match statement {
            Statement::Call {
                function,
                result: Some(r),
                ..
            } if *r == result => return Some(*function),
            Statement::Block(block) => {
                if let Some(found) = call_result_producer(block, result) {
                    return Some(found);
                }
            }
            Statement::If { accept, reject, .. } => {
                if let Some(found) = call_result_producer(accept, result) {
                    return Some(found);
                }
                if let Some(found) = call_result_producer(reject, result) {
                    return Some(found);
                }
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    if let Some(found) = call_result_producer(&case.body, result) {
                        return Some(found);
                    }
                }
            }
            Statement::Loop {
                body: loop_body,
                continuing,
                ..
            } => {
                if let Some(found) = call_result_producer(loop_body, result) {
                    return Some(found);
                }
                if let Some(found) = call_result_producer(continuing, result) {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse, validate, and lint one assembled deterministic-path module,
/// returning the naga module and validation info alongside the violations so
/// the source rewriter can resolve each violation's `Site` back to IR
/// coordinates. Module and info are `None` when parsing or validation fails.
#[allow(clippy::type_complexity)]
pub fn analyze_source(
    module_name: &str,
    source: &str,
) -> (
    Option<naga::Module>,
    Option<naga::valid::ModuleInfo>,
    Vec<Violation>,
) {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(error) => {
            return (
                None,
                None,
                vec![Violation {
                    module: module_name.to_string(),
                    function: String::new(),
                    line: 1,
                    kind: "parse",
                    detail: format!("naga rejected the assembled WGSL: {error}"),
                    site: None,
                }],
            );
        }
    };
    // Type information per expression: needed to distinguish float
    // arithmetic (the contract's domain) from exact integer arithmetic.
    let info = match naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    {
        Ok(info) => info,
        Err(error) => {
            return (
                None,
                None,
                vec![Violation {
                    module: module_name.to_string(),
                    function: String::new(),
                    line: 1,
                    kind: "validate",
                    detail: format!(
                        "naga validation failed on the assembled WGSL: {}",
                        error.emit_to_string(source)
                    ),
                    site: None,
                }],
            );
        }
    };

    // ---- call graph: which functions transitively read det_zu -------------
    // Calls can only target regular functions, so the closure lives on the
    // functions arena; entry points are checked separately for "reaches a
    // barrier reader".
    let det_zu_readers: HashSet<usize> = module
        .functions
        .iter()
        .filter(|(_, function)| function_reads_det_zu(&module, function))
        .map(|(handle, _)| handle.index())
        .collect();
    // callee index -> does it transitively reach a det_zu reader
    let mut reaches_barrier: HashSet<usize> = det_zu_readers.clone();
    loop {
        let mut grew = false;
        for (handle, function) in module.functions.iter() {
            if reaches_barrier.contains(&handle.index()) {
                continue;
            }
            if calls_in_body(&function.body).any(|callee| reaches_barrier.contains(&callee.index()))
            {
                reaches_barrier.insert(handle.index());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    let mut violations = Vec::new();
    for (handle, function) in module.functions.iter() {
        lint_function(
            module_name,
            &module,
            &info,
            function,
            &info[handle],
            function.name.as_deref().unwrap_or("?"),
            FuncSite::Regular(handle),
            &reaches_barrier,
            source,
            &mut violations,
        );
    }
    for (index, entry) in module.entry_points.iter().enumerate() {
        lint_function(
            module_name,
            &module,
            &info,
            &entry.function,
            info.get_entry_point(index),
            entry.name.as_str(),
            FuncSite::Entry(index),
            &reaches_barrier,
            source,
            &mut violations,
        );
    }
    (Some(module), Some(info), violations)
}

fn calls_in_body(body: &naga::Block) -> impl Iterator<Item = Handle<naga::Function>> + '_ {
    body.iter().flat_map(|statement| CallsInStatement {
        stack: vec![statement],
    })
}

fn calls_in_statement(statement: &Statement) -> impl Iterator<Item = Handle<naga::Function>> + '_ {
    CallsInStatement {
        stack: vec![statement],
    }
}

struct CallsInStatement<'a> {
    stack: Vec<&'a Statement>,
}

impl<'a> Iterator for CallsInStatement<'a> {
    type Item = Handle<naga::Function>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(statement) = self.stack.pop() {
            match statement {
                Statement::Call { function, .. } => return Some(*function),
                Statement::Block(block) => self.stack.extend(block.iter()),
                Statement::If { accept, reject, .. } => {
                    self.stack.extend(accept.iter());
                    self.stack.extend(reject.iter());
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        self.stack.extend(case.body.iter());
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    self.stack.extend(body.iter());
                    self.stack.extend(continuing.iter());
                }
                _ => {}
            }
        }
        None
    }
}

/// Does this function body contain a Load of the private `det_zu` global?
fn function_reads_det_zu(module: &naga::Module, function: &naga::Function) -> bool {
    function.expressions.iter().any(|(_, expression)| {
        let Expression::Load { pointer } = expression else {
            return false;
        };
        matches!(
            function.expressions[*pointer],
            Expression::GlobalVariable(global)
                if module.global_variables[global].name.as_deref() == Some("det_zu")
        )
    })
}

fn is_det_helper(name: &str) -> bool {
    name.starts_with("det_")
}

/// Scalar kind of a resolved naga type: is it float arithmetic?
fn is_float_arithmetic(inner: &TypeInner) -> bool {
    match inner {
        TypeInner::Scalar(scalar) => scalar.kind == naga::ScalarKind::Float,
        TypeInner::Vector { scalar, .. } => scalar.kind == naga::ScalarKind::Float,
        TypeInner::Matrix { .. } => true,
        _ => false,
    }
}

fn producer_kind(inner: &TypeInner, op: BinaryOperator) -> Option<ProducerKind> {
    if !is_float_arithmetic(inner) {
        return None;
    }
    Some(match op {
        BinaryOperator::Multiply => ProducerKind::Mul,
        BinaryOperator::Add | BinaryOperator::Subtract => ProducerKind::AddSub,
        BinaryOperator::Divide => ProducerKind::Div,
        _ => return None,
    })
}

fn lint_function(
    module_name: &str,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    function: &naga::Function,
    function_info: &naga::valid::FunctionInfo,
    name: &str,
    func_site: FuncSite,
    reaches_barrier: &HashSet<usize>,
    source: &str,
    violations: &mut Vec<Violation>,
) {
    let det = is_det_helper(name);
    let span_line = |handle: Handle<Expression>| {
        function
            .expressions
            .get_span(handle)
            .location(source)
            .line_number as usize
    };

    // ---- det_zu may be loaded only by det_barrier*/written only by det_seed
    for (handle, expression) in function.expressions.iter() {
        if let Expression::Load { pointer } = expression {
            if let Expression::GlobalVariable(global) = function.expressions[*pointer] {
                if module.global_variables[global].name.as_deref() == Some("det_zu")
                    && !name.starts_with("det_barrier")
                    && name != "det_seed"
                {
                    violations.push(Violation {
                        module: module_name.to_string(),
                        function: name.to_string(),
                        line: span_line(*pointer),
                        kind: "det_zu_access",
                        detail: "loads det_zu outside det_barrier*/det_seed".to_string(),
                        site: Some(Site::at(func_site, handle)),
                    });
                }
            }
        }
    }
    for statement in function.body.iter() {
        collect_det_zu_stores(
            module,
            function,
            statement,
            name,
            module_name,
            func_site,
            &span_line,
            violations,
        );
    }

    if det {
        // The pin layer is proven by the symbolic verifier; skip the
        // arithmetic/intrinsic rules here (its bodies are deliberately raw
        // barriered ops). Entry points are never det_* functions.
        return;
    }

    // var-locals: a Load's producer is whatever was stored into the local
    // (over-approximated across control flow; order-insensitive by design).
    let local_producers = function_local_producers(function);

    // Resolve an operand handle through transparent wrappers to the actual
    // producer: compose/swizzle/access/unary pass through, local loads
    // resolve to their stored expressions, and CallResults resolve to the
    // callee's tail kind (a call is inlined by the backend, so `det_fma(..)
    // + x` is an unbarriered add feeding an add). det_barrier* results are
    // opaque leaves — that is exactly what they exist to be.
    //
    // Each stack entry carries the constant component projection still to
    // apply (`v.x` of `vec3(a, b * c, d)` is `a`, not the product), so a
    // product in one lane of a composite is not blamed on another lane. A
    // projection that cannot be followed exactly is dropped, which only
    // widens the search.
    //
    // `consumer` is the expression consuming the operand. Outside a loop a
    // store can never feed a load in its own right-hand side, so the
    // consumer's own result stored back (`x.y -= b`) is not its producer.
    let self_stores: HashSet<Handle<Expression>> = local_producers
        .values()
        .flatten()
        .filter(|(_, _, _, in_loop)| !in_loop)
        .map(|(_, _, value, _)| *value)
        .collect();
    let resolve = |handle: Handle<Expression>,
                   consumer: Option<Handle<Expression>>|
     -> Option<(ProducerKind, Handle<Expression>)> {
        let mut seen = HashSet::new();
        let mut stack: Vec<(Handle<Expression>, Vec<u32>)> = vec![(handle, Vec::new())];
        while let Some((current, projection)) = stack.pop() {
            if !seen.insert((current, projection.clone())) {
                continue;
            }
            let expression = &function.expressions[current];
            match expression {
                Expression::Binary { op, .. } => {
                    let inner = function_info[current].ty.inner_with(&module.types);
                    if let Some(kind) = producer_kind(inner, *op) {
                        return Some((kind, current));
                    }
                }
                Expression::CallResult(_) => {
                    if let Some((callee, arguments)) = call_statement(&function.body, current) {
                        let callee_name = module.functions[callee].name.as_deref().unwrap_or("");
                        if !callee_name.starts_with("det_barrier") {
                            if let Some(kind) = call_tail_kind(module, info, callee) {
                                return Some((kind, current));
                            }
                            // A callee that returns a parameter unchanged
                            // is transparent after inlining: the caller's
                            // argument is the producer.
                            for index in returned_params(module, callee) {
                                if let Some(argument) = arguments.get(index as usize) {
                                    stack.push((*argument, Vec::new()));
                                }
                            }
                        }
                    }
                }
                Expression::AccessIndex { base, index } => {
                    let mut deeper = vec![*index];
                    deeper.extend(projection.iter().copied());
                    stack.push((*base, deeper));
                }
                Expression::Access { base, .. } => stack.push((*base, Vec::new())),
                Expression::Swizzle { vector, .. } => stack.push((*vector, Vec::new())),
                Expression::Splat { value, .. } => stack.push((*value, Vec::new())),
                Expression::Compose { components, .. } => {
                    let inner = function_info[current].ty.inner_with(&module.types);
                    match projection.split_first() {
                        Some((first, rest))
                            if compose_is_elementwise(inner, components.len())
                                && (*first as usize) < components.len() =>
                        {
                            stack.push((components[*first as usize], rest.to_vec()));
                        }
                        _ => stack.extend(components.iter().map(|c| (*c, Vec::new()))),
                    }
                }
                Expression::Unary { expr, .. } => stack.push((*expr, projection.clone())),
                Expression::As { expr, .. } => stack.push((*expr, projection.clone())),
                Expression::Load { pointer } => {
                    if let Some((local, mut path, exact)) = pointer_path(function, *pointer) {
                        if exact {
                            path.extend(projection.iter().copied());
                        }
                        stack.extend(
                            stores_for_load(&local_producers, local, &path, exact).filter(
                                |(value, _)| {
                                    Some(*value) != consumer || !self_stores.contains(value)
                                },
                            ),
                        );
                    }
                }
                // Select/Relational/ImageLoad/Literal/argument boundaries are
                // opaque leaves: their value is data, not a fuseable tree.
                _ => {}
            }
        }
        None
    };

    for (handle, expression) in function.expressions.iter() {
        let inner = function_info[handle].ty.inner_with(&module.types);
        match expression {
            Expression::Binary { op, left, right } => {
                let kind = producer_kind(inner, *op);
                if matches!(kind, Some(ProducerKind::Div)) {
                    violations.push(Violation {
                        module: module_name.to_string(),
                        function: name.to_string(),
                        line: span_line(handle),
                        kind: "raw_divide",
                        detail: "raw float `/` outside det_*; use det_div/det_rcp".to_string(),
                        site: Some(Site::at(func_site, handle)),
                    });
                }
                // Matrix products outside det_*: mat*vec / mat*mat / vec*mat
                // have per-compiler reduction order.
                if *op == BinaryOperator::Multiply {
                    for operand in [*left, *right] {
                        if matches!(
                            function_info[operand].ty.inner_with(&module.types),
                            TypeInner::Matrix { .. }
                        ) {
                            violations.push(Violation {
                                module: module_name.to_string(),
                                function: name.to_string(),
                                line: span_line(handle),
                                kind: "matrix_product",
                                detail: "matrix product outside det_*; use det_mat*_mul_vec*"
                                    .to_string(),
                                site: Some(Site::at(func_site, handle)),
                            });
                            break;
                        }
                    }
                }
                if let Some(kind) = kind {
                    // For each operand, find the producer through the
                    // transparent wrappers and check the edge rule.
                    for operand in [*left, *right] {
                        if let Some((producer, producer_handle)) = resolve(operand, Some(handle)) {
                            if is_bad_edge(producer, kind) {
                                let producer_op = match function.expressions[producer_handle] {
                                    Expression::Binary { op, .. } => op,
                                    _ => *op,
                                };
                                violations.push(Violation {
                                    module: module_name.to_string(),
                                    function: name.to_string(),
                                    line: span_line(producer_handle),
                                    kind: "unbarriered_edge",
                                    detail: format!(
                                        "{producer_op:?} result feeds {op:?} without a \
                                         det_barrier* between them"
                                    ),
                                    site: Some(Site::edge(func_site, operand, producer_handle)),
                                });
                            }
                        }
                    }
                }
            }
            Expression::Math { fun, .. } => {
                if let Some(intrinsic) = forbidden_math(*fun) {
                    violations.push(Violation {
                        module: module_name.to_string(),
                        function: name.to_string(),
                        line: span_line(handle),
                        kind: "forbidden_intrinsic",
                        detail: format!(
                            "{intrinsic} outside det_* is a per-backend lowering; \
                             route through the det_* pin"
                        ),
                        site: Some(Site::at(func_site, handle)),
                    });
                }
            }
            Expression::ImageSample { gather, .. } => {
                violations.push(Violation {
                    module: module_name.to_string(),
                    function: name.to_string(),
                    line: span_line(handle),
                    kind: "hardware_sample",
                    detail: if gather.is_some() {
                        "textureGather is hardware texel selection; use textureLoad"
                    } else {
                        "filtered/compared texture sampling is hardware texel \
                         selection; use textureLoad + pinned bilinear"
                    }
                    .to_string(),
                    site: Some(Site::at(func_site, handle)),
                });
            }
            Expression::Derivative { ctrl, .. } if *ctrl != naga::DerivativeControl::Coarse => {
                violations.push(Violation {
                    module: module_name.to_string(),
                    function: name.to_string(),
                    line: span_line(handle),
                    kind: "derivative_granularity",
                    detail: "derivative must spell the Coarse variant; implementation-chosen \
                             granularity is a divergence surface"
                        .to_string(),
                    site: Some(Site::at(func_site, handle)),
                });
            }
            _ => {}
        }
    }

    // ---- call arguments ----------------------------------------------------
    // Backends inline calls, so a caller-side product passed as an argument
    // that the callee adds to (or multiplies again) is the same contractible
    // edge as `a * b + c` spelled in one function.
    let mut consumer_memo = HashMap::new();
    for (callee, arguments) in call_sites(&function.body) {
        let consumers = param_consumer_kinds(module, info, callee, &mut consumer_memo);
        let callee_name = module.functions[callee].name.as_deref().unwrap_or("?");
        for (position, argument) in arguments.iter().enumerate() {
            let Some((producer, producer_handle)) = resolve(*argument, None) else {
                continue;
            };
            let Some(consumer) = consumers
                .get(position)
                .and_then(|kinds| kinds.iter().find(|kind| is_bad_edge(producer, **kind)))
            else {
                continue;
            };
            violations.push(Violation {
                module: module_name.to_string(),
                function: name.to_string(),
                line: span_line(producer_handle),
                kind: "unbarriered_edge",
                detail: format!(
                    "{producer:?} result passed as argument {position} of {callee_name}, which \
                     feeds it to {consumer:?} without a det_barrier* between them"
                ),
                site: Some(Site::edge(func_site, *argument, producer_handle)),
            });
        }
    }

    // ---- entry-point seeding ---------------------------------------------
    if !matches!(func_site, FuncSite::Entry(_)) {
        return;
    }
    let needs_seed =
        calls_in_body(&function.body).any(|callee| reaches_barrier.contains(&callee.index()));
    if !needs_seed {
        return;
    }
    // The seed must dominate every barrier-reaching call: only det_seed calls
    // in the entry's OWN top-level statement stream count (a seed inside a
    // conditional or helper does not dominate).
    let mut seeded = false;
    for statement in function.body.iter() {
        match statement {
            Statement::Call {
                function: callee, ..
            } => {
                let callee_name = module.functions[*callee].name.as_deref().unwrap_or("");
                if callee_name == "det_seed" {
                    seeded = true;
                } else if !seeded && reaches_barrier.contains(&callee.index()) {
                    violations.push(Violation {
                        module: module_name.to_string(),
                        function: name.to_string(),
                        line: 1,
                        kind: "missing_det_seed",
                        detail: format!(
                            "call to {callee_name} reaches det_barrier before det_seed \
                             runs; det_seed must lead the entry point's statement stream"
                        ),
                        site: Some(Site::func(func_site)),
                    });
                }
            }
            // A barrier-reaching call nested in control flow still requires
            // the top-level seed first.
            Statement::If { .. }
            | Statement::Switch { .. }
            | Statement::Loop { .. }
            | Statement::Block(_) => {
                let mut nested_seed_seen = false;
                for callee in calls_in_statement(statement) {
                    let callee_name = module.functions[callee].name.as_deref().unwrap_or("");
                    if callee_name == "det_seed" {
                        nested_seed_seen = true;
                    } else if !seeded && reaches_barrier.contains(&callee.index()) {
                        let note = if nested_seed_seen {
                            " (a det_seed nested in control flow does not dominate it)"
                        } else {
                            ""
                        };
                        violations.push(Violation {
                            module: module_name.to_string(),
                            function: name.to_string(),
                            line: 1,
                            kind: "missing_det_seed",
                            detail: format!(
                                "call to {callee_name} inside control flow reaches \
                                 det_barrier before a top-level det_seed{note}"
                            ),
                            site: Some(Site::func(func_site)),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    if !seeded {
        violations.push(Violation {
            module: module_name.to_string(),
            function: name.to_string(),
            line: 1,
            kind: "missing_det_seed",
            detail: "entry point reaches det_barrier but never calls det_seed".to_string(),
            site: Some(Site::func(func_site)),
        });
    }
}

/// Record `local -> stored expression` edges for every Store in a body
/// (over-approximates across control flow; order-insensitive by design).
fn collect_stores(
    function: &naga::Function,
    statement: &Statement,
    in_loop: bool,
    local_producers: &mut LocalStores,
) {
    match statement {
        Statement::Store { pointer, value } => {
            if let Some((local, path, exact)) = pointer_path(function, *pointer) {
                local_producers
                    .entry(local)
                    .or_default()
                    .push((path, exact, *value, in_loop));
            }
        }
        Statement::Block(block) => {
            for s in block.iter() {
                collect_stores(function, s, in_loop, local_producers);
            }
        }
        Statement::If { accept, reject, .. } => {
            for s in accept.iter().chain(reject.iter()) {
                collect_stores(function, s, in_loop, local_producers);
            }
        }
        Statement::Switch { cases, .. } => {
            for case in cases {
                for s in case.body.iter() {
                    collect_stores(function, s, in_loop, local_producers);
                }
            }
        }
        Statement::Loop {
            body, continuing, ..
        } => {
            for s in body.iter().chain(continuing.iter()) {
                collect_stores(function, s, true, local_producers);
            }
        }
        _ => {}
    }
}

/// Flag any Store that writes the private `det_zu` global outside det_seed.
fn collect_det_zu_stores(
    module: &naga::Module,
    function: &naga::Function,
    statement: &Statement,
    name: &str,
    module_name: &str,
    func_site: FuncSite,
    span_line: &dyn Fn(Handle<Expression>) -> usize,
    violations: &mut Vec<Violation>,
) {
    match statement {
        Statement::Store { pointer, value } => {
            if let Expression::GlobalVariable(global) = function.expressions[*pointer] {
                if module.global_variables[global].name.as_deref() == Some("det_zu")
                    && name != "det_seed"
                {
                    violations.push(Violation {
                        module: module_name.to_string(),
                        function: name.to_string(),
                        line: span_line(*value),
                        kind: "det_zu_access",
                        detail: "stores det_zu outside det_seed".to_string(),
                        site: Some(Site::at(func_site, *value)),
                    });
                }
            }
        }
        Statement::Block(block) => {
            for s in block.iter() {
                collect_det_zu_stores(
                    module,
                    function,
                    s,
                    name,
                    module_name,
                    func_site,
                    span_line,
                    violations,
                );
            }
        }
        Statement::If { accept, reject, .. } => {
            for s in accept.iter().chain(reject.iter()) {
                collect_det_zu_stores(
                    module,
                    function,
                    s,
                    name,
                    module_name,
                    func_site,
                    span_line,
                    violations,
                );
            }
        }
        Statement::Switch { cases, .. } => {
            for case in cases {
                for s in case.body.iter() {
                    collect_det_zu_stores(
                        module,
                        function,
                        s,
                        name,
                        module_name,
                        func_site,
                        span_line,
                        violations,
                    );
                }
            }
        }
        Statement::Loop {
            body, continuing, ..
        } => {
            for s in body.iter().chain(continuing.iter()) {
                collect_det_zu_stores(
                    module,
                    function,
                    s,
                    name,
                    module_name,
                    func_site,
                    span_line,
                    violations,
                );
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::analyze_source;

    fn violation_kinds(source: &str) -> Vec<&'static str> {
        let (module, info, violations) = analyze_source("test", source);
        assert!(module.is_some());
        assert!(info.is_some());
        violations
            .into_iter()
            .map(|violation| violation.kind)
            .collect()
    }

    #[test]
    fn verify_deep_call_tail_cannot_escape_lint() {
        let source = r#"
fn leaf(x: f32) -> f32 { return x * 2.0; }
fn hop0(x: f32) -> f32 { return leaf(x); }
fn hop1(x: f32) -> f32 { return hop0(x); }
fn hop2(x: f32) -> f32 { return hop1(x); }
fn hop3(x: f32) -> f32 { return hop2(x); }
fn hop4(x: f32) -> f32 { return hop3(x); }
fn hop5(x: f32) -> f32 { return hop4(x); }
fn hop6(x: f32) -> f32 { return hop5(x); }
fn hop7(x: f32) -> f32 { return hop6(x); }
fn hop8(x: f32) -> f32 { return hop7(x); }
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let product = hop8(f32(gid.x));
    let sum = product + 1.0;
}
"#;
        assert!(violation_kinds(source).contains(&"unbarriered_edge"));
    }

    #[test]
    fn verify_product_argument_fused_by_callee_is_rejected() {
        let source = r#"
fn add_one(x: f32) -> f32 { return x + 1.0; }
fn forward(x: f32) -> f32 { return add_one(x); }
fn caller(a: f32, b: f32) -> f32 { return forward(a * b); }
"#;
        assert!(violation_kinds(source).contains(&"unbarriered_edge"));
    }

    #[test]
    fn verify_barriered_argument_passes() {
        let source = r#"
var<private> det_zu: u32 = 0u;
fn det_barrier(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | det_zu);
}
fn add_one(x: f32) -> f32 { return x + 1.0; }
fn caller(a: f32, b: f32) -> f32 { return add_one(det_barrier(a * b)); }
"#;
        assert!(!violation_kinds(source).contains(&"unbarriered_edge"));
    }

    #[test]
    fn verify_identity_callee_is_transparent() {
        let source = r#"
fn identity(x: f32) -> f32 { return x; }
fn caller(a: f32, b: f32) -> f32 { return identity(a * b) + 1.0; }
"#;
        assert!(violation_kinds(source).contains(&"unbarriered_edge"));
    }

    #[test]
    fn verify_struct_field_store_edge_is_rejected() {
        let source = r#"
struct Pair { v: f32, w: f32 }
fn caller(a: f32, b: f32) -> f32 {
    var pair: Pair;
    pair.v = a * b;
    return pair.v + 1.0;
}
"#;
        assert!(violation_kinds(source).contains(&"unbarriered_edge"));
    }

    #[test]
    fn verify_product_in_another_lane_is_not_blamed() {
        let source = r#"
fn caller(receiver: f32, b: f32) -> f32 {
    var coefficients = vec3<f32>(1.0, receiver, receiver * receiver);
    coefficients.y -= b;
    return coefficients.y;
}
"#;
        assert!(!violation_kinds(source).contains(&"unbarriered_edge"));
    }

    #[test]
    fn verify_raw_float_divide_is_rejected() {
        let source = "fn quotient(x: f32) -> f32 { return x / 3.0; }";
        assert!(violation_kinds(source).contains(&"raw_divide"));
    }

    #[test]
    fn verify_barrier_reaching_entry_requires_seed() {
        let source = r#"
var<private> det_zu: u32 = 0u;
fn det_barrier(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | det_zu);
}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let value = det_barrier(f32(gid.x));
}
"#;
        assert!(violation_kinds(source).contains(&"missing_det_seed"));
    }
}
