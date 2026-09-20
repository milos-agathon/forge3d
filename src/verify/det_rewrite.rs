//! TERRA-DETERMINATA source rewriter: turns naga-IR lint violations into
//! textual edits on the `.wgsl` source files.
//!
//! The lint (`determinism_lint`) runs on the assembled module; each
//! violation carries the expression handles of its site. This module maps
//! naga spans in the *assembled* text back to byte ranges in the constituent
//! files (via `shader_sources::SourcePart`) and emits the pinned-arithmetic
//! rewrite:
//!
//! - `a*b` feeding `+`/`-`, `a+b` feeding `+`/`-`, `a*b` feeding `*` ->
//!   wrap the producer in `det_barrier{,2,3,4}`.
//! - `x / y` -> `det_div{,2,3,4}(x, y)` (scalar operands splat to the
//!   result's lane count).
//! - matrix `*` products -> `det_mat{3,4}_mul_vec{3,4}` /
//!   `det_vec{3,4}_mul_mat{3,4}` / `det_mat{3,4}_mul_mat{3,4}`.
//! - the forbidden intrinsics -> their `det_*` pins by name and arity.
//! - `textureSample*` -> `det_sample*` / `det_load*` manual reconstructions.
//! - `dpdx`/`dpdy`/`fwidth` -> the `*Coarse` spellings.
//! - entry points that reach `det_barrier*` without a leading `det_seed`
//!   get one inserted as the first statement.
//!
//! Run with: `FORGE3D_DET_REWRITE=1 cargo test det_instrument_rewrite --lib`
//! (a dry run prints every planned edit; without the env var the test only
//! reports which files would change).

#[cfg(test)]
mod instrument {
    use super::super::determinism_lint::{analyze_source, FuncSite, Violation};
    use crate::shader_sources::{deterministic_module_parts, SourcePart, MODULE_REWRITES};
    use naga::{Expression, Handle, TypeInner};
    use std::path::PathBuf;

    /// An edit in assembled-source coordinates. Nested edits (a wrap whose
    /// span contains another violation's site) are resolved by emitting the
    /// outer replacement around the recursively rewritten inner text.
    #[derive(Debug)]
    struct AsmEdit {
        start: usize,
        end: usize,
        payload: Payload,
    }

    /// One operand slot of a generated call: either a source span to fill
    /// with rewritten text or an already-materialized string (synthetic
    /// expressions such as the splat in `scalar / vec`).
    #[derive(Debug)]
    enum Operand {
        Span(usize, usize),
        Text(String),
    }

    #[derive(Debug)]
    enum Payload {
        /// `name(<rewritten inner>)`
        Wrap(&'static str),
        /// `name(<arg0>, <arg1>)`
        Call { name: String, args: Vec<Operand> },
        /// `<lhs> = name(<arg0>, <arg1>)` — a compound-assign (`x /= y`)
        /// whose divide span covers the whole statement.
        Store {
            lhs: String,
            name: String,
            args: Vec<Operand>,
        },
        /// Replace the span (a callee name) outright.
        Name(String),
        /// Zero-width insertion.
        Insert(String),
    }

    /// Assembled-text span of one source part, plus the per-surviving-line
    /// table needed to map back to file offsets when `#include` lines were
    /// stripped.
    struct PartMap {
        path: &'static str,
        /// (assembled byte offset of line start, file byte offset of line
        /// start, file line byte length) per surviving line.
        lines: Vec<(usize, usize, usize)>,
        /// File-coordinate ranges where an assembly rewrite changed the
        /// text: edits landing here would write module text into a file
        /// that differs at those bytes.
        rewritten: Vec<(usize, usize)>,
    }

    fn assemble_mapped(module_name: &str, parts: &[SourcePart]) -> (String, Vec<PartMap>) {
        let mut out = String::new();
        let mut maps = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let mut rewritten = Vec::new();
            // Read the CURRENT file text so each rewrite iteration sees the
            // edits applied so far.
            let disk = std::fs::read_to_string(part.path).unwrap_or_else(|_| part.text.to_string());
            let text: std::borrow::Cow<str> = match MODULE_REWRITES
                .iter()
                .find(|(m, path, ..)| *m == module_name && *path == part.path)
            {
                Some((_, _, from, to)) => {
                    assert_eq!(
                        from.len(),
                        to.len(),
                        "module rewrite must be length-preserving"
                    );
                    let mut start = 0;
                    while let Some(off) = disk[start..].find(from) {
                        let abs = start + off;
                        rewritten.push((abs, abs + from.len()));
                        start = abs + from.len();
                    }
                    disk.replace(from, to).into()
                }
                None => disk.into(),
            };
            let mut lines = Vec::new();
            for file_line in text.split_inclusive('\n') {
                let file_off = file_line.as_ptr() as usize - text.as_ptr() as usize;
                let trimmed = file_line.trim_start();
                if part.strip && trimmed.starts_with("#include") {
                    continue;
                }
                let asm_off = out.len();
                out.push_str(file_line);
                lines.push((asm_off, file_off, file_line.len()));
            }
            if text.is_empty() {
                lines.push((out.len(), 0, 0));
            }
            maps.push(PartMap {
                path: part.path,
                lines,
                rewritten,
            });
        }
        (out, maps)
    }

    /// Map an assembled-text byte span to a file byte span. Returns None when
    /// the span crosses a part boundary or lands on stripped text.
    fn map_span(
        part_index: &[(usize, usize)],
        maps: &[PartMap],
        start: usize,
        end: usize,
    ) -> Option<(&'static str, usize, usize)> {
        let part = part_index
            .iter()
            .position(|(s, e)| start >= *s && end <= *e)?;
        let map = &maps[part];
        let line_of = |off: usize| -> Option<usize> {
            map.lines
                .iter()
                .position(|(a, _, len)| off >= *a && off < a + len)
        };
        let ls = line_of(start)?;
        let le = line_of(end.saturating_sub(1).max(start))?;
        let (as_, fs, _) = map.lines[ls];
        let (ae, fe, _) = map.lines[le];
        Some((map.path, fs + (start - as_), fe + (end - ae)))
    }

    /// End offset (exclusive) of the call whose callee name ends at
    /// `name_end`: skips whitespace, expects `(`, balances to `)`.
    fn call_end(source: &str, name_end: usize) -> Option<usize> {
        let bytes = source.as_bytes();
        let mut i = name_end;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'(' {
            return None;
        }
        let mut depth = 0i32;
        for j in i..bytes.len() {
            match bytes[j] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(j + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn lanes(inner: &TypeInner) -> usize {
        match inner {
            TypeInner::Vector { size, .. } => match size {
                naga::VectorSize::Bi => 2,
                naga::VectorSize::Tri => 3,
                naga::VectorSize::Quad => 4,
            },
            _ => 1,
        }
    }

    fn barrier_name(lanes: usize) -> &'static str {
        match lanes {
            1 => "det_barrier",
            2 => "det_barrier2",
            3 => "det_barrier3",
            _ => "det_barrier4",
        }
    }

    fn div_name(lanes: usize) -> &'static str {
        match lanes {
            1 => "det_div",
            2 => "det_div2",
            3 => "det_div3",
            _ => "det_div4",
        }
    }

    fn det_intrinsic_name(
        fun: naga::MathFunction,
        result_lanes: usize,
        arg_lanes: usize,
        arg2_lanes: usize,
    ) -> Option<String> {
        use naga::MathFunction as M;
        let suf = |n: usize| {
            if n == 1 {
                String::new()
            } else {
                n.to_string()
            }
        };
        // digit-ending bases take an underscore separator so the vector
        // name can't collide with the scalar (det_exp2 vs det_exp2_3).
        let dsuf = |n: usize| {
            if n == 1 {
                String::new()
            } else {
                format!("_{n}")
            }
        };
        Some(match fun {
            M::Dot => format!("det_dot{}", arg_lanes.max(2)),
            M::Mix => format!(
                "det_mix{}{}",
                suf(result_lanes),
                if arg2_lanes > 1 { "v" } else { "" }
            ),
            M::Fma => format!("det_fma{}", suf(result_lanes)),
            M::Pow => format!("det_pow{}", suf(result_lanes)),
            // exp/log/atan vec2 spell det_<f>_2: the bare det_<f>2 slots
            // already name the scalar exp2/log2/atan2 pins.
            M::Exp => format!(
                "det_exp{}",
                if result_lanes == 2 {
                    "_2".into()
                } else {
                    suf(result_lanes)
                }
            ),
            M::Exp2 => format!("det_exp2{}", dsuf(result_lanes)),
            M::Log => format!(
                "det_log{}",
                if result_lanes == 2 {
                    "_2".into()
                } else {
                    suf(result_lanes)
                }
            ),
            M::Log2 => format!("det_log2{}", dsuf(result_lanes)),
            M::Sqrt => format!("det_sqrt{}", suf(result_lanes)),
            M::InverseSqrt => format!("det_inverse_sqrt{}", suf(result_lanes)),
            M::Length => format!("det_length{}", arg_lanes.max(2)),
            M::Distance => format!("det_distance{}", arg_lanes.max(2)),
            M::Normalize => format!("det_normalize{}", arg_lanes.max(2)),
            M::Cross => "det_cross3".to_string(),
            M::Reflect => format!("det_reflect{}", arg_lanes.max(2)),
            M::SmoothStep => format!("det_smoothstep{}", suf(result_lanes)),
            M::Sin => format!("det_sin{}", suf(result_lanes)),
            M::Cos => format!("det_cos{}", suf(result_lanes)),
            M::Tan => format!("det_tan{}", suf(result_lanes)),
            M::Asin => format!("det_asin{}", suf(result_lanes)),
            M::Acos => format!("det_acos{}", suf(result_lanes)),
            M::Atan => format!(
                "det_atan{}",
                if result_lanes == 2 {
                    "_2".into()
                } else {
                    suf(result_lanes)
                }
            ),
            M::Atan2 => format!("det_atan2{}", dsuf(result_lanes)),
            M::Transpose => format!("det_transpose{}", suf(result_lanes)),
            _ => return None,
        })
    }

    /// Build the structured edit for one violation. `source` is the
    /// assembled text the spans index into.
    fn edit_for(
        v: &Violation,
        module: &naga::Module,
        info: &naga::valid::ModuleInfo,
        source: &str,
    ) -> Result<AsmEdit, String> {
        let site = v.site.ok_or("no site")?;
        let (function, function_info) = match site.func {
            FuncSite::Regular(h) => (&module.functions[h], &info[h]),
            FuncSite::Entry(i) => (&module.entry_points[i].function, info.get_entry_point(i)),
        };
        let span = |h: Handle<Expression>| -> Result<(usize, usize), String> {
            function
                .expressions
                .get_span(h)
                .to_range()
                .map(|r| (r.start, r.end))
                .ok_or_else(|| "expression has no source span".to_string())
        };
        let expr_info = |h: Handle<Expression>| function_info[h].ty.inner_with(&module.types);

        // Source text for an operand. Synthetic nodes (the splat naga
        // inserts for `scalar / vec`, or the reload of `a /= b`) have no
        // span; rebuild their text structurally instead.
        fn operand_text(
            module: &naga::Module,
            function: &naga::Function,
            source: &str,
            h: Handle<Expression>,
        ) -> Result<Operand, String> {
            // A let-bound handle is shared by every use site and its span
            // points at the `let` definition, so the use-site text is the
            // bound name, not the def-site expression.
            if let Some(name) = function.named_expressions.get(&h) {
                return Ok(Operand::Text(name.clone()));
            }
            // Splat carries a span over just the scalar text even though it
            // is a synthetic vecN node; synthesize before the span lookup.
            if let Expression::Splat { size, value } = &function.expressions[h] {
                let k = match size {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                let inner = operand_string(module, function, source, *value)?;
                return Ok(Operand::Text(format!("vec{k}<f32>({inner})")));
            }
            if let Some(r) = function.expressions.get_span(h).to_range() {
                // Math/Derivative spans cover the callee name only; extend
                // through the balanced argument list so `x / max(a, b)`
                // emits `max(a, b)` rather than a bare `max`.
                if matches!(
                    function.expressions[h],
                    Expression::Math { .. } | Expression::Derivative { .. }
                ) {
                    if let Some(end) = call_end(source, r.end) {
                        return Ok(Operand::Span(r.start, end));
                    }
                }
                return Ok(Operand::Span(r.start, r.end));
            }
            match &function.expressions[h] {
                Expression::Literal(lit) => Ok(Operand::Text(match lit {
                    naga::Literal::F32(v) => format!("{v:?}"),
                    naga::Literal::F64(v) => format!("{v:?}"),
                    naga::Literal::U32(v) => format!("{v}u"),
                    naga::Literal::I32(v) => format!("{v}"),
                    naga::Literal::Bool(v) => format!("{v}"),
                    _ => return Err("unsupported literal".into()),
                })),
                Expression::Load { pointer } => operand_text(module, function, source, *pointer),
                Expression::LocalVariable(v) => function
                    .local_variables
                    .try_get(*v)
                    .ok()
                    .and_then(|v| v.name.clone())
                    .map(Operand::Text)
                    .ok_or_else(|| "unnamed local".to_string()),
                Expression::GlobalVariable(v) => module
                    .global_variables
                    .try_get(*v)
                    .ok()
                    .and_then(|v| v.name.clone())
                    .map(Operand::Text)
                    .ok_or_else(|| "unnamed global".to_string()),
                Expression::Constant(v) => module
                    .constants
                    .try_get(*v)
                    .ok()
                    .and_then(|v| v.name.clone())
                    .map(Operand::Text)
                    .ok_or_else(|| "unnamed constant".to_string()),
                Expression::FunctionArgument(i) => function
                    .arguments
                    .get(*i as usize)
                    .and_then(|a| a.name.clone())
                    .map(Operand::Text)
                    .ok_or_else(|| "unnamed argument".to_string()),
                Expression::Access { base, index } => {
                    let b = operand_string(module, function, source, *base)?;
                    let i = operand_string(module, function, source, *index)?;
                    Ok(Operand::Text(format!("{b}[{i}]")))
                }
                Expression::AccessIndex { base, index } => {
                    let b = operand_string(module, function, source, *base)?;
                    let c = "xyzw".chars().nth(*index as usize);
                    match c {
                        Some(c) => Ok(Operand::Text(format!("{b}.{c}"))),
                        None => Ok(Operand::Text(format!("{b}[{index}]"))),
                    }
                }
                other => Err(format!("unspanned expression {other:?}")),
            }
        }

        fn operand_string(
            module: &naga::Module,
            function: &naga::Function,
            source: &str,
            h: Handle<Expression>,
        ) -> Result<String, String> {
            match operand_text(module, function, source, h)? {
                Operand::Span(s, e) => Ok(source[s..e].to_string()),
                Operand::Text(t) => Ok(t),
            }
        }

        match v.kind {
            "unbarriered_edge" => {
                // Wrap the consumer's operand (the use site). For
                // let-inlined operands this is the producer handle itself —
                // the def-site span. For var loads / call results it is the
                // use-site span; the producer's own span can be a whole
                // `a += f(..)` statement, which cannot be wrapped.
                let mut target = site.operand.or(site.related).ok_or("no operand")?;
                // `vec + scalar` makes naga splat the scalar; the splat's
                // span covers the operand text, so wrap the scalar inside —
                // the parse re-splats the barriered value.
                while let Expression::Splat { value, .. } = function.expressions[target] {
                    target = value;
                }
                let (s, e, k) = match span(target) {
                    Ok((s, e)) => (s, e, lanes(expr_info(target))),
                    // Unspanned operand (synthetic load of a `var`): fall
                    // back to the producer's own span — wrapping its result
                    // barriers the value at the store, which pins the edge
                    // for every later load.
                    Err(_) => {
                        let prod = site.related.ok_or("no producer")?;
                        let (s, e) = span(prod)?;
                        (s, e, lanes(expr_info(prod)))
                    }
                };
                // det_barrier* take float scalars/vectors only: a struct or
                // matrix operand (e.g. a helper returning a struct whose
                // field is a raw product) must be barriered inside the
                // producer by hand.
                let barrierable = |h: Handle<Expression>| {
                    matches!(
                        expr_info(h),
                        TypeInner::Scalar(naga::Scalar {
                            kind: naga::ScalarKind::Float,
                            ..
                        }) | TypeInner::Vector {
                            scalar: naga::Scalar {
                                kind: naga::ScalarKind::Float,
                                ..
                            },
                            ..
                        }
                    )
                };
                if !barrierable(target) {
                    return Err(format!(
                        "operand is not a float scalar/vector ({:?}); barrier the producer by hand",
                        expr_info(target)
                    ));
                }
                // The operand is the target of a compound assignment
                // (`x.y -= rhs`): an lvalue cannot be wrapped, so expand to
                // `x.y = det_barrier(x.y) - (rhs)`, which is the same value.
                let after = source[e..].trim_start();
                if let Some(op) = ["+=", "-=", "*="]
                    .into_iter()
                    .find(|op| after.starts_with(op))
                {
                    let rhs_start = e + (source[e..].len() - after.len()) + op.len();
                    let mut depth = 0i32;
                    let rhs_end = source[rhs_start..]
                        .char_indices()
                        .find_map(|(i, c)| {
                            match c {
                                '(' | '[' => depth += 1,
                                ')' | ']' => depth -= 1,
                                ';' if depth == 0 => return Some(rhs_start + i),
                                _ => {}
                            }
                            None
                        })
                        .ok_or("compound assignment without `;`")?;
                    let lhs = &source[s..e];
                    let rhs = source[rhs_start..rhs_end].trim();
                    return Ok(AsmEdit {
                        start: s,
                        end: rhs_end,
                        payload: Payload::Name(format!(
                            "{lhs} = {}({lhs}) {} ({rhs})",
                            barrier_name(k),
                            &op[..1]
                        )),
                    });
                }
                Ok(AsmEdit {
                    start: s,
                    end: e,
                    payload: Payload::Wrap(barrier_name(k)),
                })
            }
            "raw_divide" => {
                let h = site.expr.ok_or("no expr")?;
                let Expression::Binary { left, right, .. } = function.expressions[h] else {
                    return Err("not a binary".into());
                };
                let k = lanes(function_info[h].ty.inner_with(&module.types));
                let lk = lanes(expr_info(left));
                let rk = lanes(expr_info(right));
                let splat = |op: Operand, n: usize| -> Operand {
                    match op {
                        Operand::Span(s, e) => {
                            Operand::Text(format!("vec{n}<f32>({})", &source[s..e]))
                        }
                        Operand::Text(t) => Operand::Text(format!("vec{n}<f32>({t})")),
                    }
                };
                let lop = if lk == 1 && k > 1 {
                    splat(operand_text(module, function, source, left)?, k)
                } else {
                    operand_text(module, function, source, left)?
                };
                let rop = if rk == 1 && k > 1 {
                    splat(operand_text(module, function, source, right)?, k)
                } else {
                    operand_text(module, function, source, right)?
                };
                let (s, e) = span(h).map_err(|e| format!("divide site: {e}"))?;
                // `x /= y` lowers to a store of a divide whose naga span
                // covers the whole statement. A bare `det_div(x, y)` there
                // would discard the result; emit an explicit assignment.
                if source[s..e].contains("/=") {
                    let lhs = match &lop {
                        Operand::Span(ls, le) => source[*ls..*le].to_string(),
                        Operand::Text(t) => t.clone(),
                    };
                    return Ok(AsmEdit {
                        start: s,
                        end: e,
                        payload: Payload::Store {
                            lhs,
                            name: div_name(k).to_string(),
                            args: vec![lop, rop],
                        },
                    });
                }
                Ok(AsmEdit {
                    start: s,
                    end: e,
                    payload: Payload::Call {
                        name: div_name(k).to_string(),
                        args: vec![lop, rop],
                    },
                })
            }
            "matrix_product" => {
                let h = site.expr.ok_or("no expr")?;
                let Expression::Binary { left, right, .. } = function.expressions[h] else {
                    return Err("not a binary".into());
                };
                let lop = operand_text(module, function, source, left)?;
                let rop = operand_text(module, function, source, right)?;
                let linner = expr_info(left);
                let rinner = expr_info(right);
                let mdim = |i: &TypeInner| match i {
                    TypeInner::Matrix { columns, rows, .. } => {
                        let to_i = |s: naga::VectorSize| match s {
                            naga::VectorSize::Bi => 2,
                            naga::VectorSize::Tri => 3,
                            naga::VectorSize::Quad => 4,
                        };
                        (to_i(*columns), to_i(*rows))
                    }
                    _ => (0, 0),
                };
                let name = match (linner, rinner) {
                    (TypeInner::Matrix { .. }, TypeInner::Vector { .. }) => {
                        let (c, _) = mdim(linner);
                        format!("det_mat{c}_mul_vec{c}")
                    }
                    (TypeInner::Vector { .. }, TypeInner::Matrix { .. }) => {
                        let (_, c) = mdim(rinner);
                        format!("det_vec{c}_mul_mat{c}")
                    }
                    (TypeInner::Matrix { .. }, TypeInner::Matrix { .. }) => {
                        let (c, _) = mdim(linner);
                        format!("det_mat{c}_mul_mat{c}")
                    }
                    _ => return Err("not a matrix product".into()),
                };
                let (s, e) =
                    span(h).map_err(|e| format!("mat site {:?}: {e}", function.expressions[h]))?;
                Ok(AsmEdit {
                    start: s,
                    end: e,
                    payload: Payload::Call {
                        name,
                        args: vec![lop, rop],
                    },
                })
            }
            "forbidden_intrinsic" | "derivative_granularity" => {
                // naga spans these expressions over the callee name only.
                let h = site.expr.ok_or("no expr")?;
                let name = match &function.expressions[h] {
                    Expression::Math {
                        fun,
                        arg,
                        arg1: _,
                        arg2,
                        ..
                    } => {
                        let rl = lanes(function_info[h].ty.inner_with(&module.types));
                        // argument lanes drive helpers that reduce (dot,
                        // length, distance, normalize); result lanes drive
                        // elementwise ones (pow, exp, sin, mix, ...).
                        let al = lanes(expr_info(*arg));
                        let al2 = arg2.map(|a| lanes(expr_info(a))).unwrap_or(1);
                        det_intrinsic_name(*fun, rl, al, al2)
                            .ok_or_else(|| format!("no det_ pin for {fun:?}"))?
                    }
                    Expression::Derivative { axis, .. } => match axis {
                        naga::DerivativeAxis::X => "dpdxCoarse".to_string(),
                        naga::DerivativeAxis::Y => "dpdyCoarse".to_string(),
                        naga::DerivativeAxis::Width => "fwidthCoarse".to_string(),
                    },
                    _ => return Err("not a math/derivative".into()),
                };
                let (s, e) = span(h)?;
                Ok(AsmEdit {
                    start: s,
                    end: e,
                    payload: Payload::Name(name),
                })
            }
            _ => Err(format!("kind {} has no rewrite", v.kind)),
        }
    }

    /// `det_seed` insertion for an entry point: assembled offset just after
    /// the body's opening brace plus the statement text.
    fn seed_insert(
        module: &naga::Module,
        entry_index: usize,
        source: &str,
    ) -> Result<(usize, String), String> {
        let entry = &module.entry_points[entry_index];
        let mut seed_expr: Option<String> = None;
        let seed_for = |b: &naga::BuiltIn, path: String| -> Option<String> {
            Some(match b {
                naga::BuiltIn::GlobalInvocationId => format!("f32({path}.x)"),
                naga::BuiltIn::LocalInvocationId => format!("f32({path}.x)"),
                naga::BuiltIn::VertexIndex | naga::BuiltIn::InstanceIndex => {
                    format!("f32({path})")
                }
                naga::BuiltIn::Position { .. } => format!("{path}.x"),
                _ => return None,
            })
        };
        for arg in entry.function.arguments.iter() {
            let name = arg.name.as_deref().unwrap_or("_det_arg");
            if let Some(naga::Binding::BuiltIn(b)) = &arg.binding {
                if let Some(e) = seed_for(b, name.to_string()) {
                    seed_expr = Some(e);
                    break;
                }
            }
            let TypeInner::Struct { ref members, .. } = module.types[arg.ty].inner else {
                continue;
            };
            for member in members.iter() {
                let Some(naga::Binding::BuiltIn(b)) = &member.binding else {
                    continue;
                };
                let path = format!("{name}.{}", member.name.as_deref().unwrap_or("_det_m"));
                if let Some(e) = seed_for(b, path) {
                    seed_expr = Some(e);
                    break;
                }
            }
            if seed_expr.is_some() {
                break;
            }
        }
        // Fallback: no builtin at all (e.g. clipmap vertex stage with only
        // @location inputs). Any lane-varying argument pins the barrier;
        // use the first argument, extracting a scalar f32.
        if seed_expr.is_none() {
            let lane_path = |name: &str, inner: &TypeInner| -> Option<String> {
                Some(match inner {
                    TypeInner::Vector { .. } => format!("{name}.x"),
                    TypeInner::Scalar(naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        ..
                    }) => name.to_string(),
                    TypeInner::Scalar(_) => format!("f32({name})"),
                    TypeInner::Struct { ref members, .. } => {
                        let m = members.iter().find(|m| {
                            matches!(
                                &module.types[m.ty].inner,
                                TypeInner::Vector { .. } | TypeInner::Scalar(_)
                            )
                        })?;
                        let mn = m.name.as_deref().unwrap_or("_det_m");
                        match &module.types[m.ty].inner {
                            TypeInner::Vector { .. } => format!("{name}.{mn}.x"),
                            TypeInner::Scalar(naga::Scalar {
                                kind: naga::ScalarKind::Float,
                                ..
                            }) => format!("{name}.{mn}"),
                            _ => format!("f32({name}.{mn})"),
                        }
                    }
                    _ => return None,
                })
            };
            for arg in entry.function.arguments.iter() {
                let name = arg.name.as_deref().unwrap_or("_det_arg");
                if let Some(e) = lane_path(name, &module.types[arg.ty].inner) {
                    seed_expr = Some(e);
                    break;
                }
            }
        }
        let seed = seed_expr
            .ok_or_else(|| format!("entry {} has no lane input to seed from", entry.name))?;
        let needle = format!("fn {}", entry.name);
        let pos = source
            .find(&needle)
            .ok_or_else(|| format!("cannot find {needle}"))?;
        let brace = source[pos..].find('{').ok_or("no body brace")? + pos;
        Ok((brace + 1, format!("\n    det_seed({seed});")))
    }

    /// Render one edit's replacement text directly from the assembled
    /// source. Nested violations inside the replaced text are deliberately
    /// NOT resolved here: each iteration applies exactly one edit and
    /// re-lints, so inner sites are rewritten on later passes.
    fn render_edit(source: &str, e: &AsmEdit) -> String {
        match &e.payload {
            Payload::Wrap(name) => format!("{name}({})", &source[e.start..e.end]),
            Payload::Name(name) => name.clone(),
            Payload::Insert(text) => text.clone(),
            Payload::Call { name, args } => {
                let rendered: Vec<String> = args
                    .iter()
                    .map(|a| match a {
                        Operand::Span(s, e2) => source[*s..*e2].to_string(),
                        Operand::Text(t) => t.clone(),
                    })
                    .collect();
                format!("{name}({})", rendered.join(", "))
            }
            Payload::Store { lhs, name, args } => {
                let rendered: Vec<String> = args
                    .iter()
                    .map(|a| match a {
                        Operand::Span(s, e2) => source[*s..*e2].to_string(),
                        Operand::Text(t) => t.clone(),
                    })
                    .collect();
                format!("{lhs} = {name}({})", rendered.join(", "))
            }
        }
    }

    /// Scan every deterministic module and return the first fixable
    /// violation as a file-level edit. Unfixable violations are reported
    /// once through `skipped`.
    fn first_fixable(
        skipped: &mut Vec<String>,
        seen_skips: &mut std::collections::BTreeSet<String>,
    ) -> Option<(PathBuf, usize, usize, String)> {
        for (name, parts) in deterministic_module_parts() {
            let (source, maps) = assemble_mapped(name, &parts);
            let mut part_index = Vec::new();
            let mut off = 0usize;
            for (i, part) in parts.iter().enumerate() {
                if i > 0 {
                    off += 1;
                }
                let start = off;
                let len: usize = if part.strip {
                    let disk = std::fs::read_to_string(part.path)
                        .unwrap_or_else(|_| part.text.to_string());
                    disk.split_inclusive('\n')
                        .filter(|l| !l.trim_start().starts_with("#include"))
                        .map(|l| l.len())
                        .sum()
                } else {
                    maps[part_index.len()].lines.iter().map(|(_, _, l)| l).sum()
                };
                off += len;
                part_index.push((start, off));
            }

            let (module, info, violations) = analyze_source(name, &source);
            let (Some(module), Some(info)) = (module, info) else {
                for v in violations {
                    let line = format!("{v}");
                    if seen_skips.insert(line.clone()) {
                        skipped.push(line);
                    }
                }
                continue;
            };

            for v in &violations {
                let edit = match v.kind {
                    "parse" | "validate" | "det_zu_access" => {
                        let line = format!("{v}");
                        if seen_skips.insert(line.clone()) {
                            skipped.push(line);
                        }
                        continue;
                    }
                    "missing_det_seed" => {
                        let Some(site) = v.site else { continue };
                        let FuncSite::Entry(idx) = site.func else {
                            continue;
                        };
                        match seed_insert(&module, idx, &source) {
                            Ok((off, text)) => AsmEdit {
                                start: off,
                                end: off,
                                payload: Payload::Insert(text),
                            },
                            Err(m) => {
                                let line = format!("{name}: {m}");
                                if seen_skips.insert(line.clone()) {
                                    skipped.push(line);
                                }
                                continue;
                            }
                        }
                    }
                    "hardware_sample" => {
                        let line = format!("{v}  [manual: software sample]");
                        if seen_skips.insert(line.clone()) {
                            skipped.push(line);
                        }
                        continue;
                    }
                    _ => match edit_for(v, &module, &info, &source) {
                        Ok(e) => e,
                        Err(m) => {
                            let line = format!("{name}:{}: {}: {m}", v.function, v.kind);
                            if seen_skips.insert(line.clone()) {
                                skipped.push(line);
                            }
                            continue;
                        }
                    },
                };
                let text = render_edit(&source, &edit);
                match map_span(&part_index, &maps, edit.start, edit.end) {
                    Some((path, fs, fe)) => {
                        let hit_rewrite = maps.iter().any(|m| {
                            m.path == path
                                && m.rewritten.iter().any(|(rs, re)| fs < *re && fe > *rs)
                        });
                        if hit_rewrite {
                            let line =
                                format!("{name}: {path}:{fs}..{fe} lands on an assembly rewrite");
                            if seen_skips.insert(line.clone()) {
                                skipped.push(line);
                            }
                            continue;
                        }
                        return Some((PathBuf::from(path), fs, fe, text));
                    }
                    None => {
                        let line = format!(
                            "{name}: span {}..{} crosses a part boundary",
                            edit.start, edit.end
                        );
                        if seen_skips.insert(line.clone()) {
                            skipped.push(line);
                        }
                        continue;
                    }
                }
            }
        }
        None
    }

    /// Iterate `first_fixable` until convergence. In dry-run mode
    /// (`apply == false`) nothing is written and the loop would repeat the
    /// same first edit, so a single projected pass reports instead.
    fn rewrite_all(apply: bool) -> (usize, Vec<String>) {
        let mut skipped = Vec::new();
        let mut seen_skips = std::collections::BTreeSet::new();
        if !apply {
            let mut count = 0usize;
            let mut probe = Vec::new();
            let mut probe_seen = std::collections::BTreeSet::new();
            for (name, parts) in deterministic_module_parts() {
                let (source, _) = assemble_mapped(name, &parts);
                let (_, _, violations) = analyze_source(name, &source);
                for v in &violations {
                    match v.kind {
                        "unbarriered_edge"
                        | "raw_divide"
                        | "matrix_product"
                        | "forbidden_intrinsic"
                        | "derivative_granularity"
                        | "missing_det_seed" => {
                            count += 1;
                            if std::env::var_os("FORGE3D_DET_LINT_VERBOSE").is_some() {
                                eprintln!("  fixable: {v}");
                            }
                        }
                        _ => {
                            let line = format!("{v}");
                            if probe_seen.insert(line.clone()) {
                                probe.push(line);
                            }
                        }
                    }
                }
            }
            skipped.extend(probe);
            return (count, skipped);
        }
        let mut count = 0usize;
        while let Some((path, s, e, text)) = first_fixable(&mut skipped, &mut seen_skips) {
            let mut file = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
            file.replace_range(s..e, &text);
            std::fs::write(&path, &file)
                .unwrap_or_else(|err| panic!("cannot write {}: {err}", path.display()));
            count += 1;
            if count.is_multiple_of(200) {
                eprintln!("  det rewrite: {count} edits applied...");
            }
        }
        (count, skipped)
    }

    #[test]
    fn det_instrument_rewrite() {
        let apply = std::env::var_os("FORGE3D_DET_REWRITE").is_some();
        let (count, skipped) = rewrite_all(apply);
        eprintln!(
            "determinism rewrite: {count} edits{}",
            if apply { " applied" } else { " (dry run)" }
        );
        for line in &skipped {
            eprintln!("  skip: {line}");
        }
        if !apply {
            assert_eq!(
                count, 0,
                "determinism IR lint found {count} fixable violation(s); run with FORGE3D_DET_REWRITE=1 to apply one-edit rewrites"
            );
            let hard = skipped
                .iter()
                .filter(|line| !line.contains(": hardware_sample:"))
                .collect::<Vec<_>>();
            assert!(
                hard.is_empty(),
                "determinism IR lint found non-advisory violation(s): {hard:#?}"
            );
        }
    }
}
