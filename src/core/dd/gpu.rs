use super::generator::canary_pair;
use super::gpu_exec::HarnessPipeline;
use super::gpu_report::{DdHarnessReport, DdOperation, DdSelftestReport, TwoProdVariant};
use super::proof::{bits_equal, reduce_chunk, selected_product};
use super::*;
use crate::core::error::{RenderError, RenderResult};
use once_cell::sync::OnceCell;

const ADVERSARIAL_COUNT: u64 = 1_000_000;
const MIN_GENERATED_COUNT: u64 = 100_000_000;
const CHUNK: u32 = 262_144;

static CAPABILITY: OnceCell<Result<DdSelftestReport, String>> = OnceCell::new();

fn adapter_fields(context: &crate::core::gpu::GpuContext) -> (String, String) {
    let info = context.adapter.get_info();
    (format!("{:?}", info.backend).to_lowercase(), info.name)
}

fn run_candidate(
    context: &crate::core::gpu::GpuContext,
    variant: TwoProdVariant,
    force_failure: bool,
) -> RenderResult<DdSelftestReport> {
    let pipeline = HarnessPipeline::new(&context.device, variant)?;
    let outputs = pipeline.dispatch(&context.device, &context.queue, 0, 10, 2, 0, false)?;
    let mut mismatches = 0;
    let mut failure_details = Vec::new();
    for (index, output) in outputs.iter().enumerate() {
        let (a, b) = canary_pair(index as u32);
        let sum = two_sum(a, b);
        let product = selected_product(a, b, variant);
        let exact_sum = a as f64 + b as f64;
        let exact_product = a as f64 * b as f64;
        let check_sum = index != 4;
        // The two-denormal sum verifies two_sum; its product is
        // below binary32 range and therefore has no error-free representation.
        let check_product = index != 2;
        if check_sum && (!bits_equal(output.primary, sum) || output.primary.to_f64() != exact_sum) {
            failure_details.push(format!(
                "two_sum[{index}] input={:08x}/{:08x} gpu={:08x}/{:08x} rust={:08x}/{:08x}",
                a.to_bits(),
                b.to_bits(),
                output.primary.hi.to_bits(),
                output.primary.lo.to_bits(),
                sum.hi.to_bits(),
                sum.lo.to_bits()
            ));
            mismatches += 1;
        }
        if check_product
            && (!bits_equal(output.product, product) || output.product.to_f64() != exact_product)
        {
            failure_details.push(format!(
                "two_prod_{:?}[{index}] input={:08x}/{:08x} gpu={:08x}/{:08x} rust={:08x}/{:08x}",
                variant,
                a.to_bits(),
                b.to_bits(),
                output.product.hi.to_bits(),
                output.product.lo.to_bits(),
                product.hi.to_bits(),
                product.lo.to_bits()
            ));
            mismatches += 1;
        }
    }
    if force_failure {
        mismatches += 1;
        failure_details.push(format!("forced {:?} candidate failure", variant));
    }
    let (backend, adapter) = adapter_fields(context);
    let shader_hash = crate::core::shader_registry::shader_hashes_snapshot()
        .get(pipeline.label())
        .cloned()
        .unwrap_or_default();
    Ok(DdSelftestReport {
        passed: mismatches == 0,
        backend,
        adapter,
        two_prod_variant: variant,
        shader_label: pipeline.label().to_string(),
        shader_hash,
        canary_count: 10,
        mismatch_count: mismatches,
        rejected_variants: Vec::new(),
        failure_details,
    })
}

fn initialize(context: &crate::core::gpu::GpuContext) -> Result<DdSelftestReport, String> {
    let mut rejected = Vec::new();
    let mut failure_details = Vec::new();
    let force_failure = std::env::var_os("FORGE3D_DD_FORCE_SELFTEST_FAIL").is_some();
    for variant in [TwoProdVariant::Fma, TwoProdVariant::Split] {
        match run_candidate(context, variant, force_failure) {
            Ok(mut report) if report.passed => {
                report.rejected_variants = rejected;
                return Ok(report);
            }
            Ok(report) => failure_details.extend(report.failure_details),
            Err(error) => failure_details.push(format!("{variant:?} candidate error: {error}")),
        }
        rejected.push(variant);
    }
    crate::core::degradation::record_degradation(
        "precision_selftest_failed",
        "double_float",
        "two_sum/two_prod exactness canary failed; DD execution refused",
    );
    Err(format!(
        "double-float exactness self-test failed on all product variants: {}",
        failure_details.join("; ")
    ))
}

pub(crate) fn initialize_for_context(context: &crate::core::gpu::GpuContext) -> RenderResult<()> {
    if crate::core::gpu::deterministic_mode() {
        let state = CAPABILITY.get_or_init(|| initialize(context));
        if let Err(reason) = state {
            return Err(RenderError::degraded_capability(reason));
        }
    }
    Ok(())
}

pub fn selftest() -> RenderResult<DdSelftestReport> {
    let context = crate::core::gpu::try_ctx()?;
    match CAPABILITY.get_or_init(|| initialize(context)) {
        Ok(report) => Ok(report.clone()),
        Err(reason) => Err(RenderError::degraded_capability(reason)),
    }
}

fn run_harness(op: DdOperation, generated: u64, adversarial: u64) -> RenderResult<DdHarnessReport> {
    let capability = selftest()?;
    let context = crate::core::gpu::try_ctx()?;
    let capture = crate::core::certificate::begin_render_capture("dupla.dd_harness");
    let pipeline = HarnessPipeline::new(&context.device, capability.two_prod_variant)?;
    let mut mismatches = 0;
    let mut maximum = 0.0_f64;
    for (phase, total) in [(0, adversarial), (1, generated)] {
        let mut offset = 0_u64;
        while offset < total {
            let count = (total - offset).min(CHUNK as u64) as u32;
            let outputs = pipeline.dispatch(
                &context.device,
                &context.queue,
                offset as u32,
                count,
                phase,
                op.code(),
                phase == 0 && offset == 0,
            )?;
            reduce_chunk(
                op,
                capability.two_prod_variant,
                phase,
                offset,
                &outputs,
                &mut mismatches,
                &mut maximum,
            );
            offset += count as u64;
        }
    }
    if mismatches != 0 || maximum > op.bound() {
        return Err(RenderError::render(format!(
            "DD {:?} proof failed: mismatches={mismatches}, max={maximum}u², bound={}u²",
            op,
            op.bound()
        )));
    }
    let evidence = crate::core::certificate::PrecisionEvidence {
        backend: capability.backend.clone(),
        adapter: capability.adapter.clone(),
        operation: op.as_str().to_string(),
        two_prod_variant: capability.two_prod_variant.as_str().to_string(),
        shader_label: pipeline.label().to_string(),
        shader_hash: capability.shader_hash.clone(),
        generated_count: generated,
        adversarial_count: adversarial,
        mismatch_count: mismatches,
        max_err_u2: maximum,
        cited_bound_u2: op.bound(),
    };
    crate::core::certificate::record_precision_evidence(evidence);
    capture.finish();
    let certificate_json = crate::core::certificate::execution_report_json()?;
    Ok(DdHarnessReport {
        operation: op,
        backend: capability.backend,
        adapter: capability.adapter,
        two_prod_variant: capability.two_prod_variant,
        shader_label: pipeline.label().to_string(),
        shader_hash: capability.shader_hash,
        generated_count: generated,
        adversarial_count: adversarial,
        mismatch_count: mismatches,
        max_err_u2: maximum,
        cited_bound_u2: op.bound(),
        certificate_json,
    })
}

pub fn harness(operation: &str, n: u64) -> RenderResult<DdHarnessReport> {
    if n < MIN_GENERATED_COUNT {
        return Err(RenderError::render(format!(
            "DD harness requires at least {MIN_GENERATED_COUNT} generated vectors"
        )));
    }
    if n > u32::MAX as u64 {
        return Err(RenderError::render(format!(
            "DD harness vector count must not exceed {}",
            u32::MAX
        )));
    }
    run_harness(DdOperation::parse(operation)?, n, ADVERSARIAL_COUNT)
}

#[cfg(test)]
pub(crate) fn harness_for_test(
    operation: DdOperation,
    n: u64,
    adversarial: u64,
) -> RenderResult<DdHarnessReport> {
    run_harness(operation, n, adversarial)
}

#[cfg(test)]
pub(crate) fn harness_window_for_test(
    operation: DdOperation,
    offset: u32,
    count: u32,
) -> RenderResult<(u64, f64)> {
    let capability = selftest()?;
    let context = crate::core::gpu::try_ctx()?;
    let pipeline = HarnessPipeline::new(&context.device, capability.two_prod_variant)?;
    let outputs = pipeline.dispatch(
        &context.device,
        &context.queue,
        offset,
        count,
        1,
        operation.code(),
        false,
    )?;
    let mut mismatches = 0;
    let mut maximum = 0.0;
    reduce_chunk(
        operation,
        capability.two_prod_variant,
        1,
        offset as u64,
        &outputs,
        &mut mismatches,
        &mut maximum,
    );
    Ok((mismatches, maximum))
}
