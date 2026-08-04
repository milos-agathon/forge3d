from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_STEP = re.compile(
    r"(?ms)^      - (?:name|uses):.*?"
    r"(?=^      - (?:name|uses):|^  [A-Za-z0-9_-]+:|\Z)"
)


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _workflow_data(name: str) -> dict:
    data = yaml.load(_workflow(name), Loader=yaml.BaseLoader)
    assert isinstance(data, dict), f"{name} is not a workflow mapping"
    assert isinstance(data.get("jobs"), dict), f"{name} has no jobs mapping"
    return data


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:.*?(?=^  [A-Za-z0-9_-]+:|\Z)",
        workflow,
    )
    assert match, f"missing workflow job: {name}"
    return match.group()


def _artifact_step(workflow: str, artifact: str) -> str:
    for step in _upload_steps(workflow):
        if re.search(rf"(?m)^\s+name: {re.escape(artifact)}$", step):
            return step
    raise AssertionError(f"missing artifact upload: {artifact}")


def _upload_steps(workflow: str) -> list[str]:
    return [
        step
        for step in UPLOAD_STEP.findall(workflow)
        if "uses: actions/upload-artifact@v4" in step
    ]


def test_ci_cost_controls_are_scoped_and_retained() -> None:
    workflow = _workflow("ci.yml")

    concurrency = re.search(r"(?ms)^concurrency:\n.*?(?=^env:)", workflow)
    assert concurrency
    body = concurrency.group()
    assert "format('pr-{0}', github.event.pull_request.number)" in body
    assert "format('manual-{0}', github.ref_name)" in body
    assert "format('run-{0}', github.run_id)" in body
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request'" in body

    trigger = workflow.split("\npermissions:", 1)[0]
    assert (
        "types: [opened, synchronize, reopened, ready_for_review, labeled, unlabeled]"
        in trigger
    )
    for scope in (
        "core",
        "full",
        "determinism",
        "m06",
        "f3dz",
        "anamnesis",
        "tessella",
    ):
        assert f"          - {scope}" in trigger

    preflight = _job(workflow, "preflight")
    assert "name: Checkout evaluated workflow state" in preflight
    assert "ref: ${{ github.sha }}" in preflight
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in preflight
    assert "base-added policy tests are available" in preflight
    assert "tests/test_ci_cost_controls.py" in preflight
    assert "tests/test_ci_lfs_fanout.py" in preflight
    assert "tests/test_determinism_matrix.py" in preflight
    for job_name in (
        "prepare-lfs-fixtures",
        "terrain-golden-paths",
        "test-rust",
        "build-wheel-windows",
        "build-wheel-linux",
        "build-wheel-macos",
        "build-wheel-linux-arm",
        "build-docs",
    ):
        assert "needs: preflight" in _job(workflow, job_name)

    for platform in ("windows", "linux", "macos", "linux-arm"):
        job = _job(workflow, f"build-wheel-{platform}")
        assert "uses: ./.github/workflows/build-wheel.yml" in job
        assert f"artifact: wheels-{platform}" in job

    core = _job(workflow, "test-python-core")
    assert "runner: ubuntu-latest" in core
    assert "python_versions: '[\"3.11\"]'" in core
    assert "test_mode: full" in core
    for job_name in (
        "test-python-compat-linux",
        "test-python-compat-windows",
        "test-python-compat-macos",
    ):
        assert "test_mode: compatibility" in _job(workflow, job_name)
    assert "python_versions: '[\"3.10\", \"3.13\"]'" in _job(
        workflow, "test-python-compat-linux"
    )
    for job_name in (
        "test-python-full-linux",
        "test-python-full-windows",
        "test-python-full-macos",
    ):
        job = _job(workflow, job_name)
        assert "github.event_name == 'schedule'" in job
        assert "inputs.scope == 'full'" in job
        assert "python_versions: '[\"3.10\", \"3.11\", \"3.12\", \"3.13\"]'" in job

    assert "retention-days: 1" in _artifact_step(
        _job(workflow, "prepare-lfs-fixtures"), "lfs-fixture-bundles"
    )
    wheel_workflow = _workflow("build-wheel.yml")
    assert "retention-days: 1" in _artifact_step(
        _job(wheel_workflow, "build"), "${{ inputs.artifact }}"
    )
    for job, artifact in (
        ("test-terminus-fuzz", "terminus-fuzzer-evidence"),
        ("test-golden-images", "golden-lane-marker"),
        ("test-golden-images", "visual-golden-diffs"),
    ):
        assert "retention-days: 7" in _artifact_step(_job(workflow, job), artifact)

    for job, artifact in (
        ("test-m06-full-geospatial-viewer", "m06-full-geospatial-viewer-evidence"),
        ("test-f3dz-gpu", "f3dz-physical-gpu-evidence"),
        ("test-anamnesis-portability-seed", "anamnesis-physical-portable-store"),
        ("test-anamnesis-portability", "anamnesis-portability-evidence"),
        ("test-anamnesis-production", "anamnesis-production-evidence"),
    ):
        assert "retention-days: 90" in _artifact_step(_job(workflow, job), artifact)

    assert "sphinx" not in workflow.lower()
    assert "name: documentation" not in workflow
    assert "  build-docs:" in workflow
    assert "cargo doc --workspace" in _job(workflow, "build-docs")

    assert "refresh-recipe-certificates" not in workflow
    certificate = _workflow("certificate-refresh.yml")
    certificate_trigger = certificate.split("\npermissions:", 1)[0]
    assert "  workflow_dispatch:" in certificate_trigger
    assert "  push:" not in certificate_trigger
    assert "  pull_request:" not in certificate_trigger
    assert certificate.count("uses: ./.github/workflows/build-wheel.yml") == 1
    assert "artifact: certificate-refresh-wheel-windows" in certificate
    assert "--require-nvidia-vulkan" in certificate
    assert "ref: ${{ github.sha }}" in certificate
    assert "github.ref == 'refs/heads/main'" in certificate
    assert "github.ref_protected" in certificate
    assert "environment: production-signing" in certificate
    assert "group: certificate-refresh-production" in certificate
    assert "retention-days: 90" in _artifact_step(
        _job(certificate, "refresh"), "refreshed-recipe-certificates"
    )


def test_publish_cost_controls_are_tag_only_and_retention_aware() -> None:
    workflow = _workflow("publish.yml")
    trigger = workflow.split("\npermissions:", 1)[0]
    assert "  pull_request:" not in trigger
    assert re.search(r"(?m)^  push:\n    tags:\n      - 'v\*'$", trigger)
    assert "  workflow_dispatch:" in trigger

    concurrency = re.search(r"(?ms)^concurrency:\n.*?(?=^jobs:)", workflow)
    assert concurrency
    assert "group: ${{ github.workflow }}-${{ github.ref }}" in concurrency.group()
    assert "cancel-in-progress: false" in concurrency.group()
    assert "queue: max" in concurrency.group()

    publish = _job(workflow, "publish")
    condition = re.search(r"(?m)^    if: (.+)$", publish)
    assert condition
    assert "startsWith(github.ref, 'refs/tags/v')" in condition.group(1)
    assert "github.event_name == 'push'" in condition.group(1)
    assert "github.event_name == 'workflow_dispatch'" in condition.group(1)
    assert "github.event.inputs.dry_run == 'false'" in condition.group(1)

    expected = "retention-days: ${{ startsWith(github.ref, 'refs/tags/v') && 90 || 1 }}"
    uploads = _upload_steps(workflow)
    assert uploads
    assert all(expected in upload for upload in uploads)


def test_determinism_cost_controls_keep_pr_cancellation_and_evidence() -> None:
    workflow = _workflow("determinism-matrix.yml")
    trigger = workflow.split("\npermissions:", 1)[0]
    assert "  workflow_call:" in trigger
    assert "  workflow_dispatch:" not in trigger
    assert "  push:" not in trigger
    assert "  pull_request:" not in trigger
    assert "maturin develop" not in workflow
    assert "PyO3/maturin-action" not in workflow
    assert "actions/download-artifact@v4" in workflow
    for family in ("run_render", "run_f3dz", "run_anamnesis"):
        assert family in trigger

    uploads = _upload_steps(workflow)
    assert uploads
    assert all("retention-days: 7" in upload for upload in uploads)
    for artifact in (
        "f3dz-determinism-${{ matrix.os }}",
        "determinism-hash-${{ matrix.leg }}",
        "determinism-hash-wasm",
        "anamnesis-store-linux",
        "anamnesis-hosted-consumer-evidence",
    ):
        assert "retention-days: 7" in _artifact_step(workflow, artifact)


def test_all_workflows_parse_and_local_reusable_calls_are_well_formed() -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    parsed = {path.name: _workflow_data(path.name) for path in workflow_dir.glob("*.yml")}
    assert parsed

    allowed_call_keys = {
        "name",
        "uses",
        "with",
        "secrets",
        "strategy",
        "needs",
        "if",
        "permissions",
        "concurrency",
    }
    for caller_name, workflow in parsed.items():
        for job_name, job in workflow["jobs"].items():
            if not isinstance(job, dict):
                continue
            uses = job.get("uses", "")
            if not uses.startswith("./.github/workflows/"):
                continue
            assert set(job) <= allowed_call_keys, (
                f"{caller_name}:{job_name} uses unsupported keys for a reusable call: "
                f"{sorted(set(job) - allowed_call_keys)}"
            )
            target_name = Path(uses).name
            assert target_name in parsed, f"{caller_name}:{job_name} calls missing {uses}"
            target_on = parsed[target_name].get("on", {})
            assert isinstance(target_on, dict) and "workflow_call" in target_on, (
                f"{caller_name}:{job_name} target {target_name} is not reusable"
            )
            call_contract = target_on["workflow_call"] or {}
            declared_inputs = call_contract.get("inputs", {})
            provided_inputs = job.get("with", {}) or {}
            assert set(provided_inputs) <= set(declared_inputs), (
                f"{caller_name}:{job_name} passes unknown inputs to {target_name}: "
                f"{sorted(set(provided_inputs) - set(declared_inputs))}"
            )
            required_inputs = {
                name
                for name, contract in declared_inputs.items()
                if str(contract.get("required", "false")).casefold() == "true"
            }
            assert required_inputs <= set(provided_inputs), (
                f"{caller_name}:{job_name} omits required {target_name} inputs: "
                f"{sorted(required_inputs - set(provided_inputs))}"
            )
            for input_name, value in provided_inputs.items():
                input_type = declared_inputs[input_name].get("type")
                if input_type == "boolean" and not str(value).startswith("${{"):
                    assert str(value).casefold() in {"true", "false"}, (
                        f"{caller_name}:{job_name} passes non-boolean {input_name}={value}"
                    )


def test_tessella_acceptance_is_explicitly_scoped() -> None:
    workflow = _workflow("ci.yml")
    paths = _job(workflow, "terrain-golden-paths")
    assert "tessella: ${{ steps.filter.outputs.tessella }}" in paths
    assert "            tessella:" in paths

    for path in (
        "src/terrain/renderer/virtual_texture.rs",
        "src/terrain/culling/two_phase.rs",
        "src/core/feedback_buffer.rs",
        "src/core/screen_space_effects/hzb.rs",
        "src/shaders/hzb_build.wgsl",
        "src/shaders/hzb_cull.wgsl",
        "scripts/tessella_evidence_contract.py",
        "scripts/tessella_evidence_provenance.py",
        "scripts/tessella_evidence_report.py",
        "scripts/tessella_evidence_thresholds.py",
        "tests/test_terrain_vt_pbr_families.py",
        "tests/test_tv20_virtual_texturing.py",
        "tests/test_vt_out_of_core.py",
        "tests/test_hzb_culling.py",
        "tests/test_visibility_buffer.py",
        "tests/test_bc_encoders.py",
        "tests/test_flythrough_popping.py",
        "tests/test_vt_request_retention.py",
        "tests/test_tessella_certificate_evidence.py",
        "tests/test_tessella_evidence_report.py",
    ):
        assert f"              - '{path}'" in paths

    # Lock the explicit lane and its aggregate wiring so TESSELLA cannot appear
    # under a generic name or escape the physical-runner selection policy.
    jobs = _workflow_data("ci.yml")["jobs"]
    tessella_jobs = []
    tessella_tokens = (
        "test_tv20_virtual_texturing.py",
        "test_terrain_vt_pbr_families.py",
        "hzb_build.wgsl",
        "virtual_texture.rs",
        "feedback_buffer.rs",
    )
    for name, body in jobs.items():
        if name == "terrain-golden-paths":
            continue
        serialized = yaml.safe_dump(body).casefold()
        physical_domain_job = "self-hosted" in serialized and any(
            token in serialized for token in tessella_tokens
        )
        if (
            "tessella" in name.casefold()
            or "tessella" in serialized
            or physical_domain_job
        ):
            tessella_jobs.append(name)
    assert tessella_jobs == ["test-tessella-gpu", "full-acceptance-summary"]

    lane = jobs["test-tessella-gpu"]
    condition = " ".join(lane["if"].split())
    assert condition == " ".join(
        """
        github.event_name == 'schedule' ||
        (github.event_name == 'workflow_dispatch' &&
         (inputs.scope == 'full' || inputs.scope == 'tessella')) ||
        (github.event_name == 'pull_request' &&
         github.event.pull_request.head.repo.full_name == github.repository &&
         contains(github.event.pull_request.labels.*.name, 'run-physical') &&
         needs.terrain-golden-paths.outputs.tessella == 'true')
        """.split()
    )
    assert lane["needs"] == ["build-wheel-windows", "terrain-golden-paths"]
    assert lane["env"]["FORGE3D_TESSELLA_REQUIRED_GPU"] == "1"
    assert "test-tessella-gpu" in jobs["full-acceptance-summary"]["needs"]


def test_physical_selection_truth_table_and_job_conditions() -> None:
    jobs = _workflow_data("ci.yml")["jobs"]

    def normalize(value: str) -> str:
        return " ".join(value.split())

    def expected_condition(scope: str, output: str) -> str:
        return normalize(
            f"""
            github.event_name == 'schedule' ||
            (github.event_name == 'workflow_dispatch' &&
             (inputs.scope == 'full' || inputs.scope == '{scope}')) ||
            (github.event_name == 'pull_request' &&
             github.event.pull_request.head.repo.full_name == github.repository &&
             contains(github.event.pull_request.labels.*.name, 'run-physical') &&
             needs.terrain-golden-paths.outputs.{output} == 'true')
            """
        )

    expected = {
        "test-m06-full-geospatial-viewer": expected_condition("m06", "m06"),
        "test-f3dz-gpu": expected_condition("f3dz", "f3dz"),
        "test-anamnesis-portability-seed": expected_condition(
            "anamnesis", "anamnesis"
        ),
        "test-anamnesis-portability": expected_condition("anamnesis", "anamnesis"),
        "test-anamnesis-production": expected_condition("anamnesis", "anamnesis"),
        "test-tessella-gpu": expected_condition("tessella", "tessella"),
    }
    for job_name, condition in expected.items():
        assert normalize(jobs[job_name]["if"]) == condition

    def selected(
        event: str,
        *,
        scope: str = "core",
        internal: bool = False,
        labeled: bool = False,
        path_selected: bool = False,
        family: str = "m06",
    ) -> bool:
        return event == "schedule" or (
            event == "workflow_dispatch" and scope in {"full", family}
        ) or (
            event == "pull_request" and internal and labeled and path_selected
        )

    cases = (
        ({"event": "push", "path_selected": True}, False),
        ({"event": "pull_request", "path_selected": True}, False),
        (
            {
                "event": "pull_request",
                "internal": True,
                "path_selected": True,
            },
            False,
        ),
        (
            {
                "event": "pull_request",
                "internal": True,
                "labeled": True,
                "path_selected": False,
            },
            False,
        ),
        (
            {
                "event": "pull_request",
                "internal": True,
                "labeled": True,
                "path_selected": True,
            },
            True,
        ),
        ({"event": "schedule"}, True),
        ({"event": "workflow_dispatch", "scope": "core"}, False),
        ({"event": "workflow_dispatch", "scope": "full"}, True),
        ({"event": "workflow_dispatch", "scope": "m06"}, True),
        ({"event": "workflow_dispatch", "scope": "f3dz"}, False),
    )
    for arguments, wanted in cases:
        assert selected(**arguments) is wanted
