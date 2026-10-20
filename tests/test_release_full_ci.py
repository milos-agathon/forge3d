"""Release admission must prove full scope on the exact candidate SHA."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import require_full_ci as gate

SHA = "a" * 40
ARGS = ["--repository", "milos-agathon/forge3d", "--sha", SHA,
        "--ref", "v1.42.0", "--event", "push"]


def run(event="schedule", **changes):
    return {"id": 12, "head_sha": SHA, "event": event, "status": "completed",
            "conclusion": "failure", "html_url": "https://github.com/example/actions/runs/12",
            **changes}


def summary(scope="full", **changes):
    return {"name": gate.SUMMARY_JOB, "head_sha": SHA, "status": "completed",
            "conclusion": "success", "steps": [{"name": f"Record acceptance scope: {scope}",
            "status": "completed", "conclusion": "success"}], **changes}


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_gate_accepts_full_summary_even_with_failed_optional_job(monkeypatch, event):
    calls = []

    def api(endpoint, key):
        calls.append(endpoint)
        return [run(event)] if key == "workflow_runs" else [summary()]

    monkeypatch.setattr(gate, "api_items", api)
    assert gate.find_green_full_run("milos-agathon/forge3d", SHA)["id"] == 12
    assert "workflows/ci.yml/runs?head_sha=" + SHA + "&status=completed" in calls[0]
    assert calls[1].endswith("/actions/runs/12/jobs?filter=latest")


@pytest.mark.parametrize("candidate,job", [
    (run(head_sha="b" * 40), summary()),
    (run(event="push"), summary()),
    (run(event="pull_request"), summary()),
    (run(status="in_progress"), summary()),
    (run(), summary(head_sha="b" * 40)),
    (run(), summary(name="PR Core Success")),
    (run(), summary(status="queued")),
    (run(), summary(conclusion="failure")),
    (run(), summary(conclusion="skipped")),
    (run("workflow_dispatch"), summary(scope="m06")),
    (run("workflow_dispatch"), summary(scope="determinism")),
    (run("workflow_dispatch"), summary(steps=[])),
    (run("workflow_dispatch"), summary(steps=[{"name": gate.FULL_SCOPE_STEP,
                                              "status": "completed", "conclusion": "skipped"}])),
])
def test_gate_rejects_incomplete_or_wrong_scope_evidence(monkeypatch, candidate, job):
    monkeypatch.setattr(gate, "api_items", lambda endpoint, key:
                        [candidate] if key == "workflow_runs" else [job])
    assert gate.find_green_full_run("milos-agathon/forge3d", SHA) is None


def test_gate_continues_to_eligible_run_after_bad_evidence(monkeypatch):
    def api(endpoint, key):
        if key == "workflow_runs":
            return [run(id=11), run()]
        return [summary(conclusion="failure")] if "/11/" in endpoint else [summary()]

    monkeypatch.setattr(gate, "api_items", api)
    assert gate.find_green_full_run("milos-agathon/forge3d", SHA)["id"] == 12


def test_api_collects_every_page(monkeypatch):
    def execute(command, **kwargs):
        assert command == ["gh", "api", "endpoint", "--paginate", "--slurp"]
        assert kwargs == {"text": True, "capture_output": True, "check": True}
        return SimpleNamespace(stdout=json.dumps([{"jobs": []}, {"jobs": [summary()]}]))

    monkeypatch.setattr(gate.subprocess, "run", execute)
    assert gate.api_items("endpoint", "jobs") == [summary()]


def test_missing_full_run_fails_with_exact_dispatch_command(monkeypatch, capsys):
    monkeypatch.setattr(gate, "find_green_full_run", lambda repository, sha: None)
    assert gate.main(ARGS) == 1
    output = capsys.readouterr().out
    assert f"no completed green full ci.yml run for {SHA}" in output
    assert "gh workflow run ci.yml -f scope=full --ref v1.42.0" in output


def test_api_error_fails_closed_without_exposing_api_output(monkeypatch, capsys):
    def fail(repository, sha):
        raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="private diagnostic")

    monkeypatch.setattr(gate, "find_green_full_run", fail)
    assert gate.main(ARGS) == 1
    output = capsys.readouterr().out
    assert "Cannot verify" in output and "private diagnostic" not in output
    assert "gh workflow run ci.yml -f scope=full --ref v1.42.0" in output


def test_only_explicit_dispatch_dry_run_bypasses_api(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(gate, "find_green_full_run", lambda *args: calls.append(args))
    dispatch_args = ARGS[:-1] + ["workflow_dispatch", "--dry-run", "true"]
    assert gate.main(dispatch_args) == 0
    assert calls == []
    assert "no full CI claim" in capsys.readouterr().out
    assert gate.main(ARGS + ["--dry-run", "true"]) == 1
    assert len(calls) == 1
    assert gate.main(ARGS[:-1] + ["workflow_dispatch", "--dry-run", "false"]) == 1
    assert len(calls) == 2
