"""Require exact-SHA full CI evidence before building release artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
from urllib.parse import urlencode

SUMMARY_JOB = "Full Acceptance Summary"
FULL_SCOPE_STEP = "Record acceptance scope: full"


def api_items(endpoint: str, key: str) -> list[dict]:
    result = subprocess.run(
        ["gh", "api", endpoint, "--paginate", "--slurp"],
        text=True, capture_output=True, check=True,
    )
    pages = json.loads(result.stdout)
    return [item for page in pages for item in page[key]]


def find_green_full_run(repository: str, sha: str) -> dict | None:
    query = urlencode({"head_sha": sha, "status": "completed"})
    runs = api_items(
        f"repos/{repository}/actions/workflows/ci.yml/runs?{query}", "workflow_runs"
    )
    for run in runs:
        if (run.get("head_sha") != sha or run.get("status") != "completed"
                or run.get("event") not in {"schedule", "workflow_dispatch"}):
            continue
        jobs = api_items(
            f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=latest", "jobs"
        )
        for job in jobs:
            if (job.get("name") != SUMMARY_JOB or job.get("status") != "completed"
                    or job.get("conclusion") != "success" or job.get("head_sha") != sha):
                continue
            # Dispatch inputs are not exposed by the workflow-runs API. The
            # summary's executed step name records the selected input instead.
            if run["event"] == "workflow_dispatch" and not any(
                step.get("name") == FULL_SCOPE_STEP
                and step.get("status") == "completed"
                and step.get("conclusion") == "success"
                for step in job.get("steps", [])
            ):
                continue
            return run
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--dry-run", default="false")
    args = parser.parse_args(argv)
    if args.event == "workflow_dispatch" and args.dry_run == "true":
        print("Release gate bypassed for an explicit dry run; no full CI claim is made.")
        return 0
    try:
        run = find_green_full_run(args.repository, args.sha)
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError):
        print("::error::Cannot verify full CI evidence through the Actions API.")
        run = None
    if run is None:
        print(f"::error::Release refused: no completed green full ci.yml run for {args.sha}.")
        print("Run full acceptance on this ref, wait for Full Acceptance Summary to succeed, then retry:")
        print(f"gh workflow run ci.yml -f scope=full --ref {args.ref}")
        return 1
    print(f"Release gate passed: {run['html_url']} (exact SHA {args.sha}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
