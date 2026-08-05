# tools/europe_smoke/probe.py
"""ADS pre-flight (§0.4) and the gate-16 CLI.

The constraints endpoint is unauthenticated and cheap. Running it immediately
before every fetch is mandatory: constraints.json is an ADS build-time
snapshot, so a cached copy silently describes yesterday's availability.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from typing import Callable
from urllib.request import Request, urlopen

from . import config, provenance

CACHE_CONSTRAINTS = False  # see module docstring; never make this True


def constraints(payload: dict) -> dict:
    """POST an inputs payload to the ADS constraints endpoint."""
    body = json.dumps({"inputs": payload}).encode("utf-8")
    req = Request(
        config.ADS_CONSTRAINTS_URL,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": config.USER_AGENT},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _latest_date(reply: dict) -> dt.date:
    """Largest end-date across every range the reply lists."""
    latest = None
    for spec in reply.get("date", []):
        end = spec.split("/")[-1].strip()
        d = dt.date.fromisoformat(end)
        if latest is None or d > latest:
            latest = d
    if latest is None:
        raise RuntimeError(f"constraints reply carried no date ranges: {reply}")
    return latest


@dataclass(frozen=True)
class Axis:
    d_an: dt.date            # latest analysis date available
    d_prev: dt.date          # d_an - 1 day
    start: dt.date           # first analysis date requested
    h_win: list[str]         # analysis hours on start..d_prev
    h_new: list[str]         # analysis hours on d_an
    t_now: dt.datetime       # NOW = latest analysis valid time delivered
    run_date: dt.date        # forecast run date
    run_hour: str            # "00:00" or "12:00"
    t_init: dt.datetime      # forecast initialisation

    def requested_analysis_times(self) -> list[dt.datetime]:
        out: list[dt.datetime] = []
        day = self.start
        while day <= self.d_prev:
            for h in self.h_win:
                hh, mm = h.split(":")
                out.append(dt.datetime.combine(day, dt.time(int(hh), int(mm))))
            day += dt.timedelta(days=1)
        for h in self.h_new:
            hh, mm = h.split(":")
            out.append(dt.datetime.combine(self.d_an, dt.time(int(hh), int(mm))))
        return sorted(out)


def resolve_axis(
    start_date: dt.date,
    today: dt.date,
    constraints: Callable[[dict], dict] = constraints,
) -> Axis:
    """Derive the requested axis from live availability (§0.4 steps 1-5)."""
    aer = list(config.AEROSOL_VARS)
    wind = list(config.WIND_VARS)

    d_an = _latest_date(constraints({"variable": aer, "type": ["analysis"]}))
    d_prev = d_an - dt.timedelta(days=1)

    h_new = sorted(constraints(
        {"variable": aer, "type": ["analysis"], "date": [f"{d_an}/{d_an}"]}
    )["time"])
    h_win = sorted(constraints(
        {"variable": aer, "type": ["analysis"], "date": [f"{start_date}/{d_prev}"]}
    )["time"])

    # step 4: wind must cover the same hours, else intersect and record it
    w_new = sorted(constraints(
        {"variable": wind, "type": ["analysis"], "date": [f"{d_an}/{d_an}"]}
    )["time"])
    w_win = sorted(constraints(
        {"variable": wind, "type": ["analysis"], "date": [f"{start_date}/{d_prev}"]}
    )["time"])
    h_new = sorted(set(h_new) & set(w_new)) or h_new
    h_win = sorted(set(h_win) & set(w_win)) or h_win

    hh, mm = max(h_new).split(":")
    t_now = dt.datetime.combine(d_an, dt.time(int(hh), int(mm)))

    # step 5: newest forecast run still listing all five variables
    run_date, run_hour = None, None
    for back in range(3):
        d = d_an - dt.timedelta(days=back)
        for candidate in ("12:00", "00:00"):
            reply = constraints({
                "variable": aer + wind, "type": ["forecast"],
                "date": [f"{d}/{d}"],
            })
            if candidate in reply.get("time", []):
                run_date, run_hour = d, candidate
                break
        if run_date:
            break
    if run_date is None:
        raise RuntimeError("no forecast run carries all five variables in the last 3 days")

    hh, mm = run_hour.split(":")
    t_init = dt.datetime.combine(run_date, dt.time(int(hh), int(mm)))
    return Axis(d_an, d_prev, start_date, h_win, h_new, t_now, run_date, run_hour, t_init)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify artifacts and credentials, then exit (gate 16)")
    ap.add_argument("--strict", action="store_true",
                    help="with --check: also fail on warnings (unpinned, pending). "
                         "The release gate.")
    ap.add_argument("--write-pins", action="store_true",
                    help="install observed hashes for unpinned artifacts into "
                         "manifest.toml. Operator-invoked; never runs automatically; "
                         "refuses while anything is blocking and refuses to overwrite "
                         "an existing pin. Review the diff.")
    ap.add_argument("--days", type=int, default=10, help="analysis window length")
    args = ap.parse_args(argv)

    ok, findings = provenance.check_all(strict=args.strict)
    for line in provenance.format_findings(findings):
        print(line)

    pins = provenance.propose_pins(findings)
    if pins:
        out = provenance.write_proposed_pins(pins)
        print(f"\n{len(pins)} artifact(s) unpinned; observed hashes -> {out}")
        print("Verify each against its upstream, then rerun with --write-pins.")

    if args.write_pins:
        # Key the refusal off STATUS, not level. --strict promotes `unpinned`
        # to fail, and refusing there would block the one command that clears
        # it. A genuinely blocking status (missing/size/mismatch/absent) means
        # this machine's artifacts are not trustworthy, so nothing is pinned:
        # otherwise `--write-pins` would install a fresh hash and exit 0 while
        # a required artifact sat substituted, which is the same decorative
        # gate this whole change exists to remove.
        blocking = [f for f in findings
                    if provenance.level_of(f.status) == provenance.FAIL]
        if blocking:
            print(f"\nrefusing to pin while {[f.name for f in blocking]} "
                  f"{'is' if len(blocking) == 1 else 'are'} blocking; "
                  f"fix the artifact, do not pin around it")
            return 1
        if not pins:
            print("nothing to pin.")
            return 0
        written = provenance.apply_pins(pins)
        print(f"pinned {written} in {provenance.MANIFEST_PATH}")
        print("Review the diff before committing: git diff -- "
              "tools/europe_smoke/manifest.toml")
        return 0

    if args.check:
        summary = provenance.summarise(findings)
        print("\nprovenance:", "PASS" if ok else "FAIL",
              f"({summary['counts']['ok']} ok, {summary['counts']['warn']} warn, "
              f"{summary['counts']['fail']} fail)")
        return 0 if ok else 1
    if not ok:
        print("\nprovenance FAILED; refusing to probe", file=sys.stderr)
        return 1

    today = dt.datetime.now(dt.UTC).date()
    axis = resolve_axis(today - dt.timedelta(days=args.days), today)
    print(json.dumps({
        "d_an": str(axis.d_an), "h_win": axis.h_win, "h_new": axis.h_new,
        "t_now": axis.t_now.isoformat(), "t_init": axis.t_init.isoformat(),
        "run": f"{axis.run_date} {axis.run_hour}",
        "n_analysis_steps": len(axis.requested_analysis_times()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
