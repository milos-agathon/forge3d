"""Run and reconcile the LITTERA exact-head installed-wheel shaping lane."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


HOSTS = {"linux": "Linux", "macos": "macOS", "windows": "Windows"}
HASH_TEST = "tests/test_text_no_hidden_deps.py::test_shaped_corpus_hash_is_locally_byte_identical"
HASH_LINE = re.compile(r"(?<!\w)shaped strings=([1-9]\d*) local SHA256=([0-9a-f]{64})$", re.MULTILINE)
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wheel_runtime(wheel):
    import forge3d
    from forge3d import _forge3d

    with ZipFile(wheel) as archive:
        for module in (forge3d, _forge3d):
            path = Path(module.__file__)
            actual = path.read_bytes()
            if actual != archive.read(f"forge3d/{path.name}"):
                raise ValueError(f"imported {module.__name__} differs from {wheel.name}")
    return digest(Path(_forge3d.__file__).read_bytes())


def extract_result(log, results):
    cases = list(ET.parse(results).iter("testcase"))
    matches = HASH_LINE.findall(log)
    if not cases or len(matches) != 1 or any(
        case.find(status) is not None for case in cases for status in ("failure", "error", "skipped")
    ):
        raise ValueError("expected passing shaping tests, zero skips and one shaping hash")
    if sum(case.get("name") == HASH_TEST.rsplit("::", 1)[1] for case in cases) != 1:
        raise ValueError("JUnit does not contain exactly one corpus hash test")
    count, sha256 = matches[0]
    if sum(case.get("name", "").startswith("test_shape_matches_harfbuzz[") for case in cases) != int(count):
        raise ValueError("the number of executed HarfBuzz cases differs from the serialized corpus")
    return int(count), sha256


def collect(wheels, output, identity):
    wheel, = wheels.glob("*.whl")
    native_sha256 = wheel_runtime(wheel)
    output.mkdir(parents=True, exist_ok=True)
    junit = output / "shaping.xml"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_shaping_conformance.py", HASH_TEST,
         "-q", "-s", f"--junitxml={junit}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    (output / "shaping.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="")
    result.check_returncode()
    count, sha256 = extract_result(result.stdout, junit)
    receipt = {
        **identity,
        "runner_os": os.environ["RUNNER_OS"],
        "wheel": wheel.name,
        "wheel_sha256": digest(wheel.read_bytes()),
        "native_sha256": native_sha256,
        "shaped_cases": count,
        "shaping_sha256": sha256,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


def compare(directory, identity):
    values = []
    for leg, host in HOSTS.items():
        artifacts = directory / f"littera-shaped-{leg}"
        receipt = json.loads((artifacts / "receipt.json").read_text(encoding="utf-8"))
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError(f"{leg}: receipt belongs to another revision or workflow run")
        if receipt["runner_os"] != host:
            raise ValueError(f"{leg}: expected {host}, got {receipt['runner_os']}")
        for key in ("wheel_sha256", "native_sha256", "shaping_sha256"):
            if not SHA256.fullmatch(receipt[key]):
                raise ValueError(f"{leg}: missing or invalid {key}")
        actual = extract_result(
            (artifacts / "shaping.log").read_text(encoding="utf-8"), artifacts / "shaping.xml"
        )
        if actual != (receipt["shaped_cases"], receipt["shaping_sha256"]):
            raise ValueError(f"{leg}: log or JUnit contradicts the shaping receipt")
        print(f"{host}: {actual[0]} cases, {actual[1]}, {receipt['wheel']}")
        values.append(actual)
    if len(set(values)) != 1:
        raise ValueError("hosted shaping hashes or case counts differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    capture = mode.add_parser("collect")
    capture.add_argument("wheels", type=Path)
    capture.add_argument("output", type=Path)
    reconcile = mode.add_parser("compare")
    reconcile.add_argument("directory", type=Path)
    args = parser.parse_args()

    revision = os.environ["LITTERA_EXACT_HEAD"]
    if os.environ.get("GITHUB_ACTIONS") != "true" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("requires GitHub Actions and a full pull request head SHA")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != revision:
        raise ValueError("checkout does not match the candidate head")
    identity = {
        "revision": revision,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
    }
    if args.mode == "collect":
        if any(os.environ.get(key) != "1" for key in
               ("FORGE3D_NO_BOOTSTRAP", "FORGE3D_TEST_INSTALLED_WHEEL")):
            raise ValueError("requires the installed-wheel test contract")
        collect(args.wheels, args.output, identity)
    else:
        compare(args.directory, identity)


if __name__ == "__main__":
    main()
