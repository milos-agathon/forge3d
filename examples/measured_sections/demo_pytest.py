"""Optional real-data runner using the repository's existing Pro test fixture."""

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
source = importlib.util.spec_from_file_location("_c08_repo_test_fixtures", REPO / "tests/conftest.py")
repo_tests = importlib.util.module_from_spec(source)
source.loader.exec_module(repo_tests)
pro_license = repo_tests.pro_license

def test_monterey_demo(monkeypatch, pro_license):
    """Exercise the documented CLI with the existing Pro fixture on a prepared subset."""
    demo_root = os.environ.get("FORGE3D_MEASURED_SECTIONS_DEMO")
    if demo_root is None:
        raise ValueError("Set FORGE3D_MEASURED_SECTIONS_DEMO to a prepared directory")
    variant = os.environ["FORGE3D_MEASURED_SECTIONS_RECIPE"]
    examples = Path(__file__).resolve().parent
    output = Path(demo_root).resolve()
    monkeypatch.syspath_prepend(str(examples))
    from monterey import main

    monkeypatch.setattr(sys, "argv", ["monterey.py", "render", str(output / "subset"),
                                     str(examples / f"{variant}.json"), str(output / variant)])
    main()
