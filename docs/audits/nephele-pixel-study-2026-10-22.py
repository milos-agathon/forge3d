"""Run the corrected study; the reviewed legacy script is archived in evidence.

See nephele-correction-study-2026-10-22.py --help. Raw inputs are never rewritten;
--output must name a new directory. Both lighting ablations use the same source
estimator and geometric valid-air classification; camera T is held fixed.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("nephele-correction-study-2026-10-22.py")), run_name="__main__")
