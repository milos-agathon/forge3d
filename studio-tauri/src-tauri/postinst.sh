#!/bin/sh
# The deb never requires the engine; it only names the remedy. The shell
# probes `FORGE3D_STUDIO_PYTHON`, ancestors' .venv, then system python3.
# A first-run machine needs python3 + the engine (PyPI: forge3d).
printf '%s\n' "Forge3D Studio installed. It needs the forge3d Python package at runtime:"
printf '%s\n' "  pip install --user forge3d"
printf '%s\n' "or point the shell at your interpreter:"
printf '%s\n' "  export FORGE3D_STUDIO_PYTHON=/path/to/python"
printf '%s\n' "Verify with:  forge3d-studio-shell --engine-check"
exit 0
