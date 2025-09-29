# examples/f18_gltf_import_demo.py
# Demonstrates importing the first primitive from a glTF 2.0 file and reporting counts.

from __future__ import annotations

import sys
from pathlib import Path

from forge3d.io import import_gltf, save_obj


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python examples/f18_gltf_import_demo.py <path-to-gltf-or-glb>")
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    mesh = import_gltf(str(path))
    print(f"Imported glTF: V={mesh.vertex_count} T={mesh.triangle_count}")
    try:
        out = path.with_suffix("")
        out = f"{out.name}_import.obj"
        save_obj(mesh, out)
        print(f"Saved: {out}")
    except Exception as e:
        print(f"OBJ export skipped/failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
