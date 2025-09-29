# examples/f17_curves_demo.py
# Demonstrate curves and tubes generation along a 3D path.

import os
import tempfile

import numpy as np

import forge3d.io as fio
from forge3d.geometry import generate_ribbon, generate_tube


def make_helix(turns: int = 3, steps_per_turn: int = 64, radius: float = 1.0, pitch: float = 0.25) -> np.ndarray:
    t = np.linspace(0.0, turns * 2.0 * np.pi, turns * steps_per_turn, dtype=np.float32)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t / (2.0 * np.pi)
    return np.stack([x, y, z], axis=0).T


def main() -> None:
    path = make_helix(turns=4, steps_per_turn=96, radius=1.0, pitch=0.3)

    ribbon = generate_ribbon(path, width_start=0.2, width_end=0.05)
    tube = generate_tube(path, radius_start=0.15, radius_end=0.05, radial_segments=24, cap_ends=True)

    print("Ribbon:", ribbon.vertex_count, ribbon.triangle_count)
    print("Tube:", tube.vertex_count, tube.triangle_count)

    with tempfile.TemporaryDirectory() as td:
        ribbon_path = os.path.join(td, "helix_ribbon.obj")
        tube_path = os.path.join(td, "helix_tube.obj")
        fio.save_obj(ribbon, ribbon_path)
        fio.save_obj(tube, tube_path)
        print("Wrote:", ribbon_path)
        print("Wrote:", tube_path)


if __name__ == "__main__":
    main()
