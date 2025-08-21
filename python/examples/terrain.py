# A2-BEGIN:terrain-example
from __future__ import annotations
import os

try:
    from forge3d import TerrainSpike  # available only with --features terrain_spike
except Exception as e:
    raise SystemExit("TerrainSpike not available. Rebuild with --features terrain_spike") from e

def main():
    out = os.path.abspath("terrain_spike.png")
    r = TerrainSpike(800, 600, 160)
    r.render_png(out)
    print("Wrote", out)

if __name__ == "__main__":
    main()
# A2-END:terrain-example
