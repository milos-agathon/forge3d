import numpy as np
from forge3d import Renderer

def count_covered(px: np.ndarray) -> int:
    # px: HxWx4 uint8; clear is white per A1.2
    return int(np.count_nonzero(np.any(px[:, :, :3] != 255, axis=2)))

if __name__ == "__main__":
    r = Renderer(512, 512)
    a = r.render_triangle_rgba()
    b = r.render_triangle_rgba()
    ca, cb = count_covered(a), count_covered(b)
    print("covered-pixels:", ca, cb)
    if ca != cb:
        raise SystemExit(f"NON-DETERMINISTIC EDGE COVERAGE: {ca} vs {cb}")
    print("OK: coverage stable")