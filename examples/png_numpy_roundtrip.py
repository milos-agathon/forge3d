from pathlib import Path
import numpy as np
import forge3d as f3d

def main():
    out = Path("ex_roundtrip.png")
    rgb = (np.random.default_rng(0).integers(0, 255, size=(64,64,3), dtype=np.uint8)).copy(order="C")
    f3d.numpy_to_png(out, rgb)
    back = f3d.png_to_numpy(out)
    assert back.shape == (64,64,4)
    print("roundtrip OK:", out)

if __name__ == "__main__":
    main()