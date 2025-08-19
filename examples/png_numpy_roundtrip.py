from pathlib import Path
import numpy as np
import vulkan_forge as vf

def main():
    out = Path("ex_roundtrip.png")
    rgb = (np.random.default_rng(0).integers(0, 255, size=(64,64,3), dtype=np.uint8)).copy(order="C")
    vf.numpy_to_png(out, rgb)
    back = vf.png_to_numpy(out)
    assert back.shape == (64,64,4)
    print("roundtrip OK:", out)

if __name__ == "__main__":
    main()