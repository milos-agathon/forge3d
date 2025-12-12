from PIL import Image
import numpy as np

p5 = np.array(Image.open('reports/p5/p5_terrain_csm.png'))
p5f = np.array(Image.open('reports/p5/p5_forced_colorspace.png'))
p6 = np.array(Image.open('reports/p6/p6_terrain_csm.png'))

d1 = np.abs(p5.astype(float) - p5f.astype(float))
d2 = np.abs(p5f.astype(float) - p6.astype(float))

print(f"P5 vs P5-forced: mean={d1.mean():.4f}, max={d1.max():.0f}, nonzero_pixels={np.count_nonzero(d1.sum(axis=2))}")
print(f"P5-forced vs P6: mean={d2.mean():.4f}, max={d2.max():.0f}, nonzero_pixels={np.count_nonzero(d2.sum(axis=2))}")
