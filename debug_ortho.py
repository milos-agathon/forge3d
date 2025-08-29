import forge3d as f3d
import numpy as np

print("Testing orthographic projection matrices...")

# Test WGPU matrix
proj_wgpu = f3d.camera_orthographic(-1, 1, -1, 1, 0.1, 10.0, 'wgpu')
print("\nWGPU ortho matrix:")
print(proj_wgpu)
print("Bottom row:", proj_wgpu[3,:])

# Test GL matrix  
proj_gl = f3d.camera_orthographic(-1, 1, -1, 1, 0.1, 10.0, 'gl')
print("\nGL ortho matrix:")
print(proj_gl)
print("Bottom row:", proj_gl[3,:])

# Test a point transformation
point = np.array([0.0, 0.0, -5.0, 1.0])
print("\nTesting point:", point)

ndc_wgpu = proj_wgpu @ point
ndc_wgpu = ndc_wgpu / ndc_wgpu[3] if ndc_wgpu[3] != 0 else ndc_wgpu
print("WGPU NDC:", ndc_wgpu)

ndc_gl = proj_gl @ point  
ndc_gl = ndc_gl / ndc_gl[3] if ndc_gl[3] != 0 else ndc_gl
print("GL NDC:", ndc_gl)

# Test the pixel alignment case
print("\nTesting pixel alignment (800x600):")
proj_pixel = f3d.camera_orthographic(0, 800, 0, 600, 0.1, 10.0, 'wgpu')
print("Pixel alignment matrix:")
print(proj_pixel)

# Test (0.5, height-0.5) = (0.5, 599.5)
pixel_center = np.array([0.5, 599.5, -1.0, 1.0])
ndc_pixel = proj_pixel @ pixel_center
ndc_pixel = ndc_pixel / ndc_pixel[3] if ndc_pixel[3] != 0 else ndc_pixel
print("Pixel center (0.5, 599.5) -> NDC:", ndc_pixel)
print("Expected NDC X ≈ -1.0, got:", ndc_pixel[0])
print("Expected NDC Y ≈ +1.0, got:", ndc_pixel[1])