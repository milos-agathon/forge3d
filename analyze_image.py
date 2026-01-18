
import numpy as np
from PIL import Image
import sys

def analyze_image(path):
    try:
        img = Image.open(path)
        img_data = np.array(img)
        
        # Check simple statistics
        mean_val = np.mean(img_data)
        std_val = np.std(img_data)
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        
        print(f"Image: {path}")
        print(f"Dimensions: {img.size}")
        print(f"Mean brightness: {mean_val:.2f} (0-255)")
        print(f"Contrast (std dev): {std_val:.2f}")
        print(f"Range: {min_val} - {max_val}")
        
        # Check center region vs edges (vignette check)
        h, w = img_data.shape[:2]
        center = img_data[h//2-50:h//2+50, w//2-50:w//2+50]
        print(f"Center mean: {np.mean(center):.2f}")
        
        # Check for "washed out" (high brightness, low contrast)
        if mean_val > 200 and std_val < 30:
            print("STATUS: POTENTIALLY WASHED OUT (High brightness, low contrast)")
        elif mean_val < 50:
            print("STATUS: POTENTIALLY TOO DARK")
        else:
            print("STATUS: OK (Normal exposure range)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        print("Usage: python analyze_image.py <image_path>")
