"""
Minimal SSIM (Structural Similarity Index) implementation for golden testing.

Pure NumPy implementation to avoid heavy dependencies like scikit-image.
Based on the original SSIM paper by Wang et al. (2004).
"""

import numpy as np


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian window for SSIM computation."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    return np.outer(g, g)


def ssim(img1: np.ndarray, img2: np.ndarray, 
         data_range: float = 255.0,
         k1: float = 0.01, 
         k2: float = 0.03,
         win_size: int = 11,
         sigma: float = 1.5) -> float:
    """
    Compute SSIM between two images.
    
    Args:
        img1, img2: Images to compare, same shape
        data_range: Maximum possible pixel value difference (255 for uint8)
        k1, k2: SSIM constants (default values from paper)
        win_size: Size of Gaussian window
        sigma: Standard deviation for Gaussian window
        
    Returns:
        SSIM value between -1 and 1 (1 = identical)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
    
    if img1.ndim != 2 and img1.ndim != 3:
        raise ValueError(f"Images must be 2D or 3D, got {img1.ndim}D")
    
    # Convert to float64 for computation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # For 3D images (H,W,C), compute SSIM per channel and average
    if img1.ndim == 3:
        ssim_vals = []
        for c in range(img1.shape[2]):
            ssim_vals.append(ssim(img1[:,:,c], img2[:,:,c], data_range, k1, k2, win_size, sigma))
        return np.mean(ssim_vals)
    
    # SSIM constants
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    # Create Gaussian window
    window = _gaussian_window(win_size, sigma)
    
    # Compute local means
    mu1 = _filter2d(img1, window)
    mu2 = _filter2d(img2, window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = _filter2d(img1 ** 2, window) - mu1_sq
    sigma2_sq = _filter2d(img2 ** 2, window) - mu2_sq
    sigma12 = _filter2d(img1 * img2, window) - mu1_mu2
    
    # Compute SSIM map
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = numerator / denominator
    
    # Return mean SSIM
    return float(np.mean(ssim_map))


def _filter2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution using scipy.ndimage if available, otherwise manual."""
    try:
        from scipy.ndimage import convolve
        return convolve(img, kernel, mode='constant', cval=0.0)
    except ImportError:
        # Fallback to manual convolution (slower but dependency-free)
        return _manual_convolve2d(img, kernel)


def _manual_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2D convolution implementation."""
    kh, kw = kernel.shape
    ih, iw = img.shape
    
    # Calculate padding
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Pad image
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Output array
    output = np.zeros_like(img)
    
    # Convolution
    for i in range(ih):
        for j in range(iw):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return output