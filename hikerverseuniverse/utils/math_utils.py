import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n


def gaussian_psf(radius_px: int, sigma_px: float) -> np.ndarray:
    """Generate a 2D Gaussian PSF."""
    ax = np.arange(-radius_px, radius_px + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma_px**2)
    return kernel / kernel.sum()


