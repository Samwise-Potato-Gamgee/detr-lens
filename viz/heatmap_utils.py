"""Gaussian splatting utility for deformable attention point visualization."""

import numpy as np


def splat_gaussian(
    canvas: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    weights: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Splat weighted Gaussian blobs at (pts_x, pts_y) onto canvas.

    canvas: (H, W) float32
    pts_x, pts_y: pixel coordinates (clipped to canvas bounds)
    weights: per-point scalar weights (sum need not be 1)
    radius: Gaussian sigma in pixels
    """
    H, W = canvas.shape
    if radius < 1:
        radius = 1

    # Build Gaussian kernel
    ksize = radius * 4 + 1
    ky = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    kx = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    gy, gx = np.meshgrid(ky, kx, indexing="ij")
    kernel = np.exp(-(gx**2 + gy**2) / (2 * radius**2))

    pts_x = np.clip(pts_x, 0, W - 1)
    pts_y = np.clip(pts_y, 0, H - 1)

    for xi, yi, wi in zip(pts_x, pts_y, weights):
        x0 = xi - ksize // 2
        y0 = yi - ksize // 2
        # Canvas slice bounds
        cx0 = max(0, x0)
        cy0 = max(0, y0)
        cx1 = min(W, x0 + ksize)
        cy1 = min(H, y0 + ksize)
        # Kernel slice bounds
        kx0 = cx0 - x0
        ky0 = cy0 - y0
        kx1 = kx0 + (cx1 - cx0)
        ky1 = ky0 + (cy1 - cy0)
        if cx1 > cx0 and cy1 > cy0:
            canvas[cy0:cy1, cx0:cx1] += float(wi) * kernel[ky0:ky1, kx0:kx1]

    vmax = canvas.max()
    if vmax > 0:
        canvas /= vmax
    return canvas
