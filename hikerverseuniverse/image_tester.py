# python
import sys

import numpy as np
from math import tan, radians, ceil
from typing import Tuple, Optional
from PIL import Image

from hikerverseuniverse.library.constants import L0
from hikerverseuniverse.universe_api import all_celestials_within_distance


def euler_to_rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R_yaw = np.array([[cy, -sy, 0],[sy, cy, 0],[0,0,1]])
    R_pitch = np.array([[cp, 0, sp],[0,1,0],[-sp,0,cp]])
    R_roll = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return R_yaw @ R_pitch @ R_roll

def render_starfield(
    star_positions: np.ndarray,      # shape (N,3)
    star_luminosities: np.ndarray,   # shape (N,)
    observer_pos: Tuple[float,float,float] = (0.0,0.0,0.0),
    orientation_matrix: Optional[np.ndarray] = None,  # 3x3 rotation (world->camera). If None use identity.
    fov_deg: float = 60.0,
    image_size: Tuple[int,int] = (1024,1024),
    psf_sigma_px: float = 1.0,
    magnitude_scale: float = 1.0,    # multiply luminosities before rendering
    clip_min_z: float = 1e-3         # avoid division by zero for very close stars
) -> np.ndarray:
    """
    Returns a float32 image array (H,W) with accumulated flux. Use `to_uint8` below to convert.
    - star_positions: world coordinates of stars (N,3).
    - star_luminosities: linear flux values (N,).
    - orientation_matrix: 3x3 matrix mapping world coords to camera coords (camera looks along +Z).
    - fov_deg: horizontal field of view in degrees.
    - psf_sigma_px: gaussian sigma in pixels for each star.
    """
    H, W = image_size
    stars = np.asarray(star_positions, dtype=np.float64)
    flux = np.asarray(star_luminosities, dtype=np.float64) * magnitude_scale
    assert stars.shape[0] == flux.shape[0], "positions and luminosities must match"

    # Translate to observer frame
    obs = np.asarray(observer_pos, dtype=np.float64)
    pts = stars - obs

    # Orientation: world -> camera. If None assume identity (camera aligned with world axes).
    if orientation_matrix is None:
        R = np.eye(3)
    else:
        R = np.asarray(orientation_matrix, dtype=np.float64)
        assert R.shape == (3,3)

    cam_pts = (R @ pts.T).T  # (N,3) in camera coordinates

    # Cull points behind camera (z <= 0)
    z = cam_pts[:,2]
    valid = z > clip_min_z
    if not np.any(valid):
        return np.zeros((H,W), dtype=np.float32)

    cam_pts = cam_pts[valid]
    flux = flux[valid]
    z = cam_pts[:,2]

    # Pinhole projection
    fov_rad = radians(fov_deg)
    fx = 0.5 * W / tan(0.5 * fov_rad)   # focal length in pixels (horizontal)
    fy = fx                             # assume square pixels; adjust if needed
    x_proj = (cam_pts[:,0] * fx) / z
    y_proj = (cam_pts[:,1] * fy) / z

    # Map projected coordinates to pixel coords (center origin)
    px = (W / 2.0) + x_proj
    py = (H / 2.0) - y_proj  # y image downwards

    # Cull those outside image (with margin for PSF)
    margin = ceil(3 * psf_sigma_px)
    inside = (px >= -margin) & (px <= W - 1 + margin) & (py >= -margin) & (py <= H - 1 + margin)
    if not np.any(inside):
        return np.zeros((H,W), dtype=np.float32)

    px = px[inside]
    py = py[inside]
    flux = flux[inside]

    # Prepare image
    img = np.zeros((H, W), dtype=np.float64)

    # Precompute small gaussian kernel (square) for PSF and reuse for stars with same sigma
    kradius = max(1, margin)
    ks = 2 * kradius + 1
    ax = np.arange(-kradius, kradius+1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (psf_sigma_px**2))
    kernel /= kernel.sum()  # normalize so kernel integrates to 1
    # Splat each star: distribute its flux over the kernel pixels
    for x_c, y_c, f in zip(px, py, flux):
        ix = int(round(x_c))
        iy = int(round(y_c))
        x0 = ix - kradius
        y0 = iy - kradius
        x1 = x0 + ks
        y1 = y0 + ks

        kx0 = 0
        ky0 = 0
        kx1 = ks
        ky1 = ks

        # Clip to image bounds
        if x0 < 0:
            kx0 = -x0
            x0 = 0
        if y0 < 0:
            ky0 = -y0
            y0 = 0
        if x1 > W:
            kx1 = ks - (x1 - W)
            x1 = W
        if y1 > H:
            ky1 = ks - (y1 - H)
            y1 = H

        if x0 >= x1 or y0 >= y1:
            continue

        img[y0:y1, x0:x1] += f * kernel[ky0:ky1, kx0:kx1]

    return img.astype(np.float32)

def to_uint8(img: np.ndarray, gamma: float = 0.8, clip_percentile: float = 99.5) -> np.ndarray:
    """
    Convert float image to 8-bit with simple auto-scaling and gamma.
    """
    arr = np.asarray(img, dtype=np.float32)
    vmax = np.percentile(arr, clip_percentile)
    if vmax <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = arr / vmax
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = scaled ** (1.0 / gamma)
    return (scaled * 255.0).astype(np.uint8)


# Example usage:
if __name__ == "__main__":

    stars_ = all_celestials_within_distance(coordinates=[0,0,0,], distance=10)

    star_locs = []
    star_lums = []
    for st in stars_:
        star_locs.append( (               abs(float(st.x)), abs(float(st.y)), abs(float(st.z))              ) )
        star_lums.append( 10000*float(st.luminosity)/L0 )

    star_locs = [(1.63e-05, 0.0, 0.0), (1.53958064, 1.17833026, 3.75297394), (1.61429006, 1.34955198, 3.77075724),
     (1.61436178, 1.34991384, 3.7705975), (0.05663598, 5.92215838, 0.48614098), (7.44196386, 2.11682884, 0.95210582),
     (6.51348, 1.6440343, 4.87534304), (7.40182674, 3.4131222, 2.64047938), (7.40225706, 3.41332106, 2.6406326),
     (1.61149298, 8.07414306, 2.4726611), (1.61167554, 8.07441364, 2.4726122), (1.90997858, 8.64700004, 3.91240098),
     (1.63e-05, 0.0, 0.0), (1.53958064, 1.17833026, 3.75297394), (1.61429006, 1.34955198, 3.77075724),
     (1.61436178, 1.34991384, 3.7705975), (0.05663598, 5.92215838, 0.48614098), (7.44196386, 2.11682884, 0.95210582),
     (6.51348, 1.6440343, 4.87534304), (7.40182674, 3.4131222, 2.64047938), (7.40225706, 3.41332106, 2.6406326),
     (1.61149298, 8.07414306, 2.4726611), (1.61167554, 8.07441364, 2.4726122), (1.90997858, 8.64700004, 3.91240098),
     (1.63e-05, 0.0, 0.0), (1.53958064, 1.17833026, 3.75297394), (1.61429006, 1.34955198, 3.77075724),
     (1.61436178, 1.34991384, 3.7705975), (0.05663598, 5.92215838, 0.48614098), (7.44196386, 2.11682884, 0.95210582),
     (6.51348, 1.6440343, 4.87534304), (7.40182674, 3.4131222, 2.64047938), (7.40225706, 3.41332106, 2.6406326),
     (1.61149298, 8.07414306, 2.4726611), (1.61167554, 8.07441364, 2.4726122), (1.90997858, 8.64700004, 3.91240098)]

    star_lums = [10047.021943573667, 0.579745452583711, 4430.342999135716, 15503.771208254617, 4.446695089097862, 0.2085630411265554, 57.28452955046373, 0.566027264412932, 0.5021544906013337, 229317.58042370147, 25.611637764519507, 5.4908218192357, 10047.021943573667, 0.579745452583711, 4430.342999135716, 15503.771208254617, 4.446695089097862, 0.2085630411265554, 57.28452955046373, 0.566027264412932, 0.5021544906013337, 229317.58042370147, 25.611637764519507, 5.4908218192357, 10047.021943573667, 0.579745452583711, 4430.342999135716, 15503.771208254617, 4.446695089097862, 0.2085630411265554, 57.28452955046373, 0.566027264412932, 0.5021544906013337, 229317.58042370147, 25.611637764519507, 5.4908218192357]

    # Convert to numpy arrays and align lengths
    stars = np.asarray(star_locs, dtype=np.float64)
    lum = np.asarray(star_lums, dtype=np.float64)
    n = min(len(stars), len(lum))
    stars = stars[:n]
    lum = lum[:n]

    # Ensure all stars are in front of the camera (avoid z <= 0)
    _eps_z = 1e-3
    mask_front = stars[:, 2] <= _eps_z
    if np.any(mask_front):
        stars[mask_front, 2] = _eps_z

    # Use the small catalog lists as inputs
    stars = np.asarray(star_locs, dtype=np.float64)
    lum = np.asarray(star_lums, dtype=np.float64)

    img = render_starfield(stars, lum, observer_pos=(0, 0, 0),
                           orientation_matrix=None,
                           fov_deg=90.0, image_size=(512, 512), psf_sigma_px=1.2)

    img8 = to_uint8(img, gamma=0.7)
    Image.fromarray(img8).save("starfield2.png")







    # generate example luminosities matching the number of `stars`
    n_stars = len(stars)
    rng = np.random.default_rng(42)
    # log-uniform values from 1e-1 to 1e5 to cover a wide dynamic range
    example_lums = np.exp(rng.uniform(np.log(1e-1), np.log(1e5), size=n_stars))
    # inject a few bright outliers to test clipping/visibility
    if n_stars >= 3:
        example_lums[0] = 2.3e5
        example_lums[1] = 1.5e4
        example_lums[2] = 1.0e3

    img = render_starfield(stars, example_lums, observer_pos=(0, 0, 0),
                           orientation_matrix=None,
                           fov_deg=90.0, image_size=(512, 512), psf_sigma_px=1.2)

    img8 = to_uint8(img, gamma=0.7)
    Image.fromarray(img8).save("starfield.png")


    sys.exit()

    # Create a random synthetic star catalog
    N = 5000
    # random points on a sphere at varying distances
    directions = np.random.normal(size=(N,3))
    directions /= np.linalg.norm(directions, axis=1)[:,None]
    distances = np.random.uniform(10.0, 1000.0, size=(N,1))
    stars = directions * distances
    # luminosities with large dynamic range
    lum = 10**(np.random.uniform(-3, 1, size=N))

    # render from origin, looking along +Z (identity rotation)
    img = render_starfield(stars, lum, observer_pos=(0,0,0),
                           orientation_matrix=None,
                           fov_deg=90.0, image_size=(1024,1024), psf_sigma_px=1.2)

    img8 = to_uint8(img, gamma=0.7)
    Image.fromarray(img8).save("starfield.png")
