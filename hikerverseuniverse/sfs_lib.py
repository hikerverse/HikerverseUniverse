

from __future__ import annotations

import math

import numpy as np
from math import tan, radians, pi

import scipy
from PIL import Image
from scipy.interpolate import interp1d

from hikerverseuniverse.utils.math_utils import normalize

# --- Physical constants ---

PC_TO_M = 3.085677581e16  # meters per parsec
L_SUN = 3.828e26          # W

def tanner_helland_temp_to_rgb(temp_k):
    """
    Converts a color temperature (in Kelvin) to an RGB color using the Tanner Helland algorithm.
    :param temp_k: Temperature in Kelvin (range: ~1000K to ~40000K)
    :return: Tuple of (R, G, B) values (0-255)
    """
    temp_k = max(1000, min(40000, temp_k))  # Clamp temperature range

    # Calculate red
    if temp_k <= 6600:
        red = 255
    else:
        red = temp_k / 100 - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))

    # Calculate green
    if temp_k <= 6600:
        green = temp_k / 100
        green = 99.4708025861 * (green ** -0.0755148492)
        green = max(0, min(255, green))
    else:
        green = temp_k / 100 - 60
        green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))

    # Calculate blue
    if temp_k >= 6600:
        blue = 255
    elif temp_k <= 1900:
        blue = 0
    else:
        blue = temp_k / 100 - 10
        blue = 138.5177312231 * (blue ** -0.0755148492)
        blue = max(0, min(255, blue))

    rgb =[ int(red), int(green), int(blue)]
    return rgb / np.max(rgb)


def temperature_to_rgb_new(temperature: float) -> np.ndarray:
    """
    Convert a temperature (in Kelvin) to an RGB color.

    Args:
        temperature (float): Temperature in Kelvin (2000–50000 K).

    Returns:
        np.ndarray: RGB color normalized to [0, 1].
    """
    temperature = np.clip(temperature, 2000, 50000)

    # Constants for blackbody approximation
    if temperature <= 6600:
        r = 1.0
        g = np.clip(0.3900815787690196 * np.log(temperature) - 0.6318414437886275, 0, 1)
        b = np.clip(temperature - 1000, 0, 1) / 6600 if temperature > 1900 else 0
    else:
        r = np.clip(1.292936186062745 * (temperature - 6000), 0, 1)
        g = np.clip(temperature - 1000, 0, 1) / 6600 if temperature > 1900 else 0
        b = np.clip(temperature - 1000, 0, 1) / 6600 if temperature > 1900 else 0

    rgb = np.array([r, g, b])
    return rgb / np.max(rgb)

def rgb_to_temperature_new(rgb: np.ndarray) -> float:
    """
    Estimate the temperature of a star based on its RGB color.

    Args:
        rgb (np.ndarray): RGB values of the star (normalized to [0, 1]).

    Returns:
        float: Estimated temperature in Kelvin (2000–50000 K).
    """
    # Normalize RGB values
    rgb = rgb / np.max(rgb)

    # Precomputed RGB-to-temperature mapping (example values)
    temperatures = np.array([2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 20000, 50000])
    red_ratios = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    blue_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Interpolate temperature based on red and blue ratios
    red_interp = interp1d(red_ratios, temperatures, bounds_error=False, fill_value="extrapolate")
    blue_interp = interp1d(blue_ratios, temperatures, bounds_error=False, fill_value="extrapolate")

    # Estimate temperature
    r, g, b = rgb
    temperature = (red_interp(r) + blue_interp(b)) / 2
    return max(2000, min(50000, temperature))



















































def look_at_basis(cam_dir: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal camera basis from a forward direction and an up hint.
    Returns (right, up, forward).
    """
    forward = normalize(cam_dir)

    up = normalize(up_hint)
    if abs(np.dot(forward, up)) > 0.999:
        # If up is almost parallel to forward, pick an alternate up
        if abs(forward[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

    right = np.cross(forward, up)
    right = normalize(right)
    up = np.cross(right, forward)

    return right, up, forward


def image_to_rays(
    img_height: int,
    img_width: int,
    fov_y_deg: float,
) -> np.ndarray:
    """
    Produce a ray direction per pixel in camera space, using a pinhole camera
    model with vertical FOV fov_y_deg and pixels centered at (i + 0.5, j + 0.5).
    Returns (H, W, 3) unit vectors.
    """
    fov_y_rad = radians(fov_y_deg)
    f = 1.0 / tan(fov_y_rad / 2.0)

    ys, xs = np.meshgrid(
        np.arange(img_height),
        np.arange(img_width),
        indexing="ij",
    )

    # NDC coordinates in [-1,1]
    u = (xs + 0.5) / img_width * 2.0 - 1.0
    v = -((ys + 0.5) / img_height * 2.0 - 1.0)

    x_over_z = u / f
    y_over_z = v / f

    z_cam = np.ones_like(x_over_z, dtype=np.float64)
    x_cam = x_over_z * z_cam
    y_cam = y_over_z * z_cam

    dirs = np.stack([x_cam, y_cam, z_cam], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs /= np.clip(norms, 1e-12, None)
    return dirs  # (H, W, 3)


def temperature_to_rgb222(temp_k: float) -> np.ndarray:
    """Convert temperature (K) to RGB color."""
    t = np.clip(temp_k, 2000, 40000)
    x = np.log10(t / 6500.0)
    r = np.clip(1.0 - 0.8 * x, 0.0, 1.5)
    g = np.clip(1.0 - 0.2 * x**2, 0.0, 1.0)
    b = np.clip(1.0 + 1.2 * x, 0.2, 2.0)
    rgb = np.array([r, g, b])
    return rgb / np.max(rgb)





def temperature_to_rgb2(temp_k: float) -> np.ndarray:
    """
    Very rough mapping from effective temperature (K) to linear RGB.
    Based on a simple blackbody approximation, normalized to max channel 1.0.
    """
    saturation_factor = 1  # increase to boost saturation

    t = float(np.clip(temp_k, 2000.0, 40000.0))

    # Use a simple piecewise approximation (not physically exact)
    # Treat 6500K as roughly "white"
    x = np.log10(t / 6500.0)

    # Blue vs red balance
    r = np.clip(1.0 - 0.8 * x, 0.0, 1.5)
    g = np.clip(1.0 - 0.2 * x**2, 0.0, 1.0)
    b = np.clip(1.0 + 1.2 * x, 0.2, 2.0)

    rgb = np.array([r, g, b], dtype=np.float64)
    m = np.max(rgb)
    if m > 0:
        rgb /= m

    # Apply saturation factor
    mean = np.mean(rgb)
    rgb = mean + saturation_factor * (rgb - mean)

    return rgb


def temperature_to_rgb(temp_k):
    """
    Converts from K to RGB, algorithm courtesy of
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    # range check
    if temp_k < 1000:
        temp_k = 1000
    elif temp_k > 40000:
        temp_k = 40000

    tmp_internal = temp_k / 100.0

    # red
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red

    # green
    if tmp_internal <= 66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green

    # blue
    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue

    rgb = np.array([red, green, blue], dtype=np.float64)
    return rgb


def temperature_to_luminosity(temp_k: float, radius_rsun: float = 1.0) -> float:
    """
    Approximate bolometric luminosity from temperature and radius via Stefan–Boltzmann law:
        L = 4 * pi * R^2 * sigma * T^4
    expressed in W. Radius is in units of solar radius.
    """
    # Stefan–Boltzmann constant
    sigma = 5.670374419e-8  # W / (m^2 K^4)
    R_sun_m = 6.957e8       # m

    T = float(temp_k)
    R_m = float(radius_rsun) * R_sun_m
    L = 4.0 * pi * (R_m**2) * sigma * (T**4)
    return L


def estimate_temperature_from_color(mean_rgb: np.ndarray) -> float:
    """
    Estimate an effective temperature (K) from normalized linear RGB
    by nearest neighbor over a small temperature grid.
    """
    target = mean_rgb.astype(float)
    total = target.sum()
    if total <= 0:
        return 5800.0  # default ~Sun

    target /= total

    # Build a coarse grid of prototype colors
    temps = np.array([2500, 3500, 4500, 5500, 6500, 8000, 12000, 20000, 35000], dtype=float)
    protos = np.stack([temperature_to_rgb(T) for T in temps], axis=0)
    protos /= np.clip(protos.sum(axis=1, keepdims=True), 1e-12, None)

    dists = np.linalg.norm(protos - target[None, :], axis=1)
    idx = int(np.argmin(dists))
    return float(temps[idx])



# --- Spectral class <-> RGB and luminosity ---

def spectral_class_to_rgb(spectral_class: str) -> np.ndarray:
    """
    Approximate linear RGB colors for spectral classes, not gamma-corrected.
    These are arbitrary, but should be used consistently for render and classify.
    """
    mapping = {
        "O": np.array([0.6, 0.7, 1.0]),
        "B": np.array([0.7, 0.8, 1.0]),
        "A": np.array([0.9, 0.9, 1.0]),
        "F": np.array([1.0, 1.0, 0.95]),
        "G": np.array([1.0, 0.95, 0.8]),
        "K": np.array([1.0, 0.8, 0.6]),
        "M": np.array([1.0, 0.6, 0.5]),
    }
    return mapping.get(spectral_class, np.array([1.0, 1.0, 1.0]))


def spectral_class_to_luminosity(spectral_class: str) -> float:
    """
    Approximate bolometric luminosity (W) for a spectral class.
    """
    mapping = {
        "O": 1e5 * L_SUN,
        "B": 1e3 * L_SUN,
        "A": 25.0 * L_SUN,
        "F": 6.0 * L_SUN,
        "G": 1.2 * L_SUN,
        "K": 0.4 * L_SUN,
        "M": 0.05 * L_SUN,
    }
    return mapping.get(spectral_class, L_SUN)


def estimate_spectral_class_from_color(mean_rgb: np.ndarray) -> str:
    """
    Classify a spectral type from normalized linear RGB by nearest prototype.

    Expects `mean_rgb` either raw (will be normalized inside) or already
    normalized such that sum(mean_rgb) == 1.0 for nonzero pixels.
    """
    target = mean_rgb.astype(float)
    total = target.sum()
    if total <= 0:
        return "G"
    target /= total

    # Same prototypes as spectral_class_to_rgb, but normalized
    prototypes = {
        "O": np.array([0.6, 0.7, 1.0]),
        "B": np.array([0.7, 0.8, 1.0]),
        "A": np.array([0.9, 0.9, 1.0]),
        "F": np.array([1.0, 1.0, 0.95]),
        "G": np.array([1.0, 0.95, 0.8]),
        "K": np.array([1.0, 0.8, 0.6]),
        "M": np.array([1.0, 0.6, 0.5]),
    }

    best_spec = "G"
    best_dist = float("inf")
    for spec, proto in prototypes.items():
        p = proto.astype(float)
        p /= p.sum()
        d = np.linalg.norm(target - p)
        if d < best_dist:
            best_dist = d
            best_spec = spec
    return best_spec


# --- Tone mapping ---

def tone_map(flux_image: np.ndarray, exposure: float, gamma: float = 1.0, method: str = "log") -> np.ndarray:
    """
    Very simple tone-mapping from physical flux to 8-bit.
    `flux_image` is linear in W/m^2. Returns uint8 array.
    """
    img = flux_image.astype(np.float64) * exposure

    if method == "log":
        img = np.log1p(img)
        img /= np.max(img) if np.max(img) > 0 else 1.0
    elif method == "linear":
        img /= np.max(img) if np.max(img) > 0 else 1.0
    else:
        raise ValueError("Unknown tone mapping method")

    # gamma
    img = np.clip(img, 0.0, 1.0) ** (1.0 / max(gamma, 1e-6))
    img_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img_u8


def tone_map_to_u16(flux_image: np.ndarray, exposure: float, gamma: float = 1.0, method: str = "log") -> np.ndarray:
    """
    16-bit version of tone_map. Good if you want to save images with more dynamic range.
    """
    img_u8 = tone_map(flux_image, exposure=exposure, gamma=gamma, method=method).astype(np.float32)
    img_u16 = np.clip(img_u8 / 255.0 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    return img_u16




def render_stars_rgb_temperature2(
    stars_pos_pc: np.ndarray,
    stars_lum_w: np.ndarray,    # can still pass explicit L, or compute from T
    stars_temp_k: np.ndarray,
    cam_pos_pc: np.ndarray,
    cam_dir: np.ndarray,
    up_hint: np.ndarray,
    fov_y_deg: float,
    img_width: int,
    img_height: int,
    det_threshold_w_m2: float,
    psf_kernel: np.ndarray | None = None,  # Optional PSF kernel
) -> np.ndarray:
    stars_pos_pc = np.asarray(stars_pos_pc, dtype=np.float64)
    stars_lum_w = np.asarray(stars_lum_w, dtype=np.float64)
    stars_temp_k = np.asarray(stars_temp_k, dtype=np.float64)

    H, W = img_height, img_width
    right, up, forward = look_at_basis(cam_dir, up_hint)
    fov_y_rad = radians(fov_y_deg)
    f = 1.0 / tan(fov_y_rad / 2.0)

    img = np.zeros((H, W, 3), dtype=np.float64)

    for pos, L, T in zip(stars_pos_pc, stars_lum_w, stars_temp_k):
        v = pos - cam_pos_pc
        d = np.linalg.norm(v)
        if d <= 0:
            continue

        dir_world = v / d
        if np.dot(dir_world, forward) <= 0.0:
            continue

        x_cam = np.dot(dir_world, right)
        y_cam = np.dot(dir_world, up)
        z_cam = np.dot(dir_world, forward)
        if z_cam <= 0:
            continue

        u = x_cam / (z_cam * f)
        v_ndc = y_cam / (z_cam * f)
        px = (u + 1.0) * 0.5 * W
        py = (1.0 - (v_ndc + 1.0) * 0.5) * H

        ix = int(px)
        iy = int(py)
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            continue

        flux = L / (4.0 * pi * (d * PC_TO_M) ** 2)
        if flux < det_threshold_w_m2:
            continue

        c = temperature_to_rgb(T)
        img[iy, ix, :] += flux * c

    # Apply PSF convolution if a kernel is provided
    if psf_kernel is not None:
        from scipy.signal import convolve
        for channel in range(3):
            img[:, :, channel] = convolve(img[:, :, channel], psf_kernel, mode="same")

    return img


def render_stars_rgb_temperature(
    stars_pos_pc: np.ndarray,
    stars_lum_w: np.ndarray,    # can still pass explicit L, or compute from T
    stars_temp_k: np.ndarray,
    cam_pos_pc: np.ndarray,
    cam_dir: np.ndarray,
    up_hint: np.ndarray,
    fov_y_deg: float,
    img_width: int,
    img_height: int,
    det_threshold_w_m2: float,
) -> np.ndarray:
    stars_pos_pc = np.asarray(stars_pos_pc, dtype=np.float64)
    stars_lum_w = np.asarray(stars_lum_w, dtype=np.float64)
    stars_temp_k = np.asarray(stars_temp_k, dtype=np.float64)

    H, W = img_height, img_width
    right, up, forward = look_at_basis(cam_dir, up_hint)
    fov_y_rad = radians(fov_y_deg)
    f = 1.0 / tan(fov_y_rad / 2.0)

    img = np.zeros((H, W, 3), dtype=np.float64)

    for pos, L, T in zip(stars_pos_pc, stars_lum_w, stars_temp_k):
        v = pos - cam_pos_pc
        d = np.linalg.norm(v)
        if d <= 0:
            continue

        dir_world = v / d
        if np.dot(dir_world, forward) <= 0.0:
            continue

        x_cam = np.dot(dir_world, right)
        y_cam = np.dot(dir_world, up)
        z_cam = np.dot(dir_world, forward)
        if z_cam <= 0:
            continue

        u = x_cam / (z_cam * f)
        v_ndc = y_cam / (z_cam * f)
        px = (u + 1.0) * 0.5 * W
        py = (1.0 - (v_ndc + 1.0) * 0.5) * H

        ix = int(px)
        iy = int(py)
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            continue

        flux = L / (4.0 * pi * (d * PC_TO_M) ** 2)
        if flux < det_threshold_w_m2:
            continue

        c = temperature_to_rgb(T)
        img[iy, ix, :] += flux * c

    return img

def reconstruct_star_field_from_image_per_temp(
        energy_img: np.ndarray,
        rgb_energy_img: np.ndarray,
    cam_pos_pc: np.ndarray,
    cam_dir: np.ndarray,
    up_hint: np.ndarray,
    fov_y_deg: float,
    det_threshold_energy: float,
    t_exp_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Reconstruction using time-integrated energy (J/m^2) instead of instantaneous flux.
        Internally converts back to flux by dividing by t_exp_s so physics stays the same.
        """
    # convert energy back to flux for the physical formulas
    flux_img = energy_img / float(t_exp_s)
    rgb_img = rgb_energy_img  # colors don't change with exposure


    H, W = flux_img.shape
    if rgb_img.shape != (H, W, 3):
        raise ValueError("rgb_img must be shape (H, W, 3) matching flux_img")

    mask = flux_img >= (det_threshold_energy/t_exp_s)
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    ys, xs = np.where(mask)
    flux_vals = flux_img[ys, xs]

    temps = []
    L_list = []
    for y, x in zip(ys, xs):
        color = rgb_img[y, x]
        T = estimate_temperature_from_color(color)
        L = temperature_to_luminosity(T)
        temps.append(T)
        L_list.append(L)

    temps_arr = np.array(temps, dtype=np.float64)
    L_arr = np.array(L_list, dtype=np.float64)

    rays_cam = image_to_rays(H, W, fov_y_deg)
    star_rays_cam = rays_cam[ys, xs]

    right, up, forward = look_at_basis(cam_dir, up_hint)
    R = np.stack([right, up, forward], axis=1)
    star_rays_world = (R @ star_rays_cam.T).T

    d_m = np.sqrt(L_arr / (4.0 * pi * flux_vals))
    d_pc = d_m / PC_TO_M

    star_pos_pc = cam_pos_pc[None, :] + star_rays_world * d_pc[:, None]
    return star_pos_pc, L_arr, temps_arr

# --- Rendering ---

def render_stars_rgb(
    stars_pos_pc: np.ndarray,
    stars_lum_w: np.ndarray,
    stars_spec: np.ndarray,
    cam_pos_pc: np.ndarray,
    cam_dir: np.ndarray,
    up_hint: np.ndarray,
    fov_y_deg: float,
    img_width: int,
    img_height: int,
    det_threshold_w_m2: float,
) -> np.ndarray:
    """
    Extremely simple point-star renderer.

    Each star is projected to a pixel and contributes a single-sample flux:
        flux = L / (4*pi*d^2) in W/m^2
    multiplied by a spectral RGB color.
    """
    stars_pos_pc = np.asarray(stars_pos_pc, dtype=np.float64)
    stars_lum_w = np.asarray(stars_lum_w, dtype=np.float64)
    stars_spec = np.asarray(stars_spec, dtype=object)

    H, W = img_height, img_width

    # Camera basis
    right, up, forward = look_at_basis(cam_dir, up_hint)

    # Image plane rays
    rays_cam = image_to_rays(H, W, fov_y_deg)  # (H, W, 3)

    # Precompute for pixel coordinate mapping
    fov_y_rad = radians(fov_y_deg)
    f = 1.0 / tan(fov_y_rad / 2.0)

    img = np.zeros((H, W, 3), dtype=np.float64)

    for pos, L, spec in zip(stars_pos_pc, stars_lum_w, stars_spec):
        # World-space vector from camera to star
        v = pos - cam_pos_pc
        d = np.linalg.norm(v)
        if d <= 0:
            continue

        dir_world = v / d
        # Check if in front of camera
        if np.dot(dir_world, forward) <= 0.0:
            continue

        # Transform to camera space: components along right, up, forward
        x_cam = np.dot(dir_world, right)
        y_cam = np.dot(dir_world, up)
        z_cam = np.dot(dir_world, forward)

        # Perspective divide
        if z_cam <= 0:
            continue

        u = x_cam / (z_cam * f)
        v_ndc = y_cam / (z_cam * f)

        # Convert NDC [-1,1] to pixel coordinates
        px = (u + 1.0) * 0.5 * W  # pixel-space with origin at left edge
        py = (1.0 - (v_ndc + 1.0) * 0.5) * H

        ix = int(px)  # floor
        iy = int(py)
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            continue

        # Physical flux
        flux = L / (4.0 * pi * (d * PC_TO_M) ** 2)
        if flux < det_threshold_w_m2:
            continue

        c = spectral_class_to_rgb(str(spec))
        img[iy, ix, :] += flux * c

    return img  # (H, W, 3) linear W/m^2 per channel


# --- Reconstruction ---


def reconstruct_star_field_from_image_per_lum(
    flux_img: np.ndarray,          # (H, W), physical flux (sum of channels)
    rgb_img: np.ndarray,           # (H, W, 3), normalized linear RGB for color
    cam_pos_pc: np.ndarray,
    cam_dir: np.ndarray,
    up_hint: np.ndarray,
    fov_y_deg: float,
    det_threshold_w_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct 3D star positions using per-pixel luminosity inferred
    from the pixel color (spectral class), with simple centroiding
    around local maxima for subpixel accuracy.
    """
    flux_img = np.asarray(flux_img, dtype=np.float64)
    rgb_img = np.asarray(rgb_img, dtype=np.float64)

    H, W = flux_img.shape
    if rgb_img.shape != (H, W, 3):
        raise ValueError("rgb_img must be shape (H, W, 3) matching flux_img")

    # 1) Detect local maxima above threshold (ignore 1-pixel border for simplicity)
    mask = flux_img >= det_threshold_w_m2
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.array([], dtype=object),
        )

    peaks_y = []
    peaks_x = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if not mask[y, x]:
                continue
            center = flux_img[y, x]
            # Check 8-neighborhood for strict local max
            window = flux_img[y - 1:y + 2, x - 1:x + 2]
            if center == window.max() and np.count_nonzero(window == center) == 1:
                peaks_y.append(y)
                peaks_x.append(x)

    if len(peaks_y) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.array([], dtype=object),
        )

    # 2) Centroid around each peak (3x3 window) for subpixel position
    centroids_y = []
    centroids_x = []
    flux_vals = []
    for y0, x0 in zip(peaks_y, peaks_x):
        win = flux_img[y0 - 1:y0 + 2, x0 - 1:x0 + 2]
        ys, xs = np.mgrid[y0 - 1:y0 + 2, x0 - 1:x0 + 2]
        w = win
        total = w.sum()
        if total <= 0:
            # fallback to pixel center
            cy, cx = float(y0), float(x0)
            total = flux_img[y0, x0]
        else:
            cy = float((ys * w).sum() / total)
            cx = float((xs * w).sum() / total)

        centroids_y.append(cy)
        centroids_x.append(cx)
        flux_vals.append(total)

    centroids_y = np.array(centroids_y, dtype=np.float64)
    centroids_x = np.array(centroids_x, dtype=np.float64)
    flux_vals = np.array(flux_vals, dtype=np.float64)

    # 3) Infer spectral class and luminosity at the centroid positions
    star_spec = []
    star_lum_list = []
    for cy, cx in zip(centroids_y, centroids_x):
        # Bilinear sample rgb_img at float coords (very simple version)
        y0 = int(np.clip(np.floor(cy), 0, H - 2))
        x0 = int(np.clip(np.floor(cx), 0, W - 2))
        dy = cy - y0
        dx = cx - x0

        c00 = rgb_img[y0,     x0    ]
        c10 = rgb_img[y0 + 1, x0    ]
        c01 = rgb_img[y0,     x0 + 1]
        c11 = rgb_img[y0 + 1, x0 + 1]

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy
        color = c0 * (1 - dx) + c1 * dx

        spec = estimate_spectral_class_from_color(color)
        L = spectral_class_to_luminosity(spec)
        star_spec.append(spec)
        star_lum_list.append(L)

    star_lum_vals = np.array(star_lum_list, dtype=np.float64)
    star_spec_arr = np.array(star_spec, dtype=object)

    # 4) Build rays for float pixel coordinates (inverse of image_to_rays)
    fov_y_rad = radians(fov_y_deg)
    f = 1.0 / tan(fov_y_rad / 2.0)

    # pixel centers at (i+0.5, j+0.5)
    u = (centroids_x + 0.5) / W * 2.0 - 1.0          # in [-1, 1]
    v = -((centroids_y + 0.5) / H * 2.0 - 1.0)

    x_over_z = u / f
    y_over_z = v / f
    z_cam = np.ones_like(x_over_z, dtype=np.float64)
    x_cam = x_over_z * z_cam
    y_cam = y_over_z * z_cam

    star_rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    star_rays_cam /= np.linalg.norm(star_rays_cam, axis=-1, keepdims=True)

    # 5) Camera basis and world-space directions
    right, up, forward = look_at_basis(cam_dir, up_hint)
    R = np.stack([right, up, forward], axis=1)
    star_rays_world = (R @ star_rays_cam.T).T

    # 6) Distances from per-star luminosity and measured flux
    d_m = np.sqrt(star_lum_vals / (4.0 * pi * flux_vals))
    d_pc = d_m / PC_TO_M

    # 7) Positions
    star_pos_pc = cam_pos_pc[None, :] + star_rays_world * d_pc[:, None]

    return star_pos_pc, star_lum_vals, star_spec_arr

def apply_time_integration(flux_img_rgb: np.ndarray,
                           t_exp_s: float,
                           add_noise: bool = False,
                           read_noise_std: float = 0.0) -> np.ndarray:
    """
    Apply time integration to a physical RGB flux image.

    * flux_img_rgb: per-channel flux [W/m^2]
    * t_exp_s: exposure time [s]
    * Returns an image proportional to collected energy/area [J/m^2]
    """
    flux_img_rgb = np.asarray(flux_img_rgb, dtype=np.float64)

    # Ideal integration: energy = flux * time
    energy_img_rgb = flux_img_rgb * float(t_exp_s)

    if add_noise:
        # Simple Poisson-like + read noise model
        # sigma_shot ~ sqrt(energy), sigma_read = constant
        shot_noise_std = np.sqrt(np.clip(energy_img_rgb, 0.0, None))
        noise = np.random.normal(
            loc=0.0,
            scale=np.sqrt(shot_noise_std**2 + read_noise_std**2),
            size=energy_img_rgb.shape,
        )
        energy_img_rgb = np.clip(energy_img_rgb + noise, 0.0, None)

    return energy_img_rgb

def reconstruct_star_field_from_image_per_lum2(
    flux_img: np.ndarray,          # (H, W), physical flux (sum of channels)
    rgb_img: np.ndarray,           # (H, W, 3), normalized linear RGB for color
    cam_pos_pc: np.ndarray,
    cam_dir: np.ndarray,
    up_hint: np.ndarray,
    fov_y_deg: float,
    det_threshold_w_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct 3D star positions using per-pixel luminosity inferred
    from the pixel color (spectral class).

    `flux_img` and `rgb_img` must be aligned: flux per pixel and its RGB color.

    Returns:
        star_pos_pc:  (N, 3) reconstructed positions in parsecs.
        star_lum_w:   (N,)   inferred per-star luminosities.
        star_spec:    (N,)   inferred spectral classes (dtype=object).
    """
    flux_img = np.asarray(flux_img, dtype=np.float64)
    rgb_img = np.asarray(rgb_img, dtype=np.float64)

    H, W = flux_img.shape
    if rgb_img.shape != (H, W, 3):
        raise ValueError("rgb_img must be shape (H, W, 3) matching flux_img")

    # 1) Pixels with detectable flux
    mask = flux_img >= det_threshold_w_m2
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.array([], dtype=object),
        )

    ys, xs = np.where(mask)
    flux_vals = flux_img[ys, xs]

    # 2) Infer spectral class and luminosity per pixel
    star_spec = []
    star_lum_list = []
    for y, x in zip(ys, xs):
        color = rgb_img[y, x]  # (3,)
        spec = estimate_spectral_class_from_color(color)
        L = spectral_class_to_luminosity(spec)
        star_spec.append(spec)
        star_lum_list.append(L)

    star_lum_vals = np.array(star_lum_list, dtype=np.float64)
    star_spec_arr = np.array(star_spec, dtype=object)

    # 3) Camera rays for those pixels
    rays_cam = image_to_rays(H, W, fov_y_deg)
    star_rays_cam = rays_cam[ys, xs]  # (N, 3)

    # 4) Camera basis and world-space directions
    right, up, forward = look_at_basis(cam_dir, up_hint)
    R = np.stack([right, up, forward], axis=1)  # (3, 3)
    star_rays_world = (R @ star_rays_cam.T).T  # (N, 3)

    # 5) Distances from per-star luminosity and measured flux
    # flux_i = L_i / (4*pi*d_i^2)  =>  d_i = sqrt(L_i / (4*pi*flux_i))
    d_m = np.sqrt(star_lum_vals / (4.0 * pi * flux_vals))
    d_pc = d_m / PC_TO_M

    # 6) Positions
    star_pos_pc = cam_pos_pc[None, :] + star_rays_world * d_pc[:, None]

    return star_pos_pc, star_lum_vals, star_spec_arr
