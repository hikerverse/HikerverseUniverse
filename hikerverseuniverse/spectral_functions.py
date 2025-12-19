# python
import math
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

from hikerverseuniverse.library.celestial_constants import morgan_keenan_spectral_class_ranges
from hikerverseuniverse.library.constants import Kb, h, c


def spectral_class_to_temperature(spectral: str) -> float:
    """
    Convert a spectral type like 'G2' or 'K7V' to approximate effective temperature in K.
    If subclass digit not found, uses midpoint for the class.
    """
    spectral = spectral.strip().upper()
    if not spectral:
        raise ValueError("Empty spectral class")

    class_letter = spectral[0]
    subclass = None
    # find first digit sequence after the letter
    for i in range(1, len(spectral)):
        if spectral[i].isdigit():
            j = i
            while j < len(spectral) and spectral[j].isdigit():
                j += 1
            subclass = int(spectral[i:j])
            break



    if class_letter not in morgan_keenan_spectral_class_ranges:
        raise ValueError(f"Unknown spectral class '{class_letter}'")

    tmin, tmax = morgan_keenan_spectral_class_ranges[class_letter]
    if subclass is None:
        frac = 0.5
    else:
        frac = max(0.0, min(9.0, subclass)) / 9.0

    # interpolate from hottest (subclass 0) to coolest (subclass 9)
    return tmin + (tmax - tmin) * (1 - frac)


def planck_lambda(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray:
    """
    Planck's law: spectral radiance per unit wavelength (relative units).
    wavelength_m: array of wavelengths in meters
    temperature_k: temperature in Kelvin
    Returns relative radiance (not absolute calibrated).
    """

    a = 2.0 * h * c * c
    wl5 = np.power(wavelength_m, 5)
    exponent = (h * c) / (wavelength_m * Kb * temperature_k)
    # prevent overflow for very small wavelengths
    exponent = np.clip(exponent, 1e-9, 700)
    intensity = a / (wl5 * (np.exp(exponent) - 1.0))
    return intensity


def color_temperature_to_rgb(temp_k: float, gamma: float = 0.8) -> Tuple[float, float, float]:
    """
    Approximate conversion from blackbody temperature (Kelvin) to sRGB.
    Returns tuple of floats in 0..1.
    Uses a widely used empirical approximation.
    """
    if temp_k < 1000:
        temp_k = 1000.0
    if temp_k > 40000:
        temp_k = 40000.0

    temp = temp_k / 100.0

    # Red
    if temp <= 66:
        red = 255.0
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)

    # Green
    if temp <= 66:
        green = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)

    # Blue
    if temp >= 66:
        blue = 255.0
    elif temp <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307

    def clamp_and_gamma(v: float) -> float:
        v = max(0.0, min(255.0, v))
        v = (v / 255.0) ** gamma
        return v

    return (clamp_and_gamma(red), clamp_and_gamma(green), clamp_and_gamma(blue))


def generate_star_spectrum(
        spectral_class: Optional[str] = None,
        temperature: Optional[float] = None,
        wl_min_nm: float = 100.0,
        wl_max_nm: float = 2500.0,
        n_points: int = 2000,
):
    """
    Generate a blackbody spectral curve for a star given spectral class or temperature.
    Returns (wavelengths_nm, normalized_flux, rgb_color).
    - spectral_class: e.g. 'G2V' or 'K7'
    - temperature: effective temperature in Kelvin (overrides spectral_class if provided)
    """
    if temperature is None:
        if spectral_class is None:
            raise ValueError("Provide either spectral_class or temperature")
        temperature = spectral_class_to_temperature(spectral_class)

    wavelengths_nm = np.linspace(wl_min_nm, wl_max_nm, n_points)
    wavelengths_m = wavelengths_nm * 1e-9
    flux = planck_lambda(wavelengths_m, temperature)

    # normalize for plotting
    flux_norm = flux / np.max(flux)

    rgb = color_temperature_to_rgb(temperature)

    return wavelengths_nm, flux_norm, rgb


def plot_star_spectrum(wavelengths_nm, flux_norm, rgb, title: str = "Star Spectrum"):
    """
    Plot the normalized spectrum and use the approximate star color as the plot background.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths_nm, flux_norm, color="black", lw=1.3)
    ax.set_xlim(wavelengths_nm[0], wavelengths_nm[-1])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title(title)

    # set background color as the star color but desaturated to keep grid readable
    bg = tuple(min(1.0, 0.9 * c + 0.1) for c in rgb)
    ax.set_facecolor(bg)

    # draw a small box showing the star color
    ax2 = fig.add_axes([0.82, 0.75, 0.12, 0.18])
    ax2.axis("off")
    ax2.set_facecolor(rgb)
    ax2.set_title("Color", fontsize=8)

    plt.tight_layout()
    plt.show()


def estimate_temperature_from_spectrum(
        wavelengths_nm: np.ndarray,
        flux: np.ndarray,
        temp_min: float = 1000.0,
        temp_max: float = 40000.0,
        n_grid: int = 200,
        refine_steps: int = 2,
        refine_width: float = 0.25,
) -> float:
    """
    Estimate effective temperature (K) by fitting a scaled blackbody to the input spectrum.
    - wavelengths_nm: 1D array of wavelengths in nanometers
    - flux: 1D array of flux (can be arbitrary relative units)
    Strategy: coarse log-spaced grid search over temperature, then refine around best T.
    """
    wl_m = np.asarray(wavelengths_nm, dtype=float) * 1e-9
    f = np.asarray(flux, dtype=float)
    # sanitize
    if f.size == 0 or wl_m.size == 0:
        raise ValueError("Empty wavelength or flux array")
    # remove non-finite
    mask = np.isfinite(wl_m) & np.isfinite(f) & (wl_m > 0)
    wl_m = wl_m[mask]
    f = f[mask]
    if wl_m.size == 0:
        raise ValueError("No valid data points after filtering")

    # normalize observed flux to mitigate scale
    eps = 1e-30
    f_norm = f / (np.max(f) + eps)

    def score_for_t_grid(ts: np.ndarray) -> Tuple[np.ndarray, float]:
        # returns (errors array, best_index)
        errs_ = np.empty(ts.shape)
        for i, T in enumerate(ts):
            model = planck_lambda(wl_m, T)
            model_norm = model / (np.max(model) + eps)
            # use log-MSE to reduce emphasis on absolute scale
            errs_[i] = np.mean((np.log(model_norm + eps) - np.log(f_norm + eps)) ** 2)
        return errs_, int(np.argmin(errs_))

    # coarse log-spaced grid
    grid = np.logspace(np.log10(max(100.0, temp_min)), np.log10(temp_max), n_grid)
    errs_, best_idx = score_for_t_grid(grid)
    best_t = float(grid[best_idx])

    # refinement steps
    for _ in range(refine_steps):
        lo = max(temp_min, best_t * (1 - refine_width))
        hi = min(temp_max, best_t * (1 + refine_width))
        grid = np.linspace(lo, hi, max(50, n_grid))
        errs_, best_idx = score_for_t_grid(grid)
        best_t = float(grid[best_idx])

    return best_t


def estimate_spectral_class_from_temperature(temp_k: float) -> str:
    """
    Map temperature (K) to a spectral class string like 'G2'.
    Uses the same approximate MK ranges as the module.
    """

    # find class whose range contains temp, else nearest by center distance
    chosen = None
    for letter, (tmin, tmax) in morgan_keenan_spectral_class_ranges.items():
        if tmin <= temp_k <= tmax:
            chosen = letter
            break

    if chosen is None:
        # pick nearest midpoint
        mids = {L: (rng[0] + rng[1]) / 2.0 for L, rng in morgan_keenan_spectral_class_ranges.items()}
        chosen = min(mids.keys(), key=lambda L: abs(mids[L] - temp_k))

    tmin, tmax = morgan_keenan_spectral_class_ranges[chosen]
    # invert interpolation used in `spectral_class_to_temperature`:
    # t = tmin + (tmax - tmin) * (1 - frac) where frac = subclass/9
    # => frac = 1 - (t - tmin)/(tmax - tmin)
    if tmax == tmin:
        frac = 0.5
    else:
        frac = 1.0 - (temp_k - tmin) / (tmax - tmin)
    frac = float(max(0.0, min(1.0, frac)))
    subclass = int(round(frac * 9.0))
    subclass = max(0, min(9, subclass))
    return f"{chosen}{subclass}"


def estimate_class_and_temperature(
        wavelengths_nm: np.ndarray,
        flux: np.ndarray,
        return_model: bool = True,
) -> Dict[str, Optional[object]]:
    """
    Convenience wrapper:
    - returns a dict with keys: 'temperature', 'spectral_class', 'model_flux_norm' (if requested)
    """
    temp_k = estimate_temperature_from_spectrum(wavelengths_nm, flux)
    spec_class = estimate_spectral_class_from_temperature(temp_k)

    model_flux_norm = None
    if return_model:
        wl_m = np.asarray(wavelengths_nm, dtype=float) * 1e-9
        model = planck_lambda(wl_m, temp_k)
        model_flux_norm = model / (np.max(model) + 1e-30)

    return {"temperature": temp_k, "spectral_class": spec_class, "model_flux_norm": model_flux_norm}


# Demo usage
if __name__ == "__main__":
    # Example 1: from spectral class
    wl, flux, color = generate_star_spectrum(spectral_class="G2V")
    plot_star_spectrum(wl, flux, color, title="G2 (Sun-like)")

    # Example 2: explicit temperature (hot O-type)
    wl2, flux2, color2 = generate_star_spectrum(temperature=30000)
    plot_star_spectrum(wl2, flux2, color2, title="30000 K (O-type)")

    # Example 2: explicit temperature (hot O-type)
    wl2, flux2, color2 = generate_star_spectrum(temperature=3000)
    plot_star_spectrum(wl2, flux2, color2, title="30000 K (O-type)")
