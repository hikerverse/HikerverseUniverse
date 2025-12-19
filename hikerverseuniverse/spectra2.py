# python
from typing import Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from hikerverseuniverse.spectral_functions import planck_lambda, estimate_class_and_temperature

# assumes `planck_lambda` and `spectral_class_to_temperature` are defined in this module



if __name__ == '__main__':
    # true star temperature (K)
    true_temp = 5800.0  # Sun-like (G-type)

    # wavelength grid in nanometers
    wavelengths_nm = np.linspace(100.0, 2500.0, 2000)
    wavelengths_m = wavelengths_nm * 1e-9

    # generate clean blackbody spectrum (relative units)
    clean_flux = planck_lambda(wavelengths_m, true_temp)

    # add realistic small noise and slight scaling
    rng = np.random.default_rng(12345)
    noise = 0.02 * rng.standard_normal(clean_flux.shape)
    observed_flux = clean_flux * (1.0 + noise)
    observed_flux = np.clip(observed_flux, 1e-30, None)

    # estimate temperature and spectral class
    result = estimate_class_and_temperature(wavelengths_nm, observed_flux, return_model=True)
    est_temp = result["temperature"]
    est_class = result["spectral_class"]
    model_flux_norm = result["model_flux_norm"]

    # print results
    print(f"True temperature: {true_temp:.0f} K")
    print(f"Estimated temperature: {est_temp:.0f} K")
    print(f"Estimated spectral class: {est_class}")

    # normalize observed for plotting
    obs_norm = observed_flux / np.max(observed_flux)

    # plot observed and fitted model
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths_nm, obs_norm, color="tab:blue", label="Observed (noisy)")
    if model_flux_norm is not None:
        plt.plot(wavelengths_nm, model_flux_norm, color="tab:red", lw=1.5, label=f"Blackbody fit ({est_temp:.0f} K)")
    plt.xlim(wavelengths_nm[0], wavelengths_nm[-1])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized intensity")
    plt.title(f"Estimated: {est_class}, {est_temp:.0f} K")
    plt.legend()
    plt.tight_layout()
    plt.show()

