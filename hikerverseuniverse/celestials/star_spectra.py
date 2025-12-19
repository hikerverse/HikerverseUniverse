


import numpy as np
import matplotlib.pyplot as plt


def estimate_light_spectrum(temperature: float, wavelengths: np.ndarray,
                            line_strengths: dict, noise_level: float = 0.0, noise_normalized=True) -> np.ndarray:

    # Constants
    h = 6.626e-34  # Planck's constant (Joule-seconds)
    c = 3.0e8      # Speed of light (m/s)
    k = 1.38e-23   # Boltzmann constant (Joule/Kelvin)

    # Calculate the blackbody radiation spectrum using Planck's law
    spectrum = (2 * h * c**2) / (wavelengths**5) / (np.exp((h * c) / (wavelengths * k * temperature)) - 1)

    # Add spectral lines by modifying the spectrum at specific wavelengths
    for line_wavelength, strength in line_strengths.items():
        line_index = np.argmin(np.abs(wavelengths - line_wavelength))
        spectrum[line_index] *= (1 - strength)  # Absorption line (reduce intensity)


    # Add noise to the spectrum
    if noise_level > 0:
        if noise_normalized:
            max_intensity = np.max(spectrum)
        else:
            max_intensity = spectrum
        noise = np.random.normal(0, noise_level * max_intensity, size=spectrum.shape)
        spectrum += noise

    return spectrum


def wavelength_to_rgb(wavelength):
    """
    Convert a wavelength in the visible spectrum (380-750 nm) to an RGB color.
    :param wavelength: Wavelength in nanometers.
    :return: Tuple of (R, G, B) values normalized to [0, 1].
    """
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0

    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 750:
        R = 1.0
        G = 0.0
        B = 0.0

    # Adjust intensity
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength < 645:
        factor = 1.0
    elif 645 <= wavelength <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)

    R = ((R * factor) ** gamma) * intensity_max
    G = ((G * factor) ** gamma) * intensity_max
    B = ((B * factor) ** gamma) * intensity_max

    return R / 255, G / 255, B / 255


if __name__ == '__main__':
    # Define the temperature of the star (in Kelvin)
    temperature = 6000  # Example: Sun-like star

    # Define the range of wavelengths (in meters)
    wavelengths = np.linspace(1e-7, 3e-6, 1000)  # From 100 nm to 3000 nm

    # Define spectral lines (wavelengths in meters and their strengths)
    line_strengths = {
        3.93e-7: 0.3,  # Calcium K line (393 nm)
        3.96e-7: 0.3,  # Calcium H line (396 nm)
        4.30e-7: 0.2,  # Hydrogen gamma line (430 nm)
        4.86e-7: 0.2,  # Hydrogen beta line (486 nm)
        5.17e-7: 0.1,  # Magnesium triplet (517 nm)
        5.89e-7: 0.4,  # Sodium D line (589 nm)
        6.56e-7: 0.2,  # Hydrogen alpha line (656 nm)
        7.60e-7: 0.1   # Potassium line (760 nm)
    }

    fraunhofer_lines_all = {
        "Hydrogen": {
            "H-alpha": 6.56e-7,  # 656 nm
            "H-beta": 4.86e-7,  # 486 nm
            "H-gamma": 4.34e-7,  # 434 nm
            "H-delta": 4.10e-7  # 410 nm
        },
        "Helium": {
            "He-I": 5.87e-7,  # 587 nm
            "He-II": 4.68e-7  # 468 nm
        },
        "Calcium": {
            "Ca-K": 3.93e-7,  # 393 nm
            "Ca-H": 3.96e-7  # 396 nm
        },
        "Sodium": {
            "Na-D1": 5.89e-7,  # 589 nm
            "Na-D2": 5.89e-7  # 589 nm (slightly shifted)
        },
        "Magnesium": {
            "Mg-b1": 5.18e-7,  # 518 nm
            "Mg-b2": 5.17e-7,  # 517 nm
            "Mg-b3": 5.16e-7  # 516 nm
        },
        "Iron": {
            "Fe-I": 4.37e-7,  # 437 nm
            "Fe-II": 4.58e-7  # 458 nm
        },
        "Oxygen": {
            "O-I": 7.77e-7,  # 777 nm
            "O-II": 4.65e-7  # 465 nm
        },
        "Titanium Oxide": {
            "TiO-1": 7.05e-7,  # 705 nm
            "TiO-2": 7.60e-7  # 760 nm
        },
        "Potassium": {
            "K-I": 7.69e-7  # 769 nm
        }
    }

    # Call the function to estimate the spectrum
    spectrum = estimate_light_spectrum(temperature, wavelengths, line_strengths, noise_level=1e-2)

    # Plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths * 1e9, spectrum, label=f"Temperature: {temperature} K")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Estimated Light Spectrum with Spectral Lines")
    plt.legend()
    plt.grid()
    plt.show()

    # Image dimensions
    width = 512
    height = 75

    # Generate the image
    image = np.ones((height, width, 3))

    # Map each column to a wavelength in the visible spectrum
    wavelengths = np.linspace(380, 750, width)
    for x in range(width):
        color = wavelength_to_rgb(wavelengths[x])
        image[:, x, :] = color

    # Add vertical lines at the specified wavelengths
    for line_wavelength in line_strengths.keys():
        if 380 <= line_wavelength * 1e9 <= 750:  # Convert meters to nanometers and check range
            x_position = int((line_wavelength * 1e9 - 380) / (750 - 380) * width)
            if 0 <= x_position < width:
                image[:, x_position, :] = [0,0,0]  # White vertical line


    # Display the image
    plt.figure(figsize=(12, 2))
    plt.imshow(image, extent=[380, 750, 0, height])
    plt.axis("off")
    plt.show()
