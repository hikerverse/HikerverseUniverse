import math

from hikerverseuniverse.library.constants import Kb, h, c, L0

# --- Presets ---
STAR_PRESETS = {
    "Sun": {"luminosity": L0, "temperature": 5778.0, "radio_luminosity": 1e13},
    "RedDwarf": {"luminosity": 0.01 * L0, "temperature": 3200.0, "radio_luminosity": 1e12},
    "Giant": {"luminosity": 100.0 * L0, "temperature": 4000.0, "radio_luminosity": 1e14},
}


class Star:
    """
    Simple star model with bolometric luminosity and radio spectral luminosity (W/Hz).
    - luminosity: total bolometric luminosity in W
    - temperature: photospheric temperature in K
    - radio_luminosity: spectral radio luminosity in W/Hz
    """

    def __init__(self, name: str, luminosity: float = L0, temperature: float = 5778.0,
                 radius: float = None, radio_luminosity: float = 0.0):
        self.name = name
        self.luminosity = float(luminosity)
        self.temperature = float(temperature)
        self.radius = radius
        self.radio_luminosity = float(radio_luminosity)

    @classmethod
    def from_preset(cls, preset_name: str):
        p = STAR_PRESETS.get(preset_name)
        if not p:
            raise ValueError("unknown preset")
        return cls(name=preset_name, luminosity=p["luminosity"],
                   temperature=p["temperature"], radio_luminosity=p["radio_luminosity"])

    def flux_at(self, distance_m: float) -> float:
        """Bolometric flux at distance in W/m^2 (inverse-square)."""
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        return self.luminosity / (4.0 * math.pi * distance_m * distance_m)

    def radio_flux_density(self, distance_m: float) -> float:
        """Radio flux density in W / (m^2 Hz) using inverse-square on spectral luminosity."""
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        return self.radio_luminosity / (4.0 * math.pi * distance_m * distance_m)

    def wien_peak_wavelength(self) -> float:
        """Approximate peak wavelength (m) using Wien's displacement law."""
        return 2.897771955e-3 / self.temperature

    def band_fraction(self, band_center_m: float, band_width_m: float, nsteps: int = 128) -> float:
        """
        Approximate fraction of bolometric flux in a simple band using a toy Planck integral.
        Fast trapezoidal integration of relative B_lambda shape. Game-friendly approximation.
        """
        lam_min = max(1e-9, band_center_m - band_width_m / 2.0)
        lam_max = band_center_m + band_width_m / 2.0

        def b_lambda_rel(lam: float) -> float:
            x = (h * c) / (lam * Kb * self.temperature)
            # avoid overflow and use expm1 for better small-x precision
            if x > 700:
                return 0.0
            denom = (lam ** 5) * math.expm1(x)
            return 1.0 / denom if denom > 0.0 else 0.0

        peak = self.wien_peak_wavelength()
        total_min = max(1e-9, peak * 0.01)
        total_max = peak * 30.0

        def integrate(f, a, b, steps):
            steps = max(4, int(steps))
            hstep = (b - a) / steps
            s = 0.5 * (f(a) + f(b))
            for i in range(1, steps):
                s += f(a + i * hstep)
            return s * hstep

        band = integrate(b_lambda_rel, lam_min, lam_max, max(8, nsteps // 8))
        total = integrate(b_lambda_rel, total_min, total_max, nsteps)
        frac = band / total if total > 0 else 0.0
        return max(0.0, min(1.0, frac))




