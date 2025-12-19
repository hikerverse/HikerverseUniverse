import math

from hikerverseuniverse.library.constants import L0


class MagneticScanner:
    """
    Game-friendly magnetic field sensor.

    - sensitivity_t: field (Tesla) that yields SNR=1 for 1s integration
    - integration_time: seconds (SNR scales with sqrt(time))
    - background_field_t: ambient/background field (Tesla)
    - detect_range: soft range (m) beyond which signals attenuate faster
    - steepness: logistic steepness for detection_probability
    """

    def __init__(self, sensitivity_t: float = 1e-12, integration_time: float = 60.0,
                 background_field_t: float = 1e-10, detect_range: float = 1e17,
                 steepness: float = 1.0):
        self.sensitivity_t = float(sensitivity_t)
        self.integration_time = float(integration_time)
        self.background_field_t = float(background_field_t)
        self.detect_range = float(detect_range)
        self.steepness = float(steepness)

    def _surface_field(self, star) -> float:
        # Prefer explicit attribute, else fallback scaled from luminosity
        b = getattr(star, "magnetic_field", None)
        if b is None:
            b = getattr(star, "surface_magnetic_field", None)
        if b is not None:
            return float(b)
        # gamey fallback: scale a fiducial surface field with luminosity
        return 1e-5 * ((getattr(star, "luminosity", 1.0) / L0) ** 0.2)

    def _radius(self, star) -> float:
        # prefer explicit radius, otherwise estimate from L = 4 pi R^2 sigma T^4
        if getattr(star, "radius", None):
            return float(star.radius)
        T = getattr(star, "temperature", None)
        L = getattr(star, "luminosity", L0)
        if T and T > 0:
            sigma = 5.670374419e-8
            try:
                return math.sqrt(L / (4.0 * math.pi * sigma * (T ** 4)))
            except Exception:
                pass
        # fallback to approximate solar radius
        return 6.957e8

    def field_at_distance(self, star, distance_m: float) -> float:
        """Estimate the net field magnitude (Tesla) at distance from object + background."""
        if distance_m <= 0:
            return 0.0
        B0 = abs(self._surface_field(star))
        R = self._radius(star)
        # dipole-like falloff: B ~ B0 * (R / d)^3 (clamp inside surface)
        if distance_m <= R:
            B_obj = B0
        else:
            B_obj = B0 * (R / distance_m) ** 3
        # soft extra attenuation beyond detect_range
        if distance_m > self.detect_range and self.detect_range > 0:
            excess = (distance_m - self.detect_range) / self.detect_range
            B_obj *= math.exp(-excess * 2.0)
        # combine object field and background magnitude-wise
        B_total = math.sqrt(B_obj * B_obj + (self.background_field_t ** 2))
        return float(B_total)

    def snr(self, star, distance_m: float) -> float:
        """Simple SNR model: (field / sensitivity) * sqrt(time)."""
        field = self.field_at_distance(star, distance_m)
        if self.sensitivity_t <= 0 or field <= 0:
            return 0.0
        return float((field / self.sensitivity_t) * math.sqrt(max(0.0, self.integration_time)))

    def detection_probability(self, snr: float) -> float:
        """Logistic detection curve centered around SNR ~ 4 for game feel."""
        if snr <= 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-self.steepness * (snr - 4.0)))