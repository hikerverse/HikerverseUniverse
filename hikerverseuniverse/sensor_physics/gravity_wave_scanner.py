import math

from hikerverseuniverse.library.constants import L0


class GravityWaveScanner:
    """
    Game-friendly gravity wave scanner.

    - sensitivity_h: characteristic strain that yields SNR=1 for 1s integration (dimensionless)
    - integration_time: seconds (SNR scales with sqrt(time))
    - freq_center_hz: center frequency of sensitivity (Hz)
    - freq_band_hz: approximate band width (Hz) for matching source frequency
    - chirp_factor: boosts SNR for coherent/chirping sources
    - background_h: ambient strain floor (dimensionless)
    - steepness: logistic steepness for detection_probability
    """

    def __init__(self, sensitivity_h: float = 1e-21, integration_time: float = 60.0,
                 freq_center_hz: float = 100.0, freq_band_hz: float = 200.0,
                 chirp_factor: float = 1.0, background_h: float = 1e-23, steepness: float = 1.0):
        self.sensitivity_h = float(sensitivity_h)
        self.integration_time = float(integration_time)
        self.freq_center_hz = float(freq_center_hz)
        self.freq_band_hz = float(freq_band_hz)
        self.chirp_factor = float(chirp_factor)
        self.background_h = float(background_h)
        self.steepness = float(steepness)

    def _intrinsic_strain_at_1pc(self, star) -> float:
        """
        Return an intrinsic reference strain at 1 pc (dimensionless).
        Prefer `star.gw_amplitude` if present (interpreted as strain at 1 pc).
        Fallbacks:
         - if star.mass (kg) provided, scale roughly with mass
         - else use luminosity-based heuristic
        """
        s = getattr(star, "gw_amplitude", None)
        if s is not None:
            return float(s)
        # mass-based heuristic
        M = getattr(star, "mass", None)
        if M is not None:
            M_sun = 1.98847e30
            return 1e-19 * ((M / M_sun) ** 0.7)
        # luminosity fallback
        L = getattr(star, "luminosity", None)
        if L is not None:
            return 1e-22 * ((L / L0) ** 0.3)
        # very faint generic fallback
        return 1e-24

    def _source_freq(self, star) -> float:
        """
        Estimate a GW central frequency for the source (Hz).
        Prefer `star.gw_freq` attribute, else fallback to a default.
        """
        f = getattr(star, "gw_freq", None)
        return float(f) if f is not None else 100.0

    def strain_at_distance(self, star, distance_m: float) -> float:
        """
        Estimate observed dimensionless strain at `distance_m`.
        Uses a simple 1/d scaling from the reference strain at 1 pc.
        """
        if distance_m <= 0:
            return 0.0
        intrinsic_at_1pc = self._intrinsic_strain_at_1pc(star)
        # convert distance to parsecs scale: strain scales ~ 1/d
        scale = (distance_m / pc)
        if scale <= 0:
            return 0.0
        received = intrinsic_at_1pc / scale
        # include background floor
        return float(max(self.background_h, received))

    def _band_match(self, star) -> float:
        """
        Simple Gaussian band match factor (0..1) between source freq and sensor band.
        """
        f_src = self._source_freq(star)
        hw = max(1.0, self.freq_band_hz / 2.0)
        diff = f_src - self.freq_center_hz
        # Gaussian-like match
        return math.exp(-0.5 * (diff / hw) ** 2)

    def snr(self, star, distance_m: float) -> float:
        """
        Compute a gamey SNR:
         - strain scales with 1/d (from `strain_at_distance`)
         - SNR ~ (strain / sensitivity) * sqrt(integration_time) * chirp_factor * band_match
        """
        strain = self.strain_at_distance(star, distance_m)
        if strain <= 0 or self.sensitivity_h <= 0:
            return 0.0
        match = self._band_match(star)
        snr = (strain / self.sensitivity_h) * math.sqrt(max(0.0, self.integration_time))
        snr *= (self.chirp_factor * match)
        return float(max(0.0, snr))

    def detection_probability(self, snr: float) -> float:
        """
        Logistic detection probability for game feel (centered near SNR ~ 5).
        """
        if snr <= 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-self.steepness * (snr - 5.0)))

