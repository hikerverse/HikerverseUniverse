



class SubspaceScanner:
    """
    Sci\-fi 'subspace' sensor for game use.

    - sensitivity: game units of flux required for SNR=1
    - integration_time: seconds (increases SNR with sqrt(time) as a game simplification)
    - range_max: soft range in meters; signals beyond this suffer additional attenuation
    - coherence: [0..1] boosts effective signal if the target emits coherent subspace waves
    - steepness: detection curve steepness for probability mapping
    """

    def __init__(self, sensitivity: float = 1.0, integration_time: float = 60.0,
                 range_max: float = 1e17, coherence: float = 0.8, steepness: float = 1.0):
        self.sensitivity = float(sensitivity)
        self.integration_time = float(integration_time)
        self.range_max = float(range_max)
        self.coherence = float(max(0.0, min(1.0, coherence)))
        self.steepness = float(steepness)

    def _target_subspace_power(self, star) -> float:
        """
        Return an intrinsic 'subspace' power for a star (arbitrary units).
        Prefer `star.subspace_signal` if present, otherwise fall back to `star.radio_luminosity * 1e-6`.
        """
        p = getattr(star, "subspace_signal", None)
        if p is None:
            # fallback: use radio_luminosity as a proxy (scaled to subspace units)
            p = getattr(star, "radio_luminosity", 0.0) * 1e-6
        return float(p)

    def snr(self, star, distance_m: float) -> float:
        """
        Compute a gamey SNR:
        - intrinsic subspace power spreads with inverse square
        - apply coherence boost, range falloff, and sqrt(integration_time)
        - normalize by sensor sensitivity
        """
        if distance_m <= 0:
            return 0.0
        power = self._target_subspace_power(star)
        # inverse-square propagation
        received = power / (4.0 * math.pi * distance_m * distance_m)
        # coherence multiplies effective received power
        received *= (1.0 + self.coherence)
        # soft range attenuation beyond range_max
        if distance_m > self.range_max and self.range_max > 0:
            excess = (distance_m - self.range_max) / self.range_max
            # stronger attenuation the farther beyond range_max
            received *= math.exp(-excess * 3.0)
        # integrate and normalize to sensitivity
        if self.sensitivity <= 0:
            return 0.0
        snr = (received / self.sensitivity) * math.sqrt(self.integration_time)
        return float(max(0.0, snr))

    def detection_probability(self, snr: float) -> float:
        """
        Map SNR to detection probability with a logistic curve for game feel.
        """
        if snr <= 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-self.steepness * (snr - 4.0)))

