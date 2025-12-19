
class RadioTelescope:
    """
    Radio telescope simplified radiometer model.
    - effective_area: m^2
    - system_temperature: K
    - bandwidth: Hz
    - integration_time: s
    - observing_frequency: Hz (informational)
    - sensitivity_floor_jy: optional floor for SEFD in Jy (game guard)
    """

    def __init__(self, effective_area: float, system_temperature: float = 50.0,
                 bandwidth: float = 1e7, integration_time: float = 60.0,
                 observing_frequency: float = 1.4e9, sensitivity_floor_jy: float = 1e-6):
        self.A_eff = float(effective_area)
        self.T_sys = float(system_temperature)
        self.bandwidth = float(bandwidth)
        self.integration_time = float(integration_time)
        self.freq = float(observing_frequency)
        self.sensitivity_floor_jy = float(sensitivity_floor_jy)

    def sefd_jy(self) -> float:
        """
        The selected code defines the sefd_jy method, which calculates the System Equivalent Flux Density (SEFD) of a
        radio telescope in Jansky (Jy). SEFD is a measure of the sensitivity of a radio telescope, representing
        the flux density of a source that would produce the same power as the system noise.
        SEFD = 2 * k * Tsys / A_eff (W/m^2/Hz), converted to Jansky.
        """
        if self.A_eff <= 0:
            return float('inf')
        sefd_w = 2.0 * Kb * self.T_sys / self.A_eff
        return sefd_w / 1e-26

    def snr(self, star_or_flux, distance_m: float = None) -> float:
        """
        Compute SNR either from a Star+distance or from a direct flux density (W/m^2/Hz).
        - If `distance_m` is provided and `star_or_flux` is a Star, compute flux_density.
        - Otherwise treat `star_or_flux` as flux_density_w_m2_hz.
        """
        if isinstance(star_or_flux, Star):
            if distance_m is None:
                raise ValueError("distance_m required when passing a Star")
            flux_density = star_or_flux.radio_flux_density(distance_m)
        else:
            flux_density = float(star_or_flux)

        if flux_density <= 0:
            return 0.0
        s_jy = flux_density / 1e-26
        sefd = max(self.sefd_jy(), self.sensitivity_floor_jy)
        if sefd <= 0:
            return 0.0
        return (s_jy / sefd) * math.sqrt(self.bandwidth * self.integration_time)

    def detection_probability(self, snr: float) -> float:
        """
        The selected code defines the detection_probability method, which calculates the probability of detecting a signal based on its signal-to-noise ratio (SNR). This method uses a mathematical model to map the SNR to a probability value between 0 and 1, providing a smooth transition from low to high probabilities as the SNR increases.
The method first checks if the provided snr is greater than 0. If not, it immediately returns 0.0, indicating no detection probability for non-positive SNR values:
        :param snr:
        :return:
        """
        return 1.0 - math.exp(-(snr / 4.0) ** 1.8) if snr > 0 else 0.0

    """
    Typical SNR values depend on instrument and use case; treat them as rough guides:
    - Optical (photon counting / imaging): detection thresholds ~5 (survey); useful photometry ~10–30; high‑quality photometry/astrometry >100; very bright sources can give SNRs in the 1e3+ range.
    - Radio (radiometer / continuum): survey/claim thresholds ~5–10; targeted detections typically >10–50; bright sources (or long integrations/wide bandwidth) can reach 1e2–1e6+ (your logs show very large values).
    - Gravitational waves: network SNR ~8 is commonly used as a detection floor; confident detections often >12; loud events can reach tens or more.
    - Magnetic sensors: detection threshold typically ~3–5; reliable measurements >10.
    - Gamey / synthetic sensors (Subspace, scanner): arbitrary units — in this code the logistic curves are centered near SNR ≈ 4–5, so treat that region as the mid‑point for detection probability.

    Also note: required SNR scales with sqrt(integration_time) and with bandwidth/sensitivity; many independent trials or a stricter false‑alarm probability (p_fa) require higher SNR to maintain the same confidence.

    Typical values for (p_fa) depend on the application and the tolerated trade-off between false alarms and missed detections:
    Casual/gamey or exploratory pipelines: ~1e-2 ... 1e-3 (lenient).
    Common signal‑detection / astronomy thresholds: ~1e-3 ... 1e-6.
    Strict experiments (particle physics, high‑confidence claims): ~1e-6 ... 1e-12 or tighter; the canonical "5σ" corresponds to ~2.9e-7.
    For GW / survey work people often quote a false-alarm rate (e.g. 1 per year) and convert to an equivalent per‑trial (p_fa) given the number of trials.

    Tau is the detection threshold (in the same units as SNR, i.e., roughly "sigma") chosen so the Gaussian tail probability equals the desired false‑alarm probability. In the code tau is computed from the false‑alarm probability p_fa using the inverse complementary error function so that
    p_fa = Q(tau) where Q is the Gaussian tail (complementary CDF), and
    tau is the corresponding threshold in sigma units.
    """

    def detection_probability_realistic(self, snr: float, p_fa: float = 1e-3) -> float:
        if snr <= 0:
            return 0.0
        # Q(x) = 0.5 * erfc(x / sqrt(2)); invert to get threshold tau for given P_FA
        tau = math.sqrt(2.0) * math.erfcinv(2.0 * p_fa)
        # Detection probability P_D = Q(tau - SNR)
        return 0.5 * math.erfc((tau - snr) / math.sqrt(2.0))