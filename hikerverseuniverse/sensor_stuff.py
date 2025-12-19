# python
"""
File: `sensor_stuff.py`
Self-contained combined library with examples for game use:
- constants
- STAR_PRESETS + Star model
- OpticalSensor
- RadioTelescope
- SurveySimulator
- utilities: flux_to_mag_ab, detect_optical, detect_radio
- example runners: run_example, run_simple, run_survey, show_presets
"""

import math
import random
import pprint
from typing import Dict, List, Tuple

from hikerverseuniverse.celestials.star import Star, STAR_PRESETS
from hikerverseuniverse.library.constants import L0, Kb, pc, AU, h, c
from hikerverseuniverse.sensor_physics.gravity_wave_scanner import GravityWaveScanner
from hikerverseuniverse.sensor_physics.magneto_scanner import MagneticScanner
from hikerverseuniverse.sensor_physics.optical_sensor_implementation import OpticalSensor
from hikerverseuniverse.sensor_physics.radio_sensor import RadioTelescope
from hikerverseuniverse.sensor_physics.subspace_scanner import SubspaceScanner

# --- Constants (self-contained) ---
#c = 299_792_458.0  # speed of light, m/s
#h = 6.62607015e-34  # Planck constant, J*s
#Kb = 1.380649e-23  # Boltzmann constant, J/K
#L0 = 3.828e26  # approximate solar luminosity, W
#AU = 149_597_870_700.0  # astronomical unit, m
#pc = 3.085677581491367e16  # parsec, m












class SurveySimulator:
    """
    Simple survey runner for the game:
    - sensors: list of sensors (OpticalSensor or RadioTelescope)
    - random_seed: optional for reproducibility
    """

    def __init__(self, sensors: List, random_seed: int = None):
        self.sensors = list(sensors)
        if random_seed is not None:
            random.seed(random_seed)

    def scan_target(self, star: Star, distance_m: float, pointing_offset_deg: float = 0.0) -> Dict:
        results = {}
        for s in self.sensors:
            if isinstance(s, OpticalSensor):
                snr = s.snr(star, distance_m, pointing_offset_deg)
                p = s.detection_probability(snr)
                detected = random.random() < p
                results[f"optical_{id(s)}"] = {"snr": snr, "prob": p, "detected": detected}
            elif isinstance(s, RadioTelescope):
                snr = s.snr(star, distance_m)
                p = s.detection_probability(snr)
                detected = random.random() < p
                results[f"radio_{id(s)}"] = {"snr": snr, "prob": p, "detected": detected}
            elif isinstance(s, SubspaceScanner):
                snr = s.snr(star, distance_m)
                p = s.detection_probability(snr)
                detected = random.random() < p
                results[f"subspace_{id(s)}"] = {"snr": snr, "prob": p, "detected": detected}
            elif isinstance(s, MagneticScanner):
                snr = s.snr(star, distance_m)
                p = s.detection_probability(snr)
                detected = random.random() < p
                results[f"magnetic_{id(s)}"] = {"snr": snr, "prob": p, "detected": detected}
            elif isinstance(s, GravityWaveScanner):
                snr = s.snr(star, distance_m)
                p = s.detection_probability(snr)
                detected = random.random() < p
                results[f"gw_{id(s)}"] = {"snr": snr, "prob": p, "detected": detected}
            else:
                results[f"sensor_{id(s)}"] = {"error": "unknown sensor type"}
        return results

    def scan_distances(self, star: Star, distances: List[float], pointing_offset_deg: float = 0.0) -> List[
        Tuple[float, Dict]]:
        out = []
        for d in distances:
            out.append((d, self.scan_target(star, d, pointing_offset_deg)))
        return out


# --- Utility functions ---
def flux_to_mag_ab(flux_w_m2: float, band_center_m: float) -> float:
    """
    Crude conversion of W/m^2 in a band to an approximate AB magnitude (gamey).
    This assumes a guessed narrow-band bandwidth for labeling only.
    """
    if flux_w_m2 <= 0:
        return 99.0
    approx_bandwidth = 1e14  # gamey guess
    s_jy = (flux_w_m2 / approx_bandwidth) / 1e-26
    if s_jy <= 0:
        return 99.0
    return -2.5 * math.log10(s_jy / 3631.0)


def detect_optical(sensor: OpticalSensor, star: Star, distance_m: float, snr_threshold: float = 5.0) -> Dict:
    snr = sensor.snr(star, distance_m)
    return {"snr": snr, "detected": snr >= snr_threshold}


def detect_radio(sensor: RadioTelescope, star: Star, distance_m: float, snr_threshold: float = 5.0) -> Dict:
    snr = sensor.snr(star, distance_m)
    return {"snr": snr, "detected": snr >= snr_threshold}


# --- Example runners (game/demo) ---
def run_example():
    sun = Star(name="Sun", luminosity=L0, temperature=5778.0, radio_luminosity=1e13)

    optical = OpticalSensor(aperture_diameter=1.0, throughput=0.8, qe=0.9,
                            band_center_m=550e-9, band_width_m=100e-9,
                            integration_time=60.0, background_photons_per_s=100.0,
                            read_noise_e=5.0, fov_deg=1.0)

    radio = RadioTelescope(effective_area=1000.0, system_temperature=50.0,
                           bandwidth=1e7, integration_time=60.0, observing_frequency=1.4e9)

    distances = [AU, pc, 1000 * pc]
    for d in distances:
        opt_res = detect_optical(optical, sun, d)
        rad_res = detect_radio(radio, sun, d)
        print(f"Distance: {d:.3e} m")
        print(f"  Optical SNR: {opt_res['snr']:.3f}, detected: {opt_res['detected']}")
        print(f"  Radio SNR:   {rad_res['snr']:.3e}, detected: {rad_res['detected']}")
        print("")


def run_simple():
    sun = Star.from_preset("Sun")

    optical = OpticalSensor(aperture_diameter=1.0, throughput=0.8, qe=0.9,
                            band_center_m=550e-9, band_width_m=100e-9,
                            integration_time=60.0, background_photons_per_s=100.0,
                            read_noise_e=5.0, fov_deg=1.0)

    radio = RadioTelescope(effective_area=1000.0, system_temperature=50.0,
                           bandwidth=1e7, integration_time=60.0, observing_frequency=1.4e9)

    distances = [AU, pc, 1000 * pc]
    for d in distances:
        snr_opt = optical.snr(sun, d, pointing_offset_deg=0.0)
        snr_rad = radio.snr(sun, d)
        flux = sun.flux_at(d)
        mag = flux_to_mag_ab(flux * sun.band_fraction(optical.band_center_m, optical.band_width_m),
                             optical.band_center_m)
        print(f"Distance: {d:.3e} m")
        print(
            f"  Optical SNR: {snr_opt:.3f}, detection_prob: {optical.detection_probability(snr_opt):.3f}, mag: {mag:.2f}")
        print(f"  Radio SNR:   {snr_rad:.3e}, detection_prob: {radio.detection_probability(snr_rad):.3f}")
        print("")


def run_survey():
    small_opt = OpticalSensor(aperture_diameter=0.2, band_center_m=600e-9, band_width_m=120e-9,
                              integration_time=30.0, background_photons_per_s=50.0, fov_deg=2.0)
    large_opt = OpticalSensor(aperture_diameter=2.0, band_center_m=550e-9, band_width_m=100e-9,
                              integration_time=120.0, background_photons_per_s=10.0, fov_deg=0.5)
    radio = RadioTelescope(effective_area=2000.0, bandwidth=2e7, integration_time=300.0)

    sim = SurveySimulator([small_opt, large_opt, radio], random_seed=42)

    targets = [
        (Star.from_preset("RedDwarf"), 5 * pc),
        (Star.from_preset("Sun"), 10 * pc),
        (Star.from_preset("Giant"), 50 * pc),
    ]

    for star, dist in targets:
        print(f"--- Scanning {star.name} at {dist:.3e} m ---")
        res = sim.scan_target(star, dist, pointing_offset_deg=0.2)
        pprint.pprint(res)
        print("")


def show_presets():
    optical = OpticalSensor(aperture_diameter=0.5, band_center_m=700e-9, band_width_m=150e-9,
                            integration_time=60.0, background_photons_per_s=20.0)

    dist = 1.0 * pc
    print("Preset summary at 1 pc:")
    for name in STAR_PRESETS:
        s = Star.from_preset(name)
        flux = s.flux_at(dist) * s.band_fraction(optical.band_center_m, optical.band_width_m)
        mag = flux_to_mag_ab(flux, optical.band_center_m)
        snr = optical.snr(s, dist)
        print(f"  {name}: mag ~ {mag:.2f}, optical SNR ~ {snr:.2f}")


if __name__ == "__main__":
    # run a few demos
    run_example()
    run_survey()
    run_simple()
    show_presets()
