import numpy as np


class StarField:
    def __init__(self, star_positions: np.ndarray, star_temperatures: np.ndarray, star_luminosities: np.ndarray,
                 star_radii: np.ndarray):
        self.star_positions = star_positions  # (N, 3) in parsecs
        self.star_temperatures = star_temperatures  # (N,) in Kelvin
        self.star_luminosities = star_luminosities  # (N,) in Watts
        self.star_radii = star_radii  # (N,) in Watts
