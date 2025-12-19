# python
import math
from hikerverseuniverse.library.constants import c  # re-use project light speed constant if available

# SI constants kept local to avoid depending on other constant names
G = 6.67430e-11          # gravitational constant, m^3 kg^-1 s^-2
M_SUN = 1.98847e30       # solar mass, kg
M_PROTON = 1.67262192369e-27
SIGMA_T = 6.6524587158e-29  # Thomson cross-section, m^2

class BlackHole:
    """
    Simple black hole model.
    - mass_kg: black hole mass in kg
    - spin: dimensionless spin parameter (0..1), informational only
    - accretion_efficiency: radiative efficiency (fraction of m c^2 converted to radiation)
    - radio_luminosity: spectral radio luminosity in W/Hz (optional)
    """
    def __init__(self, name: str, mass_kg: float, spin: float = 0.0,
                 accretion_efficiency: float = 0.1, radio_luminosity: float = 0.0):
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if not (0.0 <= spin <= 1.0):
            raise ValueError("spin must be between 0 and 1")
        if accretion_efficiency < 0.0:
            raise ValueError("accretion_efficiency must be non-negative")

        self.name = name
        self.mass_kg = float(mass_kg)
        self.spin = float(spin)
        self.accretion_efficiency = float(accretion_efficiency)
        self.radio_luminosity = float(radio_luminosity)

    @classmethod
    def from_solar_mass(cls, name: str, mass_msun: float, **kwargs):
        """Construct from mass in solar masses."""
        return cls(name=name, mass_kg=mass_msun * M_SUN, **kwargs)

    def schwarzschild_radius(self) -> float:
        """Schwarzschild radius (m): R_s = 2 G M / c^2."""
        return 2.0 * G * self.mass_kg / (c * c)

    def eddington_luminosity(self) -> float:
        """
        Eddington luminosity (W):
        L_Edd = 4 * pi * G * M * m_p * c / sigma_T
        """
        return 4.0 * math.pi * G * self.mass_kg * M_PROTON * c / SIGMA_T

    def accretion_luminosity(self, mdot_kg_s: float) -> float:
        """Luminosity from mass accretion rate (W): L = eta * mdot * c^2."""
        if mdot_kg_s < 0:
            raise ValueError("mdot_kg_s must be non-negative")
        return self.accretion_efficiency * mdot_kg_s * (c * c)

    def flux_at(self, distance_m: float, luminosity_w: float | None = None) -> float:
        """
        Bolometric flux at distance (W/m^2). If `luminosity_w` is None the method
        returns the flux for the Eddington luminosity as a conservative reference.
        """
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        L = luminosity_w if luminosity_w is not None else self.eddington_luminosity()
        return L / (4.0 * math.pi * distance_m * distance_m)

    def radio_flux_density(self, distance_m: float) -> float:
        """Radio flux density in W / (m^2 Hz) from `radio_luminosity` using inverse-square."""
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        return self.radio_luminosity / (4.0 * math.pi * distance_m * distance_m)


if __name__ == '__main__':

    # unit
    pc = 3.085677581e16  # m
    kpc = 1e3 * pc
    Mpc = 1e6 * pc
    JY = 1e-26  # W / (m^2 Hz)

    examples = [
        # (name, mass_Msun, distance_m, radio_luminosity_W_per_Hz)
        ("Sgr A*", 4.3e6, 8.122 * kpc, 1e16),  # Galactic center, radio L_nu ~1e16 W/Hz (approx)
        ("M87*", 6.5e9, 16.4 * Mpc, 1e25),  # M87*, powerful radio jet (illustrative)
        ("Cygnus X-1", 21.0, 1.86 * kpc, 1e15),  # XRB, small radio output (illustrative)
        ("GW150914_remnant", 62.0, 410 * Mpc, 0.0),  # GW source remnant, no radio assumed
    ]

    for name, mass_msun, dist_m, radio_L in examples:
        bh = BlackHole.from_solar_mass(name=name, mass_msun=mass_msun, radio_luminosity=radio_L)
        Rs = bh.schwarzschild_radius()
        Ledd = bh.eddington_luminosity()
        flux = bh.flux_at(dist_m)  # W / m^2 (bolometric, using L_Edd by default)
        radio_flux = bh.radio_flux_density(dist_m)  # W / (m^2 Hz)
        radio_jy = radio_flux / JY

        print(f"{name}:")
        print(f"  mass = {mass_msun:.3g} Msun")
        print(f"  distance = {dist_m:.3g} m")
        print(f"  Schwarzschild radius = {Rs:.3e} m")
        print(f"  Eddington luminosity = {Ledd:.3e} W")
        print(f"  bolometric flux at Earth ~ {flux:.3e} W/m^2")
        print(f"  radio flux density ~ {radio_flux:.3e} W/(m^2 Hz) = {radio_jy:.3e} Jy")
        print()