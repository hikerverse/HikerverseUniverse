




def hms_to_deg(h: float, m: float = 0.0, s: float = 0.0) -> float:
    """Convert signed degrees given as (h, m, s) to decimal degrees."""
    sign = 1 if h >= 0 else -1
    return sign * (abs(h) + m / 60.0 + s / 3600.0)


def ra_to_deg(h: float, m: float = 0.0, s: float = 0.0) -> float:
    """Convert right ascension (hours, minutes, seconds) to decimal degrees."""
    sign = 1 if h >= 0 else -1
    return 15.0 * sign * (abs(h) + m / 60.0 + s / 3600.0)



def galactic_coordinates_to_absolute_coordinates(right_ascension: float, declination: float, distance: float) -> tuple[float, float, float]:
    """Convert galactic coordinates to absolute Cartesian coordinates.
    right_ascension and declination are in degrees; distance is in the same units used for the returned coordinates.
    Returns (x, y, z).
    """
    import math

    # Solar offsets (adjust to your project's unit system if needed)
    x_sol = 0.0
    y_sol = -8.0   # placeholder: -8 (e.g. kpc) — keep units consistent with `distance`
    z_sol = 100.0  # placeholder: 100 (e.g. ly)

    ra_rad = math.radians(right_ascension)
    dec_rad = math.radians(declination)

    x = distance * math.sin(ra_rad) + x_sol
    y = distance * math.cos(ra_rad) + y_sol
    z = distance * math.sin(dec_rad) + z_sol

    return (x, y, z)
