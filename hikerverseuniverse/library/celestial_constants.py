import math

from hikerverseuniverse.library import unit, constants

# temperature ranges per Morgan-Keenan spectral classes (approx)
morgan_keenan_spectral_class_ranges = {
    "O": (30000, 50000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}

MBOL_SUN = 4.75

O_STAR_RADIUS = 695500 * unit.kilometer
B_STAR_RADIUS = 695500 * unit.kilometer
A_STAR_RADIUS = 695500 * unit.kilometer
F_STAR_RADIUS = 695500 * unit.kilometer

G2V_STAR_RADIUS = 695500 * unit.kilometer

K_STAR_RADIUS = 695500 * unit.kilometer
M_STAR_RADIUS = 695500 * unit.kilometer

M0V_STAR_RADIUS = constants.R0 * 0.62
M1V_STAR_RADIUS = constants.R0 * 0.49
M2V_STAR_RADIUS = constants.R0 * 0.44
M3V_STAR_RADIUS = constants.R0 * 0.39
M4V_STAR_RADIUS = constants.R0 * 0.26
M5V_STAR_RADIUS = constants.R0 * 0.20
M6V_STAR_RADIUS = constants.R0 * 0.15
M7V_STAR_RADIUS = constants.R0 * 0.12
M8V_STAR_RADIUS = constants.R0 * 0.11
M9V_STAR_RADIUS = constants.R0 * 0.08

G0V_STAR_MASS = constants.M0 * 1.15
G1V_STAR_MASS = constants.M0 * 1.10
G2V_STAR_MASS = constants.M0 * 1.07
G3V_STAR_MASS = constants.M0 * 1.04
G4V_STAR_MASS = constants.M0 * 1.00
G5V_STAR_MASS = constants.M0 * 0.98
G6V_STAR_MASS = constants.M0 * 0.93
G7V_STAR_MASS = constants.M0 * 0.90
G8V_STAR_MASS = constants.M0 * 0.87
G9V_STAR_MASS = constants.M0 * 0.84

M0V_STAR_MASS = constants.M0 * 0.6
M1V_STAR_MASS = constants.M0 * 0.49
M2V_STAR_MASS = constants.M0 * 0.44
M3V_STAR_MASS = constants.M0 * 0.36
M4V_STAR_MASS = constants.M0 * 0.20
M5V_STAR_MASS = constants.M0 * 0.14
M6V_STAR_MASS = constants.M0 * 0.10
M7V_STAR_MASS = constants.M0 * 0.09
M8V_STAR_MASS = constants.M0 * 0.08
M9V_STAR_MASS = constants.M0 * 0.075

G0V_STAR_TEMP = 5980
G1V_STAR_TEMP = 5900
G2V_STAR_TEMP = 5800
G3V_STAR_TEMP = 5710
G4V_STAR_TEMP = 5690
G5V_STAR_TEMP = 5620
G6V_STAR_TEMP = 5570
G7V_STAR_TEMP = 5500
G8V_STAR_TEMP = 5450
G9V_STAR_TEMP = 5370

M0V_STAR_TEMP = 3800
M1V_STAR_TEMP = 3600
M2V_STAR_TEMP = 3400
M3V_STAR_TEMP = 3250
M4V_STAR_TEMP = 3100
M5V_STAR_TEMP = 2800
M6V_STAR_TEMP = 2600
M7V_STAR_TEMP = 2500
M8V_STAR_TEMP = 2400
M9V_STAR_TEMP = 2300

G2V_STAR_LUMINOSITY = 3.846E26  # * unit.W

M0V_STAR_LUMINOSITY = 0.072 * constants.L0
M1V_STAR_LUMINOSITY = 0.035 * constants.L0
M2V_STAR_LUMINOSITY = 0.023 * constants.L0
M3V_STAR_LUMINOSITY = 0.015 * constants.L0
M4V_STAR_LUMINOSITY = 0.0055 * constants.L0
M5V_STAR_LUMINOSITY = 0.0022 * constants.L0
M6V_STAR_LUMINOSITY = 0.0009 * constants.L0
M7V_STAR_LUMINOSITY = 0.0005 * constants.L0
M8V_STAR_LUMINOSITY = 0.0003 * constants.L0
M9V_STAR_LUMINOSITY = 0.00015 * constants.L0

# SOLAR constants
Lsun = 3.839e33  # erg s-1
Rsun = 6.955e10  # cm

# PLANETARY constants
massMerc = 3.303e26 * unit.kg
massVenus = 4.870e27 * unit.kg
massEarth = 5.976e27 * unit.kg
massMars = 6.418e26 * unit.kg
massJup = 1.899e30 * unit.kg
massSaturn = 5.686e29 * unit.kg
massUranus = 8.66e28 * unit.kg
massNeptune = 1.030e29 * unit.kg
massPluto = 1.0e25 * unit.kg

radMerc = 2.439e8 * unit.km
radVenus = 6.050e8 * unit.km
radEarth = 6.378e8 * unit.km
radMars = 3.397e8 * unit.km
radJup = 7.140e9 * unit.km
radSaturn = 6.0e9 * unit.km
radUranus = 2.615e9 * unit.km
radNeptune = 2.43e9 * unit.km
radPluto = 1.2e8 * unit.km

# YEARS
periodMerc = 2.4085e-1 * unit.year
periodVenus = 6.1521e-1 * unit.year
periodEarth = 1.0004 * unit.year
periodMars = 1.88089 * unit.year
periodJup = 1.18622e1 * unit.year
periodSaturn = 2.94577e1 * unit.year
periodUranus = 8.40139e1 * unit.year
periodNeptune = 1.64793e2 * unit.year
periodPluto = 2.47686e2 * unit.year

# AU
smaMerc = 3.87096e-1 * unit.astronomical_unit
smaVenus = 7.23342e-1 * unit.astronomical_unit
smaEarth = 9.99987e-1 * unit.astronomical_unit
smaMars = 1.523705 * unit.astronomical_unit
smaJup = 5.204529 * unit.astronomical_unit
smaSaturn = 9.575133 * unit.astronomical_unit
smaUranus = 1.930375e1 * unit.astronomical_unit
smaNeptune = 3.020652e1 * unit.astronomical_unit
smaPluto = 3.991136e1 * unit.astronomical_unit

eccMerc = 0.205622
eccVenus = 0.006783
eccEarth = 0.016684
eccMars = 0.093404
eccJup = 0.047826
eccSaturn = 0.052754
eccUranus = 0.050363
eccNeptune = 0.004014
eccPluto = 0.256695


def l_star(m_bol):
    return math.pow(10, ((MBOL_SUN - m_bol) / 2.5))
