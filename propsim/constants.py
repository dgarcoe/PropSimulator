"""Physical and geometric constants used across the propagation chain.

All quantities are SI unless the name says otherwise.  The single unit
convention of the core is:

* lengths inside the ray tracer:            kilometres  (suffix ``_km``)
* antenna / structure heights:              metres      (suffix ``_m``)
* frequencies exposed to the user:          MHz         (suffix ``_mhz``)
* frequencies inside the plasma equations:  Hz          (suffix ``_hz``)
* powers, gains, losses:                    dBW / dB

Anything crossing a module boundary carries the unit in its name so a
kilometre can never be silently handed to something expecting metres.
"""

import math

# --- fundamental ---------------------------------------------------------
ELECTRON_CHARGE = 1.602176634e-19       # C
ELECTRON_MASS = 9.1093837015e-31        # kg
EPSILON_0 = 8.8541878128e-12            # F/m
SPEED_OF_LIGHT = 2.99792458e8           # m/s
BOLTZMANN = 1.380649e-23                # J/K
T0_KELVIN = 290.0                       # reference noise temperature

#: ``k * T0`` in dBW/Hz -- the -204 dBW/Hz used by every noise budget.
KT0_DBW_PER_HZ = 10.0 * math.log10(BOLTZMANN * T0_KELVIN)

# --- geometry ------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0088             # IUGG mean radius

# --- plasma --------------------------------------------------------------
#: ``f_p = PLASMA_FREQ_COEFF * sqrt(Ne)`` with Ne in m^-3 and f_p in Hz.
#: Derived, not copied: sqrt(e^2 / (eps0 * m_e)) / (2 pi) ~= 8.9787
PLASMA_FREQ_COEFF_HZ = math.sqrt(
    ELECTRON_CHARGE**2 / (EPSILON_0 * ELECTRON_MASS)
) / (2.0 * math.pi)

#: ``f_H = GYRO_FREQ_COEFF * B`` with B in tesla and f_H in Hz (~2.8e10).
GYRO_FREQ_COEFF_HZ = ELECTRON_CHARGE / (2.0 * math.pi * ELECTRON_MASS)

#: Non-deviative absorption coefficient e^2 / (2 eps0 m_e c), in s^-1 m^2.
ABSORPTION_COEFF = ELECTRON_CHARGE**2 / (
    2.0 * EPSILON_0 * ELECTRON_MASS * SPEED_OF_LIGHT
)

NEPER_TO_DB = 8.685889638065035

# --- solar ---------------------------------------------------------------
J2000_JD = 2451545.0
