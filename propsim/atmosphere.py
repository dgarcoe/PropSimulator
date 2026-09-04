"""Neutral atmosphere, and the electron-neutral collision frequency from it.

Absorption is proportional to the product of electron density and collision
frequency, so the collision frequency is not a detail: an error in it moves
loss between the D and E regions and changes which one dominates.

It is computed here rather than fitted.  Banks' relation

    nu_en = 5.4e-16 * n_n * sqrt(T_e)

with the neutral number density and temperature of the US Standard
Atmosphere gives a profile that is smooth over 60-120 km -- a cubic fits it
to within 4%.  An earlier version of this package used a polynomial fitted
to a table of remembered values instead, and no polynomial of any degree
could fit that table better than 40%, because the table was internally
inconsistent: it implied a factor of ten between 100 and 110 km and only
three over the next ten kilometres, which no smooth atmosphere does.  The
derived profile is a factor of three lower at 110 km than that table, which
is exactly where it mattered most.

Electron temperature is taken equal to the neutral temperature.  In the D
and lower E regions the two are close; higher up the electrons run hotter,
but there the collision frequency is already too low to absorb.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = ["neutral_density", "temperature", "collision_frequency",
           "collision_frequency_array", "STANDARD_ATMOSPHERE"]

AVOGADRO = 6.02214076e23

#: Banks' electron-neutral momentum-transfer coefficient, m^3 K^-1/2 s^-1.
BANKS_COEFFICIENT = 5.4e-16

#: US Standard Atmosphere 1976: height (km), mass density (kg/m^3), mean
#: molar mass (kg/mol), temperature (K).  The molar mass falls above 90 km
#: as the lighter species take over, which is why it is carried explicitly
#: rather than held at the sea-level value.
STANDARD_ATMOSPHERE: Sequence[Tuple[float, float, float, float]] = (
    (40.0, 3.996e-3, 0.02896, 250.4),
    (50.0, 1.027e-3, 0.02896, 270.6),
    (60.0, 3.097e-4, 0.02896, 247.0),
    (70.0, 8.283e-5, 0.02896, 219.6),
    (80.0, 1.846e-5, 0.02896, 198.6),
    (90.0, 3.416e-6, 0.02890, 186.9),
    (100.0, 5.604e-7, 0.02840, 195.1),
    (110.0, 9.708e-8, 0.02730, 240.0),
    (120.0, 2.222e-8, 0.02620, 360.0),
    (140.0, 3.831e-9, 0.02460, 559.6),
    (160.0, 1.233e-9, 0.02320, 696.3),
    (200.0, 3.318e-10, 0.02100, 845.6),
    (300.0, 5.681e-11, 0.01780, 976.0),
)

_HEIGHTS = np.array([row[0] for row in STANDARD_ATMOSPHERE])
#: Number density, m^-3.  Interpolated in the logarithm, because it falls
#: exponentially and linear interpolation between decades is meaningless.
_LOG_NUMBER = np.log(
    np.array([row[1] * AVOGADRO / row[2] for row in STANDARD_ATMOSPHERE])
)
_TEMPERATURE = np.array([row[3] for row in STANDARD_ATMOSPHERE])


def neutral_density(height_km: float) -> float:
    """Neutral number density, m^-3."""
    return float(np.exp(np.interp(height_km, _HEIGHTS, _LOG_NUMBER)))


def temperature(height_km: float) -> float:
    """Neutral temperature, K."""
    return float(np.interp(height_km, _HEIGHTS, _TEMPERATURE))


def collision_frequency(height_km: float) -> float:
    """Electron-neutral collision frequency, s^-1."""
    return BANKS_COEFFICIENT * neutral_density(height_km) * math.sqrt(
        temperature(height_km)
    )


def collision_frequency_array(heights_km) -> np.ndarray:
    """Vectorised twin of :func:`collision_frequency`.

    Absorption evaluates this at every quadrature node of every ray, so it
    is the one place where the scalar form would be a bottleneck.
    """
    heights = np.asarray(heights_km, dtype=float)
    density = np.exp(np.interp(heights, _HEIGHTS, _LOG_NUMBER))
    return BANKS_COEFFICIENT * density * np.sqrt(
        np.interp(heights, _HEIGHTS, _TEMPERATURE)
    )
