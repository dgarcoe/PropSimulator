"""Appleton-Hartree refractive index for a magnetised plasma.

The collisionless form, written so that both magnetoionic modes stay
numerically well behaved right through the reflection height:

    n^2 = 1 - X / D,
    D   = 1 - Y_T^2 / (2(1-X))  -/+  sqrt( Y_T^4 / (4(1-X)^2) + Y_L^2 )

with ``X = (f_N/f)^2``, ``Y = f_H/f``, ``Y_L = Y cos(theta)`` and
``Y_T = Y sin(theta)``, where ``theta`` is the angle between the magnetic
field and the ray.

Written naively, the ordinary-mode branch subtracts two nearly equal large
numbers near ``X = 1`` and loses every significant digit exactly where the
ray reflects.  The implementation below rearranges that difference into the
algebraically identical form ``b^2 / (sqrt(a^2+b^2) + a)``, which is exact
and stable, so the O mode reflects cleanly at ``X = 1`` and the X mode at
``X = 1 - Y`` without special-casing.
"""

from __future__ import annotations

import math
from enum import Enum

from .constants import GYRO_FREQ_COEFF_HZ, PLASMA_FREQ_COEFF_HZ

__all__ = ["Mode", "refractive_index_squared", "refractive_index",
           "reflection_density", "x_parameter", "y_parameter"]


class Mode(str, Enum):
    """Magnetoionic mode."""

    ORDINARY = "O"
    EXTRAORDINARY = "X"


def x_parameter(electron_density: float, frequency_hz: float) -> float:
    """``X = (f_N / f)^2``, the normalised electron density."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency must be positive")
    plasma_hz = PLASMA_FREQ_COEFF_HZ * math.sqrt(max(electron_density, 0.0))
    return (plasma_hz / frequency_hz) ** 2


def y_parameter(magnetic_field_tesla: float, frequency_hz: float) -> float:
    """``Y = f_H / f``, the normalised gyrofrequency."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency must be positive")
    return GYRO_FREQ_COEFF_HZ * abs(magnetic_field_tesla) / frequency_hz


def refractive_index_squared(
    electron_density: float,
    frequency_hz: float,
    magnetic_field_tesla: float = 0.0,
    theta_rad: float = math.pi / 2.0,
    mode: Mode = Mode.ORDINARY,
) -> float:
    """``n^2`` for one magnetoionic mode.

    Parameters
    ----------
    electron_density:
        Local electron density, m^-3.
    frequency_hz:
        Wave frequency, Hz.
    magnetic_field_tesla:
        Local field magnitude.  Zero reduces the result to the isotropic
        ``n^2 = 1 - X`` for both modes.
    theta_rad:
        Angle between the magnetic field and the direction of propagation.
        The extraordinary mode depends on it; the ordinary mode is only
        weakly sensitive away from exact longitudinal propagation.
    """
    x = x_parameter(electron_density, frequency_hz)
    y = y_parameter(magnetic_field_tesla, frequency_hz)

    if y == 0.0:
        return 1.0 - x

    y_long = y * math.cos(theta_rad)
    y_tran = y * math.sin(theta_rad)
    u = 1.0 - x                      # passes through zero at O-mode reflection

    # Multiplying the textbook denominator through by u clears the 1/u and
    # 1/u^2 singularities.  The factor sqrt() picks up sign(u), because
    # u * sqrt(A/u^2 + B) == sign(u) * sqrt(A + B u^2) -- dropping that sign
    # would put the reflection heights of both modes in the wrong place for
    # any over-dense sample.
    a = 0.5 * y_tran**2
    b = y_long * u
    root = math.copysign(math.hypot(a, b), u) if u != 0.0 else math.hypot(a, b)

    if mode is Mode.ORDINARY:
        if u >= 0.0:
            # Cancellation-free rearrangement, exact and finite at u -> 0:
            #   -a + sqrt(a^2+b^2) == b^2 / (sqrt(a^2+b^2) + a)
            floor = math.hypot(a, b) + a
            if floor <= 0.0:
                return 1.0 - x
            return 1.0 - x / (1.0 + (y_long**2) * u / floor)
        denominator = u - a + root
    else:
        denominator = u - a - root

    if abs(denominator) < 1e-300:
        return -math.inf             # resonance: the mode cannot propagate
    return 1.0 - x * u / denominator


def refractive_index(
    electron_density: float,
    frequency_hz: float,
    magnetic_field_tesla: float = 0.0,
    theta_rad: float = math.pi / 2.0,
    mode: Mode = Mode.ORDINARY,
) -> float:
    """Real refractive index; zero where the wave is evanescent."""
    n2 = refractive_index_squared(
        electron_density, frequency_hz, magnetic_field_tesla, theta_rad, mode
    )
    return math.sqrt(n2) if n2 > 0.0 else 0.0


def reflection_density(
    frequency_hz: float,
    magnetic_field_tesla: float = 0.0,
    mode: Mode = Mode.ORDINARY,
) -> float:
    """Electron density at which a vertically incident wave reflects.

    The O mode turns at ``X = 1`` exactly (``f_N = f``).  The X mode turns
    at ``X = 1 - Y`` exactly, i.e. ``f_N = f sqrt(1 - Y)``; the familiar
    ``f_N = f - f_H/2`` is the first-order expansion of that and is a good
    approximation only while ``Y`` is small.  Both cutoffs are independent
    of the field/ray angle, which is a useful check on the index formula.
    """
    y = y_parameter(magnetic_field_tesla, frequency_hz)
    x_reflect = 1.0 if mode is Mode.ORDINARY else max(0.0, 1.0 - y)
    plasma_hz = frequency_hz * math.sqrt(x_reflect)
    return (plasma_hz / PLASMA_FREQ_COEFF_HZ) ** 2
