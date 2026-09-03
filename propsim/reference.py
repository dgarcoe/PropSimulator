"""Independent reference models, used to cross-check the physics core.

Nothing in here feeds a prediction.  These are oracles: alternative,
externally-derived expressions for quantities PropSimulator computes from
first principles, so the core can be checked against something other than
itself.  The ray tracer already has such an oracle -- the closed-form
Bouguer integrals of a sharp reflector -- and this module supplies the
missing ones for the ionosphere and for absorption.

A caveat that belongs in the code rather than in a footnote: the absorption
index below is the *published functional form* of the George-Bradley /
ITU-R P.533 absorption term, and its numerical constants are reproduced
from that standard expression.  They could not be checked against the ITU
document in the environment this was written in.  The comparison is
therefore used the way it should be used regardless: its **scaling laws**
-- how absorption varies with frequency, obliquity, solar zenith angle and
solar activity -- are robust, widely reproduced and worth testing against.
The absolute constant is reported as a measured ratio, never silently
folded into the core as a correction factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["absorption_index_db", "AbsorptionScaling", "measure_scaling",
           "chapman_foe_mhz"]


def chapman_foe_mhz(zenith_deg: float, sunspot_number: float) -> float:
    """E-layer critical frequency from the classical empirical relation.

    ``foE = 0.9 * [(180 + 1.44 R12) cos(chi)]^0.25``, the long-standing
    expression behind every E-layer prediction.

    It is a **daytime** relation: below the horizon there is no solar
    production to describe, and the small floor on ``cos(chi)`` here only
    guards this oracle's own arithmetic.  It is not a night-time model, and
    a caller must not read it as one.
    """
    cos_chi = max(math.cos(math.radians(min(zenith_deg, 89.0))), 0.01)
    return 0.9 * ((180.0 + 1.44 * sunspot_number) * cos_chi) ** 0.25


def incidence_angle_at_110km(elevation_deg: float, earth_radius_km: float = 6371.0088) -> float:
    """Angle of incidence at the 110 km reference height, degrees."""
    beta = math.radians(elevation_deg)
    sin_i = earth_radius_km * math.cos(beta) / (earth_radius_km + 110.0)
    return math.degrees(math.asin(min(1.0, sin_i)))


def absorption_index_db(
    frequency_mhz: float,
    elevation_deg: float,
    zenith_deg: float,
    sunspot_number: float = 50.0,
    gyrofrequency_mhz: float = 1.2,
    hops: int = 1,
) -> float:
    """Non-deviative absorption per the P.533-style absorption index.

    ``677.2 sec(i110) (1 + 0.0067 R12) F(chi) / ((f + fL)^1.98 + 10.2)``

    The pieces and why each is there:

    * ``sec(i110)`` -- obliquity: a shallower ray spends longer in the
      absorbing layer.
    * ``(f + fL)^1.98`` -- the near-inverse-square frequency law, offset by
      the longitudinal gyrofrequency because the wave is absorbed hardest
      near gyroresonance.
    * ``F(chi) = cos^0.881(chi)`` -- solar control of D-region ionisation,
      with a small night-time residue.
    * ``(1 + 0.0067 R12)`` -- solar-cycle dependence.
    """
    if frequency_mhz <= 0.0:
        raise ValueError("frequency must be positive")

    incidence = math.radians(incidence_angle_at_110km(elevation_deg))
    sec_i = 1.0 / max(math.cos(incidence), 1e-3)

    if zenith_deg < 90.0:
        solar_term = max(math.cos(math.radians(zenith_deg)), 0.0) ** 0.881
    else:
        solar_term = 0.02          # night-time residue

    numerator = 677.2 * sec_i * (1.0 + 0.0067 * sunspot_number) * solar_term
    denominator = (frequency_mhz + gyrofrequency_mhz) ** 1.98 + 10.2
    return hops * numerator / denominator


@dataclass(frozen=True)
class AbsorptionScaling:
    """Power-law exponents fitted to an absorption model's behaviour.

    Comparing these between the core and the reference tests the *shape* of
    the model -- which is what a physical derivation should get right --
    independently of any overall constant.
    """

    frequency_exponent: float
    obliquity_exponent: float
    zenith_exponent: float

    def summary(self) -> dict:
        return {
            "frequency_exponent": self.frequency_exponent,
            "obliquity_exponent": self.obliquity_exponent,
            "zenith_exponent": self.zenith_exponent,
        }


def _log_slope(xs, ys) -> float:
    """Least-squares slope of log(y) against log(x)."""
    points = [
        (math.log(x), math.log(y))
        for x, y in zip(xs, ys)
        if x is not None
        and y is not None
        and math.isfinite(x)
        and math.isfinite(y)
        and x > 0.0
        and y > 0.0
    ]
    if len(points) < 2:
        return float("nan")
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    numerator = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    denominator = sum((p[0] - mean_x) ** 2 for p in points)
    return numerator / denominator if denominator else float("nan")


def measure_scaling(absorption_of) -> AbsorptionScaling:
    """Fit the three scaling exponents of any absorption function.

    ``absorption_of(frequency_mhz, elevation_deg, zenith_deg)`` returns dB,
    or ``None`` where the model has no answer -- a ray that escapes, or one
    that turns below the absorbing layer instead of crossing it.  Those
    points are dropped from the fit rather than replaced by a small number:
    substituting a floor makes the regression fit the floor, and the slope
    it returns is then a property of the substitution, not of the model.

    The same probe is applied to the core and to the reference, so the two
    are characterised identically.
    """
    frequencies = [7.0, 10.0, 14.0, 18.0, 21.0]
    frequency_slope = _log_slope(
        frequencies, [absorption_of(f, 20.0, 30.0) for f in frequencies]
    )

    elevations = [5.0, 10.0, 20.0, 30.0, 45.0]
    secants = [
        1.0 / math.cos(math.radians(incidence_angle_at_110km(e))) for e in elevations
    ]
    obliquity_slope = _log_slope(
        secants, [absorption_of(14.0, e, 30.0) for e in elevations]
    )

    zeniths = [0.0, 20.0, 40.0, 60.0, 75.0]
    cosines = [math.cos(math.radians(z)) for z in zeniths]
    zenith_slope = _log_slope(
        cosines, [absorption_of(14.0, 20.0, z) for z in zeniths]
    )

    return AbsorptionScaling(frequency_slope, obliquity_slope, zenith_slope)
