"""Chapman-layer ionosphere: D, E, F1 and F2.

A Chapman layer is the electron-density profile produced by monochromatic
ionising radiation absorbed in an atmosphere whose neutral density falls
exponentially with height.  Each layer has a peak density ``Nm`` at a height
``hm`` and a scale height ``H``:

    Ne(h) = Nm * exp(0.5 * (1 - z - sec(chi) * exp(-z))),  z = (h - hm) / H

The layer parameters below are empirical fits to the observed behaviour of
the real ionosphere -- they are engineering approximations, not first
principles, and they are labelled as such.  What they do reproduce is the
set of dependences that matter operationally: more solar flux raises the
density, geomagnetic storms depress F2, the sunlit D region absorbs, and the
equatorial anomaly moves ionisation off the magnetic equator.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

from .constants import PLASMA_FREQ_COEFF_HZ
from .geodesy import GeoPoint, path_points
from .magnetic import geomagnetic_latitude_deg
from .solar import solar_zenith_angle_deg
from .spaceweather import QUIET_XRAY_FLUX, SpaceWeather

__all__ = ["Layer", "LayerSet", "IonosphericProfile", "build_profile",
           "EquivalentColumn", "build_equivalent_column",
           "plasma_frequency_hz", "plasma_frequency_mhz",
           "electron_density_from_plasma_frequency", "collision_frequency_hz"]

#: Heights spanned by the model, km.  Below 50 km there are no free
#: electrons worth counting; above 600 km the density is too low to refract
#: HF appreciably.
MIN_HEIGHT_KM = 50.0
MAX_HEIGHT_KM = 600.0


def plasma_frequency_hz(electron_density: float) -> float:
    """Plasma frequency in **Hz** for a density in m^-3."""
    if electron_density <= 0.0:
        return 0.0
    return PLASMA_FREQ_COEFF_HZ * math.sqrt(electron_density)


def plasma_frequency_mhz(electron_density: float) -> float:
    """Plasma frequency in **MHz** for a density in m^-3.

    The conversion is a single division by 1e6 applied to the Hz form -- the
    two helpers cannot drift apart, and the unit lives in the name.
    """
    return plasma_frequency_hz(electron_density) / 1e6


def electron_density_from_plasma_frequency(frequency_hz: float) -> float:
    """Inverse of :func:`plasma_frequency_hz`."""
    if frequency_hz <= 0.0:
        return 0.0
    return (frequency_hz / PLASMA_FREQ_COEFF_HZ) ** 2


#: Quadratic-in-log fit to the electron-neutral collision frequency,
#: log10(nu) = A + B h + C h^2, anchored on the standard values
#: 1.5e7 s^-1 at 60 km, 8e5 at 90 km and 1e4 at 120 km.  A single
#: exponential cannot span that range: the neutral scale height falls from
#: about 10 km in the D region to under 6 km at the base of the E region,
#: and forcing one exponent through both leaves the collision frequency
#: several times too high at E-layer heights -- which shows up directly as
#: too much absorption for every ray that grazes the E layer.
#: Least-squares fit over all seven reference heights (60-120 km) rather
#: than an exact fit through three of them.  The reference values are not
#: perfectly smooth -- they imply a factor of ten between 100 and 110 km and
#: only three between 110 and 120 -- so no quadratic reproduces them all;
#: the worst deviation is 1.55x, at 110 km.  That residual is documented
#: rather than absorbed into a tuning constant.
_NU_A, _NU_B, _NU_C = 7.54528, 1.710708e-2, -3.933216e-4


def collision_frequency_hz(height_km: float) -> float:
    """Electron-neutral collision frequency, s^-1."""
    log_nu = _NU_A + _NU_B * height_km + _NU_C * height_km**2
    return 10.0**log_nu


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layer:
    """One Chapman layer."""

    name: str
    peak_density: float      # m^-3
    peak_height_km: float
    scale_height_km: float
    #: Solar zenith angle at the layer, carried for reporting.  It is *not*
    #: fed back into the profile shape as a sec(chi) production term: the
    #: peak density and peak height of every layer here are prescribed
    #: empirically (they already contain the illumination dependence), so
    #: re-applying sec(chi) would count the same physics twice and would
    #: push the modelled peak hundreds of kilometres above the prescribed
    #: hm at large zenith angles.
    chi_deg: float

    @property
    def critical_frequency_mhz(self) -> float:
        return plasma_frequency_mhz(self.peak_density)

    def density_at(self, height_km: float) -> float:
        """Chapman density at a height, in m^-3.

        The alpha-Chapman shape at normal incidence,
        ``exp(0.5 * (1 - z - exp(-z)))``, which peaks exactly at ``z = 0``
        and so honours the prescribed ``peak_height_km``.  It falls off with
        the neutral scale height below the peak and half as fast above it,
        which is the asymmetry that makes the topside reach so far.
        """
        if self.peak_density <= 0.0:
            return 0.0
        z = (height_km - self.peak_height_km) / self.scale_height_km
        if z > 60.0:
            return 0.0
        exponent = 0.5 * (1.0 - z - math.exp(-z))
        if exponent < -60.0:      # exp underflow guard, density is nil anyway
            return 0.0
        return self.peak_density * math.exp(exponent)


@dataclass(frozen=True)
class LayerSet:
    """The four layers at one geographic point and time."""

    d: Layer
    e: Layer
    f1: Layer
    f2: Layer

    def __iter__(self):
        return iter((self.d, self.e, self.f1, self.f2))

    @property
    def fof2_mhz(self) -> float:
        return self.f2.critical_frequency_mhz


def _solar_scaling(f107: float, sunspot_number: float) -> float:
    """Combined solar-activity factor, 1.0 at F10.7 = 100."""
    flux_term = (f107 / 100.0) ** 0.6
    sunspot_term = (1.0 + sunspot_number / 250.0) ** 0.35
    return flux_term * sunspot_term


def _chapman_illumination(zenith_deg: float) -> float:
    """Production factor from overhead (1.0) to deep night (0.0).

    Below the horizon the factor decays smoothly rather than snapping to
    zero, which is what keeps a terminator crossing continuous.
    """
    if zenith_deg < 90.0:
        return max(0.0, math.cos(math.radians(zenith_deg))) ** 0.5
    # Twilight tail: the layer is still partly illuminated at height.
    return 0.06 * math.exp(-(zenith_deg - 90.0) / 6.0)


def build_layers(
    point: GeoPoint,
    when: datetime,
    weather: SpaceWeather,
    seasonal_phase: float = 0.0,
) -> LayerSet:
    """Build the four Chapman layers at one point.

    ``weather`` is passed whole, so ``xray_flux_wm2`` reaches the D-region
    construction by the same route as F10.7.  There is no separate optional
    X-ray argument that a caller could omit -- the flare either is in the
    scenario's space weather or it does not exist.
    """
    zenith = solar_zenith_angle_deg(point, when)
    illumination = _chapman_illumination(zenith)
    solar = _solar_scaling(weather.f107, weather.sunspot_number)
    mag_lat = geomagnetic_latitude_deg(point)

    # ---- D region ------------------------------------------------------
    # Driven by Lyman-alpha in quiet conditions and by hard X-rays during a
    # flare.  The X-ray term is the model's only flare mechanism: absorption
    # downstream reads this density and is never multiplied by a second
    # empirical flare factor on top of it.
    xray_ratio = weather.xray_flux_wm2 / QUIET_XRAY_FLUX
    xray_enhancement = xray_ratio ** 0.5   # Ne ~ sqrt(q) in the recombining D
    d_density = 2.5e8 * solar * illumination * xray_enhancement

    # Auroral particle precipitation ionises the D region independently of
    # the Sun, which is why absorption survives the night at high latitude.
    auroral = 1.0 + 2.2 * (weather.kp / 9.0) ** 2 * math.exp(
        -((abs(mag_lat) - 67.0) / 11.0) ** 2
    )
    d_density *= auroral
    d_layer = Layer("D", max(d_density, 0.0), 75.0, 5.0, zenith)

    # ---- E region ------------------------------------------------------
    # The classical empirical relation, rather than a parameterisation of
    # this package's own invention:
    #
    #     foE = 0.9 * [(180 + 1.44 R12) cos(chi)]^0.25   MHz
    #
    # It is the long-standing expression behind E-layer prediction, it
    # carries both the solar-activity and the zenith-angle dependence, and
    # it is checkable against a source outside this code base.  An earlier
    # invented form matched its *shape* exactly but sat 9.9% high in foE --
    # 21% in density, which the absorption integral pays for directly,
    # since the E region dominates the loss for a ray that crosses it.
    # No clamp on the zenith angle here.  Clamping it to 89 degrees, as the
    # reference oracle does to guard its own division, would treat local
    # midnight as though the Sun sat one degree above the horizon and leave
    # the night-time E layer nearly three times too dense.  Below the
    # horizon the relation simply does not apply and the nocturnal residue
    # takes over, which it reaches smoothly as cos(chi) goes to zero.
    cos_chi = math.cos(math.radians(zenith))
    if cos_chi > 0.0:
        foe_mhz = 0.9 * ((180.0 + 1.44 * weather.sunspot_number) * cos_chi) ** 0.25
        e_density = (foe_mhz * 1e6 / PLASMA_FREQ_COEFF_HZ) ** 2
    else:
        e_density = 0.0
    e_density = max(e_density, 3.0e9)     # weak nocturnal residue
    # 8 km: E-region scale heights run 5-10 km.  The Chapman topside falls
    # with twice the scale height, so a thicker E leaks density up into the
    # 130-160 km band where F1 should already be taking over.
    e_layer = Layer("E", e_density, 110.0, 8.0, zenith)

    # ---- F1 region -----------------------------------------------------
    # F1 exists as a distinct ledge only in the sunlit, summer ionosphere.
    f1_presence = illumination * (0.55 + 0.45 * seasonal_phase)
    f1_density = 2.5e11 * solar * max(f1_presence, 0.0)
    f1_layer = Layer("F1", f1_density, 180.0, 20.0, zenith)

    # ---- F2 region -----------------------------------------------------
    # F2 keeps a large fraction of its ionisation through the night: the
    # recombination rate at 300 km is slow compared with a night's length.
    f2_day_night = 0.30 + 0.70 * illumination
    f2_density = 1.1e12 * solar * f2_day_night

    # Equatorial ionisation anomaly: fountain uplift empties the magnetic
    # equator and piles ionisation into two crests near +-15 deg dip lat.
    crest = math.exp(-((abs(mag_lat) - 15.0) / 10.0) ** 2)
    trough = math.exp(-(mag_lat / 8.0) ** 2)
    f2_density *= 1.0 + 0.45 * crest * illumination - 0.22 * trough * illumination

    # Storm-time negative phase: composition changes deplete F2 at mid and
    # high latitude while leaving the equator largely alone.
    if weather.kp > 3.0:
        storm_depth = 0.11 * (weather.kp - 3.0)
        latitude_weight = min(1.0, (abs(mag_lat) / 45.0) ** 2)
        f2_density *= max(0.25, 1.0 - storm_depth * latitude_weight)

    # Winter anomaly in the daytime F2 layer.
    f2_density *= 1.0 - 0.12 * seasonal_phase * illumination

    # Peak height rises at night and with solar activity.
    f2_height = 300.0 + 45.0 * (1.0 - illumination) + 25.0 * (solar - 1.0)
    f2_height = min(max(f2_height, 250.0), 420.0)
    f2_layer = Layer("F2", max(f2_density, 1e9), f2_height, 55.0, zenith)

    return LayerSet(d_layer, e_layer, f1_layer, f2_layer)


# --------------------------------------------------------------------------
# Vertical profile
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IonosphericProfile:
    """Electron density against height at one geographic point.

    Heights are stored strictly ascending.  Every interpolation in the code
    base goes through :meth:`density_at`, which sorts its own input, so a
    caller descending from a ray apex cannot silently get the wrong answer.
    """

    heights_km: Sequence[float]
    densities: Sequence[float]
    layers: LayerSet
    point: GeoPoint
    solar_zenith_deg: float

    def __post_init__(self) -> None:
        heights = list(self.heights_km)
        if len(heights) < 2:
            raise ValueError("a profile needs at least two heights")
        if any(b <= a for a, b in zip(heights, heights[1:])):
            raise ValueError("profile heights must be strictly ascending")
        if len(self.densities) != len(heights):
            raise ValueError("heights and densities differ in length")

    def density_at(self, height_km: float) -> float:
        """Linearly interpolated electron density, m^-3.

        Outside the modelled range the density is zero: no extrapolation, so
        a ray that wanders below 50 km or above 600 km gets vacuum rather
        than an invented number.
        """
        heights = self.heights_km
        if height_km <= heights[0] or height_km >= heights[-1]:
            return 0.0
        index = bisect_left(heights, height_km)
        h0, h1 = heights[index - 1], heights[index]
        n0, n1 = self.densities[index - 1], self.densities[index]
        weight = (height_km - h0) / (h1 - h0)
        return n0 + weight * (n1 - n0)

    def plasma_frequency_mhz_at(self, height_km: float) -> float:
        return plasma_frequency_mhz(self.density_at(height_km))

    @property
    def peak_density(self) -> float:
        return max(self.densities)

    @property
    def peak_height_km(self) -> float:
        return self.heights_km[list(self.densities).index(self.peak_density)]

    @property
    def critical_frequency_mhz(self) -> float:
        return plasma_frequency_mhz(self.peak_density)


def height_grid(step_km: float = 2.0) -> List[float]:
    steps = int(round((MAX_HEIGHT_KM - MIN_HEIGHT_KM) / step_km))
    return [MIN_HEIGHT_KM + i * step_km for i in range(steps + 1)]


def build_profile(
    point: GeoPoint,
    when: datetime,
    weather: SpaceWeather,
    seasonal_phase: float = 0.0,
    step_km: float = 2.0,
) -> IonosphericProfile:
    """Total electron density profile at one point: the sum of all layers."""
    layers = build_layers(point, when, weather, seasonal_phase)
    heights = height_grid(step_km)
    densities = [sum(layer.density_at(h) for layer in layers) for h in heights]
    return IonosphericProfile(
        heights_km=heights,
        densities=densities,
        layers=layers,
        point=point,
        solar_zenith_deg=solar_zenith_angle_deg(point, when),
    )


# --------------------------------------------------------------------------
# Equivalent column along a path
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalentColumn:
    """Path-averaged ionosphere plus the local profiles it averages.

    The ray geometry is integrated through :attr:`mean_profile` -- a single
    spherically-stratified column, which is what makes the Bouguer invariant
    applicable at all.  Absorption instead reads :meth:`profile_at_fraction`,
    the *local* ionosphere under each piece of the path, so a path crossing
    the terminator absorbs like a half-lit path rather than like its average.
    """

    mean_profile: IonosphericProfile
    profiles: Sequence[IonosphericProfile]
    fractions: Sequence[float]

    def profile_at_fraction(self, fraction: float) -> IonosphericProfile:
        """Local profile at a fraction of the way along the path."""
        clamped = min(max(fraction, 0.0), 1.0)
        index = min(
            range(len(self.fractions)),
            key=lambda i: abs(self.fractions[i] - clamped),
        )
        return self.profiles[index]

    @property
    def fof2_mhz(self) -> float:
        return self.mean_profile.critical_frequency_mhz


def build_equivalent_column(
    tx: GeoPoint,
    rx: GeoPoint,
    when: datetime,
    weather: SpaceWeather,
    seasonal_phase: float = 0.0,
    samples: int = 9,
    step_km: float = 2.0,
) -> EquivalentColumn:
    """Average the ionosphere along the great circle at each height.

    This is the compromise that lets a varying ionosphere be traced with a
    spherically-symmetric ray solver.  It captures the terminator and the
    latitude gradient in an averaged sense; it does not and cannot produce
    horizontal refraction, off-great-circle paths or ducting, and no result
    downstream claims otherwise.
    """
    if samples < 2:
        raise ValueError("need at least two sample points")
    points = path_points(tx, rx, samples)
    fractions = [i / (samples - 1) for i in range(samples)]
    profiles = [build_profile(p, when, weather, seasonal_phase, step_km) for p in points]

    heights = profiles[0].heights_km
    mean_densities = [
        sum(profile.densities[i] for profile in profiles) / len(profiles)
        for i in range(len(heights))
    ]
    midpoint = profiles[len(profiles) // 2]
    mean_profile = IonosphericProfile(
        heights_km=heights,
        densities=mean_densities,
        layers=midpoint.layers,
        point=points[len(points) // 2],
        solar_zenith_deg=midpoint.solar_zenith_deg,
    )
    return EquivalentColumn(mean_profile, profiles, fractions)
