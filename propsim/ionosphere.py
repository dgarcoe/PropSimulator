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
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Sequence

from .atmosphere import collision_frequency as atmosphere_collision_frequency
from .constants import PLASMA_FREQ_COEFF_HZ
from .geodesy import GeoPoint, path_points
from .magnetic import geomagnetic_latitude_deg
from .solar import solar_zenith_angle_deg
from .spaceweather import QUIET_XRAY_FLUX, SpaceWeather

__all__ = ["Layer", "DRegionLayer", "LayerSet", "IonosphericProfile", "build_profile",
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


def collision_frequency_hz(height_km: float) -> float:
    """Electron-neutral collision frequency, s^-1.

    Delegates to :mod:`propsim.atmosphere`, where it is derived from Banks'
    relation and the US Standard Atmosphere rather than fitted to a table.
    """
    return atmosphere_collision_frequency(height_km)


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
class DRegionLayer:
    """The D region, which is not a Chapman layer and should not be one.

    A Chapman layer has a production peak with a roughly symmetric shape
    about it.  The D region does not: its electron density climbs by two
    orders of magnitude between 70 and 90 km and then merges into the E
    layer without ever peaking on its own.  Forcing a Chapman shape on it
    produces a profile that is nearly flat across 70-90 km -- fourteen times
    too low at 85 km against published values -- and since that is exactly
    the band where the product of electron density and collision frequency
    is largest, it moves absorption out of the D region and into the E,
    which is the wrong region to be dominated by.

    The shape here is two terms fitted to the published daytime profile:

    * a ledge near 60-70 km, produced by cosmic rays and Lyman-alpha on
      nitric oxide, which survives the night at a reduced level;
    * an exponential rise with a 4.3 km scale height anchored at 90 km,
      which carries the daytime absorption.

    Both are empirical.  What matters is that they follow the measured
    shape rather than a shape borrowed from a different physical regime.
    """

    name: str
    #: Electron density at 90 km, the anchor of the rising term, m^-3.
    density_at_90km: float
    #: Density of the lower ledge at its centre, m^-3.
    ledge_density: float
    chi_deg: float
    #: Peak density of the flare component, m^-3.  Zero on a quiet sun.
    flare_density: float = 0.0

    #: Scale height of the rise, km.  Fitted to 1e8 at 70 km and 1e10 at 90.
    RISE_SCALE_KM = 4.3
    #: Centre and width of the lower ledge, km.
    LEDGE_CENTRE_KM = 62.0
    LEDGE_WIDTH_KM = 12.0
    #: Above this the E layer takes over; the D term is rolled off so the
    #: two are not both counted in the same kilometres.
    TAPER_HEIGHT_KM = 90.0
    TAPER_SCALE_KM = 3.0
    #: Centre and width of the flare component, km.  Hard X-rays penetrate
    #: deeper than Lyman-alpha, so a flare adds electrons around 75 km --
    #: below where the quiet D region lives, and where the collision
    #: frequency is an order of magnitude higher.  That is why a flare
    #: blacks out HF: the new electrons are exactly where they absorb most.
    FLARE_CENTRE_KM = 75.0
    FLARE_WIDTH_KM = 9.0

    @property
    def peak_density(self) -> float:
        return self.density_at_90km

    @property
    def peak_height_km(self) -> float:
        return self.TAPER_HEIGHT_KM

    @property
    def scale_height_km(self) -> float:
        return self.RISE_SCALE_KM

    @property
    def critical_frequency_mhz(self) -> float:
        return plasma_frequency_mhz(self.peak_density)

    def density_at(self, height_km: float) -> float:
        if height_km < 45.0 or height_km > 130.0:
            return 0.0
        if height_km <= self.TAPER_HEIGHT_KM:
            rise = self.density_at_90km * math.exp(
                (height_km - self.TAPER_HEIGHT_KM) / self.RISE_SCALE_KM
            )
        else:
            rise = self.density_at_90km * math.exp(
                -(height_km - self.TAPER_HEIGHT_KM) / self.TAPER_SCALE_KM
            )
        ledge = self.ledge_density * math.exp(
            -(((height_km - self.LEDGE_CENTRE_KM) / self.LEDGE_WIDTH_KM) ** 2)
        )
        flare = self.flare_density * math.exp(
            -(((height_km - self.FLARE_CENTRE_KM) / self.FLARE_WIDTH_KM) ** 2)
        )
        return rise + ledge + flare


@dataclass(frozen=True)
class LayerSet:
    """The four layers at one geographic point and time."""

    d: DRegionLayer
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


def _d_region_illumination(zenith_deg: float) -> float:
    """Solar control of D-region ionisation.

    Simple Chapman photochemistry gives ``Ne ~ sqrt(q) ~ sqrt(cos chi)``,
    since production is proportional to the cosine and loss is quadratic in
    the electron density.  Observed D-region absorption falls off more
    steeply than that, because the effective recombination coefficient is
    not constant: as the sun drops, negative-ion chemistry takes over and
    electrons are lost faster, so the density falls below what a fixed alpha
    would predict.  An exponent of 0.75 rather than 0.5 carries that.

    It is still shallower than the 0.881 the empirical absorption index
    uses, and the remaining gap is recorded in docs/MODEL.md rather than
    closed by fitting.
    """
    if zenith_deg < 90.0:
        return max(0.0, math.cos(math.radians(zenith_deg))) ** 0.75
    return 0.03 * math.exp(-(zenith_deg - 90.0) / 6.0)


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
    fof2_multiplier: float = 1.0,
    hmf2_offset_km: float = 0.0,
) -> LayerSet:
    """Build the four Chapman layers at one point.

    ``weather`` is passed whole, so ``xray_flux_wm2`` reaches the D-region
    construction by the same route as F10.7.  There is no separate optional
    X-ray argument that a caller could omit -- the flare either is in the
    scenario's space weather or it does not exist.

    ``fof2_multiplier`` scales foF2 away from its monthly median, which is
    how :mod:`propsim.variability` samples the day-to-day distribution.  It
    multiplies the critical *frequency*, so the density it sets is scaled by
    its square; 1.0 is the median day.
    """
    if fof2_multiplier <= 0.0:
        raise ValueError("foF2 multiplier must be positive")
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

    # Auroral particle precipitation ionises the D region independently of
    # the Sun, which is why absorption survives the night at high latitude.
    auroral = 1.0 + 2.2 * (weather.kp / 9.0) ** 2 * math.exp(
        -((abs(mag_lat) - 67.0) / 11.0) ** 2
    )

    d_illumination = _d_region_illumination(zenith)

    # 1e10 m^-3 at 90 km is the published quiet daytime value for an
    # overhead sun; illumination and solar activity scale it from there.
    # Around 90 km the ionisation is driven by Lyman-alpha, and Lyman-alpha
    # barely moves in a flare -- it is the X-ray band that spikes by orders
    # of magnitude.  So the enhancement here is slight: an exponent of 0.05
    # lifts an X1 by half again, against the six-fold rise at 75 km where
    # the X-rays actually deposit.  Applying the full square
    # root here put an X-class flare's D region above the F2 peak, which no
    # flare does, and made a C-class flare -- which does not disturb HF at
    # all -- read as a total blackout.
    d_at_90km = 1.0e10 * solar * d_illumination * xray_ratio**0.05 * auroral

    # The lower ledge is largely cosmic-ray produced, so it survives the
    # night at a quarter of its daytime value.  Cosmic rays do not care
    # about a solar flare, so no X-ray term belongs here.
    d_ledge = 1.0e8 * solar * (0.25 + 0.75 * d_illumination) * auroral

    # The flare component: hard X-rays penetrate deeper than Lyman-alpha and
    # deposit around 75 km -- below the quiet D region, and where the
    # collision frequency is an order of magnitude higher.  That is why a
    # flare blacks out HF: the new electrons land exactly where they absorb
    # most.  The amplitude is set so a quiet sun contributes 1e7 m^-3 there,
    # an order below the ledge and therefore inert, while an X1 reaches
    # 1e9, the observed flare-time value.
    d_flare = 2.5e7 * solar * d_illumination * auroral * xray_enhancement

    d_layer = DRegionLayer(
        "D", max(d_at_90km, 0.0), max(d_ledge, 0.0), zenith, max(d_flare, 0.0)
    )

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
    f2_height = min(max(f2_height + hmf2_offset_km, 150.0), 600.0)
    # foF2 goes as the square root of the peak density.
    f2_density *= fof2_multiplier**2
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

    @property
    def total_electron_content(self) -> float:
        """Vertical TEC in TECU (10^16 electrons per square metre).

        The trapezoidal integral of the profile over the modelled height
        range.  It is a *partial* TEC: this model stops at 600 km, and the
        plasmasphere above that carries a few TECU more, so the value is a
        lower bound on what a GNSS receiver would measure.
        """
        heights = self.heights_km
        densities = self.densities
        total = 0.0
        for i in range(len(heights) - 1):
            step_m = (heights[i + 1] - heights[i]) * 1000.0
            total += 0.5 * (densities[i] + densities[i + 1]) * step_m
        return total / 1e16


def height_grid(step_km: float = 2.0) -> List[float]:
    steps = int(round((MAX_HEIGHT_KM - MIN_HEIGHT_KM) / step_km))
    return [MIN_HEIGHT_KM + i * step_km for i in range(steps + 1)]


def build_profile(
    point: GeoPoint,
    when: datetime,
    weather: SpaceWeather,
    seasonal_phase: float = 0.0,
    step_km: float = 2.0,
    fof2_multiplier: float = 1.0,
    sporadic_e=None,
    hmf2_offset_km: float = 0.0,
) -> IonosphericProfile:
    """Total electron density profile at one point: the sum of all layers.

    ``sporadic_e``, when given, is a
    :class:`~propsim.sporadic_e.SporadicELayer` added on top of the regular
    layers.  The height grid is refined around it first: a patch a kilometre
    thick sampled every two kilometres is a patch the model never sees, and
    the ray would pass straight through the gap between grid points.
    """
    layers = build_layers(
        point, when, weather, seasonal_phase, fof2_multiplier, hmf2_offset_km
    )
    heights = height_grid(step_km)
    if sporadic_e is not None:
        from .sporadic_e import refine_grid_for_layer

        heights = refine_grid_for_layer(heights, sporadic_e)
    densities = [sum(layer.density_at(h) for layer in layers) for h in heights]
    if sporadic_e is not None:
        densities = [
            density + sporadic_e.density_at(h)
            for density, h in zip(densities, heights)
        ]
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
    #: Memo for :meth:`segment_profile`.  Mutating the dict is allowed on a
    #: frozen dataclass because the field itself is never rebound.
    _segments: dict = field(default_factory=dict, compare=False, repr=False)

    def segment_profile(
        self, low_fraction: float, high_fraction: float
    ) -> IonosphericProfile:
        """Average the ionosphere over one stretch of the path.

        A hop traverses a fraction of the circuit, so the column it should
        be traced through is the average over *that* stretch, not the
        average over the whole path and not the single profile at its
        midpoint.  For a one-hop circuit the stretch is the whole path and
        this reduces exactly to :attr:`mean_profile`, which is the case the
        equivalent-column approximation was built for.
        """
        low = min(max(low_fraction, 0.0), 1.0)
        high = min(max(high_fraction, 0.0), 1.0)
        if high < low:
            low, high = high, low
        key = (round(low, 4), round(high, 4))
        cached = self._segments.get(key)
        if cached is not None:
            return cached

        chosen = [
            profile
            for profile, fraction in zip(self.profiles, self.fractions)
            if low - 1e-9 <= fraction <= high + 1e-9
        ]
        if not chosen:
            chosen = [self.profile_at_fraction(0.5 * (low + high))]
        if len(chosen) == 1:
            self._segments[key] = chosen[0]
            return chosen[0]

        heights = chosen[0].heights_km
        densities = [
            sum(profile.densities[i] for profile in chosen) / len(chosen)
            for i in range(len(heights))
        ]
        middle = chosen[len(chosen) // 2]
        profile = IonosphericProfile(
            heights_km=heights,
            densities=densities,
            layers=middle.layers,
            point=middle.point,
            solar_zenith_deg=middle.solar_zenith_deg,
        )
        self._segments[key] = profile
        return profile

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
    fof2_multiplier: float = 1.0,
    sporadic_e=None,
    hmf2_offset_km: float = 0.0,
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
    profiles = [
        build_profile(
            p, when, weather, seasonal_phase, step_km, fof2_multiplier,
            sporadic_e, hmf2_offset_km,
        )
        for p in points
    ]

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
