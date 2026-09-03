"""Sporadic E: the thin, intense, unpredictable layer near 105 km.

Sporadic E is a patch of greatly enhanced ionisation, a kilometre or two
thick, that forms in the E region from wind-shear convergence of metallic
ions.  It has almost nothing to do with solar production: it appears in
summer, peaks around mid-morning and again in the early evening, and can
carry frequencies far above anything the ordinary E layer supports -- 50 MHz
openings on a path that should have closed at 20.

The honest way to model it is as a **probability**, never as a state.  There
is no fact of the matter about whether sporadic E "is present" on a circuit
at a given hour; there is an occurrence rate and a distribution of foEs.
This module supplies both, and :mod:`propsim.reliability` composes them with
the ordinary prediction:

    reliability = P(Es) x reliability_with_Es + (1 - P(Es)) x reliability_without

A model that instead switched Es on and reported a single answer would be
claiming knowledge nobody has.

The occurrence statistics here are empirical summaries of the well-known
climatology -- strong summer maximum at mid-latitudes, a weaker equatorial
population tied to the electrojet, and a separate auroral population driven
by particle precipitation rather than wind shear.  They are labelled
empirical wherever they are reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple

__all__ = ["SporadicELayer", "sporadic_e_probability", "median_foes_mhz",
           "sporadic_e_for", "refine_grid_for_layer"]


@dataclass(frozen=True)
class SporadicELayer:
    """A sporadic-E patch.

    Attributes
    ----------
    foes_mhz:
        Critical frequency of the patch.  Above ``foes_mhz * sec(i)`` the
        wave goes straight through: sporadic E is partially reflecting, and
        the transition is sharp because the layer is thin.
    height_km:
        Height of the patch, typically 95-115 km.
    thickness_km:
        Half-thickness of the density profile.  Real patches are 0.5-2 km,
        which is why the height grid has to be refined around them: on a
        2 km grid the layer would fall between samples and vanish.
    """

    foes_mhz: float
    height_km: float = 105.0
    thickness_km: float = 1.0

    def __post_init__(self) -> None:
        if not 0.5 <= self.foes_mhz <= 40.0:
            raise ValueError(f"foEs {self.foes_mhz} MHz is outside 0.5-40 MHz")
        if not 90.0 <= self.height_km <= 130.0:
            raise ValueError(f"Es height {self.height_km} km is outside 90-130 km")
        if not 0.2 <= self.thickness_km <= 5.0:
            raise ValueError(f"Es thickness {self.thickness_km} km is implausible")

    @property
    def peak_density(self) -> float:
        from .constants import PLASMA_FREQ_COEFF_HZ

        return (self.foes_mhz * 1e6 / PLASMA_FREQ_COEFF_HZ) ** 2

    def density_at(self, height_km: float) -> float:
        """Gaussian patch: thin, symmetric, and gone within a few km.

        A Chapman shape is wrong here.  Chapman describes a layer in
        photochemical equilibrium with overhead radiation; sporadic E is a
        compressed cloud of long-lived metallic ions with no such balance,
        and it is very nearly symmetric about its peak.
        """
        z = (height_km - self.height_km) / self.thickness_km
        if abs(z) > 6.0:
            return 0.0
        return self.peak_density * math.exp(-0.5 * z * z)


def refine_grid_for_layer(
    heights_km: Sequence[float], layer: SporadicELayer, step_km: float = 0.25
) -> List[float]:
    """Insert fine height samples around a thin layer.

    Without this the patch is invisible: a 1 km-thick layer sampled every
    2 km is a layer the model never sees, and the ray passes through the
    gap between grid points as though nothing were there.
    """
    span = 5.0 * layer.thickness_km
    low = layer.height_km - span
    high = layer.height_km + span
    count = int(round((high - low) / step_km))
    extra = [low + i * step_km for i in range(count + 1)]
    merged = sorted(set(list(heights_km) + [h for h in extra if h > 0.0]))
    # Strictly ascending, no duplicates within a nanometre.
    result = [merged[0]]
    for value in merged[1:]:
        if value - result[-1] > 1e-9:
            result.append(value)
    return result


# --------------------------------------------------------------------------
# Occurrence climatology
# --------------------------------------------------------------------------

def _seasonal_factor(month: int, latitude_deg: float) -> float:
    """Summer maximum, referred to the local hemisphere."""
    # Peak in June for the north, December for the south.
    phase = (month - 6.0) / 12.0 * 2.0 * math.pi
    if latitude_deg < 0.0:
        phase += math.pi
    return 0.5 * (1.0 + math.cos(phase))


def _diurnal_factor(local_solar_hour: float) -> float:
    """Two maxima: mid-morning and early evening."""
    morning = math.exp(-((local_solar_hour - 10.0) / 3.0) ** 2)
    evening = math.exp(-((local_solar_hour - 19.0) / 3.0) ** 2)
    return max(0.12, morning + 0.75 * evening)


def sporadic_e_probability(
    when: datetime,
    geomagnetic_latitude_deg: float,
    local_solar_hour: float,
    kp: float = 2.0,
    threshold_mhz: float = 5.0,
) -> float:
    """Probability that a patch with ``foEs`` above the threshold is present.

    Three distinct populations, which is why one formula cannot describe
    sporadic E everywhere:

    * mid-latitude, wind-shear driven, strongly seasonal -- the familiar one;
    * equatorial, tied to the electrojet, weakly seasonal and mostly daytime;
    * auroral, driven by particle precipitation, rising with Kp and largely
      a night-time phenomenon.
    """
    magnitude = abs(geomagnetic_latitude_deg)
    season = _seasonal_factor(when.month, geomagnetic_latitude_deg)
    diurnal = _diurnal_factor(local_solar_hour)

    mid_latitude = (
        0.42 * math.exp(-((magnitude - 40.0) / 20.0) ** 2) * season * diurnal
    )
    equatorial = 0.28 * math.exp(-(magnitude / 10.0) ** 2) * (
        1.0 if 7.0 <= local_solar_hour <= 18.0 else 0.15
    )
    auroral = (
        0.30
        * math.exp(-((magnitude - 65.0) / 12.0) ** 2)
        * min(1.0, (kp / 6.0) ** 1.5)
    )

    probability = mid_latitude + equatorial + auroral

    # Higher thresholds are rarer; the tail of the foEs distribution falls
    # roughly exponentially above the median.
    median = median_foes_mhz(when, geomagnetic_latitude_deg, local_solar_hour)
    if threshold_mhz > median:
        probability *= math.exp(-(threshold_mhz - median) / 3.0)

    return max(0.0, min(0.95, probability))


def median_foes_mhz(
    when: datetime, geomagnetic_latitude_deg: float, local_solar_hour: float
) -> float:
    """Median foEs given that a patch is present.

    Conditional on presence -- this is not an expectation over all days.
    """
    magnitude = abs(geomagnetic_latitude_deg)
    season = _seasonal_factor(when.month, geomagnetic_latitude_deg)
    base = 4.0 + 4.5 * season * math.exp(-((magnitude - 40.0) / 22.0) ** 2)
    base += 2.0 * math.exp(-(magnitude / 10.0) ** 2)
    return max(2.0, min(base * (0.8 + 0.3 * _diurnal_factor(local_solar_hour)), 18.0))


def sporadic_e_for(
    when: datetime,
    geomagnetic_latitude_deg: float,
    local_solar_hour: float,
    kp: float = 2.0,
) -> Tuple[float, SporadicELayer]:
    """Occurrence probability and the layer to use when it does occur.

    Returns ``(probability, layer)``.  The layer is the median patch; the
    probability is what makes it meaningful.
    """
    foes = median_foes_mhz(when, geomagnetic_latitude_deg, local_solar_hour)
    probability = sporadic_e_probability(
        when, geomagnetic_latitude_deg, local_solar_hour, kp, threshold_mhz=foes
    )
    return probability, SporadicELayer(foes_mhz=foes, height_km=105.0, thickness_km=1.0)
