"""Radio noise at the receiver: thermal, galactic, atmospheric, man-made.

Every source is expressed as an ITU-style noise figure ``Fa`` in dB above
the thermal floor ``kT0``, and the sources are combined **in power**, not in
dB -- adding decibels would systematically overstate the total.

The atmospheric term is a compact empirical fit, not a full implementation
of ITU-R P.372: it reproduces the frequency slope, the day/night ratio and
the broad latitude trend, and it is labelled an approximation wherever it is
reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .constants import KT0_DBW_PER_HZ

__all__ = ["NoiseEnvironment", "NoiseBudget", "noise_budget"]


class NoiseEnvironment(str, Enum):
    """ITU-R P.372 man-made noise categories."""

    QUIET_RURAL = "quiet_rural"
    RURAL = "rural"
    RESIDENTIAL = "residential"
    CITY = "city"


#: (intercept, slope) for Fam = intercept - slope * log10(f_MHz).
_MAN_MADE = {
    NoiseEnvironment.QUIET_RURAL: (53.6, 28.6),
    NoiseEnvironment.RURAL: (67.2, 27.7),
    NoiseEnvironment.RESIDENTIAL: (72.5, 27.7),
    NoiseEnvironment.CITY: (76.8, 27.7),
}


def _galactic_fa_db(frequency_mhz: float) -> float:
    """Galactic background: the irreducible floor on a quiet HF band."""
    return 52.0 - 23.0 * math.log10(max(frequency_mhz, 0.1))


def _atmospheric_fa_db(
    frequency_mhz: float, sunlit_fraction: float, geomagnetic_latitude_deg: float
) -> float:
    """Lightning-driven atmospheric noise, approximated.

    Tropical thunderstorms are the dominant source worldwide, so the level
    falls with geomagnetic latitude, and it is higher at night when the
    D region stops absorbing distant lightning on its way to the receiver.
    """
    # Anchored on the P.372 mid-latitude curves: about 70 dB at 1.8 MHz,
    # 51 dB at 7 MHz and 32 dB at 28 MHz.  An earlier, steeper fit sat some
    # 15 dB high across the band, which made atmospheric noise swamp the
    # man-made, auroral and precipitation terms so completely that changing
    # them moved the total by less than 0.1 dB.
    base = 78.0 - 32.0 * math.log10(max(frequency_mhz, 0.1))
    night_bonus = 6.0 * (1.0 - sunlit_fraction)
    magnitude = abs(geomagnetic_latitude_deg)
    if magnitude < 45.0:
        # Tropical thunderstorms are the dominant world source.
        latitude_term = 12.0 * (1.0 - magnitude / 45.0)
    else:
        latitude_term = -0.25 * (magnitude - 45.0)
    return base + night_bonus + latitude_term


def _auroral_fa_db(kp: float, geomagnetic_latitude_deg: float) -> float:
    """Auroral-zone noise: hiss and precipitation-driven emission.

    Expressed as a level in dB that rises with Kp, attenuated by the
    distance from the auroral oval in dB as well.  Scaling the *level*
    linearly instead left the term some 25 dB under the atmospheric floor at
    every Kp, so a severe storm at 67 deg magnetic moved the total noise by
    a tenth of a decibel.
    """
    proximity = math.exp(-((abs(geomagnetic_latitude_deg) - 67.0) / 12.0) ** 2)
    if proximity < 1e-6:
        return 0.0
    level = 30.0 + 30.0 * (kp / 9.0) ** 1.5
    return level + 10.0 * math.log10(proximity)


@dataclass(frozen=True)
class NoiseBudget:
    """Noise at the receiver input."""

    total_fa_db: float
    galactic_fa_db: float
    atmospheric_fa_db: float
    man_made_fa_db: float
    auroral_fa_db: float
    weather_fa_db: float
    bandwidth_hz: float
    receiver_noise_figure_db: float

    @property
    def noise_power_dbw(self) -> float:
        """``N = Fa + 10 log10(B) + kT0``, with the receiver's own noise added."""
        external = self.total_fa_db
        combined = 10.0 * math.log10(
            10.0 ** (external / 10.0) + 10.0 ** (self.receiver_noise_figure_db / 10.0)
        )
        return combined + 10.0 * math.log10(self.bandwidth_hz) + KT0_DBW_PER_HZ

    def summary(self) -> dict:
        return {
            "total_fa_db": self.total_fa_db,
            "galactic_fa_db": self.galactic_fa_db,
            "atmospheric_fa_db": self.atmospheric_fa_db,
            "man_made_fa_db": self.man_made_fa_db,
            "auroral_fa_db": self.auroral_fa_db,
            "weather_fa_db": self.weather_fa_db,
            "noise_power_dbw": self.noise_power_dbw,
            "bandwidth_hz": self.bandwidth_hz,
        }


def noise_budget(
    frequency_hz: float,
    bandwidth_hz: float,
    environment: NoiseEnvironment = NoiseEnvironment.RURAL,
    sunlit_fraction: float = 0.5,
    geomagnetic_latitude_deg: float = 45.0,
    kp: float = 2.0,
    rain_rate_mm_h: float = 0.0,
    receiver_noise_figure_db: float = 10.0,
) -> NoiseBudget:
    """Combine every noise source at the receiver.

    The individual terms are noise *figures*; they are summed as powers and
    only then converted back to dB.
    """
    if frequency_hz <= 0.0 or bandwidth_hz <= 0.0:
        raise ValueError("frequency and bandwidth must be positive")

    frequency_mhz = frequency_hz / 1e6
    galactic = _galactic_fa_db(frequency_mhz)
    atmospheric = _atmospheric_fa_db(frequency_mhz, sunlit_fraction, geomagnetic_latitude_deg)
    intercept, slope = _MAN_MADE[environment]
    man_made = intercept - slope * math.log10(max(frequency_mhz, 0.1))
    auroral = _auroral_fa_db(kp, geomagnetic_latitude_deg)

    # Local precipitation static: charged rain and nearby lightning.
    weather = 0.0
    if rain_rate_mm_h > 0.0:
        # Precipitation static: charged droplets discharging on the antenna,
        # plus nearby lightning.  Broadband and, in heavy rain, loud enough
        # to dominate everything else on the lower bands.
        weather = 40.0 + 12.0 * math.log10(1.0 + rain_rate_mm_h)

    terms = [galactic, atmospheric, man_made, weather]
    if auroral > 0.0:
        terms.append(auroral)
    total = 10.0 * math.log10(sum(10.0 ** (t / 10.0) for t in terms))

    return NoiseBudget(
        total_fa_db=total,
        galactic_fa_db=galactic,
        atmospheric_fa_db=atmospheric,
        man_made_fa_db=man_made,
        auroral_fa_db=auroral,
        weather_fa_db=weather,
        bandwidth_hz=bandwidth_hz,
        receiver_noise_figure_db=receiver_noise_figure_db,
    )
