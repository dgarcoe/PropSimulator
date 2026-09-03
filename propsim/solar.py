"""Apparent solar position from J2000 and the illumination of a path.

Accuracy is the low-precision Astronomical Almanac series: better than
0.01 deg in declination over 1950-2050, which is far tighter than anything
the ionospheric model can exploit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from .constants import J2000_JD
from .geodesy import GeoPoint, great_circle_distance_km, path_points

__all__ = ["SolarPosition", "solar_position", "solar_zenith_angle_deg",
           "subsolar_point", "PathIllumination", "illuminate_path",
           "local_solar_time_hours", "julian_day", "days_since_j2000"]

#: Sun's angular radius plus refraction: the geometric zenith angle at which
#: the disc centre sits when the upper limb touches the horizon.
SUNRISE_ZENITH_DEG = 90.833


def julian_day(when: datetime) -> float:
    """Julian Day number for a timezone-aware UTC datetime."""
    if when.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (use timezone.utc)")
    utc = when.astimezone(timezone.utc)
    year, month = utc.year, utc.month
    day = (
        utc.day
        + (utc.hour + (utc.minute + (utc.second + utc.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    )
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


def days_since_j2000(when: datetime) -> float:
    return julian_day(when) - J2000_JD


@dataclass(frozen=True)
class SolarPosition:
    """Apparent geocentric position of the Sun."""

    declination_deg: float
    right_ascension_deg: float
    ecliptic_longitude_deg: float
    equation_of_time_min: float
    greenwich_hour_angle_deg: float

    @property
    def subsolar_lat_deg(self) -> float:
        return self.declination_deg

    @property
    def subsolar_lon_deg(self) -> float:
        # GHA is measured westward from Greenwich, so the sub-solar meridian
        # is its negative wrapped into [-180, 180).
        return (-self.greenwich_hour_angle_deg + 180.0) % 360.0 - 180.0


def solar_position(when: datetime) -> SolarPosition:
    """Apparent solar position using the low-precision almanac series."""
    n = days_since_j2000(when)

    mean_longitude = (280.460 + 0.9856474 * n) % 360.0
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)

    ecliptic_longitude = (
        mean_longitude
        + 1.915 * math.sin(mean_anomaly)
        + 0.020 * math.sin(2.0 * mean_anomaly)
    ) % 360.0
    lam = math.radians(ecliptic_longitude)

    obliquity = math.radians(23.439 - 0.0000004 * n)

    declination = math.asin(math.sin(obliquity) * math.sin(lam))
    right_ascension = math.degrees(
        math.atan2(math.cos(obliquity) * math.sin(lam), math.cos(lam))
    ) % 360.0

    # Greenwich mean sidereal time -> hour angle of the true Sun.
    gmst = (280.46061837 + 360.98564736629 * n) % 360.0
    greenwich_hour_angle = (gmst - right_ascension) % 360.0

    equation_of_time = ((mean_longitude - right_ascension + 180.0) % 360.0 - 180.0) * 4.0

    return SolarPosition(
        declination_deg=math.degrees(declination),
        right_ascension_deg=right_ascension,
        ecliptic_longitude_deg=ecliptic_longitude,
        equation_of_time_min=equation_of_time,
        greenwich_hour_angle_deg=greenwich_hour_angle,
    )


def subsolar_point(when: datetime) -> GeoPoint:
    """The point where the Sun is exactly overhead."""
    sun = solar_position(when)
    return GeoPoint(sun.subsolar_lat_deg, sun.subsolar_lon_deg, "subsolar")


def solar_zenith_angle_deg(point: GeoPoint, when: datetime) -> float:
    """Zenith angle of the Sun at ``point``; 0 = overhead, >90 = below horizon."""
    sun = solar_position(when)
    dec = math.radians(sun.declination_deg)
    hour_angle = math.radians((sun.greenwich_hour_angle_deg + point.lon_deg + 180.0) % 360.0 - 180.0)
    cos_zenith = (
        math.sin(point.lat_rad) * math.sin(dec)
        + math.cos(point.lat_rad) * math.cos(dec) * math.cos(hour_angle)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def local_solar_time_hours(point: GeoPoint, when: datetime) -> float:
    """Apparent local solar time, 0-24 h."""
    sun = solar_position(when)
    hour_angle = (sun.greenwich_hour_angle_deg + point.lon_deg + 180.0) % 360.0 - 180.0
    return (hour_angle / 15.0 + 12.0) % 24.0


@dataclass(frozen=True)
class PathIllumination:
    """Where the terminator falls across a great-circle path."""

    zenith_angles_deg: List[float]
    sunlit_fraction: float
    crosses_terminator: bool
    subsolar: GeoPoint
    declination_deg: float
    #: Season as seen from the transmitter's hemisphere, for layer scaling.
    seasonal_phase: float

    @property
    def is_fully_dark(self) -> bool:
        return self.sunlit_fraction <= 0.0

    @property
    def is_fully_sunlit(self) -> bool:
        return self.sunlit_fraction >= 1.0


def illuminate_path(
    tx: GeoPoint, rx: GeoPoint, when: datetime, samples: int = 9
) -> PathIllumination:
    """Sample the solar zenith angle along the path and locate the terminator.

    ``sunlit_fraction`` is computed by linear interpolation of the zenith
    angle between samples, so a terminator crossing between two sample
    points is resolved rather than snapped to the nearest sample.
    """
    points = path_points(tx, rx, samples)
    zeniths = [solar_zenith_angle_deg(p, when) for p in points]

    lit = [z < SUNRISE_ZENITH_DEG for z in zeniths]
    segments = len(zeniths) - 1
    sunlit = 0.0
    for i in range(segments):
        z0, z1 = zeniths[i], zeniths[i + 1]
        if lit[i] and lit[i + 1]:
            sunlit += 1.0
        elif lit[i] != lit[i + 1]:
            # Fraction of this segment on the sunlit side of the crossing.
            crossing = (SUNRISE_ZENITH_DEG - z0) / (z1 - z0)
            sunlit += crossing if lit[i] else 1.0 - crossing
    sunlit_fraction = sunlit / segments if segments else float(lit[0])

    sun = solar_position(when)
    # +1 at local summer solstice, -1 at local winter, for the tx hemisphere.
    hemisphere = 1.0 if tx.lat_deg >= 0.0 else -1.0
    seasonal_phase = hemisphere * math.sin(math.radians(sun.declination_deg)) / math.sin(
        math.radians(23.439)
    )

    return PathIllumination(
        zenith_angles_deg=zeniths,
        sunlit_fraction=max(0.0, min(1.0, sunlit_fraction)),
        crosses_terminator=any(lit) and not all(lit),
        subsolar=GeoPoint(sun.subsolar_lat_deg, sun.subsolar_lon_deg, "subsolar"),
        declination_deg=sun.declination_deg,
        seasonal_phase=max(-1.0, min(1.0, seasonal_phase)),
    )
