"""Great-circle geometry on a spherical Earth.

A sphere rather than WGS-84: the resulting path-length error is a few parts
in a thousand, three orders of magnitude below the ionospheric uncertainty
that dominates every result downstream.  The choice is deliberate and the
error is bounded, not ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .constants import EARTH_RADIUS_KM

__all__ = ["GeoPoint", "great_circle_distance_km", "initial_bearing_deg",
           "intermediate_point", "path_points", "destination_point"]


@dataclass(frozen=True)
class GeoPoint:
    """A point on the Earth's surface, degrees, north/east positive."""

    lat_deg: float
    lon_deg: float
    name: str = ""

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError(f"latitude {self.lat_deg} outside [-90, 90]")
        if not -180.0 <= self.lon_deg <= 360.0:
            raise ValueError(f"longitude {self.lon_deg} outside [-180, 360]")
        # Normalise to [-180, 180) so bearing arithmetic never wraps twice.
        object.__setattr__(self, "lon_deg", (self.lon_deg + 180.0) % 360.0 - 180.0)

    @property
    def lat_rad(self) -> float:
        return math.radians(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        return math.radians(self.lon_deg)

    def __repr__(self) -> str:  # pragma: no cover - display only
        tag = f" {self.name!r}" if self.name else ""
        return f"GeoPoint({self.lat_deg:.4f}, {self.lon_deg:.4f}{tag})"


def great_circle_distance_km(a: GeoPoint, b: GeoPoint) -> float:
    """Haversine distance -- stable for the short paths where the
    law-of-cosines form loses all its significant digits."""
    dlat = b.lat_rad - a.lat_rad
    dlon = b.lon_rad - a.lon_rad
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(a.lat_rad) * math.cos(b.lat_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def central_angle_rad(a: GeoPoint, b: GeoPoint) -> float:
    return great_circle_distance_km(a, b) / EARTH_RADIUS_KM


def initial_bearing_deg(a: GeoPoint, b: GeoPoint) -> float:
    """Forward azimuth at ``a`` along the great circle to ``b``, 0-360."""
    dlon = b.lon_rad - a.lon_rad
    y = math.sin(dlon) * math.cos(b.lat_rad)
    x = math.cos(a.lat_rad) * math.sin(b.lat_rad) - math.sin(a.lat_rad) * math.cos(
        b.lat_rad
    ) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def destination_point(origin: GeoPoint, bearing_deg: float, distance_km: float) -> GeoPoint:
    """Point reached from ``origin`` on ``bearing_deg`` after ``distance_km``."""
    delta = distance_km / EARTH_RADIUS_KM
    theta = math.radians(bearing_deg)
    lat1, lon1 = origin.lat_rad, origin.lon_rad
    sin_lat = math.sin(lat1) * math.cos(delta) + math.cos(lat1) * math.sin(delta) * math.cos(theta)
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat)))
    lon2 = lon1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(lat1),
        math.cos(delta) - math.sin(lat1) * math.sin(lat2),
    )
    return GeoPoint(math.degrees(lat2), math.degrees(lon2))


def intermediate_point(a: GeoPoint, b: GeoPoint, fraction: float) -> GeoPoint:
    """Point a given fraction of the way from ``a`` to ``b`` (spherical slerp)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction {fraction} outside [0, 1]")
    delta = central_angle_rad(a, b)
    if delta < 1e-12:
        return a
    sin_delta = math.sin(delta)
    p = math.sin((1.0 - fraction) * delta) / sin_delta
    q = math.sin(fraction * delta) / sin_delta
    x = p * math.cos(a.lat_rad) * math.cos(a.lon_rad) + q * math.cos(b.lat_rad) * math.cos(b.lon_rad)
    y = p * math.cos(a.lat_rad) * math.sin(a.lon_rad) + q * math.cos(b.lat_rad) * math.sin(b.lon_rad)
    z = p * math.sin(a.lat_rad) + q * math.sin(b.lat_rad)
    return GeoPoint(
        math.degrees(math.atan2(z, math.hypot(x, y))),
        math.degrees(math.atan2(y, x)),
    )


def path_points(a: GeoPoint, b: GeoPoint, count: int) -> List[GeoPoint]:
    """``count`` points evenly spaced along the great circle, endpoints included."""
    if count < 2:
        raise ValueError("need at least two points")
    return [intermediate_point(a, b, i / (count - 1)) for i in range(count)]
