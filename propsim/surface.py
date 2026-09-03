"""Which parts of a path run over sea and which over land.

A coarse continental outline is enough for the purpose: the reflection loss
of a mid-path ground bounce differs by several dB between sea water and dry
ground, and that difference is worth capturing even from a crude polygon.
Sub-degree coastline detail is not, and is not attempted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .antenna import GROUND_CONSTANTS, GroundType, fresnel_reflection
from .coastlines import COASTLINES
from .geodesy import GeoPoint, intermediate_point

__all__ = ["classify_surface", "path_surface_profile", "ground_reflection_loss_db",
           "SurfaceProfile"]

#: Land outlines, shared with the web globe -- see
#: :mod:`propsim.coastlines` for what they simplify and why.
_LAND_POLYGONS: dict[str, Sequence[Tuple[float, float]]] = dict(COASTLINES)


def _point_in_ring(lon: float, lat: float, ring: Sequence[Tuple[float, float]]) -> bool:
    """Standard ray-casting test."""
    inside = False
    count = len(ring)
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def classify_surface(point: GeoPoint) -> GroundType:
    """Best guess at the ground type under a point."""
    if abs(point.lat_deg) > 66.5:
        # Polar: sea ice or ice cap either way, electrically similar.
        return GroundType.ICE
    for ring in _LAND_POLYGONS.values():
        if _point_in_ring(point.lon_deg, point.lat_deg, ring):
            return GroundType.AVERAGE_GROUND
    return GroundType.SALT_WATER


@dataclass(frozen=True)
class SurfaceProfile:
    """What the path runs over."""

    ground_types: Sequence[GroundType]
    sea_fraction: float

    @property
    def dominant(self) -> GroundType:
        counts: dict[GroundType, int] = {}
        for g in self.ground_types:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)


def path_surface_profile(tx: GeoPoint, rx: GeoPoint, samples: int = 21) -> SurfaceProfile:
    """Classify the surface at points along the great circle."""
    points = [intermediate_point(tx, rx, i / (samples - 1)) for i in range(samples)]
    types = [classify_surface(p) for p in points]
    sea = sum(1 for t in types if t is GroundType.SALT_WATER) / len(types)
    return SurfaceProfile(tuple(types), sea)


def ground_reflection_loss_db(
    elevation_deg: float,
    frequency_hz: float,
    ground: GroundType,
    horizontal: bool = True,
    moisture_factor: float = 1.0,
    sea_state_factor: float = 0.0,
) -> float:
    """Loss at one intermediate ground reflection, dB (positive = loss).

    ``sea_state_factor`` runs 0 (flat) to 1 (heavy swell); a rough sea
    scatters part of the energy out of the specular direction, so a stormy
    ocean is a worse mirror than a calm one even though its conductivity is
    unchanged.
    """
    gamma = fresnel_reflection(
        elevation_deg, frequency_hz, ground, horizontal, moisture_factor
    )
    power_reflected = abs(gamma) ** 2

    if ground is GroundType.SALT_WATER and sea_state_factor > 0.0:
        # Rayleigh roughness: the specular component falls as the surface
        # becomes rough compared with a wavelength at grazing incidence.
        power_reflected *= math.exp(-2.0 * sea_state_factor * math.sin(math.radians(elevation_deg)))

    return -10.0 * math.log10(max(power_reflected, 1e-6))
