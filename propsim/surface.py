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
from .geodesy import GeoPoint, intermediate_point

__all__ = ["classify_surface", "path_surface_profile", "ground_reflection_loss_db",
           "SurfaceProfile"]

#: Very coarse land outlines, (lon, lat) rings.  Deliberately simple: they
#: resolve "is this hop bouncing off the Atlantic or off the Sahara", which
#: is the question the reflection loss actually depends on.
_LAND_POLYGONS: dict[str, Sequence[Tuple[float, float]]] = {
    "eurasia_africa": (
        (-10, 36), (-6, 43), (-9, 44), (-2, 51), (2, 51), (5, 53), (8, 58),
        (12, 55), (19, 55), (21, 60), (25, 65), (30, 70), (60, 70), (100, 76),
        (140, 73), (160, 70), (142, 54), (135, 45), (127, 38), (122, 31),
        (110, 21), (105, 10), (95, 6), (80, 8), (72, 20), (60, 25), (48, 13),
        (43, 12), (35, 15), (32, 5), (40, -5), (40, -20), (35, -25), (25, -34),
        (18, -34), (12, -18), (9, 4), (-5, 5), (-17, 15), (-16, 25), (-10, 36),
    ),
    "americas": (
        (-168, 66), (-160, 71), (-140, 70), (-125, 70), (-95, 70), (-80, 73),
        (-60, 60), (-55, 52), (-65, 45), (-70, 42), (-75, 35), (-81, 26),
        (-90, 29), (-97, 26), (-95, 18), (-88, 21), (-84, 10), (-78, 8),
        (-75, 0), (-70, -18), (-70, -35), (-73, -45), (-70, -55), (-65, -55),
        (-58, -35), (-48, -25), (-35, -8), (-50, 0), (-60, 8), (-80, 22),
        (-97, 26), (-125, 40), (-125, 49), (-135, 58), (-150, 60), (-168, 66),
    ),
    "australia": (
        (113, -22), (114, -34), (129, -32), (138, -35), (146, -39), (150, -37),
        (153, -28), (145, -15), (137, -12), (130, -12), (122, -17), (113, -22),
    ),
    "greenland": ((-55, 60), (-45, 60), (-20, 70), (-20, 82), (-40, 83), (-60, 78), (-55, 60)),
    "antarctica": ((-180, -75), (180, -75), (180, -90), (-180, -90), (-180, -75)),
}


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
