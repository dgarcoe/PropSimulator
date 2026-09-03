"""Geomagnetic field as a tilted, eccentric-free dipole aligned to IGRF-2025.

A dipole cannot reproduce regional anomalies (the South Atlantic Anomaly is
the obvious casualty), but it carries the two quantities the magnetoionic
theory actually needs -- field strength and the angle between the field and
the ray -- with a few percent accuracy over most of the globe.

The field is returned as a :class:`MagneticField` dataclass rather than a
dict.  Consumers reach for ``field.east`` / ``field.north`` / ``field.up``;
a misspelt component is an ``AttributeError`` at the first call, not a
silently-zero dot product that quietly freezes the extraordinary mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import EARTH_RADIUS_KM, GYRO_FREQ_COEFF_HZ
from .geodesy import GeoPoint

__all__ = ["MagneticField", "magnetic_field", "geomagnetic_latitude_deg",
           "gyrofrequency_hz", "field_ray_angle_rad", "NORTH_GEOMAGNETIC_POLE"]

#: IGRF-2025 north geomagnetic pole (the dipole axis intersection).
NORTH_GEOMAGNETIC_POLE = GeoPoint(80.7, -72.7, "north geomagnetic pole")

#: Equatorial surface field of the IGRF-2025 dipole, tesla.
DIPOLE_MOMENT_TESLA = 3.02e-5


@dataclass(frozen=True)
class MagneticField:
    """Local geomagnetic field in the local horizon frame.

    Components are in tesla, in the East / North / Up right-handed frame.
    """

    east: float
    north: float
    up: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.east**2 + self.north**2 + self.up**2)

    @property
    def horizontal(self) -> float:
        return math.hypot(self.east, self.north)

    @property
    def inclination_deg(self) -> float:
        """Dip angle: positive downward, following the usual convention."""
        return math.degrees(math.atan2(-self.up, self.horizontal))

    @property
    def declination_deg(self) -> float:
        """Angle of the horizontal field east of true north."""
        return math.degrees(math.atan2(self.east, self.north))

    @property
    def gyrofrequency_hz(self) -> float:
        return GYRO_FREQ_COEFF_HZ * self.magnitude

    def unit(self) -> tuple[float, float, float]:
        magnitude = self.magnitude
        if magnitude <= 0.0:
            raise ValueError("zero magnetic field has no direction")
        return (self.east / magnitude, self.north / magnitude, self.up / magnitude)


def _geomagnetic_colatitude_rad(point: GeoPoint) -> float:
    """Angle between the point and the north geomagnetic pole."""
    pole = NORTH_GEOMAGNETIC_POLE
    cos_theta = math.sin(point.lat_rad) * math.sin(pole.lat_rad) + math.cos(
        point.lat_rad
    ) * math.cos(pole.lat_rad) * math.cos(point.lon_rad - pole.lon_rad)
    return math.acos(max(-1.0, min(1.0, cos_theta)))


def geomagnetic_latitude_deg(point: GeoPoint) -> float:
    """Dipole latitude: 0 at the magnetic equator, +-90 at the poles."""
    return 90.0 - math.degrees(_geomagnetic_colatitude_rad(point))


def magnetic_field(point: GeoPoint, height_km: float = 0.0) -> MagneticField:
    """Dipole field at ``point``, ``height_km`` above the surface."""
    if height_km < -1.0:
        raise ValueError(f"height {height_km} km is below the Earth")
    colatitude = _geomagnetic_colatitude_rad(point)
    radius_ratio = (EARTH_RADIUS_KM / (EARTH_RADIUS_KM + height_km)) ** 3

    # Dipole field in geomagnetic spherical coordinates.
    b_radial = -2.0 * DIPOLE_MOMENT_TESLA * radius_ratio * math.cos(colatitude)
    b_theta = -DIPOLE_MOMENT_TESLA * radius_ratio * math.sin(colatitude)

    # b_theta points toward increasing geomagnetic colatitude (magnetic
    # south); the local "up" component is the outward radial one.
    b_up = b_radial
    b_magnetic_north = -b_theta

    # Rotate the horizontal part from magnetic north to true north by the
    # bearing of the geomagnetic pole as seen from this point.
    pole = NORTH_GEOMAGNETIC_POLE
    dlon = pole.lon_rad - point.lon_rad
    y = math.sin(dlon) * math.cos(pole.lat_rad)
    x = math.cos(point.lat_rad) * math.sin(pole.lat_rad) - math.sin(
        point.lat_rad
    ) * math.cos(pole.lat_rad) * math.cos(dlon)
    pole_bearing = math.atan2(y, x)

    return MagneticField(
        east=b_magnetic_north * math.sin(pole_bearing),
        north=b_magnetic_north * math.cos(pole_bearing),
        up=b_up,
    )


def gyrofrequency_hz(point: GeoPoint, height_km: float = 0.0) -> float:
    """Electron gyrofrequency at a point and height."""
    return magnetic_field(point, height_km).gyrofrequency_hz


def ray_direction_enu(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Unit propagation vector in the East / North / Up frame."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (
        math.sin(az) * math.cos(el),
        math.cos(az) * math.cos(el),
        math.sin(el),
    )


def field_ray_angle_rad(
    field: MagneticField, azimuth_deg: float, elevation_deg: float
) -> float:
    """Angle between the field vector and the ray direction, 0..pi.

    This is the ``theta`` of the Appleton-Hartree formula.  Both operands
    are in the same East/North/Up frame and the dot product is taken over
    named attributes, so the two cannot silently disagree.
    """
    bx, by, bz = field.unit()
    kx, ky, kz = ray_direction_enu(azimuth_deg, elevation_deg)
    cos_theta = bx * kx + by * ky + bz * kz
    return math.acos(max(-1.0, min(1.0, cos_theta)))
