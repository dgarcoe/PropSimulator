"""Antenna gain against elevation, including the ground reflection.

Heights are in **metres** everywhere in this module and in the objects it
exchanges with the rest of the core.  There is no kilometre form of an
antenna height anywhere in PropSimulator, and therefore no place for a 20 m
mast to be converted to 0.02 and then floored to a 0.5 that means 500 m.
:class:`AntennaSpec` validates its height against limits that are absurd in
metres and impossible in kilometres, so a unit slip fails loudly.

A horizontal antenna over ground is treated by image theory: the ground
reflection behaves like a second, inverted antenna the same distance below
the surface.  The two interfere, and the resulting pattern -- nulls at the
horizon and at multiples of a wavelength of path difference -- is what
decides how much power is available at the low elevation angles that long
skywave paths need.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .constants import SPEED_OF_LIGHT

__all__ = ["AntennaType", "GroundType", "AntennaSpec", "GROUND_CONSTANTS",
           "fresnel_reflection", "complex_permittivity", "surface_impedance"]


class AntennaType(str, Enum):
    HORIZONTAL_DIPOLE = "horizontal_dipole"
    INVERTED_V = "inverted_v"
    VERTICAL_QUARTER_WAVE = "vertical_quarter_wave"
    SHORT_VERTICAL = "short_vertical"
    ISOTROPIC = "isotropic"


class GroundType(str, Enum):
    SALT_WATER = "salt_water"
    FRESH_WATER = "fresh_water"
    WET_GROUND = "wet_ground"
    AVERAGE_GROUND = "average_ground"
    DRY_GROUND = "dry_ground"
    URBAN = "urban"
    ICE = "ice"


#: (relative permittivity, conductivity S/m) -- the standard ITU-R P.527 set.
GROUND_CONSTANTS = {
    GroundType.SALT_WATER: (80.0, 5.0),
    GroundType.FRESH_WATER: (80.0, 3e-3),
    GroundType.WET_GROUND: (30.0, 1e-2),
    GroundType.AVERAGE_GROUND: (15.0, 5e-3),
    GroundType.DRY_GROUND: (4.0, 1e-3),
    GroundType.URBAN: (5.0, 1e-3),
    GroundType.ICE: (3.0, 3e-5),
}


def complex_permittivity(
    frequency_hz: float, ground: GroundType, moisture_factor: float = 1.0
) -> complex:
    """Relative permittivity of the ground including its conduction current.

    ``eps_r - j sigma / (omega eps0)``.  The imaginary part is what makes a
    ground lossy at all: at 3.65 MHz it is 24.6 for average ground and
    24 600 for sea water, and that four-order-of-magnitude difference is the
    whole reason a ground wave crosses an ocean and dies over a continent.
    """
    permittivity, conductivity = GROUND_CONSTANTS[ground]
    conductivity = max(conductivity * moisture_factor, 1e-6)
    return complex(
        permittivity,
        -conductivity / (2.0 * math.pi * frequency_hz * 8.8541878128e-12),
    )


def surface_impedance(
    frequency_hz: float, ground: GroundType, moisture_factor: float = 1.0
) -> complex:
    """Normalised surface impedance ``Delta = 1 / sqrt(eps_c)``.

    The single number that governs everything about a ground wave: how fast
    it is attenuated, how far its front tilts forward -- and therefore how
    much of it a horizontal antenna can pick up -- and how its field grows
    with height above the surface.  Small over sea water, an appreciable
    fraction of unity over dry rock.
    """
    return 1.0 / cmath.sqrt(complex_permittivity(frequency_hz, ground, moisture_factor))


def fresnel_reflection(
    elevation_deg: float,
    frequency_hz: float,
    ground: GroundType,
    horizontal: bool,
    moisture_factor: float = 1.0,
) -> complex:
    """Fresnel reflection coefficient of the ground at grazing incidence.

    ``moisture_factor`` scales the conductivity: rain raises it, freezing
    lowers it sharply, and both change the low-angle gain of a vertical
    antenna far more than they change anything else in the link.
    """
    epsilon = complex_permittivity(frequency_hz, ground, moisture_factor)

    psi = math.radians(elevation_deg)          # elevation above the surface
    sin_psi = math.sin(psi)
    root = cmath.sqrt(epsilon - math.cos(psi) ** 2)

    if horizontal:
        return (sin_psi - root) / (sin_psi + root)
    return (epsilon * sin_psi - root) / (epsilon * sin_psi + root)


@dataclass(frozen=True)
class AntennaSpec:
    """One antenna: what it is, how high it is and what it loses.

    Attributes
    ----------
    height_m:
        Height of the radiator above ground, **metres**.
    design_frequency_hz:
        The frequency the antenna is cut for.  Operating away from it costs
        efficiency, which :meth:`gain_dbi` charges.
    boom_length_m:
        Physical length of the radiator, metres.  A vertical much shorter
        than a quarter wave radiates poorly, and that is charged too.
    feedline_loss_db, trap_loss_db:
        Fixed losses between the transmitter and the radiator.
    azimuth_deg:
        Direction the antenna points, for the directional types.  ``None``
        means omnidirectional in azimuth.
    receive_only:
        Receiving antennas may be deliberately lossy (a beverage is);
        efficiency losses are then not charged against transmit power.
    """

    antenna_type: AntennaType = AntennaType.HORIZONTAL_DIPOLE
    height_m: float = 10.0
    design_frequency_hz: float = 14.2e6
    boom_length_m: float = 10.0
    ground: GroundType = GroundType.AVERAGE_GROUND
    feedline_loss_db: float = 0.5
    trap_loss_db: float = 0.0
    azimuth_deg: Optional[float] = None
    mobile: bool = False
    receive_only: bool = False
    #: Gain this model does not derive, added to the computed pattern: a
    #: beam's directivity, a phased array, an amplifierless preselector.
    #: It is a declaration by the operator, not a result -- the package
    #: models dipoles and verticals, and cannot invent a Yagi pattern for
    #: you.  Kept separate from the computed gain for exactly that reason.
    extra_gain_dbi: float = 0.0

    def __post_init__(self) -> None:
        # 0.1 m to 300 m: a legitimate antenna height in metres.  A value
        # that arrived as kilometres (0.02 for a 20 m mast) is rejected here
        # rather than silently floored into something 25x too high.
        if not 0.1 <= self.height_m <= 300.0:
            raise ValueError(
                f"antenna height {self.height_m} m is outside 0.1-300 m; "
                "note this field is in METRES, not kilometres"
            )
        if self.design_frequency_hz <= 0.0:
            raise ValueError("design frequency must be positive")
        if self.boom_length_m <= 0.0:
            raise ValueError("radiator length must be positive")
        for name in ("feedline_loss_db", "trap_loss_db"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if not -20.0 <= self.extra_gain_dbi <= 40.0:
            raise ValueError(
                f"extra gain {self.extra_gain_dbi} dBi is outside -20 to 40 dBi"
            )

    @property
    def height_km(self) -> float:
        """Height in kilometres, for the ray tracer's starting radius.

        Provided as a derived read-only view so the conversion happens in
        exactly one place and cannot be applied twice.
        """
        return self.height_m / 1000.0

    # -- pattern ---------------------------------------------------------
    def _free_space_gain_dbi(self, elevation_deg: float) -> float:
        """Gain of the radiator alone, before the ground reflection."""
        if self.antenna_type is AntennaType.ISOTROPIC:
            return 0.0
        if self.antenna_type in (
            AntennaType.VERTICAL_QUARTER_WAVE,
            AntennaType.SHORT_VERTICAL,
        ):
            # A short monopole's pattern is cos(elevation) in field.
            cos_el = math.cos(math.radians(elevation_deg))
            return 10.0 * math.log10(max(cos_el**2, 1e-6)) + 1.76
        # Half-wave dipole broadside; the inverted V is a little flatter.
        base = 2.15 if self.antenna_type is AntennaType.HORIZONTAL_DIPOLE else 1.6
        return base

    def _ground_reflection_gain_db(
        self, elevation_deg: float, frequency_hz: float, moisture_factor: float
    ) -> float:
        """Interference between the antenna and its image in the ground."""
        wavelength_m = SPEED_OF_LIGHT / frequency_hz
        horizontal = self.antenna_type in (
            AntennaType.HORIZONTAL_DIPOLE,
            AntennaType.INVERTED_V,
        )
        gamma = fresnel_reflection(
            elevation_deg, frequency_hz, self.ground, horizontal, moisture_factor
        )
        # Path difference between the direct ray and the ground-reflected
        # ray leaving at the same elevation: 2 h sin(elevation).
        phase = 2.0 * math.pi * (2.0 * self.height_m / wavelength_m) * math.sin(
            math.radians(elevation_deg)
        )
        # The Fresnel coefficient already carries the polarisation-dependent
        # sign: the horizontal coefficient tends to -1 at grazing incidence,
        # which is exactly what produces the horizon null, while the vertical
        # one stays positive over sea water, which is why a vertical there
        # keeps its low-angle gain.  Applying a further inversion by hand
        # would cancel that and turn the pattern inside out -- every lobe
        # would land where a null belongs.
        total = 1.0 + gamma * cmath.exp(1j * phase)
        return 10.0 * math.log10(max(abs(total) ** 2, 1e-6))

    def _efficiency_loss_db(self, frequency_hz: float) -> float:
        """Losses from a short radiator and from operating off design."""
        if self.receive_only:
            return 0.0
        loss = 0.0
        wavelength_m = SPEED_OF_LIGHT / frequency_hz

        if self.antenna_type is AntennaType.SHORT_VERTICAL:
            # Radiation resistance of a short monopole falls as (l/lambda)^2,
            # so its efficiency against a fixed loss resistance collapses.
            electrical_length = self.boom_length_m / wavelength_m
            radiation_resistance = 395.0 * electrical_length**2
            loss_resistance = 15.0 if not self.mobile else 25.0
            efficiency = radiation_resistance / (radiation_resistance + loss_resistance)
            loss += -10.0 * math.log10(max(efficiency, 1e-6))

        # Off-design operation: a resonant antenna detunes.
        octaves = abs(math.log2(frequency_hz / self.design_frequency_hz))
        loss += 3.0 * octaves**2

        if self.mobile:
            loss += 3.0        # compromised ground plane and siting
        return loss

    def gain_dbi(
        self,
        elevation_deg: float,
        frequency_hz: float,
        bearing_deg: Optional[float] = None,
        moisture_factor: float = 1.0,
    ) -> float:
        """Total gain toward a given elevation and bearing, dBi.

        Includes the free-space pattern, the ground reflection, radiator
        efficiency, feedline and traps, and the azimuth pattern where the
        antenna has one.
        """
        if not 0.0 <= elevation_deg <= 90.0:
            raise ValueError(f"elevation {elevation_deg} outside [0, 90]")

        gain = self._free_space_gain_dbi(elevation_deg)
        gain += self._ground_reflection_gain_db(
            elevation_deg, frequency_hz, moisture_factor
        )
        gain -= self._efficiency_loss_db(frequency_hz)
        gain -= self.feedline_loss_db + self.trap_loss_db
        gain += self.extra_gain_dbi

        if self.azimuth_deg is not None and bearing_deg is not None:
            offset = math.radians((bearing_deg - self.azimuth_deg + 180.0) % 360.0 - 180.0)
            # A dipole is a figure of eight broadside to the wire.
            gain += 10.0 * math.log10(max(math.cos(offset) ** 2, 1e-3))

        return gain

    def ground_wave_gain_dbi(
        self,
        frequency_hz: float,
        bearing_deg: Optional[float] = None,
        moisture_factor: float = 1.0,
    ) -> float:
        """Gain into (or out of) the surface wave running along the ground.

        This is deliberately *not* ``gain_dbi(0.0, ...)``.  At grazing
        incidence the Fresnel coefficient tends to -1 for both
        polarisations, so the direct ray and its image cancel and the
        over-ground pattern collapses to a null: the space wave really does
        vanish along the surface.  What survives there is the surface wave,
        a different field with a different excitation, and asking the
        space-wave pattern about it returns -70 dBi for every antenna ever
        built.

        Three things set the excitation:

        * **Image.**  A vertical over ground radiates into a hemisphere, so
          it gains 3 dB over the same radiator in free space.  A short
          vertical monopole therefore lands on 4.77 dBi, which is the
          300 mV/m at 1 km per kilowatt that every ground-wave chart is
          drawn against -- derived here, not asserted.
        * **Polarisation.**  The surface wave is vertically polarised.  Its
          front tilts forward by the surface impedance ``Delta``, and that
          tilt is the *only* horizontal electric field there is, so a
          horizontal antenna couples to it down by ``|Delta|`` -- 15 dB over
          average ground, 44 dB over sea.  This is an upper bound: it
          assumes the wire has a component in the plane of propagation.  A
          horizontal antenna is a poor ground-wave antenna, and over salt
          water it is hopeless.
        * **Height.**  The surface wave's field varies as
          ``1 + j k h Delta`` in the first few tens of metres.  At HF over
          sea that is nothing at all, which is why ground-wave coverage is
          famously indifferent to how high the antenna is.
        """
        if frequency_hz <= 0.0:
            raise ValueError("frequency must be positive")
        delta = surface_impedance(frequency_hz, self.ground, moisture_factor)
        horizontal = self.antenna_type in (
            AntennaType.HORIZONTAL_DIPOLE,
            AntennaType.INVERTED_V,
        )

        gain = self._free_space_gain_dbi(0.0)
        if horizontal:
            gain += 20.0 * math.log10(max(abs(delta), 1e-12))
        else:
            gain += 10.0 * math.log10(2.0)

        wavenumber = 2.0 * math.pi * frequency_hz / SPEED_OF_LIGHT
        gain += 20.0 * math.log10(
            abs(1.0 + 1j * wavenumber * self.height_m * delta)
        )

        gain -= self._efficiency_loss_db(frequency_hz)
        gain -= self.feedline_loss_db + self.trap_loss_db
        gain += self.extra_gain_dbi

        if self.azimuth_deg is not None and bearing_deg is not None:
            offset = math.radians((bearing_deg - self.azimuth_deg + 180.0) % 360.0 - 180.0)
            gain += 10.0 * math.log10(max(math.cos(offset) ** 2, 1e-3))
        return gain

    def best_elevation_deg(self, frequency_hz: float) -> float:
        """Elevation of the pattern's main lobe, for reporting."""
        return max(
            (e * 0.5 for e in range(1, 180)),
            key=lambda e: self.gain_dbi(e, frequency_hz),
        )
