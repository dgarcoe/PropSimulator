"""The complete description of a circuit: stations, path, time, conditions.

Everything the physics needs travels together in one frozen, validated
:class:`Scenario`.  No evaluation function in this package accepts a bare
frequency and a handful of optional station parameters; they all take the
scenario.  A code path therefore cannot be reached with the power, gains,
bandwidth, noise figure or required SNR missing and quietly filled in by
defaults -- the situation that produces a lowest-usable-frequency computed
from a -100 dBW transmitter into a 10 Hz bandwidth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .antenna import AntennaSpec, GroundType
from .geodesy import GeoPoint, great_circle_distance_km, initial_bearing_deg
from .noise import NoiseEnvironment
from .spaceweather import SpaceWeather

__all__ = ["Station", "Weather", "Scenario"]


@dataclass(frozen=True)
class Station:
    """One end of the circuit, complete.

    Every field here is required by the link budget.  There are no optional
    numeric parameters and no defaults that stand in for real equipment.
    """

    location: GeoPoint
    antenna: AntennaSpec
    transmit_power_w: float = 100.0
    bandwidth_hz: float = 2400.0
    receiver_noise_figure_db: float = 12.0
    required_snr_db: float = 6.0
    noise_environment: NoiseEnvironment = NoiseEnvironment.RURAL
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.001 <= self.transmit_power_w <= 1e6:
            raise ValueError(
                f"transmit power {self.transmit_power_w} W outside 1 mW - 1 MW"
            )
        if not 1.0 <= self.bandwidth_hz <= 1e6:
            raise ValueError(
                f"bandwidth {self.bandwidth_hz} Hz outside 1 Hz - 1 MHz; a value "
                "near the lower bound usually means the field was never set"
            )
        if not 0.0 <= self.receiver_noise_figure_db <= 40.0:
            raise ValueError(f"receiver noise figure {self.receiver_noise_figure_db} dB is implausible")
        if not -20.0 <= self.required_snr_db <= 60.0:
            raise ValueError(f"required SNR {self.required_snr_db} dB is implausible")

    @property
    def transmit_power_dbw(self) -> float:
        return 10.0 * math.log10(self.transmit_power_w)


@dataclass(frozen=True)
class Weather:
    """Surface weather along the path.

    Rain and humidity change the noise and the ground reflection far more
    than they change the wave itself; the direct rain attenuation at HF is
    tenths of a decibel, and is charged as such.
    """

    rain_rate_mm_h: float = 0.0
    #: Multiplies ground conductivity: >1 wet, <1 frozen.
    ground_moisture_factor: float = 1.0
    #: 0 flat, 1 heavy swell -- roughens sea reflections.
    sea_state: float = 0.0
    freezing: bool = False

    def __post_init__(self) -> None:
        if self.rain_rate_mm_h < 0.0:
            raise ValueError("rain rate cannot be negative")
        if not 0.01 <= self.ground_moisture_factor <= 100.0:
            raise ValueError("ground moisture factor outside 0.01 - 100")
        if not 0.0 <= self.sea_state <= 1.0:
            raise ValueError("sea state outside 0 - 1")

    @property
    def effective_moisture_factor(self) -> float:
        """Frozen ground is a far poorer conductor than merely dry ground."""
        return self.ground_moisture_factor * (0.1 if self.freezing else 1.0)


@dataclass(frozen=True)
class Scenario:
    """A circuit, fully specified."""

    transmitter: Station
    receiver: Station
    when: datetime
    space_weather: SpaceWeather = field(default_factory=SpaceWeather)
    weather: Weather = field(default_factory=Weather)
    #: Number of great-circle sample points for the equivalent column.
    path_samples: int = 9

    def __post_init__(self) -> None:
        if self.when.tzinfo is None:
            raise ValueError("scenario time must be timezone-aware")
        if self.path_samples < 3:
            raise ValueError("need at least three path samples")
        if self.distance_km < 1.0:
            raise ValueError("transmitter and receiver are at the same place")

    @property
    def distance_km(self) -> float:
        return great_circle_distance_km(self.transmitter.location, self.receiver.location)

    @property
    def bearing_deg(self) -> float:
        return initial_bearing_deg(self.transmitter.location, self.receiver.location)

    @property
    def reverse_bearing_deg(self) -> float:
        return initial_bearing_deg(self.receiver.location, self.transmitter.location)

    def summary(self) -> dict:
        return {
            "transmitter": self.transmitter.name or "TX",
            "receiver": self.receiver.name or "RX",
            "tx_lat": self.transmitter.location.lat_deg,
            "tx_lon": self.transmitter.location.lon_deg,
            "rx_lat": self.receiver.location.lat_deg,
            "rx_lon": self.receiver.location.lon_deg,
            "distance_km": self.distance_km,
            "bearing_deg": self.bearing_deg,
            "when": self.when.astimezone(timezone.utc).isoformat(),
            "space_weather": self.space_weather.summary(),
            "transmit_power_w": self.transmitter.transmit_power_w,
            "bandwidth_hz": self.receiver.bandwidth_hz,
            "required_snr_db": self.receiver.required_snr_db,
        }
