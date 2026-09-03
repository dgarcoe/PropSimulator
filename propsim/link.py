"""Link budget: from transmitter power to signal-to-noise ratio.

The chain is the Friis equation with the ionosphere's losses inserted:

    Prx = Ptx + Gtx + Grx - Lspread - Labsorption - Lground - Lweather

``Lspread`` is free-space spreading over the **ray's own path length**, not
over the ground range: a 2000 km hop that climbs to 300 km travels further
than 2000 km and spreads accordingly.

Every loss in the breakdown is a term that was actually subtracted.  The
summary is produced by decomposing the number that was computed, and
:meth:`LinkBudget.verify` re-adds the terms and refuses to report a budget
whose parts do not reconstruct its total.  A loss that is displayed but not
charged -- or charged twice under two names -- cannot survive that check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .noise import NoiseBudget

__all__ = ["LinkBudget", "build_link_budget", "free_space_loss_db",
           "field_strength_dbuv_per_m"]


def free_space_loss_db(distance_km: float, frequency_hz: float) -> float:
    """Spreading loss over a given path length.

    ``32.44 + 20 log10(f_MHz) + 20 log10(d_km)`` is the familiar form; it is
    written here from ``(4 pi d / lambda)^2`` directly so the constant is
    derived rather than remembered.
    """
    if distance_km <= 0.0 or frequency_hz <= 0.0:
        raise ValueError("distance and frequency must be positive")
    wavelength_m = 299792458.0 / frequency_hz
    return 20.0 * math.log10(4.0 * math.pi * distance_km * 1000.0 / wavelength_m)


def field_strength_dbuv_per_m(received_power_dbw: float, gain_dbi: float,
                              frequency_hz: float) -> float:
    """Field strength implied by a received power and antenna gain.

    Exact inverse of the effective-aperture relation, so converting to a
    field and back reproduces the original power.
    """
    wavelength_m = 299792458.0 / frequency_hz
    aperture_db = 10.0 * math.log10(wavelength_m**2 / (4.0 * math.pi))
    # P = E^2 / (120 pi) * Aeff  ->  E[dBuV/m] = P[dBW] - G - Aeff + 145.8
    return received_power_dbw - gain_dbi - aperture_db + 10.0 * math.log10(
        120.0 * math.pi
    ) + 120.0


def rain_attenuation_db(
    frequency_hz: float, rain_rate_mm_h: float, path_length_km: float
) -> float:
    """Direct absorption of the wave by rain along the path.

    At HF this is genuinely small -- tenths of a dB -- because the drops are
    minute compared with the wavelength.  It is computed, reported *and
    subtracted*: a loss that appears in the breakdown but never reaches the
    received power is worse than no loss at all, because it makes the budget
    look complete while being wrong.
    """
    if rain_rate_mm_h <= 0.0 or path_length_km <= 0.0:
        return 0.0
    frequency_ghz = frequency_hz / 1e9
    # ITU-R P.838 power law, in its low-frequency limit.
    specific_db_per_km = 4.0e-4 * (frequency_ghz**2) * (rain_rate_mm_h**0.9)
    # Only the lowest few kilometres of the path are in rain.
    effective_km = min(path_length_km, 10.0)
    return specific_db_per_km * effective_km


@dataclass(frozen=True)
class LinkBudget:
    """A complete, self-checking link budget for one propagation mode."""

    frequency_hz: float
    transmit_power_dbw: float
    transmit_gain_dbi: float
    receive_gain_dbi: float
    spreading_loss_db: float
    absorption_loss_db: float
    ground_reflection_loss_db: float
    rain_attenuation_db: float
    received_power_dbw: float
    noise: NoiseBudget
    required_snr_db: float
    path_length_km: float
    hops: int

    @property
    def snr_db(self) -> float:
        return self.received_power_dbw - self.noise.noise_power_dbw

    @property
    def margin_db(self) -> float:
        return self.snr_db - self.required_snr_db

    @property
    def total_loss_db(self) -> float:
        return (
            self.spreading_loss_db
            + self.absorption_loss_db
            + self.ground_reflection_loss_db
            + self.rain_attenuation_db
        )

    def verify(self) -> None:
        """Re-derive the received power from the parts and compare."""
        rebuilt = (
            self.transmit_power_dbw
            + self.transmit_gain_dbi
            + self.receive_gain_dbi
            - self.total_loss_db
        )
        if not math.isclose(rebuilt, self.received_power_dbw, abs_tol=1e-9):
            raise ValueError(
                f"link budget does not reconstruct: parts give {rebuilt:.4f} dBW "
                f"but the stored received power is {self.received_power_dbw:.4f} dBW. "
                "Some loss is reported without being charged, or charged twice."
            )

    def breakdown(self) -> Dict[str, float]:
        self.verify()
        return {
            "transmit_power_dbw": self.transmit_power_dbw,
            "transmit_gain_dbi": self.transmit_gain_dbi,
            "receive_gain_dbi": self.receive_gain_dbi,
            "spreading_loss_db": -self.spreading_loss_db,
            "absorption_loss_db": -self.absorption_loss_db,
            "ground_reflection_loss_db": -self.ground_reflection_loss_db,
            "rain_attenuation_db": -self.rain_attenuation_db,
            "received_power_dbw": self.received_power_dbw,
            "noise_power_dbw": self.noise.noise_power_dbw,
            "snr_db": self.snr_db,
            "required_snr_db": self.required_snr_db,
            "margin_db": self.margin_db,
        }


def build_link_budget(
    frequency_hz: float,
    transmit_power_w: float,
    transmit_gain_dbi: float,
    receive_gain_dbi: float,
    path_length_km: float,
    absorption_loss_db: float,
    ground_reflection_loss_db: float,
    noise: NoiseBudget,
    required_snr_db: float,
    hops: int = 1,
    rain_rate_mm_h: float = 0.0,
) -> LinkBudget:
    """Assemble and self-check a link budget."""
    if transmit_power_w <= 0.0:
        raise ValueError("transmit power must be positive")

    transmit_power_dbw = 10.0 * math.log10(transmit_power_w)
    spreading = free_space_loss_db(path_length_km, frequency_hz)
    rain = rain_attenuation_db(frequency_hz, rain_rate_mm_h, path_length_km)

    received = (
        transmit_power_dbw
        + transmit_gain_dbi
        + receive_gain_dbi
        - spreading
        - absorption_loss_db
        - ground_reflection_loss_db
        - rain            # subtracted, not merely displayed
    )

    budget = LinkBudget(
        frequency_hz=frequency_hz,
        transmit_power_dbw=transmit_power_dbw,
        transmit_gain_dbi=transmit_gain_dbi,
        receive_gain_dbi=receive_gain_dbi,
        spreading_loss_db=spreading,
        absorption_loss_db=absorption_loss_db,
        ground_reflection_loss_db=ground_reflection_loss_db,
        rain_attenuation_db=rain,
        received_power_dbw=received,
        noise=noise,
        required_snr_db=required_snr_db,
        path_length_km=path_length_km,
        hops=hops,
    )
    budget.verify()
    return budget
