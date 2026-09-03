"""Orchestration: scenario in, propagation prediction out.

This is where the components meet, and therefore where the mistakes that
matter live.  Three rules hold throughout:

1. **A receiver is reached only if a ray lands on it.**  Reach is decided by
   solving for the launch angle whose hop terminates at the receiver's
   distance and then confirming the traced range.  No hop is ever computed
   at one range and rescaled to another; a ray still at altitude over the
   receiver produces no mode, which is what a skip zone is.
2. **Every evaluation receives the whole scenario.**  MUF, LOF, the band
   scan and the single-frequency report all call one function that takes a
   :class:`Scenario`, so none of them can run against a station whose power
   or bandwidth was never supplied.
3. **A loss is charged exactly once.**  The launch-angle optimiser ranks
   candidates by the same total the link budget charges, so the angle it
   picks is the angle the budget was built from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .absorption import AbsorptionResult, absorption_db
from .antenna import GroundType
from .constants import EARTH_RADIUS_KM
from .geodesy import intermediate_point
from .ionosphere import EquivalentColumn, build_equivalent_column
from .link import LinkBudget, build_link_budget
from .magnetic import field_ray_angle_rad, geomagnetic_latitude_deg, magnetic_field
from .noise import noise_budget
from .raytrace import (
    RayMedium,
    RayPath,
    scan_ranges,
    skip_distance_km,
    solve_launch_angles,
    trace_ray,
)
from .refractive import Mode
from .scenario import Scenario
from .solar import PathIllumination, illuminate_path
from .surface import ground_reflection_loss_db, path_surface_profile

__all__ = ["PropagationMode", "FrequencyReport", "PropagationEngine",
           "Prediction", "STANDARD_BANDS_MHZ"]

#: Launch elevations considered.  Below 1 degree the spherical-Earth model
#: and the neglected terrain both stop being defensible.
MIN_ELEVATION_DEG = 1.0
MAX_ELEVATION_DEG = 60.0
MAX_HOPS = 5


@dataclass(frozen=True)
class PropagationMode:
    """One viable way the signal gets from A to B."""

    hops: int
    launch_elevation_deg: float
    mode: Mode
    path: RayPath
    absorption: AbsorptionResult
    budget: LinkBudget

    @property
    def total_path_km(self) -> float:
        return self.path.geometric_path_km * self.hops

    @property
    def group_delay_ms(self) -> float:
        return self.path.group_delay_ms * self.hops

    @property
    def margin_db(self) -> float:
        return self.budget.margin_db

    def summary(self) -> dict:
        return {
            "hops": self.hops,
            "launch_elevation_deg": self.launch_elevation_deg,
            "magnetoionic_mode": self.mode.value,
            "apex_height_km": self.path.apex_height_km,
            "virtual_height_km": self.path.virtual_height_km,
            "hop_range_km": self.path.ground_range_km,
            "total_path_km": self.total_path_km,
            "group_delay_ms": self.group_delay_ms,
            "geometric_delay_ms": self.path.geometric_delay_ms * self.hops,
            "absorption": self.absorption.summary(),
            "budget": self.budget.breakdown(),
        }


@dataclass(frozen=True)
class FrequencyReport:
    """What happens at one frequency."""

    frequency_hz: float
    modes: Sequence[PropagationMode]
    skip_distance_km: Optional[float]

    @property
    def frequency_mhz(self) -> float:
        return self.frequency_hz / 1e6

    @property
    def is_open(self) -> bool:
        return bool(self.modes)

    @property
    def best(self) -> Optional[PropagationMode]:
        """The mode with the most margin, or None if nothing gets through.

        Returning ``None`` rather than a fabricated SNR is deliberate: when
        no ray reaches the receiver there is no signal to quote a
        signal-to-noise ratio for.
        """
        return max(self.modes, key=lambda m: m.margin_db) if self.modes else None

    @property
    def snr_db(self) -> Optional[float]:
        best = self.best
        return best.budget.snr_db if best else None

    @property
    def margin_db(self) -> Optional[float]:
        best = self.best
        return best.margin_db if best else None

    def summary(self) -> dict:
        best = self.best
        return {
            "frequency_mhz": self.frequency_mhz,
            "open": self.is_open,
            "skip_distance_km": self.skip_distance_km,
            "snr_db": self.snr_db,
            "margin_db": self.margin_db,
            "mode_count": len(self.modes),
            "best_mode": best.summary() if best else None,
        }


class PropagationEngine:
    """Evaluates a scenario across frequency.

    The expensive, frequency-independent work -- the ionospheric column, the
    illumination, the surface classification -- is done once in the
    constructor and reused for every frequency.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        tx = scenario.transmitter.location
        rx = scenario.receiver.location

        self.illumination: PathIllumination = illuminate_path(
            tx, rx, scenario.when, scenario.path_samples
        )
        self.column: EquivalentColumn = build_equivalent_column(
            tx,
            rx,
            scenario.when,
            scenario.space_weather,
            self.illumination.seasonal_phase,
            scenario.path_samples,
        )
        self.surface = path_surface_profile(tx, rx)
        self.midpoint = intermediate_point(tx, rx, 0.5)
        self.geomagnetic_latitude_deg = geomagnetic_latitude_deg(self.midpoint)

        # Field and field/ray angle at a representative reflection height.
        self.magnetic_field = magnetic_field(self.midpoint, 250.0)
        self._distance_km = scenario.distance_km

    # -- helpers ---------------------------------------------------------
    def _theta_rad(self, elevation_deg: float) -> float:
        """Angle between the field and the ray, from the real field vector."""
        return field_ray_angle_rad(
            self.magnetic_field, self.scenario.bearing_deg, elevation_deg
        )

    def _medium(self, frequency_hz: float, mode: Mode, elevation_deg: float) -> RayMedium:
        return RayMedium(
            profile=self.column.mean_profile,
            frequency_hz=frequency_hz,
            mode=mode,
            magnetic_field=self.magnetic_field,
            theta_rad=self._theta_rad(elevation_deg),
        )

    def _ground_loss_db(self, elevation_deg: float, frequency_hz: float, hops: int) -> float:
        """Loss at the ``hops - 1`` intermediate ground reflections."""
        if hops <= 1:
            return 0.0
        ground = self.surface.dominant
        horizontal = self.scenario.transmitter.antenna.antenna_type.value.startswith(
            ("horizontal", "inverted")
        )
        per_bounce = ground_reflection_loss_db(
            elevation_deg,
            frequency_hz,
            ground,
            horizontal,
            self.scenario.weather.effective_moisture_factor,
            self.scenario.weather.sea_state,
        )
        return per_bounce * (hops - 1)

    def _build_mode(
        self,
        frequency_hz: float,
        mode: Mode,
        hops: int,
        elevation_deg: float,
        path: RayPath,
    ) -> PropagationMode:
        """Assemble the full budget for one candidate ray."""
        scenario = self.scenario
        theta = self._theta_rad(elevation_deg)

        absorption = absorption_db(
            path, self.column, frequency_hz, mode, self.magnetic_field, theta
        )
        # Absorption is per hop; the ray repeats.
        total_absorption_db = absorption.total_db * hops

        tx_gain = scenario.transmitter.antenna.gain_dbi(
            elevation_deg,
            frequency_hz,
            scenario.bearing_deg,
            scenario.weather.effective_moisture_factor,
        )
        rx_gain = scenario.receiver.antenna.gain_dbi(
            elevation_deg,
            frequency_hz,
            scenario.reverse_bearing_deg,
            scenario.weather.effective_moisture_factor,
        )

        noise = noise_budget(
            frequency_hz=frequency_hz,
            bandwidth_hz=scenario.receiver.bandwidth_hz,
            environment=scenario.receiver.noise_environment,
            sunlit_fraction=self.illumination.sunlit_fraction,
            geomagnetic_latitude_deg=self.geomagnetic_latitude_deg,
            kp=scenario.space_weather.kp,
            rain_rate_mm_h=scenario.weather.rain_rate_mm_h,
            receiver_noise_figure_db=scenario.receiver.receiver_noise_figure_db,
        )

        budget = build_link_budget(
            frequency_hz=frequency_hz,
            transmit_power_w=scenario.transmitter.transmit_power_w,
            transmit_gain_dbi=tx_gain,
            receive_gain_dbi=rx_gain,
            path_length_km=path.geometric_path_km * hops,
            absorption_loss_db=total_absorption_db,
            ground_reflection_loss_db=self._ground_loss_db(
                elevation_deg, frequency_hz, hops
            ),
            noise=noise,
            required_snr_db=scenario.receiver.required_snr_db,
            hops=hops,
            rain_rate_mm_h=scenario.weather.rain_rate_mm_h,
        )
        return PropagationMode(hops, elevation_deg, mode, path, absorption, budget)

    # -- main entry points ------------------------------------------------
    def evaluate(
        self, frequency_hz: float, modes: Sequence[Mode] = (Mode.ORDINARY,)
    ) -> FrequencyReport:
        """Every way the signal reaches the receiver at this frequency."""
        if frequency_hz <= 0.0:
            raise ValueError("frequency must be positive")

        found: List[PropagationMode] = []
        best_skip: Optional[float] = None
        start_height_km = self.scenario.transmitter.antenna.height_km

        scan_points = int((MAX_ELEVATION_DEG - MIN_ELEVATION_DEG) * 2) + 1
        for mode in modes:
            # The medium depends weakly on elevation through the field/ray
            # angle; probe at a representative angle, then rebuild at the
            # solved angle before charging anything to the budget.
            probe = self._medium(frequency_hz, mode, 15.0)
            # The range-against-elevation curve does not depend on the hop
            # count, so it is computed once and reused for all of them.
            scan = scan_ranges(
                probe, start_height_km, MIN_ELEVATION_DEG, MAX_ELEVATION_DEG, scan_points
            )
            candidate = skip_distance_km(
                probe, start_height_km, MIN_ELEVATION_DEG, MAX_ELEVATION_DEG, scan=scan
            )
            if candidate is not None:
                best_skip = candidate if best_skip is None else min(best_skip, candidate)

            for hops in range(1, MAX_HOPS + 1):
                target = self._distance_km / hops
                if target > math.pi * EARTH_RADIUS_KM:
                    continue
                angles = solve_launch_angles(
                    probe,
                    target,
                    start_height_km,
                    MIN_ELEVATION_DEG,
                    MAX_ELEVATION_DEG,
                    scan_points=scan_points,
                    scan=scan,
                )

                for elevation in angles:
                    medium = self._medium(frequency_hz, mode, elevation)
                    path = trace_ray(medium, elevation, start_height_km)
                    if path.escaped:
                        continue
                    # Confirm against the re-traced ray: the angle was solved
                    # on the probe medium, so the landing point is re-checked
                    # on the medium the budget is actually built from.
                    if abs(path.ground_range_km - target) > max(5.0, 0.01 * target):
                        continue
                    found.append(
                        self._build_mode(frequency_hz, mode, hops, elevation, path)
                    )

        return FrequencyReport(frequency_hz, tuple(found), best_skip)

    def _frequency_grid(self, low_hz: float, high_hz: float, step_hz: float) -> List[float]:
        count = max(2, int(round((high_hz - low_hz) / step_hz)) + 1)
        return [low_hz + i * step_hz for i in range(count)]

    def maximum_usable_frequency_hz(
        self,
        low_hz: float = 2e6,
        high_hz: float = 50e6,
        step_hz: float = 2.5e5,
        tolerance_hz: float = 2.5e4,
    ) -> Optional[float]:
        """Highest frequency the ionosphere still returns to the receiver.

        A purely geometric quantity: it asks whether a ray lands on the
        receiver, not whether what arrives can be heard.  That is the LOF's
        job, and keeping the two separate is the point.

        Found by scanning and then refining, **not** by bisecting from the
        band edges.  Openness is not monotonic in frequency: a path can be
        closed at one frequency because the receiver sits in the one-hop
        skip zone and open again higher up on a two-hop mode.  A bare
        bisection on a non-monotonic predicate lands wherever the first
        probe happens to fall.
        """
        grid = self._frequency_grid(low_hz, high_hz, step_hz)
        open_flags = [self.evaluate(f).is_open for f in grid]
        if not any(open_flags):
            return None

        highest_open = max(i for i, flag in enumerate(open_flags) if flag)
        if highest_open == len(grid) - 1:
            return grid[-1]

        low, high = grid[highest_open], grid[highest_open + 1]
        while high - low > tolerance_hz:
            mid = 0.5 * (low + high)
            if self.evaluate(mid).is_open:
                low = mid
            else:
                high = mid
        return low

    def lowest_usable_frequency_hz(
        self,
        low_hz: float = 1.5e6,
        high_hz: float = 30e6,
        step_hz: float = 2.5e5,
        tolerance_hz: float = 2.5e4,
    ) -> Optional[float]:
        """Lowest frequency arriving with enough SNR to be usable.

        Set by absorption and noise rather than by geometry, and evaluated
        through the same :meth:`evaluate` as everything else -- so it sees
        the scenario's real transmit power, antenna gains, bandwidth, noise
        figure and required SNR.  There is no separate code path here that
        could be handed an incomplete station and fall back on defensive
        defaults, which is how a LOF ends up computed from a -100 dBW
        transmitter into a 10 Hz bandwidth.
        """
        grid = self._frequency_grid(low_hz, high_hz, step_hz)
        margins = [self.evaluate(f).margin_db for f in grid]
        usable = [m is not None and m >= 0.0 for m in margins]
        if not any(usable):
            return None

        lowest_usable = min(i for i, flag in enumerate(usable) if flag)
        if lowest_usable == 0:
            return grid[0]

        low, high = grid[lowest_usable - 1], grid[lowest_usable]
        while high - low > tolerance_hz:
            mid = 0.5 * (low + high)
            margin = self.evaluate(mid).margin_db
            if margin is not None and margin >= 0.0:
                high = mid
            else:
                low = mid
        return high

    def predict(
        self,
        low_hz: float = 1.5e6,
        high_hz: float = 30e6,
        step_hz: float = 2.5e5,
    ) -> "Prediction":
        """One pass over the band, reusing every evaluation.

        MUF and LOF are extracted from the same scan that produces the band
        report, so the three can never disagree with each other.
        """
        grid = self._frequency_grid(low_hz, high_hz, step_hz)
        reports = [self.evaluate(f) for f in grid]

        open_indices = [i for i, r in enumerate(reports) if r.is_open]
        usable_indices = [
            i for i, r in enumerate(reports)
            if r.margin_db is not None and r.margin_db >= 0.0
        ]

        muf = reports[max(open_indices)].frequency_hz if open_indices else None
        lof = reports[min(usable_indices)].frequency_hz if usable_indices else None

        return Prediction(
            scenario=self.scenario,
            conditions=self.conditions(),
            reports=tuple(reports),
            muf_hz=muf,
            lof_hz=lof,
        )

    def conditions(self) -> dict:
        """Frequency-independent context, for reporting."""
        return {
            "distance_km": self._distance_km,
            "bearing_deg": self.scenario.bearing_deg,
            "sunlit_fraction": self.illumination.sunlit_fraction,
            "crosses_terminator": self.illumination.crosses_terminator,
            "solar_declination_deg": self.illumination.declination_deg,
            "seasonal_phase": self.illumination.seasonal_phase,
            "fof2_mhz": self.column.fof2_mhz,
            "foe_mhz": self.column.mean_profile.layers.e.critical_frequency_mhz,
            "hmf2_km": self.column.mean_profile.layers.f2.peak_height_km,
            "geomagnetic_latitude_deg": self.geomagnetic_latitude_deg,
            "gyrofrequency_mhz": self.magnetic_field.gyrofrequency_hz / 1e6,
            "magnetic_dip_deg": self.magnetic_field.inclination_deg,
            "sea_fraction": self.surface.sea_fraction,
            "dominant_surface": self.surface.dominant.value,
        }


#: Amateur and broadcast band centres used for the practical band report.
STANDARD_BANDS_MHZ = (
    ("160 m", 1.9), ("80 m", 3.65), ("60 m", 5.35), ("40 m", 7.1),
    ("30 m", 10.125), ("20 m", 14.2), ("17 m", 18.1), ("15 m", 21.2),
    ("12 m", 24.94), ("10 m", 28.5),
)


@dataclass(frozen=True)
class Prediction:
    """The complete answer for one scenario."""

    scenario: Scenario
    conditions: dict
    reports: Sequence[FrequencyReport]
    muf_hz: Optional[float]
    lof_hz: Optional[float]

    @property
    def muf_mhz(self) -> Optional[float]:
        return self.muf_hz / 1e6 if self.muf_hz else None

    @property
    def lof_mhz(self) -> Optional[float]:
        return self.lof_hz / 1e6 if self.lof_hz else None

    @property
    def optimum_working_frequency_mhz(self) -> Optional[float]:
        """The traditional FOT: 85% of the MUF.

        A rule of thumb about day-to-day variability of the real ionosphere,
        not a prediction this model derives. Labelled as such.
        """
        return self.muf_mhz * 0.85 if self.muf_mhz else None

    def best_report(self) -> Optional[FrequencyReport]:
        open_reports = [r for r in self.reports if r.margin_db is not None]
        return max(open_reports, key=lambda r: r.margin_db) if open_reports else None

    def band_report(self) -> List[dict]:
        """Score each standard band.

        The score is an **operational heuristic** -- it weighs link margin,
        how far the band sits below the MUF, and how well the antenna works
        there.  It is not a probability of contact, and nothing in this
        package treats it as one.
        """
        rows = []
        for name, centre_mhz in STANDARD_BANDS_MHZ:
            nearest = min(
                self.reports, key=lambda r: abs(r.frequency_mhz - centre_mhz)
            )
            if abs(nearest.frequency_mhz - centre_mhz) > 0.5:
                continue
            margin = nearest.margin_db
            if margin is None:
                rows.append({
                    "band": name, "frequency_mhz": centre_mhz, "open": False,
                    "margin_db": None, "score": 0.0, "note": "no ray reaches the receiver",
                })
                continue

            margin_score = max(0.0, min(1.0, (margin + 5.0) / 35.0))
            if self.muf_mhz:
                ratio = centre_mhz / self.muf_mhz
                # Best just below the MUF; penalised for being too close to it.
                headroom_score = math.exp(-((ratio - 0.82) / 0.30) ** 2)
            else:
                headroom_score = 0.0
            antenna = self.scenario.transmitter.antenna
            best = nearest.best
            practicality = math.exp(
                -((antenna.gain_dbi(best.launch_elevation_deg, nearest.frequency_hz)
                   - antenna.gain_dbi(antenna.best_elevation_deg(nearest.frequency_hz),
                                     nearest.frequency_hz)) / 12.0) ** 2
            )
            score = 0.55 * margin_score + 0.30 * headroom_score + 0.15 * practicality
            rows.append({
                "band": name,
                "frequency_mhz": centre_mhz,
                "open": True,
                "margin_db": margin,
                "snr_db": nearest.snr_db,
                "hops": best.hops,
                "elevation_deg": best.launch_elevation_deg,
                "score": score,
                "note": "heuristic score, not a contact probability",
            })
        return sorted(rows, key=lambda r: r["score"], reverse=True)

    def summary(self) -> dict:
        best = self.best_report()
        return {
            "scenario": self.scenario.summary(),
            "conditions": self.conditions,
            "muf_mhz": self.muf_mhz,
            "lof_mhz": self.lof_mhz,
            "fot_mhz": self.optimum_working_frequency_mhz,
            "best_frequency_mhz": best.frequency_mhz if best else None,
            "bands": self.band_report(),
            "spectrum": [r.summary() for r in self.reports],
        }
