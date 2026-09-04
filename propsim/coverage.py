"""Coverage against distance: what a transmitter reaches, not just its receiver.

Two sweeps, both built on one observation: for a given frequency and mode the
range-against-elevation curve does not depend on where the receiver is.  So a
sweep over distance reuses a single elevation scan per frequency instead of
recomputing one per point, which is what makes these affordable at interactive
speed.

The ionosphere stays the equivalent column built for the scenario's own path.
These curves therefore answer "how far does this signal reach along this
bearing", not "what would happen on a completely different circuit" -- a
distinction worth keeping, because the column is an average over the modelled
path and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .constants import EARTH_RADIUS_KM
from .engine import (
    MAX_ELEVATION_DEG,
    MAX_HOPS,
    MIN_ELEVATION_DEG,
    PropagationEngine,
)
from .link import field_strength_dbuv_per_m
from .refractive import Mode

__all__ = ["CoverageSample", "UsableBandSample", "coverage_vs_distance",
           "usable_band_vs_distance", "distance_grid", "frequency_grid"]


#: Where the near field stops being the interesting part of a coverage
#: curve.  Below this the ground wave carries the link and the skywave has
#: not started; above it the skip zone and the hop structure decide
#: everything, and they are features of thousands of kilometres, not tens.
NEAR_FIELD_KM = 500.0


def distance_grid(maximum_km: float = 12000.0, points: int = 40) -> List[float]:
    """Distances to sample, starting clear of the transmitter itself.

    Geometric below :data:`NEAR_FIELD_KM` and uniform above it.  A purely
    uniform grid out to 12 000 km puts its first sample at 300 km, which is
    past the whole of the ground wave on the higher bands and past the
    near edge of the skip zone on the lower ones -- so the curve begins
    after the two most interesting things on it have already happened.
    Geometric spacing near in costs nothing: the same few samples resolve
    30 km and 300 km, where a uniform grid resolves neither.
    """
    if points < 2:
        return [min(NEAR_FIELD_KM, maximum_km)]
    if maximum_km <= NEAR_FIELD_KM:
        step = maximum_km / points
        return [step * (i + 1) for i in range(points)]

    near_points = max(2, points // 4)
    far_points = points - near_points
    start = min(25.0, maximum_km / points)
    ratio = (NEAR_FIELD_KM / start) ** (1.0 / near_points)
    near = [start * ratio**i for i in range(near_points)]
    step = (maximum_km - NEAR_FIELD_KM) / max(far_points - 1, 1)
    far = [NEAR_FIELD_KM + i * step for i in range(far_points)]
    return near + far


def frequency_grid(low_hz: float = 2e6, high_hz: float = 32e6, points: int = 25) -> List[float]:
    step = (high_hz - low_hz) / max(points - 1, 1)
    return [low_hz + i * step for i in range(points)]


@dataclass(frozen=True)
class CoverageSample:
    """What arrives at one distance, at a fixed frequency."""

    distance_km: float
    reached: bool
    hops: Optional[int] = None
    launch_elevation_deg: Optional[float] = None
    apex_height_km: Optional[float] = None
    field_strength_dbuv_m: Optional[float] = None
    received_power_dbm: Optional[float] = None
    noise_floor_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    margin_db: Optional[float] = None
    #: The surface-wave route at this distance, which exists whether or not
    #: a ray lands here.  Kept as its own column rather than folded into
    #: the skywave one: a coverage chart that merged them would fill the
    #: skip zone in and hide the very thing it is drawn to show.
    ground_wave_margin_db: Optional[float] = None
    ground_wave_field_dbuv_m: Optional[float] = None
    ground_wave_loss_db: Optional[float] = None

    @property
    def best_margin_db(self) -> Optional[float]:
        """The better of the two routes, whichever it is."""
        candidates = [
            value for value in (self.margin_db, self.ground_wave_margin_db)
            if value is not None
        ]
        return max(candidates) if candidates else None

    def summary(self) -> dict:
        return {
            "distance_km": self.distance_km,
            "reached": self.reached,
            "hops": self.hops,
            "launch_elevation_deg": self.launch_elevation_deg,
            "apex_height_km": self.apex_height_km,
            "field_strength_dbuv_m": self.field_strength_dbuv_m,
            "received_power_dbm": self.received_power_dbm,
            "noise_floor_dbm": self.noise_floor_dbm,
            "snr_db": self.snr_db,
            "margin_db": self.margin_db,
            "ground_wave_margin_db": self.ground_wave_margin_db,
            "ground_wave_field_dbuv_m": self.ground_wave_field_dbuv_m,
            "ground_wave_loss_db": self.ground_wave_loss_db,
            "best_margin_db": self.best_margin_db,
        }


def coverage_vs_distance(
    engine: PropagationEngine,
    frequency_hz: float,
    distances_km: Optional[Sequence[float]] = None,
) -> List[CoverageSample]:
    """Field strength and SNR against distance at one frequency.

    ``reached`` is a statement about the **skywave**: a distance where it is
    false is one no ray lands on, so skip zones appear as genuine gaps
    rather than as a dip, which is the whole point of solving for the
    launch angle instead of scaling a hop.  The ground wave is reported
    alongside it in its own columns, because inside the skip zone it is
    frequently the only thing there is -- and because merging the two
    curves would paper over exactly the gap this function exists to show.
    """
    distances = list(distances_km) if distances_km is not None else distance_grid()
    samples: List[CoverageSample] = []

    for distance in distances:
        report = engine.evaluate(frequency_hz, distance_km=distance)
        ground_wave = report.ground_wave
        surface = {}
        if ground_wave is not None:
            surface = {
                "ground_wave_margin_db": ground_wave.margin_db,
                "ground_wave_loss_db": ground_wave.loss.total_db,
                "ground_wave_field_dbuv_m": field_strength_dbuv_per_m(
                    ground_wave.budget.received_power_dbw,
                    ground_wave.budget.receive_gain_dbi,
                    frequency_hz,
                ),
            }

        best = report.best
        if best is None:
            samples.append(
                CoverageSample(distance_km=distance, reached=False, **surface)
            )
            continue

        budget = best.budget
        receive_gain = budget.receive_gain_dbi
        field = field_strength_dbuv_per_m(
            budget.received_power_dbw, receive_gain, frequency_hz
        )
        samples.append(CoverageSample(
            distance_km=distance,
            reached=True,
            hops=best.hops,
            launch_elevation_deg=best.launch_elevation_deg,
            apex_height_km=best.apex_height_km,
            field_strength_dbuv_m=field,
            received_power_dbm=budget.received_power_dbw + 30.0,
            noise_floor_dbm=budget.noise.noise_power_dbw + 30.0,
            snr_db=budget.snr_db,
            margin_db=best.margin_db,
            **surface,
        ))
    return samples


@dataclass(frozen=True)
class UsableBandSample:
    """The frequency window that works at one distance."""

    distance_km: float
    muf_mhz: Optional[float]
    lof_mhz: Optional[float]

    @property
    def has_window(self) -> bool:
        return (
            self.muf_mhz is not None
            and self.lof_mhz is not None
            and self.muf_mhz > self.lof_mhz
        )

    def summary(self) -> dict:
        return {
            "distance_km": self.distance_km,
            "muf_mhz": self.muf_mhz,
            "lof_mhz": self.lof_mhz,
            "usable": self.has_window,
        }


def _reaches(
    engine: PropagationEngine, frequency_hz: float, distance_km: float, mode: Mode
) -> bool:
    """Does a ray land at this distance, in any hop count?

    Geometry only, and not even the full geometry.  Two economies, in order
    of how much they save.

    Building a link budget to answer this -- antenna gains, absorption,
    noise, the lot -- computes a number that is then thrown away, since the
    MUF asks only whether the ionosphere returns the signal.

    More than that: *solving* for the launch angle is also wasted here.
    ``solve_launch_angles`` finds the bracket in the range-against-elevation
    curve and then bisects inside it to locate the angle precisely.  This
    question needs the bracket alone -- whether the curve crosses the target
    range between two neighbouring elevations -- which is an array test on a
    curve already computed and cached.  Skipping the bisection is what takes
    the sweep from half a minute to something interactive, and it costs
    nothing but the elevation resolution of the scan, which is finer than
    any coverage curve can show.
    """
    _, scan, skip = engine.scan_for(frequency_hz, mode)
    ranges = scan[1]

    for hops in range(1, engine.scenario.max_hops + 1):
        target = distance_km / hops
        if target > math.pi * EARTH_RADIUS_KM:
            continue
        if skip is not None and target < skip - 1e-9:
            continue
        previous: Optional[float] = None
        for value in ranges:
            if value is None:
                previous = None
                continue
            if previous is not None and (previous - target) * (value - target) <= 0.0:
                return True
            previous = value
    return False


def usable_band_vs_distance(
    engine: PropagationEngine,
    distances_km: Optional[Sequence[float]] = None,
    frequencies_hz: Optional[Sequence[float]] = None,
) -> List[UsableBandSample]:
    """MUF and LOF against distance.

    Two passes, because the two quantities cost very different amounts.

    **MUF is geometric**: the highest frequency whose ray lands at all.  It is
    found from the cached elevation scans without ever building a budget.

    **LOF reads the budget**, so it sees the real transmit power, antenna
    gains, bandwidth, noise figure and required SNR.  It is bracketed on a
    coarse frequency scan and then bisected, rather than stepped from the
    bottom of the band at every distance.

    The loop is ordered frequency-outermost in the first pass on purpose:
    each frequency builds its elevation scan once and then answers every
    distance from it.
    """
    distances = list(distances_km) if distances_km is not None else distance_grid(points=16)
    frequencies = list(frequencies_hz) if frequencies_hz is not None else frequency_grid()

    # -- pass one: MUF, from geometry alone ------------------------------
    highest: dict[float, Optional[float]] = {d: None for d in distances}
    for frequency in frequencies:
        for distance in distances:
            if any(
                _reaches(engine, frequency, distance, mode)
                for mode in (Mode.ORDINARY, Mode.EXTRAORDINARY)
            ):
                highest[distance] = frequency

    # -- pass two: LOF, from the link budget ------------------------------
    lowest: dict[float, Optional[float]] = {d: None for d in distances}
    for distance in distances:
        ceiling = highest[distance]
        if ceiling is None:
            continue

        def usable(frequency_hz: float) -> bool:
            margin = engine.evaluate(
                frequency_hz, distance_km=distance
            ).effective_margin_db()
            return margin is not None and margin >= 0.0

        candidates = [f for f in frequencies if f <= ceiling + 1e-6]
        if not candidates:
            continue

        # Coarse bracket: the first sampled frequency that closes the link.
        # Both magnetoionic modes stay in the evaluation. Dropping the
        # extraordinary one looks like a free halving and is not: a second
        # arriving mode lowers the Rician K factor and so deepens the fade,
        # which moves the effective margin by several decibels. More modes
        # is not simply more signal.
        stride = max(1, len(candidates) // 5)
        bracket_low: Optional[float] = None
        bracket_high: Optional[float] = None
        for index in range(0, len(candidates), stride):
            if usable(candidates[index]):
                bracket_high = candidates[index]
                bracket_low = candidates[index - stride] if index >= stride else None
                break
        if bracket_high is None and usable(candidates[-1]):
            bracket_high = candidates[-1]
            bracket_low = candidates[max(0, len(candidates) - 1 - stride)]
        if bracket_high is None:
            continue                       # nothing in band closes the link
        if bracket_low is None:
            lowest[distance] = bracket_high
            continue

        low, high = bracket_low, bracket_high
        # Four halvings of a ~6 MHz bracket land inside 0.4 MHz, finer than
        # the curve is drawn.
        for _ in range(4):
            middle = 0.5 * (low + high)
            if usable(middle):
                high = middle
            else:
                low = middle
        lowest[distance] = high

    return [
        UsableBandSample(
            distance_km=distance,
            muf_mhz=None if highest[distance] is None else highest[distance] / 1e6,
            lof_mhz=None if lowest[distance] is None else lowest[distance] / 1e6,
        )
        for distance in distances
    ]
