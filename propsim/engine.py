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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .absorption import AbsorptionResult, absorption_db
from .absorption import combine as combine_absorption
from .fading import MultipathProfile, multipath_profile
from .antenna import GroundType
from .constants import EARTH_RADIUS_KM, SPEED_OF_LIGHT
from .geodesy import destination_point, intermediate_point
from .groundwave import GroundWaveLoss, ground_wave_loss_db
from .ionosphere import (
    MAX_HEIGHT_KM as IONOSPHERE_TOP_KM,
    EquivalentColumn,
    build_equivalent_column,
)
from .link import LinkBudget, build_link_budget
from .magnetic import field_ray_angle_rad, geomagnetic_latitude_deg, magnetic_field
from .noise import noise_budget
from .raytrace import (
    RayMedium,
    RayPath,
    _batch_hop_geometry,
    scan_ranges,
    skip_distance_km,
    solve_launch_angles,
    trace_ray,
)
from .refractive import Mode
from .scenario import Scenario
from .solar import PathIllumination, illuminate_path
from .surface import (
    classify_surface,
    ground_reflection_loss_db,
    path_sections,
    path_surface_profile,
)

__all__ = ["PropagationMode", "GroundWave", "FrequencyReport",
           "PropagationEngine", "Prediction", "STANDARD_BANDS_MHZ"]

#: Launch elevations considered.  Below 1 degree the spherical-Earth model
#: and the neglected terrain both stop being defensible.
MIN_ELEVATION_DEG = 1.0
MAX_ELEVATION_DEG = 60.0
MAX_HOPS = 5

#: Longest ground range one hop can possibly have, from geometry alone.
#: A ray that turns at the very top of the modelled profile and comes back
#: covers ``2 R arccos(R / (R + h))``; no ionosphere can beat it, because
#: there is no ionosphere above that height in this model to turn in.
#: Used only to skip hop counts that cannot exist -- never to accept one.
MAX_HOP_RANGE_KM = 2.0 * EARTH_RADIUS_KM * math.acos(
    EARTH_RADIUS_KM / (EARTH_RADIUS_KM + IONOSPHERE_TOP_KM)
)

#: Both magnetoionic modes are evaluated by default.  A magnetised plasma
#: splits the wave into an ordinary and an extraordinary component that
#: refract differently, turn at different heights and are absorbed
#: differently -- the extraordinary one resonates near the gyrofrequency and
#: fades first.  Evaluating only the O mode silently throws away the half of
#: the physics that decides which of the two actually arrives.
DEFAULT_MODES: Sequence[Mode] = (Mode.ORDINARY, Mode.EXTRAORDINARY)


@dataclass(frozen=True)
class PropagationMode:
    """One viable way the signal gets from A to B."""

    hops: int
    launch_elevation_deg: float
    mode: Mode
    #: The first hop, kept for display; the circuit is :attr:`paths`.
    path: RayPath
    absorption: AbsorptionResult
    budget: LinkBudget
    #: One traced ray per hop, each through the ionosphere above its own
    #: stretch of the path.  They differ: on a circuit crossing the
    #: terminator the first hop climbs into a daylit F2 and the last into a
    #: night-time one, and their ranges and apex heights differ with it.
    paths: Sequence[RayPath] = ()
    hop_ranges_km: Sequence[float] = ()

    @property
    def apex_height_km(self) -> float:
        """Highest point of the first hop, for display."""
        return self.path.apex_height_km

    @property
    def total_path_km(self) -> float:
        return sum(p.geometric_path_km for p in self.paths) or self.path.geometric_path_km

    @property
    def group_delay_ms(self) -> float:
        return sum(p.group_delay_ms for p in self.paths) or self.path.group_delay_ms

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
            "hop_ranges_km": list(self.hop_ranges_km),
            "apex_heights_km": [p.apex_height_km for p in self.paths],
            "total_path_km": self.total_path_km,
            "group_delay_ms": self.group_delay_ms,
            "geometric_delay_ms": self.path.geometric_delay_ms * self.hops,
            "absorption": self.absorption.summary(),
            "budget": self.budget.breakdown(),
            "via": "skywave",
        }


@dataclass(frozen=True)
class GroundWave:
    """The signal that reaches the receiver without leaving the ground.

    Deliberately not a :class:`PropagationMode`.  It has no launch angle to
    solve for, no hop count, no apex, no magnetoionic splitting and no
    ionospheric absorption, because it never enters the ionosphere -- and a
    class that carried all of those as zeros would invite exactly the
    confusion of counting it as a skywave hop.  What it shares with a
    skywave mode is the part that is genuinely common: a link budget, a
    delay and a margin, so anything comparing routes can compare them.
    """

    frequency_hz: float
    distance_km: float
    loss: GroundWaveLoss
    budget: LinkBudget

    #: Read as "no ionospheric hop", and true: the wave never goes up.
    hops: int = 0
    launch_elevation_deg: float = 0.0
    apex_height_km: float = 0.0
    #: A ground wave is a single vertically polarised wave, not one of the
    #: two magnetoionic components -- there is no magnetised plasma along
    #: its path to split it.
    mode: Optional[Mode] = None

    @property
    def total_path_km(self) -> float:
        return self.distance_km

    @property
    def group_delay_ms(self) -> float:
        """Surface distance at the speed of light.

        The ground wave is the *early* arrival on any path that also has a
        skywave: a 400 km circuit reaches the receiver in 1.3 ms along the
        ground and in 1.7 ms via the E region, and it is the beat between
        those two that makes the classic dusk fade on the lower bands.
        """
        return self.distance_km * 1e3 / SPEED_OF_LIGHT * 1e3

    @property
    def margin_db(self) -> float:
        return self.budget.margin_db

    def summary(self) -> dict:
        return {
            "hops": 0,
            "launch_elevation_deg": 0.0,
            "magnetoionic_mode": None,
            "apex_height_km": 0.0,
            "total_path_km": self.distance_km,
            "group_delay_ms": self.group_delay_ms,
            "loss": self.loss.summary(),
            "budget": self.budget.breakdown(),
            "via": "ground wave",
        }


@dataclass(frozen=True)
class FrequencyReport:
    """What happens at one frequency."""

    frequency_hz: float
    modes: Sequence[PropagationMode]
    skip_distance_km: Optional[float]
    #: The surface-wave route, always evaluated and usually negligible.
    #: It is kept out of :attr:`modes` on purpose: the MUF asks whether the
    #: *ionosphere* returns a ray, and a ground wave that exists at every
    #: frequency would answer "yes, 50 MHz" to that question forever.
    ground_wave: Optional["GroundWave"] = None
    #: Memo for :meth:`multipath`.  The Rician fade integral is the most
    #: expensive thing in a report and the answer cannot change: a report is
    #: a fixed set of modes.  Mutating the dict is allowed on a frozen
    #: dataclass because the field itself is never rebound.
    _multipath_cache: Dict[float, Optional[MultipathProfile]] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def frequency_mhz(self) -> float:
        return self.frequency_hz / 1e6

    @property
    def is_open(self) -> bool:
        """Whether the **ionosphere** returns a ray to the receiver.

        A skywave question, and the one the MUF is defined by.  For "can
        the receiver hear anything at all", which is what an operator
        actually wants, see :attr:`reachable`.
        """
        return bool(self.modes)

    @property
    def usable(self) -> bool:
        """Whether any route clears the operator's own required SNR.

        Not a threshold this package invented: ``required_snr_db`` is a
        field of the scenario, supplied by whoever is asking.  A ground
        wave 180 dB under the noise is a number the model can compute and
        is not a contact, and this is what says so.
        """
        margin = self.overall_margin_db
        return margin is not None and margin >= 0.0

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
    def routes(self) -> Sequence[object]:
        """Every route that reaches the receiver, skywave and ground wave.

        The ground wave goes last so that, on the many paths where it is
        200 dB down, ``max`` over margin still returns the skywave mode it
        would have returned before this existed.
        """
        routes = list(self.modes)
        if self.ground_wave is not None:
            routes.append(self.ground_wave)
        return routes

    #: An arrival this far under the strongest one cannot fade it: at 40 dB
    #: down the amplitude ratio is 1/100, so the resultant swings by at
    #: most 20 log10(1.01) = 0.086 dB between full addition and full
    #: cancellation.  Counting such an arrival as multipath would inflate
    #: the reported mode count without moving any number that depends on it.
    NEGLIGIBLE_ARRIVAL_DB = 40.0

    @property
    def interfering_routes(self) -> Sequence[object]:
        """The routes strong enough to fade against each other."""
        routes = self.routes
        if not routes:
            return ()
        strongest = max(r.budget.received_power_dbw for r in routes)
        return [
            r for r in routes
            if r.budget.received_power_dbw
            >= strongest - self.NEGLIGIBLE_ARRIVAL_DB
        ]

    @property
    def best_overall(self):
        """The strongest route by whatever mechanism, or None.

        On a 56 km path on 80 metres this is the ground wave, because there
        is no skywave at all; on a 3000 km path it is a skywave mode by a
        margin of a hundred decibels.  Nothing chooses between the two by
        distance -- they are both computed and the louder one wins.
        """
        routes = self.routes
        return max(routes, key=lambda r: r.margin_db) if routes else None

    @property
    def overall_margin_db(self) -> Optional[float]:
        best = self.best_overall
        return best.margin_db if best else None

    @property
    def overall_snr_db(self) -> Optional[float]:
        best = self.best_overall
        return best.budget.snr_db if best else None

    def multipath(self, availability: float = 0.9) -> Optional[MultipathProfile]:
        """How the arriving modes interfere, and what that costs in margin.

        Every mode in this report reaches the receiver, so they all add
        vectorially and their relative phases drift.  The link budget
        reports the *mean* power; this says how far below that mean the
        signal sits for the given fraction of the time.
        """
        if availability in self._multipath_cache:
            return self._multipath_cache[availability]
        profile = None
        routes = self.interfering_routes
        if routes:
            # The ground wave is one of the interfering arrivals, not a
            # separate world.  Where it is comparable with a skywave mode
            # the two beat against each other -- that is the dusk fade on
            # 160 and 80 metres -- and where it is far below one it has
            # already been dropped by :attr:`interfering_routes`.
            profile = multipath_profile(
                [r.budget.received_power_dbw for r in routes],
                [r.group_delay_ms for r in routes],
                routes[0].budget.noise.bandwidth_hz,
                availability,
            )
        self._multipath_cache[availability] = profile
        return profile

    def fade_margin_db(self, availability: float = 0.9) -> float:
        profile = self.multipath(availability)
        return profile.fade_margin_db if profile else 0.0

    def effective_margin_db(self, availability: float = 0.9) -> Optional[float]:
        """Margin after paying for multipath fading.

        This, not :attr:`margin_db`, is what decides whether a circuit is
        usable: a link whose *mean* signal clears the noise by 5 dB is below
        it much of the time if several modes are fading against each other.
        """
        if self.margin_db is None:
            return None
        return self.margin_db - self.fade_margin_db(availability)

    def best_of_mode(self, mode: Mode) -> Optional["PropagationMode"]:
        candidates = [m for m in self.modes if m.mode is mode]
        return max(candidates, key=lambda m: m.margin_db) if candidates else None

    @property
    def mode_splitting_db(self) -> Optional[float]:
        """Margin advantage of the ordinary mode over the extraordinary one.

        Positive is the usual case: the X mode sits closer to the
        gyrofrequency resonance and absorbs more.  ``None`` means only one of
        the two reaches the receiver at all, which happens near the MUF
        where their turning heights differ enough to matter.
        """
        ordinary = self.best_of_mode(Mode.ORDINARY)
        extraordinary = self.best_of_mode(Mode.EXTRAORDINARY)
        if ordinary is None or extraordinary is None:
            return None
        return ordinary.margin_db - extraordinary.margin_db

    @property
    def margin_db(self) -> Optional[float]:
        best = self.best
        return best.margin_db if best else None

    def summary(self) -> dict:
        best = self.best
        profile = self.multipath()
        return {
            "frequency_mhz": self.frequency_mhz,
            "open": self.is_open,
            "usable": self.usable,
            "skip_distance_km": self.skip_distance_km,
            "snr_db": self.snr_db,
            "margin_db": self.margin_db,
            "overall_snr_db": self.overall_snr_db,
            "overall_margin_db": self.overall_margin_db,
            "best_route": (self.best_overall.summary() if self.best_overall else None),
            "ground_wave": (
                self.ground_wave.summary() if self.ground_wave else None
            ),
            "mode_count": len(self.modes),
            "magnetoionic_mode": best.mode.value if best else None,
            "mode_splitting_db": self.mode_splitting_db,
            "effective_margin_db": self.effective_margin_db(),
            "multipath": profile.summary() if profile else None,
            "best_mode": best.summary() if best else None,
        }


class PropagationEngine:
    """Evaluates a scenario across frequency.

    The expensive, frequency-independent work -- the ionospheric column, the
    illumination, the surface classification -- is done once in the
    constructor and reused for every frequency.
    """

    def __init__(
        self,
        scenario: Scenario,
        fof2_multiplier: float = 1.0,
        sporadic_e=None,
    ) -> None:
        self.scenario = scenario
        self.fof2_multiplier = fof2_multiplier
        self.sporadic_e = sporadic_e
        # An engine is fixed to one scenario, one foF2 draw and one sporadic-E
        # state, so a frequency's report cannot change under it.  The band
        # scan, the MUF bisection and the LOF search all probe overlapping
        # frequencies, and the reliability sweep runs the same band centres
        # through every quantile's engine; memoising removes that repetition
        # rather than recomputing an identical answer.
        self._reports: Dict[
            Tuple[float, Tuple[Mode, ...], float], FrequencyReport
        ] = {}
        # The range-against-elevation curve depends on frequency and mode but
        # not on where the receiver is, so a distance sweep reuses one scan
        # for every target rather than recomputing it per distance.
        self._scans: Dict[Tuple[float, Mode], Tuple] = {}
        self._segment_scans: Dict[Tuple, Tuple] = {}
        # Land/sea decomposition of the path, per distance.  Frequency
        # independent, so a band scan classifies the coastlines once.
        self._sections: Dict[float, Sequence[Tuple[float, GroundType]]] = {}
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
            fof2_multiplier=fof2_multiplier * scenario.fof2_scale,
            sporadic_e=sporadic_e,
            hmf2_offset_km=scenario.hmf2_offset_km,
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

    def _medium(
        self,
        frequency_hz: float,
        mode: Mode,
        elevation_deg: float,
        segment: Optional[Tuple[float, float]] = None,
    ) -> RayMedium:
        """The refracting medium, optionally over one stretch of the path.

        ``segment`` is a ``(low, high)`` pair of path fractions selecting the
        ionosphere averaged over that stretch rather than over the whole
        path.  A circuit that crosses the terminator has a first hop
        in daylight and a last one in darkness -- on one such path foF2 runs
        from 8.3 to 6.0 MHz -- and tracing every hop through the 6.5 MHz
        average describes neither end of it.
        """
        profile = (
            self.column.mean_profile if segment is None
            else self.column.segment_profile(*segment)
        )
        return RayMedium(
            profile=profile,
            frequency_hz=frequency_hz,
            mode=mode,
            magnetic_field=self.magnetic_field,
            theta_rad=self._theta_rad(elevation_deg),
        )

    def segment_fractions(self, hops: int) -> List[Tuple[float, float]]:
        """The stretch of path each hop of an ``hops``-hop circuit covers.

        Equal shares, which the hops do not take exactly -- that is the
        whole point, since each refracts in a different ionosphere -- but
        they stay within a fifth of each other, far finer than the nine
        columns the path is sampled with.
        """
        return [(k / hops, (k + 1) / hops) for k in range(hops)]

    def scan_for_segment(
        self, frequency_hz: float, mode: Mode, segment: Tuple[float, float]
    ) -> Tuple:
        """``(medium, elevation scan, skip distance)`` for one path segment.

        Keyed on the segment, and segments repeat across hop counts -- the
        first half of the path is the first hop of a two-hop circuit and the
        first two of a four-hop one -- so the scans are shared rather than
        rebuilt.
        """
        key = (frequency_hz, mode, round(segment[0], 4), round(segment[1], 4))
        cached = self._segment_scans.get(key)
        if cached is not None:
            return cached
        start_height_km = self.scenario.transmitter.antenna.height_km
        medium = self._medium(frequency_hz, mode, 15.0, segment)
        scan = scan_ranges(
            medium, start_height_km, MIN_ELEVATION_DEG, MAX_ELEVATION_DEG,
            self.SCAN_POINTS,
        )
        skip = skip_distance_km(
            medium, start_height_km, MIN_ELEVATION_DEG, MAX_ELEVATION_DEG, scan=scan
        )
        self._segment_scans[key] = (medium, scan, skip)
        return self._segment_scans[key]

    def _hop_ranges(
        self, media: Sequence[RayMedium], elevation_deg: float
    ) -> Optional[List[float]]:
        """Ground range of each hop at one launch elevation, or None.

        One elevation for the whole circuit: a ground reflection preserves
        it, so every hop leaves at the angle the transmitter chose.  What
        differs between hops is the ionosphere they climb into, and
        therefore how far each one reaches.
        """
        start_height_km = self.scenario.transmitter.antenna.height_km
        ranges: List[float] = []
        for medium in media:
            values, _, _ = _batch_hop_geometry(
                medium, np.array([elevation_deg]), start_height_km
            )
            if not np.isfinite(values[0]):
                return None            # one hop escapes: the circuit is broken
            ranges.append(float(values[0]))
        return ranges

    def _ground_loss_db(
        self, elevation_deg: float, frequency_hz: float, hop_ranges: Sequence[float]
    ) -> float:
        """Loss at the intermediate ground reflections.

        Each bounce is charged for the surface it actually lands on, not for
        the path's dominant one.  A North Atlantic circuit reflects off sea
        water, and a polar one off ice; charging both as average ground is
        several decibels wrong per bounce in opposite directions.
        """
        if len(hop_ranges) <= 1:
            return 0.0
        horizontal = self.scenario.transmitter.antenna.antenna_type.value.startswith(
            ("horizontal", "inverted")
        )
        total = 0.0
        travelled = 0.0
        for span in hop_ranges[:-1]:
            travelled += span
            point = destination_point(
                self.scenario.transmitter.location, self.scenario.bearing_deg, travelled
            )
            total += ground_reflection_loss_db(
                elevation_deg,
                frequency_hz,
                classify_surface(point),
                horizontal,
                self.scenario.weather.effective_moisture_factor,
                self.scenario.weather.sea_state,
            )
        return total

    def _path_sections(
        self, distance_km: float
    ) -> Sequence[Tuple[float, GroundType]]:
        """The land and sea stretches out to ``distance_km`` along the bearing."""
        key = round(distance_km, 3)
        cached = self._sections.get(key)
        if cached is not None:
            return cached
        far = destination_point(
            self.scenario.transmitter.location, self.scenario.bearing_deg, distance_km
        )
        sections = tuple(
            path_sections(
                self.scenario.transmitter.location, far, distance_km,
                self.scenario.path_samples,
            )
        )
        self._sections[key] = sections
        return sections

    def _ground_wave(self, frequency_hz: float, distance_km: float) -> "GroundWave":
        """The surface-wave route to the receiver.

        Always built, never conditioned on distance.  Whether the ground
        wave matters is what the link budget is for; deciding in advance
        that "it is a short path so use ground wave, a long one so use
        skywave" is the kind of rule that answers a 56 km link with a
        two-hop F2 mode and a 3000 km one with a surface wave.
        """
        scenario = self.scenario
        moisture = scenario.weather.effective_moisture_factor
        loss = ground_wave_loss_db(
            distance_km=distance_km,
            frequency_hz=frequency_hz,
            sections=self._path_sections(distance_km),
            moisture_factor=moisture,
        )
        tx_gain = scenario.transmitter.antenna.ground_wave_gain_dbi(
            frequency_hz, scenario.bearing_deg, moisture
        )
        rx_gain = scenario.receiver.antenna.ground_wave_gain_dbi(
            frequency_hz, scenario.reverse_bearing_deg, moisture
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
            # The ground wave follows the surface, so its spreading is over
            # the great-circle distance itself -- no hop climbs above it and
            # no ray is longer than the arc.
            path_length_km=distance_km,
            absorption_loss_db=0.0,
            ground_reflection_loss_db=0.0,
            noise=noise,
            required_snr_db=scenario.receiver.required_snr_db,
            hops=0,
            rain_rate_mm_h=scenario.weather.rain_rate_mm_h,
            surface_wave_loss_db=loss.total_db,
        )
        return GroundWave(
            frequency_hz=frequency_hz,
            distance_km=distance_km,
            loss=loss,
            budget=budget,
        )

    def _build_mode(
        self,
        frequency_hz: float,
        mode: Mode,
        elevation_deg: float,
        fractions: Sequence[Tuple[float, float]],
        paths: Sequence[RayPath],
    ) -> PropagationMode:
        """Assemble the full budget for one candidate circuit.

        The hops are already traced, each in the ionosphere above its own
        stretch of the path.  What is left is to charge each one for the
        absorption where it actually is, and the reflections for the ground
        they actually land on.
        """
        scenario = self.scenario
        hops = len(paths)
        theta = self._theta_rad(elevation_deg)

        losses = [
            absorption_db(
                path, self.column, frequency_hz, mode, self.magnetic_field, theta,
                fraction_offset=index / hops, fraction_span=1.0 / hops,
            )
            for index, path in enumerate(paths)
        ]
        absorption = combine_absorption(losses)
        hop_ranges = [path.ground_range_km for path in paths]

        tx_gain = scenario.transmitter.antenna.gain_dbi(
            elevation_deg, frequency_hz, scenario.bearing_deg,
            scenario.weather.effective_moisture_factor,
        )
        rx_gain = scenario.receiver.antenna.gain_dbi(
            elevation_deg, frequency_hz, scenario.reverse_bearing_deg,
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
            path_length_km=sum(path.geometric_path_km for path in paths),
            absorption_loss_db=absorption.total_db,
            ground_reflection_loss_db=self._ground_loss_db(
                elevation_deg, frequency_hz, hop_ranges
            ),
            noise=noise,
            required_snr_db=scenario.receiver.required_snr_db,
            hops=hops,
            rain_rate_mm_h=scenario.weather.rain_rate_mm_h,
        )
        return PropagationMode(
            hops, elevation_deg, mode, paths[0], absorption, budget,
            tuple(paths), tuple(hop_ranges),
        )

    #: Elevation samples in the range-against-elevation scan.
    SCAN_POINTS = int((MAX_ELEVATION_DEG - MIN_ELEVATION_DEG) * 2) + 1

    def scan_for(self, frequency_hz: float, mode: Mode) -> Tuple:
        """``(probe medium, elevation scan, skip distance)``, memoised.

        The medium depends weakly on elevation through the field/ray angle;
        this probes at a representative angle to find candidate launch
        angles, and the medium is rebuilt at the solved angle before
        anything is charged to the budget.
        """
        key = (frequency_hz, mode)
        cached = self._scans.get(key)
        if cached is not None:
            return cached

        start_height_km = self.scenario.transmitter.antenna.height_km
        probe = self._medium(frequency_hz, mode, 15.0)
        scan = scan_ranges(
            probe, start_height_km, MIN_ELEVATION_DEG, MAX_ELEVATION_DEG, self.SCAN_POINTS
        )
        skip = skip_distance_km(
            probe, start_height_km, MIN_ELEVATION_DEG, MAX_ELEVATION_DEG, scan=scan
        )
        self._scans[key] = (probe, scan, skip)
        return self._scans[key]

    def _solve_circuit_elevation(
        self,
        media: Sequence[RayMedium],
        target_km: float,
        low_deg: float,
        high_deg: float,
        tolerance_km: float = 1.0,
        low_total_km: Optional[float] = None,
        high_total_km: Optional[float] = None,
    ) -> Optional[float]:
        """Launch elevation whose hops together cover ``target_km``.

        Solved on the summed ground range of the actual traced hops, not on
        an interpolation of the scan: the scan locates the bracket, the
        trace decides the answer.

        By the Illinois variant of false position rather than by bisection.
        Every iteration costs one traced ray *per hop* -- five of them on a
        transpacific circuit -- and this is the most expensive loop in the
        package.  Range against elevation is smooth and monotonic inside a
        bracket, so interpolating through it converges in a third of the
        steps; the Illinois halving of the stale endpoint keeps the bracket
        valid, so it cannot run away from the root the way plain regula
        falsi can on a one-sided curve.
        """
        # The caller found this bracket by summing the cached per-segment
        # scans, so it already knows both endpoints exactly -- the scans and
        # this solver trace the same media through the same function.
        # Re-tracing them here would cost two hops-worth of rays per bracket
        # to reproduce numbers already in hand.
        if low_total_km is None:
            ranges = self._hop_ranges(media, low_deg)
            if ranges is None:
                return None
            low_total_km = sum(ranges)
        if high_total_km is None:
            ranges = self._hop_ranges(media, high_deg)
            if ranges is None:
                return None
            high_total_km = sum(ranges)
        low_error = low_total_km - target_km
        high_error = high_total_km - target_km
        if low_error == 0.0:
            return low_deg
        if high_error == 0.0:
            return high_deg
        if low_error * high_error > 0.0:
            return None            # the scan's bracket did not survive tracing

        # Best elevation seen, not the last one tried.  A bracket whose
        # summed range jumps -- one hop of the circuit starts escaping part
        # way across it -- has no root at all, and iterating to a cap and
        # then returning the midpoint would hand back an elevation that is
        # worse than several already visited.  Whatever comes back still
        # has to survive the traced-range check in :meth:`evaluate`.
        best_deg, best_error = (
            (low_deg, low_error) if abs(low_error) <= abs(high_error)
            else (high_deg, high_error)
        )
        for _ in range(32):
            span = high_error - low_error
            if span == 0.0:
                break
            middle = low_deg - low_error * (high_deg - low_deg) / span
            # Keep the trial strictly inside the bracket: a flat curve can
            # otherwise put it on an endpoint and stall.
            edge = 1e-9 * max(1.0, high_deg - low_deg)
            middle = min(max(middle, low_deg + edge), high_deg - edge)
            ranges = self._hop_ranges(media, middle)
            if ranges is None:
                return None
            error = sum(ranges) - target_km
            if abs(error) < abs(best_error):
                best_deg, best_error = middle, error
            if abs(error) <= 0.05 * tolerance_km:
                return middle
            if error * low_error < 0.0:
                high_deg, high_error = middle, error
                low_error *= 0.5              # Illinois: halve the stale end
            else:
                low_deg, low_error = middle, error
                high_error *= 0.5
            if high_deg - low_deg < 1e-9:
                break
        return best_deg

    def max_hop_range_km(self, frequency_hz: float, mode: Mode = Mode.ORDINARY) -> Optional[float]:
        """Longest single hop this frequency can make, or None if none return."""
        _, scan, _ = self.scan_for(frequency_hz, mode)
        reachable = [r for r in scan[1] if r is not None]
        return max(reachable) if reachable else None

    def evaluate(
        self,
        frequency_hz: float,
        modes: Sequence[Mode] = DEFAULT_MODES,
        distance_km: Optional[float] = None,
    ) -> FrequencyReport:
        """Every way the signal reaches the receiver at this frequency.

        ``distance_km`` overrides the scenario's own path length, which is
        what a coverage sweep varies.  The ionosphere is still the column
        built for the scenario's path: the sweep asks "how far does this
        signal reach along this bearing", not "what if the receiver were
        somewhere else entirely".
        """
        if frequency_hz <= 0.0:
            raise ValueError("frequency must be positive")
        target_distance = self._distance_km if distance_km is None else distance_km
        if target_distance <= 0.0:
            raise ValueError("distance must be positive")

        key = (frequency_hz, tuple(modes), target_distance)
        cached = self._reports.get(key)
        if cached is not None:
            return cached

        found: List[PropagationMode] = []
        best_skip: Optional[float] = None

        for mode in modes:
            _, _, candidate = self.scan_for(frequency_hz, mode)
            if candidate is not None:
                best_skip = candidate if best_skip is None else min(best_skip, candidate)

            for hops in range(1, self.scenario.max_hops + 1):
                if target_distance / hops > math.pi * EARTH_RADIUS_KM:
                    continue
                if target_distance / hops > MAX_HOP_RANGE_KM:
                    # Geometry alone rules this out, whatever the ionosphere
                    # does: a ray turning at the very top of the modelled
                    # profile cannot cover more ground than this in one hop.
                    # Skipping it here saves tracing an elevation scan per
                    # hop for a circuit that cannot exist.
                    continue
                fractions = self.segment_fractions(hops)
                segments = [
                    self.scan_for_segment(frequency_hz, mode, span)
                    for span in fractions
                ]
                media = [segment[0] for segment in segments]

                # The whole circuit leaves at one elevation, so the total
                # ground range is the sum of what each hop reaches in its own
                # ionosphere.  Summing the cached per-segment scans gives that
                # curve for every candidate elevation at once; only the few
                # elevations that bracket the target are then traced properly.
                elevations = segments[0][1][0]
                per_segment = [segment[1][1] for segment in segments]
                totals: List[Optional[float]] = []
                for index in range(len(elevations)):
                    values = [ranges[index] for ranges in per_segment]
                    totals.append(
                        None if any(v is None for v in values) else sum(values)
                    )

                for index in range(len(elevations) - 1):
                    low_total, high_total = totals[index], totals[index + 1]
                    if low_total is None or high_total is None:
                        continue
                    if (low_total - target_distance) * (high_total - target_distance) > 0.0:
                        continue

                    elevation = self._solve_circuit_elevation(
                        media, target_distance,
                        float(elevations[index]), float(elevations[index + 1]),
                        low_total_km=low_total, high_total_km=high_total,
                    )
                    if elevation is None:
                        continue
                    if any(abs(elevation - m.launch_elevation_deg) < 1e-3
                           for m in found if m.hops == hops and m.mode is mode):
                        continue

                    start_height_km = self.scenario.transmitter.antenna.height_km
                    paths = []
                    for span in fractions:
                        medium = self._medium(frequency_hz, mode, elevation, span)
                        path = trace_ray(medium, elevation, start_height_km)
                        if path.escaped:
                            paths = []
                            break
                        paths.append(path)
                    if not paths:
                        continue
                    # Accept only because the traced circuit really lands on
                    # the receiver, never because the solver said it would.
                    reached = sum(path.ground_range_km for path in paths)
                    if abs(reached - target_distance) > max(5.0, 0.01 * target_distance):
                        continue
                    found.append(
                        self._build_mode(frequency_hz, mode, elevation, fractions, paths)
                    )

        report = FrequencyReport(
            frequency_hz, tuple(found), best_skip,
            self._ground_wave(frequency_hz, target_distance),
        )
        self._reports[key] = report
        return report

    def _frequency_grid(self, low_hz: float, high_hz: float, step_hz: float) -> List[float]:
        count = max(2, int(round((high_hz - low_hz) / step_hz)) + 1)
        return [low_hz + i * step_hz for i in range(count)]

    def maximum_usable_frequency_hz(
        self,
        low_hz: float = 2e6,
        high_hz: float = 50e6,
        step_hz: float = 2.5e5,
        tolerance_hz: float = 2.5e4,
        hint_hz: Optional[float] = None,
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

        ``hint_hz`` narrows the scan around a known nearby answer, which is
        what makes a MUF *distribution* affordable: the deciles sit within a
        factor of about 1.5 of the median, so the same circuit need not be
        rescanned from 2 MHz three times over.  The window is checked at its
        own edges and widened to the full range if the answer is not inside
        it, so the shortcut cannot silently return a wrong MUF.
        """
        if hint_hz is not None and hint_hz > 0.0:
            # The decile MUFs track foF2, which spans roughly x0.8 to x1.2
            # about its median, so a x0.7 to x1.4 window brackets them with
            # room to spare.  It is checked at both edges below and widened
            # to the full range if it does not, so tightening it trades scan
            # width for an occasional fallback, never for a wrong answer.
            window_low = max(low_hz, hint_hz * 0.70)
            window_high = min(high_hz, hint_hz * 1.40)
            narrow = self._frequency_grid(window_low, window_high, step_hz)
            flags = [self.evaluate(f).is_open for f in narrow]
            # Trust the window only if it brackets the transition: open at
            # the bottom, closed at the top.
            if flags and flags[0] and not flags[-1]:
                low_hz, high_hz = window_low, window_high
                grid, open_flags = narrow, flags
                highest_open = max(i for i, flag in enumerate(open_flags) if flag)
                low, high = grid[highest_open], grid[highest_open + 1]
                while high - low > tolerance_hz:
                    mid = 0.5 * (low + high)
                    if self.evaluate(mid).is_open:
                        low = mid
                    else:
                        high = mid
                return low

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
            "sporadic_e_foes_mhz": (
                self.sporadic_e.foes_mhz if self.sporadic_e is not None else None
            ),
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

    def ground_wave_report(self) -> List[dict]:
        """The surface route on each standard band.

        Reported next to the band scores rather than mixed into them: the
        two are different routes with different antennas, different
        polarisation and different reasons to fail, and a table that
        averaged them would be describing a link that does not exist.
        """
        rows = []
        for name, centre_mhz in STANDARD_BANDS_MHZ:
            nearest = min(
                self.reports, key=lambda r: abs(r.frequency_mhz - centre_mhz)
            )
            if abs(nearest.frequency_mhz - centre_mhz) > 0.5:
                continue
            wave = nearest.ground_wave
            if wave is None:
                continue
            rows.append({
                "band": name,
                "frequency_mhz": centre_mhz,
                "margin_db": wave.margin_db,
                "snr_db": wave.budget.snr_db,
                "excess_loss_db": wave.loss.total_db,
                "surface_loss_db": wave.loss.surface_db,
                "curvature_loss_db": wave.loss.curvature_db,
                "sea_fraction": wave.loss.sea_fraction,
                "usable": wave.margin_db >= 0.0,
            })
        return rows

    def summary(self) -> dict:
        best = self.best_report()
        return {
            "scenario": self.scenario.summary(),
            "ground_wave": self.ground_wave_report(),
            "conditions": self.conditions,
            "muf_mhz": self.muf_mhz,
            "lof_mhz": self.lof_mhz,
            "fot_mhz": self.optimum_working_frequency_mhz,
            "best_frequency_mhz": best.frequency_mhz if best else None,
            "bands": self.band_report(),
            "spectrum": [r.summary() for r in self.reports],
        }
