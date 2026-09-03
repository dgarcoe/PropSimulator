"""Circuit reliability: how often a link works, not whether it works today.

A single deterministic SNR answers a question nobody has.  An operator wants
to know whether a circuit is dependable, and that is a probability over the
day-to-day scatter of the ionosphere -- the quantity every serious HF
prediction reports and the one a point estimate silently withholds.

:class:`ReliabilityPredictor` evaluates the same scenario against several
draws from the foF2 distribution and reports the resulting spread of link
margin, together with the fraction of days the circuit closes.  Everything
downstream of the ionosphere -- absorption, antennas, noise, budget -- is
the existing chain, run once per draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .engine import PropagationEngine, STANDARD_BANDS_MHZ
from .scenario import Scenario
from .solar import local_solar_time_hours
from .sporadic_e import SporadicELayer, sporadic_e_for
from .variability import (
    QUANTILES,
    VariabilitySpread,
    fof2_decile_factors,
    fof2_multipliers,
    reliability_from_samples,
)

__all__ = ["FrequencyReliability", "ReliabilityPredictor"]


@dataclass(frozen=True)
class FrequencyReliability:
    """The distribution of link margin at one frequency."""

    frequency_hz: float
    #: Margin at each sampled quantile of foF2, ``None`` where no ray lands.
    margins_db: Tuple[Optional[float], ...]
    quantiles: Tuple[float, ...]
    reliability: float
    spread: VariabilitySpread
    #: Reliability of the regular ionosphere alone, before sporadic E.
    reliability_without_es: float = 0.0
    #: Reliability given that a sporadic-E patch is present.
    reliability_with_es: float = 0.0
    #: Probability such a patch is present at all.
    sporadic_e_probability: float = 0.0
    #: Margin after multipath fading, at each sampled quantile.
    effective_margins_db: Tuple[Optional[float], ...] = ()
    #: Fade margin charged, dB, on the median-ionosphere day.
    fade_margin_db: float = 0.0
    time_availability: float = 0.9

    @property
    def frequency_mhz(self) -> float:
        return self.frequency_hz / 1e6

    def _at(self, probability: float) -> Optional[float]:
        index = min(
            range(len(self.quantiles)),
            key=lambda i: abs(self.quantiles[i] - probability),
        )
        return self.margins_db[index]

    @property
    def median_margin_db(self) -> Optional[float]:
        return self._at(0.5)

    @property
    def lower_decile_margin_db(self) -> Optional[float]:
        """Margin on a bad ionospheric day -- the one that matters."""
        return self._at(0.1)

    @property
    def upper_decile_margin_db(self) -> Optional[float]:
        return self._at(0.9)

    def summary(self) -> dict:
        return {
            "frequency_mhz": self.frequency_mhz,
            "reliability": self.reliability,
            "median_margin_db": self.median_margin_db,
            "lower_decile_margin_db": self.lower_decile_margin_db,
            "upper_decile_margin_db": self.upper_decile_margin_db,
            "quantiles": list(self.quantiles),
            "margins_db": list(self.margins_db),
            "fof2_lower_decile": self.spread.lower_decile,
            "fof2_upper_decile": self.spread.upper_decile,
            "reliability_without_es": self.reliability_without_es,
            "reliability_with_es": self.reliability_with_es,
            "sporadic_e_probability": self.sporadic_e_probability,
            "fade_margin_db": self.fade_margin_db,
            "time_availability": self.time_availability,
        }


class ReliabilityPredictor:
    """Runs a scenario against several draws of the ionosphere.

    Construction builds one :class:`~propsim.engine.PropagationEngine` per
    sampled quantile.  That is the expensive part and it happens once; each
    frequency afterwards costs one evaluation per engine.
    """

    def __init__(
        self,
        scenario: Scenario,
        quantiles: Sequence[float] = QUANTILES,
        spread: Optional[VariabilitySpread] = None,
        include_sporadic_e: bool = True,
        time_availability: float = 0.9,
        median_engine: Optional[PropagationEngine] = None,
    ) -> None:
        self.scenario = scenario
        self.quantiles = tuple(quantiles)
        # Two timescales, both required for an honest answer.  The quantiles
        # sample how the ionosphere differs from day to day; the time
        # availability charges for how the signal fades within any one of
        # them, as several modes drift in and out of phase.
        self.time_availability = time_availability

        # The median engine is needed first to know where the path runs and
        # how much of it is lit, which is what sets the spread.  A caller
        # that has already built one for the same scenario can hand it over:
        # it arrives with its frequency reports memoised, so the median MUF
        # search costs nothing instead of repeating a scan already done.
        if median_engine is not None:
            if median_engine.fof2_multiplier != 1.0 or median_engine.sporadic_e is not None:
                raise ValueError(
                    "median_engine must be the undisturbed median ionosphere"
                )
            median = median_engine
        else:
            median = PropagationEngine(scenario, fof2_multiplier=1.0)
        self.spread = spread or fof2_decile_factors(
            median.geomagnetic_latitude_deg,
            median.illumination.sunlit_fraction,
            scenario.space_weather.kp,
        )
        self.multipliers = fof2_multipliers(self.spread, self.quantiles)
        self.engines: List[PropagationEngine] = [
            median if math.isclose(m, 1.0, abs_tol=1e-9)
            else PropagationEngine(scenario, fof2_multiplier=m)
            for m in self.multipliers
        ]
        self.median_engine = median

        # Sporadic E is composed in as a probability, never switched on.
        # There is no fact about whether a patch is present on a given hour;
        # there is an occurrence rate, and the honest output is the mixture.
        self.sporadic_e_probability = 0.0
        self.sporadic_e_layer: Optional[SporadicELayer] = None
        self.es_engines: List[PropagationEngine] = []
        if include_sporadic_e:
            midpoint = median.midpoint
            hour = local_solar_time_hours(midpoint, scenario.when)
            probability, layer = sporadic_e_for(
                scenario.when,
                median.geomagnetic_latitude_deg,
                hour,
                scenario.space_weather.kp,
            )
            if probability > 0.01:
                self.sporadic_e_probability = probability
                self.sporadic_e_layer = layer
                # foF2 variability barely matters when an Es patch is doing
                # the reflecting, so the Es branch is evaluated at the median
                # ionosphere only -- one engine, not five.
                self.es_engines = [
                    PropagationEngine(scenario, fof2_multiplier=1.0, sporadic_e=layer)
                ]

    def at(self, frequency_hz: float) -> FrequencyReliability:
        """Margin distribution and reliability at one frequency.

        The reported reliability is the mixture

            P(Es) x reliability_with_Es + (1 - P(Es)) x reliability_without

        so a band that is dead through the F layer but lives on sporadic E
        is reported at the rate sporadic E actually occurs, not as open or
        as closed.
        """
        reports = [engine.evaluate(frequency_hz) for engine in self.engines]
        margins = tuple(report.margin_db for report in reports)
        # Reliability is judged on the margin *after* multipath fading is
        # paid for, not on the mean-power margin the budget reports.
        effective = tuple(
            report.effective_margin_db(self.time_availability) for report in reports
        )
        without = reliability_from_samples(effective, self.quantiles)

        with_es = without
        if self.es_engines:
            es_report = self.es_engines[0].evaluate(frequency_hz)
            es_margin = es_report.effective_margin_db(self.time_availability)
            with_es = 1.0 if (es_margin is not None and es_margin >= 0.0) else 0.0

        probability = self.sporadic_e_probability
        combined = probability * with_es + (1.0 - probability) * without

        return FrequencyReliability(
            frequency_hz=frequency_hz,
            margins_db=margins,
            quantiles=self.quantiles,
            reliability=max(0.0, min(1.0, combined)),
            spread=self.spread,
            reliability_without_es=without,
            reliability_with_es=with_es,
            sporadic_e_probability=probability,
            effective_margins_db=effective,
            fade_margin_db=self.median_engine.evaluate(frequency_hz).fade_margin_db(
                self.time_availability
            ),
            time_availability=self.time_availability,
        )

    def band_reliability(self) -> List[dict]:
        """Reliability of every standard band, best first.

        This replaces the heuristic band score with a number that means
        something: the fraction of days the circuit closes.  Bands are ranked
        by it, with median margin breaking ties.
        """
        rows = []
        for name, centre_mhz in STANDARD_BANDS_MHZ:
            result = self.at(centre_mhz * 1e6)
            rows.append({
                "band": name,
                "frequency_mhz": centre_mhz,
                "reliability": result.reliability,
                "median_margin_db": result.median_margin_db,
                "lower_decile_margin_db": result.lower_decile_margin_db,
                "upper_decile_margin_db": result.upper_decile_margin_db,
                "open": result.median_margin_db is not None,
                "reliability_without_es": result.reliability_without_es,
                "sporadic_e_probability": result.sporadic_e_probability,
                "fade_margin_db": result.fade_margin_db,
            })
        return sorted(
            rows,
            key=lambda r: (
                r["reliability"],
                r["median_margin_db"] if r["median_margin_db"] is not None else -1e9,
            ),
            reverse=True,
        )

    def muf_distribution_hz(
        self, step_hz: float = 5e5, hint_hz: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """MUF at the lower decile, median and upper decile of foF2.

        The spread between them is the honest width of a MUF prediction, and
        it is usually a couple of megahertz -- which is why quoting a MUF to
        two decimal places, as a deterministic model invites, overstates what
        is known.
        """
        wanted = {"median": 0.5, "lower_decile": 0.1, "upper_decile": 0.9}
        result: Dict[str, Optional[float]] = {}
        # The median is found first and then seeds the decile searches: the
        # MUF tracks foF2, so the deciles are close by and do not need a scan
        # from the bottom of the band each.
        for label, probability in wanted.items():
            index = min(
                range(len(self.quantiles)),
                key=lambda i: abs(self.quantiles[i] - probability),
            )
            seed = hint_hz if label == "median" else result.get("median") or hint_hz
            result[label] = self.engines[index].maximum_usable_frequency_hz(
                step_hz=step_hz, hint_hz=seed
            )
        return {key: result[key] for key in ("lower_decile", "median", "upper_decile")}

    def summary(self, step_hz: float = 5e5) -> dict:
        muf = self.muf_distribution_hz(step_hz)
        return {
            "scenario": self.scenario.summary(),
            "conditions": self.median_engine.conditions(),
            "fof2_spread": {
                "lower_decile": self.spread.lower_decile,
                "upper_decile": self.spread.upper_decile,
                "log_sigma": self.spread.log_sigma,
            },
            "muf_mhz": {
                key: (value / 1e6 if value else None) for key, value in muf.items()
            },
            "sporadic_e": {
                "probability": self.sporadic_e_probability,
                "foes_mhz": (
                    self.sporadic_e_layer.foes_mhz if self.sporadic_e_layer else None
                ),
                "height_km": (
                    self.sporadic_e_layer.height_km if self.sporadic_e_layer else None
                ),
            },
            "bands": self.band_reliability(),
        }
