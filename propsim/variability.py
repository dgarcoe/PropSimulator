"""Day-to-day variability of the ionosphere.

A propagation prediction that returns one number is answering the wrong
question.  The ionosphere does not repeat itself: on a given day of a given
month, at a given hour, foF2 scatters around its monthly median by tens of
percent.  Two circuits with the same median SNR can differ completely in how
often they actually work, and only a distribution distinguishes them.

This module supplies the spread.  ``foF2`` is treated as log-normally
distributed about the monthly median -- it is a positive quantity whose
scatter is multiplicative -- and characterised by its decile factors: the
values exceeded on 10% and 90% of days.  The spread widens at equatorial
latitudes, where the fountain is irregular, and at auroral ones, where
storms bite hardest; it is narrowest at mid-latitudes, and it is wider by
night than by day.

The decile factors are empirical and are labelled as such.  What matters
structurally is that the *shape* of the answer is right: a reliability, not
a point estimate dressed up as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

__all__ = ["QUANTILES", "fof2_decile_factors", "fof2_multipliers",
           "reliability_from_samples", "VariabilitySpread"]

#: Cumulative probabilities sampled when building a distribution.  Five
#: points span the deciles without making the prediction five times slower
#: than it needs to be.
QUANTILES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class VariabilitySpread:
    """Decile factors for foF2 about its monthly median."""

    lower_decile: float          # multiplier exceeded on 90% of days
    upper_decile: float          # multiplier exceeded on 10% of days

    def __post_init__(self) -> None:
        if not 0.3 <= self.lower_decile < 1.0:
            raise ValueError(f"lower decile factor {self.lower_decile} is implausible")
        if not 1.0 < self.upper_decile <= 2.5:
            raise ValueError(f"upper decile factor {self.upper_decile} is implausible")

    @property
    def log_sigma(self) -> float:
        """Standard deviation of ln(foF2), from the decile spread.

        For a normal distribution the 10th and 90th percentiles are 2.5631
        standard deviations apart.
        """
        return math.log(self.upper_decile / self.lower_decile) / 2.5631

    def multiplier_at(self, probability: float) -> float:
        """foF2 multiplier not exceeded with the given probability."""
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly between 0 and 1")
        median = math.sqrt(self.lower_decile * self.upper_decile)
        return median * math.exp(self.log_sigma * _normal_quantile(probability))


def _normal_quantile(probability: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to better than 1.15e-9 in absolute value over the whole range,
    which is far finer than anything the decile factors themselves justify.
    """
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low, high = 0.02425, 1.0 - 0.02425

    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if probability > high:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fof2_decile_factors(
    geomagnetic_latitude_deg: float, sunlit_fraction: float, kp: float = 2.0
) -> VariabilitySpread:
    """Decile factors for foF2 at a given place, time and disturbance level.

    Mid-latitude daytime is the quiet case, roughly +-15%.  The spread grows
    towards the equatorial anomaly and towards the auroral oval, grows at
    night when the layer is unsupported by production, and grows sharply
    during a geomagnetic storm.
    """
    magnitude = abs(geomagnetic_latitude_deg)
    base = 0.15

    # Equatorial anomaly: the fountain is irregular day to day.
    base += 0.07 * math.exp(-((magnitude - 12.0) / 14.0) ** 2)
    # Auroral oval: storm-driven depletion is highly variable.
    base += 0.10 * math.exp(-((magnitude - 65.0) / 15.0) ** 2)
    # Night: no production to hold the layer steady.
    base += 0.05 * (1.0 - max(0.0, min(1.0, sunlit_fraction)))
    # Disturbance.
    base += 0.03 * max(0.0, kp - 3.0)

    spread = min(base, 0.55)
    return VariabilitySpread(lower_decile=1.0 - spread, upper_decile=1.0 + spread)


def fof2_multipliers(spread: VariabilitySpread,
                     quantiles: Sequence[float] = QUANTILES) -> Tuple[float, ...]:
    """foF2 multipliers to evaluate, one per sampled quantile."""
    return tuple(spread.multiplier_at(q) for q in quantiles)


def reliability_from_samples(
    margins_db: Sequence[float | None],
    quantiles: Sequence[float] = QUANTILES,
) -> float:
    """Fraction of days the circuit closes, from margins sampled by quantile.

    ``margins_db[i]`` is the link margin on a day at the ``quantiles[i]``
    point of the foF2 distribution; ``None`` means no ray reached the
    receiver at all, which is a failure, not missing data.

    Where every sample closes or every sample fails, the answer is 1 or 0.
    Otherwise the crossing is located by interpolating between the bracketing
    quantiles: a circuit whose margin passes through zero between the 30th
    and 50th percentile of foF2 works on somewhere between 50% and 70% of
    days, and linear interpolation in margin places it inside that window.

    A normal fit is deliberately not used here.  It would extrapolate
    confidently past the sampled range and report, say, 3% reliability for a
    circuit that in fact failed at every quantile examined.
    """
    if len(margins_db) != len(quantiles):
        raise ValueError("one margin is required per quantile")
    if not margins_db:
        raise ValueError("no samples")

    order = sorted(range(len(quantiles)), key=lambda i: quantiles[i])
    probabilities = [quantiles[i] for i in order]
    margins = [margins_db[i] for i in order]
    closes = [m is not None and m >= 0.0 for m in margins]

    if all(closes):
        return 1.0
    if not any(closes):
        return 0.0

    # The common case: the circuit fails at the poor end of the distribution
    # and closes above some crossing.  Locate the crossing by interpolating
    # the margin between the two bracketing quantiles, which resolves it far
    # more finely than the sample spacing.
    first_close = min(i for i, ok in enumerate(closes) if ok)
    monotone = all(closes[i] for i in range(first_close, len(closes)))
    if monotone:
        if first_close == 0:
            return 1.0
        below, above = margins[first_close - 1], margins[first_close]
        p_below, p_above = probabilities[first_close - 1], probabilities[first_close]
        if below is None or above is None or above == below:
            crossing = 0.5 * (p_below + p_above)
        else:
            fraction = (0.0 - below) / (above - below)
            crossing = p_below + max(0.0, min(1.0, fraction)) * (p_above - p_below)
        return max(0.0, min(1.0, 1.0 - crossing))

    # Margin is not always monotonic in foF2.  Near the critical frequency
    # the set of available modes changes discontinuously -- a hop count or a
    # reflecting layer drops out -- so a *worse* ionosphere can occasionally
    # give a *better* margin by forcing a different, less absorbed path.
    # Interpolating a single crossing is meaningless there, so fall back to
    # the probability mass carried by the quantiles that do close.  The
    # answer is coarser, and it is not an extrapolation.
    total = 0.0
    for index, ok in enumerate(closes):
        if not ok:
            continue
        lower = 0.0 if index == 0 else 0.5 * (probabilities[index - 1] + probabilities[index])
        upper = (
            1.0
            if index == len(probabilities) - 1
            else 0.5 * (probabilities[index] + probabilities[index + 1])
        )
        total += upper - lower
    return max(0.0, min(1.0, total))
