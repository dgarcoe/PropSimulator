"""Multipath and fading: the variation within a day, not between days.

:mod:`propsim.variability` describes how the ionosphere differs from one day
to the next.  This module describes what happens over seconds and minutes on
any one of them.

Several modes usually reach the receiver at once -- a one-hop and a two-hop,
a low ray and a high ray, an ordinary and an extraordinary component -- with
slightly different path lengths.  Their relative phases drift as the
reflection heights move, so the resultant amplitude fades.  A deterministic
link budget reports the *mean* power and therefore overstates how much of
the time the signal is actually usable: on a Rayleigh channel the level
exceeded 90% of the time is about 10 dB below the mean.

Two consequences are computed here.

**Fade depth.** The modes are characterised by a Rician K factor, the ratio
of the dominant mode's power to the sum of the rest.  A single clean mode is
K = infinity and does not fade; several comparable modes approach Rayleigh.
The margin a circuit needs for a given time availability follows.

**Delay spread.** The modes' group delays differ, which smears the signal in
time.  Compared with the signal bandwidth this decides whether the channel
fades flat -- every frequency in the channel rising and falling together, an
inconvenience -- or selectively, which distorts the signal itself and cannot
be cured with more power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

__all__ = ["MultipathProfile", "multipath_profile", "fade_depth_db",
           "rician_k_factor", "required_margin_db"]


def _bessel_i0(x: float) -> float:
    """Modified Bessel function of the first kind, order zero.

    Series below the crossover, asymptotic expansion above it; keeping the
    package free of a SciPy dependency for one function.
    """
    if x < 0.0:
        x = -x
    if x < 50.0:
        # All terms are positive, so there is no cancellation and the series
        # is exact to machine precision well past the crossover.  The limit
        # is overflow, not convergence, and that is far above x = 50.
        total = 1.0
        term = 1.0
        quarter = 0.25 * x * x
        for k in range(1, 300):
            term *= quarter / (k * k)
            total += term
            if term < 1e-17 * total:
                break
        return total

    # Asymptotic: e^x / sqrt(2 pi x) * sum a_k, a_k = a_{k-1} (2k-1)^2 / (8 k x).
    # Six terms rather than three: the extra ones are three multiplications
    # and they take the error at the crossover from 3e-6 to below 1e-9.
    coefficient = 1.0
    series = 1.0
    for k in range(1, 7):
        coefficient *= (2 * k - 1) ** 2 / (8.0 * k * x)
        series += coefficient
    return math.exp(x) / math.sqrt(2.0 * math.pi * x) * series


def rician_k_factor(mode_powers_db: Sequence[float]) -> float:
    """Ratio of the strongest mode's power to the sum of the others.

    ``inf`` when a single mode arrives -- nothing to interfere with, so no
    fading.  Zero when the dominant mode carries none of the power, which is
    the Rayleigh limit.
    """
    if not mode_powers_db:
        raise ValueError("no modes")
    powers = sorted((10.0 ** (p / 10.0) for p in mode_powers_db), reverse=True)
    if len(powers) == 1:
        return math.inf
    scattered = sum(powers[1:])
    if scattered <= 0.0:
        return math.inf
    return powers[0] / scattered


def _rician_power_cdf(x: float, k_factor: float) -> float:
    """P(instantaneous power <= x * mean power) for a Rician channel.

    Integrated numerically from the amplitude density.  Doing it this way
    avoids the Marcum Q-function and stays accurate at both limits, which a
    truncated series form does not.
    """
    if x <= 0.0:
        return 0.0
    if math.isinf(k_factor):
        return 0.0 if x < 1.0 else 1.0

    # Normalise so the mean power is 1: nu^2 + 2 sigma^2 = 1.
    sigma_squared = 1.0 / (2.0 * (k_factor + 1.0))
    sigma = math.sqrt(sigma_squared)
    nu = math.sqrt(k_factor / (k_factor + 1.0))

    upper = math.sqrt(x)

    # The density is a spike of width ~sigma centred near r = nu, and sigma
    # shrinks as 1/sqrt(K).  A fixed step over [0, upper] steps straight over
    # that spike once K is large, the integral comes back as zero, and the
    # bisection above then runs to its bracket and reports a *negative* fade
    # depth.  So the range is trimmed to where the density actually lives and
    # the step is tied to sigma.
    lower = max(0.0, nu - 12.0 * sigma)
    if upper <= lower:
        return 0.0
    steps = int(min(20000, max(800, (upper - lower) / (sigma / 8.0))))
    step = (upper - lower) / steps
    total = 0.0
    for i in range(steps + 1):
        r = lower + i * step
        exponent = -(r * r + nu * nu) / (2.0 * sigma_squared)
        argument = r * nu / sigma_squared
        if argument < 15.0:
            # Bessel is O(1) here, so the exponent alone decides underflow.
            value = (
                0.0
                if exponent < -700.0
                else (r / sigma_squared) * math.exp(exponent) * _bessel_i0(argument)
            )
        else:
            # I0(a) ~ e^a / sqrt(2 pi a): fold that growth into the exponent
            # rather than evaluating either factor alone.  Both are enormous
            # and they very nearly cancel -- at K = 1000 the exponent is
            # about -2000 and the argument about +2000 -- so testing the
            # exponent for underflow *before* combining them zeroes the whole
            # integrand and the distribution collapses to nothing.
            combined = exponent + argument
            if combined < -700.0:
                value = 0.0
            else:
                series = 1.0 + 1.0 / (8.0 * argument)
                value = (
                    (r / sigma_squared)
                    * math.exp(combined)
                    / math.sqrt(2.0 * math.pi * argument)
                    * series
                )
        weight = 0.5 if i in (0, steps) else 1.0
        total += weight * value * step
    return min(1.0, max(0.0, total))


def fade_depth_db(k_factor: float, availability: float = 0.9) -> float:
    """Depth below the mean exceeded for ``availability`` of the time, dB.

    ``fade_depth_db(0.0, 0.9)`` is the classic Rayleigh answer, about 10 dB:
    a signal whose mean is 10 dB above the noise floor is *below* it a tenth
    of the time.
    """
    if not 0.0 < availability < 1.0:
        raise ValueError("availability must lie strictly between 0 and 1")
    if math.isinf(k_factor):
        return 0.0

    target = 1.0 - availability          # outage probability
    low, high = 1e-6, 20.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if _rician_power_cdf(mid, k_factor) < target:
            low = mid
        else:
            high = mid
    level = 0.5 * (low + high)
    # A fade depth is a loss relative to the mean, so it cannot be negative:
    # the 10th percentile of a fading power distribution is never above its
    # own mean.  Clamping here is a guard, not the fix -- the integration
    # above is what has to be right.
    return max(0.0, -10.0 * math.log10(max(level, 1e-12)))


def required_margin_db(
    mode_powers_db: Sequence[float], availability: float = 0.9
) -> float:
    """Extra margin a circuit needs to be usable that fraction of the time."""
    return fade_depth_db(rician_k_factor(mode_powers_db), availability)


@dataclass(frozen=True)
class MultipathProfile:
    """How the arriving modes combine, and what that costs."""

    mode_count: int
    k_factor: float
    #: RMS spread of group delay across the modes, milliseconds.
    delay_spread_ms: float
    #: Frequency separation over which the channel stays correlated, Hz.
    coherence_bandwidth_hz: float
    signal_bandwidth_hz: float
    fade_margin_db: float
    availability: float

    @property
    def is_frequency_selective(self) -> bool:
        """True when the channel distorts the signal rather than just fading it.

        Selective fading cannot be cured with power: different parts of the
        occupied bandwidth fade independently, so the signal arrives
        misshapen however loudly it is sent.
        """
        return self.signal_bandwidth_hz > self.coherence_bandwidth_hz

    @property
    def is_effectively_single_mode(self) -> bool:
        return math.isinf(self.k_factor) or self.k_factor > 100.0

    def summary(self) -> dict:
        return {
            "mode_count": self.mode_count,
            "k_factor_db": (
                None if math.isinf(self.k_factor) else 10.0 * math.log10(max(self.k_factor, 1e-12))
            ),
            "delay_spread_ms": self.delay_spread_ms,
            "coherence_bandwidth_hz": self.coherence_bandwidth_hz,
            "frequency_selective": self.is_frequency_selective,
            "fade_margin_db": self.fade_margin_db,
            "availability": self.availability,
        }


def multipath_profile(
    mode_powers_db: Sequence[float],
    mode_delays_ms: Sequence[float],
    signal_bandwidth_hz: float,
    availability: float = 0.9,
) -> MultipathProfile:
    """Characterise the multipath channel formed by the arriving modes."""
    if len(mode_powers_db) != len(mode_delays_ms):
        raise ValueError("one delay is required per mode")
    if not mode_powers_db:
        raise ValueError("no modes")
    if signal_bandwidth_hz <= 0.0:
        raise ValueError("bandwidth must be positive")

    powers = [10.0 ** (p / 10.0) for p in mode_powers_db]
    total_power = sum(powers)
    mean_delay = sum(p * d for p, d in zip(powers, mode_delays_ms)) / total_power
    variance = sum(
        p * (d - mean_delay) ** 2 for p, d in zip(powers, mode_delays_ms)
    ) / total_power
    delay_spread_ms = math.sqrt(max(variance, 0.0))

    # Bc ~ 1 / (2 pi tau_rms).  A single mode has no spread and therefore an
    # unbounded coherence bandwidth.
    if delay_spread_ms <= 0.0:
        coherence_hz = math.inf
    else:
        coherence_hz = 1.0 / (2.0 * math.pi * delay_spread_ms * 1e-3)

    k_factor = rician_k_factor(mode_powers_db)
    return MultipathProfile(
        mode_count=len(mode_powers_db),
        k_factor=k_factor,
        delay_spread_ms=delay_spread_ms,
        coherence_bandwidth_hz=coherence_hz,
        signal_bandwidth_hz=signal_bandwidth_hz,
        fade_margin_db=fade_depth_db(k_factor, availability),
        availability=availability,
    )
