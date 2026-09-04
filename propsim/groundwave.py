"""Ground wave: the part of the signal that never leaves the surface.

Everything else in this package is skywave -- a ray launched into the
ionosphere and bent back down.  Below the skip distance there is no such
ray, and a skywave-only model answers a 56 km link on 80 metres with
"nothing reaches you", which is not a conservative answer but a wrong one:
at 3.65 MHz a 56 km path is an unremarkable ground-wave contact and the
ionosphere has nothing to do with it.

Three separate things set the field, and each is charged on its own so the
breakdown can be read:

1. **Surface dissipation.**  The wave drags its own return currents through
   a lossy conductor and pays for them.  This is Sommerfeld's flat-earth
   attenuation function ``A(p)``, evaluated exactly rather than through one
   of the rational fits: those are good to about a decibel over sea and
   average ground and drift to four and a half over fresh water, where the
   numerical distance turns almost pure imaginary and the fit's phase
   correction overshoots.
2. **Earth curvature.**  Beyond the horizon the surface falls away from the
   wavefront and the field enters a shadow.  This is the leading term of
   the Fock residue series.
3. **Polarisation and height coupling.**  The surface wave is vertically
   polarised and its strength varies with height above the ground.  Both
   live on :meth:`propsim.antenna.AntennaSpec.ground_wave_gain_dbi`, not
   here, because they are properties of the antenna rather than of the
   path.

Mixed land/sea paths go through Millington's method.  A sea path that ends
with 200 km of land is not the same link as one that starts with it -- the
field recovers when it moves back over water and does not when it does not
-- and averaging the ground constants along the path loses that asymmetry
entirely.

What this module does not claim
-------------------------------
The curvature term is evaluated in the **good-conductor limit** of the
residue series (the boundary condition ``w'(t) = 0``, whose leading root is
``1.01879 exp(i pi/3)``).  That is close to right over sea water, which is
where a ground wave travels far enough for curvature to matter at all.
Over land the true root moves and the shadow decays up to 2.3 times faster
-- but over land the flat-earth term has already spent 50 dB before the
horizon is reached, so the path is dead long before the difference could
show.  The model is therefore accurate where the ground wave lives and
optimistic where it does not, and the optimism is bounded by a factor this
docstring names rather than hidden.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .antenna import GroundType, complex_permittivity, surface_impedance
from .constants import SPEED_OF_LIGHT

__all__ = [
    "GroundWaveLoss",
    "numerical_distance",
    "faddeeva",
    "sommerfeld_attenuation",
    "surface_attenuation_db",
    "curvature_loss_db",
    "millington_surface_db",
    "ground_wave_loss_db",
    "SHADOW_ONSET_X",
    "EFFECTIVE_EARTH_RADIUS_KM",
]

#: Effective Earth radius under standard refraction, 4/3 of the geometric
#: one.  The ground wave is bent by the same tropospheric gradient that
#: bends a line-of-sight link, and using the geometric radius would put the
#: horizon -- and with it the onset of the shadow -- 15% too close.
EFFECTIVE_EARTH_RADIUS_KM = 4.0 / 3.0 * 6371.0088

#: First root of ``w'(t) = 0``: ``|a'_1| exp(i pi/3)`` with ``a'_1`` the
#: first zero of the Airy derivative.  Only its imaginary part is used, and
#: it is the decay rate of the field in the shadow.
_AIRY_FIRST_DERIVATIVE_ZERO = 1.0187929716474710
_SHADOW_DECAY = _AIRY_FIRST_DERIVATIVE_ZERO * math.sin(math.pi / 3.0)
_SHADOW_RESIDUE = 2.0 * math.sqrt(math.pi) / _AIRY_FIRST_DERIVATIVE_ZERO


def _shadow_factor(x: float) -> float:
    """Leading residue term of the Fock series at normalised distance ``x``."""
    return _SHADOW_RESIDUE * math.sqrt(x) * math.exp(-_SHADOW_DECAY * x)


def _solve_shadow_onset() -> float:
    """Where the single residue term first equals the flat-earth field.

    The Fock series is a shadow-region expansion: one term of it is larger
    than unity at small ``x``, which is nonsense, and smaller than unity
    beyond the crossing, which is the shadow.  Taking the crossing as the
    onset makes the curvature factor continuous at 1.0 by construction --
    there is no seam to interpolate across and no cutoff distance to
    choose.  Solved here rather than written down, so it stays correct if
    the residue constant is ever revisited.
    """
    low, high = 0.5, 20.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _shadow_factor(middle) > 1.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


#: Normalised distance at which the flat-earth description hands over to
#: the shadow.  About 1.72, which at 3.65 MHz is 213 km -- comfortably
#: beyond the geometric horizon, as it should be.
SHADOW_ONSET_X = _solve_shadow_onset()

#: Losses beyond this are reported at this value.  A ground wave 300 dB
#: below free space is absent, and the difference between 300 dB and 900 dB
#: is of no interest to anything; capping keeps a 28 MHz continental path
#: from producing an infinity that then propagates into a link budget.
MAX_LOSS_DB = 300.0


def numerical_distance(
    distance_km: float,
    frequency_hz: float,
    ground: GroundType,
    moisture_factor: float = 1.0,
) -> Tuple[float, float]:
    """Sommerfeld's ``(p, b)`` for a vertically polarised wave.

    ``p = k d |Delta|^2 / 2`` is the distance measured in units of the
    surface's own loss scale: it is what decides the attenuation, and it is
    why the same 100 km costs 0.5 dB over sea and 48 dB over average
    ground.  ``b`` is the phase angle of that distance, near zero over a
    good conductor and approaching 90 degrees over a poor one.
    """
    if distance_km < 0.0:
        raise ValueError("distance cannot be negative")
    if frequency_hz <= 0.0:
        raise ValueError("frequency must be positive")
    delta = surface_impedance(frequency_hz, ground, moisture_factor)
    wavenumber = 2.0 * math.pi * frequency_hz / SPEED_OF_LIGHT
    p = 0.5 * wavenumber * distance_km * 1000.0 * abs(delta) ** 2
    # b is the argument of the numerical distance, which for
    # Delta^2 = 1/eps_c is the argument of eps_c taken from the imaginary
    # axis.  The air's own permittivity is included in the contrast, which
    # is the usual refinement and matters only over dry ground.
    permittivity = 1.0 / delta**2
    b = math.atan2(permittivity.real + 1.0, -permittivity.imag)
    return p, b


def _weideman_coefficients(order: int = 32) -> Tuple[float, np.ndarray]:
    """Rational coefficients for the Faddeeva function, by FFT.

    Weideman's construction: sample ``exp(-t^2)(L^2 + t^2)`` on the image
    of the real line under a Cayley map and take its Fourier coefficients.
    They are *computed* here, once, at import -- there is no table of
    magic constants to mistype, and changing ``order`` changes the accuracy
    rather than invalidating a transcription.
    """
    half = 2 * order
    index = np.arange(-half + 1, half)
    scale = math.sqrt(order / math.sqrt(2.0))
    node = scale * np.tan(index * math.pi / (2.0 * half))
    sample = np.concatenate(([0.0], np.exp(-node**2) * (scale**2 + node**2)))
    spectrum = np.real(np.fft.fft(np.fft.fftshift(sample))) / (2 * half)
    return scale, np.flipud(spectrum[1:order + 1])


_WEIDEMAN_SCALE, _WEIDEMAN_COEFFICIENTS = _weideman_coefficients()


def faddeeva(z: complex) -> complex:
    """``w(z) = exp(-z^2) erfc(-i z)``, for ``z`` in the upper half plane.

    Accurate to about one part in ``1e13`` there, which the test suite
    checks against SciPy.  SciPy is a test dependency of this package and
    not a runtime one: the core computes its own physics with numpy, and
    this is the piece that would otherwise have forced an exception.
    """
    denominator = _WEIDEMAN_SCALE - 1j * z
    mapped = (_WEIDEMAN_SCALE + 1j * z) / denominator
    polynomial = complex(np.polyval(_WEIDEMAN_COEFFICIENTS, mapped))
    return 2.0 * polynomial / denominator**2 + (1.0 / math.sqrt(math.pi)) / denominator


def sommerfeld_attenuation(
    distance_km: float,
    frequency_hz: float,
    ground: GroundType,
    moisture_factor: float = 1.0,
) -> complex:
    """The attenuation function ``A(p)`` itself, complex.

    ``A(p) = 1 - j sqrt(pi p) exp(-p) erfc(j sqrt p)`` with ``p`` complex.
    Written through the Faddeeva function using ``w(-z) = 2 exp(-z^2) -
    w(z)``, which collapses the bracket to a single ``w(-sqrt p)`` -- and
    since ``arg p`` lies in ``(-90, 0)`` degrees, ``-sqrt p`` always lands
    in the upper half plane where ``w`` is bounded.  Evaluating the
    bracket as written instead would subtract two enormous nearly equal
    numbers over any lossy ground.
    """
    epsilon = complex_permittivity(frequency_hz, ground, moisture_factor)
    wavenumber = 2.0 * math.pi * frequency_hz / SPEED_OF_LIGHT
    p = -1j * wavenumber * distance_km * 1000.0 / (2.0 * epsilon)
    return 1.0 - 1j * cmath.sqrt(math.pi * p) * faddeeva(-cmath.sqrt(p))


def surface_attenuation_db(
    distance_km: float,
    frequency_hz: float,
    ground: GroundType,
    moisture_factor: float = 1.0,
) -> float:
    """Flat-earth surface loss below the inverse-distance field, dB.

    The Sommerfeld attenuation function, evaluated rather than fitted.
    ``A(0) = 1`` and ``|A| -> 1/2p`` far out, both exactly, and the
    interesting part in between -- where the ground stops behaving like a
    conductor and starts behaving like a dielectric -- is the part the
    rational fits get wrong by several decibels.
    """
    if distance_km <= 0.0:
        return 0.0
    attenuation = abs(
        sommerfeld_attenuation(distance_km, frequency_hz, ground, moisture_factor)
    )
    return min(-20.0 * math.log10(max(attenuation, 1e-15)), MAX_LOSS_DB)


def curvature_loss_db(distance_km: float, frequency_hz: float) -> float:
    """Extra loss from the Earth curving away under the wave, dB.

    Zero inside :data:`SHADOW_ONSET_X` and continuous through it.  The
    normalised distance is ``x = (k a / 2)^(1/3) d / a``, the only
    combination of frequency, distance and radius the diffraction problem
    depends on, so a 3.65 MHz path at 400 km and a 29 MHz path at 200 km
    are the same problem twice.
    """
    if distance_km <= 0.0 or frequency_hz <= 0.0:
        return 0.0
    wavenumber_per_km = 2.0 * math.pi * frequency_hz / SPEED_OF_LIGHT * 1000.0
    x = (
        (0.5 * wavenumber_per_km * EFFECTIVE_EARTH_RADIUS_KM) ** (1.0 / 3.0)
        * distance_km
        / EFFECTIVE_EARTH_RADIUS_KM
    )
    if x <= SHADOW_ONSET_X:
        return 0.0
    return min(-20.0 * math.log10(max(_shadow_factor(x), 1e-15)), MAX_LOSS_DB)


def millington_surface_db(
    sections: Sequence[Tuple[float, GroundType]],
    frequency_hz: float,
    moisture_factor: float = 1.0,
) -> float:
    """Surface loss over a path of several grounds, by Millington's method.

    Each section is charged the *increment* the homogeneous curve for its
    own ground accumulates over that section's span of the total distance,
    then the whole thing is done again with the path reversed and the two
    averaged in dB.

    The reversal is the method.  Run one way, a land-then-sea path shows the
    famous recovery as the wave reaches the water; run the other way it
    does not.  Reciprocity says a passive path cannot care which end
    transmits, and the mean of the two directions is the cheapest estimate
    that respects it.  A path of one ground reduces to the homogeneous
    answer exactly, in either direction, so nothing is paid for the
    machinery when there is no mixture.
    """
    if not sections:
        return 0.0

    def one_way(ordered: Sequence[Tuple[float, GroundType]]) -> float:
        total = 0.0
        travelled = 0.0
        for length_km, ground in ordered:
            if length_km <= 0.0:
                continue
            start = travelled
            travelled += length_km
            total += surface_attenuation_db(
                travelled, frequency_hz, ground, moisture_factor
            ) - surface_attenuation_db(start, frequency_hz, ground, moisture_factor)
        return total

    forward = one_way(sections)
    reverse = one_way(list(reversed(sections)))
    return min(0.5 * (forward + reverse), MAX_LOSS_DB)


@dataclass(frozen=True)
class GroundWaveLoss:
    """Excess loss of the ground wave over free space, with its parts."""

    distance_km: float
    frequency_hz: float
    surface_db: float
    curvature_db: float
    sections: Sequence[Tuple[float, GroundType]] = ()

    def __post_init__(self) -> None:
        if self.surface_db < -1e-9 or self.curvature_db < -1e-9:
            raise ValueError(
                "ground-wave losses are attenuations and cannot be negative: "
                f"surface {self.surface_db:.3f} dB, curvature {self.curvature_db:.3f} dB"
            )

    @property
    def total_db(self) -> float:
        return self.surface_db + self.curvature_db

    @property
    def saturated(self) -> bool:
        """Whether either part hit the cap and stopped being a measurement.

        A loss reported as exactly ``MAX_LOSS_DB`` is the model saying "past
        here I stopped counting", not "the loss is 300 dB".  Anything
        displaying the number needs to be able to tell those apart, so it
        is asked here rather than inferred from a suspicious round figure.
        """
        return (
            self.surface_db >= MAX_LOSS_DB - 1e-9
            or self.curvature_db >= MAX_LOSS_DB - 1e-9
        )

    @property
    def sea_fraction(self) -> float:
        total = sum(length for length, _ in self.sections)
        if total <= 0.0:
            return 0.0
        return sum(
            length for length, ground in self.sections
            if ground is GroundType.SALT_WATER
        ) / total

    def summary(self) -> dict:
        return {
            "total_db": self.total_db,
            "surface_db": self.surface_db,
            "curvature_db": self.curvature_db,
            "saturated": self.saturated,
            "sea_fraction": self.sea_fraction,
            "sections": [
                {"length_km": length, "ground": ground.value}
                for length, ground in self.sections
            ],
        }


def ground_wave_loss_db(
    distance_km: float,
    frequency_hz: float,
    sections: Optional[Sequence[Tuple[float, GroundType]]] = None,
    ground: GroundType = GroundType.AVERAGE_GROUND,
    moisture_factor: float = 1.0,
) -> GroundWaveLoss:
    """Total excess loss of the ground wave, over free space.

    ``sections`` is the mixed path from
    :func:`propsim.surface.path_sections`; without it the whole path is
    taken as one ``ground``.  The curvature term is applied once to the
    whole distance rather than section by section: the Earth bends under
    the path as a whole, and it does not bend more where the path happens
    to cross a coastline.
    """
    if distance_km <= 0.0:
        raise ValueError("distance must be positive")
    if sections is None:
        sections = ((distance_km, ground),)
    surface = millington_surface_db(sections, frequency_hz, moisture_factor)
    curvature = curvature_loss_db(distance_km, frequency_hz)
    return GroundWaveLoss(
        distance_km=distance_km,
        frequency_hz=frequency_hz,
        surface_db=surface,
        curvature_db=curvature,
        sections=tuple(sections),
    )
