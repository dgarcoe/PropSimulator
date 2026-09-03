"""Non-deviative ionospheric absorption.

A radio wave drives the free electrons; each electron-neutral collision
converts a little of that ordered motion into heat, and the wave loses it.
The absorption coefficient for one magnetoionic component is

    kappa = (e^2 / (2 eps0 m_e c)) * Ne * nu / (n * ((omega -/+ omega_L)^2 + nu^2))

which is large where the electron density and the collision frequency are
both appreciable.  That is the D region: above it the collision frequency
collapses, below it there are no electrons.  Absorption falls roughly as
1/f^2, which is why raising the frequency is the standard cure for a signal
lost to daytime absorption.

Two things this module refuses to do:

* It applies **no empirical flare multiplier.**  A flare raises absorption
  because its X-rays raise the D-region electron density, and that density
  arrives here through the ionospheric profile like any other.  Multiplying
  the result by a second "flare factor" would count the same event twice.
* It reports its breakdown by **decomposing** the total it computed, never
  by recomputing the pieces.  The components therefore sum to the total by
  construction, and the telemetry cannot disagree with the link budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .constants import ABSORPTION_COEFF, NEPER_TO_DB
from .ionosphere import (
    _NU_A,
    _NU_B,
    _NU_C,
    EquivalentColumn,
    collision_frequency_hz,
)
from .magnetic import MagneticField
from .raytrace import RayPath
from .refractive import Mode

__all__ = ["AbsorptionResult", "absorption_db", "REGION_BOUNDS"]

#: Height bands used for the reporting breakdown, km.
REGION_BOUNDS = {"D": (50.0, 90.0), "E": (90.0, 150.0), "F": (150.0, 600.0)}

#: Below this refractive index the non-deviative approximation stops being
#: valid: the wave is being bent strongly and spends far longer in the
#: medium than the straight-ray formula assumes.
NON_DEVIATIVE_INDEX_FLOOR = 0.8


@dataclass(frozen=True)
class AbsorptionResult:
    """Absorption for one hop, in dB, with a breakdown that adds up."""

    total_db: float
    non_deviative_db: float
    deviative_db: float
    by_region_db: Dict[str, float]
    #: Fraction of the ray's length excluded from the non-deviative integral
    #: because the refractive index had dropped too far for it to be valid.
    excluded_path_fraction: float
    turning_height_km: float

    def __post_init__(self) -> None:
        parts = sum(self.by_region_db.values())
        if not math.isclose(parts, self.non_deviative_db, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"region breakdown {parts:.6f} dB does not sum to the "
                f"non-deviative total {self.non_deviative_db:.6f} dB"
            )
        expected = self.non_deviative_db + self.deviative_db
        if not math.isclose(expected, self.total_db, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"total {self.total_db:.6f} dB is not the sum of its parts "
                f"({expected:.6f} dB)"
            )

    def summary(self) -> dict:
        return {
            "total_db": self.total_db,
            "non_deviative_db": self.non_deviative_db,
            "deviative_db": self.deviative_db,
            "by_region_db": dict(self.by_region_db),
            "turning_height_km": self.turning_height_km,
            "excluded_path_fraction": self.excluded_path_fraction,
        }


def _local_densities(
    column: EquivalentColumn, heights_km: np.ndarray, fractions: np.ndarray
) -> np.ndarray:
    """Electron density beneath each ray node, from that node's own column.

    Every node is assigned the nearest sampled profile along the great
    circle, so a path crossing the terminator absorbs like a half-lit path.
    Interpolation is always against the profile's ascending height table --
    the node heights themselves may arrive in any order, including
    descending from the apex, and that is fine.
    """
    profile_heights = np.asarray(column.mean_profile.heights_km, dtype=float)
    if not np.all(np.diff(profile_heights) > 0):
        raise ValueError("profile height table must be ascending to interpolate")

    boundaries = np.asarray(column.fractions, dtype=float)
    indices = np.abs(fractions[:, None] - boundaries[None, :]).argmin(axis=1)

    densities = np.zeros_like(heights_km)
    for profile_index in np.unique(indices):
        mask = indices == profile_index
        table = np.asarray(column.profiles[int(profile_index)].densities, dtype=float)
        densities[mask] = np.interp(
            heights_km[mask], profile_heights, table, left=0.0, right=0.0
        )
    return densities


def absorption_db(
    path: RayPath,
    column: EquivalentColumn,
    frequency_hz: float,
    mode: Mode = Mode.ORDINARY,
    magnetic_field: Optional[MagneticField] = None,
    theta_rad: float = math.pi / 2.0,
) -> AbsorptionResult:
    """Absorption suffered by one hop.

    The integral runs over the ray's own quadrature nodes, so the geometry,
    the electron density and the collision frequency are guaranteed to be
    evaluated at the same heights.
    """
    if path.escaped or path.quadrature is None:
        return AbsorptionResult(0.0, 0.0, 0.0, {k: 0.0 for k in REGION_BOUNDS}, 0.0, 0.0)
    if frequency_hz <= 0.0:
        raise ValueError("frequency must be positive")

    quadrature = path.quadrature
    heights = np.asarray(quadrature.height_km, dtype=float)
    ds_km = np.asarray(quadrature.ds_weight_km, dtype=float)
    fractions = np.clip(np.asarray(quadrature.path_fraction, dtype=float), 0.0, 1.0)

    densities = _local_densities(column, heights, fractions)
    # Same fit as ionosphere.collision_frequency_hz, evaluated vectorised.
    collisions = 10.0 ** (_NU_A + _NU_B * heights + _NU_C * heights**2)

    omega = 2.0 * math.pi * frequency_hz
    if magnetic_field is not None:
        gyro = magnetic_field.gyrofrequency_hz
        omega_longitudinal = 2.0 * math.pi * gyro * abs(math.cos(theta_rad))
    else:
        omega_longitudinal = 0.0
    # The extraordinary mode resonates at the gyrofrequency and so is the
    # more heavily absorbed of the two; the ordinary mode is detuned away
    # from it.  This is why an X-mode signal fades first in a flare.
    if mode is Mode.EXTRAORDINARY:
        omega_effective = omega - omega_longitudinal
    else:
        omega_effective = omega + omega_longitudinal

    denominator = omega_effective**2 + collisions**2
    kappa_per_m = ABSORPTION_COEFF * densities * collisions / np.maximum(denominator, 1e-30)

    # Restrict to where the non-deviative approximation holds.
    valid = _non_deviative_mask(path, column, frequency_hz, heights)
    excluded_fraction = float(1.0 - ds_km[valid].sum() / max(ds_km.sum(), 1e-12))

    # ds is in km; kappa is per metre.
    loss_np = kappa_per_m[valid] * ds_km[valid] * 1e3
    contributions_db = NEPER_TO_DB * loss_np

    by_region: Dict[str, float] = {}
    valid_heights = heights[valid]
    assigned = np.zeros(valid_heights.shape, dtype=bool)
    for name, (low, high) in REGION_BOUNDS.items():
        mask = (valid_heights >= low) & (valid_heights < high)
        by_region[name] = float(contributions_db[mask].sum())
        assigned |= mask
    # Anything outside the named bands still belongs in the total; fold it
    # into the nearest region rather than losing it from the breakdown.
    leftover = float(contributions_db[~assigned].sum())
    if leftover:
        by_region["F"] += leftover

    non_deviative = float(sum(by_region.values()))
    deviative = _deviative_penalty_db(path, frequency_hz)

    return AbsorptionResult(
        total_db=non_deviative + deviative,
        non_deviative_db=non_deviative,
        deviative_db=deviative,
        by_region_db=by_region,
        excluded_path_fraction=max(0.0, excluded_fraction),
        turning_height_km=path.apex_height_km,
    )


def _non_deviative_mask(
    path: RayPath, column: EquivalentColumn, frequency_hz: float, heights: np.ndarray
) -> np.ndarray:
    """Nodes where the straight-ray absorption formula is applicable."""
    from .refractive import refractive_index_squared

    profile_heights = np.asarray(column.mean_profile.heights_km, dtype=float)
    densities = np.asarray(column.mean_profile.densities, dtype=float)
    node_density = np.interp(heights, profile_heights, densities, left=0.0, right=0.0)
    n_squared = np.array(
        [refractive_index_squared(float(ne), frequency_hz) for ne in node_density]
    )
    return n_squared >= NON_DEVIATIVE_INDEX_FLOOR**2


def _deviative_penalty_db(path: RayPath, frequency_hz: float) -> float:
    """Extra loss from the strongly refracting region around the turning point.

    Inside the turning region the ray is bent hard, its group velocity is
    low and it lingers; the non-deviative integral above deliberately
    excludes that region, so the loss there is added here instead.  The form
    is empirical -- it is calibrated to the observation that a ray turning
    inside the E layer is far more heavily absorbed than one turning in F2 --
    and it is charged **once**, inside the absorption total.  Nothing
    downstream may add a second low-turning penalty when comparing launch
    angles: doing so would rank angles against a loss that the link budget
    never charges.
    """
    turning = path.apex_height_km
    if turning >= 150.0:
        return 0.0
    frequency_mhz = frequency_hz / 1e6
    depth = (150.0 - turning) / 100.0          # 0 at 150 km, 1.0 at 50 km
    return 6.0 * depth**2 * (10.0 / max(frequency_mhz, 1.0)) ** 1.2
