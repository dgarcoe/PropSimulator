"""Ray tracing through a spherically stratified ionosphere.

The ray obeys Bouguer's invariant, the spherical form of Snell's law and the
stationary-time statement of Fermat's principle in a radially varying medium:

    n(r) * r * cos(beta(r)) = P = const

where ``beta`` is the elevation of the ray above the local horizontal.  The
ray turns where ``n(r) r = P``.  Writing ``g(r) = (n(r) r)^2 - P^2``:

    d(theta)/dr = P / (r sqrt(g)),        geocentric angle
    ds/dr       = n r / sqrt(g),          geometric path length
    d(s_group)/dr = r / sqrt(g),          group path (n_group = 1/n)

All three integrands have an inverse-square-root singularity at the turning
point.  The substitution ``r = r_apex - w^2`` removes it exactly: ``g`` goes
to zero like ``w^2``, so ``sqrt(g)`` goes like ``w`` and cancels against the
``2w dw`` from the change of variable, leaving a bounded integrand.  Note
the sign: ``dr = -2w dw`` and the limits reverse with it, so the two minus
signs cancel and the accumulated length is a sum of positive contributions.
A path length that comes out shorter than the straight-line chord, or a hop
that needs an artificial upper clamp, means that cancellation was done
wrong; :func:`RayPath.check_consistency` asserts it on every traced ray.

What this module deliberately does **not** claim: there is no horizontal
refraction, no off-great-circle propagation and no ducting.  The medium is
the path-averaged equivalent column, and the geometry is radial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .constants import EARTH_RADIUS_KM, SPEED_OF_LIGHT
from .ionosphere import IonosphericProfile, MAX_HEIGHT_KM
from .magnetic import MagneticField
from .refractive import Mode, refractive_index_squared_array

__all__ = ["RayMedium", "RayPath", "RayQuadrature", "trace_ray",
           "hop_ground_range_km", "scan_ranges", "solve_launch_angles",
           "skip_distance_km", "RayError"]


class RayError(RuntimeError):
    """Raised when a ray cannot be traced at all."""


#: Gauss-Legendre nodes are interior to each panel, so the integrator never
#: evaluates the (removable) singularity at the turning point itself.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)


@dataclass
class RayMedium:
    """Refractive index against height for one frequency and one mode.

    The index table is built once on the profile's ascending height grid.
    Every lookup goes through :meth:`index_at`, which interpolates against
    that ascending table; the *query* points may be in any order, including
    the descending radii a tracer produces on the way down from an apex.
    """

    profile: IonosphericProfile
    frequency_hz: float
    mode: Mode = Mode.ORDINARY
    magnetic_field: Optional[MagneticField] = None
    theta_rad: float = math.pi / 2.0

    heights_km: np.ndarray = field(init=False)
    n_squared: np.ndarray = field(init=False)
    #: Cached turning-point candidates; see :meth:`apex_candidates`.
    _candidates: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _candidate_shared: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency must be positive")
        heights = np.asarray(self.profile.heights_km, dtype=float)
        if not np.all(np.diff(heights) > 0):
            raise ValueError("profile heights must be strictly ascending")
        self.heights_km = heights
        b = self.magnetic_field.magnitude if self.magnetic_field is not None else 0.0
        densities = np.asarray(self.profile.densities, dtype=float)
        self.n_squared = np.asarray(
            refractive_index_squared_array(
                densities, self.frequency_hz, b, self.theta_rad, self.mode
            ),
            dtype=float,
        )
        # An evanescent sample is a hard stop for the ray, not a number to
        # interpolate through; clip at zero so sqrt() downstream is defined.
        self.n_squared = np.clip(self.n_squared, 0.0, None)

    @property
    def top_radius_km(self) -> float:
        return EARTH_RADIUS_KM + float(self.heights_km[-1])

    def apex_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Turning-point candidate radii and ``n^2 r^2`` sampled at them.

        Depends only on the medium, not on the ray, so it is built once and
        shared by every apex search against this medium -- an elevation scan
        makes thousands of them.
        """
        if self._candidates is None:
            candidates = _turning_candidates(
                self, EARTH_RADIUS_KM + float(self.heights_km[0]), self.top_radius_km
            )
            object.__setattr__(self, "_candidates", candidates)
            object.__setattr__(
                self,
                "_candidate_shared",
                self.n_squared_at_radius(candidates) * candidates**2,
            )
        return self._candidates, self._candidate_shared

    def n_squared_at_radius(self, radius_km: np.ndarray | float) -> np.ndarray:
        """n^2 at one or many radii.  Below the grid the medium is vacuum."""
        heights = np.asarray(radius_km, dtype=float) - EARTH_RADIUS_KM
        return np.interp(
            heights, self.heights_km, self.n_squared, left=1.0, right=1.0
        )

    def index_at(self, radius_km: np.ndarray | float) -> np.ndarray:
        return np.sqrt(np.clip(self.n_squared_at_radius(radius_km), 0.0, None))

    def g_at(self, radius_km: np.ndarray | float, bouguer: float) -> np.ndarray:
        """``g(r) = (n r)^2 - P^2``; the ray turns where this reaches zero."""
        radius = np.asarray(radius_km, dtype=float)
        return self.n_squared_at_radius(radius) * radius**2 - bouguer**2


@dataclass(frozen=True)
class RayQuadrature:
    """The ray's own integration nodes, exposed for path integrals.

    Anything that must be accumulated *along* the ray -- absorption above
    all -- integrates over these nodes rather than re-deriving a geometry of
    its own.  There is therefore only one description of where the ray is,
    and no opportunity for the geometry, the electron density and the
    collision frequency to end up evaluated at three different heights.

    Attributes
    ----------
    height_km:
        Height above the surface at each node.  Derived once as
        ``radius - EARTH_RADIUS_KM``; the transmitter height is already in
        ``radius`` and is never added again.
    ds_weight_km:
        Quadrature weight for ``integral f ds`` -- geometric path length.
    path_fraction:
        Position along the hop, 0 at the transmitter to 1 at the landing
        point, used to select the local ionosphere beneath each node.
    """

    height_km: np.ndarray
    ds_weight_km: np.ndarray
    path_fraction: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.height_km)
        if len(self.ds_weight_km) != n or len(self.path_fraction) != n:
            raise ValueError("quadrature arrays must have equal length")


@dataclass(frozen=True)
class RayPath:
    """One hop, traced from the transmitter to where it comes back down."""

    launch_elevation_deg: float
    frequency_hz: float
    mode: Mode
    escaped: bool
    start_height_km: float
    apex_height_km: float = 0.0
    #: Ground range of a single hop, measured along the Earth's surface.
    ground_range_km: float = 0.0
    #: Geometric length of the ray, both legs.  Always >= the chord.
    geometric_path_km: float = 0.0
    #: Group path, the quantity that actually sets the propagation delay.
    group_path_km: float = 0.0
    #: Sampled (height, fraction-of-hop) pairs covering the whole hop.
    samples: Sequence[Tuple[float, float]] = ()
    #: The integration nodes themselves, for path integrals along the ray.
    quadrature: Optional["RayQuadrature"] = None

    @property
    def group_delay_ms(self) -> float:
        """True group delay for one hop, milliseconds.

        Uses the group path, not the geometric one: in the ionosphere the
        group index is 1/n > 1, so the signal is retarded relative to a
        vacuum ray of the same shape.
        """
        return self.group_path_km * 1e3 / SPEED_OF_LIGHT * 1e3

    @property
    def geometric_delay_ms(self) -> float:
        return self.geometric_path_km * 1e3 / SPEED_OF_LIGHT * 1e3

    @property
    def virtual_height_km(self) -> float:
        """Height of the sharp mirror that would give this group path.

        Solved on the sphere, not on a flat Earth.  With half-angle
        ``T = D / 2Re`` and half group path ``L``, the law of cosines gives

            (Re + h')^2 - 2 Re cos(T) (Re + h') + Re^2 - L^2 = 0

        whose positive root is ``Re cos(T) + sqrt(L^2 - Re^2 sin^2 T)``.
        The flat-Earth form ``sqrt(L^2 - (D/2)^2)`` overestimates h' badly at
        oblique incidence -- by 20% for a 2000 km hop -- because it ignores
        the curvature the ray actually follows.
        """
        if self.ground_range_km <= 0.0:
            return 0.0
        half_angle = self.ground_range_km / (2.0 * EARTH_RADIUS_KM)
        half_path = self.group_path_km / 2.0
        discriminant = half_path**2 - (EARTH_RADIUS_KM * math.sin(half_angle)) ** 2
        if discriminant <= 0.0:
            return 0.0
        radius = EARTH_RADIUS_KM * math.cos(half_angle) + math.sqrt(discriminant)
        return max(0.0, radius - EARTH_RADIUS_KM)

    def check_consistency(self) -> None:
        """Assert the invariants a correctly integrated ray must satisfy."""
        if self.escaped:
            return
        if self.ground_range_km <= 0.0:
            raise RayError("a returning ray must cover a positive ground range")
        # Chord subtended by the hop on the Earth's surface.
        half_angle = self.ground_range_km / (2.0 * EARTH_RADIUS_KM)
        chord = 2.0 * EARTH_RADIUS_KM * math.sin(half_angle)
        if self.geometric_path_km < chord * (1.0 - 1e-6):
            raise RayError(
                f"geometric path {self.geometric_path_km:.1f} km is shorter than "
                f"the {chord:.1f} km chord it spans -- the path-length integral "
                "has lost a sign"
            )
        if self.group_path_km < self.geometric_path_km * (1.0 - 1e-6):
            raise RayError(
                f"group path {self.group_path_km:.1f} km is shorter than the "
                f"geometric path {self.geometric_path_km:.1f} km, but the group "
                "index in a plasma is always >= 1"
            )
        # A hop cannot exceed the half-circumference of the Earth.
        if self.ground_range_km > math.pi * EARTH_RADIUS_KM:
            raise RayError(f"hop range {self.ground_range_km:.0f} km is unphysical")


def _turning_candidates(medium: RayMedium, low_km: float, high_km: float) -> np.ndarray:
    """Radii at which ``g`` must be sampled to find every crossing.

    Between two nodes of the profile grid ``n^2`` is linear in height, so
    ``g(r) = n^2(r) r^2 - P^2`` is a cubic in ``r``.  A cubic on an interval
    attains its minimum either at an endpoint or at an interior stationary
    point, and the stationary points solve

        g'(r) = r * (3 b r + 2 (a - b r_k)) = 0

    for the segment's ``n^2 = a + b (r - r_k)``.  Sampling the grid nodes
    **and** those stationary points is therefore exhaustive: no crossing can
    hide between samples.

    That matters because the dangerous case is not a wide dip but a razor
    one.  A ray launched at exactly the E-layer maximum usable frequency
    grazes the layer peak, and ``g`` dips below zero over a span of a few
    tens of metres in radius.  Any uniform scan coarse enough to be
    affordable will step straight over it, conclude the ray carried on to
    the F region, and then integrate a path that passes through a region the
    wave cannot enter.
    """
    radii = EARTH_RADIUS_KM + medium.heights_km
    n_squared = medium.n_squared

    mask = (radii >= low_km) & (radii <= high_km)
    candidates = [radii[mask], np.array([low_km, high_km])]

    dr = np.diff(radii)
    slope = np.diff(n_squared) / dr
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary = -2.0 * (n_squared[:-1] - slope * radii[:-1]) / (3.0 * slope)
    valid = (
        np.isfinite(stationary)
        & (stationary > radii[:-1])
        & (stationary < radii[1:])
        & (stationary >= low_km)
        & (stationary <= high_km)
    )
    candidates.append(stationary[valid])

    return np.unique(np.concatenate(candidates))


def _batch_apex_radii(
    medium: RayMedium, bouguer: np.ndarray, start_radius_km: float
) -> np.ndarray:
    """Turning radius for many rays at once; NaN where the ray escapes.

    Every ray in the batch sees the same medium and differs only in its
    Bouguer constant, so the expensive part -- the candidate radii and the
    refractive index sampled at them -- is computed once and shared.  The
    bisection then runs as a vector operation over the whole batch.

    The candidate set is the exhaustive one from
    :func:`_turning_candidates`, so a razor tangency at a layer peak is
    caught here exactly as it was by the scalar search this replaces.
    :func:`trace_ray` routes through this function too -- there is one apex
    search in the package, not a fast one and a careful one that could
    disagree about where a ray turns.
    """
    all_candidates, all_shared = medium.apex_candidates()
    keep = all_candidates >= start_radius_km
    candidates, shared = all_candidates[keep], all_shared[keep]
    if candidates.size < 2:
        return np.full(bouguer.shape, np.nan)

    # (rays, candidates)
    g = shared[None, :] - (bouguer**2)[:, None]
    below = g <= 0.0

    apex = np.full(bouguer.shape, np.nan)
    any_below = below.any(axis=1)
    first = np.argmax(below, axis=1)          # first True per row

    # A ray already evanescent at the start turns immediately.
    start_g = float(medium.n_squared_at_radius(start_radius_km)) * start_radius_km**2
    immediate = (start_g - bouguer**2) <= 0.0
    apex[immediate] = start_radius_km

    active = any_below & ~immediate & (first > 0)
    if not active.any():
        return apex

    low = candidates[first[active] - 1].astype(float)
    high = candidates[first[active]].astype(float)
    p_active = bouguer[active]
    # A bracket is at most one profile-grid segment wide, so ~35 halvings
    # reach the 1e-9 km tolerance; the loop leaves as soon as it does rather
    # than grinding out a fixed count for precision nothing can use.
    for _ in range(50):
        if np.all(high - low < 1e-9):
            break
        mid = 0.5 * (low + high)
        g_mid = medium.n_squared_at_radius(mid) * mid**2 - p_active**2
        positive = g_mid > 0.0
        low = np.where(positive, mid, low)
        high = np.where(positive, high, mid)
    apex[active] = low

    # Rows whose first crossing is candidate 0 turn at the very start.
    edge = any_below & ~immediate & (first == 0)
    apex[edge] = start_radius_km
    return apex


def _batch_hop_geometry(
    medium: RayMedium,
    elevations_deg: np.ndarray,
    start_height_km: float,
    panels: int = 48,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground range, geometric path and apex height for many launch angles.

    Returns three arrays, NaN wherever the ray escapes.  Used for the
    elevation scan, where only the range is wanted and building a full
    quadrature for each candidate would be wasted work.
    """
    start_radius = EARTH_RADIUS_KM + start_height_km
    n_start = float(medium.index_at(start_radius))
    beta = np.radians(np.asarray(elevations_deg, dtype=float))
    bouguer = n_start * start_radius * np.cos(beta)

    apex = _batch_apex_radii(medium, bouguer, start_radius)
    ranges = np.full(apex.shape, np.nan)
    paths = np.full(apex.shape, np.nan)
    heights = apex - EARTH_RADIUS_KM

    usable = np.isfinite(apex) & (apex > start_radius + 1e-9)
    if not usable.any():
        return ranges, paths, heights

    apex_u = apex[usable]
    p_u = bouguer[usable]
    w_max = np.sqrt(apex_u - start_radius)                    # (rays,)

    unit = np.linspace(0.0, 1.0, panels + 1)
    edges = w_max[:, None] * unit[None, :]                    # (rays, panels+1)
    half = 0.5 * np.diff(edges, axis=1)
    centre = 0.5 * (edges[:, :-1] + edges[:, 1:])

    # (rays, panels, nodes)
    w = centre[:, :, None] + half[:, :, None] * _GL_NODES[None, None, :]
    weights = _GL_WEIGHTS[None, None, :] * half[:, :, None]
    radius = apex_u[:, None, None] - w**2
    jacobian = 2.0 * w

    n2 = medium.n_squared_at_radius(radius)
    g = n2 * radius**2 - (p_u**2)[:, None, None]
    # A batch ray that lands on a non-positive integrand is dropped rather
    # than integrated: the same condition that raises in the single-ray
    # path, expressed as an escape here so one bad angle cannot poison a
    # whole scan.
    healthy = (g > 0.0).all(axis=(1, 2))
    g = np.where(g > 0.0, g, 1.0)
    sqrt_g = np.sqrt(g)
    common = weights * jacobian / sqrt_g

    angle = np.sum(common * p_u[:, None, None] / radius, axis=(1, 2))
    length = np.sum(common * np.sqrt(np.clip(n2, 0.0, None)) * radius, axis=(1, 2))

    angle = np.where(healthy, angle, np.nan)
    length = np.where(healthy, length, np.nan)

    ranges[usable] = 2.0 * EARTH_RADIUS_KM * angle
    paths[usable] = 2.0 * length
    heights[~np.isfinite(apex)] = np.nan
    return ranges, paths, heights


def _integrate_leg(
    medium: RayMedium,
    bouguer: float,
    start_radius_km: float,
    apex_radius_km: float,
    panels: int = 48,
) -> Tuple[float, float, float, List[Tuple[float, float]], dict]:
    """Integrate one leg (start -> apex) after removing the singularity.

    Returns ``(geocentric_angle, geometric_length, group_length, samples)``
    where samples are ``(radius, cumulative_geocentric_angle)`` pairs taken
    from the start radius upward.

    The whole panel set is evaluated in one vectorised pass; the per-panel
    running total needed for the samples comes from a cumulative sum rather
    than from a Python loop.
    """
    span = apex_radius_km - start_radius_km
    if span <= 0.0:
        return 0.0, 0.0, 0.0, [], {
            "radius": np.empty(0),
            "ds_weight": np.empty(0),
            "angle_to_node": np.empty(0),
        }

    w_max = math.sqrt(span)
    edges = np.linspace(0.0, w_max, panels + 1)
    half = 0.5 * np.diff(edges)                      # (panels,)
    centre = 0.5 * (edges[:-1] + edges[1:])          # (panels,)

    # (panels, nodes) grid of the substituted variable.
    w = centre[:, None] + half[:, None] * _GL_NODES[None, :]
    weights = _GL_WEIGHTS[None, :] * half[:, None]

    radius = apex_radius_km - w**2
    # dr = -2w dw, and the integration limits reverse with the substitution.
    # The two sign changes cancel, so every contribution below is added.
    jacobian = 2.0 * w

    g = medium.g_at(radius, bouguer)
    if np.any(g <= 0.0):
        # The substitution guarantees g > 0 at every interior node when the
        # apex is the true first turning point.  Reaching here means it is
        # not, and integrating anyway would emit a path length inflated by
        # many orders of magnitude instead of an error.
        raise RayError(
            "ray integrand became singular away from the turning point; "
            "the apex search returned a radius past an earlier crossing"
        )
    sqrt_g = np.sqrt(g)
    index = np.sqrt(np.clip(medium.n_squared_at_radius(radius), 0.0, None))

    common = weights * jacobian / sqrt_g
    angle_panels = np.sum(common * bouguer / radius, axis=1)
    length_panels = np.sum(common * index * radius, axis=1)
    group_panels = np.sum(common * radius, axis=1)

    angle_total = float(angle_panels.sum())
    length_total = float(length_panels.sum())
    group_total = float(group_panels.sum())

    # Panels run apex-first in w; cumulative angle measured from the apex.
    cumulative = np.cumsum(angle_panels)
    mean_radius = radius.mean(axis=1)
    samples = [
        (float(r), angle_total - float(c))
        for r, c in zip(mean_radius[::-1], cumulative[::-1])
    ]

    # Per-node data.  The angle accumulated from the transmitter up to each
    # node is the total minus the angle still to run to the apex; that
    # running total is built by exclusive cumulative sum over the panels
    # (which are ordered apex-first) plus the within-panel contribution.
    ds_nodes = (common * index * radius)                      # (panels, nodes)
    angle_nodes = common * bouguer / radius
    exclusive = np.concatenate(([0.0], cumulative[:-1]))       # angle above panel
    angle_to_node = angle_total - (
        exclusive[:, None] + np.cumsum(angle_nodes[:, ::-1], axis=1)[:, ::-1]
    )
    node_data = {
        "radius": radius.ravel(),
        "ds_weight": ds_nodes.ravel(),
        "angle_to_node": angle_to_node.ravel(),
    }
    return angle_total, length_total, group_total, samples, node_data


def trace_ray(
    medium: RayMedium,
    launch_elevation_deg: float,
    start_height_km: float = 0.0,
    panels: int = 48,
) -> RayPath:
    """Trace one hop from the ground up to the apex and back down.

    ``start_height_km`` is the height of the transmitting antenna above the
    surface and is applied exactly once, to form the starting radius.  Every
    later height in this function is derived as ``radius - EARTH_RADIUS_KM``,
    so it can never be added a second time.
    """
    if not 0.0 < launch_elevation_deg < 90.0:
        raise ValueError(f"launch elevation {launch_elevation_deg} outside (0, 90)")
    if start_height_km < 0.0:
        raise ValueError("start height cannot be negative")

    start_radius = EARTH_RADIUS_KM + start_height_km
    n_start = float(medium.index_at(start_radius))
    beta = math.radians(launch_elevation_deg)
    bouguer = n_start * start_radius * math.cos(beta)

    apex_array = _batch_apex_radii(medium, np.array([bouguer]), start_radius)
    apex_radius = None if not np.isfinite(apex_array[0]) else float(apex_array[0])
    if apex_radius is None:
        return RayPath(
            launch_elevation_deg=launch_elevation_deg,
            frequency_hz=medium.frequency_hz,
            mode=medium.mode,
            escaped=True,
            start_height_km=start_height_km,
        )

    angle, length, group, leg_samples, nodes = _integrate_leg(
        medium, bouguer, start_radius, apex_radius, panels
    )

    # The medium is the same on the way down, so the descending leg mirrors
    # the ascending one.  Total geocentric angle is 2 * angle.
    total_angle = 2.0 * angle
    ground_range = EARTH_RADIUS_KM * total_angle

    samples: List[Tuple[float, float]] = []
    if total_angle > 0.0:
        for radius, partial_angle in leg_samples:
            samples.append((radius - EARTH_RADIUS_KM, partial_angle / total_angle))
        for radius, partial_angle in reversed(leg_samples):
            samples.append(
                (radius - EARTH_RADIUS_KM, 1.0 - partial_angle / total_angle)
            )

    quadrature = None
    if total_angle > 0.0 and nodes["radius"].size:
        up_fraction = nodes["angle_to_node"] / total_angle
        # The descending leg mirrors the ascending one about the apex.
        heights = nodes["radius"] - EARTH_RADIUS_KM
        quadrature = RayQuadrature(
            height_km=np.concatenate([heights, heights]),
            ds_weight_km=np.concatenate([nodes["ds_weight"], nodes["ds_weight"]]),
            path_fraction=np.concatenate([up_fraction, 1.0 - up_fraction]),
        )

    path = RayPath(
        launch_elevation_deg=launch_elevation_deg,
        frequency_hz=medium.frequency_hz,
        mode=medium.mode,
        escaped=False,
        start_height_km=start_height_km,
        apex_height_km=apex_radius - EARTH_RADIUS_KM,
        ground_range_km=ground_range,
        geometric_path_km=2.0 * length,
        group_path_km=2.0 * group,
        samples=tuple(samples),
        quadrature=quadrature,
    )
    path.check_consistency()
    return path


def hop_ground_range_km(
    medium: RayMedium, launch_elevation_deg: float, start_height_km: float = 0.0
) -> Optional[float]:
    """Ground range of one hop, or None if the ray escapes."""
    path = trace_ray(medium, launch_elevation_deg, start_height_km)
    return None if path.escaped else path.ground_range_km


def scan_ranges(
    medium: RayMedium,
    start_height_km: float = 0.0,
    min_elevation_deg: float = 1.0,
    max_elevation_deg: float = 89.0,
    scan_points: int = 89,
) -> Tuple[np.ndarray, List[Optional[float]]]:
    """Hop range against launch elevation; ``None`` where the ray escapes.

    Computed once per frequency and reused for every hop count, since the
    curve does not depend on how many hops the circuit is being tried with.
    """
    elevations = np.linspace(min_elevation_deg, max_elevation_deg, scan_points)
    values, _, _ = _batch_hop_geometry(medium, elevations, start_height_km)
    ranges = [None if not np.isfinite(v) else float(v) for v in values]
    return elevations, ranges


def solve_launch_angles(
    medium: RayMedium,
    target_range_km: float,
    start_height_km: float = 0.0,
    min_elevation_deg: float = 1.0,
    max_elevation_deg: float = 89.0,
    scan_points: int = 89,
    tolerance_km: float = 1.0,
    scan: Optional[Tuple[np.ndarray, List[Optional[float]]]] = None,
) -> List[float]:
    """Launch elevations whose hop lands exactly at ``target_range_km``.

    This is the only sanctioned way to ask "does the signal reach a receiver
    at this distance".  The question is answered by *solving* for the launch
    angle whose hop terminates at the receiver, never by taking a hop that
    lands somewhere else and rescaling it to the wanted distance.  A ray that
    is still hundreds of kilometres up when it passes over the receiver has
    a hop range larger than the target and simply does not appear in the
    returned list -- which is what puts the receiver in the skip zone.

    Returns every solution found, low ray first.  An empty list means the
    target is unreachable in one hop at this frequency and mode.
    """
    if target_range_km <= 0.0:
        raise ValueError("target range must be positive")

    if scan is None:
        scan = scan_ranges(
            medium, start_height_km, min_elevation_deg, max_elevation_deg, scan_points
        )
    elevations, ranges = scan

    # Collect every bracket where the traced range crosses the target.
    lows: List[float] = []
    highs: List[float] = []
    low_values: List[float] = []
    for i in range(len(elevations) - 1):
        r_low, r_high = ranges[i], ranges[i + 1]
        if r_low is None or r_high is None:
            continue
        if (r_low - target_range_km) * (r_high - target_range_km) > 0.0:
            continue
        lows.append(float(elevations[i]))
        highs.append(float(elevations[i + 1]))
        low_values.append(float(r_low))

    if not lows:
        return []

    # Bisect every bracket at once: one batch trace per iteration instead of
    # one per bracket per iteration.
    low = np.array(lows)
    high = np.array(highs)
    value_at_low = np.array(low_values)
    for _ in range(48):
        mid = 0.5 * (low + high)
        values, _, _ = _batch_hop_geometry(medium, mid, start_height_km)
        finite = np.isfinite(values)
        # An escape inside a bracket collapses it; the candidate is rejected
        # by the verification below rather than silently accepted.
        crosses = finite & (
            (value_at_low - target_range_km) * (values - target_range_km) <= 0.0
        )
        high = np.where(crosses, mid, high)
        low = np.where(crosses | ~finite, low, mid)
        value_at_low = np.where(crosses | ~finite, value_at_low, values)
        if np.all(high - low < 1e-7):
            break

    candidates = 0.5 * (low + high)
    achieved, _, _ = _batch_hop_geometry(medium, candidates, start_height_km)

    solutions: List[float] = []
    for elevation, landed in zip(candidates, achieved):
        # Accepted only because the ray genuinely lands there.
        if not np.isfinite(landed) or abs(landed - target_range_km) > tolerance_km:
            continue
        value = float(elevation)
        if not any(abs(value - existing) < 1e-3 for existing in solutions):
            solutions.append(value)

    return sorted(solutions)


def skip_distance_km(
    medium: RayMedium,
    start_height_km: float = 0.0,
    min_elevation_deg: float = 1.0,
    max_elevation_deg: float = 89.0,
    scan_points: int = 89,
    scan: Optional[Tuple[np.ndarray, List[Optional[float]]]] = None,
) -> Optional[float]:
    """Shortest one-hop ground range achievable at this frequency.

    Receivers closer than this are inside the skip zone: the ray passes over
    them at altitude.  ``None`` means no ray returns at all.
    """
    if scan is None:
        scan = scan_ranges(
            medium, start_height_km, min_elevation_deg, max_elevation_deg, scan_points
        )
    valid = [r for r in scan[1] if r is not None]
    return min(valid) if valid else None
