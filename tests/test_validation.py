"""Validation against independent references, and equivalence of fast paths.

Two kinds of test live here.

*Cross-validation* compares the physics core with :mod:`propsim.reference`,
which contains externally-derived expressions the core does not use. That is
the only way to catch a model that is internally consistent and wrong.

*Equivalence* pins every optimised path to the plain one it replaced. A
vectorised twin that quietly disagrees with its scalar original is worse
than no optimisation, so each is asserted to match to machine precision.
"""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from propsim.absorption import absorption_db
from propsim.geodesy import GeoPoint
from propsim.ionosphere import build_equivalent_column, build_profile
from propsim.magnetic import magnetic_field
from propsim.raytrace import (
    RayMedium,
    _batch_hop_geometry,
    scan_ranges,
    trace_ray,
)
from propsim.reference import (
    absorption_index_db,
    chapman_foe_mhz,
    incidence_angle_at_110km,
    measure_scaling,
)
from propsim.refractive import (
    Mode,
    refractive_index_squared,
    refractive_index_squared_array,
)
from propsim.solar import solar_zenith_angle_deg
from propsim.spaceweather import SpaceWeather

UTC = timezone.utc
NOON = datetime(2025, 6, 21, 12, tzinfo=UTC)
MADRID = GeoPoint(40.4, -3.7)
LONDON = GeoPoint(51.5, -0.1)


@pytest.fixture(scope="module")
def column():
    return build_equivalent_column(
        MADRID, LONDON, NOON, SpaceWeather(f107=140, sunspot_number=60, kp=2), 0.9
    )


class TestVectorisedEquivalence:
    def test_vectorised_index_matches_the_scalar_form(self):
        """Bit-for-bit, over the whole parameter space that matters."""
        densities = np.concatenate([[0.0], np.logspace(8, 13, 40)])
        for frequency in (2e6, 7e6, 14e6, 28e6):
            for field in (0.0, 2e-5, 4.7e-5, 6e-5):
                for theta in (1e-3, 0.5, 1.0, math.pi / 2, 2.0, 3.14):
                    for mode in Mode:
                        vector = np.asarray(
                            refractive_index_squared_array(
                                densities, frequency, field, theta, mode
                            ),
                            dtype=float,
                        )
                        scalar = np.array([
                            refractive_index_squared(
                                float(density), frequency, field, theta, mode
                            )
                            for density in densities
                        ])
                        assert (np.isfinite(vector) == np.isfinite(scalar)).all()
                        finite = np.isfinite(vector)
                        assert np.array_equal(vector[finite], scalar[finite])

    def test_batch_tracer_matches_the_single_ray_tracer(self, column):
        """The scan and the full trace must agree about where a ray lands."""
        for frequency_mhz in (5, 9, 14, 20):
            medium = RayMedium(column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY)
            elevations = np.linspace(1.0, 60.0, 119)
            ranges, paths, apexes = _batch_hop_geometry(medium, elevations, 0.015)
            for index, elevation in enumerate(elevations):
                single = trace_ray(medium, float(elevation), 0.015)
                if single.escaped:
                    assert not np.isfinite(ranges[index])
                    continue
                assert np.isfinite(ranges[index])
                assert ranges[index] == pytest.approx(single.ground_range_km, abs=1e-9)
                assert paths[index] == pytest.approx(single.geometric_path_km, abs=1e-9)
                assert apexes[index] == pytest.approx(single.apex_height_km, abs=1e-9)

    def test_scan_ranges_agrees_with_individual_traces(self, column):
        medium = RayMedium(column.mean_profile, 14e6, Mode.ORDINARY)
        elevations, ranges = scan_ranges(medium, 0.015, 1.0, 60.0, 60)
        for elevation, value in zip(elevations, ranges):
            single = trace_ray(medium, float(elevation), 0.015)
            if single.escaped:
                assert value is None
            else:
                assert value == pytest.approx(single.ground_range_km, abs=1e-9)

    def test_cached_candidates_do_not_change_the_answer(self, column):
        """The apex-candidate cache must be a pure memoisation."""
        medium = RayMedium(column.mean_profile, 14e6, Mode.ORDINARY)
        assert medium._candidates is None                    # cold
        cold = trace_ray(medium, 20.0, 0.015).ground_range_km
        assert medium._candidates is not None                # now warm
        warm = trace_ray(medium, 20.0, 0.015).ground_range_km
        assert warm == cold

        # A medium built fresh must agree with a warmed one, and the cached
        # arrays must equal what an uncached build would have produced.
        fresh = RayMedium(column.mean_profile, 14e6, Mode.ORDINARY)
        assert trace_ray(fresh, 20.0, 0.015).ground_range_km == cold
        cached_candidates, cached_shared = medium.apex_candidates()
        recomputed_candidates, recomputed_shared = fresh.apex_candidates()
        assert np.array_equal(cached_candidates, recomputed_candidates)
        assert np.array_equal(cached_shared, recomputed_shared)


class TestEmpiricalELayer:
    """The E layer follows the classical relation, not one of our own."""

    @pytest.mark.parametrize("hour", [8, 10, 12, 14, 16])
    def test_foe_matches_the_classical_relation_by_day(self, hour):
        when = datetime(2025, 6, 21, hour, tzinfo=UTC)
        zenith = solar_zenith_angle_deg(MADRID, when)
        profile = build_profile(
            MADRID, when, SpaceWeather(f107=140, sunspot_number=60), 0.9
        )
        assert profile.layers.e.critical_frequency_mhz == pytest.approx(
            chapman_foe_mhz(zenith, 60), rel=1e-9
        )

    @pytest.mark.parametrize("hour", [0, 2, 22])
    def test_night_uses_the_residue_not_the_daytime_relation(self, hour):
        """The daytime relation clamped at the horizon would leave the
        night-time E layer nearly three times too dense."""
        when = datetime(2025, 6, 21, hour, tzinfo=UTC)
        assert solar_zenith_angle_deg(MADRID, when) > 90.0
        profile = build_profile(
            MADRID, when, SpaceWeather(f107=140, sunspot_number=60), 0.9
        )
        assert profile.layers.e.critical_frequency_mhz < 0.8

    def test_foe_rises_with_solar_activity(self):
        low = build_profile(MADRID, NOON, SpaceWeather(sunspot_number=5), 0.9)
        high = build_profile(MADRID, NOON, SpaceWeather(sunspot_number=200), 0.9)
        assert (
            high.layers.e.critical_frequency_mhz > low.layers.e.critical_frequency_mhz
        )


class TestAbsorptionCrossValidation:
    """The core is compared with an absorption model it does not use."""

    def _core(self, column, frequency_mhz, elevation_deg):
        field = magnetic_field(GeoPoint(45, -2), 250.0)
        medium = RayMedium(column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY)
        path = trace_ray(medium, elevation_deg, 0.015)
        # The reference describes a ray that *crosses* the absorbing layer.
        # A ray turning below 110 km is a different regime and comparing the
        # two there would be comparing two different physical situations.
        if path.escaped or path.apex_height_km < 150.0:
            return None
        return absorption_db(
            path, column, frequency_mhz * 1e6, Mode.ORDINARY, field, 1.0
        ).total_db

    @pytest.mark.parametrize("frequency_mhz", [12, 14, 16, 18, 20])
    def test_f_mode_absorption_is_within_a_quarter_of_the_reference(
        self, column, frequency_mhz
    ):
        core = self._core(column, frequency_mhz, 20.0)
        assert core is not None
        reference = absorption_index_db(frequency_mhz, 20.0, 17.3, 60, 1.23)
        assert 0.75 < core / reference < 1.35

    def test_mean_ratio_across_the_band_is_near_unity(self, column):
        ratios = []
        for frequency_mhz in (12, 14, 16, 18, 20):
            core = self._core(column, frequency_mhz, 20.0)
            if core is None:
                continue
            ratios.append(
                core / absorption_index_db(frequency_mhz, 20.0, 17.3, 60, 1.23)
            )
        assert len(ratios) >= 4
        assert 0.8 < sum(ratios) / len(ratios) < 1.2

    def test_frequency_scaling_is_near_the_inverse_square_law(self, column):
        """Non-deviative absorption goes as 1/f^2; the ray geometry steepens
        it a little because a higher frequency also turns higher."""
        scaling = measure_scaling(lambda f, e, z: self._core(column, f, e))
        assert -2.9 < scaling.frequency_exponent < -1.8

    def test_obliquity_scaling_is_near_linear_in_the_secant(self, column):
        scaling = measure_scaling(lambda f, e, z: self._core(column, f, e))
        assert 0.8 < scaling.obliquity_exponent < 1.7

    def test_reference_reproduces_its_own_stated_exponents(self):
        """A guard on the oracle itself: if it stops behaving as documented,
        every comparison against it becomes meaningless."""
        scaling = measure_scaling(lambda f, e, z: absorption_index_db(f, e, z))
        assert scaling.obliquity_exponent == pytest.approx(1.0, abs=0.02)
        assert scaling.zenith_exponent == pytest.approx(0.881, abs=0.02)

    def test_incidence_angle_is_geometrically_sound(self):
        assert incidence_angle_at_110km(90.0) == pytest.approx(0.0, abs=1e-9)
        angles = [incidence_angle_at_110km(e) for e in (5, 15, 30, 60, 85)]
        assert all(b < a for a, b in zip(angles, angles[1:]))


class TestMagnetoionicModes:
    """Both modes are evaluated, and they differ in the way theory says."""

    def test_extraordinary_mode_turns_lower(self, column):
        """The X mode reflects at X = 1 - Y, so it needs less density and
        turns below the O mode."""
        for frequency_mhz in (10, 14, 18):
            ordinary = RayMedium(
                column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY,
                magnetic_field(GeoPoint(45, -2), 250.0), 1.0,
            )
            extraordinary = RayMedium(
                column.mean_profile, frequency_mhz * 1e6, Mode.EXTRAORDINARY,
                magnetic_field(GeoPoint(45, -2), 250.0), 1.0,
            )
            o_path = trace_ray(ordinary, 25.0, 0.015)
            x_path = trace_ray(extraordinary, 25.0, 0.015)
            if o_path.escaped or x_path.escaped:
                continue
            assert x_path.apex_height_km < o_path.apex_height_km

    def test_engine_evaluates_both_modes_by_default(self):
        from propsim.engine import DEFAULT_MODES, PropagationEngine
        from propsim.antenna import AntennaSpec
        from propsim.scenario import Scenario, Station

        assert set(DEFAULT_MODES) == {Mode.ORDINARY, Mode.EXTRAORDINARY}
        engine = PropagationEngine(Scenario(
            Station(MADRID, AntennaSpec(height_m=15.0), transmit_power_w=100.0),
            Station(LONDON, AntennaSpec(height_m=12.0)),
            NOON, SpaceWeather(f107=140, kp=2),
        ))
        report = engine.evaluate(14e6)
        assert report.best_of_mode(Mode.ORDINARY) is not None
        assert report.best_of_mode(Mode.EXTRAORDINARY) is not None

    def test_ordinary_mode_is_the_less_absorbed_one(self):
        """The X mode sits nearer gyroresonance and fades first."""
        from propsim.engine import PropagationEngine
        from propsim.antenna import AntennaSpec
        from propsim.scenario import Scenario, Station

        engine = PropagationEngine(Scenario(
            Station(MADRID, AntennaSpec(height_m=15.0), transmit_power_w=100.0),
            Station(LONDON, AntennaSpec(height_m=12.0)),
            NOON, SpaceWeather(f107=140, kp=2),
        ))
        splittings = [
            engine.evaluate(f * 1e6).mode_splitting_db for f in (7, 10, 12, 14)
        ]
        present = [s for s in splittings if s is not None]
        assert present and all(s >= 0.0 for s in present)

    def test_including_the_x_mode_raises_the_muf(self):
        """It should rise by roughly half the gyrofrequency, which is the
        classic separation between the two ionogram traces."""
        from propsim.engine import PropagationEngine
        from propsim.antenna import AntennaSpec
        from propsim.scenario import Scenario, Station

        scenario = Scenario(
            Station(MADRID, AntennaSpec(height_m=15.0), transmit_power_w=100.0),
            Station(LONDON, AntennaSpec(height_m=12.0)),
            NOON, SpaceWeather(f107=140, kp=2),
        )
        engine = PropagationEngine(scenario)
        ordinary_only = engine._frequency_grid(2e6, 30e6, 5e5)
        o_max = max(
            f for f in ordinary_only if engine.evaluate(f, (Mode.ORDINARY,)).is_open
        )
        both = max(f for f in ordinary_only if engine.evaluate(f).is_open)
        gyro_half = engine.magnetic_field.gyrofrequency_hz / 2.0
        assert both >= o_max
        assert both - o_max < 2.0 * gyro_half


class TestMufSearchHint:
    """The hinted MUF search is an optimisation, not a different answer."""

    def _engine(self):
        from propsim.antenna import AntennaSpec
        from propsim.engine import PropagationEngine
        from propsim.scenario import Scenario, Station

        return PropagationEngine(Scenario(
            Station(MADRID, AntennaSpec(height_m=15.0), transmit_power_w=100.0),
            Station(LONDON, AntennaSpec(height_m=12.0)),
            NOON, SpaceWeather(f107=140, sunspot_number=60, kp=2),
        ))

    def test_hinted_search_agrees_with_the_full_scan(self):
        engine = self._engine()
        full = engine.maximum_usable_frequency_hz(step_hz=5e5)
        assert full is not None
        for hint in (full * 0.7, full, full * 1.3):
            hinted = engine.maximum_usable_frequency_hz(step_hz=5e5, hint_hz=hint)
            assert hinted == pytest.approx(full, abs=1e5)

    def test_a_useless_hint_falls_back_to_the_full_scan(self):
        """A window that does not bracket the transition must be abandoned,
        not trusted -- otherwise a bad hint silently returns a wrong MUF."""
        engine = self._engine()
        full = engine.maximum_usable_frequency_hz(step_hz=5e5)
        for hint in (2.5e6, 45e6):
            assert engine.maximum_usable_frequency_hz(
                step_hz=5e5, hint_hz=hint
            ) == pytest.approx(full, abs=1e5)
