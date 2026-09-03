"""Integration invariants.

Every test here is a regression guard on a way the chain can be wired up
wrongly while each component remains individually correct.  These are the
failures that survive a green unit-test suite.
"""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from propsim.absorption import absorption_db
from propsim.antenna import AntennaSpec, AntennaType, GroundType
from propsim.constants import EARTH_RADIUS_KM
from propsim.engine import PropagationEngine
from propsim.geodesy import GeoPoint
from propsim.ionosphere import (
    IonosphericProfile,
    build_equivalent_column,
    electron_density_from_plasma_frequency,
)
from propsim.link import build_link_budget, free_space_loss_db
from propsim.magnetic import magnetic_field
from propsim.noise import NoiseEnvironment, noise_budget
from propsim.raytrace import (
    RayError,
    RayMedium,
    solve_launch_angles,
    skip_distance_km,
    trace_ray,
)
from propsim.refractive import Mode
from propsim.scenario import Scenario, Station, Weather
from propsim.spaceweather import SpaceWeather

UTC = timezone.utc
NOON = datetime(2025, 6, 21, 12, tzinfo=UTC)


def mirror_medium(mirror_height_km=300.0, frequency_hz=10e6, step_km=0.25):
    """A sharp reflector: vacuum below, opaque at and above.

    The Bouguer integrals have closed form in this medium, which makes it
    the one case where the tracer can be checked against exact analysis
    rather than against itself.
    """
    heights = list(np.arange(50.0, 600.0 + step_km, step_km))
    opaque = electron_density_from_plasma_frequency(1e9)
    densities = [0.0 if h < mirror_height_km else opaque for h in heights]

    class _Layers:
        d = e = f1 = f2 = None

    profile = IonosphericProfile(heights, densities, _Layers(), GeoPoint(0, 0), 30.0)
    return RayMedium(profile, frequency_hz, Mode.ORDINARY)


def exact_mirror_hop(elevation_deg, apex_height_km):
    """Analytic ground range and path length for the mirror medium."""
    beta = math.radians(elevation_deg)
    bouguer = EARTH_RADIUS_KM * math.cos(beta)
    apex_radius = EARTH_RADIUS_KM + apex_height_km
    angle = math.acos(bouguer / apex_radius) - beta
    ground_range = 2.0 * EARTH_RADIUS_KM * angle
    path = 2.0 * (
        math.sqrt(apex_radius**2 - bouguer**2)
        - math.sqrt(EARTH_RADIUS_KM**2 - bouguer**2)
    )
    return ground_range, path


def default_scenario(**overrides):
    tx = Station(
        GeoPoint(40.4, -3.7, "Madrid"),
        AntennaSpec(AntennaType.HORIZONTAL_DIPOLE, height_m=15.0, design_frequency_hz=14.2e6),
        transmit_power_w=100.0,
        name="Madrid",
    )
    rx = Station(
        GeoPoint(51.5, -0.1, "London"),
        AntennaSpec(AntennaType.HORIZONTAL_DIPOLE, height_m=12.0, design_frequency_hz=14.2e6),
        name="London",
    )
    fields = {
        "transmitter": tx,
        "receiver": rx,
        "when": NOON,
        "space_weather": SpaceWeather(f107=140, kp=2),
    }
    fields.update(overrides)
    return Scenario(**fields)


class TestRayTracerAgainstExactSolution:
    """The tracer is checked against closed-form analysis, not against itself."""

    @pytest.mark.parametrize("elevation", [5, 8, 12, 20, 30, 45, 60, 75])
    def test_ground_range_matches_analysis(self, elevation):
        medium = mirror_medium()
        path = trace_ray(medium, elevation, 0.0)
        expected_range, _ = exact_mirror_hop(elevation, path.apex_height_km)
        assert path.ground_range_km == pytest.approx(expected_range, rel=1e-3)

    @pytest.mark.parametrize("elevation", [5, 12, 30, 60])
    def test_path_length_matches_analysis(self, elevation):
        medium = mirror_medium()
        path = trace_ray(medium, elevation, 0.0)
        _, expected_path = exact_mirror_hop(elevation, path.apex_height_km)
        assert path.geometric_path_km == pytest.approx(expected_path, rel=1e-3)

    def test_virtual_height_recovers_the_mirror(self):
        medium = mirror_medium(mirror_height_km=300.0)
        for elevation in (5, 12, 30, 60):
            path = trace_ray(medium, elevation, 0.0)
            assert path.virtual_height_km == pytest.approx(300.0, abs=2.0)

    def test_quadrature_reproduces_the_path_length(self):
        """The nodes absorption integrates over must describe the same ray
        the geometry does."""
        medium = mirror_medium()
        path = trace_ray(medium, 20.0, 0.0)
        assert path.quadrature.ds_weight_km.sum() == pytest.approx(
            path.geometric_path_km, rel=1e-12
        )


class TestPathLengthIntegration:
    """A path length shorter than the chord means a lost sign."""

    def test_path_is_never_shorter_than_the_chord(self):
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        for frequency_mhz in (5, 8, 12, 16, 20):
            medium = RayMedium(column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY)
            for elevation in (3, 10, 20, 35, 50):
                path = trace_ray(medium, elevation, 0.015)
                if path.escaped:
                    continue
                half = path.ground_range_km / (2.0 * EARTH_RADIUS_KM)
                chord = 2.0 * EARTH_RADIUS_KM * math.sin(half)
                assert path.geometric_path_km >= chord

    def test_group_path_is_never_shorter_than_the_geometric_path(self):
        """The group index in a plasma is 1/n >= 1, always."""
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        for frequency_mhz in (5, 10, 15):
            medium = RayMedium(column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY)
            for elevation in (5, 15, 30):
                path = trace_ray(medium, elevation, 0.015)
                if not path.escaped:
                    assert path.group_path_km >= path.geometric_path_km

    def test_hop_length_stays_in_the_right_order_of_magnitude(self):
        """A 1300 km hop is about 1400 km of ray, not 60000."""
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        medium = RayMedium(column.mean_profile, 14e6, Mode.ORDINARY)
        for elevation in (5, 10, 20, 40):
            path = trace_ray(medium, elevation, 0.015)
            if not path.escaped:
                assert path.geometric_path_km < 1.6 * max(path.ground_range_km, 500.0)

    def test_group_delay_is_physically_sized(self):
        """A 1300 km hop takes about 4.5 ms, not 200."""
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        medium = RayMedium(column.mean_profile, 14e6, Mode.ORDINARY)
        path = trace_ray(medium, 20.0, 0.015)
        light_time_ms = path.ground_range_km / 299792.458 * 1e3
        assert light_time_ms <= path.group_delay_ms < 3.0 * light_time_ms


class TestSkipZone:
    """Reach is decided by solving for the landing point, never by rescaling."""

    def test_no_solution_inside_the_skip_zone(self):
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        medium = RayMedium(column.mean_profile, 18e6, Mode.ORDINARY)
        skip = skip_distance_km(medium, 0.015)
        assert skip is not None and skip > 500.0
        for distance in (100.0, 0.5 * skip, 0.9 * skip):
            assert solve_launch_angles(medium, distance, 0.015) == []

    def test_every_returned_angle_actually_lands_on_the_target(self):
        """The core guarantee: a solution is verified against the traced ray,
        so a ray that is still hundreds of kilometres up over the receiver
        cannot be reported as reaching it."""
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        for frequency_mhz in (7, 10, 14, 18, 22):
            medium = RayMedium(column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY)
            for target in (500.0, 1000.0, 1500.0, 2000.0, 2500.0):
                for elevation in solve_launch_angles(medium, target, 0.015):
                    path = trace_ray(medium, elevation, 0.015)
                    assert not path.escaped
                    assert path.ground_range_km == pytest.approx(target, abs=1.0)

    def test_skip_zone_grows_with_frequency(self):
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        skips = []
        for frequency_mhz in (12, 15, 18):
            medium = RayMedium(column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY)
            skips.append(skip_distance_km(medium, 0.015))
        assert all(b >= a for a, b in zip(skips, skips[1:]))

    def test_engine_reports_no_mode_rather_than_a_false_snr(self):
        """When nothing arrives there is no SNR to quote."""
        scenario = default_scenario()
        engine = PropagationEngine(scenario)
        report = engine.evaluate(45e6)     # far above any MUF
        assert not report.is_open
        assert report.best is None
        assert report.snr_db is None
        assert report.margin_db is None


class TestTracerRobustness:
    def test_no_ray_fails_anywhere_in_the_operating_envelope(self):
        """A grazing ray at a layer peak used to slip past the apex search
        and produce a path length inflated by orders of magnitude."""
        scenario = default_scenario()
        engine = PropagationEngine(scenario)
        for frequency_mhz in np.arange(2.0, 30.1, 1.0):
            medium = engine._medium(float(frequency_mhz) * 1e6, Mode.ORDINARY, 15.0)
            for elevation in np.arange(1.0, 60.0, 0.5):
                trace_ray(medium, float(elevation), 0.015)  # must not raise

    def test_consistency_check_rejects_a_broken_ray(self):
        import dataclasses

        medium = mirror_medium()
        path = trace_ray(medium, 20.0, 0.0)
        broken = dataclasses.replace(path, geometric_path_km=path.ground_range_km * 0.5)
        with pytest.raises(RayError):
            broken.check_consistency()


class TestAbsorptionWiring:
    def test_absorption_samples_the_whole_path_not_just_the_endpoints(self):
        """The local ionosphere must be consulted along the ray, which means
        many distinct positions per hop rather than three."""
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        medium = RayMedium(column.mean_profile, 14e6, Mode.ORDINARY)
        path = trace_ray(medium, 20.0, 0.015)
        fractions = path.quadrature.path_fraction
        in_d_region = (path.quadrature.height_km >= 60) & (path.quadrature.height_km <= 90)
        assert in_d_region.sum() > 40
        # Both the outbound and the return crossing of the D region.
        assert fractions[in_d_region].min() < 0.3
        assert fractions[in_d_region].max() > 0.7

    def test_breakdown_sums_to_the_total(self):
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        field = magnetic_field(GeoPoint(45, -2), 250.0)
        medium = RayMedium(column.mean_profile, 10e6, Mode.ORDINARY)
        path = trace_ray(medium, 15.0, 0.015)
        result = absorption_db(path, column, 10e6, Mode.ORDINARY, field, 1.0)
        assert sum(result.by_region_db.values()) == pytest.approx(
            result.non_deviative_db, abs=1e-9
        )
        assert result.non_deviative_db + result.deviative_db == pytest.approx(
            result.total_db, abs=1e-9
        )

    def test_absorption_falls_with_frequency_on_a_fixed_path(self):
        column = build_equivalent_column(
            GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, SpaceWeather(f107=140)
        )
        field = magnetic_field(GeoPoint(45, -2), 250.0)
        losses = []
        for frequency_mhz in (7, 10, 14, 18):
            medium = RayMedium(column.mean_profile, frequency_mhz * 1e6, Mode.ORDINARY)
            angles = solve_launch_angles(medium, 1500.0, 0.015)
            best = min(
                absorption_db(
                    trace_ray(medium, a, 0.015), column, frequency_mhz * 1e6,
                    Mode.ORDINARY, field, 1.0,
                ).total_db
                for a in angles
            )
            losses.append(best)
        assert all(b < a for a, b in zip(losses, losses[1:]))

    def test_day_absorbs_far_more_than_night(self):
        field = magnetic_field(GeoPoint(45, -2), 250.0)
        results = {}
        for label, hour in (("day", 12), ("night", 1)):
            when = datetime(2025, 6, 21, hour, tzinfo=UTC)
            column = build_equivalent_column(
                GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), when, SpaceWeather(f107=140)
            )
            medium = RayMedium(column.mean_profile, 7e6, Mode.ORDINARY)
            path = trace_ray(medium, 20.0, 0.015)
            results[label] = absorption_db(
                path, column, 7e6, Mode.ORDINARY, field, 1.0
            ).total_db
        assert results["day"] > 5 * results["night"]

    def test_flare_raises_absorption_through_the_d_region_alone(self):
        """No empirical flare multiplier exists, so the blackout has to come
        from the electron density -- and it does."""
        field = magnetic_field(GeoPoint(45, -2), 250.0)
        losses = {}
        for label, weather in (
            ("quiet", SpaceWeather(f107=140)),
            ("flare", SpaceWeather(f107=140).with_flare("X1")),
        ):
            column = build_equivalent_column(
                GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1), NOON, weather
            )
            medium = RayMedium(column.mean_profile, 14e6, Mode.ORDINARY)
            path = trace_ray(medium, 20.0, 0.015)
            losses[label] = absorption_db(
                path, column, 14e6, Mode.ORDINARY, field, 1.0
            )
        assert losses["flare"].total_db > losses["quiet"].total_db + 20.0
        assert losses["flare"].by_region_db["D"] > 10 * losses["quiet"].by_region_db["D"]


class TestLinkBudget:
    def test_matches_friis(self):
        noise = noise_budget(14e6, 3000)
        budget = build_link_budget(14e6, 100.0, 0.0, 0.0, 1000.0, 0.0, 0.0, noise, 10.0)
        expected = 10 * math.log10(100.0) - free_space_loss_db(1000.0, 14e6)
        assert budget.received_power_dbw == pytest.approx(expected, abs=1e-12)

    def test_every_declared_loss_is_charged(self):
        noise = noise_budget(14e6, 3000)
        budget = build_link_budget(
            14e6, 100.0, 3.0, 2.0, 2500.0, 12.0, 1.5, noise, 6.0, rain_rate_mm_h=30.0
        )
        budget.verify()
        rebuilt = (
            budget.transmit_power_dbw
            + budget.transmit_gain_dbi
            + budget.receive_gain_dbi
            - budget.spreading_loss_db
            - budget.absorption_loss_db
            - budget.ground_reflection_loss_db
            - budget.rain_attenuation_db
        )
        assert rebuilt == pytest.approx(budget.received_power_dbw, abs=1e-12)

    def test_rain_attenuation_moves_the_received_power(self):
        """A loss that is displayed but not subtracted is worse than none."""
        noise = noise_budget(14e6, 3000)
        dry = build_link_budget(14e6, 100.0, 0, 0, 2500.0, 10, 1, noise, 6.0, rain_rate_mm_h=0)
        wet = build_link_budget(14e6, 100.0, 0, 0, 2500.0, 10, 1, noise, 6.0, rain_rate_mm_h=50)
        assert dry.received_power_dbw - wet.received_power_dbw == pytest.approx(
            wet.rain_attenuation_db, abs=1e-12
        )

    def test_verify_catches_an_uncharged_loss(self):
        import dataclasses

        noise = noise_budget(14e6, 3000)
        budget = build_link_budget(14e6, 100.0, 0, 0, 1000.0, 5, 0, noise, 6.0)
        tampered = dataclasses.replace(budget, absorption_loss_db=budget.absorption_loss_db + 20)
        with pytest.raises(ValueError, match="does not reconstruct"):
            tampered.verify()


class TestNoise:
    def test_sources_combine_in_power_not_in_decibels(self):
        budget = noise_budget(7e6, 3000)
        naive = (
            budget.galactic_fa_db + budget.atmospheric_fa_db + budget.man_made_fa_db
        )
        assert budget.total_fa_db < naive - 40.0
        assert budget.total_fa_db >= max(
            budget.galactic_fa_db, budget.atmospheric_fa_db, budget.man_made_fa_db
        )

    @pytest.mark.parametrize(
        "frequency_mhz,reference", [(1.8, 70), (7, 51), (14, 41), (28, 32)]
    )
    def test_atmospheric_noise_tracks_the_reference_curve(self, frequency_mhz, reference):
        budget = noise_budget(frequency_mhz * 1e6, 3000, sunlit_fraction=1.0)
        assert budget.atmospheric_fa_db == pytest.approx(reference, abs=3.0)

    def test_noise_falls_with_frequency(self):
        levels = [noise_budget(f * 1e6, 3000).total_fa_db for f in (2, 4, 8, 16, 28)]
        assert all(b < a for a, b in zip(levels, levels[1:]))

    def test_environment_changes_the_total(self):
        quiet = noise_budget(7e6, 3000, NoiseEnvironment.QUIET_RURAL).total_fa_db
        city = noise_budget(7e6, 3000, NoiseEnvironment.CITY).total_fa_db
        assert city > quiet + 2.0

    def test_auroral_noise_registers_during_a_storm(self):
        calm = noise_budget(7e6, 3000, geomagnetic_latitude_deg=67, kp=1).total_fa_db
        storm = noise_budget(7e6, 3000, geomagnetic_latitude_deg=67, kp=9).total_fa_db
        assert storm > calm + 8.0

    def test_auroral_noise_is_confined_to_high_latitude(self):
        equator = noise_budget(7e6, 3000, geomagnetic_latitude_deg=5, kp=9)
        assert equator.auroral_fa_db < equator.atmospheric_fa_db - 15.0


class TestAntennaUnits:
    def test_height_is_metres_and_a_kilometre_value_is_rejected(self):
        AntennaSpec(height_m=20.0)                      # fine
        with pytest.raises(ValueError, match="METRES"):
            AntennaSpec(height_m=0.02)                  # 20 m written as km

    def test_height_km_is_a_pure_derived_view(self):
        assert AntennaSpec(height_m=20.0).height_km == pytest.approx(0.02)

    @pytest.mark.parametrize("height_m", [20.0, 10.0, 5.0])
    def test_main_lobe_follows_image_theory(self, height_m):
        """First maximum at sin(elevation) = lambda / 4h."""
        frequency = 14.2e6
        wavelength = 299792458.0 / frequency
        spec = AntennaSpec(
            AntennaType.HORIZONTAL_DIPOLE, height_m=height_m,
            design_frequency_hz=frequency, feedline_loss_db=0.0,
        )
        expected = math.degrees(math.asin(min(1.0, wavelength / (4 * height_m))))
        assert spec.best_elevation_deg(frequency) == pytest.approx(expected, abs=1.5)

    def test_horizontal_antenna_has_a_horizon_null(self):
        spec = AntennaSpec(
            AntennaType.HORIZONTAL_DIPOLE, height_m=20.0,
            design_frequency_hz=14.2e6, feedline_loss_db=0.0,
        )
        assert spec.gain_dbi(0.5, 14.2e6) < spec.gain_dbi(15.0, 14.2e6) - 15.0

    def test_vertical_over_sea_beats_vertical_over_dry_ground(self):
        def gain(ground):
            return AntennaSpec(
                AntennaType.VERTICAL_QUARTER_WAVE, height_m=0.5, ground=ground,
                design_frequency_hz=14.2e6, feedline_loss_db=0.0,
            ).gain_dbi(5.0, 14.2e6)

        assert gain(GroundType.SALT_WATER) > gain(GroundType.DRY_GROUND) + 8.0

    def test_short_vertical_is_charged_for_being_short(self):
        short = AntennaSpec(
            AntennaType.SHORT_VERTICAL, height_m=3.0, boom_length_m=2.0,
            design_frequency_hz=3.6e6, feedline_loss_db=0.0,
        )
        full = AntennaSpec(
            AntennaType.SHORT_VERTICAL, height_m=3.0, boom_length_m=2.0,
            design_frequency_hz=28e6, feedline_loss_db=0.0,
        )
        assert short.gain_dbi(20.0, 3.6e6) < full.gain_dbi(20.0, 28e6)


class TestStationCompleteness:
    def test_a_station_cannot_have_an_absurd_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth"):
            Station(GeoPoint(40, -3), AntennaSpec(), bandwidth_hz=0.1)

    def test_a_station_cannot_have_an_absurd_power(self):
        with pytest.raises(ValueError, match="transmit power"):
            Station(GeoPoint(40, -3), AntennaSpec(), transmit_power_w=0.0)

    def test_lof_uses_the_real_station(self):
        """LOF must move when the station changes, which proves it is not
        being computed from defensive defaults."""
        weak = default_scenario()
        strong_tx = Station(
            weak.transmitter.location, weak.transmitter.antenna,
            transmit_power_w=1500.0, name="Madrid",
        )
        strong = default_scenario(transmitter=strong_tx)
        lof_weak = PropagationEngine(weak).lowest_usable_frequency_hz(step_hz=1e6)
        lof_strong = PropagationEngine(strong).lowest_usable_frequency_hz(step_hz=1e6)
        assert lof_weak is not None and lof_strong is not None
        assert lof_strong < lof_weak

    def test_lof_responds_to_required_snr(self):
        easy_rx = Station(
            GeoPoint(51.5, -0.1), AntennaSpec(height_m=12.0), required_snr_db=-6.0
        )
        hard_rx = Station(
            GeoPoint(51.5, -0.1), AntennaSpec(height_m=12.0), required_snr_db=30.0
        )
        lof_easy = PropagationEngine(
            default_scenario(receiver=easy_rx)
        ).lowest_usable_frequency_hz(step_hz=1e6)
        lof_hard = PropagationEngine(
            default_scenario(receiver=hard_rx)
        ).lowest_usable_frequency_hz(step_hz=1e6)
        assert lof_easy is not None and lof_hard is not None
        assert lof_easy < lof_hard


class TestMufAndLof:
    def test_muf_exceeds_lof(self):
        engine = PropagationEngine(default_scenario())
        prediction = engine.predict(2e6, 30e6, 1e6)
        assert prediction.muf_mhz is not None
        assert prediction.lof_mhz is not None
        assert prediction.muf_mhz > prediction.lof_mhz

    def test_muf_is_open_and_just_above_it_is_not(self):
        engine = PropagationEngine(default_scenario())
        muf = engine.maximum_usable_frequency_hz(step_hz=1e6)
        assert muf is not None
        assert engine.evaluate(muf).is_open
        assert not engine.evaluate(muf * 1.25).is_open

    def test_muf_rises_with_solar_activity(self):
        quiet = PropagationEngine(
            default_scenario(space_weather=SpaceWeather(f107=70, kp=2))
        ).maximum_usable_frequency_hz(step_hz=1e6)
        active = PropagationEngine(
            default_scenario(space_weather=SpaceWeather(f107=250, kp=2))
        ).maximum_usable_frequency_hz(step_hz=1e6)
        assert active > quiet

    def test_muf_is_higher_by_day_than_by_night(self):
        day = PropagationEngine(
            default_scenario(when=NOON)
        ).maximum_usable_frequency_hz(step_hz=1e6)
        night = PropagationEngine(
            default_scenario(when=datetime(2025, 6, 21, 1, tzinfo=UTC))
        ).maximum_usable_frequency_hz(step_hz=1e6)
        assert day > night

    def test_muf_is_geometric_and_ignores_the_transmitter(self):
        """MUF asks whether a ray lands, not whether it can be heard."""
        base = default_scenario()
        loud = default_scenario(
            transmitter=Station(
                base.transmitter.location, base.transmitter.antenna,
                transmit_power_w=1500.0,
            )
        )
        assert PropagationEngine(base).maximum_usable_frequency_hz(
            step_hz=1e6
        ) == PropagationEngine(loud).maximum_usable_frequency_hz(step_hz=1e6)


class TestOptimiserConsistency:
    def test_chosen_mode_is_the_one_the_budget_was_built_from(self):
        """The optimiser must rank modes by the loss the budget charges, not
        by a loss with an extra penalty added on top."""
        engine = PropagationEngine(default_scenario())
        report = engine.evaluate(14.2e6)
        assert report.is_open
        best = report.best
        assert best.margin_db == max(m.margin_db for m in report.modes)
        best.budget.verify()
        assert best.budget.absorption_loss_db == pytest.approx(
            best.absorption.total_db * best.hops, rel=1e-12
        )

    def test_every_mode_reports_a_reconstructible_budget(self):
        engine = PropagationEngine(default_scenario())
        for frequency_mhz in (7.1, 10.1, 14.2):
            for mode in engine.evaluate(frequency_mhz * 1e6).modes:
                mode.budget.verify()
                mode.summary()


class TestWeatherEffects:
    def test_rain_raises_noise_and_lowers_margin(self):
        dry = PropagationEngine(default_scenario()).evaluate(7.1e6)
        wet = PropagationEngine(
            default_scenario(weather=Weather(rain_rate_mm_h=30.0))
        ).evaluate(7.1e6)
        assert dry.margin_db is not None and wet.margin_db is not None
        assert wet.margin_db < dry.margin_db

    def test_freezing_ground_degrades_a_vertical(self):
        vertical_tx = Station(
            GeoPoint(40.4, -3.7),
            AntennaSpec(
                AntennaType.VERTICAL_QUARTER_WAVE, height_m=0.5,
                design_frequency_hz=7.1e6,
            ),
            transmit_power_w=100.0,
        )
        thawed = PropagationEngine(
            default_scenario(transmitter=vertical_tx)
        ).evaluate(7.1e6)
        frozen = PropagationEngine(
            default_scenario(transmitter=vertical_tx, weather=Weather(freezing=True))
        ).evaluate(7.1e6)
        assert thawed.margin_db is not None and frozen.margin_db is not None
        assert frozen.margin_db < thawed.margin_db


class TestEndToEnd:
    def test_a_daytime_mid_range_path_favours_the_middle_bands(self):
        prediction = PropagationEngine(default_scenario()).predict(2e6, 30e6, 1e6)
        bands = prediction.band_report()
        assert bands[0]["band"] in {"20 m", "17 m", "30 m"}

    def test_the_low_bands_open_at_night(self):
        night = default_scenario(when=datetime(2025, 6, 21, 1, tzinfo=UTC))
        day_margin = PropagationEngine(default_scenario()).evaluate(3.65e6).margin_db
        night_margin = PropagationEngine(night).evaluate(3.65e6).margin_db
        assert night_margin > day_margin

    def test_a_flare_closes_the_daytime_path(self):
        quiet = PropagationEngine(default_scenario()).evaluate(7.1e6)
        flare = PropagationEngine(
            default_scenario(space_weather=SpaceWeather(f107=140, kp=2).with_flare("X5"))
        ).evaluate(7.1e6)
        assert flare.margin_db < quiet.margin_db - 30.0

    def test_summary_is_serialisable(self):
        import json

        prediction = PropagationEngine(default_scenario()).predict(3e6, 25e6, 2e6)
        json.dumps(prediction.summary())
