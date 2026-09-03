"""Variability, reliability and sporadic E.

The claim under test is that the model answers "how often does this circuit
work" rather than "what is the SNR today", and that sporadic E enters as a
probability rather than as a switch.
"""

import math
from datetime import datetime, timezone

import pytest

from propsim.antenna import AntennaSpec
from propsim.engine import PropagationEngine
from propsim.geodesy import GeoPoint
from propsim.ionosphere import build_profile
from propsim.reliability import ReliabilityPredictor
from propsim.scenario import Scenario, Station
from propsim.spaceweather import SpaceWeather
from propsim.sporadic_e import (
    SporadicELayer,
    median_foes_mhz,
    refine_grid_for_layer,
    sporadic_e_probability,
)
from propsim.variability import (
    QUANTILES,
    VariabilitySpread,
    fof2_decile_factors,
    fof2_multipliers,
    reliability_from_samples,
)

UTC = timezone.utc
MADRID = GeoPoint(40.4, -3.7)
LONDON = GeoPoint(51.5, -0.1)
JUNE_MORNING = datetime(2025, 6, 21, 10, tzinfo=UTC)


def scenario(when=JUNE_MORNING, **overrides):
    fields = {
        "transmitter": Station(
            MADRID, AntennaSpec(height_m=15.0), transmit_power_w=100.0, name="Madrid"
        ),
        "receiver": Station(LONDON, AntennaSpec(height_m=12.0), name="London"),
        "when": when,
        "space_weather": SpaceWeather(f107=140, sunspot_number=60, kp=2),
    }
    fields.update(overrides)
    return Scenario(**fields)


class TestVariabilitySpread:
    def test_mid_latitude_daytime_is_the_quiet_case(self):
        quiet = fof2_decile_factors(45.0, 1.0, 2.0)
        assert 0.80 < quiet.lower_decile < 0.90
        assert 1.10 < quiet.upper_decile < 1.20

    def test_spread_widens_at_night(self):
        day = fof2_decile_factors(45.0, 1.0, 2.0)
        night = fof2_decile_factors(45.0, 0.0, 2.0)
        assert night.log_sigma > day.log_sigma

    def test_spread_widens_in_a_storm(self):
        calm = fof2_decile_factors(45.0, 1.0, 2.0)
        storm = fof2_decile_factors(45.0, 1.0, 8.0)
        assert storm.log_sigma > calm.log_sigma

    def test_spread_widens_towards_the_equator_and_the_aurora(self):
        mid = fof2_decile_factors(45.0, 1.0, 2.0).log_sigma
        assert fof2_decile_factors(12.0, 1.0, 2.0).log_sigma > mid
        assert fof2_decile_factors(65.0, 1.0, 2.0).log_sigma > mid

    def test_multipliers_are_ordered_and_straddle_unity(self):
        multipliers = fof2_multipliers(fof2_decile_factors(45.0, 1.0, 2.0))
        assert list(multipliers) == sorted(multipliers)
        assert multipliers[0] < 1.0 < multipliers[-1]

    def test_quantile_inversion_is_self_consistent(self):
        spread = VariabilitySpread(0.85, 1.15)
        assert spread.multiplier_at(0.1) == pytest.approx(0.85, rel=1e-6)
        assert spread.multiplier_at(0.9) == pytest.approx(1.15, rel=1e-6)

    def test_rejects_an_impossible_spread(self):
        with pytest.raises(ValueError):
            VariabilitySpread(1.2, 1.5)


class TestReliabilityFromSamples:
    def test_all_closing_is_certain(self):
        assert reliability_from_samples([2, 5, 8, 11, 14]) == 1.0

    def test_none_closing_is_impossible(self):
        assert reliability_from_samples([-9, -7, -5, -3, -1]) == 0.0

    def test_a_clean_crossing_is_interpolated(self):
        value = reliability_from_samples([-9, -4, 2, 7, 11])
        assert 0.5 < value < 0.7

    def test_a_failure_at_every_sample_is_never_extrapolated_into_hope(self):
        """A normal fit would report a few percent here. There is no
        evidence for a few percent; every sample failed."""
        assert reliability_from_samples([-30, -25, -20, -15, -9]) == 0.0

    def test_no_ray_counts_as_a_failure_not_as_missing_data(self):
        assert reliability_from_samples([None, None, None, None, None]) == 0.0
        assert 0.0 < reliability_from_samples([None, None, -1, 4, 9]) < 0.5

    def test_non_monotonic_margins_fall_back_to_probability_mass(self):
        """Near the critical frequency a worse ionosphere can give a better
        margin by forcing a different mode; interpolating one crossing there
        would be meaningless."""
        value = reliability_from_samples([5, -3, 8, 12, 15])
        assert 0.7 < value < 0.9

    def test_rejects_a_mismatched_sample_count(self):
        with pytest.raises(ValueError):
            reliability_from_samples([1, 2], QUANTILES)


class TestSporadicEClimatology:
    def test_summer_far_exceeds_winter_at_mid_latitude(self):
        summer = sporadic_e_probability(datetime(2025, 6, 15, 12, tzinfo=UTC), 42, 10)
        winter = sporadic_e_probability(datetime(2025, 12, 15, 12, tzinfo=UTC), 42, 10)
        assert summer > 10 * winter

    def test_the_season_flips_in_the_southern_hemisphere(self):
        december = sporadic_e_probability(
            datetime(2025, 12, 15, 12, tzinfo=UTC), -35, 10
        )
        june = sporadic_e_probability(datetime(2025, 6, 15, 12, tzinfo=UTC), -35, 10)
        assert december > june

    def test_mid_morning_beats_the_small_hours(self):
        when = datetime(2025, 6, 15, 12, tzinfo=UTC)
        assert sporadic_e_probability(when, 42, 10) > sporadic_e_probability(when, 42, 3)

    def test_auroral_population_responds_to_kp(self):
        when = datetime(2025, 6, 15, 12, tzinfo=UTC)
        calm = sporadic_e_probability(when, 66, 22, kp=1)
        storm = sporadic_e_probability(when, 66, 22, kp=7)
        assert storm > calm

    def test_a_higher_threshold_is_rarer(self):
        when = datetime(2025, 6, 15, 12, tzinfo=UTC)
        common = sporadic_e_probability(when, 42, 10, threshold_mhz=5)
        rare = sporadic_e_probability(when, 42, 10, threshold_mhz=20)
        assert rare < common

    def test_probability_is_always_a_probability(self):
        for month in range(1, 13):
            for latitude in (-70, -40, -10, 0, 10, 40, 70):
                for hour in (0, 6, 12, 18):
                    value = sporadic_e_probability(
                        datetime(2025, month, 15, 12, tzinfo=UTC), latitude, hour, 5.0
                    )
                    assert 0.0 <= value <= 1.0

    def test_median_foes_is_in_a_plausible_range(self):
        for month in (1, 6, 12):
            for latitude in (0, 40, 65):
                value = median_foes_mhz(
                    datetime(2025, month, 15, 12, tzinfo=UTC), latitude, 10
                )
                assert 2.0 <= value <= 18.0


class TestSporadicELayer:
    def test_patch_is_thin_and_symmetric(self):
        layer = SporadicELayer(foes_mhz=9.0, height_km=105.0, thickness_km=1.0)
        assert layer.density_at(104.0) == pytest.approx(layer.density_at(106.0))
        assert layer.density_at(105.0) > 100 * layer.density_at(110.0)

    def test_foes_round_trips_through_the_density(self):
        from propsim.ionosphere import plasma_frequency_mhz

        layer = SporadicELayer(foes_mhz=7.5)
        assert plasma_frequency_mhz(layer.peak_density) == pytest.approx(7.5, rel=1e-9)

    def test_grid_refinement_actually_resolves_the_patch(self):
        """On the plain 2 km grid a 1 km patch falls between samples and the
        model never sees it."""
        layer = SporadicELayer(foes_mhz=9.0)
        coarse = [50.0 + 2.0 * i for i in range(276)]
        refined = refine_grid_for_layer(coarse, layer)
        near = [h for h in refined if 104.0 <= h <= 106.0]
        assert len(near) >= 8
        assert all(b > a for a, b in zip(refined, refined[1:]))

    def test_patch_appears_in_the_built_profile(self):
        layer = SporadicELayer(foes_mhz=10.0)
        weather = SpaceWeather(f107=140, sunspot_number=60)
        plain = build_profile(MADRID, JUNE_MORNING, weather, 0.9)
        with_es = build_profile(MADRID, JUNE_MORNING, weather, 0.9, sporadic_e=layer)
        assert with_es.density_at(105.0) > 5 * plain.density_at(105.0)
        # Far from the patch nothing changes.
        assert with_es.density_at(300.0) == pytest.approx(
            plain.density_at(300.0), rel=1e-9
        )


class TestSporadicEPropagation:
    def test_a_patch_opens_bands_the_f_layer_cannot(self):
        base = scenario()
        plain = PropagationEngine(base)
        with_es = PropagationEngine(base, sporadic_e=SporadicELayer(foes_mhz=9.3))
        assert not plain.evaluate(28e6).is_open
        assert with_es.evaluate(28e6).is_open

    def test_a_patch_raises_the_muf_far_above_the_f_layer(self):
        base = scenario()
        plain = PropagationEngine(base).maximum_usable_frequency_hz(step_hz=1e6)
        with_es = PropagationEngine(
            base, sporadic_e=SporadicELayer(foes_mhz=9.3)
        ).maximum_usable_frequency_hz(high_hz=60e6, step_hz=1e6)
        assert with_es > 2 * plain

    def test_the_patch_reflects_the_ray_at_its_own_height(self):
        with_es = PropagationEngine(
            scenario(), sporadic_e=SporadicELayer(foes_mhz=9.3, height_km=105.0)
        )
        best = with_es.evaluate(28e6).best
        assert best is not None
        assert 95.0 < best.path.apex_height_km < 108.0

    def test_a_patch_can_screen_the_f_layer(self):
        """Es screening is a real cost, not only a bonus: the patch can steal
        a ray that would otherwise have taken a better F-layer path."""
        base = scenario()
        plain = PropagationEngine(base).evaluate(14e6)
        with_es = PropagationEngine(
            base, sporadic_e=SporadicELayer(foes_mhz=9.3)
        ).evaluate(14e6)
        assert plain.margin_db is not None and with_es.margin_db is not None
        assert with_es.best.path.apex_height_km < plain.best.path.apex_height_km


class TestReliabilityPredictor:
    def test_reliability_is_a_probability(self):
        predictor = ReliabilityPredictor(scenario())
        for row in predictor.band_reliability():
            assert 0.0 <= row["reliability"] <= 1.0

    def test_muf_is_reported_as_a_spread_not_a_point(self):
        predictor = ReliabilityPredictor(scenario())
        muf = predictor.muf_distribution_hz(step_hz=1e6)
        assert muf["lower_decile"] < muf["median"] < muf["upper_decile"]

    def test_a_strong_circuit_is_reliable_and_a_dead_one_is_not(self):
        predictor = ReliabilityPredictor(scenario())
        rows = {row["band"]: row for row in predictor.band_reliability()}
        assert rows["20 m"]["reliability"] > 0.8
        assert rows["160 m"]["reliability"] < 0.2

    def test_sporadic_e_enters_as_a_probability_not_a_switch(self):
        predictor = ReliabilityPredictor(scenario())
        assert 0.0 < predictor.sporadic_e_probability < 1.0
        result = predictor.at(28e6)
        # Dead through the F layer, alive on Es, reported at the Es rate.
        assert result.reliability_without_es == 0.0
        assert result.reliability_with_es == 1.0
        assert result.reliability == pytest.approx(
            predictor.sporadic_e_probability, abs=1e-9
        )

    def test_disabling_sporadic_e_removes_its_contribution(self):
        predictor = ReliabilityPredictor(scenario(), include_sporadic_e=False)
        assert predictor.sporadic_e_probability == 0.0
        assert predictor.at(28e6).reliability == 0.0

    def test_winter_has_no_sporadic_e_contribution(self):
        winter = ReliabilityPredictor(
            scenario(when=datetime(2025, 12, 21, 10, tzinfo=UTC))
        )
        summer = ReliabilityPredictor(scenario())
        assert summer.sporadic_e_probability > winter.sporadic_e_probability

    def test_summary_is_serialisable(self):
        import json

        json.dumps(ReliabilityPredictor(scenario()).summary(step_hz=2e6))


class TestSharedMedianEngine:
    """Handing over an already-built engine is an optimisation, not a
    different calculation."""

    def test_shared_engine_gives_the_same_answer(self):
        base = scenario()
        engine = PropagationEngine(base)
        engine.predict(2e6, 30e6, 2e6)          # warms its report cache

        shared = ReliabilityPredictor(base, median_engine=engine)
        fresh = ReliabilityPredictor(base)
        for a, b in zip(shared.band_reliability(), fresh.band_reliability()):
            assert a["band"] == b["band"]
            assert a["reliability"] == pytest.approx(b["reliability"], abs=1e-9)

    def test_a_disturbed_engine_is_refused(self):
        """Silently accepting one would compute the whole distribution about
        the wrong centre."""
        base = scenario()
        with pytest.raises(ValueError, match="undisturbed median"):
            ReliabilityPredictor(
                base, median_engine=PropagationEngine(base, fof2_multiplier=1.2)
            )
        with pytest.raises(ValueError, match="undisturbed median"):
            ReliabilityPredictor(
                base,
                median_engine=PropagationEngine(
                    base, sporadic_e=SporadicELayer(foes_mhz=8.0)
                ),
            )


class TestReportCache:
    """The frequency-report memo must be a pure memoisation."""

    def test_repeated_evaluation_returns_the_same_report(self):
        engine = PropagationEngine(scenario())
        first = engine.evaluate(14e6)
        second = engine.evaluate(14e6)
        assert first is second

    def test_different_mode_sets_are_cached_separately(self):
        from propsim.refractive import Mode

        engine = PropagationEngine(scenario())
        both = engine.evaluate(14e6)
        ordinary_only = engine.evaluate(14e6, (Mode.ORDINARY,))
        assert both is not ordinary_only
        assert len(ordinary_only.modes) <= len(both.modes)

    def test_a_cached_report_matches_an_uncached_one(self):
        base = scenario()
        warm = PropagationEngine(base)
        warm.evaluate(14e6)
        cold = PropagationEngine(base)
        assert warm.evaluate(14e6).margin_db == pytest.approx(
            cold.evaluate(14e6).margin_db, abs=1e-12
        )


class TestMultipathCache:
    def test_multipath_is_memoised_per_availability(self):
        report = PropagationEngine(scenario()).evaluate(14e6)
        assert report.multipath(0.9) is report.multipath(0.9)
        assert report.multipath(0.9) is not report.multipath(0.99)
        assert report.multipath(0.99).fade_margin_db > report.multipath(0.9).fade_margin_db
