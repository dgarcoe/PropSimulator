"""Multipath and fading: variation within a day rather than between days."""

import math

import pytest

from propsim.fading import (
    _bessel_i0,
    fade_depth_db,
    multipath_profile,
    required_margin_db,
    rician_k_factor,
)


class TestBessel:
    @pytest.mark.parametrize(
        "x,expected",
        [(0.0, 1.0), (0.5, 1.0634834), (1.0, 1.2660658),
         (5.0, 27.239872), (15.0, 339649.37), (20.0, 4.3558283e7)],
    )
    def test_matches_reference_values(self, x, expected):
        assert _bessel_i0(x) == pytest.approx(expected, rel=1e-6)

    def test_matches_scipy_across_both_branches(self):
        """SciPy is a test-only oracle; the package itself stays numpy-only."""
        scipy_special = pytest.importorskip("scipy.special")
        for x in (0.0, 0.3, 1.0, 5.0, 12.0, 49.9, 50.0, 50.1, 80.0, 200.0):
            assert _bessel_i0(x) == pytest.approx(
                float(scipy_special.i0(x)), rel=1e-9
            )

    def test_the_branches_join_smoothly(self):
        """Compared against the true growth rate, not against each other:
        I0 rises about 0.2% over this interval, so demanding the two
        endpoints be equal would test arithmetic rather than continuity."""
        scipy_special = pytest.importorskip("scipy.special")
        below, above = 49.999, 50.001
        modelled = _bessel_i0(above) / _bessel_i0(below)
        exact = float(scipy_special.i0(above)) / float(scipy_special.i0(below))
        assert modelled == pytest.approx(exact, rel=1e-9)

    def test_is_even(self):
        assert _bessel_i0(-3.0) == pytest.approx(_bessel_i0(3.0))


class TestFadeDepth:
    def test_rayleigh_matches_the_textbook(self):
        """The classic numbers: about 10 dB at 90%, 20 dB at 99%."""
        assert fade_depth_db(0.0, 0.9) == pytest.approx(9.6, abs=0.6)
        assert fade_depth_db(0.0, 0.99) == pytest.approx(20.0, abs=0.6)

    def test_a_single_mode_does_not_fade(self):
        assert fade_depth_db(math.inf, 0.9) == 0.0
        assert fade_depth_db(math.inf, 0.99) == 0.0

    def test_deeper_fades_are_rarer(self):
        depths = [fade_depth_db(1.0, a) for a in (0.5, 0.9, 0.99)]
        assert all(b > a for a, b in zip(depths, depths[1:]))

    def test_a_stronger_dominant_mode_fades_less(self):
        depths = [fade_depth_db(k, 0.9) for k in (0.0, 1.0, 10.0, 100.0)]
        assert all(b < a for a, b in zip(depths, depths[1:]))

    def test_a_dominant_mode_still_gives_a_sane_answer(self):
        """A narrow Rician spike must not fall between integration steps.
        When it did, the bisection ran to its bracket and reported a
        *negative* fade depth -- a signal stronger than its own mean."""
        for k_factor in (1e2, 1e3, 1e4, 1e6, 1e9):
            depth = fade_depth_db(k_factor, 0.9)
            assert 0.0 <= depth < 2.0

    @pytest.mark.parametrize("k_factor", [100.0, 1000.0, 10000.0])
    def test_large_k_matches_the_gaussian_limit(self, k_factor):
        """For a strongly dominant mode the power distribution tends to a
        Gaussian of mean 1 and standard deviation sqrt(2/K); the 10th
        percentile is then 1 - 1.2816 sqrt(2/K).

        Asserting a *value* rather than a range matters here: an earlier
        underflow guard collapsed the density to zero for large K, and a
        bounds-only test accepted the resulting 0.0 dB as plausible."""
        expected_level = 1.0 - 1.2816 * math.sqrt(2.0 / k_factor)
        expected_db = -10.0 * math.log10(expected_level)
        assert fade_depth_db(k_factor, 0.9) == pytest.approx(expected_db, rel=0.12)

    def test_the_power_distribution_is_normalised(self):
        """A density that integrates to well under one is a density that has
        been partly thrown away."""
        from propsim.fading import _rician_power_cdf

        for k_factor in (0.0, 1.0, 10.0, 1000.0, 1e5):
            assert _rician_power_cdf(50.0, k_factor) == pytest.approx(1.0, abs=1e-3)

    def test_fade_depth_is_never_negative(self):
        for k_factor in (0.0, 0.5, 3.0, 50.0, 1e5, 1e8):
            for availability in (0.5, 0.9, 0.99):
                assert fade_depth_db(k_factor, availability) >= 0.0

    def test_depth_falls_smoothly_towards_the_single_mode_limit(self):
        depths = [fade_depth_db(k, 0.9) for k in (1e1, 1e2, 1e3, 1e4, 1e5)]
        assert all(b <= a for a, b in zip(depths, depths[1:]))
        assert depths[-1] < 0.1

    def test_rejects_an_impossible_availability(self):
        for value in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError):
                fade_depth_db(1.0, value)


class TestKFactor:
    def test_a_lone_mode_is_infinite(self):
        assert math.isinf(rician_k_factor([-100.0]))

    def test_two_equal_modes_give_unity(self):
        assert rician_k_factor([-100.0, -100.0]) == pytest.approx(1.0)

    def test_a_dominant_mode_gives_a_large_factor(self):
        assert rician_k_factor([-100.0, -120.0]) == pytest.approx(100.0, rel=1e-6)

    def test_order_does_not_matter(self):
        assert rician_k_factor([-110.0, -100.0]) == pytest.approx(
            rician_k_factor([-100.0, -110.0])
        )

    def test_negligible_modes_barely_matter(self):
        strong = rician_k_factor([-100.0, -140.0])
        assert required_margin_db([-100.0, -140.0], 0.9) < 0.2
        assert strong > 1000.0


class TestMultipathProfile:
    def test_a_single_mode_is_flat_and_costs_nothing(self):
        profile = multipath_profile([-100.0], [4.3], 2400.0)
        assert profile.delay_spread_ms == 0.0
        assert math.isinf(profile.coherence_bandwidth_hz)
        assert not profile.is_frequency_selective
        assert profile.fade_margin_db == 0.0
        assert profile.is_effectively_single_mode

    def test_delay_spread_is_power_weighted(self):
        """A weak far-delayed mode must not smear the channel as much as a
        strong one at the same delay."""
        weak = multipath_profile([-100.0, -130.0], [4.0, 8.0], 2400.0)
        strong = multipath_profile([-100.0, -101.0], [4.0, 8.0], 2400.0)
        assert weak.delay_spread_ms < strong.delay_spread_ms

    def test_wide_spread_makes_the_channel_selective(self):
        narrow = multipath_profile([-100.0, -101.0], [4.00, 4.02], 2400.0)
        wide = multipath_profile([-100.0, -101.0], [4.0, 7.0], 2400.0)
        assert not narrow.is_frequency_selective
        assert wide.is_frequency_selective
        assert wide.coherence_bandwidth_hz < narrow.coherence_bandwidth_hz

    def test_a_narrow_signal_survives_a_channel_that_breaks_a_wide_one(self):
        powers, delays = [-100.0, -101.0], [4.0, 5.0]
        assert multipath_profile(powers, delays, 2400.0).is_frequency_selective
        assert not multipath_profile(powers, delays, 50.0).is_frequency_selective

    def test_coherence_bandwidth_follows_the_inverse_of_the_spread(self):
        profile = multipath_profile([-100.0, -100.0], [0.0, 2.0], 2400.0)
        expected = 1.0 / (2.0 * math.pi * profile.delay_spread_ms * 1e-3)
        assert profile.coherence_bandwidth_hz == pytest.approx(expected, rel=1e-9)

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError):
            multipath_profile([-100.0, -100.0], [4.0], 2400.0)
        with pytest.raises(ValueError):
            multipath_profile([], [], 2400.0)
        with pytest.raises(ValueError):
            multipath_profile([-100.0], [4.0], 0.0)


class TestEngineIntegration:
    def _report(self, frequency_hz=14e6):
        from datetime import datetime, timezone

        from propsim.antenna import AntennaSpec
        from propsim.engine import PropagationEngine
        from propsim.geodesy import GeoPoint
        from propsim.scenario import Scenario, Station
        from propsim.spaceweather import SpaceWeather

        engine = PropagationEngine(Scenario(
            Station(GeoPoint(40.4, -3.7), AntennaSpec(height_m=15.0),
                    transmit_power_w=100.0),
            Station(GeoPoint(51.5, -0.1), AntennaSpec(height_m=12.0)),
            datetime(2025, 6, 21, 12, tzinfo=timezone.utc),
            SpaceWeather(f107=140, sunspot_number=60, kp=2),
        ))
        return engine.evaluate(frequency_hz)

    def test_a_real_circuit_has_several_interfering_modes(self):
        report = self._report()
        profile = report.multipath()
        assert profile is not None
        assert profile.mode_count > 1
        assert profile.fade_margin_db > 0.0

    def test_effective_margin_is_below_the_mean_power_margin(self):
        """The budget reports mean power; the circuit lives at the fades."""
        report = self._report()
        assert report.effective_margin_db() < report.margin_db

    def test_a_stricter_availability_costs_more_margin(self):
        report = self._report()
        assert report.effective_margin_db(0.99) < report.effective_margin_db(0.9)

    def test_a_closed_band_has_nothing_to_fade_against_and_no_margin(self):
        """45 MHz on this circuit: the ionosphere returns nothing.

        What is left is the ground wave, which on a 7500 km path at 45 MHz
        is a couple of hundred decibels below free space -- a number the
        model can compute and not a signal.  It is the only arrival, so
        there is nothing for it to interfere with: one route, no fading,
        and no skywave margin at all.
        """
        report = self._report(45e6)
        assert not report.is_open
        assert report.ground_wave is not None
        assert report.ground_wave.margin_db < -100.0
        profile = report.multipath()
        assert profile is not None and profile.mode_count == 1
        assert report.fade_margin_db() == 0.0
        assert report.effective_margin_db() is None

    def test_a_hopeless_ground_wave_is_not_counted_as_multipath(self):
        """A route 40 dB down cannot fade the one above it.

        At that ratio the resultant swings by 0.086 dB between full
        addition and full cancellation, so counting it would inflate the
        mode count without moving anything that depends on it.
        """
        report = self._report()
        assert report.is_open
        assert report.ground_wave in report.routes
        assert report.ground_wave not in report.interfering_routes
        assert report.multipath().mode_count == len(report.interfering_routes)

    def test_reliability_is_judged_on_the_faded_margin(self):
        from datetime import datetime, timezone

        from propsim.antenna import AntennaSpec
        from propsim.geodesy import GeoPoint
        from propsim.reliability import ReliabilityPredictor
        from propsim.scenario import Scenario, Station
        from propsim.spaceweather import SpaceWeather

        scenario = Scenario(
            Station(GeoPoint(40.4, -3.7), AntennaSpec(height_m=15.0),
                    transmit_power_w=100.0),
            Station(GeoPoint(51.5, -0.1), AntennaSpec(height_m=12.0)),
            datetime(2025, 6, 21, 12, tzinfo=timezone.utc),
            SpaceWeather(f107=140, sunspot_number=60, kp=2),
        )
        lenient = ReliabilityPredictor(scenario, time_availability=0.5).at(10e6)
        strict = ReliabilityPredictor(scenario, time_availability=0.99).at(10e6)
        assert strict.fade_margin_db > lenient.fade_margin_db
        assert strict.reliability <= lenient.reliability
