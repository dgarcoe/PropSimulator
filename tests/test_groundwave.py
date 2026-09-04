"""The surface route: Sommerfeld, Millington, and what they must reproduce.

The core computes the Sommerfeld attenuation function itself, which needs
the Faddeeva function, which the core builds from numpy alone. SciPy has
one too, and it is used here as an oracle -- as a test dependency, never a
runtime one. A special function written by hand and never checked against
an independent implementation is the kind of thing that is wrong in the
thirteenth digit for years and then wrong in the second.
"""

import cmath
import math
from datetime import datetime, timezone

import pytest

from propsim.antenna import (
    AntennaSpec,
    AntennaType,
    GroundType,
    complex_permittivity,
    surface_impedance,
)
from propsim.constants import SPEED_OF_LIGHT
from propsim.engine import PropagationEngine
from propsim.geodesy import GeoPoint, destination_point
from propsim.groundwave import (
    SHADOW_ONSET_X,
    curvature_loss_db,
    faddeeva,
    ground_wave_loss_db,
    millington_surface_db,
    numerical_distance,
    surface_attenuation_db,
)
from propsim.scenario import Scenario, Station, Weather
from propsim.spaceweather import SpaceWeather
from propsim.surface import path_sections

HF_BAND_HZ = (1.8e6, 3.65e6, 7.1e6, 10.1e6, 14.2e6, 21.2e6, 28.5e6)
GROUNDS = tuple(GroundType)
DISTANCES_KM = (1.0, 5.0, 10.0, 30.0, 56.0, 100.0, 200.0, 400.0, 800.0)


def _exact_attenuation_db(distance_km, frequency_hz, ground):
    """The Sommerfeld attenuation function itself, not an approximation.

    ``A(p) = 1 - j sqrt(pi p) exp(-p) erfc(j sqrt(p))`` with the numerical
    distance complex, rewritten through the Faddeeva function ``w`` --
    ``exp(-p) erfc(j sqrt p) = 2 exp(-p) - w(sqrt p)`` -- because ``erfc``
    of a complex argument overflows long before ``w`` does.
    """
    wofz = pytest.importorskip("scipy.special").wofz
    epsilon = complex_permittivity(frequency_hz, ground)
    wavenumber = 2.0 * math.pi * frequency_hz / SPEED_OF_LIGHT
    p = -1j * wavenumber * distance_km * 1000.0 / (2.0 * epsilon)
    root = cmath.sqrt(p)
    value = 1.0 - 1j * cmath.sqrt(math.pi * p) * (
        2.0 * cmath.exp(-p) - complex(wofz(complex(root)))
    )
    return -20.0 * math.log10(abs(value))


class TestSurfaceAttenuation:
    def test_it_matches_scipy_across_the_whole_band(self):
        """Every HF band, every ground the package knows, 1 to 800 km.

        The core's own Faddeeva function against SciPy's, all the way
        through the physics: if the rational coefficients built at import
        were ever off, the disagreement would show here as decibels rather
        than as a number nobody looks at.
        """
        worst = 0.0
        for frequency in HF_BAND_HZ:
            for ground in GROUNDS:
                for distance in DISTANCES_KM:
                    ours = surface_attenuation_db(distance, frequency, ground)
                    theirs = _exact_attenuation_db(distance, frequency, ground)
                    worst = max(worst, abs(ours - theirs))
        # A millionth of a decibel. The two implementations share no code
        # and no constants, so anything larger would mean one of them is
        # wrong rather than merely rounded differently.
        assert worst < 1e-6, f"worst disagreement {worst:.3e} dB"

    def test_the_faddeeva_function_matches_scipy_in_the_upper_half_plane(self):
        """Directly, over seven decades of magnitude and every angle."""
        import cmath as _cmath

        numpy = pytest.importorskip("numpy")
        wofz = pytest.importorskip("scipy.special").wofz
        rng = numpy.random.default_rng(20260904)
        magnitudes = 10.0 ** rng.uniform(-4.0, 3.0, 2000)
        angles = rng.uniform(0.0, math.pi, 2000)
        worst = 0.0
        for magnitude, angle in zip(magnitudes, angles):
            z = complex(magnitude * _cmath.exp(1j * angle))
            reference = complex(wofz(z))
            worst = max(worst, abs(faddeeva(z) - reference) / abs(reference))
        assert worst < 1e-11, f"worst relative disagreement {worst:.3e}"

    def test_a_vanishing_path_costs_nothing(self):
        assert surface_attenuation_db(0.0, 3.65e6, GroundType.AVERAGE_GROUND) == 0.0

    def test_the_large_distance_limit_is_the_known_asymptote(self):
        """``|A| -> 1/2p`` far out, which is a result about the integral.

        Nothing in the implementation was arranged to make this true, so it
        is a genuine check on the evaluation rather than a restatement of a
        fitted form.
        """
        frequency, ground = 3.65e6, GroundType.AVERAGE_GROUND
        distance = 5000.0
        p, _ = numerical_distance(distance, frequency, ground)
        assert p > 1000.0
        expected = -20.0 * math.log10(1.0 / (2.0 * p))
        assert surface_attenuation_db(distance, frequency, ground) == pytest.approx(
            expected, abs=0.05
        )

    def test_sea_water_beats_every_ground_at_every_frequency(self):
        """The ordering that makes ground wave a maritime technique."""
        for frequency in HF_BAND_HZ:
            sea = surface_attenuation_db(100.0, frequency, GroundType.SALT_WATER)
            for ground in GROUNDS:
                if ground is GroundType.SALT_WATER:
                    continue
                assert sea < surface_attenuation_db(100.0, frequency, ground)

    def test_loss_grows_with_distance_and_with_frequency(self):
        for ground in GROUNDS:
            previous = -1.0
            for distance in DISTANCES_KM:
                value = surface_attenuation_db(distance, 3.65e6, ground)
                assert value > previous
                previous = value
            previous = -1.0
            for frequency in HF_BAND_HZ:
                value = surface_attenuation_db(100.0, frequency, ground)
                assert value > previous
                previous = value

    def test_wet_ground_carries_a_ground_wave_further_than_dry(self):
        """Conductivity, not permittivity, is what the wave pays for."""
        wet = surface_attenuation_db(100.0, 3.65e6, GroundType.WET_GROUND)
        dry = surface_attenuation_db(100.0, 3.65e6, GroundType.DRY_GROUND)
        assert wet < dry - 10.0

    def test_rain_on_the_ground_helps_and_frost_hurts(self):
        base = surface_attenuation_db(100.0, 3.65e6, GroundType.AVERAGE_GROUND, 1.0)
        wet = surface_attenuation_db(100.0, 3.65e6, GroundType.AVERAGE_GROUND, 3.0)
        frozen = surface_attenuation_db(100.0, 3.65e6, GroundType.AVERAGE_GROUND, 0.2)
        assert wet < base < frozen


class TestCurvature:
    def test_it_is_zero_inside_the_onset_and_continuous_through_it(self):
        """No seam: the residue term passes through unity at the onset.

        A cutoff distance chosen by hand would put a step in the answer
        right where the ground wave is most interesting. Solving for where
        the shadow expansion first equals the flat-earth field removes the
        choice and the step together.
        """
        frequency = 3.65e6
        onset_km = _onset_distance_km(frequency)
        assert curvature_loss_db(onset_km * 0.999, frequency) == 0.0
        just_beyond = curvature_loss_db(onset_km * 1.001, frequency)
        assert 0.0 <= just_beyond < 0.05

    def test_the_onset_sits_beyond_the_geometric_horizon(self):
        """Curvature must not start biting before the earth curves away."""
        for frequency in HF_BAND_HZ:
            onset_km = _onset_distance_km(frequency)
            # Horizon for two 10 m antennas over a 4/3 earth.
            horizon_km = 2.0 * math.sqrt(2.0 * (4.0 / 3.0 * 6371.0) * 0.010)
            assert onset_km > horizon_km

    def test_it_depends_only_on_the_normalised_distance(self):
        """Two very different links that are the same diffraction problem.

        ``x = (k a / 2)^(1/3) d / a`` is the only combination the shadow
        depends on, so a frequency raised eightfold and a distance halved
        must give exactly the same curvature loss.
        """
        low, far = 3.55e6, 400.0
        high, near = 3.55e6 * 8.0, 200.0
        assert curvature_loss_db(far, low) == pytest.approx(
            curvature_loss_db(near, high), rel=1e-9
        )

    def test_it_grows_without_bound_beyond_the_horizon(self):
        previous = 0.0
        for distance in (300.0, 500.0, 800.0, 1500.0):
            value = curvature_loss_db(distance, 3.65e6)
            assert value > previous
            previous = value


def _onset_distance_km(frequency_hz: float) -> float:
    """Distance at which the shadow begins, from the same normalisation."""
    from propsim.groundwave import EFFECTIVE_EARTH_RADIUS_KM

    wavenumber_per_km = 2.0 * math.pi * frequency_hz / SPEED_OF_LIGHT * 1000.0
    scale = (
        0.5 * wavenumber_per_km * EFFECTIVE_EARTH_RADIUS_KM
    ) ** (1.0 / 3.0) / EFFECTIVE_EARTH_RADIUS_KM
    return SHADOW_ONSET_X / scale


class TestMillington:
    def test_a_uniform_path_reduces_to_the_homogeneous_answer(self):
        """Exactly, and independently of how finely it was cut up."""
        frequency = 3.65e6
        for ground in GROUNDS:
            direct = surface_attenuation_db(300.0, frequency, ground)
            for pieces in (1, 2, 7, 20):
                sections = [(300.0 / pieces, ground)] * pieces
                assert millington_surface_db(sections, frequency) == pytest.approx(
                    direct, rel=1e-12
                )

    def test_it_is_reciprocal(self):
        """A passive path cannot care which end transmits."""
        frequency = 3.65e6
        sections = [
            (60.0, GroundType.AVERAGE_GROUND),
            (180.0, GroundType.SALT_WATER),
            (40.0, GroundType.DRY_GROUND),
        ]
        forward = millington_surface_db(sections, frequency)
        reverse = millington_surface_db(list(reversed(sections)), frequency)
        assert forward == pytest.approx(reverse, rel=1e-12)

    def test_a_mixed_path_lies_between_its_two_uniform_bounds(self):
        frequency = 3.65e6
        sections = [(100.0, GroundType.AVERAGE_GROUND), (100.0, GroundType.SALT_WATER)]
        mixed = millington_surface_db(sections, frequency)
        all_sea = surface_attenuation_db(200.0, frequency, GroundType.SALT_WATER)
        all_land = surface_attenuation_db(200.0, frequency, GroundType.AVERAGE_GROUND)
        assert all_sea < mixed < all_land

    def test_where_the_sea_sits_on_the_path_changes_the_answer(self):
        """The whole reason Millington exists rather than an average.

        Averaging the ground constants along a path cannot tell a link that
        starts over water from one that ends over it. Millington can, and
        must, because the two are genuinely different -- and the mean of
        the two directions, which is what the method reports, still has to
        differ from the answer for a path with the sea in the middle.
        """
        frequency = 7.1e6
        edges = [
            (100.0, GroundType.SALT_WATER),
            (100.0, GroundType.AVERAGE_GROUND),
            (100.0, GroundType.SALT_WATER),
        ]
        middle = [
            (100.0, GroundType.AVERAGE_GROUND),
            (100.0, GroundType.SALT_WATER),
            (100.0, GroundType.AVERAGE_GROUND),
        ]
        assert millington_surface_db(edges, frequency) < millington_surface_db(
            middle, frequency
        )


class TestGroundWaveLoss:
    def test_the_parts_add_up_to_the_total(self):
        loss = ground_wave_loss_db(500.0, 3.65e6, ground=GroundType.SALT_WATER)
        assert loss.total_db == pytest.approx(loss.surface_db + loss.curvature_db)

    def test_a_negative_loss_is_refused(self):
        from propsim.groundwave import GroundWaveLoss

        with pytest.raises(ValueError, match="cannot be negative"):
            GroundWaveLoss(
                distance_km=100.0, frequency_hz=3.65e6,
                surface_db=-1.0, curvature_db=0.0,
            )

    def test_the_sea_fraction_comes_from_the_sections(self):
        loss = ground_wave_loss_db(
            200.0, 3.65e6,
            sections=[(50.0, GroundType.AVERAGE_GROUND), (150.0, GroundType.SALT_WATER)],
        )
        assert loss.sea_fraction == pytest.approx(0.75)


class TestLaunchGain:
    def test_a_short_vertical_over_ground_gives_the_reference_field(self):
        """300 mV/m at 1 km per kilowatt, the number every chart is drawn to.

        A short monopole over ground is 1.76 dBi in free space plus 3 dB
        for radiating into a hemisphere, and 4.77 dBi is exactly what
        ``sqrt(30 P G) / d = 0.3 V/m`` demands at 1 kW and 1 km. The model
        has to land on it without being told.
        """
        antenna = AntennaSpec(
            antenna_type=AntennaType.SHORT_VERTICAL,
            height_m=0.1, ground=GroundType.SALT_WATER,
            design_frequency_hz=1.9e6, boom_length_m=40.0,
            feedline_loss_db=0.0,
        )
        gain_dbi = antenna.ground_wave_gain_dbi(1.9e6)
        # Recover the ideal radiator by removing the efficiency the model
        # charges a 40 m stick on 160 m.
        ideal = gain_dbi + antenna._efficiency_loss_db(1.9e6)
        assert ideal == pytest.approx(4.77, abs=0.05)
        field = math.sqrt(30.0 * 1000.0 * 10.0 ** (ideal / 10.0)) / 1000.0
        assert field == pytest.approx(0.300, rel=0.01)

    def test_a_horizontal_antenna_is_a_poor_ground_wave_radiator(self):
        """And hopeless over sea, which is where the wave travels furthest."""
        common = dict(height_m=10.0, design_frequency_hz=3.65e6, boom_length_m=20.0)
        for ground, floor_db in (
            (GroundType.AVERAGE_GROUND, 12.0),
            (GroundType.SALT_WATER, 40.0),
        ):
            horizontal = AntennaSpec(
                antenna_type=AntennaType.HORIZONTAL_DIPOLE, ground=ground, **common
            )
            vertical = AntennaSpec(
                antenna_type=AntennaType.VERTICAL_QUARTER_WAVE, ground=ground, **common
            )
            penalty = vertical.ground_wave_gain_dbi(
                3.65e6
            ) - horizontal.ground_wave_gain_dbi(3.65e6)
            assert penalty > floor_db

    def test_it_is_not_the_space_wave_pattern_at_zero_elevation(self):
        """The space wave really does vanish along the surface.

        Both Fresnel coefficients tend to -1 at grazing, so the direct ray
        and its image cancel and ``gain_dbi(0)`` returns a 70 dB null for
        every antenna ever built. The surface wave is a different field
        with a different excitation, and reading the null as its gain would
        delete the ground wave from the model entirely.
        """
        antenna = AntennaSpec(
            antenna_type=AntennaType.VERTICAL_QUARTER_WAVE, height_m=10.0,
            ground=GroundType.SALT_WATER, design_frequency_hz=3.65e6,
            boom_length_m=20.0,
        )
        assert antenna.gain_dbi(0.0, 3.65e6) < -55.0
        assert antenna.ground_wave_gain_dbi(3.65e6) > 0.0

    def test_height_barely_matters_over_sea_and_a_little_over_land(self):
        """Ground-wave coverage is famously indifferent to antenna height."""
        def gain(height_m, ground):
            return AntennaSpec(
                antenna_type=AntennaType.VERTICAL_QUARTER_WAVE, height_m=height_m,
                ground=ground, design_frequency_hz=3.65e6, boom_length_m=20.0,
            ).ground_wave_gain_dbi(3.65e6)

        sea_spread = abs(gain(30.0, GroundType.SALT_WATER)
                         - gain(1.0, GroundType.SALT_WATER))
        land_spread = abs(gain(30.0, GroundType.AVERAGE_GROUND)
                          - gain(1.0, GroundType.AVERAGE_GROUND))
        assert sea_spread < 0.1
        assert 0.1 < land_spread < 4.0


def _short_link(distance_km, bearing_deg=45.0, tx=GeoPoint(40.4, -3.7),
                antenna_type=AntennaType.VERTICAL_QUARTER_WAVE,
                ground=GroundType.AVERAGE_GROUND, design_mhz=3.65):
    antenna = AntennaSpec(
        antenna_type=antenna_type, height_m=10.0, ground=ground,
        design_frequency_hz=design_mhz * 1e6, boom_length_m=20.0,
    )
    return Scenario(
        transmitter=Station(
            location=tx, antenna=antenna, transmit_power_w=1000.0
        ),
        receiver=Station(
            location=destination_point(tx, bearing_deg, distance_km),
            antenna=antenna, transmit_power_w=1000.0,
        ),
        when=datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc),
        space_weather=SpaceWeather(f107=140.0, sunspot_number=80.0, kp=2.0),
        weather=Weather(),
    )


class TestEngineIntegration:
    def test_a_60_km_link_on_80_metres_is_no_longer_nothing(self):
        """The hole this module exists to fill.

        Below the skip distance there is no skywave, and a skywave-only
        model answered a routine local contact with 'no path'. The
        ionosphere still returns nothing here -- that part was right -- but
        the link is comfortably open along the ground.
        """
        report = PropagationEngine(_short_link(56.0)).evaluate(3.65e6)
        assert not report.is_open, "there should be no skywave at 56 km"
        assert report.ground_wave is not None
        assert report.ground_wave.margin_db > 10.0
        assert report.usable
        assert report.best_overall is report.ground_wave

    def test_a_long_path_is_not_won_by_the_ground_wave(self):
        """It is computed there too, and it loses by a hundred decibels."""
        report = PropagationEngine(_short_link(2000.0, design_mhz=14.2)).evaluate(14.2e6)
        assert report.is_open
        assert report.ground_wave.margin_db < report.margin_db - 100.0
        assert report.best_overall is not report.ground_wave

    def test_the_ground_wave_never_touches_the_muf(self):
        """The MUF asks what the ionosphere returns, and nothing else.

        A ground wave exists at every frequency. If it counted towards
        'open' the MUF would pin itself to the top of the search range for
        every path on earth, which is the failure this separation exists to
        prevent.
        """
        engine = PropagationEngine(_short_link(56.0))
        for frequency in (3.65e6, 14.2e6, 28.5e6, 45e6):
            report = engine.evaluate(frequency)
            assert report.ground_wave is not None
            assert report.is_open == bool(report.modes)
        assert engine.maximum_usable_frequency_hz(high_hz=45e6) != 45e6

    def test_the_budget_still_reconstructs_with_a_surface_wave_charged(self):
        report = PropagationEngine(_short_link(100.0)).evaluate(3.65e6)
        budget = report.ground_wave.budget
        budget.verify()
        breakdown = budget.breakdown()
        assert breakdown["surface_wave_loss_db"] == pytest.approx(
            -report.ground_wave.loss.total_db
        )
        assert breakdown["absorption_loss_db"] == 0.0

    def test_a_skywave_mode_is_charged_no_surface_loss(self):
        """The two are alternative routes, never terms of each other."""
        report = PropagationEngine(_short_link(2000.0, design_mhz=14.2)).evaluate(14.2e6)
        for mode in report.modes:
            assert mode.budget.surface_wave_loss_db == 0.0

    def test_the_ground_wave_is_the_early_arrival(self):
        """Along the surface at c, under the ionosphere at more than c."""
        report = PropagationEngine(_short_link(1500.0, design_mhz=7.1)).evaluate(7.1e6)
        assert report.modes
        assert report.ground_wave.group_delay_ms < min(
            mode.group_delay_ms for mode in report.modes
        )

    def test_a_sea_path_carries_much_further_than_a_land_one(self):
        """Same distance, same power, same antennas: only the water differs."""
        coast = GeoPoint(43.4, -8.4)
        sea = PropagationEngine(
            _short_link(300.0, bearing_deg=270.0, tx=coast,
                        ground=GroundType.SALT_WATER)
        ).evaluate(3.65e6)
        land = PropagationEngine(
            _short_link(300.0, bearing_deg=90.0, tx=GeoPoint(50.0, 30.0))
        ).evaluate(3.65e6)
        assert sea.ground_wave.loss.sea_fraction > 0.9
        assert land.ground_wave.loss.sea_fraction < 0.1
        assert sea.ground_wave.loss.total_db < land.ground_wave.loss.total_db - 30.0

    def test_the_path_sections_add_up_to_the_path(self):
        tx, rx = GeoPoint(43.4, -8.4), GeoPoint(43.4, -20.0)
        from propsim.geodesy import great_circle_distance_km

        distance = great_circle_distance_km(tx, rx)
        sections = path_sections(tx, rx, distance, 21)
        assert sum(length for length, _ in sections) == pytest.approx(distance)
