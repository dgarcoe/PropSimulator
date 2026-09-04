"""Physics invariants: values that are right or wrong independently of us."""

import math
from datetime import datetime, timezone

import pytest

from propsim.constants import (
    ABSORPTION_COEFF,
    GYRO_FREQ_COEFF_HZ,
    KT0_DBW_PER_HZ,
    PLASMA_FREQ_COEFF_HZ,
)
from propsim.geodesy import (
    GeoPoint,
    destination_point,
    great_circle_distance_km,
    initial_bearing_deg,
    intermediate_point,
    path_points,
)
from propsim.ionosphere import (
    build_profile,
    collision_frequency_hz,
    electron_density_from_plasma_frequency,
    plasma_frequency_hz,
    plasma_frequency_mhz,
)
from propsim.magnetic import (
    field_ray_angle_rad,
    geomagnetic_latitude_deg,
    magnetic_field,
)
from propsim.refractive import Mode, refractive_index, refractive_index_squared
from propsim.solar import (
    illuminate_path,
    local_solar_time_hours,
    solar_position,
    solar_zenith_angle_deg,
    subsolar_point,
)
from propsim.spaceweather import SpaceWeather

UTC = timezone.utc


class TestConstants:
    def test_plasma_frequency_coefficient(self):
        assert PLASMA_FREQ_COEFF_HZ == pytest.approx(8.9787, rel=1e-4)

    def test_gyrofrequency_coefficient(self):
        assert GYRO_FREQ_COEFF_HZ == pytest.approx(2.799e10, rel=1e-3)

    def test_absorption_coefficient(self):
        assert ABSORPTION_COEFF == pytest.approx(5.31e-6, rel=1e-2)

    def test_thermal_noise_floor(self):
        assert KT0_DBW_PER_HZ == pytest.approx(-204.0, abs=0.1)


class TestPlasmaFrequency:
    def test_known_value(self):
        # 1e12 m^-3 is the textbook 8.98 MHz.
        assert plasma_frequency_mhz(1e12) == pytest.approx(8.979, rel=1e-3)

    def test_hz_and_mhz_agree(self):
        """The MHz helper must be the Hz helper over 1e6 -- no stray 1000."""
        for density in (1e9, 1e10, 1e11, 1e12):
            assert plasma_frequency_mhz(density) == pytest.approx(
                plasma_frequency_hz(density) / 1e6, rel=1e-12
            )

    def test_round_trip(self):
        for frequency in (1e6, 5e6, 1e7):
            density = electron_density_from_plasma_frequency(frequency)
            assert plasma_frequency_hz(density) == pytest.approx(frequency, rel=1e-9)

    def test_zero_density(self):
        assert plasma_frequency_hz(0.0) == 0.0


class TestCollisionFrequency:
    """Derived from Banks' relation and the standard atmosphere.

    These used to assert against a remembered table of values. That table
    was internally inconsistent -- it implied a factor of ten between 100
    and 110 km and only three over the next ten, which no smooth atmosphere
    does -- and no polynomial of any degree fitted it better than 40%. The
    tests now check the derivation instead of the table.
    """

    @pytest.mark.parametrize("height_km", [50, 60, 70, 80, 90, 100, 110, 120])
    def test_matches_banks_from_the_standard_atmosphere(self, height_km):
        from propsim.atmosphere import (
            BANKS_COEFFICIENT,
            neutral_density,
            temperature,
        )

        expected = (
            BANKS_COEFFICIENT
            * neutral_density(height_km)
            * math.sqrt(temperature(height_km))
        )
        assert collision_frequency_hz(height_km) == pytest.approx(expected, rel=1e-9)

    def test_sits_in_the_published_range_where_absorption_happens(self):
        """Order of magnitude, across the band that carries the loss."""
        assert 1e7 < collision_frequency_hz(60) < 1e8
        assert 1e6 < collision_frequency_hz(80) < 1e7
        assert 1e5 < collision_frequency_hz(90) < 1e6
        assert 1e4 < collision_frequency_hz(110) < 1e5

    def test_falls_monotonically(self):
        values = [collision_frequency_hz(h) for h in range(45, 130, 5)]
        assert all(b < a for a, b in zip(values, values[1:]))

    def test_is_smooth(self):
        """Successive octaves of decay must not jump around: a kink here is
        the signature of a table stitched from inconsistent sources."""
        heights = list(range(55, 125, 5))
        slopes = [
            (math.log(collision_frequency_hz(b)) - math.log(collision_frequency_hz(a)))
            / (b - a)
            for a, b in zip(heights, heights[1:])
        ]
        for a, b in zip(slopes, slopes[1:]):
            assert abs(b - a) < 0.05

    def test_scalar_and_array_forms_agree(self):
        from propsim.atmosphere import collision_frequency_array

        heights = [55.0, 68.0, 83.5, 97.0, 112.0]
        assert collision_frequency_array(heights) == pytest.approx(
            [collision_frequency_hz(h) for h in heights], rel=1e-12
        )


class TestGeodesy:
    def test_known_great_circle(self):
        jfk = GeoPoint(40.6397, -73.7789)
        lhr = GeoPoint(51.4775, -0.4614)
        assert great_circle_distance_km(jfk, lhr) == pytest.approx(5540, rel=5e-3)
        assert initial_bearing_deg(jfk, lhr) == pytest.approx(51.3, abs=0.5)

    def test_destination_is_self_consistent(self):
        origin = GeoPoint(40.0, -3.0)
        for bearing in (0, 45, 90, 180, 270, 359):
            for distance in (10, 500, 5000):
                target = destination_point(origin, bearing, distance)
                assert great_circle_distance_km(origin, target) == pytest.approx(
                    distance, rel=1e-9
                )
                assert initial_bearing_deg(origin, target) % 360 == pytest.approx(
                    bearing % 360, abs=1e-6
                )

    def test_midpoint_is_equidistant(self):
        a, b = GeoPoint(35.0, 139.0), GeoPoint(-33.9, 151.2)
        mid = intermediate_point(a, b, 0.5)
        assert great_circle_distance_km(a, mid) == pytest.approx(
            great_circle_distance_km(mid, b), rel=1e-9
        )

    def test_path_points_span_the_path(self):
        a, b = GeoPoint(40.4, -3.7), GeoPoint(51.5, -0.1)
        points = path_points(a, b, 9)
        assert len(points) == 9
        assert points[0].lat_deg == pytest.approx(a.lat_deg)
        assert points[-1].lat_deg == pytest.approx(b.lat_deg)

    def test_rejects_impossible_latitude(self):
        with pytest.raises(ValueError):
            GeoPoint(91.0, 0.0)


class TestSolarGeometry:
    @pytest.mark.parametrize(
        "when,declination",
        [
            (datetime(2025, 6, 21, 3, tzinfo=UTC), 23.44),
            (datetime(2025, 12, 21, 15, tzinfo=UTC), -23.44),
            (datetime(2025, 3, 20, 9, tzinfo=UTC), 0.0),
        ],
    )
    def test_declination_at_solstices_and_equinox(self, when, declination):
        assert solar_position(when).declination_deg == pytest.approx(declination, abs=0.1)

    def test_sun_is_overhead_at_the_subsolar_point(self):
        for hour in range(0, 24, 3):
            when = datetime(2025, 8, 15, hour, tzinfo=UTC)
            point = subsolar_point(when)
            assert solar_zenith_angle_deg(point, when) == pytest.approx(0.0, abs=1e-6)

    def test_subsolar_longitude_tracks_earth_rotation(self):
        """It must move 15 degrees west per hour, not jump by 180."""
        when = datetime(2025, 3, 20, 12, tzinfo=UTC)
        later = datetime(2025, 3, 20, 18, tzinfo=UTC)
        step = (subsolar_point(when).lon_deg - subsolar_point(later).lon_deg) % 360
        assert step == pytest.approx(90.0, abs=0.5)

    def test_local_solar_noon(self):
        greenwich = GeoPoint(51.5, 0.0)
        when = datetime(2025, 6, 21, 12, tzinfo=UTC)
        assert local_solar_time_hours(greenwich, when) == pytest.approx(12.0, abs=0.2)

    def test_summer_noon_zenith_matches_latitude_minus_declination(self):
        madrid = GeoPoint(40.4, -3.7)
        when = datetime(2025, 6, 21, 12, tzinfo=UTC)
        expected = abs(40.4 - solar_position(when).declination_deg)
        assert solar_zenith_angle_deg(madrid, when) == pytest.approx(expected, abs=1.5)

    def test_terminator_is_detected(self):
        tx, rx = GeoPoint(40.4, -3.7), GeoPoint(35.7, 139.7)
        lit = illuminate_path(tx, rx, datetime(2025, 6, 21, 18, tzinfo=UTC))
        assert lit.crosses_terminator
        assert 0.0 < lit.sunlit_fraction < 1.0

    def test_seasonal_phase_flips_between_hemispheres(self):
        when = datetime(2025, 6, 21, 12, tzinfo=UTC)
        north = illuminate_path(GeoPoint(50, 0), GeoPoint(52, 2), when).seasonal_phase
        south = illuminate_path(GeoPoint(-50, 0), GeoPoint(-52, 2), when).seasonal_phase
        assert north > 0 and south < 0


class TestMagneticField:
    def test_field_strength_is_plausible(self):
        for point in (GeoPoint(40.4, -3.7), GeoPoint(0, 0), GeoPoint(69.6, 18.9)):
            magnitude = magnetic_field(point).magnitude
            assert 2.0e-5 < magnitude < 7.0e-5

    def test_dip_sign_follows_hemisphere(self):
        assert magnetic_field(GeoPoint(60, 0)).inclination_deg > 0
        assert magnetic_field(GeoPoint(-60, 0)).inclination_deg < 0

    def test_field_falls_with_height(self):
        ground = magnetic_field(GeoPoint(40, 0), 0.0).magnitude
        aloft = magnetic_field(GeoPoint(40, 0), 300.0).magnitude
        assert aloft < ground
        assert ground / aloft == pytest.approx((6371 + 300) ** 3 / 6371**3, rel=1e-6)

    def test_geomagnetic_equator_has_zero_dip(self):
        # Find the dip equator near a fixed longitude.
        best = min(
            (abs(geomagnetic_latitude_deg(GeoPoint(lat / 10, -45))), lat / 10)
            for lat in range(-200, 200)
        )
        assert magnetic_field(GeoPoint(best[1], -45)).inclination_deg == pytest.approx(
            0.0, abs=1.0
        )

    def test_reversing_the_path_changes_the_field_ray_angle(self):
        """The regression that matters: a mistyped component name would make
        the dot product constant and this test would fail."""
        field = magnetic_field(GeoPoint(40.4, -3.7), 250.0)
        forward = field_ray_angle_rad(field, 30.0, 12.0)
        reverse = field_ray_angle_rad(field, 210.0, 12.0)
        assert abs(math.degrees(forward - reverse)) > 20.0

    def test_field_ray_angle_spans_the_full_range(self):
        field = magnetic_field(GeoPoint(40.4, -3.7), 250.0)
        angles = [
            math.degrees(field_ray_angle_rad(field, az, el))
            for az in range(0, 360, 30)
            for el in (5, 30, 60)
        ]
        assert max(angles) - min(angles) > 45.0


class TestAppletonHartree:
    def test_reduces_to_isotropic_without_a_field(self):
        for plasma_mhz in (2.0, 5.0, 9.0):
            density = electron_density_from_plasma_frequency(plasma_mhz * 1e6)
            expected = 1.0 - (plasma_mhz / 10.0) ** 2
            for mode in Mode:
                assert refractive_index_squared(density, 10e6, 0.0, 1.0, mode) == (
                    pytest.approx(expected, rel=1e-12)
                )

    @pytest.mark.parametrize("theta_deg", [5, 30, 60, 89, 120, 175])
    def test_ordinary_mode_reflects_at_x_equals_one(self, theta_deg):
        """foF2 is where the O mode turns, whatever the field geometry."""
        density = electron_density_from_plasma_frequency(10e6)
        n2 = refractive_index_squared(
            density, 10e6, 4.7e-5, math.radians(theta_deg), Mode.ORDINARY
        )
        assert n2 == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("theta_deg", [5, 30, 60, 89, 120])
    def test_extraordinary_mode_reflects_at_x_equals_one_minus_y(self, theta_deg):
        frequency, field = 10e6, 4.7e-5
        y = GYRO_FREQ_COEFF_HZ * field / frequency
        density = electron_density_from_plasma_frequency(frequency * math.sqrt(1 - y))
        n2 = refractive_index_squared(
            density, frequency, field, math.radians(theta_deg), Mode.EXTRAORDINARY
        )
        assert n2 == pytest.approx(0.0, abs=1e-4)

    def test_extraordinary_mode_depends_on_the_field_angle(self):
        density = electron_density_from_plasma_frequency(7e6)
        values = [
            refractive_index(density, 10e6, 4.7e-5, math.radians(t), Mode.EXTRAORDINARY)
            for t in (1, 45, 90)
        ]
        assert max(values) - min(values) > 0.02

    def test_index_is_stable_through_the_turning_point(self):
        """No catastrophic cancellation where the ray actually reflects."""
        previous = 1.0
        for plasma_mhz in (9.0, 9.9, 9.99, 9.999, 9.99999):
            density = electron_density_from_plasma_frequency(plasma_mhz * 1e6)
            n = refractive_index(density, 10e6, 4.7e-5, math.radians(45))
            assert 0.0 <= n < previous
            previous = n

    def test_index_falls_with_density(self):
        previous = 1.0
        for plasma_mhz in (1, 3, 5, 7, 9):
            density = electron_density_from_plasma_frequency(plasma_mhz * 1e6)
            n = refractive_index(density, 10e6, 4.7e-5, math.radians(45))
            assert n < previous
            previous = n


class TestIonosphere:
    def _profile(self, hour=12, **weather):
        return build_profile(
            GeoPoint(40.4, -3.7),
            datetime(2025, 6, 21, hour, tzinfo=UTC),
            SpaceWeather(**weather),
            seasonal_phase=0.9,
        )

    def test_profile_peak_sits_at_the_f2_peak_height(self):
        """A Chapman layer must peak where its peak height says it does."""
        for hour in (2, 6, 12, 18):
            profile = self._profile(hour)
            assert profile.peak_height_km == pytest.approx(
                profile.layers.f2.peak_height_km, abs=4.0
            )

    def test_critical_frequencies_are_plausible(self):
        day = self._profile(12, f107=140)
        assert 5.0 < day.layers.fof2_mhz < 20.0
        assert 2.5 < day.layers.e.critical_frequency_mhz < 5.0

    def test_f2_survives_the_night_but_e_collapses(self):
        day, night = self._profile(12, f107=140), self._profile(2, f107=140)
        assert night.layers.fof2_mhz > 0.4 * day.layers.fof2_mhz
        assert night.layers.e.critical_frequency_mhz < 0.3 * day.layers.e.critical_frequency_mhz

    def test_solar_activity_raises_density(self):
        quiet = self._profile(12, f107=70).layers.fof2_mhz
        active = self._profile(12, f107=250).layers.fof2_mhz
        assert active > quiet

    def test_storm_depresses_f2(self):
        calm = self._profile(12, f107=140, kp=1).layers.fof2_mhz
        storm = self._profile(12, f107=140, kp=8).layers.fof2_mhz
        assert storm < calm

    def _flare_profile(self, label):
        return build_profile(
            GeoPoint(40.4, -3.7),
            datetime(2025, 6, 21, 12, tzinfo=UTC),
            SpaceWeather(f107=140).with_flare(label),
            seasonal_phase=0.9,
        )

    def test_xray_flux_reaches_the_d_region(self):
        """The flare must change the D layer through the ionosphere, not
        through a multiplier bolted on to absorption afterwards."""
        quiet = self._profile(12, f107=140)
        flare = self._flare_profile("X1")
        assert flare.density_at(75.0) > 5 * quiet.density_at(75.0)

    def test_a_flare_deposits_deep(self):
        """Hard X-rays penetrate below the quiet D region. The enhancement
        must therefore be much larger at 75 km than at 90 km -- that is the
        whole reason a flare blacks out HF, since the collision frequency at
        75 km is an order of magnitude higher."""
        quiet = self._profile(12, f107=140)
        flare = self._flare_profile("X1")
        deep = flare.density_at(75.0) / quiet.density_at(75.0)
        high = flare.density_at(90.0) / quiet.density_at(90.0)
        assert deep > 2 * high

    def test_flare_density_rises_with_class(self):
        heights = [70.0, 75.0, 80.0]
        previous = None
        for label in ("A1", "C1", "M1", "X1", "X5"):
            profile = self._flare_profile(label)
            total = sum(profile.density_at(h) for h in heights)
            if previous is not None:
                assert total > previous
            previous = total

    def test_the_d_region_follows_the_published_profile(self):
        """It climbs by two orders of magnitude from 70 to 90 km. A Chapman
        layer forced on it is flat across that band and fourteen times too
        low at 85 km, which moves absorption into the wrong region."""
        profile = self._profile(12, f107=100)
        for height, reference in ((70, 1e8), (80, 1e9), (85, 3e9), (90, 1e10)):
            assert profile.density_at(height) == pytest.approx(reference, rel=1.2)

    def test_flare_leaves_f2_alone(self):
        quiet = self._profile(12, f107=140)
        flare = build_profile(
            GeoPoint(40.4, -3.7),
            datetime(2025, 6, 21, 12, tzinfo=UTC),
            SpaceWeather(f107=140).with_flare("X1"),
            seasonal_phase=0.9,
        )
        assert flare.layers.fof2_mhz == pytest.approx(quiet.layers.fof2_mhz, rel=1e-9)

    def test_profile_heights_are_ascending(self):
        heights = self._profile().heights_km
        assert all(b > a for a, b in zip(heights, heights[1:]))

    def test_no_extrapolation_outside_the_model(self):
        profile = self._profile()
        assert profile.density_at(10.0) == 0.0
        assert profile.density_at(2000.0) == 0.0
