"""Coverage sweeps, the dashboard API and the coastline data.

The web view is driven entirely by these three, so a change that keeps the
physics right but breaks what the page reads is still a break.
"""

import math
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api.main import app
from propsim.antenna import AntennaSpec
from propsim.coastlines import COASTLINES
from propsim.coverage import (
    coverage_vs_distance,
    distance_grid,
    frequency_grid,
    usable_band_vs_distance,
)
from propsim.engine import PropagationEngine
from propsim.geodesy import GeoPoint
from propsim.raytrace import solve_launch_angles
from propsim.refractive import Mode
from propsim.scenario import Scenario, Station
from propsim.spaceweather import SpaceWeather
from propsim.surface import classify_surface

UTC = timezone.utc
WHEN = datetime(2026, 8, 30, 13, tzinfo=UTC)
client = TestClient(app)


def scenario(**overrides):
    fields = {
        "transmitter": Station(GeoPoint(40.71, -74.01, "TX"),
                               AntennaSpec(height_m=15.0), transmit_power_w=100.0),
        "receiver": Station(GeoPoint(55.67, 37.20, "RX"), AntennaSpec(height_m=12.0)),
        "when": WHEN,
        "space_weather": SpaceWeather(f107=140, sunspot_number=95, kp=2.3),
    }
    fields.update(overrides)
    return Scenario(**fields)


def request_body(**overrides):
    body = {
        "transmitter": {"lat": 40.71, "lon": -74.01, "height_m": 15, "power_w": 100},
        "receiver": {"lat": 55.67, "lon": 37.20, "height_m": 12},
        "when": "2026-08-30T13:00:00Z",
        "space_weather": {"f107": 140, "kp": 2.3, "sunspot_number": 95},
        "frequency_mhz": 14.2, "launch_angle_deg": 12.0, "max_hops": 4,
    }
    body.update(overrides)
    return body


class TestCoastlines:
    @pytest.mark.parametrize("name,lat,lon,expect_sea", [
        ("mid-Atlantic", 40, -40, True), ("Pacific", 0, -150, True),
        ("North Sea", 56, 3, True), ("Caribbean", 15, -75, True),
        ("New York", 40.7, -74.0, False), ("Madrid", 40.4, -3.7, False),
        ("Moscow", 55.7, 37.6, False), ("Sahara", 25, 10, False),
        ("Tokyo", 35.7, 139.7, False), ("Sydney", -33.9, 151.2, False),
        ("Amazon", -3, -60, False), ("Cape Town", -33.9, 18.4, False),
        ("Mongolia", 47, 105, False), ("Buenos Aires", -34.6, -58.4, False),
    ])
    def test_land_and_sea_are_placed_correctly(self, name, lat, lon, expect_sea):
        surface = classify_surface(GeoPoint(lat, lon)).value
        assert (surface == "salt_water") is expect_sea, name

    def test_no_ring_crosses_the_antimeridian(self):
        """The point-in-polygon test assumes an edge does not wrap.

        The one legitimate exception is a seam drawn *along* a pole, which
        Antarctica uses to close its band: both ends sit at latitude -90, so
        ray casting never counts it and the globe projects both ends to the
        same point.
        """
        for name, ring in COASTLINES.items():
            for (lon_a, lat_a), (lon_b, lat_b) in zip(ring, ring[1:]):
                if abs(lon_b - lon_a) < 180.0:
                    continue
                assert abs(lat_a) >= 89.0 and abs(lat_b) >= 89.0, (
                    f"{name} wraps the antimeridian away from a pole"
                )

    def test_rings_are_closed_implicitly_and_have_enough_points(self):
        for name, ring in COASTLINES.items():
            assert len(ring) >= 3, name
            assert ring[0] != ring[-1], f"{name} repeats its first point"

    def test_latitudes_are_valid(self):
        for name, ring in COASTLINES.items():
            for lon, lat in ring:
                assert -90 <= lat <= 90 and -180 <= lon <= 180, name


class TestCoverageSweep:
    def test_unreached_distances_are_gaps_not_weak_signals(self):
        engine = PropagationEngine(scenario())
        samples = coverage_vs_distance(engine, 18e6, distance_grid(6000, 20))
        missed = [s for s in samples if not s.reached]
        assert missed, "an 18 MHz path should have a skip zone in range"
        for s in missed:
            assert s.snr_db is None and s.field_strength_dbuv_m is None

    def test_signal_falls_with_distance(self):
        engine = PropagationEngine(scenario())
        reached = [s for s in coverage_vs_distance(engine, 14.2e6, distance_grid(9000, 24))
                   if s.reached]
        assert len(reached) > 6
        assert reached[-1].snr_db < reached[0].snr_db

    def test_the_bracket_test_agrees_with_solving_for_the_angle(self):
        """The MUF pass skips the bisection and reads the bracket straight
        off the cached scan. If the two ever disagree the sweep is drawing a
        different ionosphere from the one the link budget uses."""
        from propsim.coverage import _reaches

        engine = PropagationEngine(scenario())
        for frequency in (5e6, 10e6, 14e6, 20e6, 26e6):
            for distance in (600.0, 1800.0, 3000.0, 5000.0, 7800.0):
                for mode in (Mode.ORDINARY, Mode.EXTRAORDINARY):
                    probe, scan, _ = engine.scan_for(frequency, mode)
                    solved = any(
                        solve_launch_angles(probe, distance / hops, 0.015, 1.0, 60.0,
                                            scan_points=engine.SCAN_POINTS, scan=scan)
                        for hops in range(1, engine.scenario.max_hops + 1)
                        if distance / hops < 20000
                    )
                    assert _reaches(engine, frequency, distance, mode) is solved

    def test_usable_band_brackets_the_working_frequency(self):
        engine = PropagationEngine(scenario())
        samples = usable_band_vs_distance(engine, distance_grid(6000, 8), frequency_grid(points=12))
        assert samples
        for s in samples:
            if s.has_window:
                assert s.lof_mhz < s.muf_mhz

    def test_muf_is_geometric_and_ignores_transmit_power(self):
        weak = PropagationEngine(scenario())
        loud = PropagationEngine(scenario(
            transmitter=Station(GeoPoint(40.71, -74.01), AntennaSpec(height_m=15.0),
                                transmit_power_w=1500.0)))
        distances, frequencies = distance_grid(5000, 6), frequency_grid(points=10)
        a = [s.muf_mhz for s in usable_band_vs_distance(weak, distances, frequencies)]
        b = [s.muf_mhz for s in usable_band_vs_distance(loud, distances, frequencies)]
        assert a == b

    def test_lof_does_not_ignore_transmit_power(self):
        """LOF reads the budget, so unlike the MUF it must move."""
        distances, frequencies = distance_grid(5000, 6), frequency_grid(points=12)
        weak = [s.lof_mhz for s in usable_band_vs_distance(
            PropagationEngine(scenario()), distances, frequencies)]
        loud = [s.lof_mhz for s in usable_band_vs_distance(
            PropagationEngine(scenario(
                transmitter=Station(GeoPoint(40.71, -74.01), AntennaSpec(height_m=15.0),
                                    transmit_power_w=1500.0))), distances, frequencies)]
        pairs = [(a, b) for a, b in zip(weak, loud) if a is not None and b is not None]
        assert pairs
        assert any(b < a for a, b in pairs)


class TestDashboardApi:
    def test_link_returns_every_block_the_page_reads(self):
        d = client.post("/api/link", json=request_body()).json()
        for key in ("status", "ionosphere", "ray", "geometry", "space_weather", "when"):
            assert key in d
        for key in ("fof2_local_mhz", "fof2_path_mhz", "hmf2_km", "foe_mhz",
                    "fod_mhz", "tec_tecu"):
            assert isinstance(d["ionosphere"][key], float)
        assert d["ray"]["arc"] and len(d["ray"]["arc"][0]) == 2

    def test_the_drawn_link_path_lands_on_the_receiver(self):
        """The globe repeats one hop along the bearing, so the hops times the
        hop range has to come out at the path length or the drawn line stops
        short of the marker."""
        d = client.post("/api/link", json=request_body()).json()
        lp = d["link_path"]
        assert lp is not None
        total = lp["hops"] * lp["hop_range_km"]
        assert total == pytest.approx(d["geometry"]["distance_km"], rel=2e-3)

    def test_an_escaping_ray_is_reported_not_hidden(self):
        d = client.post("/api/link", json=request_body(frequency_mhz=29.0,
                                                       launch_angle_deg=45.0)).json()
        assert d["ray"]["state"] == "escaped"
        assert d["ray"]["apex_height_km"] is None
        assert d["ray"]["arc"]                      # still drawable

    def test_launch_angle_moves_the_ray_but_not_the_budget(self):
        low = client.post("/api/link", json=request_body(launch_angle_deg=8.0)).json()
        high = client.post("/api/link", json=request_body(launch_angle_deg=28.0)).json()
        assert high["ray"]["apex_height_km"] > low["ray"]["apex_height_km"]
        assert high["ray"]["hop_range_km"] < low["ray"]["hop_range_km"]
        # The link budget is about the receiver, not about the slider.
        assert high["budget"]["margin_db"] == pytest.approx(low["budget"]["margin_db"])

    def test_status_tracks_the_margin(self):
        """An easy circuit reads good, a hopeless one reads bad, and more
        power never makes the status worse."""
        easy = request_body(
            receiver={"lat": 44.0, "lon": -71.0, "height_m": 12},
            transmitter={"lat": 40.71, "lon": -74.01, "height_m": 15,
                         "power_w": 1000, "extra_gain_dbi": 10},
            frequency_mhz=7.1)
        assert client.post("/api/link", json=easy).json()["status"]["tone"] == "good"

        hopeless = request_body(transmitter={"lat": 40.71, "lon": -74.01,
                                             "height_m": 15, "power_w": 0.01,
                                             "extra_gain_dbi": -20})
        assert client.post("/api/link", json=hopeless).json()["status"]["tone"] == "bad"

        quiet = client.post("/api/link", json=request_body()).json()
        loud = client.post("/api/link", json=request_body(
            transmitter={"lat": 40.71, "lon": -74.01, "height_m": 15,
                         "power_w": 1500, "extra_gain_dbi": 12})).json()
        assert loud["budget"]["margin_db"] > quiet["budget"]["margin_db"]

    def test_coverage_and_band_endpoints(self):
        cov = client.post("/api/coverage", json=request_body()).json()
        assert cov["samples"] and "distance_km" in cov["samples"][0]
        band = client.post("/api/band", json=request_body()).json()
        assert band["samples"] and "muf_mhz" in band["samples"][0]

    def test_land_endpoint_matches_the_physics_polygons(self):
        """The globe must draw the coastline the surface model uses, not a
        decorative one that disagrees with it."""
        polygons = client.get("/api/land").json()["polygons"]
        assert len(polygons) == len(COASTLINES)
        rings = {len(r) for r in polygons}
        assert rings == {len(r) for r in COASTLINES.values()}

    def test_new_controls_reach_the_physics(self):
        base = client.post("/api/link", json=request_body()).json()
        scaled = client.post("/api/link", json=request_body(fof2_scale=1.3)).json()
        shifted = client.post("/api/link", json=request_body(hmf2_offset_km=-60)).json()
        assert scaled["ionosphere"]["fof2_path_mhz"] > base["ionosphere"]["fof2_path_mhz"]
        assert shifted["ionosphere"]["hmf2_km"] < base["ionosphere"]["hmf2_km"]

    def test_invalid_input_is_refused(self):
        for bad in ({"fof2_scale": 9.0}, {"launch_angle_deg": 0.0},
                    {"frequency_mhz": 999.0}, {"max_hops": 0}):
            assert client.post("/api/link", json=request_body(**bad)).status_code == 422
