"""HTTP API for PropSimulator.

Thin: it validates a request into a :class:`propsim.Scenario` and hands it
to the engine.  No physics lives here, and no defaults are invented for
station parameters -- a request missing them is rejected rather than filled
in, so the API cannot become a second, weaker way of building a scenario.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from propsim.antenna import AntennaSpec, AntennaType, GroundType
from propsim.constants import EARTH_RADIUS_KM
from propsim.coverage import (
    coverage_vs_distance,
    distance_grid,
    usable_band_vs_distance,
)
from propsim.engine import PropagationEngine
from propsim.magnetic import geomagnetic_latitude_deg
from propsim.raytrace import RayMedium, trace_ray
from propsim.refractive import Mode
from propsim.reliability import ReliabilityPredictor
from propsim.solar import local_solar_time_hours, subsolar_point
from propsim.surface import _LAND_POLYGONS
from propsim.geodesy import GeoPoint
from propsim.noise import NoiseEnvironment
from propsim.scenario import Scenario, Station, Weather
from propsim.spaceweather import SpaceWeather, SpaceWeatherError, fetch_space_weather

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="PropSimulator",
    version="0.1.0",
    description="HF skywave propagation: solar activity to signal-to-noise ratio.",
)


class StationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=360)
    name: str = ""
    antenna_type: AntennaType = AntennaType.HORIZONTAL_DIPOLE
    #: METRES. There is no kilometre form of this field anywhere.
    height_m: float = Field(15.0, gt=0.1, le=300)
    ground: GroundType = GroundType.AVERAGE_GROUND
    feedline_loss_db: float = Field(0.5, ge=0, le=20)
    #: Gain the model does not derive (a beam's directivity), declared by
    #: the operator and added to the computed pattern.
    extra_gain_dbi: float = Field(0.0, ge=-20, le=40)
    design_frequency_mhz: float = Field(14.2, gt=0.1, le=60)
    power_w: float = Field(100.0, gt=0.001, le=1e6)
    bandwidth_hz: float = Field(2400.0, ge=1, le=1e6)
    noise_figure_db: float = Field(12.0, ge=0, le=40)
    required_snr_db: float = Field(6.0, ge=-20, le=60)
    noise_environment: NoiseEnvironment = NoiseEnvironment.RURAL

    def to_station(self) -> Station:
        return Station(
            location=GeoPoint(self.lat, self.lon, self.name),
            antenna=AntennaSpec(
                antenna_type=self.antenna_type,
                height_m=self.height_m,
                design_frequency_hz=self.design_frequency_mhz * 1e6,
                ground=self.ground,
                feedline_loss_db=self.feedline_loss_db,
                extra_gain_dbi=self.extra_gain_dbi,
            ),
            transmit_power_w=self.power_w,
            bandwidth_hz=self.bandwidth_hz,
            receiver_noise_figure_db=self.noise_figure_db,
            required_snr_db=self.required_snr_db,
            noise_environment=self.noise_environment,
            name=self.name,
        )


class SpaceWeatherRequest(BaseModel):
    live: bool = False
    f107: Optional[float] = Field(None, ge=60, le=400)
    sunspot_number: Optional[float] = Field(None, ge=0, le=400)
    kp: Optional[float] = Field(None, ge=0, le=9)
    xray_class: Optional[str] = None
    solar_wind_speed_kms: Optional[float] = Field(None, ge=200, le=3000)
    bz_nt: Optional[float] = Field(None, ge=-100, le=100)

    def to_space_weather(self) -> SpaceWeather:
        base = fetch_space_weather() if self.live else SpaceWeather()
        overrides = {
            key: value
            for key, value in (
                ("f107", self.f107),
                ("sunspot_number", self.sunspot_number),
                ("kp", self.kp),
                ("solar_wind_speed_kms", self.solar_wind_speed_kms),
                ("bz_nt", self.bz_nt),
            )
            if value is not None
        }
        if overrides:
            base = dataclasses.replace(
                base, source="manual" if not self.live else f"{base.source}+manual",
                **overrides,
            )
        if self.xray_class:
            base = base.with_flare(self.xray_class)
        return base


class WeatherRequest(BaseModel):
    rain_rate_mm_h: float = Field(0.0, ge=0, le=200)
    ground_moisture_factor: float = Field(1.0, gt=0.01, le=100)
    sea_state: float = Field(0.0, ge=0, le=1)
    freezing: bool = False

    def to_weather(self) -> Weather:
        return Weather(
            rain_rate_mm_h=self.rain_rate_mm_h,
            ground_moisture_factor=self.ground_moisture_factor,
            sea_state=self.sea_state,
            freezing=self.freezing,
        )


class PredictRequest(BaseModel):
    transmitter: StationRequest
    receiver: StationRequest
    when: Optional[datetime] = None
    space_weather: SpaceWeatherRequest = Field(default_factory=SpaceWeatherRequest)
    weather: WeatherRequest = Field(default_factory=WeatherRequest)
    max_hops: int = Field(5, ge=1, le=10)
    fof2_scale: float = Field(1.0, ge=0.2, le=3.0)
    hmf2_offset_km: float = Field(0.0, ge=-150, le=150)
    low_mhz: float = Field(2.0, ge=0.5, le=60)
    high_mhz: float = Field(30.0, ge=1.0, le=60)
    step_mhz: float = Field(0.5, ge=0.05, le=5)
    #: Also compute how often the circuit works, not just whether it works
    #: on the median day.  Costs several extra ionospheres.
    reliability: bool = False
    include_sporadic_e: bool = True
    #: Fraction of the time the circuit must be usable, which sets how much
    #: margin multipath fading is charged.
    time_availability: float = Field(0.9, gt=0.0, lt=1.0)

    def to_scenario(self) -> Scenario:
        when = self.when or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return Scenario(
            transmitter=self.transmitter.to_station(),
            receiver=self.receiver.to_station(),
            when=when,
            space_weather=self.space_weather.to_space_weather(),
            weather=self.weather.to_weather(),
            max_hops=self.max_hops,
            fof2_scale=self.fof2_scale,
            hmf2_offset_km=self.hmf2_offset_km,
        )


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict:
    """Full band prediction for a circuit."""
    if request.high_mhz <= request.low_mhz:
        raise HTTPException(422, "high_mhz must exceed low_mhz")
    try:
        scenario = request.to_scenario()
    except (ValueError, SpaceWeatherError) as exc:
        raise HTTPException(422, str(exc)) from exc

    engine = PropagationEngine(scenario)
    prediction = engine.predict(
        request.low_mhz * 1e6, request.high_mhz * 1e6, request.step_mhz * 1e6
    )
    result = prediction.summary()

    if request.reliability:
        predictor = ReliabilityPredictor(
            scenario,
            include_sporadic_e=request.include_sporadic_e,
            time_availability=request.time_availability,
            # Hand over the engine the band scan already used, so its cached
            # frequency reports serve the median MUF search too.
            median_engine=engine,
        )
        result["reliability"] = {
            "bands": predictor.band_reliability(),
            "time_availability": predictor.time_availability,
            "fof2_spread": {
                "lower_decile": predictor.spread.lower_decile,
                "upper_decile": predictor.spread.upper_decile,
            },
            "sporadic_e": {
                "probability": predictor.sporadic_e_probability,
                "foes_mhz": (
                    predictor.sporadic_e_layer.foes_mhz
                    if predictor.sporadic_e_layer
                    else None
                ),
            },
            "muf_mhz": {
                key: (value / 1e6 if value else None)
                for key, value in predictor.muf_distribution_hz(
                    step_hz=max(request.step_mhz * 1e6, 5e5),
                    hint_hz=prediction.muf_hz,
                ).items()
            },
        }
    return result


@app.post("/api/frequency")
def single_frequency(request: PredictRequest, frequency_mhz: float) -> dict:
    """Every mode at one frequency, with the full link budget for each."""
    try:
        scenario = request.to_scenario()
    except (ValueError, SpaceWeatherError) as exc:
        raise HTTPException(422, str(exc)) from exc

    engine = PropagationEngine(scenario)
    report = engine.evaluate(frequency_mhz * 1e6)
    return {
        "conditions": engine.conditions(),
        "report": report.summary(),
        "modes": [mode.summary() for mode in report.modes],
    }


@app.get("/api/space-weather")
def space_weather() -> dict:
    """Current conditions from NOAA SWPC, with cache and fallback."""
    return fetch_space_weather().summary()


@app.get("/api/health")
def health() -> dict:
    from propsim import __version__

    return {"status": "ok", "version": __version__}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Inline mark, so the page does not 404 on its own icon request."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#2f6f4f"/>'
        '<path d="M4 22C10 8 22 8 28 22" stroke="#fff" stroke-width="2.5" fill="none"/>'
        '<circle cx="6" cy="24" r="2.5" fill="#fff"/>'
        '<circle cx="26" cy="24" r="2.5" fill="#fff"/></svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")



# --------------------------------------------------------------------------
# Dashboard endpoints
#
# Split by cost, not by topic.  The single-link call answers in a fraction of
# a second and drives the readouts and the globe, so it can run on every
# slider movement; the two sweeps take seconds and are fetched separately so
# a slow chart never holds up a fast number.
# --------------------------------------------------------------------------


class DashboardRequest(PredictRequest):
    """A scenario plus the two knobs the dashboard adds to it."""

    frequency_mhz: float = Field(14.2, gt=0.5, le=60)
    #: The launch angle whose ray is drawn.  Independent of the receiver:
    #: the point is to watch where a chosen ray actually goes.
    launch_angle_deg: float = Field(12.0, gt=0.5, lt=89.0)
    magnetoionic_mode: Mode = Mode.ORDINARY
    #: Extra rays drawn around the chosen one, to show the fan.
    secondary_rays: int = Field(9, ge=0, le=24)


def _engine_for(request: DashboardRequest):
    try:
        scenario = request.to_scenario()
    except (ValueError, SpaceWeatherError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return scenario, PropagationEngine(scenario)


def _ray_arc(engine, frequency_hz: float, elevation_deg: float, mode: Mode):
    """Height against fraction of the hop, for one ray."""
    medium = engine._medium(frequency_hz, mode, elevation_deg)
    start_height_km = engine.scenario.transmitter.antenna.height_km
    path = trace_ray(medium, elevation_deg, start_height_km)
    if path.escaped:
        # An escaping ray still has to be drawn, or the display would show
        # nothing at exactly the moment the physics is most interesting.
        # A straight climb along the launch angle is the honest shape.
        arc = [
            [fraction, fraction * 1200.0]
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        return path, arc
    step = max(1, len(path.samples) // 60)
    arc = [[fraction, height] for height, fraction in path.samples[::step]]
    return path, arc


@app.post("/api/link")
def link(request: DashboardRequest) -> dict:
    """Everything the readouts and the globe need. Fast enough to be live."""
    scenario, engine = _engine_for(request)
    frequency_hz = request.frequency_mhz * 1e6
    mode = request.magnetoionic_mode

    path, arc = _ray_arc(engine, frequency_hz, request.launch_angle_deg, mode)
    max_range = engine.max_hop_range_km(frequency_hz, mode)

    fan = []
    if request.secondary_rays:
        span = 18.0
        lowest = max(1.5, request.launch_angle_deg - span / 2.0)
        highest = min(85.0, request.launch_angle_deg + span / 2.0)
        step = (highest - lowest) / max(request.secondary_rays - 1, 1)
        for index in range(request.secondary_rays):
            elevation = lowest + index * step
            if abs(elevation - request.launch_angle_deg) < 1e-6:
                continue
            _, secondary = _ray_arc(engine, frequency_hz, elevation, mode)
            fan.append({"elevation_deg": elevation, "arc": secondary})

    report = engine.evaluate(frequency_hz)
    best = report.best
    profile = engine.column.mean_profile
    local = engine.column.profiles[0]
    layers = local.layers

    link_path = None
    if best is not None:
        step = max(1, len(best.path.samples) // 40)
        link_path = {
            "hops": best.hops,
            "hop_range_km": best.path.ground_range_km,
            "arc": [[fraction, height] for height, fraction in best.path.samples[::step]],
        }

    if best is not None:
        budget = best.budget
        from propsim.link import field_strength_dbuv_per_m

        field = field_strength_dbuv_per_m(
            budget.received_power_dbw, budget.receive_gain_dbi, frequency_hz
        )
        budget_out = {
            "ionospheric_loss_db": best.absorption.total_db * best.hops,
            "ground_loss_db": budget.ground_reflection_loss_db,
            "field_strength_dbuv_m": field,
            "received_power_dbm": budget.received_power_dbw + 30.0,
            "noise_floor_dbm": budget.noise.noise_power_dbw + 30.0,
            "snr_db": budget.snr_db,
            "margin_db": best.margin_db,
            "effective_margin_db": report.effective_margin_db(),
            "fade_margin_db": report.fade_margin_db(),
            "hops": best.hops,
            "launch_elevation_deg": best.launch_elevation_deg,
            "mode": best.mode.value,
        }
    else:
        budget_out = None

    margin = report.effective_margin_db()
    if margin is None:
        status = {"label": "NO PATH", "tone": "bad", "note": "no ray reaches the receiver"}
    elif margin >= 10.0:
        status = {"label": "SOLID LINK", "tone": "good", "note": "comfortable margin"}
    elif margin >= 0.0:
        status = {"label": "WEAK COVERAGE", "tone": "warn", "note": "marginal margin"}
    else:
        status = {"label": "BELOW THRESHOLD", "tone": "bad", "note": "margin is negative"}

    sun = subsolar_point(scenario.when)
    tx, rx = scenario.transmitter.location, scenario.receiver.location
    from propsim.geodesy import path_points

    great_circle = [[p.lat_deg, p.lon_deg] for p in path_points(tx, rx, 64)]

    return {
        "status": status,
        "ionosphere": {
            "fof2_local_mhz": layers.fof2_mhz,
            "fof2_path_mhz": profile.critical_frequency_mhz,
            "hmf2_km": layers.f2.peak_height_km,
            "foe_mhz": layers.e.critical_frequency_mhz,
            "fod_mhz": layers.d.critical_frequency_mhz,
            "tec_tecu": profile.total_electron_content,
        },
        "ray": {
            "state": "escaped" if path.escaped else "reflected",
            "apex_height_km": None if path.escaped else path.apex_height_km,
            "hop_range_km": None if path.escaped else path.ground_range_km,
            "max_range_km": max_range,
            "delay_ms": None if path.escaped else path.group_delay_ms,
            "virtual_height_km": None if path.escaped else path.virtual_height_km,
            "arc": arc,
            "fan": fan,
        },
        "budget": budget_out,
        "link_path": link_path,
        "geometry": {
            "tx": [tx.lat_deg, tx.lon_deg],
            "rx": [rx.lat_deg, rx.lon_deg],
            "distance_km": scenario.distance_km,
            "bearing_deg": scenario.bearing_deg,
            "great_circle": great_circle,
            "subsolar": [sun.lat_deg, sun.lon_deg],
            "sunlit_fraction": engine.illumination.sunlit_fraction,
            "crosses_terminator": engine.illumination.crosses_terminator,
            "local_solar_hour": local_solar_time_hours(tx, scenario.when),
            "geomagnetic_latitude_deg": geomagnetic_latitude_deg(engine.midpoint),
            "gyrofrequency_mhz": engine.magnetic_field.gyrofrequency_hz / 1e6,
        },
        "space_weather": scenario.space_weather.summary(),
        "when": scenario.when.isoformat(),
    }


@app.post("/api/coverage")
def coverage(request: DashboardRequest) -> dict:
    """Field strength and SNR against distance, at the chosen frequency."""
    scenario, engine = _engine_for(request)
    frequency_hz = request.frequency_mhz * 1e6
    reach = max(scenario.distance_km * 1.5, 4000.0)
    samples = coverage_vs_distance(
        engine, frequency_hz, distance_grid(min(reach, 18000.0), 44)
    )
    return {
        "frequency_mhz": request.frequency_mhz,
        "target_distance_km": scenario.distance_km,
        "required_snr_db": scenario.receiver.required_snr_db,
        "samples": [s.summary() for s in samples],
    }


@app.post("/api/band")
def band(request: DashboardRequest) -> dict:
    """MUF and LOF against distance. The slowest call; fetch it separately."""
    scenario, engine = _engine_for(request)
    reach = max(scenario.distance_km * 1.5, 4000.0)
    samples = usable_band_vs_distance(
        engine, distance_grid(min(reach, 18000.0), 16)
    )
    return {
        "target_distance_km": scenario.distance_km,
        "frequency_mhz": request.frequency_mhz,
        "samples": [s.summary() for s in samples],
    }


@app.get("/api/land")
def land() -> dict:
    """The land outlines the globe draws.

    These are the same coarse polygons the surface classifier uses to decide
    whether a ground reflection lands on sea water or on soil.  Drawing a
    prettier basemap would show a coastline the physics does not use; this
    way what you see is what the model reflects off.
    """
    return {
        "polygons": [
            [[lat, lon] for lon, lat in ring]
            for ring in _LAND_POLYGONS.values()
        ],
        "note": "coarse outlines, shared with the surface classifier",
    }
