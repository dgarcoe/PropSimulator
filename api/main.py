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
from propsim.engine import PropagationEngine
from propsim.reliability import ReliabilityPredictor
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
