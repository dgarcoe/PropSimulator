"""Space-weather inputs: validated container plus NOAA SWPC retrieval.

The container is a frozen dataclass with range validation, so every driver
that the ionosphere needs has exactly one representation and cannot be
half-populated.  In particular ``xray_flux_wm2`` is a first-class field: it
is carried into the D-region builder by the same object that carries F10.7,
never by a separate optional argument that a caller can forget to pass.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

__all__ = [
    "SpaceWeather",
    "SpaceWeatherError",
    "fetch_space_weather",
    "XRAY_CLASS_FLUX",
    "xray_class_to_flux",
    "flux_to_xray_class",
]


class SpaceWeatherError(ValueError):
    """Raised when space-weather input is missing, malformed or unusable."""


#: Background flux of a quiet sun (A-class), W/m^2 in the 0.1-0.8 nm band.
QUIET_XRAY_FLUX = 1e-8

XRAY_CLASS_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def xray_class_to_flux(label: str) -> float:
    """Convert a GOES class label such as ``"M2.5"`` to W/m^2."""
    text = str(label).strip().upper()
    if not text:
        raise SpaceWeatherError("empty X-ray class label")
    letter, rest = text[0], text[1:]
    if letter not in XRAY_CLASS_FLUX:
        raise SpaceWeatherError(f"unknown X-ray class letter {letter!r}")
    try:
        multiplier = float(rest) if rest else 1.0
    except ValueError as exc:
        raise SpaceWeatherError(f"bad X-ray class magnitude in {label!r}") from exc
    if multiplier <= 0:
        raise SpaceWeatherError(f"non-positive X-ray magnitude in {label!r}")
    return XRAY_CLASS_FLUX[letter] * multiplier


def flux_to_xray_class(flux_wm2: float) -> str:
    """Inverse of :func:`xray_class_to_flux`, for display."""
    if flux_wm2 < XRAY_CLASS_FLUX["A"]:
        return "<A1.0"
    for letter in ("X", "M", "C", "B", "A"):
        base = XRAY_CLASS_FLUX[letter]
        if flux_wm2 >= base:
            return f"{letter}{flux_wm2 / base:.1f}"
    return "<A1.0"


def _check_range(name: str, value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SpaceWeatherError(f"{name} is not a number: {value!r}") from exc
    if math.isnan(number) or math.isinf(number):
        raise SpaceWeatherError(f"{name} is not finite: {value!r}")
    if not low <= number <= high:
        raise SpaceWeatherError(
            f"{name}={number:g} outside the physically meaningful range "
            f"[{low:g}, {high:g}]"
        )
    return number


@dataclass(frozen=True)
class SpaceWeather:
    """Solar and geomagnetic drivers of the ionosphere.

    Attributes
    ----------
    f107:
        10.7 cm radio flux, solar flux units.  Drives the overall ionisation
        level of every layer.
    sunspot_number:
        International sunspot number.  Used where a model is historically
        expressed in R12 rather than F10.7.
    kp:
        Planetary K index, 0-9.  Drives the storm-time F2 depression and the
        auroral D-region enhancement.
    xray_flux_wm2:
        GOES 0.1-0.8 nm flux.  Drives D-region electron density directly;
        this is the *only* flare mechanism in the model, so absorption is
        never multiplied by a second empirical flare factor on top of it.
    solar_wind_speed_kms, bz_nt:
        Carried for display and for the storm-onset heuristic only.  They do
        not enter the core equations, and the field docs say so rather than
        implying a coupling that does not exist.
    source:
        Free-form provenance string ("manual", "noaa-swpc", "cache", ...).
    """

    f107: float = 100.0
    sunspot_number: float = 50.0
    kp: float = 2.0
    xray_flux_wm2: float = QUIET_XRAY_FLUX
    solar_wind_speed_kms: float = 400.0
    bz_nt: float = 0.0
    source: str = "manual"

    def __post_init__(self) -> None:
        object.__setattr__(self, "f107", _check_range("f107", self.f107, 60.0, 400.0))
        object.__setattr__(
            self,
            "sunspot_number",
            _check_range("sunspot_number", self.sunspot_number, 0.0, 400.0),
        )
        object.__setattr__(self, "kp", _check_range("kp", self.kp, 0.0, 9.0))
        object.__setattr__(
            self,
            "xray_flux_wm2",
            _check_range("xray_flux_wm2", self.xray_flux_wm2, 1e-9, 1e-2),
        )
        object.__setattr__(
            self,
            "solar_wind_speed_kms",
            _check_range("solar_wind_speed_kms", self.solar_wind_speed_kms, 200.0, 3000.0),
        )
        object.__setattr__(self, "bz_nt", _check_range("bz_nt", self.bz_nt, -100.0, 100.0))
        if not str(self.source).strip():
            raise SpaceWeatherError("source must be a non-empty label")

    # -- derived, read-only views ----------------------------------------
    @property
    def xray_class(self) -> str:
        return flux_to_xray_class(self.xray_flux_wm2)

    @property
    def is_flare(self) -> bool:
        """True from M-class up, where D-region absorption becomes dramatic."""
        return self.xray_flux_wm2 >= XRAY_CLASS_FLUX["M"]

    @property
    def is_storm(self) -> bool:
        return self.kp >= 5.0

    def with_flare(self, label: str) -> "SpaceWeather":
        """Return a copy at the given GOES class, e.g. ``sw.with_flare("X1")``."""
        return replace(self, xray_flux_wm2=xray_class_to_flux(label))

    def summary(self) -> dict:
        return {
            "f107": self.f107,
            "sunspot_number": self.sunspot_number,
            "kp": self.kp,
            "xray_flux_wm2": self.xray_flux_wm2,
            "xray_class": self.xray_class,
            "solar_wind_speed_kms": self.solar_wind_speed_kms,
            "bz_nt": self.bz_nt,
            "is_flare": self.is_flare,
            "is_storm": self.is_storm,
            "source": self.source,
        }


# --------------------------------------------------------------------------
# NOAA SWPC retrieval
# --------------------------------------------------------------------------

NOAA_F107_URLS = (
    "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
    "https://services.swpc.noaa.gov/products/summary/10cm-flux.json",
)
NOAA_KP_URLS = (
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
)
NOAA_XRAY_URLS = (
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json",
    "https://services.swpc.noaa.gov/products/summary/xray-flux.json",
)
NOAA_SSN_URLS = ("https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json",)
NOAA_WIND_URLS = ("https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json",)
NOAA_MAG_URLS = ("https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json",)


class _Cache:
    """Tiny in-process TTL cache; the last good value survives an outage."""

    def __init__(self, ttl_seconds: float = 900.0) -> None:
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, allow_stale: bool = False) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        stamp, value = entry
        if allow_stale or (time.time() - stamp) < self.ttl:
            return value
        return None

    def put(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        self._store.clear()


_CACHE = _Cache()


def _http_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "PropSimulator/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise SpaceWeatherError(f"HTTP {response.status} from {url}")
        return json.loads(response.read().decode("utf-8"))


def _fetch_first(
    urls: Sequence[str],
    parser,
    timeout: float,
    retries: int,
    backoff: float,
) -> Optional[float]:
    """Try each URL in turn, retrying transient failures with backoff."""
    for url in urls:
        delay = backoff
        for attempt in range(retries + 1):
            try:
                return parser(_http_json(url, timeout))
            except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2.0
            except (KeyError, IndexError, TypeError, ValueError):
                break  # malformed payload: another retry will not help
    return None


def _parse_f107(payload: Any) -> float:
    if isinstance(payload, dict) and "Flux" in payload:
        return float(payload["Flux"])
    latest = payload[-1]
    for key in ("flux", "f10.7", "observed_flux", "Flux"):
        if key in latest:
            return float(latest[key])
    raise KeyError("no F10.7 field")


def _parse_kp(payload: Any) -> float:
    if payload and isinstance(payload[0], list):  # header row + data rows
        header, *rows = payload
        column = header.index("Kp") if "Kp" in header else len(header) - 1
        return float(rows[-1][column])
    latest = payload[-1]
    for key in ("kp_index", "kp", "estimated_kp"):
        if key in latest:
            return float(latest[key])
    raise KeyError("no Kp field")


def _parse_xray(payload: Any) -> float:
    if isinstance(payload, dict) and "Flux" in payload:
        return float(payload["Flux"])
    long_band = [
        row
        for row in payload
        if str(row.get("energy", "")).startswith("0.1-0.8")
    ]
    latest = (long_band or payload)[-1]
    return float(latest["flux"])


def _parse_ssn(payload: Any) -> float:
    return float(payload[-1]["ssn"])


def _parse_wind_speed(payload: Any) -> float:
    header, *rows = payload
    column = header.index("speed")
    for row in reversed(rows):
        if row[column] is not None:
            return float(row[column])
    raise ValueError("no valid wind speed")


def _parse_bz(payload: Any) -> float:
    header, *rows = payload
    column = header.index("bz_gsm")
    for row in reversed(rows):
        if row[column] is not None:
            return float(row[column])
    raise ValueError("no valid Bz")


def fetch_space_weather(
    timeout: float = 8.0,
    retries: int = 2,
    backoff: float = 1.0,
    fallback: Optional[SpaceWeather] = None,
    use_cache: bool = True,
) -> SpaceWeather:
    """Fetch current conditions from NOAA SWPC.

    Every product is fetched independently with its own alternate URL, so a
    single dead endpoint degrades one driver instead of the whole set.  Any
    driver that cannot be retrieved falls back to the cached value, then to
    ``fallback``, then to the dataclass default.  The returned object is
    always fully validated -- a partially-fetched result is never handed to
    the ionosphere.
    """
    base = fallback or SpaceWeather()
    if use_cache:
        cached = _CACHE.get("space_weather")
        if cached is not None:
            return cached

    def pick(key: str, urls, parser, default: float) -> float:
        value = _fetch_first(urls, parser, timeout, retries, backoff)
        if value is None:
            stale = _CACHE.get(key, allow_stale=True)
            return stale if stale is not None else default
        _CACHE.put(key, value)
        return value

    fields: dict[str, Any] = {
        "f107": pick("f107", NOAA_F107_URLS, _parse_f107, base.f107),
        "kp": pick("kp", NOAA_KP_URLS, _parse_kp, base.kp),
        "xray_flux_wm2": pick("xray", NOAA_XRAY_URLS, _parse_xray, base.xray_flux_wm2),
        "sunspot_number": pick("ssn", NOAA_SSN_URLS, _parse_ssn, base.sunspot_number),
        "solar_wind_speed_kms": pick(
            "wind", NOAA_WIND_URLS, _parse_wind_speed, base.solar_wind_speed_kms
        ),
        "bz_nt": pick("bz", NOAA_MAG_URLS, _parse_bz, base.bz_nt),
        "source": "noaa-swpc",
    }
    try:
        weather = SpaceWeather(**fields)
    except SpaceWeatherError:
        # A live value out of range is worse than no live value at all.
        clipped = {
            "f107": min(max(fields["f107"], 60.0), 400.0),
            "kp": min(max(fields["kp"], 0.0), 9.0),
            "xray_flux_wm2": min(max(fields["xray_flux_wm2"], 1e-9), 1e-2),
            "sunspot_number": min(max(fields["sunspot_number"], 0.0), 400.0),
            "solar_wind_speed_kms": min(max(fields["solar_wind_speed_kms"], 200.0), 3000.0),
            "bz_nt": min(max(fields["bz_nt"], -100.0), 100.0),
            "source": "noaa-swpc (clipped)",
        }
        weather = SpaceWeather(**clipped)
    _CACHE.put("space_weather", weather)
    return weather
