"""PropSimulator: an HF propagation model, solar activity to SNR.

The chain runs: space weather -> Chapman ionosphere -> Appleton-Hartree
refractive index -> Bouguer ray tracing -> absorption -> antennas -> noise
-> link budget -> MUF, LOF and a band report.

Alongside it, and competing with it on the link budget rather than on a
distance rule, runs the ground wave: Sommerfeld surface attenuation over a
Millington land/sea path, with a Fock shadow beyond the horizon.  Below the
skip distance it is the only route there is.

See ``docs/MODEL.md`` for what the model does and does not claim.
"""

from .absorption import AbsorptionResult, absorption_db
from .antenna import AntennaSpec, AntennaType, GroundType
from .engine import (
    FrequencyReport,
    GroundWave,
    Prediction,
    PropagationEngine,
    PropagationMode,
)
from .geodesy import GeoPoint
from .groundwave import GroundWaveLoss, ground_wave_loss_db
from .ionosphere import build_equivalent_column, build_profile
from .link import LinkBudget
from .noise import NoiseEnvironment, noise_budget
from .raytrace import RayMedium, RayPath, trace_ray
from .refractive import Mode
from .scenario import Scenario, Station, Weather
from .spaceweather import SpaceWeather, fetch_space_weather

__version__ = "0.1.0"

__all__ = [
    "AbsorptionResult", "absorption_db", "AntennaSpec", "AntennaType",
    "GroundType", "Prediction", "PropagationEngine", "PropagationMode",
    "FrequencyReport", "GroundWave", "GroundWaveLoss", "ground_wave_loss_db",
    "GeoPoint", "build_equivalent_column", "build_profile",
    "LinkBudget", "NoiseEnvironment", "noise_budget", "RayMedium", "RayPath",
    "trace_ray", "Mode", "Scenario", "Station", "Weather", "SpaceWeather",
    "fetch_space_weather", "__version__",
]
