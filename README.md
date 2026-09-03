# PropSimulator

An HF skywave propagation model that runs the whole chain — solar activity,
ionospheric electron density, refractive index, ray path, absorption, antennas,
noise, link budget — and ends at a signal-to-noise ratio, a MUF, a LOF and a
band recommendation.

```
F10.7 / Kp / X-ray  ->  Chapman D,E,F1,F2  ->  Appleton-Hartree n(h)
                                                      |
        MUF / LOF / bands  <-  SNR  <-  link budget  <-  Bouguer ray path
                                          ^                    |
                                       noise               absorption
```

## Install and run

```bash
pip install -e ".[api,dev]"

# command line
propsim --tx-lat 40.4 --tx-lon -3.7 --tx-name Madrid \
        --rx-lat 51.5 --rx-lon -0.1 --rx-name London \
        --time 2025-06-21T12:00:00 --f107 140 --kp 2

# web interface at http://127.0.0.1:8000
uvicorn api.main:app --reload

# tests
pytest -q
```

## Example

```
Madrid -> London
  1265 km on bearing 11.4 deg

Conditions
  F10.7 140   Kp 2.0   X-ray A1.0   (manual)
  foF2 10.06 MHz   foE 3.89 MHz   hmF2 309 km
  path 100% sunlit   5% over sea

MUF 17.00 MHz
FOT 14.45 MHz (0.85 x MUF, a variability rule of thumb)
LOF 10.00 MHz

  band     MHz     SNR   margin  hops   elev  score
  20 m   14.20    31.9     25.9     1   23.5  0.935
  30 m   10.12    10.0      4.0     2   42.8  0.460
  40 m    7.10   -13.1    -19.1     1    5.9  0.102
  17 m   18.10  closed        -     -      -  0.000
```

## Library

```python
from datetime import datetime, timezone
from propsim import (AntennaSpec, AntennaType, GeoPoint, PropagationEngine,
                     Scenario, SpaceWeather, Station)

tx = Station(GeoPoint(40.4, -3.7, "Madrid"),
             AntennaSpec(AntennaType.HORIZONTAL_DIPOLE, height_m=15.0),
             transmit_power_w=100.0, name="Madrid")
rx = Station(GeoPoint(51.5, -0.1, "London"),
             AntennaSpec(AntennaType.HORIZONTAL_DIPOLE, height_m=12.0), name="London")

engine = PropagationEngine(Scenario(
    tx, rx, datetime(2025, 6, 21, 12, tzinfo=timezone.utc),
    SpaceWeather(f107=140, kp=2)))

prediction = engine.predict()
print(prediction.muf_mhz, prediction.lof_mhz)
print(prediction.band_report()[0])
```

## Design rules that hold throughout

These are the failure modes that survive a green unit-test suite, so they are
enforced structurally rather than by convention. Each has a regression test.

**A receiver is reached only if a ray lands on it.** Reach is decided by
solving for the launch elevation whose hop terminates at the receiver's
distance, then confirming against the traced ray. No hop is computed at one
range and rescaled to another. A ray still at altitude over a receiver
produces no mode — that is what a skip zone is.

**Units live in names, and conversions happen once.** Antenna heights are in
metres everywhere, with `height_km` a read-only derived view. `AntennaSpec`
rejects a height outside 0.1–300 m, so a 20 m mast written as `0.02` fails
loudly instead of becoming 500 m.

**Vectors are dataclasses, not dicts.** The magnetic field exposes
`.east/.north/.up`; a mistyped component is an `AttributeError` at the first
call rather than a silently zero dot product that freezes the extraordinary
mode.

**Every evaluation receives the whole scenario.** MUF, LOF and the band scan
all call one function taking a `Scenario`, so none can run against a station
whose power or bandwidth was never supplied.

**A loss is charged exactly once.** `LinkBudget.verify()` re-adds the terms
and refuses to report a budget whose parts do not reconstruct its total, so a
loss that is displayed but never subtracted — or subtracted twice under two
names — cannot survive. The launch-angle optimiser ranks candidates by the
same total the budget charges.

**One flare mechanism.** X-ray flux raises D-region electron density, and
absorption reads that density. There is no empirical flare multiplier
anywhere, so a blackout cannot be counted twice.

**Interpolation tables are ascending; queries need not be.** Every lookup
goes through the profile's ascending height table, so a tracer descending
from an apex cannot silently get the wrong answer.

**Failures raise.** When the apex search lands past a crossing, the
integrator raises instead of emitting a path length inflated by orders of
magnitude.

## Numerics worth knowing about

- **Turning-point regularisation.** The substitution `r = r_apex − w²` removes
  the inverse-square-root singularity exactly; `dr = −2w dw` and the reversed
  limits cancel, so lengths accumulate as positive sums.
- **Exhaustive apex search.** Between profile nodes `g(r) = n²r² − P²` is a
  cubic, so its stationary points are computed in closed form and sampled
  alongside the nodes. This catches the razor case: a ray at exactly the
  E-layer MUF grazes the peak and `g` dips negative over **tens of metres** —
  invisible to any affordable uniform scan.
- **Cancellation-free Appleton–Hartree.** The O-mode denominator is
  rearranged as `b²/(√(a²+b²)+a)`, exact and stable right through reflection,
  and `sign(u)` is kept when clearing the `1/u` singularity.
- **Group delay, not geometric.** The delay integrates the group index
  `1/n`, so it is the retardation a receiver measures.

## Honesty about limits

See [`docs/MODEL.md`](docs/MODEL.md) for what is derived, what is empirical,
and what is cross-validated. In short: F-region absorption agrees with an
independent absorption index to within 7% over 12–20 MHz, and foE now equals
the classical empirical relation by construction. The known remaining gap is
the split of absorption between the D and E regions, which shows up as a
weaker solar-zenith dependence than the reference near sunrise and sunset.

## Layout

```
propsim/
  constants.py     derived physical constants
  spaceweather.py  NOAA SWPC retrieval, validation, cache, fallback
  geodesy.py       great-circle geometry
  solar.py         solar position, terminator, illumination
  magnetic.py      dipole field, gyrofrequency, field/ray angle
  ionosphere.py    Chapman layers, profiles, equivalent column
  refractive.py    Appleton-Hartree
  raytrace.py      Bouguer tracer, apex search, quadrature
  absorption.py    non-deviative absorption
  antenna.py       image theory, Fresnel ground, efficiency
  surface.py       land/sea classification, reflection loss
  noise.py         thermal, galactic, atmospheric, man-made, auroral
  link.py          self-checking link budget
  scenario.py      Station, Weather, Scenario
  engine.py        orchestration, MUF/LOF, band report
  cli.py
api/main.py        FastAPI
web/index.html     single-page interface
tests/             152 tests
```
