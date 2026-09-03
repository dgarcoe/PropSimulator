# What the model does, and what it does not

PropSimulator computes HF skywave propagation as an unbroken physical chain.
Where a step is empirical rather than derived, this document says so. The
distinction matters: an empirical fit reproduces a trend and can be wrong by
a factor; a derived relation is either right or a bug.

## The chain

| Step | Basis | Status |
|---|---|---|
| Space-weather drivers | NOAA SWPC or manual, range-validated | data |
| Path geometry | great circles on a sphere | derived |
| Solar position | low-precision almanac series from J2000 | derived, <0.01° |
| Ionospheric layers | alpha-Chapman D/E/F1/F2 | **empirical parameters**, Chapman shape derived |
| Refractive index | Appleton–Hartree, collisionless, both modes | derived, exact |
| Magnetic field | tilted dipole aligned to IGRF-2025 | approximation |
| Ray path | Bouguer invariant / Snell / Fermat | derived, exact |
| Turning point | bisection with `r = r_apex − w²` regularisation | derived, exact |
| Absorption | non-deviative, electron–neutral collisions | derived form, **empirical ν(h)** |
| Antenna | image theory + Fresnel ground | derived |
| Noise | thermal + galactic + atmospheric + man-made + auroral | **empirical**, P.372-shaped |
| Link budget | Friis with ionospheric losses | derived |
| MUF / LOF | frequency search over the above | derived from the above |

## Verified against exact solutions

The ray tracer is not checked against itself. For a sharply bounded
reflector the Bouguer integrals have closed form, and the tracer is compared
with that analysis over the full elevation range:

- ground range and geometric path length: agreement within **0.06 %**
- a 300 km mirror is recovered as a **299.8–300.1 km** virtual height

Other externally anchored checks:

- plasma-frequency coefficient 8.9787, gyrofrequency 2.799 × 10¹⁰ Hz/T,
  thermal floor −204 dBW/Hz — all *derived*, then compared with the
  textbook values
- Appleton–Hartree cutoffs land at `X = 1` (O) and `X = 1 − Y` (X) exactly,
  and are independent of the field/ray angle, as theory requires
- horizontal-dipole main lobe at `sin θ = λ/4h` within 1.5°
- atmospheric noise within 3 dB of the ITU-R P.372 mid-latitude curves
- Madrid–London great circle 5539 km / 51.3° against published values

## Known calibration bias

**Absorption runs high**, by roughly a factor of two to three against
published one-hop values, and the excess is concentrated in the E region.
The cause is structural rather than a wiring error: the Chapman E layer used
here keeps appreciable density from 95 km to about 130 km, and a ray at
oblique incidence traverses several hundred kilometres of it. Standard
empirical absorption models are D-region-weighted and calibrated directly to
observation, and they attribute less loss to that traverse.

The consequence is that absolute SNR and margin figures are pessimistic on
the lower bands, while the *relative* ordering across frequency, time of day
and solar conditions — which is what drives band selection — behaves
correctly. This is stated rather than tuned away with a correction factor,
because a fudge factor here would make the numbers look right for the wrong
reason and would hide the real cause.

## Deliberate approximations

- **Spherical Earth, not WGS-84.** The path-length error is a few parts per
  thousand, three orders of magnitude below the ionospheric uncertainty.
- **Equivalent column.** Nine profiles along the great circle are averaged
  at each height, and the ray is traced through that single column. This is
  what makes a radially symmetric solver applicable to a varying ionosphere.
  Absorption escapes the averaging: it reads the *local* profile beneath
  each ray node, so a path crossing the terminator absorbs like a half-lit
  path.
- **Dipole field.** No regional anomalies, the South Atlantic Anomaly
  included.
- **Group index `1/n`.** Exact for an isotropic cold plasma and a good
  approximation at HF, where `Y` is small.

## Not implemented

Named so they are not mistaken for something the model quietly covers:

IRI, a full IGRF expansion, horizontal refraction, three-dimensional ray
tracing, off-great-circle propagation, sporadic E, travelling ionospheric
disturbances, spread F, multipath and fading, Doppler, detailed terrain,
hour-by-hour MUF probability distributions, and the positive/negative phase
structure of geomagnetic storms.

Solar wind speed and Bz are carried and displayed but do **not** enter the
core equations. Their field documentation says so, rather than implying a
coupling that does not exist.

## What the band score is

An operational heuristic combining link margin, headroom below the MUF, and
how well the antenna performs at the required elevation. It is **not** a
probability of contact, and nothing in the package treats it as one.

## Standing on

The model is honest as an educational architecture and as a demonstrator of
HF mechanisms. Its relative predictions are trustworthy; its absolute SNR is
not yet calibrated well enough for operational planning.
