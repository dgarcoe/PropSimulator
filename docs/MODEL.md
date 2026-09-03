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
| Ionospheric layers | alpha-Chapman D/E/F1/F2 | Chapman shape derived; E from the classical foE relation, D/F1/F2 **empirical** |
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
- F-region absorption within 7% of an independent absorption index (mean over
  12–20 MHz), and foE equal to the classical relation by construction
- every vectorised fast path is asserted bit-identical to the plain one it
  replaced: the batch ray tracer against the single-ray tracer, and the
  array form of Appleton–Hartree against its scalar original
- Madrid–London great circle 5539 km / 51.3° against published values

## Absorption: measured against an independent reference

An earlier version of this document claimed absorption ran "a factor of two
to three" high. That claim was an impression, not a measurement, and it was
wrong. `propsim/reference.py` now supplies an externally-derived absorption
index that the core does not use, and the comparison is quantified.

Two things came out of measuring rather than guessing.

**The E layer was 9.9% high in foE**, exactly and at every solar zenith
angle — a constant ratio, meaning the *shape* was right and only the
amplitude was wrong. It has been replaced with the classical relation

    foE = 0.9 [(180 + 1.44 R12) cos χ]^0.25   MHz

which is an established empirical expression rather than one of our own.
A 9.9% error in foE is 21% in density, and the E region dominates the
absorption integral for any ray that crosses it.

**After that fix, F-region absorption agrees with the reference to within
7% on average** (individual ratios 0.84–1.07 over 12–20 MHz). The
comparison is restricted to F modes deliberately: the reference describes a
ray that *crosses* the absorbing layer, and a ray that turns below 110 km is
a different physical situation, not a disagreement.

### What still differs, and why

| scaling | core | reference |
|---|---|---|
| frequency | −2.45 | −1.68 |
| obliquity (sec i) | +1.28 | +1.00 |
| solar zenith (cos χ) | **+0.47** | **+0.88** |

The frequency and obliquity exponents are close, and the core's are the more
physical of the two: non-deviative absorption goes as 1/f² exactly, steepened
a little because a higher frequency also turns higher, and the reference is
linear in sec *i* only because it is written that way.

The zenith exponent is a genuine structural difference. The core attributes
most of the loss to the E region, whose density follows cos^0.5 χ through the
foE relation; the reference attributes it to the D region, with its cos^0.881
χ law. At large zenith angles (χ > 70°) the core therefore absorbs up to 1.8×
more than the reference.

So: the **total** is cross-validated; the **split between D and E** is not,
and the discrepancy shows up near sunrise and sunset. This is recorded rather
than tuned away, because matching the exponent by scaling constants would
hide which region is actually being modelled wrongly.

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

Relative predictions across frequency, time of day and solar activity are
sound, and F-region absorption is now cross-validated against an independent
model. The remaining known weakness is the D/E attribution near sunrise and
sunset, and the absence of any day-to-day variability model: a single
deterministic SNR overstates how much a real circuit can be relied on.
