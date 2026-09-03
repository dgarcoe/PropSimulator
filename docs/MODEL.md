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
| Refractive index | Appleton–Hartree, collisionless, both modes evaluated | derived, exact |
| Sporadic E | Gaussian patch + occurrence climatology | **empirical**, entered as a probability |
| Day-to-day spread | log-normal foF2 about the median | **empirical decile factors** |
| Multipath fading | Rician statistics over the arriving modes | derived, validated |
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

## Reliability, not a point estimate

A prediction that returns one SNR is answering a question nobody has. The
ionosphere does not repeat itself: at a given hour of a given month, foF2
scatters around its monthly median by tens of percent, and two circuits with
identical median SNR can differ completely in how often they actually work.

`propsim.variability` treats foF2 as log-normally distributed about the
median and characterises it by decile factors — the multipliers exceeded on
10% and 90% of days. The spread is narrowest at mid-latitudes in daylight
(about ±15%), and widens towards the equatorial anomaly, towards the auroral
oval, at night, and during a storm. `propsim.reliability` evaluates the whole
chain once per sampled quantile and reports the fraction of days the circuit
closes.

Reliability is estimated by locating where the margin crosses zero between
bracketing quantiles. Two deliberate refusals:

* **No normal fit.** It would extrapolate past the sampled range and report,
  say, 3% for a circuit that failed at every quantile examined.
* **No assumed monotonicity.** Near the critical frequency a *worse*
  ionosphere can give a *better* margin, because the set of available modes
  changes discontinuously and the ray is forced onto a different, less
  absorbed path. Where the samples are not monotonic the estimate falls back
  to the probability mass of the quantiles that close — coarser, but not a
  fiction.

The MUF is reported the same way, as a lower decile / median / upper decile
spread. It is typically ±3 MHz wide, which is why quoting a MUF to two
decimal places overstates what is known.

## Sporadic E

Sporadic E is a patch of intense ionisation a kilometre or two thick near
105 km, formed by wind-shear convergence of metallic ions. It has almost
nothing to do with solar production, and it can carry frequencies far above
anything the regular E layer supports.

It is modelled as a **probability, never as a state**. There is no fact of
the matter about whether a patch is present on a circuit at a given hour;
there is an occurrence rate and a distribution of foEs. Reliability composes
the two branches:

    P(Es) × reliability_with_Es + (1 − P(Es)) × reliability_without

Three distinct populations are represented, because one formula cannot
describe sporadic E everywhere: the mid-latitude wind-shear population with
its strong summer maximum and twin morning/evening peaks; a weakly seasonal
equatorial population tied to the electrojet; and an auroral population
driven by particle precipitation and rising with Kp.

Two details that decide whether the model works at all:

* The patch is **Gaussian, not Chapman.** Chapman describes a layer in
  photochemical equilibrium with overhead radiation; sporadic E is a
  compressed cloud of long-lived metallic ions with no such balance, and is
  very nearly symmetric about its peak.
* The height grid is **refined around the patch** to 0.25 km. On the plain
  2 km grid a 1 km-thick layer falls between samples, and the ray passes
  straight through the gap as though nothing were there.

Screening emerges from the ray tracing rather than being coded in: a patch
can *cost* a band its reliability by stealing a ray that would otherwise
have taken a better F-layer path. On one summer mid-latitude circuit, 10 m
goes from dead to 36% reliable while 30 m falls from 100% to 64%.

## Fading: variation within a day

Day-to-day variability is only half the story. Several modes normally reach
the receiver at once — a one-hop and a two-hop, a low ray and a high ray, an
ordinary and an extraordinary component — with slightly different path
lengths. Their relative phases drift as the reflection heights move, so the
resultant amplitude fades. A deterministic link budget reports the **mean**
power and therefore overstates how much of the time a signal is usable.

`propsim.fading` characterises the arriving modes by a Rician K factor (the
dominant mode's power over the sum of the rest) and reports the depth below
the mean exceeded for a given time availability. Reliability is judged on
the margin *after* that fade is paid for, not on the mean-power margin.

The effect is not small. On the reference circuit at 10 MHz the mean margin
is 8.8 dB and the 90%-of-the-time margin is 1.6 dB: a link the budget calls
comfortable is in fact marginal.

Delay spread is computed from the same modes and compared with the signal
bandwidth. Where the coherence bandwidth `1/(2π τ_rms)` falls below the
occupied bandwidth the channel fades **selectively** rather than flat, which
distorts the signal and cannot be cured with more power. A 2.4 kHz voice
channel on two modes 0.6 ms apart is selective — which is why HF voice on a
multipath path sounds the way it does.

Validated against the textbook Rayleigh figures (9.77 dB at 90%, 19.98 dB at
99%, against ~9.6 and ~20), against the large-K Gaussian limit
`1 − 1.2816 √(2/K)`, and the Bessel `I₀` against SciPy to 1e-9 — SciPy being
a test-only oracle, since the package itself depends on numpy alone.

## Not implemented

Named so they are not mistaken for something the model quietly covers:

IRI, a full IGRF expansion, horizontal refraction, three-dimensional ray
tracing, off-great-circle propagation, travelling ionospheric disturbances,
spread F, Doppler, detailed terrain, and the positive/negative phase
structure of geomagnetic storms.

Solar wind speed and Bz are carried and displayed but do **not** enter the
core equations. Their field documentation says so, rather than implying a
coupling that does not exist.

## What the band score is

An operational heuristic combining link margin, headroom below the MUF, and
how well the antenna performs at the required elevation. It is **not** a
probability of contact, and nothing in the package treats it as one.

## The gap that remains open

The ray tracer is validated against closed-form analysis, absorption against
an independent index, fading against Rayleigh and Gaussian limits, noise
against P.372, and the antenna against image theory. **The ionosphere itself
is validated against nothing.** Its layer parameterisation for D, F1 and F2
has never been compared with IRI or with ionosonde measurements, so its
error is not large or small — it is unknown, which is worse.

`scripts/validate_against_ionosondes.py` closes that gap: it pulls foF2, foE
and hmF2 from the GIRO/DIDBase archive, runs the model for the same place
and hour, and reports the distribution of modelled/observed.

It has **never been run against the live service.** The environment it was
written in blocks outbound HTTPS by policy, so its request shapes and
response parsing come from the documented interface and are unconfirmed.
Its offline parts — the model lookups, the ratio and summary statistics, the
dry-run URL construction — are exercised; the network path is not. Treat the
first real run as part of the work rather than as a finished result.

## Standing on

Relative predictions across frequency, time of day and solar activity are
sound; F-region absorption is cross-validated against an independent model;
and the output is now a reliability rather than a point estimate, which is
the form an operational answer has to take.

The known remaining weaknesses, in order: the layer parameterisation for D,
F1 and F2 has never been checked against observation, so its error is
unknown; the split of absorption between the D and E regions is uncalibrated
and diverges near sunrise and sunset; and the equivalent column cannot
produce horizontal refraction or off-great-circle paths at all.
