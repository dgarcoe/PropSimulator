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
| Ionospheric layers | alpha-Chapman E/F1/F2, exponential-plus-ledge D | Chapman shape derived; E from the classical foE relation, D/F1/F2 **empirical** |
| Refractive index | Appleton–Hartree, collisionless, both modes evaluated | derived, exact |
| Sporadic E | Gaussian patch + occurrence climatology | **empirical**, entered as a probability |
| Day-to-day spread | log-normal foF2 about the median | **empirical decile factors** |
| Multipath fading | Rician statistics over the arriving modes | derived, validated |
| Magnetic field | tilted dipole aligned to IGRF-2025 | approximation |
| Ray path | Bouguer invariant / Snell / Fermat | derived, exact |
| Turning point | bisection with `r = r_apex − w²` regularisation | derived, exact |
| Absorption | non-deviative, electron–neutral collisions | derived form; ν(h) from Banks over the US Standard Atmosphere |
| Ground wave | Sommerfeld attenuation + Fock shadow + Millington | derived, exact where stated below |
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
- the Sommerfeld ground-wave attenuation function, computed from a
  Faddeeva function the package builds itself out of numpy, agrees with
  SciPy's to better than 10⁻⁶ dB across every HF band, every ground
  constant and 1–800 km
- a short vertical monopole over ground comes out at 4.77 dBi, which is the
  300 mV/m at 1 km per kilowatt every ground-wave chart is drawn against —
  derived from image theory, not entered
- horizontal-dipole main lobe at `sin θ = λ/4h` within 1.5°
- atmospheric noise within 3 dB of the ITU-R P.372 mid-latitude curves
- F-region absorption within 6% of an independent absorption index (mean over
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

**F-region absorption now agrees with the reference to within 6% on
average** (mean ratio 0.94, individual ratios 0.92–0.98 over 12–20 MHz).
The comparison is restricted to F modes deliberately: the reference
describes a ray that *crosses* the absorbing layer, and a ray that turns
below 110 km is a different physical situation, not a disagreement.

### Where the loss is charged, and why that changed

Getting the total right is not the same as getting the physics right, and
for a while this model had the first without the second. It attributed
about 5% of the loss to the D region and 93% to the E region — a split that
is backwards. Two things were wrong beneath it:

- **The D-layer profile was a Chapman layer.** The real D region is not
  one. Its density climbs roughly exponentially with a scale height of a
  few kilometres, with a ledge near 60 km where the chemistry changes and a
  flare-driven enhancement around 75 km. The model was 14× too low at 85 km
  and *flat* over the stretch where the real profile rises two orders of
  magnitude. `DRegionLayer` replaces it with the exponential-plus-ledge
  form and now matches published daytime densities to about 2×.
- **The collision frequency was a remembered table.** It implied a factor
  of ten between 100 and 110 km and only three over the next ten
  kilometres, which no smooth atmosphere does. `propsim/atmosphere.py`
  derives ν(h) instead, from Banks' relation `ν = 5.4e-16 n √T` over the US
  Standard Atmosphere — one interpolation of two tabulated atmospheric
  quantities rather than a curve remembered from a figure.

The split is now roughly **50% D, 47% E** across 12–20 MHz, which is the
right order, and the total moved from 0.93 to 0.94 of the reference in the
process — the corrections were to *where* the loss is, not to how much.

### What still differs, and why

| scaling | core | reference |
|---|---|---|
| frequency | −1.99 | −1.68 |
| obliquity (sec i) | +1.15 | +1.00 |
| solar zenith (cos χ) | **+0.62** | **+0.88** |

The frequency and obliquity exponents are close, and the core's are the
more physical of the two: non-deviative absorption goes as 1/f² exactly —
which is what −1.99 says — and the reference is linear in sec *i* only
because it is written that way.

The zenith exponent is a genuine residual difference: +0.62 against +0.88,
so at large zenith angles (χ > 70°) the core absorbs somewhat more than the
reference. It closed from +0.47 as a *consequence* of fixing the D-region
profile, which is the right way for it to move — nothing was fitted to it.
The gap that remains is recorded rather than tuned away, because matching
an exponent by scaling constants hides which region is being modelled
wrongly instead of showing it.

## The ground wave

Every other part of this package models skywave. Below the skip distance
there is no skywave, and a skywave-only model answers a 56 km link on 80
metres with "no path" — which is not a conservative answer but a wrong one.
`propsim/groundwave.py` fills that hole, and three things go into it.

**Surface dissipation.** Sommerfeld's flat-earth attenuation function
`A(p)`, evaluated rather than fitted. The usual rational approximations
(Norton's, Terman's) are good to about a decibel over sea water and average
ground, and drift to 4.6 dB over fresh water, where the numerical distance
turns nearly pure imaginary and the fit's phase-correction term overshoots.
The exact function needs the Faddeeva function `w(z)`, which the core
builds itself: Weideman's rational approximation, whose coefficients are
computed at import by an FFT rather than transcribed from a table. It
agrees with SciPy's to one part in 10¹³, and the whole attenuation function
to better than 10⁻⁶ dB.

The evaluation is arranged so nothing large cancels. Written directly,
`A(p) = 1 − j√(πp)·exp(−p)·erfc(j√p)` subtracts two enormous nearly equal
numbers over any lossy ground; using `w(−z) = 2exp(−z²) − w(z)` collapses
the bracket to a single `w(−√p)`, and since `arg p` always lies in
(−90°, 0°), `−√p` always lands in the upper half plane where `w` is bounded.

**Earth curvature.** Beyond the horizon the surface falls away from the
wavefront. The leading term of the Fock residue series supplies the shadow,
and it is joined to the flat-earth result **without a seam**: a
single-residue expansion exceeds unity at small normalised distance and
falls below it beyond, so the crossing is where the shadow begins. That
crossing is solved for (x ≈ 1.72, which at 3.65 MHz is 213 km — comfortably
outside the geometric horizon) rather than chosen, and the curvature factor
is therefore continuous at 1.0 by construction.

This term is evaluated in the **good-conductor limit** of the residue
series (the boundary condition `w'(t) = 0`, leading root
`1.01879 exp(iπ/3)`). That is close to right over sea water, which is where
a ground wave travels far enough for curvature to matter. Over land the
true root moves and the shadow decays up to 2.3× faster — but over land the
flat-earth term has already spent 50 dB before the horizon, so the path is
dead long before the difference could show. The model is accurate where the
ground wave lives and optimistic where it does not, by a factor named here
rather than hidden.

**Mixed land and sea.** Millington's method, over the same coastline
polygons the globe is drawn from. A sea path that ends with 200 km of land
is not the same link as one that starts with it, and averaging the ground
constants along the path loses that asymmetry completely. Millington's
forward-and-reverse mean restores reciprocity — which a passive path must
have — and a path of one ground reduces to the homogeneous answer exactly,
independently of how finely it was sampled.

**Polarisation and height** live on the antenna, not the path. The ground
wave is vertically polarised; its front tilts forward by the surface
impedance Δ, and that tilt is the only horizontal electric field there is,
so a horizontal antenna couples down by |Δ| — about 15 dB over average
ground and 44 dB over sea. That is an upper bound, since it assumes the
wire has a component in the plane of propagation. The field also varies as
`1 + jkhΔ` in the first tens of metres, which at HF over sea is nothing at
all — the reason ground-wave coverage is famously indifferent to antenna
height.

Note that this is emphatically **not** `gain_dbi(0°)`. Both Fresnel
coefficients tend to −1 at grazing, so the direct ray and its image cancel
and the over-ground pattern collapses to a 60 dB null for every antenna
ever built. The space wave really does vanish along the surface; what
survives is a different field with a different excitation, and reading the
null as its gain would delete the ground wave from the model entirely.

### How the two routes are kept apart

The ground wave is **not** a `PropagationMode`. It has no launch angle, no
hop count, no apex, no magnetoionic splitting and no ionospheric
absorption, and a class carrying all of those as zeros would invite
counting it as a hop.

It is also deliberately kept out of `FrequencyReport.modes`, because a
ground wave exists at *every* frequency: if it counted towards "open", the
MUF would pin itself to the top of the search range for every path on
earth. `is_open` therefore remains a statement about what the ionosphere
returns, which is what a MUF is defined by, and `usable` — any route
clearing the operator's own required SNR — is the question an operator
actually has. Coverage curves report the two in separate columns for the
same reason: merging them would fill in the skip zone the chart exists to
show.

The two do interfere where both arrive. The ground wave is the early
arrival — along the surface at *c*, under the ionosphere at more than *c* —
and the beat between them is the classic dusk fade on 160 and 80 metres, so
it enters the multipath sum alongside the skywave modes. Arrivals more than
40 dB below the strongest are dropped from that sum: at that ratio the
resultant swings by at most 0.086 dB between full addition and full
cancellation, so counting them would inflate the reported mode count
without moving any number that depends on it.

## The coastlines

`propsim/coastlines.py` holds hand-simplified outlines at roughly 1:110 million
detail. They are enough to place a mid-path ground bounce on the right side of
a coastline, which is all the reflection loss depends on, and they are checked
against fourteen named points of land and sea. They are not a survey product.

Hudson Bay, the Baltic, the Black Sea, the Caspian and the Persian Gulf are
treated as land: each is small against a hop length and enclosed by the
landmass around it. Rings never cross the antimeridian, because both the
point-in-polygon test and the globe projection assume a ring lives inside one
−180…180 span; Antarctica's band closes along the pole instead, where the seam
cannot be crossed by a ray-casting test.

## Deliberate approximations

- **Spherical Earth, not WGS-84.** The path-length error is a few parts per
  thousand, three orders of magnitude below the ionospheric uncertainty.
- **Equivalent column, per hop.** Nine profiles along the great circle are
  averaged at each height, which is what makes a radially symmetric solver
  applicable to a varying ionosphere. But each hop of a multi-hop circuit
  is traced through the ionosphere averaged over **its own stretch** of the
  path, not over the whole of it: on Madrid–Tokyo at 22% sunlit, foF2 runs
  from 8.3 MHz at the near end to 6.0 MHz at the far one, and tracing every
  hop through the 6.5 MHz average describes neither end. The hops of one
  circuit therefore reach different distances, and the circuit's launch
  angle is solved on their *sum*. A one-hop circuit reduces exactly to the
  whole-path average, so nothing is paid for the machinery when there is
  no gradient to resolve.
  Absorption escapes the averaging entirely: it reads the *local* profile
  beneath each ray node, so a path crossing the terminator absorbs like a
  half-lit path.
- **Ground reflections are charged where they land.** Each intermediate
  bounce of a multi-hop circuit is classified against the coastlines at its
  own reflection point, not against the path's dominant surface. A North
  Atlantic circuit bounces off sea water and a polar one off ice, and
  charging both as average ground is several dB per bounce in opposite
  directions.
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

The ground wave *is* implemented, but its shadow region is the
good-conductor limit of the residue series rather than a solution of the
root equation, and it carries no terrain: a mountain between two stations
is invisible to it.

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
unknown; the solar-zenith scaling of absorption still runs shallower than
the independent reference (+0.62 against +0.88), so the two diverge near
sunrise and sunset; the ground wave's shadow region uses the
good-conductor root and is optimistic over land beyond the horizon, where
it is already 50 dB down; and the equivalent column cannot produce
horizontal refraction or off-great-circle paths at all.
