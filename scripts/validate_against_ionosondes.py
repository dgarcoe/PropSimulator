#!/usr/bin/env python3
"""Compare the modelled ionosphere with real ionosonde measurements.

The ray tracer is validated against closed-form analysis and absorption
against an independent index, but the **ionosphere itself has never been
checked against observation**.  Its error is therefore not large or small;
it is unknown, which is worse.  This script closes that gap when network
access is available.

It pulls foF2, foE and hmF2 from the GIRO / DIDBase archive for a set of
stations and a time range, runs the model for the same place and hour, and
reports the distribution of the ratio between them.

    python scripts/validate_against_ionosondes.py --days 7 --out report.json

IMPORTANT -- this script has never been executed against the live service.
The environment it was written in blocks outbound HTTPS by policy, so the
request shapes and the response parsing below are written from the
documented interface and have not been confirmed against a real response.
Treat the first run as part of the work, not as a finished result: check
that ``--dry-run`` prints the URLs you expect, and that a single station
returns sane numbers, before trusting an aggregate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from propsim.geodesy import GeoPoint                       # noqa: E402
from propsim.ionosphere import build_profile               # noqa: E402
from propsim.solar import illuminate_path                  # noqa: E402
from propsim.spaceweather import SpaceWeather              # noqa: E402

DIDB_URL = "https://lgdc.uml.edu/common/DIDBGetValues"

#: A deliberately spread set: mid-latitude north and south, equatorial and
#: auroral, so a model that is right only at mid-latitudes is caught.
STATIONS = {
    "EA036": ("El Arenosillo", GeoPoint(37.1, -6.7)),
    "JR055": ("Juliusruh", GeoPoint(54.6, 13.4)),
    "BC840": ("Boulder", GeoPoint(40.0, -105.3)),
    "PRJ18": ("Jicamarca", GeoPoint(-12.0, -76.8)),
    "TR170": ("Tromso", GeoPoint(69.6, 19.2)),
    "CAN__": ("Canberra", GeoPoint(-35.3, 149.0)),
}

CHARACTERISTICS = ("foF2", "foE", "hmF2")


@dataclass
class Comparison:
    station: str
    characteristic: str
    when: datetime
    observed: float
    modelled: float

    @property
    def ratio(self) -> float:
        return self.modelled / self.observed if self.observed else math.nan


def fetch_didb(
    ursi_code: str, characteristic: str, start: datetime, end: datetime,
    timeout: float = 30.0, dry_run: bool = False,
) -> List[tuple[datetime, float]]:
    """Fetch one characteristic for one station over a time range."""
    query = urllib.parse.urlencode({
        "ursiCode": ursi_code,
        "charName": characteristic,
        "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": end.strftime("%Y-%m-%d %H:%M:%S"),
    })
    url = f"{DIDB_URL}?{query}"
    if dry_run:
        print(f"  would fetch {url}")
        return []

    request = urllib.request.Request(url, headers={"User-Agent": "PropSimulator/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  {ursi_code}/{characteristic}: fetch failed ({exc})", file=sys.stderr)
        return []

    rows: List[tuple[datetime, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            stamp = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
            value = float(parts[2])
        except (ValueError, IndexError):
            continue
        if value > 0.0:
            rows.append((stamp, value))
    return rows


def model_value(
    characteristic: str, point: GeoPoint, when: datetime, weather: SpaceWeather
) -> Optional[float]:
    season = illuminate_path(point, point, when, 3).seasonal_phase
    profile = build_profile(point, when, weather, season)
    if characteristic == "foF2":
        return profile.layers.fof2_mhz
    if characteristic == "foE":
        return profile.layers.e.critical_frequency_mhz
    if characteristic == "hmF2":
        return profile.layers.f2.peak_height_km
    return None


def summarise(comparisons: List[Comparison]) -> Dict[str, dict]:
    grouped: Dict[str, List[float]] = {}
    for comparison in comparisons:
        ratio = comparison.ratio
        if math.isfinite(ratio):
            grouped.setdefault(comparison.characteristic, []).append(ratio)

    summary = {}
    for characteristic, ratios in grouped.items():
        ratios.sort()
        summary[characteristic] = {
            "samples": len(ratios),
            "median_ratio": statistics.median(ratios),
            "mean_ratio": statistics.fmean(ratios),
            "p10_ratio": ratios[len(ratios) // 10] if len(ratios) >= 10 else None,
            "p90_ratio": ratios[-len(ratios) // 10] if len(ratios) >= 10 else None,
            "rms_relative_error": math.sqrt(
                statistics.fmean([(r - 1.0) ** 2 for r in ratios])
            ),
        }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--f107", type=float, default=140.0)
    parser.add_argument("--sunspots", type=float, default=60.0)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--out", default=None, help="write a JSON report here")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the URLs that would be fetched and stop")
    args = parser.parse_args(argv)

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    weather = SpaceWeather(
        f107=args.f107, sunspot_number=args.sunspots, kp=args.kp, source="manual"
    )

    comparisons: List[Comparison] = []
    for code, (name, point) in STATIONS.items():
        print(f"{name} ({code})")
        for characteristic in CHARACTERISTICS:
            for stamp, observed in fetch_didb(
                code, characteristic, start, end, dry_run=args.dry_run
            ):
                modelled = model_value(characteristic, point, stamp, weather)
                if modelled is None:
                    continue
                comparisons.append(
                    Comparison(name, characteristic, stamp, observed, modelled)
                )

    if args.dry_run:
        return 0
    if not comparisons:
        print("no measurements retrieved; nothing to compare", file=sys.stderr)
        return 1

    summary = summarise(comparisons)
    print(f"\n{'characteristic':>15} {'n':>6} {'median':>8} {'p10':>7} {'p90':>7} {'rms err':>8}")
    for characteristic, stats in summary.items():
        p10 = f"{stats['p10_ratio']:.3f}" if stats["p10_ratio"] else "  -  "
        p90 = f"{stats['p90_ratio']:.3f}" if stats["p90_ratio"] else "  -  "
        print(
            f"{characteristic:>15} {stats['samples']:6d} {stats['median_ratio']:8.3f} "
            f"{p10:>7} {p90:>7} {stats['rms_relative_error'] * 100:7.1f}%"
        )
    print("\nRatios are modelled / observed. A median far from 1.0 is a bias;")
    print("a wide p10-p90 spread is scatter the model does not capture.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({
                "summary": summary,
                "comparisons": [
                    {
                        "station": c.station,
                        "characteristic": c.characteristic,
                        "when": c.when.isoformat(),
                        "observed": c.observed,
                        "modelled": c.modelled,
                        "ratio": c.ratio,
                    }
                    for c in comparisons
                ],
            }, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
