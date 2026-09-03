"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from .antenna import AntennaSpec, AntennaType, GroundType
from .engine import PropagationEngine
from .geodesy import GeoPoint
from .noise import NoiseEnvironment
from .scenario import Scenario, Station, Weather
from .spaceweather import SpaceWeather, SpaceWeatherError, fetch_space_weather


def _station(prefix, args, is_transmitter):
    return Station(
        location=GeoPoint(
            getattr(args, f"{prefix}_lat"),
            getattr(args, f"{prefix}_lon"),
            getattr(args, f"{prefix}_name"),
        ),
        antenna=AntennaSpec(
            antenna_type=AntennaType(getattr(args, f"{prefix}_antenna")),
            height_m=getattr(args, f"{prefix}_height"),
            design_frequency_hz=args.design_freq * 1e6,
            ground=GroundType(getattr(args, f"{prefix}_ground")),
            feedline_loss_db=getattr(args, f"{prefix}_feedline"),
        ),
        transmit_power_w=args.power if is_transmitter else 100.0,
        bandwidth_hz=args.bandwidth,
        receiver_noise_figure_db=args.noise_figure,
        required_snr_db=args.required_snr,
        noise_environment=NoiseEnvironment(args.noise_environment),
        name=getattr(args, f"{prefix}_name"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="propsim", description="HF skywave propagation prediction"
    )
    parser.add_argument("--tx-lat", type=float, required=True)
    parser.add_argument("--tx-lon", type=float, required=True)
    parser.add_argument("--tx-name", default="TX")
    parser.add_argument("--rx-lat", type=float, required=True)
    parser.add_argument("--rx-lon", type=float, required=True)
    parser.add_argument("--rx-name", default="RX")

    for prefix, height in (("tx", 15.0), ("rx", 12.0)):
        parser.add_argument(
            f"--{prefix}-antenna",
            default=AntennaType.HORIZONTAL_DIPOLE.value,
            choices=[a.value for a in AntennaType],
        )
        parser.add_argument(
            f"--{prefix}-height", type=float, default=height,
            help="antenna height in METRES",
        )
        parser.add_argument(
            f"--{prefix}-ground", default=GroundType.AVERAGE_GROUND.value,
            choices=[g.value for g in GroundType],
        )
        parser.add_argument(f"--{prefix}-feedline", type=float, default=0.5)

    parser.add_argument("--power", type=float, default=100.0, help="watts")
    parser.add_argument("--bandwidth", type=float, default=2400.0, help="Hz")
    parser.add_argument("--noise-figure", type=float, default=12.0, help="dB")
    parser.add_argument("--required-snr", type=float, default=6.0, help="dB")
    parser.add_argument("--design-freq", type=float, default=14.2, help="MHz")
    parser.add_argument(
        "--noise-environment", default=NoiseEnvironment.RURAL.value,
        choices=[n.value for n in NoiseEnvironment],
    )

    parser.add_argument("--time", default=None, help="ISO 8601 UTC, default now")
    parser.add_argument("--f107", type=float, default=None)
    parser.add_argument("--sunspots", type=float, default=None)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--xray-class", default=None, help="e.g. C2, M1, X5")
    parser.add_argument("--live", action="store_true", help="fetch NOAA SWPC data")

    parser.add_argument("--rain", type=float, default=0.0, help="mm/h")
    parser.add_argument("--sea-state", type=float, default=0.0)
    parser.add_argument("--freezing", action="store_true")

    parser.add_argument("--low-mhz", type=float, default=2.0)
    parser.add_argument("--high-mhz", type=float, default=30.0)
    parser.add_argument("--step-mhz", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--reliability", action="store_true",
        help="report how often the circuit works, not just the median day",
    )
    parser.add_argument(
        "--no-sporadic-e", action="store_true",
        help="exclude the sporadic-E branch from the reliability mixture",
    )
    return parser


def build_scenario(args) -> Scenario:
    when = (
        datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)
        if args.time
        else datetime.now(timezone.utc)
    )

    if args.live:
        weather = fetch_space_weather()
    else:
        weather = SpaceWeather()
    fields = {}
    if args.f107 is not None:
        fields["f107"] = args.f107
    if args.sunspots is not None:
        fields["sunspot_number"] = args.sunspots
    if args.kp is not None:
        fields["kp"] = args.kp
    if fields:
        import dataclasses

        weather = dataclasses.replace(weather, source="manual", **fields)
    if args.xray_class:
        weather = weather.with_flare(args.xray_class)

    return Scenario(
        transmitter=_station("tx", args, True),
        receiver=_station("rx", args, False),
        when=when,
        space_weather=weather,
        weather=Weather(
            rain_rate_mm_h=args.rain,
            sea_state=args.sea_state,
            freezing=args.freezing,
        ),
    )


def render_reliability(predictor, hint_hz) -> str:
    muf = predictor.muf_distribution_hz(step_hz=5e5, hint_hz=hint_hz)
    lines = [
        "",
        "Reliability -- fraction of days the circuit closes",
        f"  MUF   bad day {muf['lower_decile'] / 1e6:.2f}   median "
        f"{muf['median'] / 1e6:.2f}   good day {muf['upper_decile'] / 1e6:.2f} MHz",
        f"  foF2 spread x{predictor.spread.lower_decile:.2f}-"
        f"{predictor.spread.upper_decile:.2f}",
    ]
    if predictor.sporadic_e_layer:
        lines.append(
            f"  sporadic E {predictor.sporadic_e_probability * 100:.0f}% likely, "
            f"foEs {predictor.sporadic_e_layer.foes_mhz:.1f} MHz"
        )
    else:
        lines.append("  sporadic E not expected on this path at this time of year")
    lines += [
        "",
        f"{'band':>6} {'MHz':>7} {'works':>7} {'p10':>8} {'median':>8} {'p90':>8}  via",
    ]
    for row in predictor.band_reliability():
        def margin(value):
            return f"{value:8.1f}" if value is not None else f"{'closed':>8}"

        # Sporadic E can subtract as well as add: a patch that screens a
        # better F-layer path costs the band reliability, and saying only
        # "F layer" there would hide why the number fell.
        without = row["reliability_without_es"]
        combined = row["reliability"]
        if combined > without + 1e-9:
            via = "sporadic E" if without <= 1e-9 else "F layer + Es"
        elif combined < without - 1e-9:
            via = "F layer, Es screens"
        else:
            via = "F layer"
        lines.append(
            f"{row['band']:>6} {row['frequency_mhz']:7.2f} "
            f"{row['reliability'] * 100:6.0f}% {margin(row['lower_decile_margin_db'])} "
            f"{margin(row['median_margin_db'])} {margin(row['upper_decile_margin_db'])}  {via}"
        )
    return "\n".join(lines)


def render(prediction) -> str:
    conditions = prediction.conditions
    scenario = prediction.scenario
    lines = [
        f"{scenario.transmitter.name} -> {scenario.receiver.name}",
        f"  {conditions['distance_km']:.0f} km on bearing {conditions['bearing_deg']:.1f} deg",
        f"  {scenario.when.isoformat()}",
        "",
        "Conditions",
        f"  F10.7 {scenario.space_weather.f107:.0f}   Kp {scenario.space_weather.kp:.1f}"
        f"   X-ray {scenario.space_weather.xray_class}   ({scenario.space_weather.source})",
        f"  foF2 {conditions['fof2_mhz']:.2f} MHz   foE {conditions['foe_mhz']:.2f} MHz"
        f"   hmF2 {conditions['hmf2_km']:.0f} km",
        f"  path {conditions['sunlit_fraction'] * 100:.0f}% sunlit"
        f"{', crosses the terminator' if conditions['crosses_terminator'] else ''}"
        f"   {conditions['sea_fraction'] * 100:.0f}% over sea",
        f"  gyrofrequency {conditions['gyrofrequency_mhz']:.2f} MHz"
        f"   magnetic dip {conditions['magnetic_dip_deg']:.1f} deg",
        "",
        f"MUF {prediction.muf_mhz:.2f} MHz" if prediction.muf_mhz else "MUF none",
        f"FOT {prediction.optimum_working_frequency_mhz:.2f} MHz (0.85 x MUF, a "
        "variability rule of thumb)" if prediction.muf_mhz else "",
        f"LOF {prediction.lof_mhz:.2f} MHz" if prediction.lof_mhz
        else "LOF none - no frequency closes the link",
        "",
        f"{'band':>6} {'MHz':>7} {'SNR':>7} {'margin':>8} {'hops':>5} {'elev':>6} {'score':>6}",
    ]
    for row in prediction.band_report():
        if row["open"]:
            lines.append(
                f"{row['band']:>6} {row['frequency_mhz']:7.2f} {row['snr_db']:7.1f} "
                f"{row['margin_db']:8.1f} {row['hops']:5d} {row['elevation_deg']:6.1f} "
                f"{row['score']:6.3f}"
            )
        else:
            lines.append(
                f"{row['band']:>6} {row['frequency_mhz']:7.2f} {'closed':>7} "
                f"{'-':>8} {'-':>5} {'-':>6} {row['score']:6.3f}"
            )
    lines += [
        "",
        "Band scores are an operational heuristic combining margin, headroom",
        "below the MUF and antenna practicality. They are not probabilities.",
    ]
    return "\n".join(line for line in lines if line is not None)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenario = build_scenario(args)
    except (ValueError, SpaceWeatherError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    engine = PropagationEngine(scenario)
    prediction = engine.predict(
        args.low_mhz * 1e6, args.high_mhz * 1e6, args.step_mhz * 1e6
    )

    predictor = None
    if args.reliability:
        from .reliability import ReliabilityPredictor

        predictor = ReliabilityPredictor(
            scenario, include_sporadic_e=not args.no_sporadic_e
        )

    if args.json:
        payload = prediction.summary()
        if predictor is not None:
            payload["reliability"] = predictor.summary()
        print(json.dumps(payload, indent=2))
    else:
        print(render(prediction))
        if predictor is not None:
            print(render_reliability(predictor, prediction.muf_hz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
