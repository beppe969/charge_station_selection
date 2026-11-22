"""Retest the frozen Washington target and control on the held-out layouts with routed distances."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from planning_study import DEFAULT_OUT, SEED, binomial_interval, draw, evaluate_layout, interval_mean
from revised_city_study import city_inventory
from road_study import StreetDistances


@lru_cache(maxsize=1)
def geometries():
    city = city_inventory("Washington")
    return {
        "Planar": None,
        "Street": StreetDistances(city),
        "Connected core": StreetDistances(city, connected_core=True),
    }


def run_one(task):
    city = city_inventory("Washington")
    number, target, control = task
    observed = draw(city, 3, number)
    rows = []
    for geometry, route in geometries().items():
        layout = observed if route is None else replace(observed, distance=route(observed.xy, city.xy))
        for metrics in evaluate_layout(city, layout, target, control):
            rows.append(dict(layout=number, geometry=geometry, **metrics))
    return rows


def summarize(raw, target, control):
    rows = []
    for geometry, group in raw.groupby("geometry", sort=False):
        additions = group.pivot(index="layout", columns="policy", values="additional")
        base = group[group.policy == "baseline"]
        target_gain, control_gain = additions.target_socket, additions.control_socket
        discount = additions.target_discount
        delta = target_gain - control_gain
        lo, hi = interval_mean(delta, np.random.default_rng(SEED))
        target_wins = int((delta > 0).sum())
        win_lo, win_hi = binomial_interval(target_wins, len(delta))
        rows.append(
            dict(
                geometry=geometry,
                layouts=len(base),
                target_site=target,
                control_site=control,
                mean_baseline_served=base.served.mean(),
                mean_baseline_deficit=base.deficit.mean(),
                baseline_full=(base.deficit == 0).sum(),
                target_improved=(target_gain > 0).sum(),
                control_improved=(control_gain > 0).sum(),
                both_improved=((target_gain > 0) & (control_gain > 0)).sum(),
                target_extra_mean=target_gain.mean(),
                control_extra_mean=control_gain.mean(),
                target_minus_control_mean=delta.mean(),
                lower95=lo,
                upper95=hi,
                target_wins=target_wins,
                win_probability_lower95=win_lo,
                win_probability_upper95=win_hi,
                discount_improved=(discount > 0).sum(),
                discount_extra_mean=discount.mean(),
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_OUT,
        help="directory containing frozen planning study.json and intervention_layouts.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("At least one worker is required")
    design = json.loads((args.reference / "study.json").read_text())
    target, control = design["target"], design["control"]
    city = city_inventory("Washington")
    for label, site in (("target", target), ("control", control)):
        np.testing.assert_equal(city.latlon[site], design[label + "_latlon"])
        np.testing.assert_equal(city.levels[site], design[label + "_ports"])
    tasks = [(i, target, control) for i in range(design["test_layouts"])]
    if args.workers == 1:
        records = [row for task in tasks for row in run_one(task)]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = [row for batch in pool.map(run_one, tasks, chunksize=2) for row in batch]
    raw = pd.DataFrame(records)
    original = pd.read_csv(args.reference / "intervention_layouts.csv").sort_values(
        ["layout", "policy"]
    )
    planar = raw[raw.geometry == "Planar"][original.columns].sort_values(["layout", "policy"])
    pd.testing.assert_frame_equal(
        planar.reset_index(drop=True), original.reset_index(drop=True), rtol=1e-10, atol=1e-10
    )
    summary = summarize(raw, target, control)
    args.output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output / "route_intervention_layouts.csv", index=False, float_format="%.12g")
    summary.to_csv(args.output / "route_interventions.csv", index=False, float_format="%.12g")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
