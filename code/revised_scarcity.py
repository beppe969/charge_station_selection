"""Compare a fixed forecast opportunity-cost policy with sequential and batch dispatch."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from revised_city_study import BOOTSTRAPS, CASES, RESULTS, city_inventory, draw, evaluate, summarize


SCARCITY = "Forecast scarcity"
FALLBACK = 100.0
POLICIES = ("Batch cost", "Sequential cost", SCARCITY)


def forecast_surcharges(city, rep, scenario):
    """Forecast LP capacity prices. Site/class copies are interchangeable here."""
    if scenario.varied_power:
        raise ValueError("This forecast LP groups equal nominal powers within each site/class")
    inventory = city_inventory(city)
    samples = draw(inventory, city, rep, scenario, forecast=True, return_port_classes=True)
    sites, _, _, _, _, _, _, eligible, cost, _, classes = samples
    _, first, group, capacity = np.unique(
        np.column_stack((sites, classes)),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    n = len(eligible)
    groups = len(first)
    # Subtract the all-fallback cost. A forecast assignment at cost above
    # the fallback is dominated. Individual request and group upper bounds
    # suffice after those unserved forecast requests have been eliminated.
    reduced_cost = cost[:, first] - FALLBACK
    ev, dest = np.nonzero(eligible[:, first] & (reduced_cost < 0))
    if not len(ev):
        return np.zeros(len(sites))
    cols = np.arange(len(ev))
    incidence = coo_matrix(
        (np.ones(2 * len(cols)), (np.r_[ev, n + dest], np.r_[cols, cols])),
        shape=(n + groups, len(cols)),
    ).tocsc()
    fit = linprog(
        reduced_cost[ev, dest],
        A_ub=incidence,
        b_ub=np.r_[np.ones(n), capacity],
        bounds=(0, None),
        method="highs-ds",
    )
    if not fit.success:
        raise RuntimeError(f"Forecast matching LP failed in layout {rep}: {fit.message}")
    assert np.isclose(fit.fun, np.dot(fit.ineqlin.marginals, np.r_[np.ones(n), capacity]))
    shadow = -fit.ineqlin.marginals[n:]
    assert np.min(shadow) >= -1e-7
    return np.maximum(shadow, 0)[group]


def run_one(task):
    case, rep = task
    city = "Washington"
    fees = forecast_surcharges(city, rep, case)
    rows = evaluate(city, rep, case, policies=POLICIES, surcharges={SCARCITY: fees})
    for row in rows:
        row["forecast_positive_ports"] = (fees > 1e-6).sum()
        row["forecast_max_shadow_usd"] = fees.max()
    return rows


def paired_changes(raw):
    rows = []
    for case, group in raw.groupby("case", sort=False):
        informed = group[group.policy == "Sequential cost"].set_index("layout").sort_index()
        forecast = group[group.policy == SCARCITY].set_index("layout").sort_index()
        assert informed.index.equals(forecast.index)
        n = len(informed)
        samples = np.random.default_rng(4577).integers(n, size=(BOOTSTRAPS, n))
        measures = {
            key: forecast[key] - informed[key]
            for key in ("service_pct", "cost_saving_pct", "fallback_100_usd")
        }
        measures["common_cost_difference_usd"] = forecast.common_cost_difference_usd
        for key, v in measures.items():
            change = v.to_numpy()
            lo, hi = np.quantile(change[samples].mean(axis=1), (0.025, 0.975))
            rows.append(
                dict(
                    city="Washington",
                    case=case,
                    metric=key,
                    layouts=n,
                    contrast="forecast minus informed",
                    change=change.mean(),
                    lo=lo,
                    hi=hi,
                )
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layouts < 2 or args.workers < 1:
        parser.error("Require at least two layouts and one worker")
    tasks = [
        (case, rep)
        for case in CASES
        if case.name in ("base", "high load")
        for rep in range(args.layouts)
    ]
    if args.workers == 1:
        rows = [row for task in tasks for row in run_one(task)]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = [row for batch in executor.map(run_one, tasks, chunksize=1) for row in batch]
    args.output.mkdir(exist_ok=True, parents=True)
    raw = pd.DataFrame(rows)
    for name, frame in (("layouts", raw), ("summary", summarize(raw)), ("paired", paired_changes(raw))):
        frame.to_csv(args.output / f"scarcity_{name}.csv", index=False, float_format="%.10g")
    print(f"{len(tasks)} layouts, {len(rows)} policy-layout records. Results in {args.output}")


if __name__ == "__main__":
    main()
