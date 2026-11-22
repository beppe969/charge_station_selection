"""Per-driver outcomes of joint reassignment and a DC charging-taper sensitivity.

Part 1 (all three cities, source-reconciled public inventories, half load).
For every layout and processing order, take the drivers served by sequential
cost booking and compare each driver's generalized cost under that booking
with the same driver's cost in the minimum-cost reassignment of exactly that
group. Positive differences are gains from joint assignment.

It also solves reassignments in which no served driver may lose more than
2, 5, 10 or 20 dollars, by pruning the pairs that would exceed the cap.
The code also solves the reassignment restricted to pairs that cost each
driver no more than the sequential booking. Sequential least-cost booking is
a serial dictatorship, so this no-loss saving is zero whenever distinct port
classes never tie in cost; the column noloss_saving_pct checks this.

Part 2 (Washington, half load). Repeat the sequential-cost comparison when the
average power of a DC session declines linearly with the request, from the
nominal effective power at 20 kWh to half of it at 60 kWh. AC sessions are
unchanged. With the factor set to one the code reproduces the published base
row of the sensitivity table.

Usage: python code/driver_outcomes.py [--layouts 30] [--workers 2]
Writes results/tits/driver_outcomes_drivers.csv.gz, driver_outcomes_summary.csv,
driver_rows.tex, taper_layouts.csv and taper_summary.csv.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from revised_city_data import FILES
from revised_city_study import BOOTSTRAPS, RESULTS, Scenario, choose_sequential, draw, exact_assignment
from revised_primary_cities import public_inventory

OUT = RESULTS / "tits"
CAPS = (2, 5, 10, 20)  # dollars: largest loss any served driver may bear
TOL = 0.005  # dollars; differences below half a cent count as ties
_INVENTORIES = {}


def inventory(city):
    if city not in _INVENTORIES:
        _INVENTORIES[city] = public_inventory(city)
    return _INVENTORIES[city]


def tapered(cost_parts, taper, scenario):
    """Rebuild charge time and cost with the DC taper factor."""
    sites, distance, q, prices, waits, bill, charge, eligible, cost, orders, level = cost_parts
    if taper:
        factor = 1 - 0.5 * (q - 20) / 40
        dc = level == 2
        charge = charge.copy()
        charge[:, dc] = charge[:, dc] / factor[:, None]
        cost = (
            scenario.distance_value * distance[:, sites]
            + bill
            + scenario.time_value * (waits[sites] + charge)
        )
    return sites, distance, q, prices, waits, bill, charge, eligible, cost, orders, level


def per_driver(task):
    city, rep = task
    scenario = Scenario()
    parts = draw(inventory(city), city, rep, scenario, return_port_classes=True)
    sites, distance, q, prices, waits, bill, charge, eligible, cost, orders, level = parts
    rows = []
    for k, order in enumerate(orders):
        chosen = choose_sequential("Sequential cost", order, eligible, cost, distance, sites, prices)
        ev = np.flatnonzero(chosen >= 0)
        joint = exact_assignment(cost[ev], eligible[ev])
        assert (joint >= 0).all()
        seq_cost = cost[ev, chosen[ev]]
        joint_cost = cost[ev, joint]
        gain = seq_cost - joint_cost
        # No-loss reassignment: keep only pairs that cost each driver no more
        # than the sequential booking. The sequential assignment stays feasible.
        noloss_allowed = eligible[ev] & (cost[ev] <= seq_cost[:, None] + 1e-9)
        noloss = exact_assignment(cost[ev], noloss_allowed)
        assert (noloss >= 0).all()
        noloss_cost = cost[ev, noloss]
        capped = {}
        for cap in CAPS:
            allowed = eligible[ev] & (cost[ev] <= seq_cost[:, None] + cap + 1e-9)
            choice = exact_assignment(cost[ev], allowed)
            assert (choice >= 0).all()
            capped[cap] = cost[ev, choice]
        seq_dc = level[chosen[ev]] == 2
        joint_dc = level[joint] == 2
        for i, j in enumerate(ev):
            rows.append(
                dict(
                    city=city,
                    layout=rep,
                    order=k,
                    driver=int(j),
                    request_kwh=q[j],
                    gain_usd=gain[i],
                    seq_cost_usd=seq_cost[i],
                    joint_cost_usd=joint_cost[i],
                    noloss_cost_usd=noloss_cost[i],
                    **{f"cap{cap}_cost_usd": capped[cap][i] for cap in CAPS},
                    seq_dc=bool(seq_dc[i]),
                    joint_dc=bool(joint_dc[i]),
                    seq_km=distance[j, sites[chosen[j]]],
                    joint_km=distance[j, sites[joint[i]]],
                )
            )
    return rows


def taper_task(task):
    rep, taper = task
    scenario = Scenario()
    city = "Washington"
    parts = tapered(draw(inventory(city), city, rep, scenario, return_port_classes=True), taper, scenario)
    sites, distance, q, prices, waits, bill, charge, eligible, cost, orders, level = parts
    n = len(q)
    max_served = np.count_nonzero(exact_assignment(cost, eligible) >= 0)
    out = []
    for order in orders:
        chosen = choose_sequential("Sequential cost", order, eligible, cost, distance, sites, prices)
        ev = np.flatnonzero(chosen >= 0)
        observed = cost[ev, chosen[ev]].sum()
        joint = exact_assignment(cost[ev], eligible[ev])
        reference = cost[ev, joint].sum()
        out.append(
            dict(
                saving_pct=100 * (observed - reference) / observed,
                deficit_pp=100 * (max_served - len(ev)) / n,
                bill_usd=bill[ev, chosen[ev]].mean(),
                charge_min=charge[ev, chosen[ev]].mean(),
                charge_saving_usd=scenario.time_value
                * (charge[ev, chosen[ev]].sum() - charge[ev, joint].sum())
                / len(ev),
                saving_usd=(observed - reference) / len(ev),
            )
        )
    row = {key: np.mean([o[key] for o in out]) for key in out[0]}
    row.update(layout=rep, taper=taper, full=int(max_served == n))
    return row


def interval(values, seed=4577):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(values), size=(BOOTSTRAPS, len(values)))
    lo, hi = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
    return values.mean(), lo, hi


def summarize_drivers(raw):
    rows = []
    for city, g in raw.groupby("city", sort=False):
        per_layout = g.groupby("layout").apply(
            lambda x: pd.Series(
                dict(
                    worse=100 * np.mean(x.gain_usd < -TOL),
                    better=100 * np.mean(x.gain_usd > TOL),
                    loss_given_worse=-x.gain_usd[x.gain_usd < -TOL].mean(),
                    gain_given_better=x.gain_usd[x.gain_usd > TOL].mean(),
                    worse_5=100 * np.mean(x.gain_usd < -5),
                    q_worse=x.request_kwh[x.gain_usd < -TOL].mean(),
                    q_better=x.request_kwh[x.gain_usd > TOL].mean(),
                    joint_saving_pct=100 * (x.seq_cost_usd.sum() - x.joint_cost_usd.sum()) / x.seq_cost_usd.sum(),
                    noloss_saving_pct=100 * (x.seq_cost_usd.sum() - x.noloss_cost_usd.sum()) / x.seq_cost_usd.sum(),
                    noloss_better=100 * np.mean(x.seq_cost_usd - x.noloss_cost_usd > TOL),
                    **{f"cap{cap}_saving_pct": 100 * (x.seq_cost_usd.sum() - x[f"cap{cap}_cost_usd"].sum())
                       / x.seq_cost_usd.sum() for cap in CAPS},
                    gain_loss_ratio=x.gain_usd[x.gain_usd > 0].sum() / -x.gain_usd[x.gain_usd < 0].sum(),
                )
            )
        )
        row = dict(city=city, layouts=len(per_layout), drivers=len(g))
        for key in per_layout:
            m, lo, hi = interval(per_layout[key].to_numpy())
            row[key], row[key + "_lo"], row[key + "_hi"] = m, lo, hi
        for p in (5, 50, 95):
            row[f"gain_p{p}"] = np.percentile(g.gain_usd, p)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    driver_tasks = [(city, r) for city in FILES for r in range(args.layouts)]
    taper_tasks = [(r, t) for t in (False, True) for r in range(args.layouts)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        raw = pd.DataFrame([row for rows in ex.map(per_driver, driver_tasks) for row in rows])
        taper = pd.DataFrame(list(ex.map(taper_task, taper_tasks)))
    raw.to_csv(OUT / "driver_outcomes_drivers.csv.gz", index=False, float_format="%.10g")
    summary = summarize_drivers(raw)
    summary.to_csv(OUT / "driver_outcomes_summary.csv", index=False, float_format="%.10g")
    taper.to_csv(OUT / "taper_layouts.csv", index=False, float_format="%.10g")
    tsum = []
    for flag, g in taper.groupby("taper"):
        g = g.sort_values("layout")
        row = dict(taper=flag, full=int(g.full.sum()), layouts=len(g))
        for key in ("saving_pct", "deficit_pp", "bill_usd", "charge_min", "charge_saving_usd", "saving_usd"):
            row[key], row[key + "_lo"], row[key + "_hi"] = interval(g[key].to_numpy())
        tsum.append(row)
    pd.DataFrame(tsum).to_csv(OUT / "taper_summary.csv", index=False, float_format="%.10g")
    lines = []
    for r in summary.itertuples():
        lines.append(
            f"{r.city} & {r.better:.1f} & {r.worse:.1f} & {r.gain_given_better:.2f} & {r.loss_given_worse:.2f} & "
            f"{r.worse_5:.1f} & {r.gain_loss_ratio:.2f} & {r.cap5_saving_pct:.2f} & "
            f"{r.cap20_saving_pct:.2f} \\\\"
        )
    (OUT / "driver_rows.tex").write_text("\n".join(lines) + "\n")
    print(summary.round(3).T)
    print(pd.DataFrame(tsum).round(3).T)


if __name__ == "__main__":
    main()
