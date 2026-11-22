"""Regenerate the city, load and input-sensitivity results in the revised paper."""

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist

from revised_city_data import FILES, draw_positions, load_city


RESULTS = Path(__file__).resolve().parents[1] / "results"
POLICIES = ("Batch cost", "Batch distance", "Sequential cost", "Nearest", "Cheapest of three")
SEED = 103721
ORDERS = 3
BOOTSTRAPS = 2000


@dataclass(frozen=True)
class Scenario:
    name: str = "base"
    load: float = 0.50
    circuity: float = 1.0
    multicentre: bool = False
    varied_power: bool = False
    correlated_tariff: bool = False
    distance_value: float = 2.52
    time_value: float = 0.224


CASES = (
    Scenario(),
    Scenario("low load", load=0.25),
    Scenario("medium-high load", load=0.75),
    Scenario("high load", load=0.95),
    Scenario("circuity", circuity=1.30),
    Scenario("two centres", multicentre=True),
    Scenario("variable power", varied_power=True),
    Scenario("linked tariffs and waits", correlated_tariff=True),
    Scenario("lower time value", time_value=0.10),
    Scenario("higher distance value", distance_value=5.04),
)


def exact_assignment(scores, allowed):
    """Maximize cardinality and then minimize scores; never relax eligibility.

    Try the fully served problem first. If it has no solution, a separate
    maximum matching gives the exact number of zero-cost dummy columns needed.
    Every feasible completion must then use exactly that number of dummies.
    """
    n, m = allowed.shape
    if n == 0:
        return np.empty(0, dtype=int)
    cost = np.where(allowed, scores, np.inf)
    try:
        if n > m:
            raise ValueError("fewer ports than requests")
        ev, port = linear_sum_assignment(cost)
    except ValueError:
        matching = maximum_bipartite_matching(csr_matrix(allowed), perm_type="column")
        k = np.count_nonzero(matching >= 0)
        cost = np.column_stack((cost, np.zeros((n, n - k))))
        ev, port = linear_sum_assignment(cost)
    choice = np.full(n, -1, dtype=int)
    used = port < m
    assert allowed[ev[used], port[used]].all()
    choice[ev[used]] = port[used]
    return choice


def choose_sequential(policy, order, eligible, cost, distance, sites, prices):
    n, m = eligible.shape
    chosen = np.full(n, -1, dtype=int)
    free = np.ones(m, dtype=bool)
    for ev in order:
        options = np.flatnonzero(eligible[ev] & free)
        if not len(options):
            continue
        if policy in ("Nearest", "Cheapest of three"):
            candidates = np.unique(sites[options])
            if policy == "Cheapest of three":
                sorted_sites = candidates[np.argsort(distance[ev, candidates], kind="stable")[:3]]
                site = sorted_sites[np.argmin(prices[sorted_sites])]
            else:
                site = candidates[np.argmin(distance[ev, candidates])]
            options = options[sites[options] == site]
        # Break same-site ties by full cost, then the shared random port ordering.
        best = options[np.argmin(cost[ev, options])]
        free[best] = False
        chosen[ev] = best
    return chosen


def distance_assignment(distance, sites, eligible, cost):
    chosen = exact_assignment(distance[:, sites], eligible)
    served = np.flatnonzero(chosen >= 0)
    assignments = sites[chosen[served]]
    # Distance cannot distinguish sockets at one coordinate. Select the best
    # cost assignment within each chosen site while preserving EV/site choices.
    for site in np.unique(assignments):
        ev = served[assignments == site]
        ports = np.flatnonzero(sites == site)
        local = np.ix_(ev, ports)
        _, local_ports = linear_sum_assignment(np.where(eligible[local], cost[local], np.inf))
        chosen[ev] = ports[local_ports]
    return chosen


def draw(
    inventory,
    city,
    rep,
    scenario,
    distance_function=cdist,
    accessible_levels=None,
    forecast=False,
    return_port_classes=False,
):
    seeds = np.random.SeedSequence([SEED, list(FILES).index(city), rep]).spawn(10)
    if forecast:
        independent = np.random.SeedSequence([SEED, list(FILES).index(city), rep, 99401]).spawn(10)
        # Forecast independent origins, energy, reservation tariff and power caps.
        # Preserve the observed tariffs, waits, port order and installed powers.
        for i in (1, 2, 3, 6, 7):
            seeds[i] = independent[i]
    rng = [np.random.default_rng(s) for s in seeds]
    site_index, level = np.nonzero(inventory.levels)
    counts = inventory.levels[site_index, level]
    if accessible_levels is not None:
        public_counts = np.asarray(accessible_levels)
        if public_counts.shape != inventory.levels.shape or not np.all(
            (public_counts >= 0)
            & (public_counts <= inventory.levels)
            & (public_counts == np.floor(public_counts))
        ):
            raise ValueError("Access counts must be a subset of the installed counts by site and class")
        public_counts = public_counts[site_index, level]
        # Within a colocated class records have identical modeled attributes
        # in the reference scenario. Retain the specified number of copies
        # before applying the shared random port permutation.
        accessible = np.concatenate(
            [np.arange(count) < cap for count, cap in zip(counts, public_counts)]
        )
    site_index = np.repeat(site_index, counts)
    level = np.repeat(level, counts)
    m = len(level)
    n = int(np.floor(m * scenario.load + 0.5))
    perm = rng[0].permutation(m)
    sites, level = site_index[perm], level[perm]
    positions = draw_positions(inventory, n, rng[1], scenario.multicentre)
    distance = distance_function(positions, inventory.xy) * scenario.circuity
    q = rng[2].uniform(20, 60, n)
    threshold = rng[3].uniform(0.40, 0.60, n)
    price_draw = rng[4].random(len(inventory.xy))
    wait_draw = rng[5].random(len(inventory.xy))
    if scenario.correlated_tariff:
        # A site with a DC outlet carries a fast-service premium. The cheaper
        # the resulting price, the higher the conditional mean external queue.
        fast_site = inventory.levels[:, 2] > 0
        price_draw = 0.70 * price_draw + 0.30 * fast_site
        wait_draw = 0.5 * wait_draw + 0.5 * (1 - price_draw)
    prices = 0.303 + (0.530 - 0.303) * price_draw
    waits = 15 * wait_draw
    ac_cap = rng[6].choice([7.2, 11.0], n)
    dc_cap = rng[7].choice([50.0, 100.0], n)
    power = np.array([1.4, 7.2, 50.0])[level]
    if scenario.varied_power:
        for cls, rates in ((1, [7.2, 11.0, 19.0]), (2, [50.0, 100.0, 150.0])):
            power[level == cls] = rng[8].choice(rates, np.count_nonzero(level == cls))
    effective = np.minimum(power, np.where(level == 2, dc_cap[:, None], ac_cap[:, None]))
    charge = 60 * q[:, None] / effective
    bill = q[:, None] * prices[sites]
    eligible = (distance[:, sites] <= 5) & (prices[sites] <= threshold[:, None])
    if accessible_levels is not None:
        eligible &= accessible[perm]
    cost = (
        scenario.distance_value * distance[:, sites]
        + bill
        + scenario.time_value * (waits[sites] + charge)
    )
    orders = [rng[9].permutation(n) for _ in range(ORDERS)]
    result = sites, distance, q, prices, waits, bill, charge, eligible, cost, orders
    return (*result, level) if return_port_classes else result


def evaluate(
    city,
    rep,
    scenario,
    distance_function=cdist,
    accessible_levels=None,
    inventory=None,
    trip_tail=False,
    policies=POLICIES,
    surcharges=None,
):
    if inventory is None:
        inventory = city_inventory(city)
    sites, distance, q, prices, waits, bill, charge, eligible, cost, orders = draw(
        inventory, city, rep, scenario, distance_function, accessible_levels
    )
    n = len(q)
    batch = exact_assignment(cost, eligible)
    max_served = np.count_nonzero(batch >= 0)
    if "Batch distance" in policies:
        min_distance = distance_assignment(distance, sites, eligible, cost)
        assert np.count_nonzero(min_distance >= 0) == max_served

    def total(ev, assigned):
        return float(cost[ev, assigned].sum())

    all_ev = np.arange(n)
    best_total = total(all_ev[batch >= 0], batch[batch >= 0])
    cohort_optima = {tuple(all_ev[batch >= 0]): best_total}

    def record(chosen):
        ev = all_ev[chosen >= 0]
        ports = chosen[ev]
        served = len(ev)
        assert np.unique(ports).size == served and eligible[ev, ports].all()
        observed = total(ev, ports)
        ids = tuple(ev)
        if ids not in cohort_optima:
            restricted = exact_assignment(cost[ev], eligible[ev])
            assert (restricted >= 0).all()
            cohort_optima[ids] = total(ev, restricted)
        ref_cost = cohort_optima[ids]
        assert observed >= ref_cost - 1e-6
        trips = distance[ev, sites[ports]]
        trip, fee = trips.sum(), bill[ev, ports].sum()
        wait, plugged = waits[sites[ports]].sum(), charge[ev, ports].sum()
        assert np.isclose(
            observed, scenario.distance_value * trip + fee + scenario.time_value * (wait + plugged)
        )
        k = served if served else np.nan
        tail = {}
        if trip_tail:
            tail["distance_p95_km"] = np.quantile(trips, 0.95) if served else np.nan
        return {
            "missed_ids": np.flatnonzero(chosen < 0).tolist(),
            "served": served,
            "service_pct": 100 * served / n,
            "unserved_kwh": q[chosen < 0].sum(),
            "distance_km": trip / k,
            **tail,
            "bill_usd": fee / k,
            "wait_min": wait / k,
            "charge_min": plugged / k,
            "cost_total": observed,
            "cohort_opt_total": ref_cost,
            "cost_saving_pct": 100 * (observed - ref_cost) / observed if served else np.nan,
            "service_deficit_pp": 100 * (max_served - served) / n,
            **{f"fallback_{p}_usd": (observed + p * (n - served)) / n for p in (50, 100, 200)},
        }

    rows = []
    ordinary_sequential = None
    for policy in policies:
        if policy == "Batch cost":
            per_order = [record(batch)]
        elif policy == "Batch distance":
            per_order = [record(min_distance)]
        else:
            scores = cost + surcharges[policy] if surcharges and policy in surcharges else cost
            choices = [
                choose_sequential(policy, order, eligible, scores, distance, sites, prices)
                for order in orders
            ]
            per_order = [record(assignment) for assignment in choices]
            if surcharges and policy == "Sequential cost":
                ordinary_sequential = choices
            if surcharges and policy in surcharges and ordinary_sequential is not None:
                for choice, reference, report in zip(choices, ordinary_sequential, per_order):
                    common = np.flatnonzero((choice >= 0) & (reference >= 0))
                    report["common_served"] = len(common)
                    report["common_cost_difference_usd"] = (
                        np.mean(cost[common, choice[common]] - cost[common, reference[common]])
                        if len(common)
                        else np.nan
                    )
        # Within-layout EV indices, zero-based. The list order follows the common EV orders.
        missed = json.dumps([x.pop("missed_ids") for x in per_order], separators=(",", ":"))
        measures = {key: np.mean([x[key] for x in per_order]) for key in per_order[0]}
        rows.append(
            dict(
                city=city,
                case=scenario.name,
                load=scenario.load,
                layout=rep,
                requests=n,
                max_served=max_served,
                isolated=int(np.sum(~eligible.any(axis=1))),
                full_feasible=int(max_served == n),
                policy=policy,
                missed_by_order=missed,
                **measures,
            )
        )
    return rows


@lru_cache(None)
def city_inventory(city):
    return load_city(city)


def run_task(task):
    city, rep, case = task
    return evaluate(city, rep, case)


def summarize(raw):
    summaries = []
    for (city, case), group in raw.groupby(["city", "case"], sort=False):
        full_cost = group[group.policy == "Batch cost"].set_index("layout")
        rng = np.random.default_rng(4577)
        indices = rng.integers(len(full_cost), size=(BOOTSTRAPS, len(full_cost)))
        for name, g in group.groupby("policy", sort=False):
            g = g.set_index("layout").sort_index()
            base = full_cost.loc[g.index]
            stats = {
                "service_pct": g.service_pct,
                "service_deficit_pp": g.service_deficit_pp,
                "cost_saving_pct": g.cost_saving_pct,
                "unserved_kwh": g.unserved_kwh,
                "distance_km": g.distance_km,
                "bill_usd": g.bill_usd,
                "wait_min": g.wait_min,
                "charge_min": g.charge_min,
            }
            if "distance_p95_km" in g:
                stats["distance_p95_km"] = g.distance_p95_km
                stats["distance_p95_minus_batch_km"] = g.distance_p95_km - base.distance_p95_km
            for p in (50, 100, 200):
                stats[f"fallback_{p}_excess"] = g[f"fallback_{p}_usd"] - base[f"fallback_{p}_usd"]
            row = dict(
                city=city,
                case=case,
                load=g.load.iloc[0],
                policy=name,
                layouts=len(g),
                feasible=int(g.full_feasible.sum()),
                n=g.requests.iloc[0],
            )
            for key, v in stats.items():
                a = v.to_numpy()
                row[key] = np.mean(a)
                row[key + "_lo"], row[key + "_hi"] = np.quantile(
                    np.mean(a[indices], axis=1), [0.025, 0.975]
                )
            summaries.append(row)
    return pd.DataFrame(summaries)


def factorial(raw):
    """Whole-fleet gain from batching, on the same requested EVs and common orders."""
    rows = []
    for (city, case), g in raw.groupby(["city", "case"], sort=False):
        for penalty in (50, 100, 200):
            pivot = g.pivot(index="layout", columns="policy", values=f"fallback_{penalty}_usd")
            full = pivot["Sequential cost"] - pivot["Batch cost"]
            distance = pivot["Nearest"] - pivot["Batch distance"]
            rng = np.random.default_rng(4577)
            samples = rng.integers(len(pivot), size=(BOOTSTRAPS, len(pivot)))
            for name, values in [
                ("batching given full cost", full),
                ("batching given distance", distance),
                ("difference of batching effects", full - distance),
            ]:
                v = values.to_numpy()
                lo, hi = np.quantile(v[samples].mean(axis=1), [0.025, 0.975])
                rows.append(
                    dict(
                        city=city,
                        case=case,
                        fallback_usd=penalty,
                        contrast=name,
                        usd_per_request=v.mean(),
                        lo=lo,
                        hi=hi,
                    )
                )
    return pd.DataFrame(rows)


def interval(row, key, digits=2):
    nums = [row[key + s] for s in ("", "_lo", "_hi")]
    return f"{nums[0]:.{digits}f} [{nums[1]:.{digits}f}, {nums[2]:.{digits}f}]"


def write_rows(path, rows):
    path.write_text("\n".join(" & ".join(row) + r" \\" for row in rows) + "\n")


def city_table(summary, path):
    cities = []
    for city in FILES:
        for i, p in enumerate(POLICIES):
            r = summary[(summary.city == city) & (summary.case == "base") & (summary.policy == p)].iloc[
                0
            ]
            cities.append(
                [
                    city if i == 0 else "",
                    p,
                    f"{r.service_pct:.2f}",
                    interval(r, "cost_saving_pct") if i else "---",
                    f"{r.distance_km:.2f}",
                    f"{r.bill_usd:.2f}",
                    f"{r.wait_min:.1f}",
                    f"{r.charge_min:.1f}",
                ]
            )
    write_rows(path, cities)


def latex_rows(summary, destination):
    def selected(city, case, name):
        return summary[(summary.city == city) & (summary.case == case) & (summary.policy == name)].iloc[
            0
        ]

    city_table(summary, destination / "city_rows.tex")
    load = []
    for case in ("low load", "base", "medium-high load", "high load"):
        b = selected("Washington", case, "Batch cost")
        g = selected("Washington", case, "Sequential cost")
        load.append(
            [
                f"{b.load:.2f}",
                str(b.n),
                f"{b.feasible}/{b.layouts}",
                f"{b.service_pct:.2f}",
                f"{g.service_pct:.2f}",
                interval(g, "service_deficit_pp"),
                interval(g, "cost_saving_pct"),
            ]
        )
    write_rows(destination / "load_rows.tex", load)
    stress = []
    for case in (c for c in CASES if c.load == 0.5):
        g = selected("Washington", case.name, "Sequential cost")
        stress.append(
            [
                case.name.capitalize(),
                f"{g.feasible}/{g.layouts}",
                interval(g, "service_deficit_pp"),
                interval(g, "cost_saving_pct"),
                f"{g.bill_usd:.2f}",
                f"{g.charge_min:.1f}",
            ]
        )
    write_rows(destination / "sensitivity_rows.tex", stress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layouts < 2 or args.workers < 1:
        parser.error("Require at least two layouts and one worker")
    tasks = [
        (city, r, case)
        for city in FILES
        for case in (CASES if city == "Washington" else CASES[:1])
        for r in range(args.layouts)
    ]
    if args.workers == 1:
        records = [row for task in tasks for row in run_task(task)]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = [row for result in executor.map(run_task, tasks, chunksize=2) for row in result]
    args.output.mkdir(exist_ok=True, parents=True)
    raw = pd.DataFrame(records)
    summary = summarize(raw)
    raw.to_csv(args.output / "city_layouts.csv", index=False, float_format="%.10g")
    summary.to_csv(args.output / "city_summary.csv", index=False, float_format="%.10g")
    factorial(raw).to_csv(args.output / "city_factorial.csv", index=False, float_format="%.10g")
    latex_rows(summary, args.output)
    print(f"{len(tasks)} scenarios/layouts, {len(raw)} policy-layout records. Results in {args.output}")


if __name__ == "__main__":
    main()
