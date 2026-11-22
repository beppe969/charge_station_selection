"""Pair all-listed, publicly supported, and source-reconciled city capacity."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from access_audit import public_ports_by_coordinate
from revised_city_data import FILES
from revised_city_study import BOOTSTRAPS, RESULTS, Scenario, city_inventory, evaluate


ALL = "all listed"
PUBLIC = "public only"
RECONCILED = "source-reconciled public"


def capacities(city):
    inventory = city_inventory(city)
    arms = {ALL: inventory.levels}
    for arm in [PUBLIC, RECONCILED] if city == "Chicago" else [PUBLIC]:
        lookup = public_ports_by_coordinate(city, cap_source=(arm == RECONCILED))
        arms[arm] = np.array([lookup.get(tuple(loc), (0, 0, 0)) for loc in inventory.latlon])
    for counts in arms.values():
        assert np.all((counts >= 0) & (counts <= inventory.levels))
    if RECONCILED in arms:
        assert np.all(arms[RECONCILED] <= arms[PUBLIC])
    return arms


def run_one(task):
    city, rep, arms = task
    records = []
    for arm, counts in arms.items():
        selected = None if arm == ALL else counts
        rows = evaluate(city, rep, Scenario(), accessible_levels=selected)
        for r in rows:
            r.update(access=arm, sites=int(np.any(counts > 0, axis=1).sum()), ports=int(counts.sum()))
        records.extend(rows)
    return records


def access_summary(raw):
    rows = []
    for city, group in raw.groupby("city", sort=False):
        for policy, data in group.groupby("policy", sort=False):
            base = data[data.access == ALL].set_index("layout").sort_index()
            for arm, subset in data[data.access != ALL].groupby("access", sort=False):
                restricted = subset.set_index("layout").sort_index()
                assert base.index.equals(restricted.index)
                assert (restricted.requests == base.requests).all()
                assert (restricted.max_served <= base.max_served).all()
                layouts = len(base)
                samples = np.random.default_rng(4577).integers(layouts, size=(BOOTSTRAPS, layouts))
                row = dict(
                    city=city,
                    access=arm,
                    policy=policy,
                    layouts=layouts,
                    requested=base.requests.iloc[0],
                    ports_all=base.ports.iloc[0],
                    ports_retained=restricted.ports.iloc[0],
                    sites_all=base.sites.iloc[0],
                    sites_retained=restricted.sites.iloc[0],
                    feasible_all=base.full_feasible.sum(),
                    feasible_restricted=restricted.full_feasible.sum(),
                )
                for metric in ("max_served", "service_pct", "cost_saving_pct", "fallback_100_usd"):
                    original, public = base[metric].to_numpy(), restricted[metric].to_numpy()
                    if metric == "max_served":
                        metric = "max_service_pct"
                        original = 100 * original / base.requests.to_numpy()
                        public = 100 * public / restricted.requests.to_numpy()
                    for suffix, values in (
                        ("all", original),
                        ("restricted", public),
                        ("change", public - original),
                    ):
                        key = f"{metric}_{suffix}"
                        row[key] = values.mean()
                        row[key + "_lo"], row[key + "_hi"] = np.quantile(
                            values[samples].mean(axis=1), (0.025, 0.975)
                        )
                rows.append(row)
    return pd.DataFrame(rows)


def table(summary, destination):
    lines = []
    for _, r in summary[summary.policy == "Sequential cost"].iterrows():
        label = "Source-capped public" if r.access == RECONCILED else "Public"
        lines.append(
            " & ".join(
                [
                    r.city,
                    label,
                    f"{r.ports_retained}/{r.ports_all}",
                    f"{r.max_service_pct_all:.2f}/{r.max_service_pct_restricted:.2f}",
                    f"{r.service_pct_all:.2f}/{r.service_pct_restricted:.2f}",
                    f"{r.cost_saving_pct_all:.2f}",
                    f"{r.cost_saving_pct_restricted:.2f}",
                    f"[{r.cost_saving_pct_restricted_lo:.2f}, {r.cost_saving_pct_restricted_hi:.2f}]",
                ]
            )
            + r" \\"
        )
    (destination / "access_rows.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layouts < 2 or args.workers < 1:
        parser.error("Require at least two layouts and one worker")
    tasks = []
    for city in FILES:
        arms = capacities(city)
        tasks.extend((city, rep, arms) for rep in range(args.layouts))
    if args.workers == 1:
        rows = [row for task in tasks for row in run_one(task)]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = [row for batch in executor.map(run_one, tasks, chunksize=1) for row in batch]
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    summary = access_summary(raw)
    raw.to_csv(args.output / "access_layouts.csv", index=False, float_format="%.10g")
    summary.to_csv(args.output / "access_summary.csv", index=False, float_format="%.10g")
    table(summary, args.output)
    print(f"{len(tasks)} paired layouts, {len(raw)} policy-layout records. Results in {args.output}")


if __name__ == "__main__":
    main()
