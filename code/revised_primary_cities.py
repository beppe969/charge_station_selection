"""Run the primary half-load comparisons on source-reconciled public inventories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from access_audit import audit_records, public_ports_by_coordinate
from revised_city_data import FILES, LEVELS, inventory_from_records
from revised_city_study import RESULTS, Scenario, city_table, evaluate, factorial, summarize


def public_inventory(city):
    audit = audit_records(city)
    public = audit[(audit.classification == "public_only") & audit.within_source_count]
    columns = dict(
        zip(("latitude", "longitude", "lvl1", "lvl2", "dcfast"), ("Latitude", "Longitude", *LEVELS))
    )
    inventory = inventory_from_records(public.rename(columns=columns))
    lookup = public_ports_by_coordinate(city, cap_source=True)
    assert np.array_equal(inventory.levels, [lookup[tuple(loc)] for loc in inventory.latlon])
    return inventory


def run_one(task):
    city, rep, inventory = task
    return evaluate(city, rep, Scenario(), inventory=inventory, trip_tail=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layouts < 2 or args.workers < 1:
        parser.error("Require at least two layouts and one worker")
    tasks, inventory_rows = [], []
    for city in FILES:
        inventory = public_inventory(city)
        l1, l2, dc = inventory.levels.sum(axis=0)
        ports = l1 + l2 + dc
        inventory_rows.append(
            dict(
                city=city,
                records=inventory.n_records,
                sites=len(inventory.xy),
                L1=l1,
                L2=l2,
                DC=dc,
                ports=ports,
                requests=(ports + 1) // 2,
                latitude0=inventory.latitude0,
                longitude0=inventory.longitude0,
            )
        )
        tasks.extend((city, rep, inventory) for rep in range(args.layouts))
    if args.workers == 1:
        rows = [row for task in tasks for row in run_one(task)]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = [row for batch in executor.map(run_one, tasks, chunksize=2) for row in batch]
    args.output.mkdir(exist_ok=True, parents=True)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    for name, frame in (
        ("layouts", raw),
        ("summary", summary),
        ("factorial", factorial(raw)),
        ("inventory", pd.DataFrame(inventory_rows)),
    ):
        frame.to_csv(args.output / f"primary_city_{name}.csv", index=False, float_format="%.10g")
    city_table(summary, args.output / "primary_city_rows.tex")
    print(f"{len(tasks)} layouts, {len(rows)} policy-layout records. Results in {args.output}")


if __name__ == "__main__":
    main()
