"""Reproduce the travel-limit, vehicle-connector and incremental batching-delay results.

Run from the package root with python code/submission_sensitivities.py.
Uniform and independently drawn travel allowances are evaluated on the same
held-out Washington planning batches. In the three-city reference experiments,
connector access changes only the feasible offers for each driver. The mixed
scenario independently restricts each driver's access with probability one half.
Both mixtures are illustrative assumptions, not estimates of driver behavior.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import planning_study as planning
from access_audit import audit_records
from extended_city_analysis import REFERENCE, compare
from revised_city_data import DATA, FILES, load_city
from revised_city_study import BOOTSTRAPS, RESULTS, draw
from revised_primary_cities import public_inventory


TRAVEL_LIMITS = (2.0, 2.5, 3.0)
TRAVEL_SEED = 62119
CONNECTOR_SEED = 20261122
BOOTSTRAP_SEED = 4577


def travel_scenarios():
    directory = RESULTS / "planning"
    settings = json.loads((directory / "study.json").read_text())
    reference = pd.read_csv(directory / "intervention_layouts.csv").pivot(
        index="layout", columns="policy", values="served")
    city = load_city("Washington")
    ports, _ = planning.port_inventory(city.levels)
    candidates = np.flatnonzero(city.levels[:, 1] > 0)
    target, comparison = settings["target"], settings["control"]
    n = settings["test_layouts"]
    cases = [f"{radius:.1f}" for radius in TRAVEL_LIMITS] + ["Mixed"]
    counts = {case: np.zeros(len(city.xy), dtype=int) for case in cases}
    layout_records = []
    for layout in range(n):
        batch = planning.draw(city, 3, layout)
        rng = np.random.default_rng([TRAVEL_SEED, layout])
        mixed = rng.choice(TRAVEL_LIMITS, size=(len(batch.xy), 1))
        for case, radius in zip(cases, [*TRAVEL_LIMITS, mixed]):
            offers = planning.admissible(batch, radius)
            chosen = planning.match(offers, ports)
            unserved = np.flatnonzero(chosen < 0)
            served = len(chosen) - len(unserved)
            _, reachable = planning.hall_witness(offers, chosen, ports, unserved)
            counts[case] += reachable
            row = dict(travel_limit_km=case, layout=layout, requests=len(chosen),
                       baseline_served=served, baseline_full=not len(unserved))
            radii = np.broadcast_to(radius, (len(chosen), 1))
            for limit in TRAVEL_LIMITS:
                row[f"drivers_with_{limit:.1f}_km"] = np.count_nonzero(radii == limit)
            for name, site in (("target", target), ("comparison", comparison)):
                expanded = planning.match(offers, np.append(ports, site))
                additional = np.count_nonzero(expanded >= 0) - served
                if additional != int(reachable[site]):
                    raise AssertionError("Alternating reachability disagrees with rematching")
                row[f"{name}_additional"] = additional
            if case == f"{planning.DESIGN_RADIUS:.1f}":
                observed = reference.loc[layout, ["baseline", "target_socket", "control_socket"]]
                np.testing.assert_array_equal(observed, [served, served + row["target_additional"],
                                                         served + row["comparison_additional"]])
            layout_records.append(row)

    layouts = pd.DataFrame(layout_records)
    summary, rankings = [], []
    for case, wins in counts.items():
        by_case = layouts[layouts.travel_limit_km == case]
        best = wins[candidates].max()
        lo, hi = planning.binomial_interval(wins[target], n)
        summary.append(dict(travel_limit_km=case, batches=n, baseline_full=by_case.baseline_full.sum(),
                            target_improved=wins[target], comparison_improved=wins[comparison],
                            candidates=len(candidates), best_improved=best,
                            n_sites_tied_for_best=np.count_nonzero(wins[candidates] == best),
                            target_success_pct=100 * wins[target] / n,
                            target_exact95_low_pct=100 * lo, target_exact95_high_pct=100 * hi))
        rankings.extend(dict(travel_limit_km=case, site=site, latitude=city.latlon[site, 0],
                             longitude=city.latlon[site, 1], improved=wins[site], batches=n)
                        for site in candidates)
    return layouts, pd.DataFrame(summary), pd.DataFrame(rankings)


def connector_counts(city, inventory, national):
    # Only a reconciled public record whose matching national records ALL say
    # TESLA alone can remove fast offers. Preserve the slower ports at that site.
    records = audit_records(city)
    fast = records[(records.classification == "public_only") & records.within_source_count
                   & (records.dcfast > 0)]
    excluded = Counter()
    for record in fast.itertuples():
        connectors = {national[int(i) - 1]["EV Connector Types"].strip().upper()
                      for i in record.national_records.split("|")}
        if connectors == {"TESLA"}:
            excluded[(record.latitude, record.longitude)] += record.dcfast
    allowed = inventory.levels.copy()
    for site, coord in enumerate(inventory.latlon):
        allowed[site, 2] -= excluded[tuple(coord)]
    if np.any(allowed < 0) or (inventory.levels - allowed).sum() != sum(excluded.values()):
        raise AssertionError("Excluded connector count does not reconcile with the inventory")
    return allowed


def city_scenarios():
    with (DATA / "USA_EV_CS.csv").open(encoding="cp1252", newline="") as stream:
        national = list(csv.DictReader(stream, delimiter=";"))
    ref = pd.read_csv(RESULTS / "extended" / "city_cost_by_order.csv")
    ref = ref[ref.power_case == "Class ratings"].set_index(["city", "layout", "order"])
    records, summary = [], []
    for city_index, city in enumerate(FILES):
        inventory = public_inventory(city)
        allowed = connector_counts(city, inventory, national)
        city_records = []
        for layout in ref.loc[city].index.get_level_values("layout").unique():
            nominal = draw(inventory, city, layout, REFERENCE, return_port_classes=True)
            restricted = draw(inventory, city, layout, REFERENCE, accessible_levels=allowed,
                              return_port_classes=True)
            rng = np.random.default_rng([CONNECTOR_SEED, city_index, layout])
            restricted_driver = rng.random(len(nominal[2])) < 0.5
            mixed_mask = np.where(restricted_driver[:, None], restricted[7], nominal[7])
            np.testing.assert_array_equal(nominal[9], restricted[9])
            np.testing.assert_array_equal(nominal[8], restricted[8])
            if np.any(restricted[7] & ~nominal[7]):
                raise AssertionError("Connector restriction introduced a previously infeasible offer")
            for case, mask, count in (("reference", nominal[7], 0),
                                      ("mixed", mixed_mask, restricted_driver.sum()),
                                      ("restricted", restricted[7], len(restricted_driver))):
                inputs = (*nominal[:7], mask, *nominal[8:])
                observations = compare(inputs[8], inputs[6], inputs, layout, city, case)
                for row in observations:
                    row["connector_scenario"] = row.pop("power_case")
                    row["restricted_drivers"] = count
                    row["requests"] = len(restricted_driver)
                    if case == "reference":
                        keys = ["served", "cost_total", "cohort_opt_total", "gain", "saving_pct"]
                        np.testing.assert_allclose([row[k] for k in keys],
                                                   ref.loc[(city, layout, row["order"]), keys].to_numpy(dtype=float),
                                                   rtol=2e-9)
                city_records.extend(observations)
        records.extend(city_records)
        frame = pd.DataFrame(city_records)
        means = frame.groupby(["layout", "connector_scenario"]).mean(numeric_only=True)
        savings = means.saving_pct.unstack()
        gain = means.gain.unstack()
        samples = np.random.default_rng(BOOTSTRAP_SEED).integers(
            len(savings), size=(BOOTSTRAPS, len(savings)))
        stats = dict(city=city, layouts=len(savings), orders=frame.order.nunique(),
                     total_fast_ports=inventory.levels[:, 2].sum(),
                     restricted_fast_ports=(inventory.levels - allowed).sum())
        values = {f"{case}_saving_pct": savings[case] for case in savings}
        for case in ("mixed", "restricted"):
            values[f"{case}_minus_reference_pp"] = savings[case] - savings.reference
        for case in gain:
            # Break-even mean extra minutes per sequentially served driver.
            values[f"{case}_gain_usd_per_driver"] = gain[case]
            values[f"{case}_breakeven_min"] = gain[case] / REFERENCE.time_value
        for name, values_by_layout in values.items():
            vector = values_by_layout.to_numpy()
            stats[name] = vector.mean()
            lo, hi = np.quantile(vector[samples].mean(axis=1), [.025, .975])
            stats[f"{name}_low95"], stats[f"{name}_high95"] = lo, hi
        summary.append(stats)
    return pd.DataFrame(records), pd.DataFrame(summary)


def write_table(destination, name, rows):
    lines = [" & ".join(row) + r" \\" for row in rows]
    (destination / f"sensitivity_{name}_rows.tex").write_text("\n".join(lines) + "\n")


def interval_endpoint(percent):
    # Keep small, positive exact limits distinguishable from an exact zero.
    return f"{percent:.2f}" if 0 < percent < 0.1 else f"{percent:.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "submission")
    out = parser.parse_args().output
    layouts, radii, candidates = travel_scenarios()
    orders, cities = city_scenarios()
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("radius_layouts", layouts), ("radius_summary", radii),
                       ("radius_candidates", candidates), ("city_orders", orders),
                       ("city_summary", cities)):
        data.to_csv(out / f"sensitivity_{name}.csv", index=False, float_format="%.10g")
    write_table(out, "radius", [
        [r.travel_limit_km, str(r.baseline_full), str(r.target_improved), str(r.comparison_improved),
         str(r.best_improved),
         (f"{r.target_success_pct:.1f} [{interval_endpoint(r.target_exact95_low_pct)}, "
          f"{interval_endpoint(r.target_exact95_high_pct)}]")]
        for r in radii.itertuples()])
    write_table(out, "city", [
        [r.city, f"{r.restricted_fast_ports}/{r.total_fast_ports}", f"{r.reference_saving_pct:.2f}",
         f"{r.mixed_saving_pct:.2f}", f"{r.restricted_saving_pct:.2f}",
         (f"${r.restricted_minus_reference_pp:.2f}$ "
          f"[${r.restricted_minus_reference_pp_low95:.2f}, {r.restricted_minus_reference_pp_high95:.2f}$]"),
         f"{r.reference_breakeven_min:.1f}"]
        for r in cities.itertuples()])
    print(radii.to_string(index=False))
    print(cities[["city", "reference_saving_pct", "mixed_saving_pct", "restricted_saving_pct",
                  "reference_breakeven_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
