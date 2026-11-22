"""Replay Tivoli Square reservations and measure delivery under shared site power.

Run this file with the environment in requirements.txt, after planning_study.py.
Each test uses the archived planning inputs. A scenario's time window starts
when the assigned vehicles plug in together. The two original customers retain
their bookings. See results/submission/power_summary.csv for the results
averaged over all test batches. Session finish times assume a driver can stay
until the entire request is filled, independently of the three departure windows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import planning_study as planning
from revised_city_data import load_city
from revised_city_study import exact_assignment

RESULTS = Path(__file__).resolve().parents[1] / "results"
WINDOWS = (4.0, 6.0, 8.0)
PORT_KW = float(planning.POWER[1])
CASES = (("Two ports", 2, 14.4), ("Three, shared", 3, 14.4), ("Three, upgraded", 3, 21.6))


def shortest_chain(offers, chosen, port_sites, target):
    """Fewest displaced reservations; ascending vehicle ID breaks equal-depth ties."""
    served = np.flatnonzero(chosen >= 0)
    parent = np.full(len(chosen), -2, dtype=int)
    frontier = np.flatnonzero(chosen < 0)
    parent[frontier] = -1
    while len(frontier):
        reached_target = frontier[offers[frontier, target]]
        if len(reached_target):
            vehicle = reached_target[0]
            chain = []
            while vehicle >= 0:
                chain.append(vehicle)
                vehicle = parent[vehicle]
            return np.asarray(chain[::-1])
        # Each driver can move to the port currently held by any new child.
        for vehicle in frontier:
            unseen = served[parent[served] == -2]
            children = unseen[offers[vehicle, port_sites[chosen[unseen]]]]
            parent[children] = vehicle
        frontier = np.flatnonzero(np.isin(parent, frontier))
    return np.empty(0, dtype=int)


def charge(demand, cap_kw):
    """Finish times and kWh delivered by each cutoff for simultaneous sessions."""
    if not len(demand):
        return np.empty(0), np.empty((len(WINDOWS), 0))
    levels = np.r_[0.0, np.unique(demand)]
    active = np.array([np.count_nonzero(demand > left) for left in levels[:-1]])
    rates = np.minimum(PORT_KW, cap_kw / active)
    times = np.r_[0.0, np.cumsum(np.diff(levels) / rates)]
    finish = times[np.searchsorted(levels, demand)]
    progress = np.interp(WINDOWS, times, levels)
    delivered = np.minimum(progress[:, None], demand)
    for window, energy in zip(WINDOWS, delivered):
        assert energy.sum() <= cap_kw * window + 1e-8
        assert np.all(energy <= PORT_KW * window + 1e-8)
        assert np.array_equal(finish <= window + 1e-10, energy >= demand - 1e-9)
    return finish, delivered


def assignment(layout, sites, kinds, target):
    offers = planning.admissible(layout, planning.DESIGN_RADIUS)
    cost = (
        planning.DIST_PRICE * layout.distance[:, sites]
        + layout.demand[:, None] * layout.tariff[sites]
        + planning.TIME_PRICE
        * (layout.wait[sites] + 60 * layout.demand[:, None] / planning.POWER[kinds])
    )
    chosen = exact_assignment(cost, offers[:, sites])
    served = np.flatnonzero(chosen >= 0)
    installed_cost = cost[served, chosen[served]].sum()
    incumbents = served[sites[chosen[served]] == target]
    chain = shortest_chain(offers, chosen, sites, target)
    if len(chain):
        assert len(incumbents) == 2 and not np.isin(chain, incumbents).any()
        expanded = chosen.copy()
        expanded[chain[:-1]] = chosen[chain[1:]]
        expanded[chain[-1]] = len(sites)
        served_after = np.flatnonzero(expanded >= 0)
        assert len(served_after) == len(served) + 1
        assert len(np.unique(expanded[served_after])) == len(served_after)
        assert offers[served_after, np.append(sites, target)[expanded[served_after]]].all()
    return incumbents, chain, len(served), installed_cost


def main():
    archive = RESULTS / "planning"
    meta = json.loads((archive / "study.json").read_text())
    city = load_city("Washington")
    sites, kinds = planning.port_inventory(city.levels)
    assert list(city.levels[meta["target"]]) == [0, 2, 0]
    baseline = pd.read_csv(archive / "intervention_layouts.csv").set_index(["layout", "policy"])
    # At 4 h, one session has finished and the other two share the released power.
    finish, delivered = charge(np.array([10.0, 30.0, 50.0]), 14.4)
    np.testing.assert_allclose(finish, 10 / 4.8 + np.array([0.0, 20.0, 40.0]) / 7.2)
    np.testing.assert_allclose(delivered[0], [10.0, 23.8, 23.8])

    records, sessions = [], []
    for rep in range(meta["test_layouts"]):
        layout = planning.draw(city, 3, rep)
        original, chain, served, cost = assignment(layout, sites, kinds, meta["target"])
        old = baseline.loc[(rep, "baseline")]
        assert served == old.served and abs(cost - old.common_cohort_cost) < 1e-6
        assert bool(len(chain)) == bool(baseline.loc[(rep, "target_socket")].additional)
        for case, ports, cap in CASES:
            customers = np.r_[original, chain[-1:]] if ports == 3 else original
            demand = layout.demand[customers]
            finish, energy = charge(demand, cap)
            delays = finish[: len(original)] - layout.demand[original] / PORT_KW
            if ports == 2 or cap >= 3 * PORT_KW - 1e-9:
                np.testing.assert_allclose(finish, demand / PORT_KW, atol=1e-10)
            inc = np.arange(len(customers)) < len(original)
            for j, vehicle in enumerate(customers):
                sessions.append(
                    dict(
                        layout=rep,
                        case=case,
                        vehicle=int(vehicle),
                        incumbent=bool(inc[j]),
                        request_kwh=demand[j],
                        untruncated_finish_h=finish[j],
                        **{f"delivered_{int(h)}h_kwh": energy[k, j] for k, h in enumerate(WINDOWS)},
                    )
                )
            for h, delivered in zip(WINDOWS, energy):
                complete = finish <= h + 1e-10
                records.append(
                    dict(
                        layout=rep,
                        case=case,
                        window_h=h,
                        ports=ports,
                        site_kw=cap,
                        rescued=int(len(chain) > 0),
                        displaced=max(len(chain) - 1, 0),
                        admitted=len(customers),
                        incumbent_count=len(original),
                        completed=int(complete.sum()),
                        delivered_kwh=delivered.sum(),
                        unmet_booked_kwh=(demand - delivered).sum(),
                        incumbent_completed=int(complete[inc].sum()),
                        extra_completed=int(complete[~inc].sum()),
                        incumbent_delivered_kwh=delivered[inc].sum(),
                        incumbent_delay_total_h=delays.sum(),
                    )
                )
    rows = pd.DataFrame(records)
    summary = rows.groupby(["window_h", "case"], sort=False).agg(
        layouts=("layout", "size"),
        mean_admitted=("admitted", "mean"),
        mean_completed=("completed", "mean"),
        mean_kwh=("delivered_kwh", "mean"),
        completed_incumbents=("incumbent_completed", "sum"),
        completed_extra=("extra_completed", "sum"),
    )
    paired = rows.set_index(["layout", "window_h", "case"])
    metrics = ["completed", "delivered_kwh", "incumbent_completed"]
    installed = paired.xs("Two ports", level="case")[metrics]
    for case, _, _ in CASES:
        current = paired.xs(case, level="case")
        diff = current[metrics] - installed
        for h in WINDOWS:
            by_batch = diff.xs(h, level="window_h")
            for metric in ("completed", "delivered_kwh"):
                # Each model sees the same 96 batches. Resample paired changes by batch.
                changes = by_batch[metric].to_numpy()
                lo, hi = planning.interval_mean(changes, np.random.default_rng(planning.SEED))
                summary.loc[(h, case), f"change_{metric}"] = np.mean(changes)
                summary.loc[(h, case), f"change_{metric}_lo95"] = lo
                summary.loc[(h, case), f"change_{metric}_hi95"] = hi
            summary.loc[(h, case), "improved_batches"] = int((by_batch.completed > 0).sum())
            summary.loc[(h, case), "worsened_batches"] = int((by_batch.completed < 0).sum())
            summary.loc[(h, case), "original_missed_deadline"] = int(
                -by_batch.incumbent_completed.sum()
            )
            assert np.all(by_batch.delivered_kwh >= -1e-8)
    dest = RESULTS / "submission"
    dest.mkdir(parents=True, exist_ok=True)
    rows.to_csv(dest / "power_layouts.csv", index=False, float_format="%.10g")
    pd.DataFrame(sessions).to_csv(dest / "power_sessions.csv", index=False, float_format="%.10g")
    summary.to_csv(dest / "power_summary.csv", float_format="%.10g")
    tex = []
    for case, ports, cap in CASES:
        cells = [str(ports), f"{cap:.1f}"]
        for h in WINDOWS:
            s = summary.loc[(h, case)]
            cells.extend([f"{s.mean_completed:.2f}", f"{s.mean_kwh:.1f}"])
        tex.append(" & ".join(cells) + r" \\")
    (dest / "power_rows.tex").write_text("\n".join(tex) + "\n")
    print(summary.round(3).to_string())
    shared = rows[(rows.case == "Three, shared") & (rows.window_h == 6)]
    touched = shared[shared.rescued == 1]
    print(
        "Existing sessions in affected layouts, mean untruncated delay (min):",
        touched.incumbent_count.sum(),
        60 * touched.incumbent_delay_total_h.sum() / touched.incumbent_count.sum(),
    )
    print("Reassignments on successful paths:", touched.displaced.value_counts().to_dict())


if __name__ == "__main__":
    main()
