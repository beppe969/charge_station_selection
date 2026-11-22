"""Washington feasibility frontier and a held-out charging infrastructure experiment.

Run from any directory: python planning_study.py
The inventory, spatial sampler and coordinate origin are shared with the city study.
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist
from scipy.stats import beta

from revised_city_data import draw_positions, load_city


RANGES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, np.inf)
CENTERS = (0.42, 0.46, 0.50, 0.55, 0.60)
NOMINAL = 0.50
DESIGN_RADIUS = 2.0
POWER = np.array([1.4, 7.2, 50.0])
DIST_PRICE = 2.52
TIME_PRICE = 0.224
TARIFF_MIN = 0.303
SEED = 34019
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "planning"


@dataclass
class Layout:
    xy: np.ndarray
    distance: np.ndarray
    tariff: np.ndarray
    rank: np.ndarray
    demand: np.ndarray
    wait: np.ndarray


def draw(city, split, number):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, split, number]))
    vehicles = (city.levels.sum() + 1) // 2
    xy = draw_positions(city, vehicles, rng)
    return Layout(xy, cdist(xy, city.xy), rng.uniform(TARIFF_MIN, .530, len(city.xy)),
                  rng.random(vehicles), rng.uniform(20, 60, vehicles),
                  rng.uniform(0, 15, len(city.xy)))


def budget_cap(layout, center=NOMINAL):
    return center - .1 + .2 * layout.rank


def admissible(layout, radius, tariff=None):
    if tariff is None:
        tariff = layout.tariff
    return ((layout.distance <= radius) &
            (tariff[None, :] <= budget_cap(layout)[:, None]))


def port_inventory(levels):
    site, kind = np.nonzero(levels)
    counts = levels[site, kind]
    return np.repeat(site, counts), np.repeat(kind, counts)


def match(graph_by_site, port_sites):
    graph = csr_matrix(graph_by_site[:, port_sites])
    return maximum_bipartite_matching(graph, perm_type="column")


def binomial_interval(success, n):
    # SciPy 1.7's Boost implementation can overflow internally while finding a
    # finite beta quantile near a boundary. Validate both returned endpoints.
    with np.errstate(over="ignore"):
        lo = 0.0 if success == 0 else beta.ppf(.025, success, n - success + 1)
        hi = 1.0 if success == n else beta.ppf(.975, success + 1, n - success)
    if not 0 <= lo <= hi <= 1:
        raise ArithmeticError(f"Invalid exact binomial interval for {success} of {n}")
    return float(lo), float(hi)


def exact_threshold(distance, affordable, sites, feasible_at_full_range, deficits):
    """First attainable edge length at which a complete matching exists."""
    if not feasible_at_full_range:
        return np.inf
    low = max((r for r, deficit in deficits.items() if deficit), default=-np.inf)
    high = min((r for r, deficit in deficits.items() if not deficit), default=np.inf)
    candidates = np.unique(distance[affordable & (distance > low) & (distance <= high)])
    left, right = 0, len(candidates) - 1
    if right < 0:
        raise AssertionError("No eligible distances in threshold bracket")
    while left < right:
        mid = (left + right) // 2
        chosen = match(affordable & (distance <= candidates[mid]), sites)
        if np.all(chosen >= 0):
            right = mid
        else:
            left = mid + 1
    return float(candidates[left])


def sweep(city, sites, n_draw):
    cap = city.levels.sum(axis=1)
    layouts, thresholds = [], []
    for number in range(n_draw):
        l = draw(city, 1, number)
        n = len(l.xy)
        for center in CENTERS:
            affordable = l.tariff[None, :] <= budget_cap(l, center)[:, None]
            deficits = {}
            for radius in RANGES:
                allowed = affordable & (l.distance <= radius)
                degrees = np.sum(allowed * cap, axis=1)
                ordered = np.sort(degrees)
                cert = bool(np.all(ordered >= np.arange(1, n + 1)))
                if cert:
                    deficit = 0
                elif np.isinf(radius):
                    # Nested affordable sets. The largest prefix deficit is exactly
                    # the number left unmatched at infinite range.
                    deficit = int(np.max(np.arange(1, n + 1) - ordered))
                else:
                    deficit = int(np.sum(match(allowed, sites) < 0))
                if deficits and deficit > min(deficits.values()):
                    raise AssertionError("Service decreased as range increased")
                deficits[radius] = deficit
                layouts.append((number, radius, center, deficit, np.any(degrees == 0),
                                cert, ordered[0] >= n))
            threshold = exact_threshold(l.distance, affordable, sites, deficits[np.inf] == 0,
                                        {r: d for r, d in deficits.items() if np.isfinite(r)})
            thresholds.append(dict(layout=number, budget_center=center, min_full_range_km=threshold))
    cells = []
    for center in CENTERS:
        for radius in RANGES:
            group = [row for row in layouts if row[1] == radius and row[2] == center]
            full = sum(r[3] == 0 for r in group)
            lo, hi = binomial_interval(full, n_draw)
            cells.append(dict(budget_center=center, radius_km=radius, layouts=n_draw,
                              full_count=full, probability=full/n_draw, lower95=lo, upper95=hi,
                              mean_deficit=float(np.mean([r[3] for r in group])),
                              no_isolates=sum(int(not r[4]) for r in group),
                              collective_only=sum(int(not r[4] and r[3] > 0) for r in group),
                              degree_certified=sum(int(r[5]) for r in group),
                              min_degree_certified=sum(int(r[6]) for r in group)))
    return cells, thresholds


def hall_witness(allowed, chosen, sites, roots):
    """Alternating reachability from free vehicles; ports at a site share eligibility."""
    ev = np.zeros(len(chosen), dtype=bool)
    ev[roots] = True
    covered = np.zeros(allowed.shape[1], dtype=bool)
    matched = np.flatnonzero(chosen >= 0)
    neighbors_of_matched = sites[chosen[matched]]
    while True:
        reached = np.any(allowed[ev], axis=0)
        if np.array_equal(covered, reached):
            break
        covered = reached
        ev[matched[covered[neighbors_of_matched]]] = True
    if int(ev.sum()) - int(covered[sites].sum()) != len(roots):
        raise AssertionError("Invalid Hall deficiency certificate")
    return ev, covered


def select_sites(city, sites, discovery):
    """Rank single-port additions by the number of development layouts rescued by one EV."""
    scores = np.zeros(len(city.xy), dtype=int)
    cache = []
    for number in range(discovery):
        l = draw(city, 2, number)
        allowed = admissible(l, DESIGN_RADIUS)
        chosen = match(allowed, sites)
        roots = np.flatnonzero(chosen < 0)
        if not len(roots):
            continue
        _, neighbors = hall_witness(allowed, chosen, sites, roots)
        scores += neighbors
        cache.append((number, allowed, chosen, roots))
    candidates = np.flatnonzero(city.levels[:, 1] > 0)
    target = int(candidates[np.argmax(scores[candidates])])
    if scores[target] == 0:
        raise RuntimeError("Discovery sample contains no eligible contested site")

    # Select a reproducible, concise collective witness for this intervention.
    # Prefer a layout with no isolated vehicle, then a small witness, then layout/root.
    best_key = None
    for number, allowed, chosen, roots in cache:
        # A free EV with no edge is a singleton witness; it cannot gain from an extra port.
        no_isolates = np.all(np.any(allowed, axis=1))
        for root in roots[np.any(allowed[roots], axis=1)]:
            ev, neighbor = hall_witness(allowed, chosen, sites, [root])
            if neighbor[target]:
                key = (not no_isolates, ev.sum(), number, root)
                if best_key is None or key < best_key:
                    best_key = key
                    witness = (number, int(root), ev, neighbor, int(len(roots)))
    if best_key is None:
        raise AssertionError("Target lacks the witness used for site selection")
    _, _, _, neighbor, _ = witness
    # Equal capacity intervention at a site of similar size/radial location.
    # Require a separate part of the city, outside the illustrated neighborhood.
    distance_from_target = np.linalg.norm(city.xy - city.xy[target], axis=1)
    controls = candidates[(~neighbor[candidates]) & (distance_from_target[candidates] >= 1.0)]
    if not len(controls):
        raise RuntimeError("No physically separated comparison location")
    size = city.levels.sum(axis=1)
    radial = np.linalg.norm(city.xy, axis=1)
    control = min(controls, key=lambda i: (abs(int(size[i])-int(size[target])),
                                          abs(radial[i]-radial[target]), int(i)))
    return target, int(control), witness, scores


def solve_cost(layout, sites, kinds, tariff, allowed, cohort=None):
    """Minimum generalized cost at maximum cardinality, or complete a specified cohort."""
    vehicles = np.arange(len(layout.xy)) if cohort is None else cohort
    demand = layout.demand[vehicles, None]
    # All modeled vehicle acceptance rates are at least the nominal 7.2/50 kW.
    duration = 60 * demand / POWER[kinds]
    costs = (DIST_PRICE * layout.distance[vehicles][:, sites] + demand * tariff[sites] +
             TIME_PRICE * (layout.wait[sites] + duration))
    legitimate = allowed[vehicles][:, sites]
    if cohort is None:
        # Removing one dummy saves more than the largest possible sum of real-edge costs.
        largest_finite = np.max(costs, where=np.isfinite(costs), initial=0.0)
        penalty = len(vehicles) * float(largest_finite) + 1.0
        extended = np.full((len(vehicles), len(sites) + len(vehicles)), np.inf)
        extended[:, :len(sites)] = np.where(legitimate, costs, np.inf)
        extended[np.arange(len(vehicles)), len(sites) + np.arange(len(vehicles))] = penalty
    else:
        extended = np.where(legitimate, costs, np.inf)
    row, col = linear_sum_assignment(extended)
    service = col < len(sites)
    row, col = row[service], col[service]
    if not np.all(legitimate[row, col]):
        raise AssertionError("An assigned port violates the original eligibility rules")
    return vehicles[row], float(np.sum(costs[row, col]))


def evaluate_layout(city, layout, target, control):
    sites, kinds = port_inventory(city.levels)
    allowed = admissible(layout, DESIGN_RADIUS)
    cohort, base_cost = solve_cost(layout, sites, kinds, layout.tariff, allowed)
    bserved = len(cohort)
    if bserved != np.count_nonzero(match(allowed, sites) >= 0):
        raise AssertionError("Cardinality and cost algorithms disagree")
    base = dict(policy="baseline", served=bserved, deficit=len(layout.xy)-bserved, additional=0,
                common_cohort_cost=base_cost, cohort_saving_usd_per_ev=0.0)
    rows = [base]
    for name, location in (("target_socket", target), ("control_socket", control),
                           ("target_discount", target)):
        rates = layout.tariff.copy()
        if name == "target_discount":
            rates[location] = max(TARIFF_MIN, rates[location] - .05)
            newsites, newkinds = sites, kinds
        else:
            newsites, newkinds = np.append(sites, location), np.append(kinds, 1)
        possible = admissible(layout, DESIGN_RADIUS, rates)
        count = int(np.sum(match(possible, newsites) >= 0))
        _, cost = solve_cost(layout, newsites, newkinds, rates, possible, cohort=cohort)
        if count < bserved or (name.endswith("socket") and count > bserved + 1):
            raise AssertionError("Capacity intervention cardinality is inconsistent")
        if cost > base_cost + 1e-7:
            raise AssertionError("Extending options increased cost for a fixed cohort")
        rows.append(dict(policy=name, served=count, deficit=len(layout.xy)-count,
                         additional=count-bserved, common_cohort_cost=cost,
                         cohort_saving_usd_per_ev=(base_cost-cost)/bserved))
    return rows


def interval_mean(values, rng):
    values = np.asarray(values)
    means = np.mean(values[rng.integers(len(values), size=(5000, len(values)))], axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def write_rows(file, rows):
    with file.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_witness(out, city, sites, target, control, witness, scores):
    number, root, e, neighbors, deficit = witness
    l = draw(city, 2, number)
    rows = evaluate_layout(city, l, target, control)
    if rows[1]["additional"] != 1 or rows[2]["additional"] not in (0, 1):
        raise AssertionError("Target socket failed to increase maximum cardinality")
    idx = np.flatnonzero(e)
    site_idx = np.flatnonzero(neighbors)
    # EVs sharing this group must collectively forgo at least its smallest kWh requests
    # up to the certified deficit (one for this single-root witness).
    data = dict(discovery_layout=number, root_vehicle=root, range_km=DESIGN_RADIUS,
                budget_center=NOMINAL, full_network_deficit=deficit,
                witness_ev_count=len(idx), witness_port_count=int(neighbors[sites].sum()),
                witness_requested_kwh=float(l.demand[e].sum()),
                witness_min_unserved_kwh=float(l.demand[e].min()),
                any_isolated=bool(np.any(~np.any(admissible(l, DESIGN_RADIUS), axis=1))),
                vehicles=[dict(id=int(i), east_km=float(l.xy[i, 0]), north_km=float(l.xy[i, 1]),
                               kwh=float(l.demand[i]), budget_per_kwh=float(budget_cap(l)[i]))
                          for i in idx],
                neighbor_sites=[dict(id=int(i), latitude=float(city.latlon[i, 0]),
                                     longitude=float(city.latlon[i, 1]),
                                     ports=int(city.levels[i].sum()), tariff=float(l.tariff[i]))
                                for i in site_idx],
                target_site=int(target), control_site=int(control),
                discovery_target_hits=int(scores[target]), discovery_control_hits=int(scores[control]),
                intervention_results=rows)
    (out/"witness.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def figure(path, city, cells, example, target, control):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.1, 4.0), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.05, 1]})
    for center, color in zip(CENTERS, ["#94411b", "#bc6931", "#13665c", "#336f9c", "#6a5c8b"]):
        group = [r for r in cells if np.isfinite(r["radius_km"]) and r["budget_center"] == center]
        x = [r["radius_km"] for r in group]
        y = [r["probability"] for r in group]
        left.plot(x, y, marker="o", ms=3.5, color=color, label=f"{center:.2f}", lw=1.6)
        if center == NOMINAL:
            left.fill_between(x, [r["lower95"] for r in group], [r["upper95"] for r in group],
                              color=color, alpha=.13)
    left.set(xlabel="Driving radius (km)", ylabel="Fraction with complete service",
             ylim=(-.035, 1.055), title="(a)  Budget and range")
    left.grid(axis="y", alpha=.23)
    left.legend(loc="lower right", title="Band centre ($/kWh)", ncol=2, fontsize=8, title_fontsize=8)
    xy = np.array([[r["east_km"], r["north_km"]] for r in example["vehicles"]])
    ni = [r["id"] for r in example["neighbor_sites"]]
    right.scatter(*city.xy.T, s=5, fc="#bbc5c3", ec="none", label="Other site")
    right.scatter(*city.xy[ni].T, s=22, fc="#4c77aa", ec="none", label="Eligible site")
    right.scatter(*xy.T, s=21, marker="^", fc="#a55520", ec="none", label="Witness EV")
    right.scatter(*city.xy[target], s=103, marker="*", color="#0a4d44", zorder=5, label="+1 target")
    right.scatter(*city.xy[control], s=59, marker="+", color="#39234b", linewidths=1.6, label="+1 control")
    right.set(xlabel="East of origin (km)", ylabel="North of origin (km)",
              title="(b)  Deficient group in Washington")
    # The map covers all witness EVs, their complete neighborhood and the control.
    focus = np.vstack((xy, city.xy[ni], city.xy[control]))
    right.set_xlim(focus[:, 0].min()-.8, focus[:, 0].max()+.8)
    right.set_ylim(focus[:, 1].min()-.8, focus[:, 1].max()+.8)
    right.set_aspect("equal", adjustable="datalim")
    right.legend(loc="best", fontsize=7.2, framealpha=.93)
    for ax in (left, right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sweep", type=int, default=64)
    parser.add_argument("--discovery", type=int, default=32)
    parser.add_argument("--test", type=int, default=96)
    args = parser.parse_args()
    if min(args.sweep, args.discovery, args.test) < 1:
        parser.error("All three sample sizes must be positive")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    city = load_city("Washington")
    sites, _ = port_inventory(city.levels)
    cells, thresholds = sweep(city, sites, args.sweep)
    write_rows(out/"feasibility.csv", cells)
    write_rows(out/"thresholds.csv", thresholds)
    target, control, witness, scores = select_sites(city, sites, args.discovery)
    example = save_witness(out, city, sites, target, control, witness, scores)
    rows = []
    for number in range(args.test):
        for row in evaluate_layout(city, draw(city, 3, number), target, control):
            rows.append(dict(layout=number, **row))
    write_rows(out/"intervention_layouts.csv", rows)
    summary = []
    for name in ("baseline", "target_socket", "control_socket", "target_discount"):
        subset = [r for r in rows if r["policy"] == name]
        d = [r["additional"] for r in subset]
        k = sum(i > 0 for i in d)
        lo, hi = binomial_interval(k, args.test)
        gain = [r["cohort_saving_usd_per_ev"] for r in subset]
        gl, gh = interval_mean(gain, np.random.default_rng(SEED))
        summary.append(dict(policy=name, layouts=args.test, mean_additional=float(np.mean(d)),
                            count_improved=k, improvement_probability_lower95=lo,
                            improvement_probability_upper95=hi,
                            full_service=sum(r["deficit"] == 0 for r in subset),
                            mean_deficit=float(np.mean([r["deficit"] for r in subset])),
                            mean_common_saving_usd_per_ev=float(np.mean(gain)),
                            saving_lower95=gl, saving_upper95=gh))
    write_rows(out/"interventions.csv", summary)
    groups = {name:np.array([r["additional"] for r in rows if r["policy"]==name])
              for name in ("target_socket", "control_socket")}
    difference = groups["target_socket"]-groups["control_socket"]
    meta = dict(seed=SEED, sweep_layouts=args.sweep, discovery_layouts=args.discovery,
                test_layouts=args.test, ports=len(sites), vehicles=(len(sites)+1)//2,
                sites=len(city.xy), target=int(target), control=int(control),
                target_latlon=city.latlon[target].tolist(), control_latlon=city.latlon[control].tolist(),
                target_ports=city.levels[target].tolist(), control_ports=city.levels[control].tolist(),
                target_minus_control_additional_mean=float(difference.mean()),
                target_minus_control_paired_bootstrap95=interval_mean(difference, np.random.default_rng(SEED)),
                bootstrap_resamples=5000, infrastructure_assumption="7.2 kW incremental headroom at each candidate",
                site_id="zero-based index in increasing latitude then longitude")
    (out/"study.json").write_text(json.dumps(meta, indent=2)+"\n")
    figure(out/"planning.pdf", city, cells, example, target, control)
    print(json.dumps(meta))
    print("Held-out intervention means:", [(s["policy"], s["mean_additional"]) for s in summary])


if __name__ == "__main__":
    main()
