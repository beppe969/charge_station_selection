"""Build the city comparisons and figures added to main_extended.tex.

Run after the primary study. Nominal totals and cohort optima are checked
against the archived policy observations for every city and layout.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import planning_study
from revised_city_data import FILES, load_city
from revised_city_study import RESULTS, Scenario, choose_sequential, draw, exact_assignment
from revised_primary_cities import public_inventory

OUTPUT = RESULTS / "extended"
PARTS = ["travel", "bill", "wait", "charging"]
REFERENCE = Scenario()


def compare(cost, charge, inputs, rep, city, power_case):
    sites, distance, q, prices, waits, bill, _, eligible, _, orders, _ = inputs
    records = []
    for order_id, order in enumerate(orders):
        choice = choose_sequential("Sequential cost", order, eligible, cost, distance, sites, prices)
        ev = np.flatnonzero(choice >= 0)
        seq_ports = choice[ev]
        joint_ports = exact_assignment(cost[ev], eligible[ev])
        assert np.all(joint_ports >= 0)
        assert eligible[ev, joint_ports].all() and len(np.unique(joint_ports)) == len(ev)
        before = cost[ev, seq_ports].sum()
        after = cost[ev, joint_ports].sum()
        gains = [
            REFERENCE.distance_value * np.mean(distance[ev, sites[seq_ports]] - distance[ev, sites[joint_ports]]),
            np.mean(bill[ev, seq_ports] - bill[ev, joint_ports]),
            REFERENCE.time_value * np.mean(waits[sites[seq_ports]] - waits[sites[joint_ports]]),
            REFERENCE.time_value * np.mean(charge[ev, seq_ports] - charge[ev, joint_ports]),
        ]
        assert np.isclose(sum(gains), (before - after) / len(ev))
        records.append(dict(city=city, layout=rep, order=order_id, power_case=power_case,
                            served=len(ev), service_pct=100 * len(ev) / len(q),
                            cost_total=before, cohort_opt_total=after, gain=sum(gains),
                            saving_pct=100 * (before - after) / before, **dict(zip(PARTS, gains))))
    return records


def city_results(city, inventory):
    costs, options = [], []
    for rep in range(30):
        inputs = draw(inventory, city, rep, REFERENCE, return_port_classes=True)
        sites, distance, q, _, waits, bill, charging, eligible, cost, _, _ = inputs
        _, first = np.unique(sites, return_index=True)
        fast_sites = inventory.levels[:, 2] > 0
        site_options = eligible[:, first]
        options.append(dict(city=city, layout=rep, nearest_site_km=distance.min(axis=1).mean(),
                            eligible_sites=site_options.sum(axis=1).mean(),
                            no_fast_site_pct=100 * np.mean(~site_options[:, fast_sites].any(axis=1))))
        costs.extend(compare(cost, charging, inputs, rep, city, "Class ratings"))
        flat_charge = np.broadcast_to((60 * q / 7.2)[:, None], cost.shape)
        flat_cost = (REFERENCE.distance_value * distance[:, sites] + bill
                     + REFERENCE.time_value * (flat_charge + waits[sites]))
        costs.extend(compare(flat_cost, flat_charge, inputs, rep, city, "Common rating"))
    return costs, options


def write_rows(filename, rows):
    (OUTPUT / filename).write_text("".join(" & ".join(row) + r" \\" + "\n" for row in rows))


def planning_tables():
    archive = RESULTS / "planning"
    witness = json.loads((archive / "witness.json").read_text())
    city = load_city("Washington")
    example = planning_study.draw(city, 2, witness["discovery_layout"])
    prices = planning_study.budget_cap(example, witness["budget_center"])
    within_range = example.distance <= witness["range_km"]
    eligible = within_range & (example.tariff <= prices[:, None])
    target = witness["target_site"]
    rows = []
    for sequence, vehicle in enumerate(witness["vehicles"], start=1):
        j = vehicle["id"]
        quantity = example.demand[j]
        assert np.isclose(quantity, vehicle["kwh"]) and np.isclose(prices[j], vehicle["budget_per_kwh"])
        assert list(np.flatnonzero(eligible[j])) == [target]
        rows.append([str(sequence), f"{quantity:.2f}", f"{prices[j]:.3f}", f"{quantity * prices[j]:.2f}",
                     f"{example.distance[j, target]:.2f}", str(within_range[j].sum()), str(eligible[j].sum())])
    write_rows("witness_rows.tex", rows)

    routes = pd.read_csv(archive / "route_interventions.csv").set_index("geometry")
    labels = {"Planar": "Straight line", "Street": "Directed streets", "Connected core": "Connected street endpoints"}
    rows = []
    for geometry, label in labels.items():
        route = routes.loc[geometry]
        rows.append([label, f"{route.mean_baseline_served:.2f}", str(int(route.target_improved)),
                     str(int(route.control_improved)), str(int(route.discount_improved))])
    write_rows("route_plan_rows.tex", rows)
    planning_figure(city, witness, pd.read_csv(archive / "feasibility.csv"))


def planning_figure(city, witness, frontier):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.45), constrained_layout=True)
    series = frontier.loc[np.isfinite(frontier.radius_km)].groupby("budget_center")
    styles = [("#97461d", "s"), ("#bf7037", "^"), ("#116b70", "o"), ("#3d75a3", "D"), ("#705c90", "v")]
    for (center, values), (color, marker) in zip(series, styles):
        radii = values.radius_km.to_numpy()
        left.plot(radii, values.probability, color=color, marker=marker, ms=3.5, lw=1.5, label=f"{center:.2f}")
        if center == planning_study.NOMINAL:
            left.fill_between(radii, values.lower95.to_numpy(), values.upper95.to_numpy(), color=color, alpha=.12)
    left.set(xlabel="Driving radius (km)", ylabel="Fraction with full service", ylim=(-.04, 1.06),
             title="(a) Full service across 64 batches")
    left.grid(axis="y", alpha=.2)
    left.legend(title="Price-cap band centre ($/kWh)", ncol=2, loc="lower right", fontsize=7.5, title_fontsize=8)

    drivers = np.array([[r["east_km"], r["north_km"]] for r in witness["vehicles"]])
    target, control = city.xy[[witness["target_site"], witness["control_site"]]]
    right.scatter(*city.xy.T, s=8, color="#bdc9cb", edgecolors="none")
    for xy in drivers:
        right.plot([xy[0], target[0]], [xy[1], target[1]], ls="--", lw=.85, color="#757575")
    right.scatter(*drivers.T, s=28, marker="^", color="#af511f", zorder=3)
    right.scatter(*target, s=90, marker="*", color="#116b70", zorder=4)
    right.scatter(*control, s=43, marker="+", lw=1.7, color="#4a3b62", zorder=4)
    labels = [("Drivers 1 and 3", drivers[[0, 2]].mean(axis=0), (-3.85, 4.7)),
              ("Driver 2", drivers[1], (.72, 4.95)),
              ("Tivoli Square", target, (.63, 2.2)),
              ("Comparison site", control, (-2.1, -2.75))]
    for label, xy, position in labels:
        right.annotate(label, xy=xy, xytext=position, fontsize=7.5,
                       arrowprops={"arrowstyle": "-", "lw": .65, "color": "#393939"},
                       bbox={"facecolor": "white", "edgecolor": "none", "pad": .3})
    right.set(xlabel="East of origin (km)", ylabel="North of origin (km)", xlim=(-4.15, 3.55),
              ylim=(-3.55, 5.55), title="(b) Three requests for two sockets")
    right.set_aspect("equal", adjustable="box")
    for ax in (left, right):
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(OUTPUT / "planning_decision.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def supply_map(inventories):
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.8), sharex=True, sharey=True)
    for ax, (city, supply) in zip(axes, inventories.items()):
        x, y = supply.xy.T
        fast = supply.levels[:, 2] > 0
        ax.scatter(x[~fast], y[~fast], s=9, facecolor="#79919c", edgecolors="none")
        ax.scatter(x[fast], y[fast], s=27, marker="D", linewidth=.5,
                   facecolor="#b24e19", edgecolors="white", zorder=3)
        ax.set_title(city, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-14, 9)
        ax.set_ylim(-24, 16)
        ax.set_xticks([-10, 0, 5])
        ax.set_yticks([-20, -10, 0, 10])
        ax.axhline(0, linewidth=.45, color=".8", zorder=0)
        ax.axvline(0, linewidth=.45, color=".8", zorder=0)
        ax.set_xlabel("East (km)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("North (km)")
    keys = [Line2D([], [], marker="o", ms=5, color="#79919c", lw=0),
            Line2D([], [], marker="D", ms=6, color="#b24e19", lw=0)]
    fig.legend(keys, ["Alternating-current ports only", "At least one direct-current fast port"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=.14, bottom=.23, top=.91, left=.07, right=.99)
    fig.savefig(OUTPUT / "city_sites.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def example_plot():
    v = np.linspace(0, .25, 200)
    # Two feasible assignments. A requests 25 kWh and B requests 50 kWh.
    bills = [25 * .50 + 50 * .35, 25 * .35 + 50 * .50]
    minutes = [60 * (25 / 50 + 50 / 7.2), 60 * (25 / 7.2 + 50 / 50)]
    fixed = np.array(bills) + 2.52 * np.array([1, 4])
    switch = (fixed[1] - fixed[0]) / (minutes[0] - minutes[1])
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.plot(v, fixed[0] + v * minutes[0], color="#b24e19", lw=2, label="25 kWh request at the fast port")
    ax.plot(v, fixed[1] + v * minutes[1], color="#116b70", lw=2, ls="--",
            label="50 kWh request at the fast port")
    ax.plot(switch, fixed[0] + switch * minutes[0], "ko", ms=4, zorder=4)
    ax.annotate("Equal cost", xy=(switch, fixed[0] + switch * minutes[0]),
                xytext=(.024, 86), arrowprops={"arrowstyle":"->", "lw":.8}, fontsize=9)
    ax.axvline(REFERENCE.time_value, color=".45", lw=1, ls=":")
    ax.text(REFERENCE.time_value - .003, 38, "City reference", ha="right", fontsize=9)
    ax.set(xlim=(0, .25), ylim=(22, 152), xlabel="Value of time at the station (dollars per minute)",
           ylabel="Cost of both trips (dollars)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "two_driver_cost.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True, parents=True)
    all_costs, all_options = [], []
    inventories = {city: public_inventory(city) for city in FILES}
    for city, inventory in inventories.items():
        costs, options = city_results(city, inventory)
        all_costs.extend(costs)
        all_options.extend(options)
    all_costs = pd.DataFrame(all_costs)
    by_layout = all_costs.groupby(["city", "power_case", "layout"], sort=False).mean(numeric_only=True)
    nominal = by_layout.xs("Class ratings", level="power_case")
    original = pd.read_csv(RESULTS / "primary_city_layouts.csv")
    original = original[original.policy == "Sequential cost"].set_index(["city", "layout"]).loc[nominal.index]
    for key in ("service_pct", "cost_total", "cohort_opt_total"):
        np.testing.assert_allclose(nominal[key], original[key], rtol=2e-9)
    np.testing.assert_allclose(nominal.saving_pct, original.cost_saving_pct, rtol=2e-9)
    all_costs.to_csv(OUTPUT / "city_cost_by_order.csv", index=False, float_format="%.10g")
    options = pd.DataFrame(all_options).groupby("city", sort=False).mean(numeric_only=True).drop(columns="layout")
    options.to_csv(OUTPUT / "city_access.csv", float_format="%.10g")
    summary = by_layout.groupby(level=["city", "power_case"], sort=False).mean().drop(columns="order")
    summary.to_csv(OUTPUT / "city_cost_components.csv", float_format="%.10g")
    rows = []
    for city in inventories:
        nominal = summary.loc[(city, "Class ratings")]
        flat = summary.loc[(city, "Common rating")]
        rows.append([city] + [f"{nominal[k]:.2f}" for k in PARTS + ["gain"]]
                    + [f"{flat.gain:.2f}", f"{flat.saving_pct:.2f}"])
    write_rows("component_rows.tex", rows)
    example_plot()
    supply_map(inventories)
    planning_tables()
    print(summary[PARTS + ["gain", "saving_pct"]].round(3))
    print(options.round(3))


if __name__ == "__main__":
    main()
