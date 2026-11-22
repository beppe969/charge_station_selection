"""Build the revised manuscript's figures and numerical text from saved study outputs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
CITY = ["Washington", "Chicago", "Manhattan"]
LOAD_CASES = ["low load", "base", "medium-high load", "high load"]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_load(index, output):
    policies = [
        ("Sequential cost", "Sequential, full cost", "#075c68", "o", "-"),
        ("Nearest", "Sequential, nearest", "#a54c16", "^", "--"),
        ("Cheapest of three", "Sequential, cheapest of three", "#6156a1", "s", ":"),
        ("Batch distance", "Batch, distance", "#4d5153", "D", "-."),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    for name, label, color, marker, line in policies:
        rows = [index["Washington", case, name] for case in LOAD_CASES]
        x = [float(r["load"]) for r in rows]
        for ax, metric in zip(axes, ["service_deficit_pp", "cost_saving_pct"]):
            y = np.array([float(r[metric]) for r in rows])
            lower = [float(r[metric + "_lo"]) for r in rows]
            upper = [float(r[metric + "_hi"]) for r in rows]
            ax.errorbar(x, y, yerr=[np.maximum(0, y-lower), np.maximum(0, upper-y)],
                        color=color, ls=line, marker=marker, label=label, ms=4.5,
                        capsize=2.3, lw=1.4)
    for ax in axes:
        ax.set_xlabel("Requests per installed port")
        ax.set_xticks([.25, .50, .75, .95])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dde4e4", linewidth=.6)
    axes[0].set_ylabel("Service deficit (percentage points)")
    axes[1].set_ylabel("Saving on the same EVs (%)")
    for ax, title in zip(axes, ["(a)  Requests left unserved", "(b)  Cost of serving the selected EVs"]):
        ax.set_title(title, fontsize=9, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(.52,-.004))
    fig.subplots_adjust(left=.092, bottom=.32, right=.99, top=.91, wspace=.32)
    fig.savefig(output, metadata={"CreationDate": None})
    plt.close(fig)


def write_rows(path, rows):
    path.write_text("".join(" & ".join(row) + r" \\" + "\n" for row in rows), encoding="utf-8")


def main():
    city = read_rows(RESULTS / "city_summary.csv")
    index = {(r["city"], r["case"], r["policy"]): r for r in city}
    primary = read_rows(RESULTS / "primary_city_summary.csv")
    primary_index = {(r["city"], r["policy"]): r for r in primary}
    inventory = read_rows(RESULTS / "primary_city_inventory.csv")
    write_rows(RESULTS / "inventory_rows.tex", [
        [row["city"]] + [f'{int(row[key]):,}' for key in
                         ("records", "sites", "L1", "L2", "DC", "ports", "requests")]
        for row in inventory])
    planning = RESULTS / "planning"
    meta = json.loads((planning / "study.json").read_text())
    witness = json.loads((planning / "witness.json").read_text())
    interventions = {r["policy"]: r for r in read_rows(planning / "interventions.csv")}
    high = index["Washington", "high load", "Sequential cost"]
    stats = {}
    def number(name, value, digits=0):
        stats[name] = f"{float(value):.{digits}f}"
    for city_name in CITY:
        base = primary_index[city_name, "Sequential cost"]
        number(city_name+"Saving", base["cost_saving_pct"], 2)
    for name, field in [("HighGreedyService","service_pct"), ("HighGap","service_deficit_pp"),
                        ("HighGapLo","service_deficit_pp_lo"), ("HighGapHi","service_deficit_pp_hi"),
                        ("HighSaving","cost_saving_pct")]:
        number(name, high[field], 2)
    number("HighUnservedEnergy", float(high["unserved_kwh"])/1000, 2)
    for n, suffix in [(50,"Fifty"),(100,"Hundred"),(200,"TwoHundred")]:
        number("Fallback"+suffix, high[f"fallback_{n}_excess"], 2)
    for tag, idx in [("EffectLo",0),("EffectHi",1)]:
        number(tag,meta["target_minus_control_paired_bootstrap95"][idx],2)
    number("Effect",meta["target_minus_control_additional_mean"],2)
    number("Improved",interventions["target_socket"]["count_improved"])
    number("Holdout",meta["test_layouts"])
    number("WitnessEV",witness["witness_ev_count"])
    number("WitnessPorts",witness["witness_port_count"])
    number("WitnessEnergy",witness["witness_requested_kwh"],1)
    number("WitnessMinEnergy",witness["witness_min_unserved_kwh"],1)
    number("SweepN",meta["sweep_layouts"])
    number("TrainN",meta["discovery_layouts"])
    number("TrainHits",witness["discovery_target_hits"])
    for coord, key in [(meta["target_latlon"],"Target"),(meta["control_latlon"],"Control")]:
        number(key+"North",coord[0],5)
        number(key+"West",abs(coord[1]),5)
    thresholds = read_rows(planning / "thresholds.csv")
    nom = [float(r["min_full_range_km"]) for r in thresholds if float(r["budget_center"]) == .50]
    for key, val in [("Mid",np.median(nom)),("Ninety",np.quantile(nom,.9)),("Max",max(nom))]:
        number("Radius"+key,val,2)
    pnames = {"baseline":"Installed network", "target_socket":"Add a port at target",
              "control_socket":"Add a port at comparison site", "target_discount":"Discount target tariff"}
    write_rows(planning / "intervention_rows.tex", [
        [pnames[k],f'{float(r["mean_additional"]):.3f}',r["count_improved"],r["full_service"],
         f'{float(r["mean_deficit"]):.2f}',f'{float(r["mean_common_saving_usd_per_ev"]):.3f}']
        for k,r in interventions.items()])
    fr = [r for r in read_rows(planning/"feasibility.csv") if float(r["budget_center"]) == .5]
    write_rows(planning/"frontier_rows.tex", [
        [r"$\infty$" if np.isinf(float(r["radius_km"])) else f'{float(r["radius_km"]):.1f}',
         r["full_count"],f'{100*float(r["probability"]):.1f} '
         f'[{100*float(r["lower95"]):.1f}, {100*float(r["upper95"]):.1f}]',
         r["no_isolates"],r["degree_certified"],f'{float(r["mean_deficit"]):.2f}'] for r in fr])
    timings = read_rows(RESULTS/"solver"/"benchmark_summary.csv")
    groups = {}
    for row in timings:
        key = (int(row["evs"]), float(row["draw_probability"]))
        groups.setdefault(key,{})[row["method"]] = row
    methods = ["matchpairs_dense","matchpairs_sparse","filtered_LP","filtered_MILP"]
    write_rows(RESULTS/"solver"/"benchmark_rows.tex", [
        [str(n), f"{prob:.2f}", str(int(float(g[methods[0]]["edges_median"])))] +
        [f'{1000*float(g[m]["end_to_end_s_median"]):.2f}' for m in methods]
        for (n,prob),g in sorted(groups.items())])
    out = RESULTS/"study_numbers.tex"
    out.write_text("% Generated by code/manuscript_assets.py\n" +
                   "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in stats.items()),
                   encoding="utf-8")
    plot_load(index, RESULTS/"load_response.pdf")


if __name__ == "__main__":
    main()
