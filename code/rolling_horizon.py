"""Rolling-horizon comparison of immediate and batched port reservation.

Washington, source-reconciled public inventory (1,053 ports). Requests arrive
over a six-hour horizon as a Poisson stream conditioned on its count, at 1.5,
3.0 or 4.5 requests per minute. Each request keeps the half-load study's demand
model: origin, energy, unit-price cap, vehicle power limits, 5 km straight-line
eligibility, site tariffs and external waits, and the generalized cost
c_ij = a d_ij + p_i q_j + v (w_i + 60 q_j / P_ij).

A reservation holds the port from confirmation until the vehicle leaves:
travel at 22 km/h, the site's external wait, then the charging session. A
port can be reserved only when it is free. A request still unassigned 30
minutes after arrival is lost.

Policies
  Immediate     least-cost free eligible port at arrival; a request that finds
                none retries each minute, in arrival order, until it expires.
  Batch(D)      every D minutes, a maximum-service minimum-cost assignment of
                all pending requests to the free ports; leftovers carry over.

Each served driver's reported cost adds the confirmation delay valued at v.
Full-fleet cost per request adds a $100 fallback for each lost request.

Usage: python code/rolling_horizon.py [--layouts 30] [--workers 2]
       python code/rolling_horizon.py --extensions [--workers 2]

The --extensions run (i) tunes a fast-port occupancy price kappa on layouts
100-109 and evaluates kappa = 0.8 $/min on layouts 0-29, and (ii) repeats the
comparison with a three-hour warm-up in a nine-hour horizon, patience of 15 or
60 minutes, and fallback costs of 50 or 200 dollars.
Writes results/tits/rolling_layouts.csv, rolling_summary.csv, rolling_rows.tex
and rolling_horizon.pdf.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from revised_city_study import BOOTSTRAPS, RESULTS, SEED, Scenario, draw, exact_assignment
from revised_primary_cities import public_inventory

OUT = RESULTS / "tits"
HORIZON = 360.0  # minutes of arrivals
PATIENCE = 30.0  # minutes a request waits for a reservation
SPEED = 22.0  # km/h, average urban speed used for port holding times
FALLBACK = 100.0
RATES = {"moderate": 1.5, "heavy": 3.0, "saturated": 4.5}  # requests per minute
WINDOWS = (1, 2, 5, 10, 15, 20, 30)
TIME_VALUE = Scenario().time_value
_INV = {}


def inventory():
    if "w" not in _INV:
        _INV["w"] = public_inventory("Washington")
    return _INV["w"]


def instance(rep, load_name, horizon=HORIZON):
    inv = inventory()
    m = int(inv.levels.sum())
    n = int(round(RATES[load_name] * horizon))
    parts = draw(inv, "Washington", rep, Scenario(load=n / m), return_port_classes=True)
    sites, distance, q, prices, waits, bill, charge, eligible, cost, orders, level = parts
    assert len(q) == n
    k = list(RATES).index(load_name)
    tag = [SEED, 0, rep, 60607, k] if horizon == HORIZON else [SEED, 0, rep, 60607, k, int(horizon)]
    rng = np.random.default_rng(np.random.SeedSequence(tag))
    arrival = np.sort(rng.uniform(0, horizon, n))
    hold = 60 * distance[:, sites] / SPEED + waits[sites][None, :] + charge
    return dict(arrival=arrival, eligible=eligible, cost=cost, hold=hold, q=q, n=n, m=len(sites), fast=level == 2,
                horizon=horizon)


def simulate(data, window, patience=PATIENCE, kappa=0.0, kappa_ports="fast", warmup=0.0, fallback=FALLBACK):
    """window=None: immediate booking; otherwise batch interval in minutes.

    kappa > 0 adds an occupancy price of kappa dollars per minute of port holding
    time to the dispatcher's scores, on fast ports only (kappa_ports="fast") or on
    every port ("all"). Reported costs exclude it. Metrics count only requests
    arriving at or after `warmup` minutes.
    """
    a, E, C, H = data["arrival"], data["eligible"], data["cost"], data["hold"]
    n, m = data["n"], data["m"]
    PATIENCE_ = patience
    if kappa > 0:
        mask = data["fast"] if kappa_ports == "fast" else np.ones(m, dtype=bool)
        S = C + kappa * H * mask[None, :]
    else:
        S = C
    release = np.zeros(m)
    port = np.full(n, -1)
    decided = np.full(n, np.nan)

    def book(j, i, t):
        assert release[i] <= t + 1e-9 and E[j, i]
        port[j], decided[j] = i, t
        release[i] = t + H[j, i]

    if window is None:
        queue = []
        nxt = 0
        t = 0.0
        while nxt < n or queue:
            t += 1.0
            while nxt < n and a[nxt] <= t:
                j = nxt
                options = np.flatnonzero(E[j] & (release <= a[j]))
                if len(options):
                    book(j, options[np.argmin(S[j, options])], a[j])
                else:
                    queue.append(j)
                nxt += 1
            keep = []
            for j in queue:
                if t - a[j] > PATIENCE_:
                    continue
                options = np.flatnonzero(E[j] & (release <= t))
                if len(options):
                    book(j, options[np.argmin(S[j, options])], t)
                else:
                    keep.append(j)
            queue = keep
    else:
        t = 0.0
        last = data["horizon"] + PATIENCE_ + window
        while t <= last:
            t += window
            pending = np.flatnonzero((a <= t) & (port < 0) & (t - a <= PATIENCE_))
            if not len(pending):
                continue
            free = np.flatnonzero(release <= t)
            if not len(free):
                continue
            allowed = E[np.ix_(pending, free)]
            rows = np.flatnonzero(allowed.any(axis=1))
            if not len(rows):
                continue
            choice = exact_assignment(S[np.ix_(pending[rows], free)], allowed[rows])
            for r, c in zip(pending[rows], choice):
                if c >= 0:
                    book(r, free[c], t)
    assert E[np.flatnonzero(port >= 0), port[port >= 0]].all()
    counted = a >= warmup
    served = (port >= 0) & counted
    n = int(counted.sum())
    delay = decided[served] - a[served]
    assert (delay >= -1e-9).all() and (delay <= PATIENCE_ + 1e-9).all()
    base = C[np.flatnonzero(served), port[served]]
    total = base + TIME_VALUE * delay
    lost = n - served.sum()
    return dict(
        requests=n,
        served_total_usd=total.sum(),
        service_pct=100 * served.sum() / n,
        lost=int(lost),
        delay_min=delay.mean(),
        cost_served_usd=base.mean(),
        cost_with_delay_usd=total.mean(),
        fleet_cost_usd=(total.sum() + fallback * lost) / n,
        served_kwh=data["q"][served].sum(),
        hold_min=H[np.flatnonzero(served), port[served]].mean(),
        fast_share_pct=100 * data["fast"][port[served]].mean(),
        fast_request_kwh=data["q"][served][data["fast"][port[served]]].mean(),
    )


def run(task):
    rep, load_name = task
    data = instance(rep, load_name)
    rows = []
    for window in (None,) + WINDOWS:
        r = simulate(data, window)
        r.update(layout=rep, load=load_name, policy="Immediate" if window is None else f"Batch {window} min",
                 window=0 if window is None else window)
        rows.append(r)
    return rows


def ci(values, seed=4577):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(values), size=(BOOTSTRAPS, len(values)))
    lo, hi = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
    return values.mean(), lo, hi


def summarize(raw):
    rows = []
    keys = ("service_pct", "delay_min", "cost_served_usd", "cost_with_delay_usd", "fleet_cost_usd", "served_kwh",
            "hold_min", "fast_share_pct", "fast_request_kwh")
    for load_name, g in raw.groupby("load", sort=False):
        ref = g[g.window == 0].set_index("layout").sort_index()
        for window, h in g.groupby("window"):
            h = h.set_index("layout").sort_index()
            row = dict(load=load_name, window=window, policy=h.policy.iloc[0], layouts=len(h),
                       requests=int(h.requests.iloc[0]))
            for key in keys:
                row[key] = h[key].mean()
                row[key + "_diff"], row[key + "_diff_lo"], row[key + "_diff_hi"] = ci(h[key] - ref[key])
            rows.append(row)
    return pd.DataFrame(rows)


def figure(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    styles = {"moderate": dict(color="#2b6cb0", marker="o"), "heavy": dict(color="#c05621", marker="s"),
              "saturated": dict(color="#2f855a", marker="^")}
    for load_name, g in summary.groupby("load", sort=False):
        g = g.sort_values("window")
        b = g[g.window > 0]
        label = f"{RATES[load_name]:g} requests/min"
        for ax, key in zip(axes, ("service_pct_diff", "fleet_cost_usd_diff")):
            ax.errorbar(b.window, b[key], yerr=[b[key] - b[key + "_lo"], b[key + "_hi"] - b[key]],
                        capsize=2, lw=1.2, ms=3.5, label=label, **styles[load_name])
    for ax in axes:
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel("Batching window (min)")
        ax.set_xticks(WINDOWS)
    axes[0].set_ylabel("Service minus immediate (pp)")
    axes[1].set_ylabel("Cost per request minus immediate ($)")
    axes[0].set_title("(a) Requests served", loc="left")
    axes[1].set_title("(b) Full-fleet cost per request", loc="left")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--extensions", action="store_true",
                        help="run the occupancy-price and robustness studies instead of the base comparison")
    args = parser.parse_args()
    if args.extensions:
        return extensions(args.workers)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(r, load_name) for load_name in RATES for r in range(args.layouts)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        raw = pd.DataFrame([row for rows in ex.map(run, tasks) for row in rows])
    raw.to_csv(OUT / "rolling_layouts.csv", index=False, float_format="%.10g")
    summary = summarize(raw)
    summary.to_csv(OUT / "rolling_summary.csv", index=False, float_format="%.10g")
    def signed(x):
        return f"{x:+.2f}".replace("+", "$+$").replace("-", "$-$")

    lines = []
    for load_name in RATES:
        for r in summary[summary.load == load_name].sort_values("window").itertuples():
            if r.window not in (0, 1, 5, 10, 20, 30):
                continue
            first = f"{RATES[load_name]:g}" if r.window == 0 else ""
            name = "Immediate" if r.window == 0 else f"{r.window:g}"
            diff = "---" if r.window == 0 else (
                f"{signed(r.fleet_cost_usd_diff)} [{signed(r.fleet_cost_usd_diff_lo)}, {signed(r.fleet_cost_usd_diff_hi)}]")
            lines.append(
                f"{first} & {name} & {r.service_pct:.2f} & {r.delay_min:.1f} & {r.cost_served_usd:.2f} & "
                f"{r.fleet_cost_usd:.2f} & {diff} \\\\"
            )
    (OUT / "rolling_rows.tex").write_text("\n".join(lines) + "\n")
    figure(summary, OUT / "rolling_horizon.pdf")
    pd.set_option("display.width", 200)
    print(summary[["load", "window", "service_pct", "service_pct_diff", "service_pct_diff_lo", "service_pct_diff_hi",
                   "delay_min", "cost_served_usd", "cost_with_delay_usd", "fleet_cost_usd", "fleet_cost_usd_diff",
                   "fleet_cost_usd_diff_lo", "fleet_cost_usd_diff_hi"]].round(3).to_string())


# ----------------------------------------------------------------------------
# Extensions: occupancy price on fast ports, and robustness of the comparison
# ----------------------------------------------------------------------------
KAPPA = 0.8  # dollars per minute of fast-port holding time, tuned on layouts 100-109
TUNING_LAYOUTS = range(100, 110)


def run_kappa_tuning(task):
    rep, load_name = task
    d = instance(rep, load_name)
    rows = [dict(simulate(d, None), layout=rep, load=load_name, window=0, kappa=0.0)]
    for window in (1, 2, 5):
        for kappa in (0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
            rows.append(dict(simulate(d, window, kappa=kappa), layout=rep, load=load_name, window=window, kappa=kappa))
    return rows


def run_throughput(task):
    rep, load_name = task
    d = instance(rep, load_name)
    rows = [dict(simulate(d, None), layout=rep, load=load_name, window=0, kappa=0.0)]
    for window in (1, 2, 5, 10):
        for kappa in (0.0, KAPPA):
            rows.append(dict(simulate(d, window, kappa=kappa), layout=rep, load=load_name, window=window, kappa=kappa))
    return rows


ROBUST_WINDOWS = (None, 1, 5, 10, 20)


def run_robustness(task):
    rep, load_name = task
    rows = []
    base = instance(rep, load_name)
    long = instance(rep, load_name, horizon=540.0)
    cases = [("base", base, dict()),
             ("patience 15", base, dict(patience=15.0)),
             ("patience 60", base, dict(patience=60.0)),
             ("warm-up", long, dict(warmup=180.0))]
    for case, d, kw in cases:
        for window in ROBUST_WINDOWS:
            r = simulate(d, window, **kw)
            for h in (50.0, 100.0, 200.0):
                if case != "base" and h != FALLBACK:
                    continue
                fleet = (r["served_total_usd"] + h * r["lost"]) / r["requests"]
                rows.append(dict(case=case if h == FALLBACK else f"fallback {h:g}", layout=rep, load=load_name,
                                 window=0 if window is None else window, service_pct=r["service_pct"],
                                 delay_min=r["delay_min"], fleet_cost_usd=fleet, requests=r["requests"]))
    return rows


def paired(raw, group_cols, ref_mask, keys):
    out = []
    for name, g in raw.groupby(group_cols, sort=False):
        ref = g[ref_mask(g)].set_index("layout").sort_index()
        for key2, h in g.groupby([c for c in ("window", "kappa") if c in g], sort=False):
            h = h.set_index("layout").sort_index()
            row = dict(zip(group_cols, name if isinstance(name, tuple) else (name,)))
            row.update(dict(zip([c for c in ("window", "kappa") if c in g], key2 if isinstance(key2, tuple) else (key2,))))
            for key in keys:
                row[key] = h[key].mean()
                row[key + "_diff"], row[key + "_diff_lo"], row[key + "_diff_hi"] = ci(h[key] - ref[key])
            out.append(row)
    return pd.DataFrame(out)


def extensions(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ("service_pct", "delay_min", "fleet_cost_usd", "fast_share_pct", "fast_request_kwh")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        tune = pd.DataFrame([r for rows in ex.map(run_kappa_tuning, [(r, l) for l in RATES for r in TUNING_LAYOUTS])
                             for r in rows])
        thr = pd.DataFrame([r for rows in ex.map(run_throughput, [(r, l) for l in RATES for r in range(30)])
                            for r in rows])
        rob = pd.DataFrame([r for rows in ex.map(run_robustness, [(r, l) for l in RATES for r in range(30)])
                            for r in rows])
    tune.groupby(["load", "window", "kappa"])[["service_pct", "fleet_cost_usd", "fast_share_pct"]].mean().to_csv(
        OUT / "rolling_kappa_tuning.csv", float_format="%.6g")
    thr.to_csv(OUT / "rolling_throughput_layouts.csv", index=False, float_format="%.10g")
    thr_sum = paired(thr, ["load"], lambda g: g.window == 0, keys)
    thr_sum.to_csv(OUT / "rolling_throughput_summary.csv", index=False, float_format="%.10g")
    rob.to_csv(OUT / "rolling_robustness_layouts.csv", index=False, float_format="%.10g")
    rob_sum = paired(rob, ["case", "load"], lambda g: g.window == 0, ("service_pct", "delay_min", "fleet_cost_usd"))
    rob_sum.to_csv(OUT / "rolling_robustness_summary.csv", index=False, float_format="%.10g")
    pd.set_option("display.width", 220)
    print(thr_sum[["load", "window", "kappa", "service_pct", "service_pct_diff", "service_pct_diff_lo",
                   "service_pct_diff_hi", "fleet_cost_usd_diff", "fleet_cost_usd_diff_lo", "fleet_cost_usd_diff_hi",
                   "fast_share_pct"]].round(3).to_string())
    print(rob_sum[["case", "load", "window", "service_pct", "service_pct_diff", "fleet_cost_usd_diff",
                   "fleet_cost_usd_diff_lo", "fleet_cost_usd_diff_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
