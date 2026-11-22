"""Rerun Washington half-load assignments on directed District roadway blocks."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from revised_city_data import DATA
from revised_city_study import BOOTSTRAPS, RESULTS, Scenario, city_inventory, evaluate, summarize


class StreetDistances:
    """Directed shortest paths with straight access legs to the nearest block ends."""

    def __init__(self, inventory, connected_core=False):
        node_index = {}
        arcs = {}
        directions = Counter()
        with gzip.open(DATA / "dc_road_blocks.csv.gz", "rt", newline="") as source:
            for block in csv.DictReader(source):
                direction = block["direction"]
                directions[direction or "missing"] += 1
                if direction not in {"IB", "OB", "BD"}:
                    continue
                ends = []
                for which in ("from", "to"):
                    # Coincident block ends can differ by sub-metre rounding.
                    point = tuple(round(float(block[f"{which}_{axis}"]), 5) for axis in ("lon", "lat"))
                    ends.append(node_index.setdefault(point, len(node_index)))
                a, b = ends
                if a == b:
                    continue
                length = float(block["length_m"]) / 1000
                if direction in {"BD", "OB"}:
                    arcs[a, b] = min(length, arcs.get((a, b), np.inf))
                if direction in {"BD", "IB"}:
                    arcs[b, a] = min(length, arcs.get((b, a), np.inf))
        n = len(node_index)
        from_nodes, to_nodes = zip(*arcs)
        graph = csr_matrix((list(arcs.values()), (from_nodes, to_nodes)), shape=(n, n))
        lonlat = np.array(list(node_index))
        scales = [111.320 * np.cos(np.deg2rad(inventory.latitude0)), 110.574]
        _, component = connected_components(graph, connection="strong")
        largest = np.flatnonzero(component == np.argmax(np.bincount(component)))
        self.available = largest if connected_core else np.arange(n)
        self.nodes = cKDTree((lonlat[self.available] - [inventory.longitude0, inventory.latitude0]) * scales)
        self.sites = inventory.xy
        self.site_offsets, target = self.nodes.query(self.sites)
        targets, self.site_to_target = np.unique(self.available[target], return_inverse=True)
        # A path from a driver to a target is a path from that target in the transpose.
        self.to_targets = dijkstra(graph.T.tocsr(), directed=True, indices=targets)
        self.metadata = dict(source_blocks=sum(directions.values()), direction_counts=dict(directions),
                             largest_strong_component=len(largest), snap_nodes=len(self.available),
                             rounded_endpoint_decimals=5, graph_nodes=n, directed_arcs=graph.nnz,
                             site_offset_median_km=float(np.median(self.site_offsets)),
                             site_offset_max_km=float(self.site_offsets.max()))
        self.origin_offsets = []

    def __call__(self, origins, sites):
        if not np.array_equal(self.sites, sites):
            raise ValueError("The street network was prepared for a different set of sites")
        offsets, source_nodes = self.nodes.query(origins)
        self.origin_offsets.append(offsets)
        street_km = self.to_targets[self.site_to_target][:, self.available[source_nodes]].T
        return offsets[:, None] + street_km + self.site_offsets


def paired_changes(raw):
    rows = []
    for policy, group in raw.groupby("policy", sort=False):
        flat = group[group.geometry == "Planar"].set_index("layout").sort_index()
        n = len(flat)
        samples = np.random.default_rng(4577).integers(n, size=(BOOTSTRAPS, n))
        for geometry in ("Street", "Connected core"):
            road = group[group.geometry == geometry].set_index("layout").sort_index()
            if not flat.index.equals(road.index):
                raise ValueError("Geometry scenarios must use the same layouts")
            for key in ("service_pct", "service_deficit_pp", "distance_km", "cost_saving_pct"):
                differences = (road[key] - flat[key]).to_numpy()
                lo, hi = np.quantile(differences[samples].mean(axis=1), (.025, .975))
                rows.append(dict(policy=policy, geometry=geometry, metric=key,
                                 street_minus_planar=differences.mean(), lower95=lo, upper95=hi))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=30)
    parser.add_argument("--output", type=Path, default=RESULTS / "roads")
    args = parser.parse_args()
    if args.layouts < 2:
        parser.error("At least two layouts are required")
    args.output.mkdir(parents=True, exist_ok=True)
    streets = StreetDistances(city_inventory("Washington"))
    core = StreetDistances(city_inventory("Washington"), connected_core=True)
    rows = []
    for rep in range(args.layouts):
        for geometry, kwargs in (("Planar", {}), ("Street", {"distance_function": streets}),
                                 ("Connected core", {"distance_function": core})):
            for row in evaluate("Washington", rep, Scenario(), **kwargs):
                rows.append(dict(row, geometry=geometry))
    raw = pd.DataFrame(rows)
    summary = pd.concat([summarize(g).assign(geometry=name)
                         for name, g in raw.groupby("geometry", sort=False)], ignore_index=True)
    for name, table in (("layouts", raw), ("summary", summary), ("paired", paired_changes(raw))):
        table.to_csv(args.output / (name + ".csv"), index=False, float_format="%.10g")
    meta = {"layouts": args.layouts}
    for geometry, network in (("Street", streets), ("Connected core", core)):
        offsets = np.concatenate(network.origin_offsets)
        meta[geometry] = dict(network.metadata, origin_offset_median_km=float(np.median(offsets)),
                              origin_offset_p95_km=float(np.quantile(offsets, .95)),
                              origin_offset_max_km=float(np.max(offsets)))
    (args.output / "network.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{args.layouts} paired layouts and {len(rows)} policy observations in {args.output}")


if __name__ == "__main__":
    main()
