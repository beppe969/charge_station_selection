"""Read the three input inventories and draw reproducible spatial demand in km."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


DATA = Path(__file__).resolve().parents[1] / "data"
FILES = {
    "Washington": "CS_location_DC_complete.csv",
    "Chicago": "CS_location_Chicago.xlsx",
    "Manhattan": "CS_location_Manhattan.xlsx",
}
LEVELS = ["EVLevel1EVSENum", "EVLevel2EVSENum", "EVDCFastCount"]


@dataclass
class Inventory:
    xy: np.ndarray
    latlon: np.ndarray
    levels: np.ndarray
    n_records: int
    hull: Delaunay
    latitude0: float
    longitude0: float


def load_city(name):
    path = DATA / FILES[name]
    if path.suffix == ".csv":
        records = pd.read_csv(path, sep=";", decimal=",")
    else:
        records = pd.read_excel(path, engine="openpyxl")
    return inventory_from_records(records)


def inventory_from_records(records):
    """Rebuild the projection, port counts and footprint from the selected records."""
    records = records.dropna(subset=["Latitude", "Longitude"]).copy()
    records[LEVELS] = records[LEVELS].fillna(0)
    counts = records[LEVELS].to_numpy()
    if not np.all((counts >= 0) & (counts == np.floor(counts))):
        raise ValueError("Unexpected connector count")
    # Retain every reported port; no capacity is imputed for unspecified outlets.
    sites = records.groupby(["Latitude", "Longitude"], sort=True)[LEVELS].sum()
    sites = sites[sites.sum(axis=1) > 0]
    latlon = sites.index.to_frame(index=False).to_numpy()
    lat0, lon0 = records[["Latitude", "Longitude"]].mean()
    x = (latlon[:, 1] - lon0) * (111.320 * np.cos(np.deg2rad(lat0)))
    y = (latlon[:, 0] - lat0) * 110.574
    xy = np.column_stack([x, y])
    return Inventory(xy, latlon, sites.to_numpy(dtype=int), len(records), Delaunay(xy), lat0, lon0)


def draw_positions(inventory, n, rng, multicentre=False):
    """Rejection sample central or two-centre Gaussian demand inside the footprint."""
    coords = inventory.xy
    if multicentre:
        # Two actual installation coordinates at opposite quartiles along the
        # first principal axis. A sign flip merely swaps their labels.
        _, _, axes = np.linalg.svd(coords - coords.mean(axis=0), full_matrices=False)
        ordered = coords[np.argsort(coords @ axes[0])]
        centers = ordered[[len(ordered) // 4, 3 * len(ordered) // 4]]
    pieces = []
    remaining = n
    while remaining:
        batch = max(64, remaining * 2)
        if multicentre:
            candidates = centers[rng.integers(2, size=batch)] + rng.normal(0, 1.5, size=(batch, 2))
        else:
            candidates = rng.normal(0, 2.5, size=(batch, 2))
        inside = candidates[inventory.hull.find_simplex(candidates) >= 0][:remaining]
        pieces.append(inside)
        remaining -= len(inside)
    return np.concatenate(pieces)
