"""Crosswalk the shipped inventories to access codes in the frozen national extract."""
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from revised_city_data import DATA, FILES, LEVELS

NATIONAL_COUNTS = ("EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count")
RESULT = Path(__file__).resolve().parents[1] / "results" / "inventory_access.csv"


def normal(value):
    return " ".join(str(value).strip().casefold().split())


def zip5(value):
    """Normalize ZIPs stored as Excel numbers, ZIP+4 strings or decimal-comma text."""
    text = str(value).strip().split("-")[0].split(".")[0].split(",")[0]
    return text.zfill(5) if text.isdigit() else ""


def key(address, city, state, zipcode, counts):
    values = tuple(float(x or 0) for x in counts)
    if any(not x.is_integer() or x < 0 for x in values):
        raise ValueError("Expected nonnegative integer port counts")
    numbers = tuple(int(x) for x in values)
    return normal(address), normal(city), normal(state), zip5(zipcode), numbers


@lru_cache(maxsize=1)
def source_index():
    source = defaultdict(list)
    with (DATA / "USA_EV_CS.csv").open(encoding="cp1252", newline="") as fh:
        for record_id, r in enumerate(csv.DictReader(fh, delimiter=";"), start=1):
            k = key(r["Street Address"], r["City"], r["State"], r["ZIP"],
                    (r[col] for col in NATIONAL_COUNTS))
            source[k].append((record_id, normal(r["Access Code"])))
    return source


@lru_cache(maxsize=3)
def audit_records(city):
    path = DATA / FILES[city]
    records = (pd.read_csv(path, sep=";", decimal=",") if path.suffix == ".csv"
               else pd.read_excel(path, engine="openpyxl"))
    national = source_index()
    results = []
    for i, r in records.iterrows():
        k = key(r["StreetAddress"], r["City"], r["State"], r["ZIP"],
                r[LEVELS].fillna(0))
        levels = k[-1]
        candidates = national.get(k, [])
        codes = {v for _, v in candidates}
        if codes == {"public"}:
            status = "public_only"
        elif codes == {"private"}:
            status = "private_only"
        elif len(codes) > 1:
            status = "ambiguous"
        else:
            status = "unresolved"
        results.append(dict(city=city, inventory_record=i + 1, latitude=r["Latitude"],
                            longitude=r["Longitude"], lvl1=levels[0], lvl2=levels[1],
                            dcfast=levels[2], classification=status,
                            national_matches=len(candidates),
                            national_records="|".join(str(j) for j, _ in candidates)))
    df = pd.DataFrame(results)
    df["within_source_count"] = False
    usable = df.dropna(subset=["latitude", "longitude"])
    for _, group in usable.loc[usable.national_matches > 0].groupby("national_records"):
        limit = int(group.national_matches.iloc[0])
        if len(group) > limit and len(group[["latitude", "longitude"]].drop_duplicates()) > 1:
            raise ValueError("Manual reconciliation needed for a repeated key at different coordinates")
        # When rows exceed the national multiplicity, tied records are identical
        # in study location and class counts. Which tied row is retained is immaterial.
        df.loc[group.index[:limit], "within_source_count"] = True
    return df


@lru_cache(maxsize=6)
def public_ports_by_coordinate(city, cap_source=False):
    """Public-coded L1/L2/DC counts by coordinate, optionally capped at source multiplicity."""
    df = audit_records(city).dropna(subset=["latitude", "longitude"])
    if cap_source:
        df = df.loc[df.within_source_count]
    grouped = df.loc[df.classification == "public_only"].groupby(
        ["latitude", "longitude"], sort=True)[["lvl1", "lvl2", "dcfast"]].sum()
    return {loc: values.to_numpy(dtype=int) for loc, values in grouped.iterrows()}


def main():
    rows = pd.concat([audit_records(city) for city in FILES], ignore_index=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(RESULT, index=False)
    usable = rows.dropna(subset=["latitude", "longitude"]).copy()
    usable["ports"] = usable[["lvl1", "lvl2", "dcfast"]].sum(axis=1)
    print(usable.groupby(["city", "classification", "within_source_count"]).agg(
        records=("ports", "size"), ports=("ports", "sum")).to_string())


if __name__ == "__main__":
    main()
