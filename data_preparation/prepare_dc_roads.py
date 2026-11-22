"""Extract the routable block endpoints and official lengths from DDOT layer 163."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

LAYER = ("https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
         "Transportation_WebMercator/MapServer/163")
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "dc_road_blocks.csv.gz"


def download_pages():
    offset = 0
    while True:
        params = {"where": "1=1", "outFields": "OBJECTID,ROADTYPE,SUMMARYDIRECTION,LENGTH",
                  "returnGeometry": "true", "outSR": "4326", "orderByFields": "OBJECTID",
                  "resultRecordCount": 1000, "resultOffset": offset, "f": "geojson"}
        with urlopen(LAYER + "/query?" + urlencode(params), timeout=60) as response:
            page = json.load(response)
        yield page
        if not page.get("exceededTransferLimit"):
            break
        if not page["features"]:
            raise ValueError("GIS service returned an empty nonfinal page")
        offset += len(page["features"])


def extract(pages, output):
    rows = []
    seen = set()
    for page in pages:
        for feature in page["features"]:
            attrs = feature["properties"]
            oid = attrs["OBJECTID"]
            if oid in seen:
                raise ValueError(f"Duplicate block ID during download: {oid}")
            seen.add(oid)
            if str(attrs["ROADTYPE"]) not in {"1", "2", "3", "4", "5"}:
                continue
            geometry = feature["geometry"]
            if geometry["type"] != "LineString" or not attrs["LENGTH"] > 0:
                raise ValueError(f"Check geometry or length for block {oid}")
            points = geometry["coordinates"]
            rows.append((oid, attrs["ROADTYPE"], attrs["SUMMARYDIRECTION"],
                         *points[0][:2], *points[-1][:2], attrs["LENGTH"]))
    if not rows:
        raise ValueError("No motor-vehicle roadway blocks returned")
    # Freeze gzip timestamps so the same snapshot yields the same artifact.
    with output.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(("id", "roadtype", "direction", "from_lon", "from_lat",
                             "to_lon", "to_lat", "length_m"))
            writer.writerows(sorted(rows))
    print(f"{len(rows)} motor-vehicle blocks in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", type=Path, help="local ordered GeoJSON response pages")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    pages = (json.loads(p.read_bytes()) for p in args.pages) if args.pages else download_pages()
    extract(pages, args.output)


if __name__ == "__main__":
    main()
