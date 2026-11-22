"""Reproduce the charging study from its archived inputs."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check_inventories():
    data = ROOT / "data"
    for line in (data / "inventory.sha256").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        actual = hashlib.sha256((data / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Inventory differs from the manuscript input: {filename}")


def python_study(filename, *arguments):
    print(f"Running {filename}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "code" / filename), *arguments],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel workers for the city experiments (default: 2)")
    parser.add_argument("--matlab", metavar="EXECUTABLE",
                        help="rerun the MATLAB solver tests and machine-specific timings")
    parser.add_argument("--submission", action="store_true",
                        help="also rebuild the extended city, connector, travel and site-power results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    check_inventories()
    python_study("access_audit.py")
    for study in ("revised_city_study.py", "revised_primary_cities.py",
                  "revised_public_access.py", "revised_scarcity.py"):
        python_study(study, "--workers", str(args.workers))
    python_study("road_study.py")
    python_study("planning_study.py")
    python_study("planning_routes.py", "--workers", str(args.workers))
    if args.matlab:
        print("Running MATLAB assignment verification and benchmark", flush=True)
        subprocess.run([args.matlab, "-batch", "addpath('code'); verify_assignment;"],
                       cwd=ROOT, check=True)
    else:
        print("Using archived MATLAB verification timings in results/solver/.", flush=True)
    python_study("manuscript_assets.py")
    if args.submission:
        for study in ("extended_city_analysis.py", "submission_sensitivities.py",
                      "tivoli_power_study.py"):
            python_study(study)


if __name__ == "__main__":
    main()
