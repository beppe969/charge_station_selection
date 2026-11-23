"""Reproduce the results of the IEEE T-ITS paper from the archived inputs.

"Joint Reservation of Electric Vehicle Charging Ports: Who Gains and When to
Batch." The runner regenerates the three-city comparisons, the per-driver and
taper analyses and the rolling-horizon studies. Studies that also serve the
companion case-study paper read its archived outputs in results/planning/;
run_revised.py regenerates everything. See PAPERS.md for the map from each
table and figure to the script and output file.

    python run_tits.py --workers 2
"""
import argparse

from run_revised import check_inventories, python_study


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2, help="parallel workers (default: 2)")
    workers = str(parser.parse_args().workers)
    check_inventories()
    python_study("access_audit.py")
    for study in ("revised_city_study.py", "revised_primary_cities.py",
                  "revised_public_access.py", "revised_scarcity.py"):
        python_study(study, "--workers", workers)
    python_study("road_study.py")
    python_study("manuscript_assets.py")
    python_study("extended_city_analysis.py")
    python_study("submission_sensitivities.py")
    python_study("driver_outcomes.py", "--workers", workers)
    python_study("rolling_horizon.py", "--workers", workers)
    python_study("rolling_horizon.py", "--extensions", "--workers", workers)


if __name__ == "__main__":
    main()
