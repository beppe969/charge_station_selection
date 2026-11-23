"""Reproduce the results of the companion case-study paper from the archived inputs.

"Where would one more charging socket help? A matching-based case study." The
runner regenerates the Washington travel-range sweep, the site selection and
shortage witness, the held-out validation, the travel-allowance and street-
routing retests and the shared-power replay at Tivoli Square. See PAPERS.md for
the map from each table and figure to the script and output file.

    python run_casestudy.py --workers 2
"""
import argparse

from run_revised import check_inventories, python_study


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2, help="parallel workers (default: 2)")
    workers = str(parser.parse_args().workers)
    check_inventories()
    python_study("planning_study.py")
    python_study("planning_routes.py", "--workers", workers)
    python_study("manuscript_assets.py")
    python_study("extended_city_analysis.py")
    python_study("submission_sensitivities.py")
    python_study("tivoli_power_study.py")


if __name__ == "__main__":
    main()
