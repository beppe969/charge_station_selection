# Code and results by paper

This repository supports two manuscripts built on the same model, data and core
code. The code stays in `code/` because both papers import the same modules;
this page says which scripts and outputs belong to which paper.

| Paper | Venue | Runner |
| --- | --- | --- |
| *Joint Reservation of Electric Vehicle Charging Ports: Who Gains and When to Batch* | IEEE Transactions on Intelligent Transportation Systems | `python run_tits.py --workers 2` |
| *Where would one more charging socket help? A matching-based case study* | Sustainable Energy, Grids and Networks | `python run_casestudy.py --workers 2` |

`python run_revised.py --submission` regenerates everything, including the
intermediate files that one paper's runner reads from the other's archived
outputs. Install the pinned environment first:
`python -m pip install -r requirements.txt` (CPython 3.9).

## Shared core

| Module | Role |
| --- | --- |
| `code/revised_city_data.py` | Reads the station inventories and draws demand layouts |
| `code/revised_city_study.py` | Generalized cost, screening, exact assignment (`exact_assignment`), sequential policies |
| `code/revised_primary_cities.py` | Source-reconciled public inventories (`public_inventory`) |
| `code/manuscript_assets.py` | Numerical macros (`results/study_numbers.tex`) and table rows for both papers |
| `code/extended_city_analysis.py` | Cost decomposition and planning tables and figures for both papers |
| `data/`, `data_preparation/` | Frozen inventories, Washington street network, registry queries |

## Paper 1: IEEE T-ITS

| Manuscript item | Script | Output |
| --- | --- | --- |
| Table II, inventories | `revised_primary_cities.py`, `manuscript_assets.py` | `results/primary_city_inventory.csv`, `results/inventory_rows.tex` |
| Table III, five policies at half load | `revised_primary_cities.py` | `results/primary_city_rows.tex`, `results/primary_city_summary.csv` |
| Table IV, cost decomposition and common power | `extended_city_analysis.py` | `results/extended/component_rows.tex` |
| Table V, per-driver gains and losses, loss caps | `driver_outcomes.py` | `results/tits/driver_rows.tex`, `results/tits/driver_outcomes_summary.csv` |
| Table VI, connector restrictions and break-even delay | `submission_sensitivities.py` | `results/submission/sensitivity_city_rows.tex` |
| Fig. 2, service and saving under load; Section V-E numbers | `revised_city_study.py`, `manuscript_assets.py` | `results/load_response.pdf`, `results/study_numbers.tex` |
| Section V-F, forecast-guided booking | `revised_scarcity.py` | `results/scarcity_summary.csv` |
| Fig. 3, Table VII, rolling-horizon dispatch | `rolling_horizon.py` | `results/tits/rolling_horizon.pdf`, `results/tits/rolling_rows.tex` |
| Table VIII, robustness; occupancy price | `rolling_horizon.py --extensions` | `results/tits/rolling_robustness_summary.csv`, `results/tits/rolling_throughput_summary.csv`, `results/tits/rolling_kappa_tuning.csv` |
| Table IX, sensitivity (without the taper row) | `revised_city_study.py` | `results/sensitivity_rows.tex` |
| Table IX, DC charging taper row | `driver_outcomes.py` | `results/tits/taper_summary.csv` |
| Section V-I, street-network distances | `road_study.py` | `results/roads/` |
| Section V-J, inventory classification | `access_audit.py`, `revised_public_access.py` | `results/inventory_access.csv`, `results/access_summary.csv` |
| Appendix A, weight calibration | `calibration_pareto.m` (MATLAB) | `results/extended/calibration_grid.csv` |
| Appendix B, solver checks and timings | `verify_assignment.m` (MATLAB) | `results/solver/` |

Fig. 1 is a hand-drawn illustration (`acceptable_assignment_graph.drawio`).
Details of the T-ITS analyses are in `REPRODUCE_TITS.txt`.

## Paper 2: case study

| Manuscript item | Script | Output |
| --- | --- | --- |
| Table 1, travel range for full service; Section 4.1 numbers | `planning_study.py`, `manuscript_assets.py` | `results/planning/feasibility.csv`, `results/planning/frontier_rows.tex`, `results/planning/thresholds.csv` |
| Fig. 1, full service by radius and the selected site | `planning_study.py`, `extended_city_analysis.py` | `results/extended/planning_decision.pdf` |
| Table 2, shortage witness | `planning_study.py`, `extended_city_analysis.py` | `results/planning/witness.json`, `results/extended/witness_rows.tex` |
| Table 3, held-out validation | `planning_study.py`, `manuscript_assets.py` | `results/planning/interventions.csv`, `results/planning/intervention_rows.tex` |
| Table 4, travel allowance | `submission_sensitivities.py` | `results/submission/sensitivity_radius_rows.tex` |
| Table 5, street routing and site choice | `planning_routes.py`, `extended_city_analysis.py` | `results/planning/route_interventions.csv`, `results/extended/route_plan_rows.tex` |
| Table 6, energy delivered under a shared connection | `tivoli_power_study.py` | `results/submission/power_rows.tex`, `results/submission/power_summary.csv` |
| Street network preparation | `data_preparation/prepare_dc_roads.py` | `data/dc_road_blocks.csv.gz` |
| Registry check for Tivoli Square | archived queries | `data_preparation/tivoli_registry_2026-11-22.json` |

Details of the case-study analyses are in `REPRODUCE_SUBMISSION.txt`
(questions 1 to 3) and `README.pdf`.

## Checking a rerun

The runners write over the archived files in `results/`. On another machine,
numbers can differ in the last floating-point digits and figure files can
differ in their bytes; compare values rather than checksums.
`git diff --stat results/` after a run shows what changed. One known case: on street
distances, the distance-only joint policy ("Batch distance" in `results/roads/`) has
ties between equally short assignments, and a different platform can break them
differently. A rerun moved its cost figures by up to a few percent of their value;
no number quoted in either paper depends on that policy.
