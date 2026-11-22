# Charging-port assignment for electric vehicles

This repository reproduces the analyses in *Cost and service trade-offs in budget-constrained electric vehicle charging-port assignment* by Luca Ambrosino, Hong T. M. Chu, Laurent El Ghaoui and Giuseppe C. Calafiore. The study compares sequential and joint booking in Washington, Chicago and Manhattan, then tests what a third charging port at Washington’s historical Tivoli Square site could deliver before drivers leave.

The station inventories and Washington street network are archived inputs. Driver requests, tariffs, waiting times, connector access and site power are modeled. All inputs needed to reproduce the results are included in this checkout.

## Run the submission analysis

Use 64-bit Python 3.9 in a virtual environment. The reference run used Python 3.9.7. From the repository root, install the pinned dependencies and run the following commands.

```sh
python -m pip install -r requirements.txt
python run_revised.py --workers 2 --submission
```

The runner verifies the five input checksums before computing the city comparisons, road distances, site-selection experiments and power-sharing results. It writes into `results/`, which already contains the paper’s outputs. The Python run uses the archived MATLAB solver timings. The calibration, synthetic experiments and independent solver checks have their own MATLAB commands in the [reproduction guide](REPRODUCE_SUBMISSION.txt).

To repeat just the travel, connector, booking-delay and power-sharing comparisons using the archived baseline results, run

```sh
python code/submission_sensitivities.py
python code/tivoli_power_study.py
```

## Find a result

| Question | Output |
| --- | --- |
| How much does joint booking save for the same drivers in each city? | [City policy comparisons](results/primary_city_summary.csv) and [cost components](results/extended/city_cost_components.csv) |
| What changes with connector restrictions, and how long can confirmation wait before the savings disappear? | [Connector and delay results by city](results/submission/sensitivity_city_summary.csv) |
| How often does another Washington port help as travel allowances change? | [Travel-limit results](results/submission/sensitivity_radius_summary.csv) and [all candidate sites](results/submission/sensitivity_radius_candidates.csv) |
| How much electricity can the Tivoli Square ports deliver by departure? | [Completions and energy at four, six and eight hours](results/submission/power_summary.csv) |

For example, across the 96 Washington test batches, two ports on a 14.4-kilowatt connection completed an average of 1.73 charges within six hours. Three ports sharing that connection equally completed 1.22, even though they delivered more energy. On a 21.6-kilowatt connection, three ports completed 1.92. These are simulations with simultaneous plug-in. The historical inventory and the later check on the site’s listing are documented in the [provenance note](data_preparation/inventory_provenance.txt).

## Check the files

The release includes a SHA-256 (Secure Hash Algorithm, 256-bit) manifest. On a fresh checkout, run `shasum -a 256 -c SHA256SUMS.txt` on macOS or `sha256sum -c SHA256SUMS.txt` on Linux. The command checks the files exactly as distributed. When comparing a new run, use the numerical summaries in the reproduction guide. Figure creation dates and machine-specific solver timings can differ.

The [detailed guide](REPRODUCE_SUBMISSION.txt) describes the random seeds, scenarios, output columns and MATLAB commands. The [baseline technical guide](README.pdf) documents the inventory reconciliation, road network and solver benchmarks. The earlier study’s plots are in [figures](figures/), and the revised paper’s plots and tables are in [results](results/). The original release identifies the project code license as MIT. The archived data retain their source terms, and third-party code retains its attribution.

## Analyses for the IEEE T-ITS version

The per-driver outcomes, DC charging-taper sensitivity, rolling-horizon dispatch, fast-port occupancy price and robustness checks added for the IEEE Transactions on Intelligent Transportation Systems version are described in `REPRODUCE_TITS.txt` and produced by `code/driver_outcomes.py` and `code/rolling_horizon.py`. Outputs are in `results/tits/`.
