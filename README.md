# QFAMES: error and cost scaling

Numerical experiments for **Quantum Spectral Estimation with Multiple Initial States: Improved Error and Cost Scaling**, by Lingwei Zhang and Zhiyan Ding.

The three experiments compare independent and shared time sampling for eigenvalue estimation, multiplicity detection, and observable estimation.

Based on Zhiyan Ding, Lin Lin, Yilun Yang, and Ruizhe Zhang, *Quantum Filtering and Analysis of Multiplicities in Eigenvalue Spectra*, PRX Quantum **7**, 020318 (2026), [doi:10.1103/jch7-734h](https://doi.org/10.1103/jch7-734h).

## Installation and notebook

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab run_experiments.ipynb
```

Select the Python environment with these dependencies and run the notebook cells in order. Each experiment has an action selector:

- `"plot"` (default): regenerate figures from saved data.
- `"view"`: display saved figures.
- `"run"`: compute or resume trials.

`MODE` selects `"paper"` or a small `"quick"` check. Settings are in `configs/`; `WORKERS` controls parallelism. Full runs can take substantial time and memory. When settings change, choose a new `RUN_NAME` instead of resuming an existing run.

## Command line

For command-line use only, install `requirements.txt`. Run the experiments with:

```bash
python -m experiments.eigenvalue_scaling --workers 2 --resume
python -m experiments.multiplicity_scaling --workers 2 --resume
python -m experiments.observable_scaling --workers 2 --resume
```

Add `--quick` for a smoke test or `--output results/my-run` for a separate run. Resume requires compatible settings, source code, and NumPy/SciPy versions. Avoid running two jobs against the same output directory.

To regenerate figures without simulation:

```bash
python -m qfames.plotting results/eigenvalue
python -m qfames.plotting results/multiplicity
python -m qfames.plotting results/observable
```

## Results

Saved results are in `results/eigenvalue/`, `results/multiplicity/`, and `results/observable/`. Each contains configuration metadata (`metadata.json`), individual trials (`trials.csv`), summaries (`summary.csv`), and PDF/PNG figures (`figures/`).

Eigenvalue and observable plots show 90th-percentile error versus state count and mean total evolution time. Multiplicity plots show normalized singular values at estimated peaks; the shaded threshold band is illustrative. Evolution costs count all sampled state-pair times, including repeats; observable costs also include both observable evolution legs.

## Code and checks

- `qfames/`: sampling, estimators, and plotting.
- `experiments/`: experiment runners.
- `configs/`: experiment settings.
- `tests/`: numerical checks.

```bash
python -m unittest discover -s tests -v
```

## License

This code is released under the [MIT License](LICENSE).
