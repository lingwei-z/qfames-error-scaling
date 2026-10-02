# QFAMES: error and cost scaling

Numerical experiments for **Quantum Spectral Estimation with Multiple Initial States: Improved Error and Cost Scaling**, by Lingwei Zhang and Zhiyan Ding.

This repository compares independent and shared time sampling in QFAMES, using controlled toy Hamiltonians. It contains experiments for eigenvalue estimation, multiplicity structure, and observable estimation within a degenerate eigenspace. The code follows the revised experiment plan in the manuscript draft; it is not a reproduction of every figure in version 1.0.

The underlying framework is introduced in Zhiyan Ding, Lin Lin, Yilun Yang, and Ruizhe Zhang, *Quantum Filtering and Analysis of Multiplicities in Eigenvalue Spectra*, PRX Quantum **7**, 020318 (2026), [doi:10.1103/jch7-734h](https://doi.org/10.1103/jch7-734h).

## Installation

Python 3.10 or later is required. From the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Only NumPy, SciPy, and Matplotlib are required. No quantum hardware or LaTeX installation is needed. Commands below should be run from the repository root.

## Quick check

```bash
python -m unittest discover -s tests -v
python -m experiments.eigenvalue_scaling --quick
python -m experiments.multiplicity_scaling --quick
python -m experiments.observable_scaling --quick
```

Quick runs use three small state counts, N in {40, 80, 160}, and four trials per point. Their figures are explicitly marked as smoke tests. They check the computation and output pipeline, not asymptotic scaling or publication-quality quantiles.

## Run the experiments

```bash
python -m experiments.eigenvalue_scaling --workers 2
python -m experiments.multiplicity_scaling --workers 2
python -m experiments.observable_scaling --workers 2
```

The default is one worker. Use a modest worker count: each worker holds an entire spectral data tensor, and the large observable configurations can use over 1 GB per worker. Full runs involve thousands of trials and can be expensive, especially the global peak searches. Run the quick checks first. The scripts print progress, checkpoint each completed trial, and do not overwrite previous runs.

| Experiment | Spectrum | Fixed-N state-count figure | N values in total-time figure | Figures |
| --- | --- | --- | --- | --- |
| Eigenvalue | {0, 0.25} | N = 200 | 50, 200, 800 | Error vs L=R; error vs sampled total time |
| Multiplicity | {0, 0, 0.02} | N = 500 | Not applicable | Two panels of normalized singular values at estimated peaks |
| Observable | {0, 0, 0.1} | N = 5000 | 1000, 5000, 10000 | Error vs L=R; error vs sampled total time |

The eigenvalue and observable total-time figures each contain **six measured curves**: three independent and three shared, with N fixed along each curve and L=R varied. The state-count figure reuses the corresponding fixed-N trials from the same sweep. It compares both sampling schemes.

Edit `configs/eigenvalue.json`, `configs/multiplicity.json`, or `configs/observable.json` to change the settings. For a separate run:

```bash
python -m experiments.eigenvalue_scaling --config configs/eigenvalue.json --output results/my-run --trials 100 --seed 123
```

Resume interrupted computation with the same arguments plus `--resume`:

```bash
python -m experiments.observable_scaling --workers 2 --resume
```

Resuming checks the resolved configuration, source hashes, and NumPy/SciPy versions. Trial seeds depend on experiment, N, L, trial index, and sampling scheme, so changing the worker count does not change numerical results. States are paired between the schemes at each (N,L,trial); time and noise streams are separate. Changing chunk size changes the observable RNG draw order and therefore requires a separate run.

## Replot without simulation

```bash
python -m qfames.plotting results/paper/eigenvalue
python -m qfames.plotting results/paper/multiplicity
python -m qfames.plotting results/paper/observable
```

Equivalently, add `--plot-only` to an experiment command. This reads the saved metadata and summary, not the current configuration file. Each figure is exported as both vector PDF and 300-dpi PNG.

Output names are:

```text
results/paper/eigenvalue/figures/eigenvalue_error_vs_state_count.{pdf,png}
results/paper/eigenvalue/figures/eigenvalue_error_vs_total_time.{pdf,png}
results/paper/multiplicity/figures/multiplicity_singular_values_vs_state_count.{pdf,png}
results/paper/observable/figures/observable_error_vs_state_count.{pdf,png}
results/paper/observable/figures/observable_error_vs_total_time.{pdf,png}
```

Each experiment directory also contains:

- `metadata.json`: resolved settings, seed, software versions, and source hashes.
- `trials.csv`: individual errors, estimated peaks, actual evolution costs, failure status, and relevant numerical diagnostics. For multiplicity, `sv_<cluster>_<k>` stores **s_k/L**, not s_k.
- `summary.csv`: 90th-percentile errors, mean sampled costs, failure counts, and mean normalized singular values.

Time tensors are not written to disk: their realized absolute sums and maxima are recorded per trial, and the seed plus configuration and code identify the generating computation. Keeping every L x R x N tensor would create unnecessarily large data files. Generated results are ignored by Git; archive selected paper data explicitly when preparing a release.

## Time distributions and resource accounting

Spectral data use the truncated density

$$a_T^{\mathrm{trunc}}(t) \propto e^{-t^2/(4T^2)}, \qquad |t|\leq\sigma T.$$

Thus the untruncated standard deviation is sqrt(2)T and the ideal spectral filter is exp[-T²(theta-lambda)²]. **T is a filter scale, not the maximum evolution time.** The sampled maximum is recorded separately.

For eigenvalue and multiplicity experiments, every trial computes

$$T_{\mathrm{spectral}}=\sum_{l,r,n}|t_{l,r,n}|.$$

For shared sampling this is LR times the sum of the absolute sampled times. Reusing a time value does not eliminate the evolution for each state pair. No plotted cost is replaced by LRNT, and signed times are never summed as a cost.

The observable numerator uses independent pairs (t,t') at scale T/sqrt(2), with each leg truncated at sigma*T/sqrt(2). The shared scheme shares each pair across state pairs, while measurement noise remains independent. Its complete cost is

$$T_{\mathrm{total}}=T_{\mathrm{spectral}}+\sum_{l,r,n}(|t_{l,r,n}|+|t'_{l,r,n}|).$$

The first term includes peak localization and the denominator matrix; the second includes both observable evolution legs. The code records both contributions separately. `max_time` is the maximum spectral |t| or observable |t|+|t'| in the trial.

Costs use the manuscript's unit-per-complex-datum convention. A real/imaginary circuit repetition factor, shot count, or observable block-encoding overhead is not added to this Gaussian-noise simulation. For literal hardware resource counts, those overheads must be specified separately. The plotted horizontal coordinate is the arithmetic mean of the actual costs at a fixed (scheme,N,L); the vertical coordinate is the empirical 90th-percentile error at that same configuration.

## Numerical choices

- **Noise:** exact correlations plus independent real and imaginary Gaussian noise, each with the configured standard deviation. This is a surrogate model, not bounded single-shot Hadamard outcomes and not a direct verification of every theorem assumption.
- **Initial states:** experiment 1 repeats a fixed equal-weight vector to keep spectral weights fixed across L and trials. Experiments 2 and 3 use independent normalized complex Gaussian vectors with every component magnitude greater than 0.5. A feasibility check and attempt limit prevent impossible rejection loops.
- **Truncation:** sigma is 6 rather than 4. The omitted Gaussian mass decreases from erfc(2), about 4.7e-3, to erfc(3), about 2.2e-5. The truncated kernel is still not exactly the ideal Gaussian kernel.
- **Peak search:** all experiments search the declared spectral interval using the Frobenius norm, then refine locally. The grid step is 0.4/T, equivalent to the original code's 20q/T at q=0.02; there is no hidden factor of 20. The search intervals and number of distinct clusters are supplied for these toy problems. True eigenvalues are used for error evaluation, not as evaluation locations. This is continuous refinement of the grid algorithm, not an exact implementation of a finite-grid theorem.
- **Multiplicity:** only the original second singular-value figure is retained. T increases from 50 to 200: for Delta=0.02, the ideal cross-cluster spectral factor decreases from exp(-1) to exp(-16). Singular values are evaluated at estimated peaks. The figure studies signal/noise separation; it does not implement a universally calibrated rank-recovery threshold.
- **Observable:** the target lower cluster has known multiplicity 2 and projected observable eigenvalues {1,2}. The observable is the original block-supported matrix with leading block [[1.5,0.5],[0.5,1.5]]. T increases from 35 to 50 to reduce finite-filter leakage. Both numerator and denominator quadrature noise standard deviations are 0.015, following the original state-count experiment. The same settings apply to both schemes.
- **Generalized eigenvalues:** the projected problem is solved directly, without an explicit inverse. Errors use bottleneck matching in the complex plane, retaining imaginary parts rather than silently dropping them. `max_imaginary` and `projected_condition` are saved for diagnosis. This can produce larger reported errors than the old real-part-only metric.
- **Failures:** failed peak searches or singular projected denominators receive infinite error. Quantiles use NumPy's `inverted_cdf` definition and include failures. Infinite quantiles appear as gaps with a warning, not fabricated finite points. Singular-value means necessarily use successful peak searches, with failures reported. A peak-search failure stops before observable acquisition, so its recorded time is the actual partial cost.
- **Reference lines:** dotted curves are slope guides anchored at the first finite independent point (or first noise singular-value point). They are not fitted estimates or claimed convergence rates. All measured points are shown; no early or late points are discarded to improve agreement.

At fixed T, varying L changes the statistical budget. The cost guide is therefore T_total^(-1/2), **not** a numerical test of Heisenberg scaling obtained by varying T. Shared-sampling error need not be flat for every finite-size toy model: independent measurement noise still averages across entries, and the block-supported observable may suppress some shared-time fluctuations. The code does not force saturation or curve collapse. Publication claims should be based on the full runs and their diagnostics.

## Repository layout

```text
qfames/
  core.py                         # Sampling, filters, peak search, observable estimator
  plotting.py                     # All figure generation
experiments/
  common.py                       # Trial execution, seeds, checkpointing, summaries
  eigenvalue_scaling.py            # Experiment 1
  multiplicity_scaling.py          # Experiment 2
  observable_scaling.py            # Experiment 3
configs/                          # Editable experiment settings
tests/test_numerics.py             # Numerical identity and cost checks
requirements.txt
```

This refactors `QFAMES.py`, `Experiment_3.py`, `Experiment_4.py`, and `Experiment_5.py`. Obsolete N-only, tail, observable-norm, and first multiplicity panels are omitted from the current experiment suite. No unpublished manuscript PDF or inherited plot is bundled.

## Citation and release

Use the manuscript's final bibliographic entry when available. The title and authors above identify the associated work; no publication venue or preprint identifier is assumed here. Before the public release, select a code license with the coauthor and add its LICENSE file. This bundle intentionally does not assign a license on the authors' behalf.
