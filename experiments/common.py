"""Experiment execution, reproducible seeds, and incremental result files."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
from concurrent.futures import ProcessPoolExecutor

# One BLAS thread per worker avoids oversubscription on typical laptops.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
from qfames.core import (initial_states, spectral_samples, filtered_matrix,
                         find_peaks, observable_filtered, observable_eigenvalues,
                         matching_error)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_IDS = {"eigenvalue": 1, "multiplicity": 2, "observable": 3}
BASE_FIELDS = ["scheme", "N", "L", "trial", "status", "error", "peak_error",
               "spectral_time", "observable_time", "total_time", "max_time",
               "peak_0", "peak_1", "projected_condition", "max_imaginary",
               "message"]
SV_FIELDS = [f"sv_{cluster}_{k}" for cluster in range(2) for k in range(1, 5)]
FIELDS = BASE_FIELDS + SV_FIELDS


def run_trial(job):
    name, cfg, scheme, N, L, trial = job
    seed = [cfg["seed"], EXPERIMENT_IDS[name], L, N, trial]
    state_rng = np.random.default_rng(np.random.SeedSequence(seed + [0]))
    scheme_id = 1 if scheme == "independent" else 2
    rng = np.random.default_rng(np.random.SeedSequence(seed + [scheme_id]))
    obs_rng = np.random.default_rng(np.random.SeedSequence(seed + [scheme_id + 2]))
    row = dict.fromkeys(FIELDS, "")
    row.update(scheme=scheme, N=N, L=L, trial=trial, status="ok", error=np.inf,
               peak_error=np.inf, spectral_time=0.0, observable_time=0.0,
               total_time=0.0, max_time=0.0)
    spectrum = np.array(cfg["eigenvalues"])
    centers = np.unique(spectrum)
    phi, psi = initial_states(state_rng, len(spectrum), L, L,
                              replicated=name == "eigenvalue",
                              min_overlap=cfg["min_overlap"])
    data = spectral_samples(rng, spectrum, phi, psi, N, cfg["T"], cfg["sigma"],
                            scheme, cfg["noise_std"])
    row.update(spectral_time=data.total_time, total_time=data.total_time,
               max_time=data.max_time)
    peaks = find_peaks(data, cfg["T"], len(centers), cfg["search_bounds"],
                       cfg["grid_step"], cfg["alpha"], cfg["threshold"])
    if len(peaks) != len(centers):
        row.update(status="peak_failure", message=f"Found {len(peaks)} of {len(centers)} peaks")
        return row
    row.update(peak_0=peaks[0], peak_1=peaks[1], peak_error=matching_error(peaks, centers))
    if name == "eigenvalue":
        row["error"] = row["peak_error"]
    elif name == "multiplicity":
        for cluster, peak in enumerate(peaks):
            singular = np.linalg.svd(filtered_matrix(data, peak), compute_uv=False)
            for k in range(1, 5):
                row[f"sv_{cluster}_{k}"] = singular[k - 1] / L
        row["error"] = row["peak_error"]
    else:
        # The target is the lower cluster, whose multiplicity is assumed known.
        theta = peaks[0]
        base = filtered_matrix(data, theta)
        del data
        observable = np.array(cfg["observable"], dtype=complex)
        filtered, obs_cost, obs_max = observable_filtered(
            obs_rng, spectrum, phi, psi, observable, N, cfg["T"], cfg["sigma"],
            scheme, cfg["observable_noise_std"], theta, cfg["chunk_size"])
        row.update(observable_time=obs_cost, total_time=row["spectral_time"] + obs_cost,
                   max_time=max(row["max_time"], obs_max))
        try:
            estimates, condition = observable_eigenvalues(base, filtered, cfg["multiplicity"])
            row.update(error=matching_error(estimates, cfg["observable_targets"]),
                       projected_condition=condition,
                       max_imaginary=float(np.max(np.abs(estimates.imag))))
        except np.linalg.LinAlgError as exc:
            row.update(status="eigensolver_failure", message=str(exc))
    return row


def key(row):
    return row["scheme"], int(row["N"]), int(row["L"]), int(row["trial"])


def summarize(raw_path, output):
    with raw_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        groups.setdefault(key(row)[:3], []).append(row)
    summaries = []
    for (scheme, N, L), group in sorted(groups.items()):
        errors = np.array([float(r["error"]) for r in group])
        item = dict(scheme=scheme, N=N, L=L, trials=len(group),
                    failed=sum(r["status"] != "ok" for r in group),
                    error_q90=float(np.quantile(errors, 0.9, method="inverted_cdf")))
        for field in ["total_time", "spectral_time", "observable_time"]:
            item["mean_" + field] = float(np.mean([float(r[field]) for r in group]))
        item["max_time"] = max(float(r["max_time"]) for r in group)
        for field in SV_FIELDS:
            vals = [float(r[field]) for r in group if r[field] != ""]
            item[field] = float(np.mean(vals)) if vals else np.nan
        summaries.append(item)
    if summaries:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def validate_config(name, cfg):
    if cfg["trials"] < 1 or cfg["seed"] < 0:
        raise ValueError("trials must be positive; seed must be nonnegative")
    minimum = 4 if name == "multiplicity" else (3 if name == "observable" else 2)
    if len(set(cfg["L_values"])) != len(cfg["L_values"]) or any(L < minimum for L in cfg["L_values"]):
        raise ValueError(f"L_values must be unique integers >= {minimum}")
    if name != "multiplicity" and (len(cfg["N_values"]) != 3 or len(set(cfg["N_values"])) != 3):
        raise ValueError("Provide three distinct N values for the six-curve cost plot")
    if min(cfg["N_values"]) < 2 or cfg["fixed_N"] not in cfg["N_values"]:
        raise ValueError("N >= 2 is required, and fixed_N must occur in N_values")
    if cfg["T"] <= 0 or cfg["sigma"] <= 0 or cfg["noise_std"] < 0:
        raise ValueError("Invalid time scale, truncation, or noise parameter")


def main(name):
    parser = argparse.ArgumentParser(description=f"QFAMES {name} experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / f"{name}.json")
    parser.add_argument("--output", type=Path, help="Run directory; use a new directory for changed settings")
    parser.add_argument("--quick", action="store_true", help="Small end-to-end check, not paper data")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "results" / ("quick" if args.quick else "paper") / name
    if args.plot_only:
        from qfames.plotting import plot_run
        plot_run(output)
        return
    cfg = json.loads(args.config.read_text())
    if args.quick:
        cfg.update(L_values=[4, 6, 8], N_values=[40, 80, 160], fixed_N=80, trials=4)
    if args.trials is not None:
        cfg["trials"] = args.trials
    if args.seed is not None:
        cfg["seed"] = args.seed
    validate_config(name, cfg)
    if args.workers < 1:
        parser.error("--workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    raw_path = output / "trials.csv"
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ["qfames", "experiments"] for p in sorted((ROOT / folder).glob("*.py"))}
    metadata = dict(experiment=name, config=cfg, quick=args.quick, code_sha256=hashes,
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    cost_convention="absolute sampled evolution time; unit per complex datum",
                    workers=args.workers)
    completed = set()
    if metadata_path.exists() or raw_path.exists():
        if not args.resume or not metadata_path.exists() or not raw_path.exists():
            parser.error("Output exists. Use --resume for the same settings, or choose a new --output.")
        old = json.loads(metadata_path.read_text())
        if any(old[k] != metadata[k] for k in ["experiment", "config", "code_sha256", "numpy", "scipy"]):
            parser.error("Resume requires unchanged settings, source code, NumPy, and SciPy.")
        with raw_path.open(newline="") as stream:
            completed = {key(row) for row in csv.DictReader(stream)}
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        with raw_path.open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    schemes = ["independent"] if name == "multiplicity" else ["independent", "shared"]
    ns = [cfg["fixed_N"]] if name == "multiplicity" else cfg["N_values"]
    jobs = [(name, cfg, scheme, N, L, trial)
            for scheme in schemes for N in ns for L in cfg["L_values"]
            for trial in range(cfg["trials"]) if (scheme, N, L, trial) not in completed]
    print(f"{name}: {len(jobs)} trials remaining; output: {output}", flush=True)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = pool.map(run_trial, jobs, chunksize=1) if pool else map(run_trial, jobs)
        with raw_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            for index, row in enumerate(results, 1):
                writer.writerow(row)
                stream.flush()
                if index % cfg["trials"] == 0 or index == len(jobs):
                    print(f"  {index}/{len(jobs)}  {row['scheme']} N={row['N']} L={row['L']}", flush=True)
    finally:
        if pool:
            pool.shutdown()
        summarize(raw_path, output / "summary.csv")
    from qfames.plotting import plot_run
    plot_run(output)

