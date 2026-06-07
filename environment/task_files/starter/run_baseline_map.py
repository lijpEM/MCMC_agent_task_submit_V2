from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from load_profile_data import load_profile_data
from pseudo2d_model import forward_profile


ROOT = Path(__file__).resolve().parents[1]


def objective(theta: np.ndarray, profile: dict) -> float:
    x_m = profile["x_m"]
    water = profile["water_depth_m"]
    freqs = profile["freqs_hz"]
    water_km = water / 1000.0
    pred_rhoa, pred_phase, props = forward_profile(freqs, x_m, water, theta)
    z_lab = props["z_lab_km"]
    if np.any(z_lab <= water_km + 0.5) or theta[3] <= 1.0:
        return 1e12
    r = (profile["TM_log10_rhoa"] - pred_rhoa) / profile["TM_log10_rhoa_err"]
    p = (profile["TM_phase_deg"] - pred_phase) / profile["TM_phase_err_deg"]
    smooth = np.sum(np.diff(z_lab, 2) ** 2) if len(z_lab) > 2 else 0.0
    return float(np.sum(r * r) + np.sum(p * p) + 0.01 * smooth)


def main(input_path: str | None = None, output_dir: str | None = None) -> None:
    start = time.time()
    input_path = input_path or os.environ.get("INPUT_PROFILE_NPZ") or os.environ.get("PROFILE_DATA_PATH")
    out = Path(output_dir or os.environ.get("OUTPUT_DIR") or (ROOT / "outputs")).resolve()
    hidden_smoke_mode = os.environ.get("PSEUDO2D_HIDDEN") == "1"
    out.mkdir(parents=True, exist_ok=True)
    if hidden_smoke_mode:
        (out / "starter_hidden_smoke_skipped.txt").write_text(
            "The starter baseline intentionally does not solve hidden cases.\n",
            encoding="utf-8",
        )
        print(f"Starter hidden smoke skipped; wrote marker to {out}")
        return
    profile = load_profile_data(input_path)

    bounds = [
        (15.0, 120.0),
        (15.0, 120.0),
        (15.0, 120.0),
        (5.0, 80.0),
        (1.5, 3.5),
        (-0.5, 1.5),
        (1.0, 3.0),
    ]
    x0 = np.array([55.0, 60.0, 45.0, 35.0, 2.5, 0.8, 2.0], dtype=float)
    result = minimize(objective, x0, args=(profile,), method="Nelder-Mead", options={"maxiter": 500})
    theta = result.x
    for i, (lo, hi) in enumerate(bounds):
        theta[i] = np.clip(theta[i], lo, hi)

    pred_rhoa, pred_phase, props = forward_profile(
        profile["freqs_hz"], profile["x_m"], profile["water_depth_m"], theta
    )

    rng = np.random.default_rng(42)
    sample_scale = np.array([5.0, 5.0, 5.0, 4.0, 0.15, 0.15, 0.15])
    # Keep the starter intentionally weak: it demonstrates file formats but
    # should not be mistaken for a calibrated Bayesian posterior.
    samples = theta + rng.normal(scale=sample_scale, size=(300, theta.size))
    for i, (lo, hi) in enumerate(bounds):
        samples[:, i] = np.clip(samples[:, i], lo, hi)
    np.save(out / "posterior_samples.npy", samples)

    z_samples = []
    bottom_samples = []
    rho_cond_samples = []
    for sample in samples:
        _, _, sample_props = forward_profile(
            profile["freqs_hz"], profile["x_m"], profile["water_depth_m"], sample
        )
        z = np.asarray(sample_props["z_lab_km"], dtype=float)
        z_samples.append(z)
        bottom_samples.append(z + float(sample_props["h_cond_km"]))
        rho_cond_samples.append(np.full_like(z, float(sample_props["rho_cond_ohm_m"])))
    z_samples = np.asarray(z_samples)
    bottom_samples = np.asarray(bottom_samples)
    rho_cond_samples = np.asarray(rho_cond_samples)

    np.savez_compressed(
        out / "model_posterior_profile.npz",
        x_m=profile["x_m"],
        station_names=profile["station_names"],
        water_depth_m=profile["water_depth_m"],
        z_lab_mean_km=np.mean(z_samples, axis=0),
        z_lab_p05_km=np.percentile(z_samples, 5, axis=0),
        z_lab_p50_km=np.percentile(z_samples, 50, axis=0),
        z_lab_p95_km=np.percentile(z_samples, 95, axis=0),
        z_bottom_mean_km=np.mean(bottom_samples, axis=0),
        z_bottom_p05_km=np.percentile(bottom_samples, 5, axis=0),
        z_bottom_p50_km=np.percentile(bottom_samples, 50, axis=0),
        z_bottom_p95_km=np.percentile(bottom_samples, 95, axis=0),
        rho_cond_mean_ohm_m=np.mean(rho_cond_samples, axis=0),
        rho_cond_p50_ohm_m=np.percentile(rho_cond_samples, 50, axis=0),
    )

    np.savez_compressed(
        out / "data_pred_mean.npz",
        station_names=profile["station_names"],
        freqs_hz=profile["freqs_hz"],
        TM_obs_log10_rhoa=profile["TM_log10_rhoa"],
        TM_pred_log10_rhoa=pred_rhoa,
        TM_sigma_log10_rhoa=profile["TM_log10_rhoa_err"],
        TM_obs_phase_deg=profile["TM_phase_deg"],
        TM_pred_phase_deg=pred_phase,
        TM_sigma_phase_deg=profile["TM_phase_err_deg"],
    )

    residual_rhoa = profile["TM_log10_rhoa"] - pred_rhoa
    residual_phase = profile["TM_phase_deg"] - pred_phase
    chi2 = np.sum((residual_rhoa / profile["TM_log10_rhoa_err"]) ** 2)
    chi2 += np.sum((residual_phase / profile["TM_phase_err_deg"]) ** 2)
    dof = residual_rhoa.size + residual_phase.size - theta.size
    summary = {
        "method": "starter_map_plus_gaussian_approximation",
        "mode_used": "TM",
        "n_stations": int(len(profile["station_names"])),
        "n_freqs": int(len(profile["freqs_hz"])),
        "n_parameters": int(theta.size),
        "n_samples": int(samples.shape[0]),
        "burn_in": 0,
        "acceptance_rate": 0.0,
        "ess_min": None,
        "rhat_max": None,
        "chi2_per_dof": float(chi2 / max(dof, 1)),
        "rmse_log10_rhoa": float(np.sqrt(np.mean(residual_rhoa ** 2))),
        "rmse_phase_deg": float(np.sqrt(np.mean(residual_phase ** 2))),
        "runtime_seconds": float(time.time() - start),
        "prior_description": "bounded starter parameters; not a full Bayesian prior",
        "forward_description": "laterally coupled 1D MT forward with three LAB controls",
        "optimizer_success": bool(result.success),
        "starter_hidden_smoke_mode": bool(hidden_smoke_mode),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (out / "report.md").write_text(
        "# Starter Baseline Report\n\n"
        "This run uses TM apparent resistivity and phase with a simple three-control "
        "pseudo-2D LAB curve, a global conductor thickness, and global lithosphere, "
        "conductor, and deep resistivities. The model includes station-specific "
        "seawater thickness from the observed water depth.\n\n"
        "The posterior samples written here are only a Gaussian approximation around "
        "a deterministic MAP-style fit. They are included to document the expected "
        "file format, not as a complete Bayesian solution. A competitive submission "
        "should replace this with MCMC, SMC, VI, or another posterior approximation "
        "using explicit prior and likelihood definitions.\n\n"
        "Important limitations are the 1D forward assumption at each station, the "
        "lack of true 2D current flow, omission of TE mode, simplified smoothness "
        "control, and approximate uncertainty calibration.\n",
        encoding="utf-8",
    )

    source_dir = Path(__file__).resolve().parent
    for name in ["load_profile_data.py", "mt1d_forward.py", "pseudo2d_model.py"]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, out / name)
    shutil.copy2(Path(__file__).resolve(), out / "run_inversion.py")
    print(f"Wrote starter outputs to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to profile NPZ. Defaults to data/mt_profile_20_public.npz.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to outputs/.")
    args = parser.parse_args()
    main(input_path=args.input, output_dir=args.output)
