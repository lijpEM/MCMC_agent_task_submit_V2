#!/usr/bin/env python3
"""
Formal scorer for the pseudo-2D Bayesian marine MT LAB inversion task.

This evaluator follows the v0.2 scoring design:

    A. Runability and artifact completeness          8
    B. MT data handling and 1D forward modelling    14
    C. Pseudo-2D LAB Bayesian modelling             35
    D. LAB result quality and posterior calibration 25
    E. Generalization, robustness, anti-hardcoding  10
    F. Report and geophysical explanation            8

The final score is:

    final_score = raw_total * structural_penalty

where structural_penalty is 1.00, 0.50, 0.34, or 0.25 for severe
MAP/Laplace-only fast pipelines. Severe hidden result-quality failures can
also apply a final correctness ceiling without becoming structural failures.

Usage:

    python evaluate.py /path/to/submission

The submission may contain outputs directly or under an outputs/ subdirectory.
The evaluator writes eval_result.json next to the evaluated outputs and prints
a SE-Bench-style structured JSON block.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPONENT_MAX = {
    "A": 8.0,
    "B": 14.0,
    "C": 35.0,
    "D": 25.0,
    "E": 10.0,
    "F": 8.0,
}

REQUIRED_OUTPUTS = [
    "run_inversion.py",
    "pseudo2d_model.py",
    "posterior_samples.npy",
    "model_posterior_profile.npz",
    "data_pred_mean.npz",
    "summary.json",
    "report.md",
]

REQUIRED_PROFILE_KEYS = [
    "x_m",
    "station_names",
    "water_depth_m",
    "z_lab_mean_km",
    "z_lab_p05_km",
    "z_lab_p50_km",
    "z_lab_p95_km",
]

PROFILE_EXTRA_KEYS = [
    "z_bottom_mean_km",
    "z_bottom_p05_km",
    "z_bottom_p50_km",
    "z_bottom_p95_km",
    "h_cond_mean_km",
    "h_cond_p05_km",
    "h_cond_p50_km",
    "h_cond_p95_km",
    "rho_cond_mean_ohm_m",
    "rho_cond_p50_ohm_m",
    "log10_rho_cond_mean",
    "log10_rho_cond_p50",
]

SUMMARY_KEYS = [
    "method",
    "mode_used",
    "n_stations",
    "n_freqs",
    "n_parameters",
    "n_samples",
    "burn_in",
    "acceptance_rate",
    "chi2_per_dof",
    "rmse_log10_rhoa",
    "rmse_phase_deg",
    "runtime_seconds",
    "prior_description",
    "forward_description",
]

HIDDEN_INPUT_EXCLUDE_KEYS = {
    "case_id",
    "z_lab_true_km",
    "h_cond_true_km",
    "z_bottom_true_km",
    "log10_rho_cond_true",
}

PUBLIC_STATION_SUBSET = [
    "s01", "s04", "s06", "s08", "s10", "s12", "s14", "s16", "s18", "s20",
    "s22", "s24", "s26", "s35", "s37", "s39", "s42", "s44", "s46", "s48",
]

REPORT_TERMS = {
    "lab": ["lab", "lithosphere-asthenosphere", "岩石圈", "软流圈"],
    "bayes": ["bayesian", "posterior", "prior", "likelihood", "贝叶斯", "后验", "先验"],
    "uncertainty": ["uncertainty", "credible", "interval", "p05", "p95", "不确定", "可信区间"],
    "pseudo2d": ["pseudo-2d", "pseudo2d", "拟二维", "laterally", "横向", "control point", "spline"],
    "mt": ["magnetotelluric", "mt", "apparent resistivity", "rhoa", "phase", "大地电磁"],
    "water": ["seawater", "water depth", "bathymetry", "海水", "水深"],
    "limitations": ["limitation", "assumption", "1d", "2d", "static", "anisotropy", "局限", "假设"],
    "results": ["rmse", "chi2", "credible", "coverage", "residual", "结果", "误差"],
}

MU0 = 4.0 * math.pi * 1e-7


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _item(score: float, max_score: float, **extra: Any) -> dict[str, Any]:
    return {"score": float(np.clip(score, 0.0, max_score)), "max_score": float(max_score), **extra}


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size <= 24:
            return _json_safe(value.tolist())
        summary: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if np.issubdtype(value.dtype, np.number):
            finite = value[np.isfinite(value)]
            if finite.size:
                summary.update(min=float(np.min(finite)), max=float(np.max(finite)))
        return summary
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def finite_fraction(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).sum() / arr.size)


def resolve_output_dir(submission_dir: Path) -> Path:
    outputs = submission_dir / "outputs"
    return outputs if outputs.is_dir() else submission_dir


def load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj, None
        return {}, "JSON root is not an object"
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], str | None]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}, None
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def load_npy(path: Path) -> tuple[np.ndarray | None, str | None]:
    try:
        return np.load(path, allow_pickle=False), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def posterior_sample_diagnostics(samples: np.ndarray | None) -> dict[str, Any]:
    """Estimate lightweight diagnostics from submitted posterior samples.

    These diagnostics are not a replacement for a full chain analysis, but they
    prevent summary.json from being the sole source of ESS/Rhat evidence.
    """
    details: dict[str, Any] = {
        "available": False,
        "n_samples": None,
        "n_parameters": None,
        "active_parameter_count": 0,
        "finite_fraction": 0.0,
        "ess_min_estimate": None,
        "split_rhat_estimate": None,
        "ess_min_fraction": None,
        "lag1_abs_median": None,
        "lag1_abs_min": None,
        "diagnostics_good": False,
        "diagnostics_present": False,
    }
    if samples is None:
        return details
    arr = np.asarray(samples)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        return details
    arr = arr.astype(float, copy=False)
    n_samples, n_params = arr.shape
    details.update(
        available=True,
        n_samples=int(n_samples),
        n_parameters=int(n_params),
        finite_fraction=finite_fraction(arr),
    )
    if n_samples < 4 or n_params < 1 or details["finite_fraction"] < 0.99:
        return details

    std = np.nanstd(arr, axis=0)
    active_mask = np.isfinite(std) & (std > 1e-8)
    active = int(np.sum(active_mask))
    details["active_parameter_count"] = active
    if active == 0:
        return details

    use = arr[:, active_mask]
    if use.shape[1] > 24:
        order = np.argsort(std[active_mask])[::-1][:24]
        use = use[:, order]

    # Split-Rhat from four contiguous chunks. It is intentionally conservative
    # for particle clouds, but catches obvious repeated or collapsed samples.
    n_chains = 4
    chain_len = n_samples // n_chains
    if chain_len >= 4:
        trimmed = use[: n_chains * chain_len].reshape(n_chains, chain_len, use.shape[1])
        chain_means = np.mean(trimmed, axis=1)
        chain_vars = np.var(trimmed, axis=1, ddof=1)
        b = chain_len * np.var(chain_means, axis=0, ddof=1)
        w = np.mean(chain_vars, axis=0)
        var_hat = ((chain_len - 1.0) / chain_len) * w + b / chain_len
        rhat = np.sqrt(np.divide(var_hat, w, out=np.ones_like(var_hat), where=w > 0.0))
        details["split_rhat_estimate"] = _finite_float(np.nanmax(rhat))

    ess_values: list[float] = []
    lag1_abs_values: list[float] = []
    max_cols = min(use.shape[1], 12)
    max_lag = min(100, n_samples // 2)
    for j in range(max_cols):
        x = use[:, j] - np.mean(use[:, j])
        var = float(np.var(x))
        if not np.isfinite(var) or var <= 0.0:
            ess_values.append(1.0)
            continue
        rho_sum = 0.0
        for lag in range(1, max_lag):
            ac = float(np.dot(x[:-lag], x[lag:]) / ((n_samples - lag) * var))
            if lag == 1 and np.isfinite(ac):
                lag1_abs_values.append(abs(ac))
            if not np.isfinite(ac) or ac <= 0.0:
                break
            rho_sum += ac
        ess_values.append(float(n_samples / max(1.0 + 2.0 * rho_sum, 1.0)))
    if ess_values:
        ess_min = _finite_float(np.nanmin(ess_values))
        details["ess_min_estimate"] = ess_min
        if ess_min is not None and n_samples > 0:
            details["ess_min_fraction"] = _finite_float(ess_min / float(n_samples))
    if lag1_abs_values:
        details["lag1_abs_median"] = _finite_float(np.nanmedian(lag1_abs_values))
        details["lag1_abs_min"] = _finite_float(np.nanmin(lag1_abs_values))

    ess_est = _finite_float(details.get("ess_min_estimate"))
    rhat_est = _finite_float(details.get("split_rhat_estimate"))
    enough_variation = active >= max(3, min(8, n_params // 2))
    details["diagnostics_present"] = bool(
        enough_variation
        and n_samples >= 200
        and (
            (ess_est is not None and ess_est >= 50.0)
            or (rhat_est is not None and 0.85 <= rhat_est <= 1.30)
        )
    )
    details["diagnostics_good"] = bool(
        enough_variation
        and n_samples >= 1000
        and ess_est is not None
        and ess_est >= 100.0
        and (rhat_est is None or 0.90 <= rhat_est <= 1.20)
    )
    return details


def hidden_case_input(case: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the NPZ payload given to submissions during hidden reruns.

    Truth arrays are evaluator-only labels. They must not be placed in the
    hidden input file, otherwise a submission can read the answer directly or
    branch on case identifiers instead of solving the inverse problem.
    """
    return {k: v for k, v in case.items() if k not in HIDDEN_INPUT_EXCLUDE_KEYS}


def as_numeric_1d(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.ndim != 1:
        return None
    try:
        return arr.astype(float)
    except Exception:
        return None


def arr_key(data: dict[str, np.ndarray], candidates: list[str]) -> str | None:
    lower = {k.lower(): k for k in data}
    for cand in candidates:
        c = cand.lower()
        if c in lower:
            return lower[c]
    for cand in candidates:
        parts = cand.lower().split()
        for lk, original in lower.items():
            if all(part in lk for part in parts):
                return original
    return None


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return math.inf
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def ci_coverage(true: np.ndarray, p05: np.ndarray, p95: np.ndarray) -> float:
    true = np.asarray(true, dtype=float)
    p05 = np.asarray(p05, dtype=float)
    p95 = np.asarray(p95, dtype=float)
    if true.shape != p05.shape or true.shape != p95.shape or true.size == 0:
        return 0.0
    return float(np.mean((true >= p05) & (true <= p95)))


# ---------------------------------------------------------------------------
# Public data loading
# ---------------------------------------------------------------------------

def candidate_public_paths() -> list[Path]:
    here = Path(__file__).resolve()
    roots = [
        here.parent,
        here.parents[1] if len(here.parents) > 1 else here.parent,
        here.parents[2] if len(here.parents) > 2 else here.parent,
        Path("/opt/mcmc_task_files"),
        Path("/opt/pseudo2d_mt_lab"),
        Path("/home/workspace/pseudo2d_mt_lab"),
        Path("/workspace"),
        Path.cwd(),
    ]
    paths: list[Path] = []
    for root in roots:
        paths += [
            root / "data" / "mt_profile_20_public.npz",
            root / "data" / "serpent_mt" / "mt_profile_20_public.npz",
            root / "data" / "SERPENT_fullMTdataSet.txt",
            root / "data" / "serpent_mt" / "SERPENT_fullMTdataSet.txt",
        ]
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            out.append(p)
    return out


def read_profile_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def parse_mare2dem_mt(path: Path, station_subset: list[str] | None = None) -> dict[str, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    freqs: list[float] = []
    receivers: list[dict[str, Any]] = []
    data_rows: list[tuple[int, int, int, int, float, float]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("# MT Frequencies"):
            n = int(re.findall(r"\d+", line)[-1])
            for j in range(n):
                freqs.append(float(lines[i + 1 + j].strip()))
            i += n + 1
            continue
        if line.startswith("# MT Receivers"):
            n = int(re.findall(r"\d+", line)[-1])
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("!"):
                j += 1
            for k in range(n):
                parts = lines[j + k].split()
                if len(parts) >= 8:
                    receivers.append({
                        "name": parts[-1].strip(),
                        "x": float(parts[0]),
                        "y": float(parts[1]),
                        "z": float(parts[2]),
                    })
            i = j + n
            continue
        if line.startswith("# Data"):
            n = int(re.findall(r"\d+", line)[-1])
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("!"):
                j += 1
            for k in range(n):
                parts = lines[j + k].split()
                if len(parts) >= 6:
                    data_rows.append((
                        int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),
                        float(parts[4]), float(parts[5]),
                    ))
            break
        i += 1

    if not freqs or not receivers or not data_rows:
        raise ValueError(f"could not parse MT data from {path}")

    names_all = [r["name"] for r in receivers]
    if station_subset is None:
        station_subset = names_all
    keep = [name for name in station_subset if name in names_all]
    if not keep:
        raise ValueError("station subset not found in MT file")

    nsta = len(keep)
    nfreq = len(freqs)
    name_to_rxid = {r["name"]: idx + 1 for idx, r in enumerate(receivers)}
    rxid_to_outidx = {name_to_rxid[name]: idx for idx, name in enumerate(keep)}

    log10_rhoa = np.full((nsta, nfreq), np.nan)
    log10_rhoa_std = np.full((nsta, nfreq), np.nan)
    phase = np.full((nsta, nfreq), np.nan)
    phase_std = np.full((nsta, nfreq), np.nan)

    # TM mode: 105 = linear apparent resistivity, 106 = phase.
    for code, freq_id, _tx, rx_id, val, err in data_rows:
        if rx_id not in rxid_to_outidx:
            continue
        fi = freq_id - 1
        si = rxid_to_outidx[rx_id]
        if fi < 0 or fi >= nfreq:
            continue
        if code == 105 and val > 0:
            log10_rhoa[si, fi] = math.log10(val)
            log10_rhoa_std[si, fi] = max(err / (val * math.log(10.0)), 0.03)
        elif code == 106:
            ph = val + 180.0 if val < 0 else val
            phase[si, fi] = ph
            phase_std[si, fi] = max(err, 2.0)

    rec_by_name = {r["name"]: r for r in receivers}
    x = np.asarray([rec_by_name[name]["y"] for name in keep], dtype=float)
    x = x - np.nanmin(x)
    wd = np.asarray([rec_by_name[name]["z"] for name in keep], dtype=float)
    return {
        "station_names": np.asarray(keep),
        "x_m": x,
        "water_depth_m": wd,
        "freqs_hz": np.asarray(freqs, dtype=float),
        "log10_rhoa_tm": log10_rhoa,
        "log10_rhoa_tm_std": log10_rhoa_std,
        "phase_tm_deg": phase,
        "phase_tm_std_deg": phase_std,
    }


def load_public_profile() -> tuple[dict[str, np.ndarray] | None, str | None]:
    for path in candidate_public_paths():
        if not path.exists():
            continue
        try:
            if path.suffix == ".npz":
                return read_profile_npz(path), str(path)
            return parse_mare2dem_mt(path, PUBLIC_STATION_SUBSET), str(path)
        except Exception:
            continue
    return None, None


# ---------------------------------------------------------------------------
# 1D MT forward and hidden data generation
# ---------------------------------------------------------------------------

def mt1d_forward(rho: np.ndarray, thickness: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return apparent resistivity and phase for a 1D layered earth.

    rho has length n_layers. thickness has length n_layers - 1. The final
    layer is a half-space.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    thickness = np.asarray(thickness, dtype=float).ravel()
    freqs = np.asarray(freqs, dtype=float).ravel()
    if rho.size < 1 or thickness.size != max(0, rho.size - 1):
        raise ValueError("rho/thickness size mismatch")

    app = np.zeros_like(freqs, dtype=float)
    phase = np.zeros_like(freqs, dtype=float)
    for i, f in enumerate(freqs):
        omega = 2.0 * math.pi * float(f)
        z_imp = np.sqrt(1j * omega * MU0 * rho[-1])
        for layer in range(rho.size - 2, -1, -1):
            r = rho[layer]
            h = thickness[layer]
            dj = np.sqrt(1j * omega * MU0 / r)
            wj = dj * r
            ej = np.exp(-2.0 * h * dj)
            rj = (wj - z_imp) / (wj + z_imp)
            z_imp = wj * (1.0 - rj * ej) / (1.0 + rj * ej)
        app[i] = (abs(z_imp) ** 2) / (MU0 * omega)
        phase[i] = math.degrees(math.atan2(z_imp.imag, z_imp.real))
    return app, phase


def layer_model_for_station(
    water_depth_m: float,
    z_lab_km: float,
    thickness_km: float,
    log10_rho_cond: float,
    log10_rho_lith: float = 3.1,
    log10_rho_deep: float = 2.7,
) -> tuple[np.ndarray, np.ndarray]:
    wd = max(float(water_depth_m), 1.0)
    z_lab_m = max(float(z_lab_km) * 1000.0, 1000.0)
    cond_h_m = max(float(thickness_km) * 1000.0, 500.0)
    rho = np.asarray([
        0.3,
        10.0 ** log10_rho_lith,
        10.0 ** log10_rho_cond,
        10.0 ** log10_rho_deep,
    ], dtype=float)
    # z_lab is below seafloor. The lithosphere layer thickness starts below water.
    thickness = np.asarray([wd, z_lab_m, cond_h_m], dtype=float)
    return rho, thickness


def predict_profile(
    x_m: np.ndarray,
    water_depth_m: np.ndarray,
    freqs_hz: np.ndarray,
    z_lab_km: np.ndarray,
    thickness_km: np.ndarray,
    log10_rho_cond: float | np.ndarray,
    log10_rho_lith: float = 3.1,
    log10_rho_deep: float = 2.7,
) -> tuple[np.ndarray, np.ndarray]:
    nsta = len(x_m)
    nfreq = len(freqs_hz)
    out_log = np.zeros((nsta, nfreq), dtype=float)
    out_phase = np.zeros((nsta, nfreq), dtype=float)
    cond_arr = np.full(nsta, float(log10_rho_cond)) if np.ndim(log10_rho_cond) == 0 else np.asarray(log10_rho_cond, dtype=float)
    for i in range(nsta):
        rho, thick = layer_model_for_station(
            water_depth_m[i], z_lab_km[i], thickness_km[i], cond_arr[i],
            log10_rho_lith=log10_rho_lith,
            log10_rho_deep=log10_rho_deep,
        )
        app, ph = mt1d_forward(rho, thick, freqs_hz)
        out_log[i, :] = np.log10(np.maximum(app, 1e-30))
        out_phase[i, :] = ph
    return out_log, out_phase


def make_hidden_case(case_id: str, n_stations: int, n_freqs: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 220_000.0, n_stations)
    xn = (x - x.min()) / max(float(np.ptp(x)), 1.0)
    jitter = rng.normal(0.0, 0.018, size=n_stations)
    x = np.maximum.accumulate(x + np.ptp(x) * jitter)
    x -= x.min()
    x *= 220_000.0 / max(float(np.ptp(x)), 1.0)
    xn = (x - x.min()) / max(float(np.ptp(x)), 1.0)

    water = 1800.0 + 1300.0 * np.sin(np.linspace(0.0, 1.6 * math.pi, n_stations) + 0.3)
    water += 350.0 * np.cos(np.linspace(0.0, 4.0 * math.pi, n_stations))
    water += rng.normal(0.0, 80.0, size=n_stations)
    water = np.clip(water, 120.0, 4200.0)
    f_hi = float(rng.uniform(0.035, 0.065))
    f_lo = float(rng.uniform(0.00065, 0.00135))
    freqs = np.geomspace(f_hi, f_lo, n_freqs)

    if case_id == "smooth_deepening":
        z_lab = 30.0 + 46.0 * xn + 4.0 * np.sin(2.0 * math.pi * xn + 0.2)
        h = 22.0 + 8.0 * np.cos(2.0 * math.pi * xn + 0.15)
        logr = float(rng.uniform(0.95, 1.15))
    elif case_id == "local_upwarp":
        center = float(rng.uniform(0.42, 0.66))
        width = float(rng.uniform(0.11, 0.19))
        z_lab = 64.0 - 26.0 * np.exp(-0.5 * ((xn - center) / width) ** 2) + 7.0 * xn
        h = 17.0 + 12.0 * np.exp(-0.5 * ((xn - min(center + 0.03, 0.85)) / 0.23) ** 2)
        logr = float(rng.uniform(0.75, 0.98))
    elif case_id == "thin_conductor":
        z_lab = 45.0 + 12.0 * np.sin(2.0 * math.pi * xn + 0.4) + 14.0 * xn
        h = 10.0 + 9.0 / (1.0 + np.exp(-(xn - 0.48) / 0.035))
        logr = float(rng.uniform(1.25, 1.45))
    elif case_id == "rugged_transition":
        z_lab = 55.0 + 18.0 * np.sin(1.5 * math.pi * xn + 0.25) + 9.0 * np.sin(5.0 * math.pi * xn)
        z_lab += 10.0 * (xn - 0.5)
        h = 16.0 + 5.0 * np.sin(3.0 * math.pi * xn + 0.6)
        logr = float(rng.uniform(0.7, 1.35))
    else:
        z_lab = 42.0 + 38.0 * xn + 7.0 * np.sin(4.0 * math.pi * xn + 0.9)
        h = 14.0 + 7.0 * np.cos(2.5 * math.pi * xn)
        logr = float(rng.uniform(0.9, 1.4))

    log_pred, ph_pred = predict_profile(x, water, freqs, z_lab, h, logr)
    sig_log = np.full_like(log_pred, 0.055)
    sig_ph = np.full_like(ph_pred, 2.5)
    log_obs = log_pred + rng.normal(0.0, sig_log)
    ph_obs = ph_pred + rng.normal(0.0, sig_ph)
    return {
        "case_id": np.asarray(case_id),
        "station_names": np.asarray([f"h{i+1:02d}" for i in range(n_stations)]),
        "x_m": x,
        "water_depth_m": water,
        "freqs_hz": freqs,
        "log10_rhoa_tm": log_obs,
        "log10_rhoa_tm_std": sig_log,
        "phase_tm_deg": ph_obs,
        "phase_tm_std_deg": sig_ph,
        "z_lab_true_km": z_lab,
        "h_cond_true_km": h,
        "z_bottom_true_km": z_lab + h,
        "log10_rho_cond_true": np.asarray(logr),
    }


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------

class StaticVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.calls: set[str] = set()
        self.names: set[str] = set()
        self.strings: list[str] = []
        self.float_literals: list[float] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0].lower())
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split(".")[0].lower())
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            self.calls.add(func.id.lower())
        elif isinstance(func, ast.Attribute):
            self.calls.add(func.attr.lower())
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id.lower())

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.append(node.value.lower())
        elif isinstance(node.value, (int, float)):
            self.float_literals.append(float(node.value))


def parse_submission_code(out_dir: Path) -> tuple[StaticVisitor, list[str], str]:
    visitor = StaticVisitor()
    errors: list[str] = []
    for rel in ["run_inversion.py", "pseudo2d_model.py", "load_profile_data.py", "mt1d_forward.py"]:
        path = out_dir / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(path))
            visitor.visit(tree)
        except Exception as exc:
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")
    # Use AST-visible names/calls/strings only. Raw source text would let
    # comments trigger method evidence without executable implementation.
    joined = " ".join(
        list(visitor.imports) + list(visitor.names) + list(visitor.calls) + visitor.strings
    )
    return visitor, errors, joined


def static_features(out_dir: Path) -> dict[str, Any]:
    visitor, errors, text = parse_submission_code(out_dir)
    station_hits = sum(1 for s in PUBLIC_STATION_SUBSET if re.search(rf"\b{s}\b", text))
    return {
        "parse_errors": errors,
        "uses_numpy_or_scipy": bool({"numpy", "scipy"} & visitor.imports) or "np" in visitor.names,
        "has_mt_forward_terms": any(t in text for t in ["mt1d", "z1d", "impedance", "apparent resistivity", "rhoa", "phase"]),
        "has_water_depth_terms": any(t in text for t in ["water_depth", "seawater", "bathymetry", "水深", "海水"]),
        "has_log10_terms": any(t in text for t in ["log10", "apparent resistivity", "rhoa"]),
        "has_lateral_basis_terms": any(t in text for t in ["spline", "basis", "control", "low-rank", "gp", "gaussianprocess", "pca", "横向"]),
        "has_bayesian_terms": any(t in text for t in ["posterior", "prior", "likelihood", "mcmc", "metropolis", "vi", "variational", "smc", "sample"]),
        "has_mcmc_or_vi": any(t in text for t in ["mcmc", "metropolis", "acceptance", "burn", "rhat", "ess", "elbo", "variational", "smc"]),
        "has_uncertainty_terms": any(t in text for t in ["credible", "p05", "p95", "percentile", "quantile", "uncertainty"]),
        "has_adaptive_or_surrogate": any(t in text for t in ["adaptive", "surrogate", "delayed", "proposal", "covariance", "reduced"]),
        "has_optimizer_terms": any(t in text for t in ["least_squares", "curve_fit", "minimize", "differential_evolution", "optimizer"]),
        "has_laplace_terms": any(t in text for t in ["laplace", "hessian", "jacobian", "covariance"]),
        "has_sample_order_destroying_terms": (
            ("permutation" in visitor.calls or "shuffle" in visitor.calls or "permutation" in visitor.names or "shuffle" in visitor.names)
            and ("posterior_samples.npy" in visitor.strings or "posterior_samples" in text)
        ),
        "has_hidden_specific_terms": any(t in text for t in ["pseudo2d_hidden", "hidden_profile", "hidden synthetic"]),
        "has_hidden_model_branch_terms": any(
            t in text
            for t in [
                "canonical_marine_water",
                "water_response_mix",
                "maybe_write_canonical_solution",
                "canonical_laplace_samples",
            ]
        ),
        "map_only_terms": any(t in text for t in ["least_squares", "curve_fit", "minimize", "differential_evolution"]) and not any(
            t in text for t in ["posterior", "mcmc", "variational", "sample"]
        ),
        "hardcodes_public_station_names": station_hits >= 10,
        "station_name_hit_count": station_hits,
        "possibly_per_station_independent": any(t in text for t in ["for station", "for sta", "for i, station", "independent"]) and not any(
            t in text for t in ["spline", "control", "basis", "smooth", "横向"]
        ),
        "hidden_specific_branch_risk": (
            any(t in text for t in ["canonical_marine_water", "water_response_mix", "maybe_write_canonical_solution"])
            and any(t in text for t in ["pseudo2d_hidden", "hidden_profile", "hidden synthetic"])
        ),
    }


# ---------------------------------------------------------------------------
# Scoring subroutines
# ---------------------------------------------------------------------------

def score_A(out_dir: Path) -> tuple[float, dict[str, Any]]:
    missing = [name for name in REQUIRED_OUTPUTS if not (out_dir / name).exists()]
    score = 4.0 * (len(REQUIRED_OUTPUTS) - len(missing)) / len(REQUIRED_OUTPUTS)
    details: dict[str, Any] = {"missing": missing, "output_dir": str(out_dir)}

    summary, summary_err = load_json(out_dir / "summary.json")
    profile, profile_err = load_npz(out_dir / "model_posterior_profile.npz")
    pred, pred_err = load_npz(out_dir / "data_pred_mean.npz")
    samples, sample_err = load_npy(out_dir / "posterior_samples.npy")

    if summary_err is None:
        score += 0.75
    else:
        details["summary_error"] = summary_err
    if profile_err is None:
        score += 0.75
    else:
        details["profile_error"] = profile_err
    if pred_err is None:
        score += 0.75
    else:
        details["prediction_error"] = pred_err
    if sample_err is None and samples is not None:
        score += 0.75
        details["posterior_shape"] = list(samples.shape)
    else:
        details["posterior_error"] = sample_err

    finite_ok = True
    for obj in [profile, pred]:
        for arr in obj.values():
            if np.issubdtype(arr.dtype, np.number) and finite_fraction(arr) < 0.995:
                finite_ok = False
    if samples is not None and np.issubdtype(samples.dtype, np.number) and finite_fraction(samples) < 0.995:
        finite_ok = False
    details["finite_outputs"] = finite_ok
    if finite_ok and not missing:
        score += 1.0

    return float(np.clip(score, 0.0, COMPONENT_MAX["A"])), details


def prediction_keymap(pred: dict[str, np.ndarray]) -> dict[str, str | None]:
    return {
        "log_obs": arr_key(pred, ["log10_rhoa_obs", "obs rhoa", "data rhoa"]),
        "log_pred": arr_key(pred, ["log10_rhoa_pred_mean", "pred rhoa", "mean rhoa"]),
        "log_std": arr_key(pred, ["log10_rhoa_std", "sigma rhoa", "err rhoa"]),
        "phase_obs": arr_key(pred, ["phase_obs_deg", "obs phase", "data phase"]),
        "phase_pred": arr_key(pred, ["phase_pred_mean_deg", "pred phase", "mean phase"]),
        "phase_std": arr_key(pred, ["phase_std_deg", "sigma phase", "err phase"]),
        "freqs": arr_key(pred, ["freqs_hz", "frequency"]),
        "station_names": arr_key(pred, ["station_names"]),
    }


def trusted_profile_keymap(profile: dict[str, np.ndarray]) -> dict[str, str | None]:
    return {
        "log_obs": arr_key(profile, ["TM_log10_rhoa", "log10_rhoa_tm", "log10_rhoa"]),
        "phase_obs": arr_key(profile, ["TM_phase_deg", "phase_tm_deg", "phase"]),
        "log_std": arr_key(profile, ["TM_log10_rhoa_err", "log10_rhoa_tm_std", "log10_rhoa_std", "rhoa_std"]),
        "phase_std": arr_key(profile, ["TM_phase_err_deg", "phase_tm_std_deg", "phase_std_deg", "phase_std"]),
    }


def align_reference_to_prediction(reference: np.ndarray | None, pred_arr: np.ndarray) -> np.ndarray | None:
    if reference is None:
        return None
    ref = np.asarray(reference, dtype=float)
    pp = np.asarray(pred_arr, dtype=float)
    if ref.shape == pp.shape:
        return ref
    if ref.ndim == 2 and pp.ndim == 2 and ref.T.shape == pp.shape:
        return ref.T
    if pp.ndim == 1 and ref.size == pp.size:
        return ref.reshape(pp.shape)
    return None


def trusted_prediction_metrics(
    pred: dict[str, np.ndarray],
    reference_profile: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    """Compute prediction fit against evaluator-trusted MT observations.

    Submitted ``data_pred_mean.npz`` files often include both observed and
    predicted arrays for plotting. Those observed arrays are submission-owned,
    so they cannot be the authority for scoring residuals. This helper uses
    the public/hidden profile given by the evaluator as the observation source.
    """
    details: dict[str, Any] = {
        "available": False,
        "prediction_keymap": prediction_keymap(pred),
        "reference_keymap": {},
        "rmse_log10_rhoa": None,
        "rmse_phase_deg": None,
        "chi2_per_dof": None,
        "submitted_log_obs_matches_reference": None,
        "submitted_phase_obs_matches_reference": None,
    }
    if reference_profile is None:
        details["error"] = "trusted_reference_profile_unavailable"
        return details

    pmap = details["prediction_keymap"]
    rmap = trusted_profile_keymap(reference_profile)
    details["reference_keymap"] = rmap
    chi_terms: list[np.ndarray] = []

    def metric_pair(pred_key: str | None, ref_key: str | None, std_key: str | None) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        if not pred_key or not ref_key:
            return None, None, None
        try:
            pp = np.asarray(pred[pred_key], dtype=float)
            rr = align_reference_to_prediction(reference_profile[ref_key], pp)
        except Exception:
            return None, None, None
        if rr is None or pp.shape != rr.shape or pp.size == 0:
            return None, None, None
        mask = np.isfinite(pp) & np.isfinite(rr)
        if not np.any(mask):
            return None, pp, rr
        val = float(np.sqrt(np.nanmean((pp[mask] - rr[mask]) ** 2)))
        if std_key and std_key in reference_profile:
            ss = align_reference_to_prediction(reference_profile[std_key], pp)
            if ss is not None and ss.shape == pp.shape:
                smask = mask & np.isfinite(ss) & (ss > 0.0)
                if np.any(smask):
                    chi_terms.append(((pp[smask] - rr[smask]) / np.maximum(ss[smask], 1e-12)) ** 2)
        return _finite_float(val), pp, rr

    log_rmse, log_pred_arr, log_ref_arr = metric_pair(pmap["log_pred"], rmap["log_obs"], rmap["log_std"])
    phase_rmse, phase_pred_arr, phase_ref_arr = metric_pair(pmap["phase_pred"], rmap["phase_obs"], rmap["phase_std"])
    details["rmse_log10_rhoa"] = log_rmse
    details["rmse_phase_deg"] = phase_rmse
    if chi_terms:
        details["chi2_per_dof"] = _finite_float(np.nanmean(np.concatenate([x.ravel() for x in chi_terms])))

    def submitted_obs_matches(obs_key: str | None, ref_arr: np.ndarray | None) -> bool | None:
        if not obs_key or ref_arr is None:
            return None
        try:
            obs = np.asarray(pred[obs_key], dtype=float)
        except Exception:
            return None
        rr = align_reference_to_prediction(ref_arr, obs)
        if rr is None or obs.shape != rr.shape:
            return False
        mask = np.isfinite(obs) & np.isfinite(rr)
        if not np.any(mask):
            return False
        return bool(np.nanmax(np.abs(obs[mask] - rr[mask])) <= 1e-6)

    details["submitted_log_obs_matches_reference"] = submitted_obs_matches(pmap["log_obs"], log_ref_arr)
    details["submitted_phase_obs_matches_reference"] = submitted_obs_matches(pmap["phase_obs"], phase_ref_arr)
    details["available"] = bool(log_rmse is not None or phase_rmse is not None)
    return details


def score_B(out_dir: Path, summary: dict[str, Any], features: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
    pred, pred_err = load_npz(out_dir / "data_pred_mean.npz")
    details: dict[str, Any] = {"prediction_error": pred_err, "static": features}
    failures: list[str] = []
    score = 0.0

    if pred_err is None:
        keymap = prediction_keymap(pred)
        details["recognized_prediction_keys"] = keymap
        required = ["log_obs", "log_pred", "phase_obs", "phase_pred"]
        score += 3.0 * sum(1 for k in required if keymap[k]) / len(required)

        if keymap["log_obs"] and keymap["log_pred"]:
            obs = np.asarray(pred[keymap["log_obs"]], dtype=float)
            pp = np.asarray(pred[keymap["log_pred"]], dtype=float)
            if obs.shape == pp.shape and obs.size:
                score += 1.0
                public_profile, public_path = load_public_profile()
                fit = trusted_prediction_metrics(pred, public_profile)
                details["trusted_public_fit"] = fit
                details["trusted_public_profile_path"] = public_path
                computed = _finite_float(fit.get("rmse_log10_rhoa"))
                details["computed_rmse_log10_rhoa"] = computed
                if computed is not None and computed < 0.25:
                    score += 1.0

        if keymap["phase_obs"] and keymap["phase_pred"]:
            obs = np.asarray(pred[keymap["phase_obs"]], dtype=float)
            pp = np.asarray(pred[keymap["phase_pred"]], dtype=float)
            if obs.shape == pp.shape and obs.size:
                score += 1.0
                fit = details.get("trusted_public_fit")
                if not isinstance(fit, dict):
                    public_profile, public_path = load_public_profile()
                    fit = trusted_prediction_metrics(pred, public_profile)
                    details["trusted_public_fit"] = fit
                    details["trusted_public_profile_path"] = public_path
                computed = _finite_float(fit.get("rmse_phase_deg"))
                details["computed_rmse_phase_deg"] = computed
                if computed is not None and computed < 12.0:
                    score += 1.0

        shapes = []
        for k in ["log_obs", "log_pred", "phase_obs", "phase_pred"]:
            if keymap[k]:
                shapes.append(tuple(np.asarray(pred[keymap[k]]).shape))
        details["prediction_shapes"] = [list(s) for s in shapes]
        if shapes and len(set(shapes)) <= 2:
            score += 1.0

    if features["has_mt_forward_terms"]:
        score += 2.0
    else:
        failures.append("no_real_1d_mt_forward_evidence")

    if features["has_water_depth_terms"]:
        score += 2.0
    else:
        failures.append("water_depth_not_used")

    if features["has_log10_terms"]:
        score += 1.0
    else:
        failures.append("apparent_resistivity_log10_handling_missing")

    mode = str(summary.get("mode_used", "")).upper()
    details["mode_used"] = mode
    if "TM" in mode:
        score += 1.0

    fit = details.get("trusted_public_fit")
    chi2 = _finite_float(fit.get("chi2_per_dof")) if isinstance(fit, dict) else None
    if chi2 is not None:
        details["chi2_per_dof_computed"] = chi2
        if 0.4 <= chi2 <= 3.0:
            score += 1.0

    # Hard caps from the rubric.
    cap = COMPONENT_MAX["B"]
    if "water_depth_not_used" in failures:
        cap = min(cap, 6.0)
    if "no_real_1d_mt_forward_evidence" in failures:
        cap = min(cap, 5.0)
    if "apparent_resistivity_log10_handling_missing" in failures:
        cap = min(cap, 8.0)
    if pred_err is None:
        keymap = prediction_keymap(pred)
        if not (keymap["log_obs"] and keymap["phase_obs"]):
            cap = min(cap, 10.0)

    return float(np.clip(score, 0.0, cap)), details, failures


def profile_quality(out_dir: Path) -> tuple[dict[str, Any], list[str]]:
    profile, err = load_npz(out_dir / "model_posterior_profile.npz")
    details: dict[str, Any] = {"error": err}
    failures: list[str] = []
    if err is not None:
        return details, ["profile_missing_or_unreadable"]

    details["keys"] = sorted(profile.keys())
    missing = [k for k in REQUIRED_PROFILE_KEYS if k not in profile]
    details["missing_profile_keys"] = missing
    x = as_numeric_1d(profile, "x_m")
    wd = as_numeric_1d(profile, "water_depth_m")
    z_mean = as_numeric_1d(profile, "z_lab_mean_km")
    z05 = as_numeric_1d(profile, "z_lab_p05_km")
    z50 = as_numeric_1d(profile, "z_lab_p50_km")
    z95 = as_numeric_1d(profile, "z_lab_p95_km")

    details.update(
        n_stations=int(x.size) if x is not None else None,
        has_required_keys=(len(missing) == 0),
    )

    if x is not None and wd is not None and z_mean is not None:
        same = x.size == wd.size == z_mean.size
        details["profile_lengths_consistent"] = same
        if same:
            details["median_z_lab_km"] = _finite_float(np.nanmedian(z_mean))
            details["median_water_depth_km"] = _finite_float(np.nanmedian(wd / 1000.0))
            details["lab_below_seafloor"] = bool(np.all(z_mean > wd / 1000.0))
            details["lab_plausible_range"] = bool(5.0 <= np.nanmedian(z_mean) <= 180.0)
            rough = np.diff(z_mean, n=2) if z_mean.size >= 3 else np.array([])
            details["lab_second_diff_rms_km"] = _finite_float(np.sqrt(np.nanmean(rough ** 2))) if rough.size else 0.0
            if not details["lab_below_seafloor"]:
                failures.append("lab_not_below_seafloor")
            if not details["lab_plausible_range"]:
                failures.append("lab_depth_implausible")

    if z05 is not None and z50 is not None and z95 is not None and z05.shape == z50.shape == z95.shape:
        ordered = bool(np.all((z05 <= z50) & (z50 <= z95)))
        widths = z95 - z05
        details["credible_intervals_ordered"] = ordered
        details["median_ci_width_km"] = _finite_float(np.nanmedian(widths))
        details["ci_width_std_km"] = _finite_float(np.nanstd(widths))
        if not ordered:
            failures.append("credible_interval_order_invalid")
        if np.nanmedian(widths) <= 0.05:
            failures.append("credible_interval_too_narrow_or_fixed")
    else:
        failures.append("credible_interval_missing")

    bottom = None
    for key in ["z_bottom_mean_km", "z_bottom_p50_km"]:
        if key in profile:
            bottom = as_numeric_1d(profile, key)
            break
    thick = None
    for key in ["h_cond_mean_km", "h_cond_p50_km"]:
        if key in profile:
            thick = as_numeric_1d(profile, key)
            break
    if z_mean is not None and bottom is not None and bottom.size == z_mean.size:
        ok = bool(np.all(bottom > z_mean))
        details["bottom_deeper_than_lab"] = ok
        if not ok:
            failures.append("negative_or_crossing_conductor_thickness")
    elif thick is not None:
        ok = bool(np.all(thick > 0))
        details["positive_conductor_thickness"] = ok
        if not ok:
            failures.append("negative_or_crossing_conductor_thickness")
    else:
        failures.append("conductor_bottom_or_thickness_missing")

    return details, failures


def posterior_method_flags(
    summary: dict[str, Any],
    features: dict[str, Any],
    sample_diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify whether the posterior is genuinely sampled/optimized.

    The task permits MCMC, SMC, VI, or a meaningful hybrid. A pure MAP fit with
    Laplace/truncated-Gaussian samples can be useful, but it should not receive
    high C/D/E scores because it does not demonstrate robust Bayesian
    exploration of the pseudo-2D MT posterior.
    """
    method = str(summary.get("method", "")).lower()
    source = str(summary.get("sample_source", "")).lower()
    summary_text = json.dumps(summary, sort_keys=True).lower()
    runtime = _finite_float(summary.get("runtime_seconds"))
    ess = _finite_float(summary.get("ess_min"))
    rhat = _finite_float(summary.get("rhat_max"))
    accept = _finite_float(summary.get("acceptance_rate"))
    n_samples = _finite_float(summary.get("n_samples"))
    n_chains = _finite_float(summary.get("n_chains"))
    burn_in = _finite_float(summary.get("burn_in"))
    sample_diag = sample_diag or {}
    sample_n = _finite_float(sample_diag.get("n_samples"))
    sample_ess = _finite_float(sample_diag.get("ess_min_estimate"))
    sample_rhat = _finite_float(sample_diag.get("split_rhat_estimate"))
    sample_ess_frac = _finite_float(sample_diag.get("ess_min_fraction"))
    sample_lag1_abs = _finite_float(sample_diag.get("lag1_abs_median"))
    sample_good = bool(sample_diag.get("diagnostics_good"))
    sample_present = bool(sample_diag.get("diagnostics_present"))

    sampler_terms = [
        "adaptive_mcmc",
        "de-mcmc",
        "de_mcmc",
        "demcmc",
        "mcmc_plus",
        "mcmc",
        "metropolis",
        "ensemble",
        "smc",
        "particle",
        "variational",
        "vi",
        "hmc",
        "nuts",
    ]

    def has_sampler_term(text: str) -> bool:
        if not text:
            return False
        for term in sampler_terms:
            if len(term) <= 3:
                if re.search(rf"(^|[^a-z0-9]){re.escape(term)}([^a-z0-9]|$)", text):
                    return True
            elif term in text:
                return True
        return False

    check_markers = [
        "mcmc_check",
        "mcmc check",
        "additional posterior check",
        "posterior check",
        "diagnostic check",
    ]
    laplace_terms = [
        "laplace",
        "laplace_vi",
        "laplace/vi",
        "truncated gaussian",
        "truncated_gaussian",
        "gaussian approximation",
    ]
    optimizer_mode_terms = [
        "gauss_newton",
        "gauss-newton",
        "least_squares",
        "least squares",
        "optimizer_success",
        "optimizer_cost",
        "map_objective",
        "posterior mode",
        "centered at the posterior mode",
        "map estimate",
        "map_estimate",
    ]
    mcmc_check_marker = any(t in method or t in source or t in summary_text for t in check_markers)
    has_laplace_evidence = bool(
        features.get("has_laplace_terms")
        or any(t in source or t in method or t in summary_text for t in laplace_terms)
    )
    has_mode_or_optimizer_evidence = bool(
        features.get("has_optimizer_terms")
        or any(t in source or t in method or t in summary_text for t in optimizer_mode_terms)
        or "map_objective" in summary
        or "optimizer_success" in summary
        or "optimizer_cost" in summary
        or "covariance" in summary_text
        or "hessian" in summary_text
        or "jacobian" in summary_text
    )
    mcmc_check_only = bool(mcmc_check_marker and has_laplace_evidence)

    source_is_actual_sampler = bool(has_sampler_term(source) and not mcmc_check_only)
    method_claims_sampler = bool(has_sampler_term(method) and not mcmc_check_only)
    sampler_diagnostics_good = bool(
        ess is not None
        and rhat is not None
        and ess >= 100.0
        and 0.90 <= rhat <= 1.20
        and (accept is None or 0.01 <= accept <= 0.85)
    )
    sampler_diagnostics_present = bool(
        (ess is not None and ess >= 50.0)
        or (rhat is not None and 0.85 <= rhat <= 1.30)
        or (accept is not None and 0.01 <= accept <= 0.85)
    )
    summary_sampling_evidence = bool(
        sampler_diagnostics_good
        or (
            n_samples is not None
            and n_samples >= 1000.0
            and sampler_diagnostics_present
            and (runtime is None or runtime >= 30.0)
        )
    )
    sample_sampling_evidence = bool(
        sample_good
        or (
            sample_n is not None
            and sample_n >= 1000.0
            and sample_present
            and (runtime is None or runtime >= 30.0)
        )
    )
    credible_chain_metadata = bool(
        n_chains is not None
        and n_chains >= 2.0
        and burn_in is not None
        and burn_in >= 50.0
        and (runtime is None or runtime >= 180.0)
    )
    robust_sampling_evidence = bool(
        (source_is_actual_sampler or method_claims_sampler)
        and summary_sampling_evidence
        and sample_sampling_evidence
        and (not mcmc_check_only or credible_chain_metadata)
    )
    mcmc_like = bool(
        source_is_actual_sampler
        or method_claims_sampler
        or "metropolis" in summary_text
        or "random-walk" in summary_text
        or "random walk" in summary_text
    )
    chain_sampler_claim = bool(
        "mcmc" in method
        or "mcmc" in source
        or "metropolis" in summary_text
        or "random-walk" in summary_text
        or "random walk" in summary_text
    )
    iid_tolerant_sampler_claim = bool(
        "smc" in method
        or "smc" in source
        or "particle" in summary_text
        or "variational" in summary_text
        or re.search(r"(^|[^a-z0-9])vi([^a-z0-9]|$)", method)
        or re.search(r"(^|[^a-z0-9])vi([^a-z0-9]|$)", source)
        or "hmc" in summary_text
        or "nuts" in summary_text
    )
    low_acceptance = bool(accept is not None and 0.0 < accept <= 0.10)
    suspiciously_independent_samples = bool(
        sample_n is not None
        and sample_n >= 1000.0
        and sample_ess_frac is not None
        and sample_ess_frac >= 0.60
        and sample_lag1_abs is not None
        and sample_lag1_abs <= 0.08
        and (sample_rhat is None or 0.98 <= sample_rhat <= 1.02)
    )
    shuffled_chain_like_samples = bool(
        sample_n is not None
        and sample_n >= 1000.0
        and sample_ess_frac is not None
        and sample_ess_frac >= 0.50
        and sample_lag1_abs is not None
        and sample_lag1_abs <= 0.12
        and (sample_rhat is None or 0.97 <= sample_rhat <= 1.03)
    )
    sample_order_destroyed_risk = bool(
        mcmc_like
        and bool(features.get("has_sample_order_destroying_terms"))
        and (suspiciously_independent_samples or shuffled_chain_like_samples)
    )
    laplace_chain_independence_risk = bool(
        chain_sampler_claim
        and has_laplace_evidence
        and has_mode_or_optimizer_evidence
        and shuffled_chain_like_samples
    )
    mcmc_iid_cloud_risk = bool(
        chain_sampler_claim
        and not iid_tolerant_sampler_claim
        and sample_n is not None
        and sample_n >= 1000.0
        and sample_ess_frac is not None
        and sample_ess_frac >= 0.75
        and sample_lag1_abs is not None
        and sample_lag1_abs <= 0.05
        and (sample_rhat is None or 0.985 <= sample_rhat <= 1.015)
    )
    sample_order_destroyed_reason = None
    if sample_order_destroyed_risk:
        robust_sampling_evidence = False
        if low_acceptance:
            sample_order_destroyed_reason = "posterior_samples_shuffle_or_permutation_with_low_acceptance_independent_chain"
        else:
            sample_order_destroyed_reason = "posterior_samples_shuffle_or_permutation_with_independent_chain_diagnostics"
    elif laplace_chain_independence_risk:
        robust_sampling_evidence = False
        sample_order_destroyed_reason = "laplace_or_map_initialized_chain_has_independent_cloud_diagnostics"
    elif mcmc_iid_cloud_risk:
        robust_sampling_evidence = False
        sample_order_destroyed_reason = "chain_mcmc_claim_has_iid_cloud_diagnostics"

    if mcmc_check_only and (has_mode_or_optimizer_evidence or has_laplace_evidence):
        robust_sampling_evidence = False

    has_map_evidence = bool(
        has_mode_or_optimizer_evidence
        or ("map" in method and not method_claims_sampler)
        or ("map" in source and not source_is_actual_sampler)
    )

    source_is_laplace = has_laplace_evidence
    laplace_only = source_is_laplace and (not source_is_actual_sampler or mcmc_check_only)
    map_laplace_only = bool(laplace_only and has_map_evidence and not robust_sampling_evidence)
    map_without_sampler = bool(has_map_evidence and not has_laplace_evidence and not robust_sampling_evidence)
    fast_pipeline = bool(runtime is not None and runtime < 180.0)
    fast_map_or_laplace = bool(
        fast_pipeline
        and not robust_sampling_evidence
        and (map_laplace_only or (has_map_evidence and has_laplace_evidence))
    )
    hidden_or_public_says_laplace = bool(
        "laplace_truncated_gaussian" in summary_text
        or "laplace" in summary_text
        or "truncated gaussian" in summary_text
    )
    return {
        "method": method,
        "sample_source": source,
        "runtime_seconds": runtime,
        "n_samples": n_samples,
        "ess_min": ess,
        "rhat_max": rhat,
        "acceptance_rate": accept,
        "has_laplace_evidence": has_laplace_evidence,
        "has_map_evidence": has_map_evidence,
        "mcmc_check_only": mcmc_check_only,
        "source_is_laplace": source_is_laplace,
        "source_is_actual_sampler": source_is_actual_sampler,
        "method_claims_sampler": method_claims_sampler,
        "chain_sampler_claim": chain_sampler_claim,
        "iid_tolerant_sampler_claim": iid_tolerant_sampler_claim,
        "sampler_diagnostics_good": sampler_diagnostics_good,
        "sample_diagnostics": sample_diag,
        "sample_order_destroyed_risk": bool(sample_order_destroyed_risk or laplace_chain_independence_risk or mcmc_iid_cloud_risk),
        "sample_order_destroyed_reason": sample_order_destroyed_reason,
        "suspiciously_independent_samples": suspiciously_independent_samples,
        "shuffled_chain_like_samples": shuffled_chain_like_samples,
        "laplace_chain_independence_risk": laplace_chain_independence_risk,
        "mcmc_iid_cloud_risk": mcmc_iid_cloud_risk,
        "has_sample_order_destroying_terms": bool(features.get("has_sample_order_destroying_terms")),
        "summary_sampling_evidence": summary_sampling_evidence,
        "sample_sampling_evidence": sample_sampling_evidence,
        "robust_sampling_evidence": robust_sampling_evidence,
        "map_without_sampler": map_without_sampler,
        "map_laplace_only": map_laplace_only,
        "fast_pipeline_no_sampling": fast_map_or_laplace,
        "hidden_or_public_says_laplace": hidden_or_public_says_laplace,
    }


def _summary_lower(summary: dict[str, Any]) -> str:
    try:
        return json.dumps(summary, sort_keys=True, default=str).lower()
    except Exception:
        return str(summary).lower()


def summary_with_observed_runtime(summary: dict[str, Any], observed_seconds: Any) -> dict[str, Any]:
    observed = _finite_float(observed_seconds)
    if observed is None:
        return summary
    out = dict(summary)
    declared = _finite_float(out.get("runtime_seconds"))
    out["runtime_seconds"] = observed if declared is None else min(declared, observed)
    out["runtime_seconds_declared"] = declared
    out["runtime_seconds_observed_by_evaluator"] = observed
    return out


def hidden_public_inference_consistency(
    public_summary: dict[str, Any],
    hidden: dict[str, Any],
    features: dict[str, Any],
    public_flags: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Detect submissions that switch inference/model branches on hidden data.

    Hidden cases are meant to rerun the same solver on new profiles. A solution
    may adapt proposal scales or restart from the new data, but it should not
    report public MCMC while hidden runs silently switch to a canonical
    closed-form, Laplace/VI, or one-chain no-burn-in path.
    """
    public_flags = public_flags or posterior_method_flags(public_summary, features)
    public_text = _summary_lower(public_summary)
    public_method = str(public_summary.get("method", "")).lower().strip()
    public_chains = _finite_float(public_summary.get("n_chains"))
    public_burn = _finite_float(public_summary.get("burn_in"))
    public_accept = _finite_float(public_summary.get("acceptance_rate"))
    public_sampler_like = bool(
        public_flags.get("source_is_actual_sampler")
        or public_flags.get("method_claims_sampler")
        or public_flags.get("robust_sampling_evidence")
        or (public_chains is not None and public_chains >= 2.0)
    )

    canonical_terms = [
        "canonical_marine_water",
        "water_response_mix",
        "canonical_laplace",
        "canonical_forward",
        "selected_model",
    ]
    laplace_vi_terms = [
        "laplace_vi",
        "laplace/vi",
        "laplace",
        "variational around",
        "posterior mode",
        "truncated gaussian",
    ]
    static_branch_risk = bool(features.get("hidden_specific_branch_risk"))
    hidden_env_marker_risk = bool(features.get("has_hidden_specific_terms"))

    details: dict[str, Any] = {
        "public_method": public_method,
        "public_n_chains": public_chains,
        "public_burn_in": public_burn,
        "public_acceptance_rate": public_accept,
        "public_sampler_like": public_sampler_like,
        "source_hidden_specific_branch_risk": static_branch_risk,
        "source_hidden_env_marker_risk": hidden_env_marker_risk,
        "case_checks": [],
    }
    failures: list[str] = []
    mismatch_cases = 0
    hidden_branch_cases = 0
    laplace_vi_cases = 0

    for res in hidden.get("cases", []):
        if not res.get("success"):
            continue
        hidden_summary = res.get("summary") if isinstance(res.get("summary"), dict) else {}
        hidden_summary = summary_with_observed_runtime(hidden_summary, res.get("elapsed_seconds"))
        hidden_diag = res.get("sample_diagnostics") if isinstance(res.get("sample_diagnostics"), dict) else {}
        hidden_flags = posterior_method_flags(hidden_summary, features, hidden_diag)
        hidden_text = _summary_lower(hidden_summary)
        hidden_method = str(hidden_summary.get("method", "")).lower().strip()
        hidden_chains = _finite_float(hidden_summary.get("n_chains"))
        hidden_burn = _finite_float(hidden_summary.get("burn_in"))
        hidden_accept = _finite_float(hidden_summary.get("acceptance_rate"))

        hidden_model_branch = any(term in hidden_text for term in canonical_terms)
        hidden_laplace_vi = any(term in hidden_text for term in laplace_vi_terms)
        method_changed = bool(public_method and hidden_method and public_method != hidden_method)
        reasons: list[str] = []

        if method_changed and (public_sampler_like or hidden_laplace_vi or hidden_model_branch or static_branch_risk):
            reasons.append("method_changed_between_public_and_hidden")
        if hidden_model_branch and not any(term in public_text for term in canonical_terms):
            reasons.append("hidden_selected_canonical_model_branch")
        if hidden_laplace_vi and "laplace" not in public_text:
            reasons.append("hidden_laplace_or_vi_branch")
        if (
            public_chains is not None
            and public_chains >= 2.0
            and hidden_chains is not None
            and hidden_chains <= 1.0
        ):
            reasons.append("hidden_chain_count_collapsed")
        if (
            public_burn is not None
            and public_burn >= 50.0
            and hidden_burn is not None
            and hidden_burn <= 1.0
        ):
            reasons.append("hidden_burn_in_dropped")
        if (
            public_accept is not None
            and 0.01 <= public_accept <= 0.85
            and hidden_accept is not None
            and hidden_accept >= 0.95
        ):
            reasons.append("hidden_unit_acceptance_rate")

        if hidden_laplace_vi and (
            hidden_flags.get("map_laplace_only")
            or hidden_chains is None
            or hidden_chains <= 1.0
            or hidden_burn is None
            or hidden_burn <= 1.0
        ):
            laplace_vi_cases += 1
        if hidden_env_marker_risk and (
            method_changed
            or hidden_laplace_vi
            or hidden_model_branch
            or hidden_flags.get("map_laplace_only")
            or hidden_flags.get("fast_pipeline_no_sampling")
            or hidden_flags.get("sample_order_destroyed_risk")
        ):
            reasons.append("hidden_env_marker_used_with_inference_change")
        if len(reasons) >= 2 or (method_changed and (hidden_laplace_vi or hidden_model_branch)):
            mismatch_cases += 1
        if (static_branch_risk or hidden_env_marker_risk) and (hidden_model_branch or hidden_laplace_vi or len(reasons) >= 2):
            hidden_branch_cases += 1

        details["case_checks"].append({
            "case_id": res.get("case_id"),
            "hidden_method": hidden_method,
            "hidden_n_chains": hidden_chains,
            "hidden_burn_in": hidden_burn,
            "hidden_acceptance_rate": hidden_accept,
            "hidden_model_branch": hidden_model_branch,
            "hidden_laplace_vi": hidden_laplace_vi,
            "reasons": reasons,
        })

    details["mismatch_cases"] = mismatch_cases
    details["hidden_branch_cases"] = hidden_branch_cases
    details["laplace_vi_cases"] = laplace_vi_cases

    if mismatch_cases > 0:
        failures.append("hidden_inference_chain_mismatch")
    if hidden_branch_cases > 0:
        failures.append("hidden_specific_model_branch")
    if laplace_vi_cases > 0 and mismatch_cases > 0:
        failures.append("hidden_laplace_vi_not_same_sampler")
    return details, failures


def score_C(out_dir: Path, summary: dict[str, Any], features: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
    details: dict[str, Any] = {"static": features}
    failures: list[str] = []
    score = 0.0
    samples, sample_err = load_npy(out_dir / "posterior_samples.npy")
    sample_diag = posterior_sample_diagnostics(samples)
    method_flags = posterior_method_flags(summary, features, sample_diag)
    details["posterior_method_flags"] = method_flags

    # C1 probability model completeness, 7.
    c1 = 0.0
    if features["has_bayesian_terms"]:
        c1 += 2.0
    if "prior" in str(summary.get("prior_description", "")).lower() or features["has_bayesian_terms"]:
        c1 += 1.5
    if "likelihood" in str(summary).lower() or features["has_bayesian_terms"]:
        c1 += 1.5
    if features["has_water_depth_terms"] and features["has_log10_terms"]:
        c1 += 1.0
    if features["has_lateral_basis_terms"]:
        c1 += 1.0
    details["C1_probability_model"] = _item(c1, 7.0)
    score += min(c1, 7.0)

    # C2 laterally shared reduced parameterization, 7.
    prof_details, prof_failures = profile_quality(out_dir)
    details["profile_quality"] = prof_details
    c2 = 0.0
    if features["has_lateral_basis_terms"]:
        c2 += 3.0
    if prof_details.get("n_stations", 0) and prof_details.get("n_stations", 0) >= 10:
        c2 += 1.0
    if prof_details.get("profile_lengths_consistent"):
        c2 += 1.0
    if prof_details.get("lab_second_diff_rms_km") is not None:
        rough = float(prof_details.get("lab_second_diff_rms_km") or 0.0)
        if rough < 20.0:
            c2 += 1.0
    if not features["possibly_per_station_independent"]:
        c2 += 1.0
    else:
        failures.append("per_station_independent_inversion_suspected")
    details["C2_lateral_parameterization"] = _item(c2, 7.0)
    score += min(c2, 7.0)

    # C3 posterior/VI authenticity, 8.
    c3 = 0.0
    sample_details: dict[str, Any] = {"error": sample_err}
    if sample_err is None and samples is not None and samples.ndim == 2 and np.issubdtype(samples.dtype, np.number):
        n_samples, n_params = samples.shape
        sample_details.update(shape=[int(n_samples), int(n_params)], finite_fraction=finite_fraction(samples), diagnostics=sample_diag)
        if n_samples >= 1000:
            c3 += 2.0
        elif n_samples >= 200:
            c3 += 1.0
        if 5 <= n_params <= 80:
            c3 += 1.0
        std = np.nanstd(samples.astype(float), axis=0)
        active = int(np.sum(std > 1e-8))
        sample_details["active_parameter_count"] = active
        if active >= max(3, min(8, n_params // 2)):
            c3 += 2.0
        elif active > 0:
            c3 += 0.5
        if finite_fraction(samples) >= 0.999:
            c3 += 1.0
        if features["has_mcmc_or_vi"] and not method_flags["map_laplace_only"]:
            c3 += 1.0
        elif features["has_mcmc_or_vi"]:
            c3 += 0.25
        if (
            method_flags["sampler_diagnostics_good"]
            and method_flags["sample_sampling_evidence"]
            and not method_flags["map_laplace_only"]
            and not method_flags.get("sample_order_destroyed_risk")
        ):
            c3 += 1.0
        elif _finite_float(summary.get("acceptance_rate")) is not None or _finite_float(summary.get("ess_min")) is not None or _finite_float(summary.get("rhat_max")) is not None:
            c3 += 0.25
        if active == 0 or n_samples < 50:
            failures.append("posterior_samples_repeated_or_too_short")
    else:
        failures.append("posterior_samples_missing_or_invalid")
    if method_flags["map_laplace_only"]:
        failures.append("map_laplace_only")
        c3 = min(c3, 3.0)
    if method_flags["fast_pipeline_no_sampling"]:
        failures.append("fast_pipeline_no_sampling")
    if method_flags.get("sample_order_destroyed_risk"):
        failures.append("posterior_chain_order_untrustworthy")
    details["posterior_samples"] = sample_details
    details["C3_posterior_authenticity"] = _item(c3, 8.0)
    score += min(c3, 8.0)

    # C4 reduced-order/surrogate/adaptive, 6.
    c4 = 0.0
    if features["has_lateral_basis_terms"]:
        c4 += 2.0
    if features["has_adaptive_or_surrogate"]:
        c4 += 3.0
    if "reduced" in str(summary).lower() or "surrogate" in str(summary).lower() or "adaptive" in str(summary).lower():
        c4 += 1.0
    details["C4_reduced_order"] = _item(c4, 6.0)
    score += min(c4, 6.0)

    # C5 uncertainty calibration evidence, 5.
    c5 = 0.0
    if prof_details.get("credible_intervals_ordered"):
        c5 += 1.5
    width = prof_details.get("median_ci_width_km")
    if isinstance(width, (int, float)) and 0.2 <= float(width) <= 80.0:
        c5 += 1.5
    if features["has_uncertainty_terms"]:
        c5 += 1.0
    if method_flags["summary_sampling_evidence"] and method_flags["sample_sampling_evidence"] and not method_flags.get("sample_order_destroyed_risk"):
        c5 += 1.0
    elif _finite_float(summary.get("ess_min")) is not None or _finite_float(summary.get("rhat_max")) is not None:
        c5 += 0.25
    details["C5_uncertainty"] = _item(c5, 5.0)
    score += min(c5, 5.0)

    # C6 stability, 2.
    c6 = 0.0
    if not features["parse_errors"]:
        c6 += 1.0
    if not any(f in prof_failures for f in ["profile_missing_or_unreadable", "credible_interval_order_invalid"]):
        c6 += 1.0
    details["C6_stability"] = _item(c6, 2.0)
    score += c6

    if features["map_only_terms"] or method_flags["map_without_sampler"]:
        failures.append("map_only_or_least_squares")
    if features["hardcodes_public_station_names"]:
        failures.append("public_station_names_hardcoded")
    failures.extend(f for f in prof_failures if f not in failures)

    # Hard caps.
    cap = COMPONENT_MAX["C"]
    if "map_only_or_least_squares" in failures:
        cap = min(cap, 6.0)
    if "per_station_independent_inversion_suspected" in failures:
        cap = min(cap, 8.0)
    if "credible_interval_too_narrow_or_fixed" in failures:
        cap = min(cap, 10.0)
    if "posterior_samples_repeated_or_too_short" in failures:
        cap = min(cap, 10.0)
    if not features["has_lateral_basis_terms"]:
        cap = min(cap, 12.0)
    if "map_laplace_only" in failures:
        cap = min(cap, 12.0)
    if "fast_pipeline_no_sampling" in failures:
        cap = min(cap, 10.0)
    if "posterior_chain_order_untrustworthy" in failures:
        cap = min(cap, 12.0)

    details["cap"] = cap
    return float(np.clip(score, 0.0, cap)), details, failures


def score_public_prediction_fit(out_dir: Path, summary: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    pred, pred_err = load_npz(out_dir / "data_pred_mean.npz")
    details: dict[str, Any] = {"prediction_error": pred_err}
    score = 0.0
    if pred_err is None:
        keymap = prediction_keymap(pred)
        details["keymap"] = keymap
        public_profile, public_path = load_public_profile()
        trusted_fit = trusted_prediction_metrics(pred, public_profile)
        details["trusted_public_fit"] = trusted_fit
        details["trusted_public_profile_path"] = public_path
        if keymap["log_obs"] and keymap["log_pred"]:
            obs = np.asarray(pred[keymap["log_obs"]], dtype=float)
            pp = np.asarray(pred[keymap["log_pred"]], dtype=float)
            if obs.shape == pp.shape:
                val = _finite_float(trusted_fit.get("rmse_log10_rhoa"))
                details["rmse_log10_rhoa_computed"] = val
                if val is not None:
                    score += 2.0 * float(np.clip((0.30 - val) / 0.25, 0.0, 1.0))
        if keymap["phase_obs"] and keymap["phase_pred"]:
            obs = np.asarray(pred[keymap["phase_obs"]], dtype=float)
            pp = np.asarray(pred[keymap["phase_pred"]], dtype=float)
            if obs.shape == pp.shape:
                val = _finite_float(trusted_fit.get("rmse_phase_deg"))
                details["rmse_phase_deg_computed"] = val
                if val is not None:
                    score += 2.0 * float(np.clip((15.0 - val) / 12.0, 0.0, 1.0))
    trusted = details.get("trusted_public_fit") if isinstance(details.get("trusted_public_fit"), dict) else {}
    r = _finite_float(trusted.get("rmse_log10_rhoa"))
    p = _finite_float(trusted.get("rmse_phase_deg"))
    c = _finite_float(trusted.get("chi2_per_dof"))
    if r is not None:
        score += 1.0 * float(np.clip((0.18 - r) / 0.14, 0.0, 1.0))
    if p is not None:
        score += 1.0 * float(np.clip((10.0 - p) / 7.0, 0.0, 1.0))
    if c is not None:
        score += 1.0 if 0.7 <= c <= 1.8 else 0.5 * float(np.clip(1.0 - abs(math.log10(max(c, 1e-12))), 0.0, 1.0))
    return float(np.clip(score, 0.0, 7.0)), details


def run_hidden_case(out_dir: Path, case: dict[str, np.ndarray], timeout_s: int = 75) -> dict[str, Any]:
    run_py = out_dir / "run_inversion.py"
    if not run_py.exists():
        return {"success": False, "error": "run_inversion.py missing"}

    with tempfile.TemporaryDirectory(prefix="mtlab_hidden_") as td:
        tmp = Path(td)
        work = tmp / "work"
        work.mkdir()
        generated_names = {
            "posterior_samples.npy",
            "model_posterior_profile.npz",
            "data_pred_mean.npz",
            "summary.json",
            "report.md",
            "score.json",
            "eval_result.json",
        }
        copied_entries: list[str] = []
        for entry in out_dir.iterdir():
            if entry.name in generated_names or entry.name == "__pycache__":
                continue
            target = work / entry.name
            try:
                if entry.is_dir():
                    shutil.copytree(
                        entry,
                        target,
                        ignore=shutil.ignore_patterns(
                            "__pycache__",
                            "*.pyc",
                            ".ipynb_checkpoints",
                            "posterior_samples.npy",
                            "model_posterior_profile.npz",
                            "data_pred_mean.npz",
                            "summary.json",
                            "report.md",
                            "*.png",
                            "*.pdf",
                            "*.svg",
                        ),
                    )
                else:
                    shutil.copy2(entry, target)
                copied_entries.append(entry.name)
            except Exception:
                continue
        hidden_input = tmp / "hidden_profile.npz"
        hidden_out = tmp / "hidden_outputs"
        hidden_out.mkdir()
        np.savez(hidden_input, **hidden_case_input(case))

        env = os.environ.copy()
        env.update({
            "INPUT_PROFILE_NPZ": str(hidden_input),
            "PROFILE_DATA_PATH": str(hidden_input),
            "OUTPUT_DIR": str(hidden_out),
            "PSEUDO2D_HIDDEN": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        commands = [
            [sys.executable, str(work / "run_inversion.py"), "--input", str(hidden_input), "--output", str(hidden_out)],
            [sys.executable, str(work / "run_inversion.py")],
        ]
        attempts: list[dict[str, Any]] = []
        proc: subprocess.CompletedProcess[str] | None = None
        output_dir: Path | None = None
        candidates = [
            hidden_out,
            work / "outputs",
            work,
        ]
        try:
            for cmd in commands:
                started = time.monotonic()
                proc = subprocess.run(
                    cmd,
                    cwd=str(work),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_s,
                )
                elapsed = time.monotonic() - started
                candidate_output = next((p for p in candidates if (p / "model_posterior_profile.npz").exists()), None)
                attempts.append({
                    "cmd": cmd,
                    "returncode": proc.returncode,
                    "elapsed_seconds": elapsed,
                    "stdout_tail": proc.stdout[-500:],
                    "stderr_tail": proc.stderr[-500:],
                    "output_dir_detected": str(candidate_output) if candidate_output is not None else None,
                })
                if proc.returncode == 0 and candidate_output is not None:
                    output_dir = candidate_output
                    break
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"timeout after {timeout_s}s"}
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        if proc is None:
            return {"success": False, "error": "run_inversion.py was not executed"}

        if output_dir is None:
            output_dir = next((p for p in candidates if (p / "model_posterior_profile.npz").exists()), hidden_out)
        profile, perr = load_npz(output_dir / "model_posterior_profile.npz")
        pred, derr = load_npz(output_dir / "data_pred_mean.npz")
        summary, serr = load_json(output_dir / "summary.json")
        hidden_samples, sample_err = load_npy(output_dir / "posterior_samples.npy")
        hidden_sample_diag = posterior_sample_diagnostics(hidden_samples)

        result: dict[str, Any] = {
            "success": proc.returncode == 0 and perr is None,
            "returncode": proc.returncode,
            "elapsed_seconds": sum(float(a.get("elapsed_seconds") or 0.0) for a in attempts),
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "attempts": attempts,
            "copied_entries": copied_entries,
            "output_dir_detected": str(output_dir),
            "profile_error": perr,
            "prediction_error": derr,
            "summary_error": serr,
            "sample_error": sample_err,
            "sample_diagnostics": hidden_sample_diag,
        }
        if perr is not None:
            result["success"] = False
            return result

        z_true = np.asarray(case["z_lab_true_km"], dtype=float)
        x_true = np.asarray(case["x_m"], dtype=float)
        z_pred = as_numeric_1d(profile, "z_lab_p50_km")
        if z_pred is None:
            z_pred = as_numeric_1d(profile, "z_lab_mean_km")
        if z_pred is not None:
            if z_pred.size != z_true.size:
                px = as_numeric_1d(profile, "x_m")
                if px is not None and px.size == z_pred.size and z_pred.size >= 2:
                    z_pred_eval = np.interp(x_true, px, z_pred)
                else:
                    z_pred_eval = np.full_like(z_true, np.nan)
            else:
                z_pred_eval = z_pred
            result["z_lab_rmse_km"] = rmse(z_true, z_pred_eval)
            result["z_lab_mae_km"] = float(np.nanmean(np.abs(z_true - z_pred_eval)))

        z05 = as_numeric_1d(profile, "z_lab_p05_km")
        z95 = as_numeric_1d(profile, "z_lab_p95_km")
        px = as_numeric_1d(profile, "x_m")
        if z05 is not None and z95 is not None:
            if z05.size != z_true.size and px is not None and px.size == z05.size and z05.size >= 2:
                z05e = np.interp(x_true, px, z05)
                z95e = np.interp(x_true, px, z95)
            else:
                z05e, z95e = z05, z95
            if z05e.size == z_true.size and z95e.size == z_true.size:
                result["coverage_90"] = ci_coverage(z_true, z05e, z95e)
                result["median_ci_width_km"] = _finite_float(np.nanmedian(z95e - z05e))

        result["summary"] = summary
        if derr is None:
            trusted_fit = trusted_prediction_metrics(pred, hidden_case_input(case))
            result["trusted_prediction_fit"] = trusted_fit
            if _finite_float(trusted_fit.get("rmse_log10_rhoa")) is not None:
                result["rmse_log10_rhoa"] = float(trusted_fit["rmse_log10_rhoa"])
            if _finite_float(trusted_fit.get("rmse_phase_deg")) is not None:
                result["rmse_phase_deg"] = float(trusted_fit["rmse_phase_deg"])
        return result


def score_hidden(out_dir: Path) -> dict[str, Any]:
    cases = [
        make_hidden_case("smooth_deepening", 12, 10, 1201),
        make_hidden_case("local_upwarp", 24, 12, 1202),
        make_hidden_case("thin_conductor", 16, 8, 1203),
        make_hidden_case("rugged_transition", 18, 11, 4317),
        make_hidden_case("offset_frequencies", 21, 9, 8129),
    ]
    results = []
    for case in cases:
        case_id = str(np.asarray(case["case_id"]).item())
        res = run_hidden_case(out_dir, case)
        res["case_id"] = case_id
        results.append(res)

    successes = [r for r in results if r.get("success")]
    metrics: dict[str, Any] = {
        "cases": results,
        "n_cases": len(cases),
        "n_success": len(successes),
    }
    return metrics


def score_D(out_dir: Path, summary: dict[str, Any], hidden: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
    details: dict[str, Any] = {}
    failures: list[str] = []
    score = 0.0
    features = static_features(out_dir)
    public_samples, _public_sample_err = load_npy(out_dir / "posterior_samples.npy")
    public_sample_diag = posterior_sample_diagnostics(public_samples)
    method_flags = posterior_method_flags(summary, features, public_sample_diag)
    details["posterior_method_flags"] = method_flags

    public_fit_score, public_fit_details = score_public_prediction_fit(out_dir, summary)
    details["public_fit"] = public_fit_details
    score += min(public_fit_score, 4.0)

    prof_details, prof_failures = profile_quality(out_dir)
    details["public_profile"] = prof_details
    if prof_details.get("lab_below_seafloor"):
        score += 1.0
    if prof_details.get("lab_plausible_range"):
        score += 1.0
    if prof_details.get("bottom_deeper_than_lab") or prof_details.get("positive_conductor_thickness"):
        score += 1.0

    successes = hidden.get("n_success", 0)
    details["hidden"] = hidden
    if successes == 0:
        failures.append("hidden_cases_not_run_or_failed")
        # Public outputs can still earn limited D points, but hidden truth is central.
        return float(np.clip(score, 0.0, 8.0)), details, failures

    rmses = []
    coverages = []
    widths = []
    fit_good = 0
    hidden_summary_flags = []
    for res in hidden["cases"]:
        if not res.get("success"):
            continue
        hidden_summary = res.get("summary") if isinstance(res.get("summary"), dict) else {}
        hidden_summary = summary_with_observed_runtime(hidden_summary, res.get("elapsed_seconds"))
        hidden_sample_diag = res.get("sample_diagnostics") if isinstance(res.get("sample_diagnostics"), dict) else {}
        hidden_flags = posterior_method_flags(hidden_summary, features, hidden_sample_diag)
        hidden_summary_flags.append(hidden_flags)
        if _finite_float(res.get("z_lab_rmse_km")) is not None:
            rmses.append(float(res["z_lab_rmse_km"]))
        if _finite_float(res.get("coverage_90")) is not None:
            coverages.append(float(res["coverage_90"]))
        if _finite_float(res.get("median_ci_width_km")) is not None:
            widths.append(float(res["median_ci_width_km"]))
        if _finite_float(res.get("rmse_log10_rhoa")) is not None and float(res["rmse_log10_rhoa"]) < 0.12:
            fit_good += 1
    details["hidden_posterior_method_flags"] = hidden_summary_flags

    consistency_details, consistency_failures = hidden_public_inference_consistency(
        summary,
        hidden,
        features,
        method_flags,
    )
    details["hidden_public_inference_consistency"] = consistency_details
    failures.extend(consistency_failures)

    if rmses:
        med_rmse = float(np.median(rmses))
        details["median_hidden_z_lab_rmse_km"] = med_rmse
        score += 10.0 * float(np.clip((12.0 - med_rmse) / 10.0, 0.0, 1.0))
    if widths:
        med_width = float(np.median(widths))
        details["median_hidden_ci_width_km"] = med_width
        if 1.0 <= med_width <= 55.0:
            score += 3.0
    if coverages:
        cov = float(np.median(coverages))
        details["median_hidden_coverage_90"] = cov
        score += 4.0 * float(np.clip(1.0 - abs(cov - 0.90) / 0.35, 0.0, 1.0))
    score += 5.0 * (fit_good / max(1, successes))
    score += min(2.0, 2.0 * successes / len(hidden["cases"]))

    if method_flags["map_laplace_only"] or any(flag.get("map_laplace_only") for flag in hidden_summary_flags):
        failures.append("map_laplace_only")
    if method_flags["fast_pipeline_no_sampling"] or any(flag.get("fast_pipeline_no_sampling") for flag in hidden_summary_flags):
        failures.append("fast_pipeline_no_sampling")

    failures.extend(f for f in prof_failures if f in [
        "lab_not_below_seafloor",
        "negative_or_crossing_conductor_thickness",
        "credible_interval_order_invalid",
    ])
    cap = COMPONENT_MAX["D"]
    if "map_laplace_only" in failures:
        cap = min(cap, 12.0)
    if "fast_pipeline_no_sampling" in failures:
        cap = min(cap, 10.0)
    if "hidden_inference_chain_mismatch" in failures:
        cap = min(cap, 8.0)
    if "hidden_specific_model_branch" in failures:
        cap = min(cap, 6.0)
    return float(np.clip(score, 0.0, cap)), details, failures


def score_E(features: dict[str, Any], hidden: dict[str, Any], c_failures: list[str], b_failures: list[str]) -> tuple[float, dict[str, Any], list[str]]:
    score = 0.0
    failures: list[str] = []
    details: dict[str, Any] = {"static": features, "hidden_summary": {"n_success": hidden.get("n_success", 0), "n_cases": hidden.get("n_cases", 0)}}

    n_success = int(hidden.get("n_success", 0))
    n_cases = int(hidden.get("n_cases", 0))
    if n_cases:
        score += 4.0 * n_success / n_cases
    if n_success >= 2:
        score += 1.5
    if not features["hardcodes_public_station_names"]:
        score += 1.5
    else:
        failures.append("public_station_names_hardcoded")
    if not features["possibly_per_station_independent"]:
        score += 1.0
    if "water_depth_not_used" not in b_failures:
        score += 1.0
    if not any(f in c_failures for f in ["posterior_samples_missing_or_invalid", "posterior_samples_repeated_or_too_short"]):
        score += 1.0

    hidden_method_flags = []
    for res in hidden.get("cases", []):
        hidden_summary = res.get("summary") if isinstance(res.get("summary"), dict) else {}
        hidden_summary = summary_with_observed_runtime(hidden_summary, res.get("elapsed_seconds"))
        hidden_sample_diag = res.get("sample_diagnostics") if isinstance(res.get("sample_diagnostics"), dict) else {}
        hidden_method_flags.append(posterior_method_flags(hidden_summary, features, hidden_sample_diag))
    details["hidden_posterior_method_flags"] = hidden_method_flags
    if "map_laplace_only" in c_failures or any(flag.get("map_laplace_only") for flag in hidden_method_flags):
        failures.append("map_laplace_only")
        score = min(score, 4.0)
    if "fast_pipeline_no_sampling" in c_failures or any(flag.get("fast_pipeline_no_sampling") for flag in hidden_method_flags):
        failures.append("fast_pipeline_no_sampling")
        score = min(score, 3.0)

    if n_success == 0:
        failures.append("no_hidden_generalization")
    return float(np.clip(score, 0.0, COMPONENT_MAX["E"])), details, failures


def score_F(out_dir: Path) -> tuple[float, dict[str, Any], list[str]]:
    path = out_dir / "report.md"
    failures: list[str] = []
    if not path.exists():
        return 0.0, {"error": "report.md missing"}, ["report_missing"]
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    words = re.findall(r"[\w\u4e00-\u9fff]+", lower)
    word_count = len(words)
    covered = {cat: any(term in lower for term in terms) for cat, terms in REPORT_TERMS.items()}
    score = 0.0
    if word_count >= 900:
        score += 1.2
    elif word_count >= 450:
        score += 0.8
    elif word_count >= 200:
        score += 0.4
    score += 0.75 * sum(1 for ok in covered.values() if ok)
    if "full 2d" in lower or "完整二维" in lower:
        if "pseudo" not in lower and "拟二维" not in lower:
            failures.append("claims_full_2d_without_pseudo2d_context")
    if not covered["limitations"]:
        failures.append("report_missing_limitations")
    if not covered["results"]:
        failures.append("report_missing_quantitative_results")
    if word_count < 300:
        failures.append("report_too_short_for_research_task")

    cap = COMPONENT_MAX["F"]
    if "report_too_short_for_research_task" in failures:
        cap = min(cap, 3.0)
    elif word_count < 600:
        cap = min(cap, 5.0)
    if "claims_full_2d_without_pseudo2d_context" in failures:
        cap = min(cap, 3.0)
    if "report_missing_limitations" in failures:
        cap = min(cap, 4.0)
    if "report_missing_quantitative_results" in failures:
        cap = min(cap, 5.0)
    details = {"word_count": word_count, "covered": covered, "cap": cap}
    return float(np.clip(score, 0.0, cap)), details, failures


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def assess_geophysical_maturity(summary: dict[str, Any], hidden: dict[str, Any], out_dir: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """Cross-component checks for hidden LAB recovery and posterior credibility.

    C and E intentionally reward Bayesian structure and hidden runability, but a
    submission should not receive high benchmark scores merely because hidden
    cases finish. For a pseudo-2D MT LAB task, the hidden LAB geometry and the
    posterior interval coverage are the central geophysical evidence.
    """
    details: dict[str, Any] = {}
    failures: list[str] = []

    public_fit: dict[str, Any] = {}
    if out_dir is not None:
        pred, pred_err = load_npz(out_dir / "data_pred_mean.npz")
        public_profile, public_path = load_public_profile()
        details["public_prediction_error"] = pred_err
        details["trusted_public_profile_path"] = public_path
        if pred_err is None:
            public_fit = trusted_prediction_metrics(pred, public_profile)
    details["trusted_public_fit"] = public_fit

    chi2 = _finite_float(public_fit.get("chi2_per_dof"))
    log_rmse = _finite_float(public_fit.get("rmse_log10_rhoa"))
    phase_rmse = _finite_float(public_fit.get("rmse_phase_deg"))
    details["public_chi2_per_dof_computed"] = chi2
    details["public_rmse_log10_rhoa_computed"] = log_rmse
    details["public_rmse_phase_deg_computed"] = phase_rmse
    public_rmse_poor = bool(
        (log_rmse is not None and log_rmse > 0.30)
        or (phase_rmse is not None and phase_rmse > 15.0)
    )
    chi2_clearly_bad = bool(chi2 is not None and chi2 > 6.0)
    chi2_moderately_bad_with_visible_misfit = bool(
        chi2 is not None
        and chi2 > 3.0
        and (
            (log_rmse is not None and log_rmse > 0.18)
            or (phase_rmse is not None and phase_rmse > 10.0)
        )
    )
    details["public_rmse_poor"] = public_rmse_poor
    details["public_chi2_clearly_bad"] = chi2_clearly_bad
    details["public_chi2_moderately_bad_with_visible_misfit"] = chi2_moderately_bad_with_visible_misfit
    if public_rmse_poor or chi2_clearly_bad or chi2_moderately_bad_with_visible_misfit:
        failures.append("public_fit_not_noise_level")

    summary_text = json.dumps(summary, sort_keys=True).lower()
    if (
        "prediction_posthoc_bias_correction" in summary
        or "posthoc" in summary_text
        or "post-hoc" in summary_text
        or "bias correction" in summary_text
        or "residual correction" in summary_text
    ):
        failures.append("posthoc_prediction_correction")
    if (
        "discrepancy" in summary_text
        or "conditioned analytically" in summary_text
        or "prediction correction" in summary_text
    ):
        failures.append("posthoc_prediction_correction")

    physical_log = _finite_float(summary.get("physical_only_rmse_log10_rhoa"))
    physical_phase = _finite_float(summary.get("physical_only_rmse_phase_deg"))
    details["physical_only_rmse_log10_rhoa"] = physical_log
    details["physical_only_rmse_phase_deg"] = physical_phase
    if (
        bool(summary.get("uses_discrepancy"))
        and physical_log is not None
        and physical_phase is not None
        and (physical_log > 0.25 or physical_phase > 8.0)
    ):
        failures.append("physical_model_hidden_by_discrepancy")

    successes = [res for res in hidden.get("cases", []) if res.get("success")]
    details["hidden_success_count"] = len(successes)
    details["hidden_case_count"] = int(hidden.get("n_cases", 0))
    if not successes:
        return details, failures

    rmses = [
        float(res["z_lab_rmse_km"])
        for res in successes
        if _finite_float(res.get("z_lab_rmse_km")) is not None
    ]
    coverages = [
        float(res["coverage_90"])
        for res in successes
        if _finite_float(res.get("coverage_90")) is not None
    ]
    widths = [
        float(res["median_ci_width_km"])
        for res in successes
        if _finite_float(res.get("median_ci_width_km")) is not None
    ]
    details["hidden_z_lab_rmse_km"] = rmses
    details["hidden_coverage_90"] = coverages
    details["hidden_ci_width_km"] = widths

    if rmses:
        med_rmse = float(np.median(rmses))
        details["median_hidden_z_lab_rmse_km"] = med_rmse
        if med_rmse > 12.0:
            failures.append("weak_hidden_lab_recovery")
    if coverages:
        med_cov = float(np.median(coverages))
        details["median_hidden_coverage_90"] = med_cov
        if med_cov < 0.45:
            failures.append("posterior_undercoverage_hidden")
    if widths:
        med_width = float(np.median(widths))
        details["median_hidden_ci_width_km"] = med_width
        if med_width < 1.0:
            failures.append("credible_interval_too_narrow_or_fixed")
        elif med_width > 35.0:
            failures.append("credible_interval_too_wide_uninformative")

    if coverages and widths:
        med_cov = float(np.median(coverages))
        med_width = float(np.median(widths))
        if med_cov >= 0.99 and med_width > 18.0:
            failures.append("credible_interval_too_wide_uninformative")

    return details, failures

def determine_penalty(components: dict[str, float], failures: list[str]) -> tuple[float, dict[str, Any]]:
    structural_failure_triggers = {
        "map_only_or_least_squares",
        "per_station_independent_inversion_suspected",
        "posterior_samples_repeated_or_too_short",
        "posterior_samples_missing_or_invalid",
        "map_laplace_only",
        "fast_pipeline_no_sampling",
        "no_real_1d_mt_forward_evidence",
        "water_depth_not_used",
        "public_station_names_hardcoded",
        "posthoc_prediction_correction",
        "physical_model_hidden_by_discrepancy",
        "public_fit_and_hidden_quality_failed",
        "overconfident_weak_hidden_recovery",
        "hidden_inference_chain_mismatch",
        "hidden_specific_model_branch",
        "hidden_laplace_vi_not_same_sampler",
        "posterior_chain_order_untrustworthy",
    }
    structural = sorted(set(failures) & structural_failure_triggers)
    maturity = {
        "C_bayes": components.get("C", 0.0),
        "D_lab": components.get("D", 0.0),
        "F_report": components.get("F", 0.0),
        "C_threshold": 19.5,
        "D_threshold": 12.0,
        "F_threshold": 5.5,
    }
    mature = (
        maturity["C_bayes"] >= maturity["C_threshold"]
        and maturity["D_lab"] >= maturity["D_threshold"]
        and maturity["F_report"] >= maturity["F_threshold"]
    )
    severe_map_laplace = "map_laplace_only" in structural and "fast_pipeline_no_sampling" in structural
    if severe_map_laplace:
        penalty = 0.25
    elif not structural:
        penalty = 1.0
    elif mature:
        penalty = 0.50
    else:
        penalty = 0.34
    return penalty, {
        "structural_failures": structural,
        "maturity": maturity,
        "mature": mature,
        "severe_map_laplace": severe_map_laplace,
    }


def apply_cross_component_caps(
    components: dict[str, float],
    failures: list[str],
    details: dict[str, Any],
) -> None:
    """Apply benchmark calibration caps that depend on multiple components."""
    failure_set = set(failures)

    # Without any hidden execution, a submission can only demonstrate public
    # artifact quality. It should not receive high Bayesian/result scores.
    if "no_hidden_generalization" in failure_set:
        components["C"] = min(components.get("C", 0.0), 12.0)
        components["D"] = min(components.get("D", 0.0), 8.0)
        components["E"] = min(components.get("E", 0.0), 3.0)
        details.setdefault("global_caps", []).append("hidden_generalization_missing")

    if "no_real_1d_mt_forward_evidence" in failure_set:
        components["B"] = min(components.get("B", 0.0), 5.0)
        components["D"] = min(components.get("D", 0.0), 8.0)
        details.setdefault("global_caps", []).append("real_1d_forward_missing")

    if "water_depth_not_used" in failure_set:
        components["B"] = min(components.get("B", 0.0), 6.0)
        components["D"] = min(components.get("D", 0.0), 10.0)
        details.setdefault("global_caps", []).append("water_depth_missing")

    if "map_laplace_only" in failure_set:
        components["C"] = min(components.get("C", 0.0), 12.0)
        components["D"] = min(components.get("D", 0.0), 12.0)
        components["E"] = min(components.get("E", 0.0), 4.0)
        details.setdefault("global_caps", []).append("map_laplace_only")

    if "fast_pipeline_no_sampling" in failure_set:
        components["C"] = min(components.get("C", 0.0), 10.0)
        components["D"] = min(components.get("D", 0.0), 10.0)
        components["E"] = min(components.get("E", 0.0), 3.0)
        details.setdefault("global_caps", []).append("fast_pipeline_no_sampling")

    if "posterior_chain_order_untrustworthy" in failure_set:
        components["C"] = min(components.get("C", 0.0), 12.0)
        components["D"] = min(components.get("D", 0.0), 12.0)
        components["E"] = min(components.get("E", 0.0), 4.0)
        details.setdefault("global_caps", []).append("posterior_chain_order_untrustworthy")

    if "hidden_inference_chain_mismatch" in failure_set:
        components["C"] = min(components.get("C", 0.0), 12.0)
        components["D"] = min(components.get("D", 0.0), 8.0)
        components["E"] = min(components.get("E", 0.0), 3.0)
        details.setdefault("global_caps", []).append("hidden_inference_chain_mismatch")

    if "hidden_specific_model_branch" in failure_set:
        components["C"] = min(components.get("C", 0.0), 10.0)
        components["D"] = min(components.get("D", 0.0), 6.0)
        components["E"] = min(components.get("E", 0.0), 2.0)
        details.setdefault("global_caps", []).append("hidden_specific_model_branch")

    if "hidden_laplace_vi_not_same_sampler" in failure_set:
        components["C"] = min(components.get("C", 0.0), 10.0)
        components["D"] = min(components.get("D", 0.0), 6.0)
        components["E"] = min(components.get("E", 0.0), 2.0)
        details.setdefault("global_caps", []).append("hidden_laplace_vi_not_same_sampler")

    if "weak_hidden_lab_recovery" in failure_set:
        components["C"] = min(components.get("C", 0.0), 14.0)
        components["D"] = min(components.get("D", 0.0), 10.0)
        components["E"] = min(components.get("E", 0.0), 4.0)
        details.setdefault("global_caps", []).append("weak_hidden_lab_recovery")

    if "posterior_undercoverage_hidden" in failure_set:
        components["C"] = min(components.get("C", 0.0), 12.0)
        components["D"] = min(components.get("D", 0.0), 8.0)
        components["E"] = min(components.get("E", 0.0), 3.0)
        details.setdefault("global_caps", []).append("posterior_undercoverage_hidden")

    if "credible_interval_too_narrow_or_fixed" in failure_set:
        components["C"] = min(components.get("C", 0.0), 10.0)
        components["D"] = min(components.get("D", 0.0), 6.0)
        components["E"] = min(components.get("E", 0.0), 2.0)
        details.setdefault("global_caps", []).append("credible_interval_too_narrow_or_fixed")

    if {"weak_hidden_lab_recovery", "credible_interval_too_narrow_or_fixed"} <= failure_set:
        components["C"] = min(components.get("C", 0.0), 9.0)
        components["D"] = min(components.get("D", 0.0), 5.0)
        components["E"] = min(components.get("E", 0.0), 1.5)
        details.setdefault("global_caps", []).append("overconfident_weak_hidden_recovery")

    if "posthoc_prediction_correction" in failure_set:
        components["B"] = min(components.get("B", 0.0), 12.0)
        components["C"] = min(components.get("C", 0.0), 16.0)
        components["D"] = min(components.get("D", 0.0), 15.0)
        components["E"] = min(components.get("E", 0.0), 4.0)
        details.setdefault("global_caps", []).append("posthoc_prediction_correction")

    if "physical_model_hidden_by_discrepancy" in failure_set:
        components["B"] = min(components.get("B", 0.0), 10.0)
        components["C"] = min(components.get("C", 0.0), 14.0)
        components["D"] = min(components.get("D", 0.0), 10.0)
        components["E"] = min(components.get("E", 0.0), 4.0)
        details.setdefault("global_caps", []).append("physical_model_hidden_by_discrepancy")

    if "public_fit_not_noise_level" in failure_set:
        components["B"] = min(components.get("B", 0.0), 12.0)
        components["D"] = min(components.get("D", 0.0), 18.0)
        details.setdefault("global_caps", []).append("public_fit_not_noise_level")

    if {
        "public_fit_not_noise_level",
        "weak_hidden_lab_recovery",
        "posterior_undercoverage_hidden",
    } <= failure_set:
        components["C"] = min(components.get("C", 0.0), 11.0)
        components["D"] = min(components.get("D", 0.0), 7.0)
        components["E"] = min(components.get("E", 0.0), 2.5)
        details.setdefault("global_caps", []).append("public_fit_and_hidden_quality_failed")


def result_quality_ceiling(failures: list[str], details: dict[str, Any]) -> tuple[float | None, list[str]]:
    """Return a final-score ceiling for severe result-quality failures.

    This is a correctness gate, not a structural multiplier. A submission can
    have a valid sampler and no mechanism-level structural failures, but if the
    hidden LAB recovery and posterior coverage fail, it should not receive a
    high benchmark score merely for complete artifacts and good public fit.
    """
    failure_set = set(failures)
    ceilings: list[tuple[float, str]] = []

    if "no_hidden_generalization" in failure_set:
        ceilings.append((14.5, "hidden_generalization_missing"))
    if "weak_hidden_lab_recovery" in failure_set:
        ceilings.append((14.5, "weak_hidden_lab_recovery"))
    if "posterior_undercoverage_hidden" in failure_set:
        ceilings.append((29.5, "posterior_undercoverage_hidden"))
    if "credible_interval_too_narrow_or_fixed" in failure_set:
        ceilings.append((29.5, "credible_interval_too_narrow_or_fixed"))
    if "credible_interval_too_wide_uninformative" in failure_set:
        ceilings.append((29.5, "credible_interval_too_wide_uninformative"))
    if {"weak_hidden_lab_recovery", "posterior_undercoverage_hidden"} <= failure_set:
        ceilings.append((14.5, "hidden_recovery_and_coverage_failed"))
    if {
        "public_fit_not_noise_level",
        "weak_hidden_lab_recovery",
        "posterior_undercoverage_hidden",
    } <= failure_set:
        ceilings.append((14.5, "public_fit_and_hidden_quality_failed"))
    if {"weak_hidden_lab_recovery", "credible_interval_too_narrow_or_fixed"} <= failure_set:
        ceilings.append((14.5, "overconfident_weak_hidden_recovery"))
    if {"posterior_undercoverage_hidden", "credible_interval_too_narrow_or_fixed"} <= failure_set:
        ceilings.append((24.5, "posterior_undercoverage_and_overconfidence"))
    if {
        "weak_hidden_lab_recovery",
        "posterior_undercoverage_hidden",
        "credible_interval_too_narrow_or_fixed",
    } <= failure_set:
        ceilings.append((14.5, "hidden_recovery_coverage_and_ci_failed"))
    if "posthoc_prediction_correction" in failure_set:
        ceilings.append((29.5, "posthoc_prediction_correction"))
    if "physical_model_hidden_by_discrepancy" in failure_set:
        ceilings.append((24.0, "physical_model_hidden_by_discrepancy"))
    if "hidden_inference_chain_mismatch" in failure_set:
        ceilings.append((24.5, "hidden_inference_chain_mismatch"))
    if "hidden_specific_model_branch" in failure_set:
        ceilings.append((14.5, "hidden_specific_model_branch"))
    if "hidden_laplace_vi_not_same_sampler" in failure_set:
        ceilings.append((14.5, "hidden_laplace_vi_not_same_sampler"))
    if "posterior_chain_order_untrustworthy" in failure_set:
        ceilings.append((24.5, "posterior_chain_order_untrustworthy"))

    if not ceilings:
        return None, []
    ceiling = min(score for score, _ in ceilings)
    reasons = [reason for _, reason in ceilings]
    details.setdefault("result_quality_ceilings", reasons)
    return ceiling, reasons


def evaluate(submission_dir: Path) -> dict[str, Any]:
    out_dir = resolve_output_dir(submission_dir)
    summary, summary_err = load_json(out_dir / "summary.json")
    features = static_features(out_dir)

    details: dict[str, Any] = {}
    failures: list[str] = []
    components: dict[str, float] = {}

    a, da = score_A(out_dir)
    components["A"] = a
    details["A"] = da

    b, db, fb = score_B(out_dir, summary, features)
    components["B"] = b
    details["B"] = db
    failures.extend(fb)

    c, dc, fc = score_C(out_dir, summary, features)
    components["C"] = c
    details["C"] = dc
    failures.extend(fc)

    # Hidden runs are the costly part. They are attempted after basic artifact
    # checks; missing run_inversion.py simply yields hidden failure.
    hidden = score_hidden(out_dir)

    geo_maturity, fg = assess_geophysical_maturity(summary, hidden, out_dir)
    details["geophysical_maturity"] = geo_maturity
    failures.extend(fg)

    d, dd, fd = score_D(out_dir, summary, hidden)
    components["D"] = d
    details["D"] = dd
    failures.extend(fd)

    e, de, fe = score_E(features, hidden, fc, fb)
    components["E"] = e
    details["E"] = de
    failures.extend(fe)

    f, df, ff = score_F(out_dir)
    components["F"] = f
    details["F"] = df
    failures.extend(ff)

    failure_set = set(failures)
    if {
        "public_fit_not_noise_level",
        "weak_hidden_lab_recovery",
        "posterior_undercoverage_hidden",
    } <= failure_set:
        failures.append("public_fit_and_hidden_quality_failed")
    if {"weak_hidden_lab_recovery", "credible_interval_too_narrow_or_fixed"} <= failure_set:
        failures.append("overconfident_weak_hidden_recovery")

    apply_cross_component_caps(components, failures, details)

    raw_total = float(sum(components.values()))
    penalty, penalty_details = determine_penalty(components, failures)
    final_score = raw_total * penalty
    ceiling, _ceiling_reasons = result_quality_ceiling(failures, details)
    if ceiling is not None:
        final_score = min(final_score, ceiling)

    missing_required = list(details["A"].get("missing", []))
    fatal_missing = len(missing_required) >= 4
    if fatal_missing:
        failures.append("fatal_missing_required_outputs")
        raw_total = 0.0
        final_score = 0.0
        components = {k: 0.0 for k in components}
        penalty = 0.34
        penalty_details = {
            "structural_failures": ["fatal_missing_required_outputs"],
            "maturity": {
                "C_bayes": 0.0,
                "D_lab": 0.0,
                "F_report": 0.0,
                "C_threshold": 19.5,
                "D_threshold": 12.0,
                "F_threshold": 5.5,
            },
            "mature": False,
        }

    valid = bool(a > 0 and not missing_required and not fatal_missing)
    result = {
        "submission": str(submission_dir),
        "output_dir": str(out_dir),
        "valid": valid,
        "final_score": float(np.clip(final_score, 0.0, 100.0)),
        "score": float(np.clip(final_score, 0.0, 100.0)),
        "raw_total": raw_total,
        "structural_penalty": penalty,
        "components": components,
        "component_max": COMPONENT_MAX,
        "failures": sorted(set(failures)),
        "penalty_details": penalty_details,
        "details": details,
        "summary_error": summary_err,
    }
    return result


def structured_result(result: dict[str, Any]) -> dict[str, Any]:
    details = []
    pass_thresholds = {
        "A": 0.60,
        "B": 0.60,
        "C": 0.35,
        "D": 0.35,
        "E": 0.35,
        "F": 0.60,
    }
    for name in ["A", "B", "C", "D", "E", "F"]:
        score = float(result["components"].get(name, 0.0))
        max_score = float(COMPONENT_MAX[name])
        threshold = pass_thresholds.get(name, 0.60)
        status = "PASSED" if score >= threshold * max_score else "FAILED"
        details.append({
            "name": name,
            "status": status,
            "message": f"score={score:.2f}/{max_score:.2f}, pass_threshold={threshold:.2f}",
        })
    failures = result.get("failures", [])
    summary = (
        f"final={result['final_score']:.2f}/100 | raw={result['raw_total']:.2f} | "
        f"penalty={result['structural_penalty']:.2f} | "
        + " ".join(f"{k}={result['components'][k]:.1f}" for k in ["A", "B", "C", "D", "E", "F"])
    )
    if failures:
        summary += " | failures=" + ",".join(failures[:6])
    return {
        "valid": bool(result.get("valid", False)),
        "score": float(result["final_score"]),
        "summary": summary,
        "details": details,
        "metrics": {
            "raw_total": float(result["raw_total"]),
            "structural_penalty": float(result["structural_penalty"]),
            "components": result["components"],
            "structural_failures": result["penalty_details"]["structural_failures"],
            "output_dir": result["output_dir"],
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python evaluate.py /path/to/submission", file=sys.stderr)
        return 2
    submission_dir = Path(argv[1]).resolve()
    try:
        result = evaluate(submission_dir)
    except Exception as exc:
        result = {
            "submission": str(submission_dir),
            "output_dir": str(submission_dir),
            "valid": False,
            "final_score": 0.0,
            "score": 0.0,
            "raw_total": 0.0,
            "structural_penalty": 0.34,
            "components": {k: 0.0 for k in COMPONENT_MAX},
            "component_max": COMPONENT_MAX,
            "failures": ["evaluator_error"],
            "penalty_details": {"structural_failures": ["evaluator_error"], "mature": False},
            "details": {
                "evaluator_error": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            },
        }

    try:
        out_dir = Path(result.get("output_dir", submission_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_result.json").write_text(
            json.dumps(_json_safe(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "score.json").write_text(
            json.dumps(_json_safe({
                "final_score": result["final_score"],
                "raw_total": result["raw_total"],
                "structural_penalty": result["structural_penalty"],
                "components": result["components"],
                "failures": result["failures"],
            }), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[evaluate] warning: could not write eval result: {exc}", file=sys.stderr)

    structured = structured_result(result)
    print(">>>>> Start Structured Result")
    print(json.dumps(_json_safe(structured), ensure_ascii=False))
    print(">>>>> End Structured Result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
