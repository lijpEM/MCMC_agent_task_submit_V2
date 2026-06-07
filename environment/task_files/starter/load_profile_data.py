from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "mt_profile_20_public.npz"


def load_profile_data(path: str | Path | None = None) -> dict[str, Any]:
    """Load the public 20-station marine MT profile.

    Apparent resistivity arrays are already stored as log10(ohm m). Phase is
    stored in degrees. The returned dictionary contains NumPy arrays copied out
    of the NPZ file so the file handle can close immediately.
    """
    data_path = Path(path) if path is not None else default_data_path()
    with np.load(data_path, allow_pickle=False) as npz:
        out = {key: npz[key].copy() for key in npz.files}
    aliases = {
        "TM_log10_rhoa": ["log10_rhoa_tm", "tm_log10_rhoa"],
        "TM_log10_rhoa_err": ["log10_rhoa_tm_std", "tm_log10_rhoa_std", "tm_log10_rhoa_err"],
        "TM_phase_deg": ["phase_tm_deg", "tm_phase_deg"],
        "TM_phase_err_deg": ["phase_tm_std_deg", "tm_phase_std_deg", "tm_phase_err_deg"],
        "TE_log10_rhoa": ["log10_rhoa_te", "te_log10_rhoa"],
        "TE_log10_rhoa_err": ["log10_rhoa_te_std", "te_log10_rhoa_std", "te_log10_rhoa_err"],
        "TE_phase_deg": ["phase_te_deg", "te_phase_deg"],
        "TE_phase_err_deg": ["phase_te_std_deg", "te_phase_std_deg", "te_phase_err_deg"],
    }
    lower_lookup = {key.lower(): key for key in out}
    for canonical, candidates in aliases.items():
        if canonical in out:
            continue
        for candidate in candidates:
            actual = lower_lookup.get(candidate.lower())
            if actual is not None:
                out[canonical] = np.asarray(out[actual]).copy()
                break
    if "periods_s" not in out and "freqs_hz" in out:
        out["periods_s"] = 1.0 / np.asarray(out["freqs_hz"], dtype=float)
    out["data_path"] = str(data_path)
    return out


def tm_observation_vector(profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened TM observations and standard deviations.

    The order is `[log10_rhoa values, phase values]`, flattened station-major.
    """
    obs = np.r_[
        np.asarray(profile["TM_log10_rhoa"], dtype=float).ravel(),
        np.asarray(profile["TM_phase_deg"], dtype=float).ravel(),
    ]
    sigma = np.r_[
        np.asarray(profile["TM_log10_rhoa_err"], dtype=float).ravel(),
        np.asarray(profile["TM_phase_err_deg"], dtype=float).ravel(),
    ]
    return obs, sigma


def summarize_profile(profile: dict[str, Any]) -> str:
    names = profile["station_names"]
    freqs = profile["freqs_hz"]
    water = profile["water_depth_m"]
    return (
        f"{len(names)} stations, {len(freqs)} frequencies, "
        f"water depth {water.min():.1f}-{water.max():.1f} m"
    )


if __name__ == "__main__":
    prof = load_profile_data()
    print(summarize_profile(prof))
    print("Keys:")
    for key in sorted(prof):
        value = prof[key]
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
