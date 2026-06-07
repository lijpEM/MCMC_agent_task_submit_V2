from __future__ import annotations

import numpy as np

from mt1d_forward import marine_lab_forward


def interpolate_controls(x_m: np.ndarray, control_x_m: np.ndarray, control_values: np.ndarray) -> np.ndarray:
    """Linear low-rank lateral interpolation helper.

    Stronger submissions can replace this with cubic splines, B-splines, GP
    bases, PCA bases, or another smooth parameterization.
    """
    x = np.asarray(x_m, dtype=float)
    cx = np.asarray(control_x_m, dtype=float)
    cv = np.asarray(control_values, dtype=float)
    order = np.argsort(cx)
    return np.interp(x, cx[order], cv[order])


def unpack_baseline_parameters(theta: np.ndarray, x_m: np.ndarray) -> dict[str, np.ndarray | float]:
    """Map a compact parameter vector to station-wise pseudo-2D properties.

    Parameter order:
        0: z_lab left control, km
        1: z_lab middle control, km
        2: z_lab right control, km
        3: global conductor thickness, km
        4: log10 lithosphere resistivity
        5: log10 conductor resistivity
        6: log10 deep resistivity
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size != 7:
        raise ValueError("baseline theta must have 7 parameters")
    control_x = np.linspace(float(np.min(x_m)), float(np.max(x_m)), 3)
    z_lab = interpolate_controls(x_m, control_x, theta[:3])
    return {
        "z_lab_km": z_lab,
        "h_cond_km": float(theta[3]),
        "rho_lith_ohm_m": float(10.0 ** theta[4]),
        "rho_cond_ohm_m": float(10.0 ** theta[5]),
        "rho_deep_ohm_m": float(10.0 ** theta[6]),
    }


def forward_profile(
    freqs_hz: np.ndarray,
    x_m: np.ndarray,
    water_depth_m: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    """Predict log10 apparent resistivity and phase at all stations."""
    props = unpack_baseline_parameters(theta, x_m)
    n_station = len(x_m)
    n_freq = len(freqs_hz)
    log10_rhoa = np.zeros((n_station, n_freq), dtype=float)
    phase_deg = np.zeros((n_station, n_freq), dtype=float)
    for i in range(n_station):
        rhoa, phase = marine_lab_forward(
            freqs_hz,
            water_depth_m=float(water_depth_m[i]),
            z_lab_km=float(props["z_lab_km"][i]),
            h_cond_km=float(props["h_cond_km"]),
            rho_lith_ohm_m=float(props["rho_lith_ohm_m"]),
            rho_cond_ohm_m=float(props["rho_cond_ohm_m"]),
            rho_deep_ohm_m=float(props["rho_deep_ohm_m"]),
        )
        log10_rhoa[i] = np.log10(np.maximum(rhoa, 1e-300))
        phase_deg[i] = phase
    return log10_rhoa, phase_deg, props
