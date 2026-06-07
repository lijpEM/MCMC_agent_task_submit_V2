from __future__ import annotations

import numpy as np


MU0 = 4.0 * np.pi * 1e-7


def mt1d_forward(
    freqs_hz: np.ndarray,
    resistivity_ohm_m: np.ndarray,
    thickness_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D MT apparent resistivity and phase for layered earth.

    Parameters
    ----------
    freqs_hz
        Frequencies in Hz.
    resistivity_ohm_m
        Resistivity for each layer, including the bottom half-space.
    thickness_m
        Thickness for all finite layers. Its length must be one less than
        `resistivity_ohm_m`.

    Returns
    -------
    rhoa_ohm_m, phase_deg
        Apparent resistivity and impedance phase at the top of the first layer.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    rho = np.asarray(resistivity_ohm_m, dtype=float)
    thick = np.asarray(thickness_m, dtype=float)
    if rho.ndim != 1 or thick.ndim != 1:
        raise ValueError("resistivity_ohm_m and thickness_m must be 1D arrays")
    if rho.size != thick.size + 1:
        raise ValueError("thickness_m must have len(resistivity_ohm_m) - 1 entries")
    if np.any(freqs <= 0) or np.any(rho <= 0) or np.any(thick < 0):
        raise ValueError("frequencies/resistivities must be positive and thickness non-negative")

    omega = 2.0 * np.pi * freqs
    iwmu = 1j * omega * MU0

    impedance = np.sqrt(iwmu * rho[-1])
    for layer in range(rho.size - 2, -1, -1):
        z0 = np.sqrt(iwmu * rho[layer])
        gamma = np.sqrt(iwmu / rho[layer])
        tanh_term = np.tanh(gamma * thick[layer])
        impedance = z0 * (impedance + z0 * tanh_term) / (z0 + impedance * tanh_term)

    rhoa = np.abs(impedance) ** 2 / (MU0 * omega)
    phase = np.degrees(np.arctan2(impedance.imag, impedance.real))
    return rhoa.real, phase.real


def marine_lab_forward(
    freqs_hz: np.ndarray,
    water_depth_m: float,
    z_lab_km: float,
    h_cond_km: float,
    rho_lith_ohm_m: float = 300.0,
    rho_cond_ohm_m: float = 10.0,
    rho_deep_ohm_m: float = 100.0,
    rho_water_ohm_m: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward response for a seawater/lithosphere/conductor/deep model."""
    water_km = float(water_depth_m) / 1000.0
    z_lab_km = max(float(z_lab_km), water_km + 0.1)
    h_cond_km = max(float(h_cond_km), 0.1)
    lith_thick_m = max((z_lab_km - water_km) * 1000.0, 1.0)
    cond_thick_m = h_cond_km * 1000.0
    resistivity = np.array(
        [rho_water_ohm_m, rho_lith_ohm_m, rho_cond_ohm_m, rho_deep_ohm_m],
        dtype=float,
    )
    thickness = np.array([water_depth_m, lith_thick_m, cond_thick_m], dtype=float)
    return mt1d_forward(freqs_hz, resistivity, thickness)


if __name__ == "__main__":
    freqs = np.logspace(-3, -1, 5)
    rhoa, phase = marine_lab_forward(freqs, water_depth_m=3000.0, z_lab_km=55.0, h_cond_km=35.0)
    for f, r, p in zip(freqs, rhoa, phase):
        print(f"{f:.5g} Hz  rhoa={r:.4g} ohm m  phase={p:.2f} deg")
