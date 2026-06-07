from __future__ import annotations

import numpy as np

from load_profile_data import load_profile_data, summarize_profile
from mt1d_forward import marine_lab_forward


def main() -> None:
    profile = load_profile_data()
    print(summarize_profile(profile))
    idx = 0
    rhoa, phase = marine_lab_forward(
        profile["freqs_hz"],
        water_depth_m=float(profile["water_depth_m"][idx]),
        z_lab_km=55.0,
        h_cond_km=35.0,
        rho_lith_ohm_m=300.0,
        rho_cond_ohm_m=8.0,
        rho_deep_ohm_m=100.0,
    )
    print(f"Station {profile['station_names'][idx]} water_depth={profile['water_depth_m'][idx]:.1f} m")
    for freq, obs, pred, obs_phase, pred_phase in zip(
        profile["freqs_hz"],
        profile["TM_log10_rhoa"][idx],
        np.log10(rhoa),
        profile["TM_phase_deg"][idx],
        phase,
    ):
        print(
            f"{freq:.6g} Hz  obs_log10_rhoa={obs:.3f}  "
            f"pred_log10_rhoa={pred:.3g}  obs_phase={obs_phase:.2f}  "
            f"pred_phase={pred_phase:.2f}"
        )


if __name__ == "__main__":
    main()
