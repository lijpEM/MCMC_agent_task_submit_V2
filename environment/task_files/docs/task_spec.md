# Pseudo-2D Bayesian Marine MT LAB Inversion

## Goal

Implement a reduced-order Bayesian inversion for a marine magnetotelluric
profile and estimate the posterior uncertainty of the oceanic
lithosphere-asthenosphere boundary (LAB) and the underlying conductive layer.

This is not a full 2D or 3D Maxwell solver task. Each station may use a 1D
layered-earth MT forward model, but all stations must be coupled through a
shared low-dimensional lateral model for LAB depth, conductor thickness, and
resistivity.

## Provided Files

- `data/mt_profile_20_public.npz`
  - 20 representative marine MT stations.
  - 12 frequencies.
  - station names, profile coordinates, water depth, TE/TM log10 apparent
    resistivity, phase, uncertainty arrays, and observed masks.
  - Apparent resistivity is already stored as `log10(ohm m)`.
- `data/serpent_mt/SERPENT_fullMTdataSet.txt`
  - Original Mare2DEM-style multi-station data file for reference.
- `data/serpent_mt/maxrho_1300C_MPT.txt`
  - Depth-dependent resistivity upper-bound prior reference.
- `docs/references/`
  - Background papers on marine MT, LAB imaging, and reduced-order
    probabilistic inversion.
- `starter/`
  - Lightweight data loading, 1D MT forward, pseudo-2D scaffold, and baseline
    scripts.
- `starter/serpent_python_port/`
  - Full SERPENT-style reference port. You may inspect or reuse it, but the
    expected task solution should write its final files to `outputs/`.

## Required Work

1. Read the public profile data and correctly handle water depth, frequency,
   log10 apparent resistivity, phase, and uncertainty arrays.
2. Implement or reuse a 1D layered-earth MT forward solver.
3. Build a pseudo-2D model in which all stations share a laterally continuous,
   low-dimensional LAB/conductor parameterization.
4. Define priors, likelihood, and posterior. At minimum, use TM `log10(rhoa)`
   and TM phase with uncertainty-normalized residuals:

   ```text
   log L(m) = -0.5 * sum(((d_obs - d_pred) / sigma)^2)
   ```

5. Run MCMC, variational inference, SMC, Laplace approximation, or an
   equivalent Bayesian posterior approximation.
6. Output posterior samples, posterior profile summaries, data prediction
   summaries, diagnostics, and a short scientific report.

## Required Outputs

Write all final files under `outputs/`:

```text
outputs/run_inversion.py
outputs/pseudo2d_model.py
outputs/posterior_samples.npy
outputs/model_posterior_profile.npz
outputs/data_pred_mean.npz
outputs/summary.json
outputs/report.md
```

Recommended figures:

```text
outputs/fig_profile_posterior.png
outputs/fig_data_fit_sections.png
outputs/fig_trace_diagnostics.png
outputs/fig_uncertainty_vs_station.png
```

## `summary.json` Minimum Fields

```json
{
  "method": "adaptive_mcmc",
  "mode_used": "TM",
  "n_stations": 20,
  "n_freqs": 12,
  "n_parameters": 14,
  "n_samples": 10000,
  "burn_in": 2000,
  "acceptance_rate": 0.25,
  "ess_min": 100,
  "rhat_max": 1.2,
  "chi2_per_dof": 1.5,
  "rmse_log10_rhoa": 0.08,
  "rmse_phase_deg": 4.0,
  "runtime_seconds": 300,
  "prior_description": "spline LAB depth, positive conductor thickness, bounded resistivities",
  "forward_description": "laterally coupled 1D MT forward"
}
```

## `model_posterior_profile.npz` Recommended Keys

```text
x_m
station_names
water_depth_m
z_lab_mean_km
z_lab_p05_km
z_lab_p50_km
z_lab_p95_km
z_bottom_mean_km
z_bottom_p05_km
z_bottom_p50_km
z_bottom_p95_km
h_cond_mean_km
h_cond_p05_km
h_cond_p50_km
h_cond_p95_km
rho_cond_mean_ohm_m
rho_cond_p50_ohm_m
```

## `data_pred_mean.npz` Recommended Keys

```text
station_names
freqs_hz
TM_obs_log10_rhoa
TM_pred_log10_rhoa
TM_sigma_log10_rhoa
TM_obs_phase_deg
TM_pred_phase_deg
TM_sigma_phase_deg
TE_obs_log10_rhoa
TE_pred_log10_rhoa
TE_sigma_log10_rhoa
TE_obs_phase_deg
TE_pred_phase_deg
TE_sigma_phase_deg
```

## Rules

- Do not implement a full 2D/3D finite-element or finite-difference MT solver.
- Do not invert each station independently and simply stitch the results.
- Do not output only deterministic MAP/least-squares results with artificial
  fixed-width error bars.
- Do not fabricate posterior samples by repeating one model or adding
  unjustified noise to a single optimum.
- Do not hard-code public station names, array shapes, responses, or a fixed
  LAB curve; hidden scenarios may alter station and frequency counts.
- You must use station-specific water depth to build the seawater layer.
- You must handle apparent resistivity units correctly: public data use
  `log10(ohm m)`.
- At minimum use TM `log10(rhoa)` and TM phase. If you omit TE, explain why in
  `report.md`.

## Scoring

The scorer checks required artifacts, static evidence of a pseudo-2D Bayesian
implementation, posterior sample quality, profile physical constraints,
prediction fit metrics, diagnostics in `summary.json`, and scientific content
in `report.md`.
