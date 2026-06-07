# Starter Guide

This directory contains a lightweight starting point for the pseudo-2D marine
MT Bayesian LAB inversion task.

## Files

```text
load_profile_data.py        Load data/mt_profile_20_public.npz.
mt1d_forward.py             1D layered-earth MT forward model.
pseudo2d_model.py           Small pseudo-2D scaffold using shared lateral controls.
example_forward_one_station.py
                            Single-station forward smoke test.
run_baseline_map.py         Simple deterministic baseline that writes outputs/.
serpent_python_port/        Larger SERPENT-style reference port.
```

The baseline is intentionally simple. It is useful for checking file formats,
but it is not a strong task solution because the task asks for a real
pseudo-2D Bayesian posterior.

## Quick Start

From the task root:

```bash
python starter/example_forward_one_station.py
python starter/run_baseline_map.py
```

The baseline writes:

```text
outputs/run_inversion.py
outputs/pseudo2d_model.py
outputs/posterior_samples.npy
outputs/model_posterior_profile.npz
outputs/data_pred_mean.npz
outputs/summary.json
outputs/report.md
```

## Data Conventions

- `freqs_hz` are frequencies in Hz.
- `water_depth_m` is positive downward and should be used as seawater layer
  thickness.
- Apparent resistivity arrays are `log10(ohm m)`, not linear resistivity.
- Phase arrays are in degrees.
- `*_observed_mask` arrays identify entries present in the original
  Mare2DEM-style file. Some missing public entries were interpolated only to
  provide a rectangular 20 x 12 array.

## Expected Modeling Direction

A stronger solution should replace the baseline with:

- a shared lateral parameterization, such as spline control points for
  `z_LAB(x)`;
- positive conductor thickness or bottom depth constraints;
- bounded lithosphere, conductor, and deep resistivity priors;
- an uncertainty-aware likelihood using normalized residuals;
- posterior samples from MCMC, SMC, VI, Laplace/VI hybrid, or an equivalent
  Bayesian approximation;
- credible intervals and diagnostics that are actually derived from the
  posterior.
