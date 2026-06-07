# SERPENT 1D Bayesian Inversion - Python port

Python re-implementation of the SERPENT (Blatter et al. 2022) PT-RJMCMC
1D Bayesian inversion code, with support for marine MT, marine obCSEM
(reciprocity-swapped, Type 41 Pmax_E), surface-towed CSEM (stCSEM) and
DC resistivity (DCR).

## File inventory

### Core engine (rarely need to edit)

| file | purpose |
|---|---|
| `rjmcmc_inversion.py` | PT-RJMCMC core (chain steps, swap, posterior storage), plot routines, multiprocessing parallel-chain support, depth-dependent prior, dependency-injection slots for forward solvers |
| `MT.py` | EDI parser, Mare2DEM multi-station MT reader (`read_mt_mare2dem`), `edi_to_S_MTdat`, 1D MT analytical forward (`MT1D`) |
| `csem_forward.py` | empymod wrapper for 1D CSEM forward (`dipole1d_wrapper`, `pmax_E_amplitude` with finite TxLength support); single entry point `register_with_serpent()` plugs it into the inversion |
| `emdata.py` | Mare2DEM `.emdata` parser, reciprocity-swap helper `extract_serpent_obDat` |
| `make_maxrho_1300C_MPT.py` | Generator for `maxrho_1300C_MPT.txt`: half-space cooling T(z) + dry-olivine Arrhenius |
| `maxrho_1300C_MPT.txt` | 2-column depth-dependent rho_max prior (Blatter 2022 style; 1300 deg C MPT, 22 Myr Cocos Plate) |

### Workflow scripts (the ones you edit)

| file | use for | edit per run |
|---|---|---|
| `Step1_initialize.py` | **MT** workflow (EDI or Mare2DEM multi-station .txt) | `FileNameRoot`, `edi_path`, `station_name` (if multi-station), `TE/TM` mode, `rho_max_profile_path` |
| `Step1_initialize_obCSEM.py` | **obCSEM** workflow (Mare2DEM .emdata) | `FileNameRoot`, `emdata_path`, `rx_name` (which Rx in the file) |
| `Step2_run.py` | run PT-RJMCMC | just `FileNameRoot` |
| `Step3_assess.py` | plot RMS / k / AR convergence diagnostics, pick burn-in | `FileNameRoot` |
| `Step4_combine.py` | merge T=1 chains, plot model responses vs data, RMS histogram | `FileNameRoot`, `burnIn` |
| `Step5_plot.py` | plot posterior PDF (paper-style parula + linear) | `FileNameRoot` |

## Run sequence

```
python Step1_initialize.py            # or Step1_initialize_obCSEM.py
python Step2_run.py
python Step3_assess.py                # inspect figures, pick burnIn
# edit burnIn in Step4_combine.py
python Step4_combine.py
python Step5_plot.py
```

## Key features

1. **Multi-data joint inversion**: MT, obCSEM, stCSEM, DCR can be combined
   (currently auto-mixed for joint data sets).
2. **Multi-chain parallelisation**: `n_workers` setting in Step1.  Set to
   `'auto'` (default) - 1 for MT/DCR-only, 8 for CSEM.  Linux fork only.
3. **Depth-dependent rho_max prior**: feed any 2-col txt file via
   `rho_max_profile_path` in Step1.  Default is `maxrho_1300C_MPT.txt`.
4. **Frequency vectorisation**: empymod called with all frequencies in
   one Hankel-transform call; ~2-3x speedup on the CSEM forward.
5. **Headless matplotlib (Agg backend)**: Step3/4/5 force Agg so they
   never hang on an SSH X11 socket.
6. **Convergence-plot thinning**: long chains (1e6+ iter) are
   auto-thinned to 5k points for plotting, no more multi-minute hangs.

## What MATLAB calls vs Python equivalents

| MATLAB | Python |
|---|---|
| `mexDipole1D` | `csem_forward.dipole1d_wrapper` (empymod backend) |
| `MT1D.m` | `rjmcmc_inversion.MT1D` |
| `getMisfit.m` | `rjmcmc_inversion.getMisfit` |
| `PT_RJMCMC.m` | `rjmcmc_inversion.PT_RJMCMC` |
| `CombineChains.m` | `rjmcmc_inversion.CombineChains` |
| `plot_convergence_PT_RJMCMC.m` | `rjmcmc_inversion.plot_convergence_PT_RJMCMC` |
| `plot_RJMCMC.m` | `rjmcmc_inversion.plot_RJMCMC` / `plot_RJMCMC_paper` |
| `maxrho_1300C_MPT.mat` | `maxrho_1300C_MPT.txt` (text 2-col) |

## Verified against published results

- **MT s08 (Blatter 2022 Cocos Plate)**: reproduces Extended Data Fig. 2a
  qualitatively, 20k iter RMS ~ 0.92 (paper RMS 0.73 at 2M iter).
- **MT s10**: reproduces Extended Data Fig. 2b, 20k iter RMS ~ 0.69.
- **MT EM5 / s02 / WPB**: clean RMS-1 fits.
- **obCSEM L3 R05**: see Step1 / Step4 diagnostics; in active use.

## Dependencies

```
numpy
scipy
matplotlib
empymod         # required for any CSEM workflow
```

## Common gotchas

- After ANY change to Step1 config (incl. n_workers), you MUST re-run
  Step1 to refresh the pickles - Step2 reads `n_workers` from the
  saved pickle, not from Step1 source.
- Step2 must print `[PT_RJMCMC] using N worker process(es)` for the
  parallel path to be active.  If missing, the pickle was written by
  an older Step1.
- For SSH-X11 users: scripts force `MPLBACKEND=Agg` -- safe.  If you
  want figures to display interactively, run with `MPLBACKEND=TkAgg`
  on your local machine after copying the PNGs back.
