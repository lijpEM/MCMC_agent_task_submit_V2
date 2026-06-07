"""
Step5_plot.py
=====================================================================
Plot the posterior PDF of log10(rho) vs depth and related diagnostics
(KL divergence, layer-count, etc.) from the combined ensemble file.

Default uses LINEAR depth (km) from 0 to 200 km, which is more readable
for typical MT-inversion ranges than the log-depth view.  For marine
data the water column (0..seafloor) is shown; for land data the plot
starts from the surface.

Set `depth_axis = 'log'` to fall back to the original log-depth plot.
=====================================================================
"""
import os
import pickle
from types import SimpleNamespace

import numpy as np
# Force the headless Agg backend BEFORE importing pyplot.  Matches
# Step3/Step4: when the server is reached via `ssh -X`, DISPLAY is
# set and the default matplotlib backend becomes TkAgg / Qt5Agg,
# which can hang on big plots over slow X11 tunnels.
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rjmcmc_inversion import (plot_RJMCMC,
                              plot_RJMCMC_linear,
                              plot_RJMCMC_paper,
                              _load_input)


FileNameRoot = 's08_TM'
out_dir      = 'Trash'

# --- USER PARAMETERS --------------------------------------------------
thin               = 10           # thin ensemble before binning
depth_axis         = 'linear'     # 'linear' (recommended) or 'log'

# Linear-depth settings (used when depth_axis == 'linear')
depth_max_km       = 200.0        # plot from 0 to this depth (km)
depth_step_km      = 1.0          # depth bin size (km)
rho_step           = 0.05         # log10(rho) bin size
show_water_column  = True         # show 0..seafloor in marine plots
shallow_zoom_km    = 10.0         # add a shallow-zoom subplot (set None to skip)

# Paper-style single panel (publication aesthetic).
# Set produce_paper_plot=False to skip.
produce_paper_plot      = True
paper_depth_min_km      = None     # top of y-axis; None = auto:
                                   #   marine -> seafloor (skip water column)
                                   #   land   -> 0
paper_depth_max_km      = None     # bottom of y-axis; None = inversion zMax
paper_pdf_log_range     = (-3.0, -0.5)   # fixed PDF colour range
paper_overlay_depths_km = None     # e.g. [40, 73] for LAB/Moho dashes
paper_title             = None     # e.g. 'EM5' shown inside the plot
paper_rho_xlim          = (-1.0, 5.0)
paper_figsize           = (4.5, 8.0)
paper_smooth_sigma      = (1.0, 1.0)  # Gaussian smoothing in (z, rho) bins
                                       # to remove sampling-noise speckle.
                                       # (0,0) = raw histogram (shows speckle)
                                       # (1,1) = mild, recommended
                                       # (2,2) = heavier, more "painted" look
paper_zero_pdf_color    = 'low'    # 'low' = dark blue (paper-style fill),
                                   # 'white' = NaN (diagnostic - shows zeros)

# Log-depth settings (only used when depth_axis == 'log')
G                  = SimpleNamespace(dz=0.05, drho=0.05)

show_plot          = False
# ---------------------------------------------------------------------


if __name__ == '__main__':
    S  = _load_input(os.path.join(out_dir, FileNameRoot + '.pkl'))
    S1 = _load_input(os.path.join(out_dir, 'WaterColumn.pkl'))
    S2 = _load_input(os.path.join(out_dir, 'Subsurface.pkl'))

    print('Loading model ensemble...')
    combined_fp = os.path.join(out_dir, FileNameRoot + '_PT_RJMCMC_Combined.pkl')
    with open(combined_fp, 'rb') as fh:
        U = pickle.load(fh)

    # thin
    k2 = np.asarray(U['k2_ll']).ravel()[::thin]
    s  = U['s_ll'][::thin]
    print(f'  thinned ensemble: {len(s)} models, k2 range {k2.min():.0f}..{k2.max():.0f}')

    # rho axis bounds & uniform prior (same for both depth axes)
    if getattr(S, 'transform01_ab', False):
        S.rhoMin = float(np.min(S.minRho))
        S.rhoMax = float(np.max(S.maxRho))
    else:
        S.rhoMin = S2.rhoMin
        S.rhoMax = S2.rhoMax

    rho_edges_uniform = np.arange(S.rhoMin, S.rhoMax + 1e-12,
                                  rho_step if depth_axis == 'linear' else G.drho)
    prior_pdf = np.full(len(rho_edges_uniform) - 1,
                        1.0 / (S.rhoMax - S.rhoMin))

    if depth_axis == 'linear':
        # The inversion only places interfaces between S2.zMin and S2.zMax.
        # Below S2.zMax, every sample is forced to the deepest halfspace
        # rho - that's a 1-parameter marginal, not a depth-resolved
        # posterior - so we MASK it out to avoid misinterpretation.
        inv_zMax_m  = 10.0 ** S2.zMax if getattr(S, 'logZ', False) else S2.zMax
        inv_zMax_km = inv_zMax_m / 1000.0
        if depth_max_km > inv_zMax_km + 5.0:
            print(f'  [info] requested depth_max_km={depth_max_km:g} > '
                  f'inversion zMax={inv_zMax_km:.1f} km;')
            print(f'         depths > {inv_zMax_km:.1f} km will be masked '
                  f'(no real info there).')
        print(f'Computing posterior PDF (linear depth, 0..{depth_max_km:g} km, '
              f'inv zMax = {inv_zMax_km:.1f} km)...')
        fig, posteriorPDF, p5, p95, KLd = plot_RJMCMC_linear(
            s, k2, S,
            depth_max_km      = depth_max_km,
            depth_step_km     = depth_step_km,
            rho_step          = rho_step,
            show_water_column = show_water_column,
            prior_pdf         = prior_pdf,
            shallow_zoom_km   = shallow_zoom_km,
            inversion_zMax_km = inv_zMax_km,
        )
        out_path = os.path.join(out_dir,
                                f'{FileNameRoot}_step5_posterior_linear.png')
    else:
        # log-depth fallback (original behaviour)
        if not S.landData:
            S1.zMin = 0.0
        if getattr(S, 'transform01_ab', False):
            S.zMax = S2.zMax; S.zMin = S2.zMin
            S.zRhoLim = np.log10(np.asarray(S.zRhoLim))
        else:
            S.zMax = S2.zMax
            S.zMin = S2.zMin if S.landData else S1.zMin
        G.prior = prior_pdf

        print('Computing posterior PDF (log depth)...')
        fig, posteriorPDF, p5, p95, KLd = plot_RJMCMC(s, k2, G, S)
        out_path = os.path.join(out_dir,
                                f'{FileNameRoot}_step5_posterior.png')

    fig.suptitle(f'{FileNameRoot}: posterior of $\\log_{{10}}\\rho$',
                 y=1.02, fontsize=11)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  Saved {out_path}')

    # ---------------------------------------------------------------
    # Publication-style single-panel figure (paper aesthetic).
    # ---------------------------------------------------------------
    if produce_paper_plot and depth_axis == 'linear':
        # default bottom of y-axis = inversion zMax
        zmax_paper = paper_depth_max_km if paper_depth_max_km is not None \
                     else inv_zMax_km
        # default top of y-axis: skip water column for marine, 0 for land
        if paper_depth_min_km is not None:
            zmin_paper = paper_depth_min_km
        elif not S.landData:
            zmin_paper = (10.0 ** S.regionBoundaryDepth) / 1000.0
        else:
            zmin_paper = 0.0
        title_str = paper_title if paper_title is not None else FileNameRoot

        print(f'Computing paper-style posterior ({zmin_paper:.1f}..'
              f'{zmax_paper:.1f} km, PDF range '
              f'{paper_pdf_log_range[0]:.1f}..{paper_pdf_log_range[1]:.1f})...')
        fig_p, pdf_p, p5_p, p50_p, p95_p, _KLd_p = plot_RJMCMC_paper(
            s, k2, S,
            depth_min_km      = zmin_paper,
            depth_max_km      = zmax_paper,
            depth_step_km     = 0.5,
            rho_step          = rho_step,
            show_water_column = False,    # paper plot always shows subsurface only
            inversion_zMax_km = inv_zMax_km,
            pdf_log_range     = paper_pdf_log_range,
            overlay_depths_km = paper_overlay_depths_km,
            title             = title_str,
            figsize           = paper_figsize,
            rho_xlim          = paper_rho_xlim,
            smooth_sigma      = paper_smooth_sigma,
            zero_pdf_color    = paper_zero_pdf_color,
        )
        paper_path = os.path.join(out_dir,
                                  f'{FileNameRoot}_step5_paper.png')
        fig_p.savefig(paper_path, dpi=200, bbox_inches='tight')
        print(f'  Saved {paper_path}')

    if show_plot:
        plt.show()
    plt.close('all')
    print('Step 5 done.')
