"""
Step3_assess.py
====================================================================
Plot misfit and MCMC acceptance-rate diagnostics for every chain so
you can pick the burn-in length for Step 4.

Reads `nChains` from the main pickle written by Step 1, so you only
need to change FileNameRoot / out_dir here.
====================================================================
"""
import os
# Force the headless Agg backend BEFORE importing pyplot.  When the
# server is reached via `ssh -X` (or -Y) DISPLAY gets set, and the
# default matplotlib backend becomes TkAgg / Qt5Agg.  Those backends
# build their windows through X11 round-trips -- with 8 chains x 4x2
# subplots over a slow SSH tunnel the script can hang in 'S' state at
# 0 % CPU waiting on the X server.  Agg renders straight to PNG with
# no GUI calls and never blocks.
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rjmcmc_inversion import plot_convergence_PT_RJMCMC, _load_input


FileNameRoot = 's08_TM'
out_dir      = 'Trash'
show_plot    = False              # ignored under Agg; left for symmetry


if __name__ == '__main__':
    # auto-detect nChains from the Step 1 pickle
    S = _load_input(os.path.join(out_dir, FileNameRoot + '.pkl'))
    nChains = int(getattr(S, 'nChains', 8))

    print(f'Plotting convergence for {nChains} chains...')
    plot_convergence_PT_RJMCMC(os.path.join(out_dir, FileNameRoot), nChains)

    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig  = plt.figure(fig_num)
        path = os.path.join(out_dir, f'{FileNameRoot}_step3_fig{i}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        print(f'  Saved {path}')

    plt.close('all')
    print('Done.  Inspect the figures to choose a burn-in length, then run Step 4.')
