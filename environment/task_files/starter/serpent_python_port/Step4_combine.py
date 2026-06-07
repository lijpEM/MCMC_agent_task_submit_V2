"""
Step4_combine.py
=====================================================================
Combine post-burn-in samples from all the T=1 chains into one model
ensemble file, and plot data-fit for a sample of the posterior.

Reads `nChains` / `nChainsAtOne` from the main Step-1 pickle.
=====================================================================
"""
import os
import pickle
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('MPLBACKEND'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rjmcmc_inversion import (CombineChains, PlotCSEM_MT_ModelResponsesAndData,
                              _load_input)


FileNameRoot = 's08_TM'
out_dir      = 'Trash'

# --- USER PARAMETERS --------------------------------------------------
burnIn       = 5000     # number of iterations to discard from start of each chain
nthin        = 1        # keep every nthin-th model
NtoCalc      = 200      # how many models to forward-model for the response plot
NtoPlot      = 200      # how many of those to actually draw (sub-sampled)
show_plot    = False
# ---------------------------------------------------------------------


def _auto_register_csem_solver(main_pkl):
    """If the run uses obCSEM (or stCSEM), register empymod via
    csem_forward.register_with_serpent().  Silent no-op for MT-only runs.

    Step4 calls PlotCSEM_MT_ModelResponsesAndData which internally
    re-runs get_field_obCSEM / get_field_stCSEM on the posterior models
    to overlay the response onto the data plot -- without this hook the
    forward solver is not registered and the call raises
    NotImplementedError.
    """
    with open(main_pkl, 'rb') as fh:
        S = pickle.load(fh)
    dt = S.get('dataTypes')
    needs_csem = (dt is not None) and (bool(dt[0]) or bool(dt[1]))
    if needs_csem:
        try:
            import csem_forward
            csem_forward.register_with_serpent()
        except ImportError as e:
            raise RuntimeError(
                "obCSEM/stCSEM run requested but empymod / csem_forward "
                "is unavailable.  Install:  pip install empymod\n"
                f"Underlying error: {e}")


if __name__ == '__main__':
    _auto_register_csem_solver(os.path.join(out_dir, FileNameRoot + '.pkl'))

    # auto-detect chain layout from the Step-1 pickle
    S = _load_input(os.path.join(out_dir, FileNameRoot + '.pkl'))
    nChains      = int(getattr(S, 'nChains', 8))
    nChainsAtOne = int(getattr(S, 'nChainsAtOne',
                       max(1, nChains - int(getattr(S, 'nTemps', 6)) + 1)))

    print(f'Combining {nChainsAtOne} T=1 chains out of {nChains} total chains '
          f'(burnIn={burnIn}, nthin={nthin})...')

    FileName = os.path.join(out_dir, FileNameRoot + '_PT_RJMCMC')
    totalRMS = CombineChains(FileName, burnIn, nthin, nChains, nChainsAtOne)

    # which data types are active?
    dt = list(map(bool, getattr(S, 'dataTypes', [False, False, True, False])))
    stCSEM, obCSEM, MT, DCR = (dt + [False] * 4)[:4]

    print('Plotting model responses...')
    figs = PlotCSEM_MT_ModelResponsesAndData(
        os.path.join(out_dir, FileNameRoot),
        MT=MT, DCR=DCR, stCSEM=stCSEM, obCSEM=obCSEM,
        NtoCalc=NtoCalc, NtoPlot=NtoPlot, seed=0)
    for i, fig in enumerate(figs, start=1):
        path = os.path.join(out_dir, f'{FileNameRoot}_step4_responses{i}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        print(f'  Saved {path}')

    print('Histogram of data-fit across the ensemble...')
    fig_h = plt.figure(figsize=(7, 4.5))
    plt.hist(totalRMS, bins=50, density=True, edgecolor='white', color='steelblue')
    plt.axvline(1.0, color='k', linestyle='--', label='RMS = 1')
    plt.xlabel('RMS misfit'); plt.ylabel('probability density')
    plt.title(f'{FileNameRoot} -- ensemble RMS')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    fig_h.savefig(os.path.join(out_dir, f'{FileNameRoot}_step4_rms_hist.png'),
                  dpi=120, bbox_inches='tight')

    if show_plot:
        plt.show()
    plt.close('all')
    print('Done.  Proceed to Step 5 (posterior plot).')
