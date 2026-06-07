"""
Step2_run.py
=====================================================================
Run the PT-RJMCMC Bayesian inversion.  Reads the three pickles
produced by Step1_initialize.py (MT) or Step1_initialize_obCSEM.py.

For obCSEM runs, empymod is registered automatically as the 1D CSEM
forward solver (replaces the SERPENT MATLAB chain's mexDipole1D).
=====================================================================
"""
import os
import time
import pickle

import rjmcmc_inversion as RJ
from rjmcmc_inversion import PT_RJMCMC


FileNameRoot = 's08_TM'      # set to 'EM5_MT' or 'EM5_obCSEM' etc.
out_dir      = 'Trash'

DataFileMain = os.path.join(out_dir, FileNameRoot + '.pkl')
DataFile1    = os.path.join(out_dir, 'WaterColumn.pkl')
DataFile2    = os.path.join(out_dir, 'Subsurface.pkl')


def _auto_register_csem_solver(main_pkl):
    """If the run uses obCSEM (or stCSEM), register empymod via
    csem_forward.dipole1d_wrapper.  Silent no-op for MT-only runs."""
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
    _auto_register_csem_solver(DataFileMain)
    t0 = time.time()
    PT_RJMCMC(DataFileMain, DataFile1, DataFile2, out_dir, seed=None)
    print(f'Done with PT-RJ-MCMC.  Elapsed: {(time.time()-t0)/60:.1f} min.')
    print('Proceed to Step 3 (assess convergence).')
