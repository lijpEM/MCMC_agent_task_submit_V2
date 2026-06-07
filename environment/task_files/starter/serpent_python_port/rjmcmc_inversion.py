"""
rjmcmc_inversion.py
=====================================================================
Python translation of MATLAB sub-programs for Parallel-Tempering
Reversible-Jump MCMC (PT-RJMCMC) Bayesian inversion of electromagnetic
(MT / CSEM / DCR) geophysical data.

This module packages the following MATLAB sub-programs into a single
Python file:

  Forward modelling
  -----------------
    MT1D.m               -> MT1D(), Z1D()
    Transform01_ab.m     -> Transform01_ab()
    get_fieldMT.m        -> get_fieldMT()
    get_fieldDCR.m       -> get_fieldDCR()
    get_field_obCSEM.m   -> get_field_obCSEM()
    get_field_stCSEM.m   -> get_field_stCSEM()
    getMisfit.m          -> getMisfit()

  PT-RJMCMC engine
  -----------------
    PT_RJMCMC.m          -> PT_RJMCMC()  (+ inner funcs birth/death/
                             move/rhoUpdate/RJ_MCMC_step/determinPerm)

  Post-processing / plotting
  --------------------------
    CombineChains.m              -> CombineChains()
    plotModel1D.m                -> plotModel1D()
    plot_RJMCMC.m                -> plot_RJMCMC()
    plot_RJMCMC_waterColumn.m    -> plot_RJMCMC_waterColumn()
    plot_convergence_PT_RJMCMC.m -> plot_convergence_PT_RJMCMC()
    PlotCSEM_MT_ModelResponsesAndData.m
                                 -> PlotCSEM_MT_ModelResponsesAndData()
    RMShistogramsPlotting.m      -> RMShistogramsPlotting()

External dependencies that were MEX / compiled in MATLAB:
  - Dipole1D  (CSEM 1D dipole forward code, used by get_field_*CSEM)
  - SFilt     (DC-resistivity Schlumberger forward filter)
  You must supply Python equivalents of these two routines (or ctypes
  wrappers around the original Fortran/C libraries) and register them
  via `register_external_solver()` before running CSEM / DCR inversions.
  Stub functions raising NotImplementedError are provided below.

Author: translated from MATLAB originals (Kerry Key / serpent authors).
=====================================================================
"""

from __future__ import annotations

import os
import pickle
from copy import copy, deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.io import loadmat, savemat
except ImportError:  # scipy is optional for pure-algorithm use
    loadmat = savemat = None


# =====================================================================
# 0.  External forward-solver registration
# ---------------------------------------------------------------------
# The original MATLAB code relies on Dipole1D (CSEM) and SFilt (DCR)
# which are compiled MEX routines. Provide your own Python / ctypes
# wrappers and register them here.
# =====================================================================

_DIPOLE1D: Optional[Callable] = None
_SFILT:    Optional[Callable] = None
_PMAX_E:   Optional[Callable] = None


def register_external_solver(dipole1d: Optional[Callable] = None,
                             sfilt:    Optional[Callable] = None,
                             pmax_E:   Optional[Callable] = None) -> None:
    """
    Register user-supplied Python wrappers for the CSEM (Dipole1D) and
    DCR (SFilt) forward solvers, and optionally a Pmax_E (polarisation-
    ellipse amplitude) helper for Mare2DEM Type 41 data.

    Dipole1D signature expected:
        allFields = dipole1d(Tx, freq, model, Rx, hankel_filt, HT_filt,
                             TxLength, nOutputFields)
        -> returns ndarray of shape (nRx, nFieldComponents)
           (column 4 in MATLAB == column 4 in Python-0index == inline Er)
    SFilt signature expected:
        appRes = sfilt(electrode_spacings, layer_thicknesses, rho)
        -> returns 1D ndarray of apparent resistivities.
    pmax_E signature expected:
        amp = pmax_E(Tx, freq, model, Rx)
        -> returns 1D real ndarray of Pmax_E amplitudes (V/m, positive),
           one value per Rx row.
    """
    global _DIPOLE1D, _SFILT, _PMAX_E
    if dipole1d is not None:
        _DIPOLE1D = dipole1d
    if sfilt is not None:
        _SFILT = sfilt
    if pmax_E is not None:
        _PMAX_E = pmax_E


def _dipole1d(*args, **kwargs):
    if _DIPOLE1D is None:
        raise NotImplementedError(
            "Dipole1D is not registered. Call register_external_solver(dipole1d=...)"
        )
    return _DIPOLE1D(*args, **kwargs)


def _sfilt(*args, **kwargs):
    if _SFILT is None:
        raise NotImplementedError(
            "SFilt is not registered. Call register_external_solver(sfilt=...)"
        )
    return _SFILT(*args, **kwargs)


def _pmax_E(*args, **kwargs):
    if _PMAX_E is None:
        raise NotImplementedError(
            "pmax_E is not registered.  Call "
            "register_external_solver(pmax_E=...) or "
            "csem_forward.register_with_serpent()."
        )
    return _PMAX_E(*args, **kwargs)


# =====================================================================
# 1.  Small utilities
# =====================================================================

def _as_ns(d):
    """
    Recursively convert dict / mat_struct / list into SimpleNamespace for
    MATLAB-struct-like attribute access (x.a.b).  Leaves ndarrays, scalars
    and SimpleNamespaces untouched.
    """
    if isinstance(d, SimpleNamespace):
        return d
    # scipy.io.loadmat(struct_as_record=False) returns mat_struct objects
    if hasattr(d, "_fieldnames"):
        return SimpleNamespace(**{n: _as_ns(getattr(d, n)) for n in d._fieldnames})
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _as_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_as_ns(x) for x in d]
    return d


def _load_input(src):
    """
    Generic loader for Step-1-style inputs.  Accepts:
      - a string path to a .pkl file          (preferred, pure Python)
      - a string path to a .mat file          (legacy MATLAB)
      - a dict / SimpleNamespace / mat_struct (in-memory)
    Returns a SimpleNamespace.
    """
    if isinstance(src, (dict, SimpleNamespace)) or hasattr(src, "_fieldnames"):
        return _as_ns(src)
    if isinstance(src, str):
        ext = os.path.splitext(src)[1].lower()
        if ext == ".pkl":
            with open(src, "rb") as fh:
                return _as_ns(pickle.load(fh))
        # default to MAT (covers '.mat' and no-extension legacy paths)
        if loadmat is None:
            raise RuntimeError("scipy.io is required for .mat files")
        return _as_ns(_clean_matlab_dict(
            loadmat(src, squeeze_me=True, struct_as_record=False)))
    raise TypeError(f"_load_input: unsupported source type {type(src).__name__}")


def _hasfield(S, name):
    """Mimic MATLAB isfield(S,'name') — works for SimpleNamespace or dict."""
    if isinstance(S, dict):
        return name in S
    return hasattr(S, name)


def _getfield(S, name, default=None):
    if isinstance(S, dict):
        return S.get(name, default)
    return getattr(S, name, default)


def _setfield(S, name, value):
    if isinstance(S, dict):
        S[name] = value
    else:
        setattr(S, name, value)


def _ismember_legacy_zero(bool_vec):
    """
    MATLAB: [~,pos] = ismember(0, bool_vec, 'legacy')
      bool_vec is a logical (0/1) array, we search for 0 (i.e. False).
      'legacy' returns the *last* index (1-based) where 0 appears,
      or 0 if 0 is not present.
    Returns: 1-based position (integer), 0 if no False was found.
    """
    bool_vec = np.asarray(bool_vec).ravel()
    # positions where element is False (==0)
    idx = np.where(~bool_vec.astype(bool))[0]
    if idx.size == 0:
        return 0
    return int(idx[-1]) + 1  # convert to 1-based


# ---------------------------------------------------------------------
# Depth-dependent maximum-resistivity prior support
# ---------------------------------------------------------------------
def load_rho_max_profile(path):
    """
    Load a two-column ASCII profile  depth_km  log10_rho_max
    (Blatter et al. 2022 'maxrho_1300C_MPT' style) and return a
    (depth_km, log10_rho_max) tuple of float arrays.  Lines starting
    with '#' or '%' are treated as comments.

    The profile gives, for every absolute depth below seafloor, the
    upper bound used by the inversion's uniform-on-log10rho prior.
    Below the largest depth in the file the value is extrapolated as
    a constant equal to the last entry.

    Pass the result into Step1's `S` as

        S['rhoMaxProfile_path'] = 'maxrho_1300C_MPT.txt'

    or directly as

        S['rhoMaxProfile'] = load_rho_max_profile('maxrho_1300C_MPT.txt')
    """
    try:
        arr = np.loadtxt(path, comments=('#', '%'))
    except Exception as exc:
        raise IOError(f'Cannot read rho-max profile {path}: {exc}') from exc
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f'{path}: expected at least 2 columns '
                         f'(depth_km, log10_rho_max); got shape {arr.shape}')
    z_km   = np.ascontiguousarray(arr[:, 0], dtype=float)
    rmax   = np.ascontiguousarray(arr[:, 1], dtype=float)
    order  = np.argsort(z_km)
    z_km, rmax = z_km[order], rmax[order]
    return z_km, rmax


def _rhoMaxAt(z_inv, S):
    """
    Depth-dependent upper bound on log10(rho).

    `z_inv` is in the same coordinate system as `x.z` (the inversion's
    subsurface partition):  log10 of metres if S.logZ is True, else
    metres, in either case measured from the surface (z=0 = sea
    surface for marine data, z=0 = ground surface for land data).

    Returns a scalar if z_inv is scalar, else an ndarray of the same
    shape.  When no profile is configured this is just S.rhoMax
    everywhere.

    The profile is stored in S.rhoMaxProfile as a (z_km, log10_rho_max)
    tuple of 1-D float arrays where z_km is depth BELOW the seafloor
    (or BELOW the surface for land data) in kilometres.  Extrapolation
    beyond the profile is constant (last-value on both ends).
    """
    prof = _getfield(S, 'rhoMaxProfile', None)
    z_arr = np.asarray(z_inv, dtype=float)
    scalar_in = (z_arr.ndim == 0)
    if prof is None:
        if scalar_in:
            return float(S.rhoMax)
        return np.full(z_arr.shape, float(S.rhoMax))

    z_km_p, rmax_p = prof
    rbd = float(_getfield(S, 'regionBoundaryDepth', 0.0))
    if _getfield(S, 'logZ', False):
        z_m_abs = 10.0 ** z_arr
        rbd_m   = 10.0 ** rbd
    else:
        z_m_abs = z_arr
        rbd_m   = rbd
    depth_below_seafloor_km = (z_m_abs - rbd_m) / 1000.0
    out = np.interp(depth_below_seafloor_km, z_km_p, rmax_p,
                    left=rmax_p[0], right=rmax_p[-1])
    # Never let the depth-dependent cap be looser than the scalar one
    # (so existing S.rhoMax acts as an absolute global ceiling).
    out = np.minimum(out, float(S.rhoMax))
    if scalar_in:
        return float(out)
    return out


def _nansum_all(a):
    """MATLAB-style nansum over all elements."""
    a = np.asarray(a, dtype=float)
    return float(np.nansum(a))


# =====================================================================
# 2.  FORWARD MODELLING
# =====================================================================

# ---------------------------- MT1D.m ---------------------------------

def Z1D(rho, h, f):
    """
    Complex surface impedance for a 1D layered resistivity model.
      rho : 1-D ndarray of layer resistivities (Ohm-m)
      h   : 1-D ndarray of layer thicknesses (bottom layer thickness
            ignored, so len(h) may be len(rho) or len(rho)-1)
      f   : 1-D ndarray of frequencies (Hz)
    Returns complex ndarray Z of length len(f).
    """
    rho = np.asarray(rho, dtype=float).ravel()
    h   = np.asarray(h,   dtype=float).ravel()
    f   = np.asarray(f,   dtype=float).ravel()

    mu = 1.256637062e-06
    w  = 2.0 * np.pi * f

    # Bottom layer intrinsic Z
    ki = np.sqrt(-1j * w * mu / rho[-1])
    Z  = w * mu / ki

    # Recursion upward through the model layers
    for i in range(len(rho) - 2, -1, -1):
        ki = np.sqrt(-1j * w * mu / rho[i])
        Zi = w * mu / ki
        Z  = Zi * (Z + Zi * np.tanh(1j * ki * h[i])) / \
                  (Zi + Z  * np.tanh(1j * ki * h[i]))
    return Z


def MT1D(rho, h, f):
    """
    1D MT forward.  Returns (apparent_resistivity, phase_deg, Z).
    """
    Z = Z1D(rho, h, f)
    mu = 4.0 * np.pi * 1e-7
    f = np.asarray(f, dtype=float).ravel()
    app_res = (Z * np.conj(Z)).real / (mu * 2.0 * np.pi * f)
    phase   = 180.0 / np.pi * np.angle(Z)
    return app_res, phase, Z


# -------------------------- Transform01_ab.m -------------------------

def Transform01_ab(x, S):
    """
    Take a standard 1D trans-d layer model with log(rho) values in (0,1),
    bin them in depth and transform to (minRho, maxRho).
    Returns (rhoBinned, zBinned).
    """
    if not _getfield(S, "logZ", False):
        zBinned = np.arange(S.zMin + S.dz / 2.0, S.zMax + 1e-12, S.dz)
    else:
        tmp = np.arange(S.zMin + S.dz / 2.0, S.zMax + 1e-12, S.dz)
        zMin = S.zMin if S.zMin > 0 else 1.0
        zBinned = np.logspace(np.log10(zMin), np.log10(S.zMax), len(tmp))

    nz = len(zBinned)
    rhoBinned = np.zeros(nz)
    zRhoLim = np.asarray(S.zRhoLim).ravel()
    maxRho  = np.asarray(S.maxRho).ravel()
    minRho  = np.asarray(S.minRho).ravel()
    x_z     = np.asarray(x.z).ravel()
    x_rho   = np.asarray(x.rho).ravel()

    for iz in range(nz):
        z_ = zBinned[iz]

        # find first index where zRhoLim > z_  (MATLAB: find(...,1))
        pos1_arr = np.where(zRhoLim > z_)[0]
        if pos1_arr.size == 0:
            pos1 = len(zRhoLim) - 1
        else:
            pos1 = int(pos1_arr[0])
        if pos1 <= 0:
            raise ValueError("Transform01_ab: pos1<=0, check S.zRhoLim / S.zMin")

        frac = (z_ - zRhoLim[pos1 - 1]) / (zRhoLim[pos1] - zRhoLim[pos1 - 1])
        local_rhoMax = maxRho[pos1 - 1] + frac * (maxRho[pos1] - maxRho[pos1 - 1])
        local_rhoMin = minRho[pos1 - 1] + frac * (minRho[pos1] - minRho[pos1 - 1])

        # find first index where x.z > z_
        pos2_arr = np.where(x_z > z_)[0]
        if pos2_arr.size == 0:
            pos2 = len(x_z)          # "beyond last interface"
            if len(x_rho) < pos2 + 1:
                pos2 = len(x_rho) - 1
        else:
            pos2 = int(pos2_arr[0])

        rhoBinned[iz] = local_rhoMin + x_rho[pos2] * (local_rhoMax - local_rhoMin)

    return rhoBinned, zBinned


# ---------------------------- get_fieldMT.m --------------------------

def get_fieldMT(S, x, yespause=False):
    """
    MT 1D forward given composite model x (water + subsurface).
    Returns (appRes_log10, phase_deg, Z).
    """
    x = copy(x)
    S_ = copy(S)

    # Find rbd position in the ORIGINAL log10 space *before* converting:
    # smp.z[nw] is a direct bit-copy of S.regionBoundaryDepth, so the
    # equality test is bit-exact.  Converting both to linear first would
    # fail by 1 ULP because numpy uses a different pow routine for scalar
    # `**` than for vector `**` on non-trivial log10 values like log10(3000).
    x_z_raw = np.asarray(x.z, dtype=float).ravel()
    if _getfield(S_, "logZ", False):
        pos_arr = np.where(x_z_raw == S_.regionBoundaryDepth)[0]
    else:
        pos_arr = np.where(x_z_raw == S_.regionBoundaryDepth)[0]
    if pos_arr.size == 0:
        # FP fall-back; should never trigger but kept for safety
        pos_arr = np.where(np.isclose(x_z_raw, S_.regionBoundaryDepth,
                                      rtol=1e-9, atol=1e-12))[0]
    pos = int(pos_arr[0]) if pos_arr.size else 0   # 0-indexed

    if _getfield(S_, "logZ", False):
        x.z = 10.0 ** x_z_raw
        S_.regionBoundaryDepth = 10.0 ** S_.regionBoundaryDepth
    else:
        x.z = x_z_raw

    # re-zero the model to z=0 at the sea floor; neglect water column
    x_z = np.asarray(x.z).ravel()

    x.z    = x_z[pos + 1:] - S_.regionBoundaryDepth
    # ---- LAND-DATA FIX ------------------------------------------------
    # For marine data the composite is [water_rhoh ... | sub_rhoh ...] and
    # stripping pos+1 leaves exactly the sub_rhoh array (length k2+1).
    # For land data x_water is empty, so the composite.rhoh == sub_rhoh
    # entirely and stripping pos+1 would silently drop the topmost
    # subsurface resistivity, causing the forward model to use the
    # wrong number of layers.  Keep all rhoh in the land case.
    if _getfield(S_, "landData", False):
        x.rhoh = np.asarray(x.rhoh).ravel()[pos:]
    else:
        x.rhoh = np.asarray(x.rhoh).ravel()[pos + 1:]

    if _getfield(S_, "transform01_ab", False):
        if _getfield(S_, "logZ", False):
            S_.zMin = 10.0 ** S_.zMin
            S_.zMax = 10.0 ** S_.zMax
        S_.zMin = S_.zMin - S_.regionBoundaryDepth
        S_.zMax = S_.zMax - S_.regionBoundaryDepth
        x.rho = x.rhoh
        rhoBinned, zBinned = Transform01_ab(x, S_)
        x.rhoh = np.concatenate([rhoBinned, rhoBinned[-1:]])
        x.z    = zBinned

    rho = 10.0 ** np.asarray(x.rhoh, dtype=float).ravel()

    z = np.concatenate([[0.0], np.asarray(x.z, dtype=float).ravel()])
    h = np.diff(z)

    freqs = np.asarray(S_.MTdat.freqs).ravel()
    appRes, phase, Z = MT1D(rho, h, freqs)
    appRes = np.log10(appRes)

    if yespause:
        _plot_mt_diag(freqs, appRes, phase, S_.MTdat)

    return appRes, phase, Z


def _plot_mt_diag(freqs, appRes, phase, MTdat):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(1.0 / freqs, appRes, 'bo-', linewidth=2)
    plt.semilogx(1.0 / freqs, MTdat.TEappRes, 'ko-', linewidth=2)
    plt.title('Apparent Resistivity'); plt.ylabel('ohm-m')
    plt.subplot(2, 1, 2)
    plt.semilogx(1.0 / freqs, phase, 'bo-', linewidth=2)
    plt.semilogx(1.0 / freqs, MTdat.TEphase, 'ko-', linewidth=2)
    plt.show()


# --------------------------- get_fieldDCR.m --------------------------

def get_fieldDCR(S, x, yespause=False):
    """
    DC-resistivity Schlumberger apparent resistivity for a 1D model.
    Requires user-registered `sfilt` external solver.
    """
    x  = deepcopy(x)
    S_ = deepcopy(S)

    # Find rbd position in the ORIGINAL log10 space *before* converting
    # (see get_fieldMT for full rationale of the FP-mismatch issue).
    x_z_raw = np.asarray(x.z, dtype=float).ravel()
    pos_arr = np.where(x_z_raw == S_.regionBoundaryDepth)[0]
    if pos_arr.size == 0:
        pos_arr = np.where(np.isclose(x_z_raw, S_.regionBoundaryDepth,
                                      rtol=1e-9, atol=1e-12))[0]
    pos = int(pos_arr[0]) if pos_arr.size else 0

    if _getfield(S_, "logZ", False):
        x.z = 10.0 ** x_z_raw
        S_.regionBoundaryDepth = 10.0 ** S_.regionBoundaryDepth
        S_.zMax                = 10.0 ** S_.zMax
    else:
        x.z = x_z_raw

    x_z = np.asarray(x.z).ravel()
    x.z    = x_z[pos + 1:] - S_.regionBoundaryDepth
    # land-data fix: same as get_fieldMT — don't drop sub_rhoh[0] when
    # the composite has no water-column entries above rbd.
    if _getfield(S_, "landData", False):
        x.rhoh = np.asarray(x.rhoh).ravel()[pos:]
    else:
        x.rhoh = np.asarray(x.rhoh).ravel()[pos + 1:]

    if _getfield(S_, "transform01_ab", False):
        if _getfield(S_, "logZ", False):
            S_.zMin = 10.0 ** S_.zMin
            S_.zMax = 10.0 ** S_.zMax
        S_.zMin = S_.zMin - S_.regionBoundaryDepth
        S_.zMax = S_.zMax - S_.regionBoundaryDepth
        x.rho = x.rhoh
        rhoBinned, zBinned = Transform01_ab(x, S_)
        x.rhoh = np.concatenate([rhoBinned, rhoBinned[-1:]])
        x.z    = zBinned

    es = np.asarray(S_.dcrDat.es).ravel()
    z  = np.concatenate([[0.0], np.asarray(x.z).ravel(), [S_.zMax]])
    h  = np.diff(z)
    rho = np.asarray(x.rhoh).ravel()

    resp = _sfilt(es, h, rho)
    resp = np.asarray(resp).ravel()
    if np.any(np.iscomplex(resp)):
        resp = np.abs(resp)
    return resp


# ------------------------ get_field_stCSEM.m -------------------------

def get_field_stCSEM(S, x, yespause=False):
    """
    Surface-towed CSEM inline Er response.  Requires registered Dipole1D.
    Returns ErResponse shape (nFreq, nRx, nSoundings) complex.
    """
    x  = deepcopy(x)
    nFreq      = len(np.asarray(S.stDat.Freqs).ravel())
    nRx        = np.asarray(S.stRx.X).shape[0]
    nSound     = len(np.asarray(S.stTx.Soundings).ravel())
    Er = np.zeros((nFreq, nRx, nSound), dtype=complex)

    if _getfield(S, "logZ", False):
        x.z = 10.0 ** np.asarray(x.z, dtype=float)

    z   = np.concatenate([[-100e3, 0.0], np.asarray(x.z).ravel()])
    rho = np.concatenate([[1e12], 10.0 ** np.asarray(x.rhoh).ravel()])
    model = np.column_stack([z, rho])

    freqs = np.asarray(S.stDat.Freqs).ravel()
    for j in range(nSound):
        Rx = np.column_stack([S.stRx.X[:, j], S.stRx.Y[:, j], S.stRx.Z[:, j]])
        Tx = np.array([S.stTx.X[j], S.stTx.Y[j], S.stTx.Z[j],
                       S.stTx.Azimuth, S.stTx.Dip])
        for k, fk in enumerate(freqs):
            allFields = _dipole1d(Tx, fk, model, Rx, 0, 0, S.stTx.Length, 3)
            # column index 4 in MATLAB (1-based #5) == inline Er
            Er[k, :, j] = np.asarray(allFields)[:, 4]
    return Er


# ------------------------ get_field_obCSEM.m -------------------------

def get_field_obCSEM(S, x, yespause=False):
    """
    Ocean-bottom CSEM forward.  Returns complex inline E-field by
    default (SERPENT convention).  If the registered Dipole1D wrapper
    is `csem_forward.dipole1d_wrapper` and the obDat object has the
    flag `obDat.observable == 'Pmax_E'`, we instead return the
    polarisation-ellipse Pmax (real-valued amplitude in V/m, packed
    as a real array with zero imaginary part for getMisfit
    compatibility).
    """
    x = deepcopy(x)
    nFreq  = len(np.asarray(S.obDat.Freqs).ravel())
    nRx    = np.asarray(S.obRx.X).shape[0]
    nSound = len(np.asarray(S.obTx.Soundings).ravel())
    observable = _getfield(S.obDat, 'observable', 'inline_E')
    is_pmax    = (observable == 'Pmax_E')

    if is_pmax:
        Er = np.zeros((nFreq, nRx, nSound), dtype=float)
    else:
        Er = np.zeros((nFreq, nRx, nSound), dtype=complex)

    if _getfield(S, "logZ", False):
        x.z = 10.0 ** np.asarray(x.z, dtype=float)

    z   = np.concatenate([[-100e3, 0.0], np.asarray(x.z).ravel()])
    rho = np.concatenate([[1e12], 10.0 ** np.asarray(x.rhoh).ravel()])
    model = np.column_stack([z, rho])

    freqs = np.asarray(S.obDat.Freqs).ravel()
    TxLength = float(_getfield(S.obTx, 'Length', 0.0))
    for j in range(nSound):
        Rx = np.column_stack([S.obRx.X[:, j], S.obRx.Y[:, j], S.obRx.Z[:, j]])
        Tx = np.array([S.obTx.X, S.obTx.Y, S.obTx.Z,
                       S.obTx.Azimuth, S.obTx.Dip])
        if is_pmax:
            # **vectorised over frequency**: empymod.bipole accepts an
            # array of freqs in `freqtime` and returns (nFreq, nRx) in
            # ONE Hankel-transform call.  This is ~2-3x faster than
            # the old per-frequency loop because the kernel evaluation
            # is shared across freqs.  Saves several hours over a 1e5
            # iter run when nFreq = 3.
            Er[:, :, j] = _pmax_E(Tx, freqs, model, Rx,
                                  TxLength=TxLength)
        else:
            # Dipole1D inline-E case: empymod's wrapper still loops over
            # freqs internally; we keep the explicit loop here for
            # backward compatibility with custom `dipole1d` wrappers
            # registered via register_external_solver().
            for k, fk in enumerate(freqs):
                allFields = _dipole1d(Tx, fk, model, Rx, 0, 0,
                                      TxLength, 3)
                Er[k, :, j] = np.asarray(allFields)[:, 4]
    return Er


# ---------------------------- getMisfit.m ----------------------------

def getMisfit(x, S, yespause=False):
    """
    Compute the total data misfit (Chi^2 / 2, RMS) and per-data-type RMS
    vector. `S.dataTypes` is a 4-tuple (stCSEM, obCSEM, MT, DCR).
    Returns (misfit[2], misfitVect).
    """
    if _getfield(S, "debug_prior", False):
        return np.array([0.0, 0.0]), np.zeros(int(np.sum(np.asarray(S.dataTypes))))

    Chi2 = 0.0
    N    = 0
    stCSEMchi2 = obCSEMchi2 = MTchi2 = DCRchi2 = 0.0
    misfitVect: List[float] = []

    dt = np.asarray(S.dataTypes).astype(bool).ravel()

    # 1) surface-towed CSEM
    if len(dt) >= 1 and dt[0]:
        QQ = get_field_stCSEM(S, x, yespause)
        ErResp    = np.log10(np.abs(QQ))
        PhaseResp = (180.0 / np.pi) * np.arctan2(QQ.imag, QQ.real)
        # Squeeze trailing nSound dim if it's 1 (the typical case)
        if ErResp.ndim == 3 and ErResp.shape[-1] == 1:
            ErResp    = ErResp[..., 0]
            PhaseResp = PhaseResp[..., 0]
        er_err = ((np.asarray(S.stDat.Er)    - ErResp)    / np.asarray(S.stDat.ErErr))    ** 2
        ph_err = ((np.asarray(S.stDat.Phase) - PhaseResp) / np.asarray(S.stDat.PhaseErr)) ** 2
        N1 = np.sum(~np.isnan(er_err)) + np.sum(~np.isnan(ph_err))
        N += int(N1)
        stCSEMchi2 = _nansum_all(er_err) + _nansum_all(ph_err)
        Chi2 += stCSEMchi2
        misfitVect.append(np.sqrt(stCSEMchi2 / N1))

    # 2) ocean-bottom CSEM
    if len(dt) >= 2 and dt[1]:
        QQ = get_field_obCSEM(S, x, yespause)
        observable    = _getfield(S.obDat, 'observable',    'inline_E')
        amplitudeOnly = bool(_getfield(S.obDat, 'amplitudeOnly', False))
        if observable == 'Pmax_E':
            # QQ is already real-valued amplitude (V/m, positive)
            ErResp    = np.log10(np.maximum(np.asarray(QQ, dtype=float), 1e-300))
            PhaseResp = np.zeros_like(ErResp)        # no phase observable
        else:
            ErResp    = np.log10(np.abs(QQ))
            PhaseResp = (180.0 / np.pi) * np.arctan2(QQ.imag, QQ.real)
        # Squeeze trailing nSound dim if it's 1, so shape matches the
        # 2D data matrices (nFreq, nRx).  The reciprocity-swap workflow
        # always has nSound = 1.
        if ErResp.ndim == 3 and ErResp.shape[-1] == 1:
            ErResp    = ErResp[..., 0]
            PhaseResp = PhaseResp[..., 0]
        er_err = ((np.asarray(S.obDat.Er) - ErResp) / np.asarray(S.obDat.ErErr)) ** 2
        if amplitudeOnly:
            ph_err = np.zeros_like(er_err)       # contributes 0 to chi2 and N
            N2     = int(np.sum(~np.isnan(er_err)))
            obCSEMchi2 = float(_nansum_all(er_err))
        else:
            ph_err = ((np.asarray(S.obDat.Phase) - PhaseResp) / np.asarray(S.obDat.PhaseErr)) ** 2
            N2     = int(np.sum(~np.isnan(er_err)) + np.sum(~np.isnan(ph_err)))
            obCSEMchi2 = float(_nansum_all(er_err) + _nansum_all(ph_err))
        N += int(N2)
        Chi2 += obCSEMchi2
        misfitVect.append(np.sqrt(obCSEMchi2 / max(N2, 1)))

    # 3) MT
    if len(dt) >= 3 and dt[2]:
        appRes, phase, _ = get_fieldMT(S, x, yespause)
        appRes_err = ((appRes - np.asarray(S.MTdat.TEappRes))  / np.asarray(S.MTdat.TEappResErr)) ** 2
        phase_err  = ((phase  - np.asarray(S.MTdat.TEphase))   / np.asarray(S.MTdat.TEphaseErr)) ** 2
        N3 = len(appRes_err) + len(phase_err)
        MTchi2 = float(np.sum(appRes_err) + np.sum(phase_err))
        Chi2 += MTchi2
        N += N3
        misfitVect.append(np.sqrt(MTchi2 / N3))

    # 4) DCR
    if len(dt) >= 4 and dt[3]:
        DCRappRes = get_fieldDCR(S, x, yespause)
        a_err = ((DCRappRes - np.asarray(S.dcrDat.appRes)) / np.asarray(S.dcrDat.appResErr)) ** 2
        N4 = len(a_err)
        N += N4
        DCRchi2 = float(np.sum(a_err))
        Chi2 += DCRchi2
        misfitVect.append(np.sqrt(DCRchi2 / N4))

    total_chi2_rms = np.sqrt((stCSEMchi2 + obCSEMchi2 + MTchi2 + DCRchi2) / max(N, 1))
    return np.array([Chi2 / 2.0, total_chi2_rms]), np.asarray(misfitVect)


# =====================================================================
# 3.  PT-RJMCMC ENGINE
# =====================================================================

# ---------------------------- inner moves ----------------------------

def _birth(k, x, S, rng):
    """
    Add a new interface drawn from the prior (or Gaussian proposal).
    Returns (pertNorm, xNew, priorViolate).  xNew is a SimpleNamespace.
    """
    xNew = SimpleNamespace(z=np.full(k + 1, np.nan),
                           rhoh=np.full(k + 2, np.nan),
                           rhov=np.full(k + 2, np.nan))
    pertNorm = 0.0
    priorViolate = 0

    # propose new interface depth
    zProp = S.zMin + rng.random() * (S.zMax - S.zMin)

    x_z    = np.asarray(x.z).ravel()
    x_rhoh = np.asarray(x.rhoh).ravel()
    x_rhov = np.asarray(x.rhov).ravel()

    # MATLAB: [~,pos]=ismember(0,(x.z >= zProp),'legacy') ->
    # last layer above zProp (1-based); 0 if all x.z >= zProp
    pos = _ismember_legacy_zero(x_z >= zProp)   # 1-based

    if _getfield(S, "birth_death_from_prior", False):
        unifDraw = rng.random(2)
        if _getfield(S, "isotropic", True):
            unifDraw[0] = unifDraw[1]
        # depth-dependent upper bound (falls back to S.rhoMax if no profile)
        rmax_local = _rhoMaxAt(zProp, S)
        newRho = S.rhoMin + unifDraw * (rmax_local - S.rhoMin)
    else:
        mu = np.array([x_rhoh[pos], x_rhov[pos]])  # pos+1 in MATLAB (1-based) = pos in 0-based
        normDraw = rng.standard_normal(2)
        if _getfield(S, "isotropic", True):
            normDraw[0] = normDraw[1]
            mu[0] = mu[1]
        newRho = mu + normDraw * S.rSD2
        pertNorm = float(np.sum(normDraw ** 2))
        rmax_local = _rhoMaxAt(zProp, S)

    if (newRho[0] < S.rhoMin or newRho[0] > rmax_local or
        newRho[1] < S.rhoMin or newRho[1] > rmax_local):
        priorViolate = 1
        return pertNorm, xNew, priorViolate

    # insert zProp at position 'pos' (0-based: pos), then NaN-fill in old values
    # MATLAB: xNew.z(pos+1) = zProp; xNew.z(isnan(xNew.z)) = x.z;
    xNew.z[pos] = zProp
    nanmask = np.isnan(xNew.z)
    xNew.z[nanmask] = x_z

    # pos = pos+1 with 50% probability (in MATLAB that means put rho in layer below)
    if rng.random() < 0.5:
        pos = pos + 1

    xNew.rhoh[pos] = newRho[0]
    mask_h = np.isnan(xNew.rhoh)
    xNew.rhoh[mask_h] = x_rhoh

    xNew.rhov[pos] = newRho[1]
    mask_v = np.isnan(xNew.rhov)
    xNew.rhov[mask_v] = x_rhov

    return pertNorm, xNew, priorViolate


def _death(x, k, S, rng):
    """Delete a randomly-chosen interface. Returns (pertNorm, xNew)."""
    xNew = SimpleNamespace(z=np.asarray(x.z, dtype=float).copy(),
                           rhoh=np.asarray(x.rhoh, dtype=float).copy(),
                           rhov=np.asarray(x.rhov, dtype=float).copy())
    # pick interface 1..k (1-based) -> 0..k-1 (0-based)
    l = int(np.floor(rng.random() * k))  # 0-based
    if _getfield(S, "isotropic", True):
        # enforce H = V on the *removed* layer for consistency
        xNew.rhoh[l]     = xNew.rhov[l]
        xNew.rhoh[l + 1] = xNew.rhov[l + 1]
    pertNorm = float(np.sum((
        np.array([xNew.rhoh[l] - xNew.rhoh[l + 1],
                  xNew.rhov[l] - xNew.rhov[l + 1]]) / S.rSD2) ** 2))

    # delete interface z(l)
    xNew.z = np.delete(xNew.z, l)
    # possibly bump l to l+1 for rho deletion
    if rng.random() < 0.5:
        l = l + 1
    xNew.rhoh = np.delete(xNew.rhoh, l)
    xNew.rhov = np.delete(xNew.rhov, l)
    return pertNorm, xNew


def _move(x, k, S, rng):
    """Move an interface. Returns (xNew, priorViolate)."""
    x_z    = np.asarray(x.z,    dtype=float).copy()
    x_rhoh = np.asarray(x.rhoh, dtype=float).copy()
    x_rhov = np.asarray(x.rhov, dtype=float).copy()

    l = int(np.floor(rng.random() * k))          # 0-based index in [0, k-1]
    zProp = x_z[l] + S.MoveSd * rng.standard_normal()
    if zProp < S.zMin or zProp > S.zMax:
        xNew = SimpleNamespace(z=x_z, rhoh=x_rhoh, rhov=x_rhov)
        return xNew, 1

    priorViolate = 0
    # Where does zProp fall now (in the current first k interfaces)?
    pos = _ismember_legacy_zero(x_z[:k] >= zProp)   # 1-based; 0 if none
    # MATLAB: if pos==l || pos==l-1 -> no shift needed
    # MATLAB l is 1-based; l (here, 0-based) -> MATLAB_l = l+1
    Ml = l + 1
    if pos == Ml or pos == Ml - 1:
        xNew = SimpleNamespace(z=x_z.copy(), rhoh=x_rhoh.copy(), rhov=x_rhov.copy())
        xNew.z[l] = zProp
        return xNew, 0

    # else: remove interface l, delete rho above or below, re-insert at new position
    x_z_r    = np.delete(x_z,    l)
    kk = k - 1
    if rng.random() < 0.5:
        rho_idx_to_remove = l + 1
    else:
        rho_idx_to_remove = l
    temp_rhoh = x_rhoh[rho_idx_to_remove]
    temp_rhov = x_rhov[rho_idx_to_remove]
    x_rhoh_r = np.delete(x_rhoh, rho_idx_to_remove)
    x_rhov_r = np.delete(x_rhov, rho_idx_to_remove)

    # birth at zProp
    xNew = SimpleNamespace(z=np.full(kk + 1, np.nan),
                           rhoh=np.full(kk + 2, np.nan),
                           rhov=np.full(kk + 2, np.nan))
    pos2 = _ismember_legacy_zero(x_z_r[:kk] >= zProp)  # 1-based
    xNew.z[pos2] = zProp
    xNew.z[np.isnan(xNew.z)] = x_z_r

    pos_rho = pos2
    if rng.random() < 0.5:
        pos_rho = pos2 + 1

    xNew.rhoh[pos_rho] = temp_rhoh
    xNew.rhoh[np.isnan(xNew.rhoh)] = x_rhoh_r
    xNew.rhov[pos_rho] = temp_rhov
    xNew.rhov[np.isnan(xNew.rhov)] = x_rhov_r

    return xNew, priorViolate


def _rhoUpdate(k, x, S, smallFlag, rng):
    """Perturb all layer resistivities with a Gaussian update. Returns (xNew, priorViolate)."""
    rSD1 = S.rSD1
    if smallFlag == 'small':
        rSD1 = S.rSD1 / _getfield(S, "localFac", 1.0)
    xNew = SimpleNamespace(z=np.asarray(x.z, dtype=float).copy(),
                           rhoh=np.asarray(x.rhoh, dtype=float).copy(),
                           rhov=np.asarray(x.rhov, dtype=float).copy())
    normDrawH = rng.standard_normal(k + 1)
    normDrawV = rng.standard_normal(k + 1)
    newRhoH = xNew.rhoh + normDrawH * rSD1
    newRhoV = xNew.rhov + normDrawV * rSD1
    if _getfield(S, "isotropic", True):
        newRhoH = newRhoV

    # Depth-dependent upper bound: for each layer use the BOTTOM-of-layer
    # depth (the deepest, coldest -> tightest cap).  Layer i has bottom
    # x.z[i] for i = 0..k-1, and zMax for the deepest layer i = k.
    # When no profile is registered _rhoMaxAt() returns the scalar S.rhoMax
    # so the comparison degrades cleanly to the legacy [rhoMin, rhoMax]
    # check.
    z_arr = np.asarray(x.z, dtype=float).ravel()
    if z_arr.size > 0:
        layer_bottoms = np.concatenate([z_arr, [float(S.zMax)]])
    else:
        layer_bottoms = np.array([float(S.zMax)])
    caps = _rhoMaxAt(layer_bottoms, S)
    if np.any(newRhoH < S.rhoMin) or np.any(newRhoH > caps) \
       or np.any(newRhoV < S.rhoMin) or np.any(newRhoV > caps):
        return xNew, 1
    xNew.rhoh = newRhoH
    xNew.rhov = newRhoV
    return xNew, 0


def _determinPerm(n, rng):
    """
    Random permutation of upper-triangular index pairs (MATLAB determinPerm).
    Returns (p, q) 1-based arrays each of length n*(n-1)/2.
    Note: original MATLAB has `n/2*(n-1)` — for odd n this is exact only as
    integer arithmetic in double; we coerce to int.
    """
    m = n * (n - 1) // 2
    k = rng.permutation(m) + 1           # 1-based like MATLAB
    q = np.floor(np.sqrt(8 * (k - 1) + 1) / 2 + 3 / 2).astype(int)
    p = (k - (q - 1) * (q - 2) // 2).astype(int)
    return p, q


def _composite(x_water, x_sub, regionBoundaryDepth):
    """Concatenate the water-column and subsurface models into a full model."""
    z1 = np.asarray(x_water.z, dtype=float).ravel()
    z2 = np.asarray(x_sub.z,   dtype=float).ravel()
    x = SimpleNamespace()
    x.z    = np.concatenate([z1, [regionBoundaryDepth], z2])
    x.rhov = np.concatenate([np.asarray(x_water.rhov).ravel(),
                             np.asarray(x_sub.rhov).ravel()])
    x.rhoh = np.concatenate([np.asarray(x_water.rhoh).ravel(),
                             np.asarray(x_sub.rhoh).ravel()])
    return x


def _compose_for_active(x_active, x_passive, S_0):
    """
    When we perturb x_active, build the composite model for forward calls.
    If active lives above the regionBoundaryDepth, put active first.
    Mirrors the original MATLAB conditional on x.z(end) vs regionBoundaryDepth.
    """
    z_a = np.asarray(x_active.z, dtype=float).ravel()
    # MATLAB uses the *old* x.z(end) to decide which side the active block is on
    x_composite = SimpleNamespace()
    rbd = S_0.regionBoundaryDepth
    if z_a.size and z_a[-1] < rbd:
        # active is water column (above seafloor)
        x_composite.z    = np.concatenate([z_a, [rbd],
                                           np.asarray(x_passive.z).ravel()])
        x_composite.rhov = np.concatenate([np.asarray(x_active.rhov).ravel(),
                                           np.asarray(x_passive.rhov).ravel()])
        # NOTE: this mirrors the original MATLAB line, which uses rhov for rhoh
        # (see the original code): [ xNew.rhov x_passive.rhoh ]
        # This is a faithful reproduction of MATLAB behaviour; it has no
        # effect when isotropic=True since rhov == rhoh in that case.
        x_composite.rhoh = np.concatenate([np.asarray(x_active.rhov).ravel(),
                                           np.asarray(x_passive.rhoh).ravel()])
    else:
        x_composite.z    = np.concatenate([np.asarray(x_passive.z).ravel(),
                                           [rbd], z_a])
        x_composite.rhov = np.concatenate([np.asarray(x_passive.rhov).ravel(),
                                       np.asarray(x_active.rhov).ravel()])
        # NOTE: mirrors MATLAB which uses rhov for rhoh (no effect when isotropic=True)
        x_composite.rhoh = np.concatenate([np.asarray(x_passive.rhoh).ravel(),
                                           np.asarray(x_active.rhov).ravel()])
    return x_composite


def _rj_mcmc_step(x, k, x_passive, oldMisfit, oldMisfitVect, ConvStat,
                  S, S_0, B, yespause, rng):
    """One RJ-MCMC move (birth/death/move/update) at inverse-temperature B."""
    del_ = S.rhoMax - S.rhoMin
    dart = rng.random()
    CDF  = np.array([0.25, 0.5, 0.75, 1.0])
    # MATLAB: [~,pos]=ismember(0,(CDF>=dart),'legacy'); pos=pos+1;
    pos = _ismember_legacy_zero(CDF >= dart) + 1    # becomes 1..4

    priorViolate = 0
    if _getfield(S_0, "fixedDimension", False):
        pos = 4  # force update-only

    newMisfit      = oldMisfit
    newMisfitVect  = oldMisfitVect
    logalpha       = -np.inf

    if pos == 1:  # birth
        ConvStat["bc"] += 1
        if k == S.kMax:
            priorViolate = 1
        if not priorViolate:
            pertNorm, xNew, priorViolate = _birth(k, x, S, rng)
        if not priorViolate:
            x_composite = _compose_for_active(xNew, x_passive, S_0)
            newMisfit, newMisfitVect = getMisfit(x_composite, S_0, yespause)
            ConvStat["evalCount"] += 1
            if not _getfield(S, "birth_death_from_prior", False):
                logalpha = 2 * np.log(S.rSD2) + np.log(2 * np.pi) \
                           - 2 * np.log(del_) + (pertNorm / 2.0)
                if _getfield(S, "isotropic", True):
                    logalpha *= 0.5
                if _getfield(S_0, "jeffereys_prior", False):
                    logalpha += np.log(k) - np.log(k + 1)
                logalpha -= (newMisfit[0] - oldMisfit[0]) * B
            else:
                logalpha = -(newMisfit[0] - oldMisfit[0]) * B
        if np.log(rng.random()) < logalpha:
            k += 1
            x  = xNew
            oldMisfit     = newMisfit
            oldMisfitVect = newMisfitVect
            ConvStat["bA"] += 1

    elif pos == 2:  # death
        ConvStat["dc"] += 1
        if k == S.kMin:
            priorViolate = 1
        if not priorViolate:
            pertNorm, xNew = _death(x, k, S, rng)
            x_composite = _compose_for_active(xNew, x_passive, S_0)
            newMisfit, newMisfitVect = getMisfit(x_composite, S_0, yespause)
            ConvStat["evalCount"] += 1
            if not _getfield(S, "birth_death_from_prior", False):
                logalpha = -2 * np.log(S.rSD2) - np.log(2 * np.pi) \
                           + 2 * np.log(del_) - (pertNorm / 2.0)
                if _getfield(S, "isotropic", True):
                    logalpha *= 0.5
                if _getfield(S_0, "jeffereys_prior", False):
                    logalpha += np.log(k) - np.log(k - 1)
                logalpha -= (newMisfit[0] - oldMisfit[0]) * B
            else:
                logalpha = -(newMisfit[0] - oldMisfit[0]) * B
        if np.log(rng.random()) < logalpha:
            k -= 1
            x  = xNew
            oldMisfit     = newMisfit
            oldMisfitVect = newMisfitVect
            ConvStat["dA"] += 1

    elif pos == 3:  # move
        ConvStat["mc"] += 1
        xNew, priorViolate = _move(x, k, S, rng)
        if not priorViolate:
            x_composite = _compose_for_active(xNew, x_passive, S_0)
            newMisfit, newMisfitVect = getMisfit(x_composite, S_0, yespause)
            ConvStat["evalCount"] += 1
            logalpha = -(newMisfit[0] - oldMisfit[0]) * B
        if np.log(rng.random()) < logalpha:
            x = xNew
            oldMisfit     = newMisfit
            oldMisfitVect = newMisfitVect
            ConvStat["mA"] += 1

    elif pos == 4:  # update
        ConvStat["uc"] += 1
        xNew, priorViolate = _rhoUpdate(k, x, S, 'large', rng)
        if not priorViolate:
            x_composite = _compose_for_active(xNew, x_passive, S_0)
            newMisfit, newMisfitVect = getMisfit(x_composite, S_0, yespause)
            ConvStat["evalCount"] += 1
            logalpha = -(newMisfit[0] - oldMisfit[0]) * B
        if np.log(rng.random()) < logalpha:
            x = xNew
            oldMisfit     = newMisfit
            oldMisfitVect = newMisfitVect
            ConvStat["uA"] += 1

    return x, k, oldMisfit, oldMisfitVect, ConvStat


def _dummy_ar():
    return dict(uAR=0.0, bAR=0.0, dAR=0.0, mAR=0.0,
                TotalAR=0.0, evalCount=0, swapRate=0.0)


def _dummy_convstat():
    return dict(uA=0, bA=0, dA=0, mA=0,
                uc=0, bc=0, dc=0, mc=0, evalCount=0)


# ---------------------------------------------------------------------
# Multi-chain parallelisation infrastructure
# ---------------------------------------------------------------------
# Module-level globals populated by PT_RJMCMC before forking workers.
# After fork(), workers inherit these by COW, so a single per-call
# pickle only needs to carry the per-chain state (x1, x2, k1, k2,
# oldMisfit, ...) which is at most a few KB.  S_0, S_1, S_2 and the
# registered forward solvers (_DIPOLE1D, _PMAX_E, _SFILT) live in
# the inherited memory and are NOT pickled per call.
_PT_S_0       = None
_PT_S_1_LIST  = None    # list[nChains] of per-chain S_1
_PT_S_2_LIST  = None    # list[nChains] of per-chain S_2
_PT_yespause  = False


def _chain_step_worker(jj, x1_jj, x2_jj, k1_jj, k2_jj,
                       oldMisfit_jj, oldMisfitVect_jj,
                       ConvStat1_jj, ConvStat2_jj,
                       B_jj, rng_state):
    """Run one MCMC step for chain `jj`. Designed for multiprocessing.

    Lives at module level so it is picklable. Reads the (large, static)
    inversion state -- S_0, S_1[jj], S_2[jj], and the registered forward
    solvers -- from module globals inherited via fork(), so the only
    per-call pickle payload is the small per-chain state (x1[jj]
    etc., ~1-2 KB).

    Returns the updated chain state plus the new RNG state, all
    picklable.  The main process re-assigns the state in place and
    records `samples`, `kTracker`, `en`, etc. in the parent.
    """
    # Rehydrate the chain's RNG from its serialised state.  Using
    # per-chain RNGs keeps Markov-chain decorrelation correct under
    # parallel execution and lets the user reproduce a run with the
    # same `seed` as long as n_workers stays the same.
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    S_0     = _PT_S_0
    S_1_jj  = _PT_S_1_LIST[jj]
    S_2_jj  = _PT_S_2_LIST[jj]
    yespause = _PT_yespause

    dart = rng.random()
    if dart < _getfield(S_0, "beta", 0.0):
        # perturb water column
        x1_new, k1_new, om_new, omv_new, cs1_new = _rj_mcmc_step(
            x1_jj, k1_jj, x2_jj, oldMisfit_jj, oldMisfitVect_jj,
            ConvStat1_jj, S_1_jj, S_0, B_jj, yespause, rng)
        x2_new, k2_new, cs2_new = x2_jj, k2_jj, ConvStat2_jj
    else:
        # perturb subsurface
        x2_new, k2_new, om_new, omv_new, cs2_new = _rj_mcmc_step(
            x2_jj, k2_jj, x1_jj, oldMisfit_jj, oldMisfitVect_jj,
            ConvStat2_jj, S_2_jj, S_0, B_jj, yespause, rng)
        x1_new, k1_new, cs1_new = x1_jj, k1_jj, ConvStat1_jj

    return (x1_new, x2_new, k1_new, k2_new,
            om_new, omv_new, cs1_new, cs2_new,
            rng.bit_generator.state)


def PT_RJMCMC(DataFileMain, DataFile1, DataFile2, outputFolder,
              restart=False, seed=None):
    """
    Parallel-Tempering Reversible-Jump MCMC driver.
    Faithful Python translation of PT_RJMCMC.m.
    Inputs (each) may be:
        - a path to a .pkl file     (preferred, produced by Step1_initialize.py)
        - a path to a .mat file     (legacy MATLAB Step1 output)
        - a dict / SimpleNamespace  (in-memory)
    Results are pickled to outputFolder/<FileRoot>_PT_RJMCMC_<chain>.pkl
    (one file per chain).
    """
    S_0 = _load_input(DataFileMain)
    S_1 = _load_input(DataFile1)
    S_2 = _load_input(DataFile2)

    if not _hasfield(S_0, "landData"):
        print("**** Assuming marine data. Set S.landData in Step1 for land. ****")
        S_0.landData = False; S_1.landData = False; S_2.landData = False

    landData = bool(S_0.landData)
    if landData:
        print("You are using land data")
        S_0.beta = -1

    yespause = False

    # defaults
    if not _hasfield(S_0, "debug_prior"):     S_0.debug_prior = False
    if not _hasfield(S_0, "jeffereys_prior"): S_0.jeffereys_prior = False
    if not landData and not _hasfield(S_1, "birth_death_from_prior"):
        S_1.birth_death_from_prior = False
    if not _hasfield(S_2, "birth_death_from_prior"):
        S_2.birth_death_from_prior = False

    if isinstance(DataFileMain, str):
        FileRoot = os.path.splitext(os.path.basename(DataFileMain))[0]
    else:
        FileRoot = _getfield(S_0, "FileNameRoot", "PT_RJMCMC_run")
    N        = int(S_0.numIterations)
    ARwindow = int(_getfield(S_0, "ARwindow", 500))
    saveWindow      = int(S_0.saveEvery)
    nDisplayEvery   = int(S_0.displayEvery)
    nChains         = int(S_0.nChains)
    B               = np.asarray(S_0.B).ravel().astype(float)
    pSwap           = 1.0

    if not landData:
        for field_ in ("UstepSize", "BstepSize", "MstepSize"):
            if len(np.asarray(getattr(S_1, field_)).ravel()) != nChains:
                raise ValueError(f"S_1.{field_} has fewer entries than nChains")

    os.makedirs(outputFolder, exist_ok=True)
    # If no `seed` kwarg was given, look in S_0 (set by Step1) so users
    # can pin reproducibility from the config file instead of code edits.
    if seed is None:
        seed = _getfield(S_0, "seed", None)
    rng = np.random.default_rng(seed)

    # state for every chain
    x  = [None] * nChains       # composite model
    x1 = [None] * nChains       # water-column half
    x2 = [None] * nChains       # subsurface half
    k1 = [0] * nChains
    k2 = [0] * nChains
    oldMisfit     = [None] * nChains
    oldMisfitVect = [None] * nChains
    S1 = [deepcopy(S_1) for _ in range(nChains)]
    S2 = [deepcopy(S_2) for _ in range(nChains)]
    ConvStat1 = [_dummy_convstat() for _ in range(nChains)]
    ConvStat2 = [_dummy_convstat() for _ in range(nChains)]

    samples   = [[None] * N for _ in range(nChains)]
    kTracker1 = [np.zeros(N) for _ in range(nChains)]
    kTracker2 = [np.zeros(N) for _ in range(nChains)]
    en        = np.zeros((N, 2, nChains))
    swapCount = [np.zeros(N) for _ in range(nChains)]
    AR1 = [[_dummy_ar() for _ in range(N // ARwindow)] for _ in range(nChains)]
    AR2 = [[_dummy_ar() for _ in range(N // ARwindow)] for _ in range(nChains)]
    indivRMSs = np.zeros((N, 1, nChains))  # grown later if multiple data types

    # --- initialize models -----------------------------------------------
    for ii in range(nChains):
        # --- water column (model 1) ---
        if not landData:
            k1[ii] = int(S_1.kInit)
            z1_init = S_1.zMin + (S_1.zMax - S_1.zMin) * rng.random(k1[ii])
            z1_init = np.sort(z1_init)
            if _getfield(S_0, "fixedDimension", False):
                tm_z = np.asarray(S_0.TrueModel.z).ravel()
                posSF = int(np.where(tm_z == S_0.regionBoundaryDepth)[0][0])
                z1_init = tm_z[:posSF]
            rhov1 = S_1.rhoMin + (S_1.rhoMax - S_1.rhoMin) * rng.random(k1[ii] + 1)
            if _getfield(S_0, "fixedWater", False):
                tm_z = np.asarray(S_0.TrueModel.z).ravel()
                tm_r = np.asarray(S_0.TrueModel.rho).ravel()
                posSF = int(np.where(tm_z == S_0.regionBoundaryDepth)[0][0])
                rhov1 = tm_r[:posSF + 1]
                S_0.beta = 0.0
            if _getfield(S_1, "isotropic", True):
                rhoh1 = rhov1.copy()
            else:
                rhoh1 = S_1.rhoMin + (S_1.rhoMax - S_1.rhoMin) * rng.random(k1[ii] + 1)
            x1[ii] = SimpleNamespace(z=z1_init, rhov=rhov1, rhoh=rhoh1)
        else:
            x1[ii] = SimpleNamespace(z=np.array([]),
                                    rhov=np.array([]),
                                    rhoh=np.array([]))

        # --- subsurface (model 2) ---
        k2[ii] = int(S_2.kInit)
        z2_init = S_2.zMin + (S_2.zMax - S_2.zMin) * rng.random(k2[ii])
        z2_init = np.sort(z2_init)
        if _getfield(S_0, "fixedDimension", False):
            tm_z = np.asarray(S_0.TrueModel.z).ravel()
            posSF = int(np.where(tm_z == S_0.regionBoundaryDepth)[0][0])
            z2_init = tm_z[posSF + 1:]
        # If a depth-dependent rho_max profile is registered we draw
        # the initial half-space resistivity from [rhoMin, rho_max(zMax)]
        # -- the TIGHTEST cap anywhere in the inversion domain -- so
        # that every layer of the initial model is guaranteed to be
        # inside its local prior support.  Otherwise the chain would
        # be born outside its own support and any subsequent rho move
        # would be rejected before it could repair the violation.
        init_cap = float(_rhoMaxAt(S_2.zMax, S_2))
        rhov2 = S_2.rhoMin + (init_cap - S_2.rhoMin) * rng.random(k2[ii] + 1)
        if _getfield(S_2, "isotropic", True):
            rhoh2 = rhov2.copy()
        else:
            rhoh2 = S_2.rhoMin + (init_cap - S_2.rhoMin) * rng.random(k2[ii] + 1)
        x2[ii] = SimpleNamespace(z=z2_init, rhov=rhov2, rhoh=rhoh2)

        # composite model & initial misfit
        x[ii] = _composite(x1[ii], x2[ii], S_0.regionBoundaryDepth)
        mis, misV = getMisfit(x[ii], S_0, yespause)
        oldMisfit[ii]     = mis
        oldMisfitVect[ii] = misV

        if not landData:
            S1[ii].rSD1    = np.asarray(S_1.UstepSize).ravel()[ii]
            S1[ii].rSD2    = np.asarray(S_1.BstepSize).ravel()[ii]
            S1[ii].MoveSd  = np.asarray(S_1.MstepSize).ravel()[ii]
        S2[ii].rSD1    = np.asarray(S_2.UstepSize).ravel()[ii]
        S2[ii].rSD2    = np.asarray(S_2.BstepSize).ravel()[ii]
        S2[ii].MoveSd  = np.asarray(S_2.MstepSize).ravel()[ii]

    if len(oldMisfitVect[0]) > 1:
        indivRMSs = np.zeros((N, len(oldMisfitVect[0]), nChains))

    # --- multiprocessing setup ------------------------------------------
    # If S_0.n_workers > 1, run the per-chain MCMC step inside a pool of
    # worker processes (one process can handle multiple chains if
    # n_workers < nChains).  Workers are FORKED so they inherit the
    # registered forward solvers (_DIPOLE1D, _PMAX_E, _SFILT) and the
    # large static structures (S_0, S_1, S_2) via copy-on-write -- this
    # avoids re-pickling these on every call.
    #
    # n_workers = 1 (default) -> identical serial code path as before.
    n_workers = int(_getfield(S_0, "n_workers", 1))
    pool = None
    chain_rng_states = None
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        # populate module-level globals so workers can read them
        global _PT_S_0, _PT_S_1_LIST, _PT_S_2_LIST, _PT_yespause
        _PT_S_0       = S_0
        _PT_S_1_LIST  = S1
        _PT_S_2_LIST  = S2
        _PT_yespause  = yespause

        # one independent RNG per chain, deterministically derived from
        # the user seed via SeedSequence (numpy's recommended pattern).
        # Worker calls receive/return the PCG64 state so each chain's
        # stream advances correctly across iterations regardless of
        # which worker process happens to run a given call.
        ss        = np.random.SeedSequence(seed)
        chain_rngs = [np.random.default_rng(s) for s in ss.spawn(nChains)]
        chain_rng_states = [r.bit_generator.state for r in chain_rngs]

        # Use 'fork' on Linux so worker children inherit the empymod
        # registration done in the parent.  On macOS/Windows fall back
        # to the platform default (spawn) -- in that case the user
        # must ensure csem_forward.register_with_serpent() is called
        # in EVERY worker, which our code does NOT currently do.  On
        # Linux this is fine.
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context()
        pool = ProcessPoolExecutor(max_workers=int(n_workers),
                                   mp_context=ctx)
        print(f"  [PT_RJMCMC] using {n_workers} worker process(es)"
              f"  (nChains={nChains}, fork mode='{ctx.get_start_method()}')")

    # --- MCMC main loop ---------------------------------------------------
    import time
    t0 = time.time()
    for count in range(1, N + 1):
        if count % nDisplayEvery == 0:
            elapsed = time.time() - t0
            per_iter = elapsed / count
            eta_s    = (N - count) * per_iter
            print(f"Iteration {count}/{N}.  t/iter = {per_iter:.5f}s.  "
                  f"ETA = {eta_s/60:.1f} min.  RMS = {oldMisfit[-1][1]:.3f}")

        # ---- chain swap ----
        if rng.random() < pSwap:
            p, q = _determinPerm(nChains, rng)
            for iT in range(len(p)):
                first, second = int(p[iT]) - 1, int(q[iT]) - 1   # 0-based
                logAlphaSwap = (oldMisfit[first][0] - oldMisfit[second][0]) \
                               * (B[first] - B[second])
                if np.log(rng.random()) < logAlphaSwap:
                    if not landData:
                        x1[first], x1[second] = x1[second], x1[first]
                        k1[first], k1[second] = k1[second], k1[first]
                    x2[first], x2[second] = x2[second], x2[first]
                    k2[first], k2[second] = k2[second], k2[first]
                    oldMisfit[first], oldMisfit[second] = oldMisfit[second], oldMisfit[first]
                    # Also swap per-data-type RMS vector so that the recorded
                    # individual misfits stay consistent with the swapped model.
                    oldMisfitVect[first], oldMisfitVect[second] = \
                        oldMisfitVect[second], oldMisfitVect[first]
                    swapCount[first][count - 1]  = 1
                    swapCount[second][count - 1] = 1

        # ---- one MCMC step per chain ----
        # When n_workers > 1 we farm the 8 chains' steps out to worker
        # processes in parallel.  Each chain has its own RNG (state
        # passed in/out per call) so the chain trajectories are
        # statistically equivalent to the serial path and reproducible
        # given the same seed AND the same n_workers setting.
        if pool is not None:
            futures = [pool.submit(_chain_step_worker, jj,
                                   x1[jj], x2[jj], k1[jj], k2[jj],
                                   oldMisfit[jj], oldMisfitVect[jj],
                                   ConvStat1[jj], ConvStat2[jj],
                                   B[jj], chain_rng_states[jj])
                       for jj in range(nChains)]
            for jj, fut in enumerate(futures):
                (x1_new, x2_new, k1_new, k2_new,
                 om_new, omv_new, cs1_new, cs2_new,
                 new_rng_state) = fut.result()
                x1[jj] = x1_new
                x2[jj] = x2_new
                k1[jj] = k1_new
                k2[jj] = k2_new
                oldMisfit[jj]     = om_new
                oldMisfitVect[jj] = omv_new
                ConvStat1[jj]     = cs1_new
                ConvStat2[jj]     = cs2_new
                chain_rng_states[jj] = new_rng_state

        # ---- per-chain bookkeeping (always sequential, cheap) ----
        for jj in range(nChains):
            if pool is None:
                # legacy serial path: do the MCMC step here, identical
                # to the original code (uses the shared main rng).
                dart = rng.random()
                if dart < _getfield(S_0, "beta", 0.0):      # perturb water column
                    x1[jj], k1[jj], oldMisfit[jj], oldMisfitVect[jj], ConvStat1[jj] = \
                        _rj_mcmc_step(x1[jj], k1[jj], x2[jj], oldMisfit[jj],
                                      oldMisfitVect[jj], ConvStat1[jj],
                                      S1[jj], S_0, B[jj], yespause, rng)
                else:                                        # perturb subsurface
                    x2[jj], k2[jj], oldMisfit[jj], oldMisfitVect[jj], ConvStat2[jj] = \
                        _rj_mcmc_step(x2[jj], k2[jj], x1[jj], oldMisfit[jj],
                                      oldMisfitVect[jj], ConvStat2[jj],
                                      S2[jj], S_0, B[jj], yespause, rng)

            # composite model for storage
            x[jj] = _composite(x1[jj], x2[jj], S_0.regionBoundaryDepth)
            samples[jj][count - 1] = deepcopy(x[jj])
            if not landData:
                kTracker1[jj][count - 1] = k1[jj]
            kTracker2[jj][count - 1] = k2[jj]
            en[count - 1, :, jj] = oldMisfit[jj]
            v = np.asarray(oldMisfitVect[jj]).ravel()
            indivRMSs[count - 1, :len(v), jj] = v

            # periodic acceptance-rate bookkeeping
            if count % ARwindow == 0:
                idx = count // ARwindow - 1
                for ARcell, CS in ((AR1, ConvStat1), (AR2, ConvStat2)):
                    ARcell[jj][idx] = dict(
                        uAR=CS[jj]["uA"] / max(CS[jj]["uc"], 1) * 100,
                        bAR=CS[jj]["bA"] / max(CS[jj]["bc"], 1) * 100,
                        dAR=CS[jj]["dA"] / max(CS[jj]["dc"], 1) * 100,
                        mAR=CS[jj]["mA"] / max(CS[jj]["mc"], 1) * 100,
                        TotalAR=(CS[jj]["uA"] + CS[jj]["bA"] +
                                 CS[jj]["dA"] + CS[jj]["mA"]) / ARwindow * 100,
                        evalCount=CS[jj]["evalCount"],
                        swapRate=np.sum(swapCount[jj][count - ARwindow:count])
                                 / ARwindow * 100,
                    )
                    for key in ("uA", "bA", "dA", "mA", "uc", "bc", "dc", "mc"):
                        CS[jj][key] = 0

        # ---- periodic save ----
        if count % saveWindow == 0 or count == N:
            # Trim every per-iter array to the iterations actually completed
            # so the pickles look like MATLAB's (no zero tails).  Pre-allocated
            # AR_ll has N//ARwindow entries; only nWindowsSoFar are real, the
            # rest are _dummy_ar() and would otherwise confuse plot_convergence.
            nWindowsSoFar = count // ARwindow

            def _atomic_dump(obj, dest):
                tmp = dest + '.tmp'
                with open(tmp, 'wb') as fh:
                    pickle.dump(obj, fh)
                os.replace(tmp, dest)

            for ll in range(nChains):
                out = dict(
                    s_ll=samples[ll][:count],
                    en_ll=en[:count, :, ll],
                    AR1_ll=AR1[ll][:nWindowsSoFar],
                    AR2_ll=AR2[ll][:nWindowsSoFar],
                    k1_ll=None if landData else kTracker1[ll][:count],
                    k2_ll=kTracker2[ll][:count],
                    S1_ll=None if landData else S1[ll],
                    S2_ll=S2[ll], S_0=S_0,
                    indivRMSs_ll=indivRMSs[:count, :, ll],
                )
                path = os.path.join(outputFolder,
                                    f"{FileRoot}_PT_RJMCMC_{ll + 1}.pkl")
                _atomic_dump(out, path)
                conv_path = os.path.join(outputFolder,
                                         f"{FileRoot}_PT_RJMCMC_{ll + 1}_conv.pkl")
                conv_out = dict(
                    en_ll=en[:count, :, ll],
                    AR1_ll=AR1[ll][:nWindowsSoFar],
                    AR2_ll=AR2[ll][:nWindowsSoFar],
                    k1_ll=None if landData else kTracker1[ll][:count],
                    k2_ll=kTracker2[ll][:count],
                    indivRMSs_ll=indivRMSs[:count, :, ll],
                )
                _atomic_dump(conv_out, conv_path)
            swaps_trimmed = [swapCount[ll][:count] for ll in range(nChains)]
            _atomic_dump(swaps_trimmed,
                         os.path.join(outputFolder,
                                      f"{FileRoot}_PT_RJMCMC_swaps.pkl"))

    # tear down the parallel pool (if any) so children exit cleanly
    if pool is not None:
        pool.shutdown(wait=True)

    return dict(samples=samples, en=en, AR1=AR1, AR2=AR2,
                k1=kTracker1, k2=kTracker2, indivRMSs=indivRMSs,
                swapCount=swapCount)


def _clean_matlab_dict(d):
    """Drop scipy.io private keys (__header__ etc.)."""
    return {k: v for k, v in d.items() if not k.startswith("__")}


# =====================================================================
# 4.  POST-PROCESSING
# =====================================================================

# -------------------------- CombineChains.m --------------------------

def CombineChains(FileName, burnIn, nthin, nChains, nChainsAtOne):
    """
    Combine post-burn-in samples of the nChainsAtOne T=1 chains.
    Reads <FileName>_<i>.pkl files produced by PT_RJMCMC, saves
    <FileName>_Combined.pkl and returns the concatenated RMS array.
    """
    s_ll   : List[Any] = []
    k1_ll  : List[Any] = []
    k2_ll  : List[Any] = []
    totalRMS: List[Any] = []

    for k in range(1, nChainsAtOne + 1):
        print(f"Processing chain {k} out of {nChainsAtOne}")
        chainInd = nChains - nChainsAtOne + k
        with open(f"{FileName}_{chainInd}.pkl", "rb") as fh:
            U = pickle.load(fh)

        k2 = np.asarray(U["k2_ll"]).ravel()
        j  = int(np.where(k2 != 0)[0][-1]) + 1   # MATLAB 'last'
        totalRMS.append(U["en_ll"][burnIn:j, 1])
        s_ll.extend(U["s_ll"][burnIn:j:nthin])
        if not U["S_0"].landData:
            k1_ll.append(np.asarray(U["k1_ll"]).ravel()[burnIn:j:nthin])
        k2_ll.append(k2[burnIn:j:nthin])

    totalRMS = np.concatenate(totalRMS) if totalRMS else np.array([])
    out = dict(s_ll=s_ll,
               k1_ll=np.concatenate(k1_ll) if k1_ll else np.array([]),
               k2_ll=np.concatenate(k2_ll) if k2_ll else np.array([]))
    print("Saving to disk...")
    with open(f"{FileName}_Combined.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return totalRMS


# --------------------------- plotModel1D.m ---------------------------

def plotModel1D(x, ax=None, style='--c', lw=2):
    """Plot a 1D step-like resistivity profile."""
    z = np.concatenate([[0.0], np.asarray(x.z).ravel()])
    if _hasfield(x, "rhoh"):
        r = np.asarray(x.rhoh).ravel()
    elif _hasfield(x, "rho"):
        r = np.asarray(x.rho).ravel()
    else:
        raise ValueError("x must have field 'rhoh' or 'rho'")

    rr, zz = [], []
    for i, ri in enumerate(r):
        rr.extend([ri, ri])
    for i in range(len(z) - 1):
        zz.extend([z[i], z[i + 1]])
    zz.append(z[-1])
    zz.append(1.1 * z[-1])

    # pad rr to match zz length
    while len(rr) < len(zz):
        rr.append(rr[-1])
    if ax is None:
        ax = plt.gca()
    ax.plot(rr, zz, style, linewidth=lw)
    ax.invert_yaxis()
    return ax


# ------------------------- KL divergence -----------------------------

def KLdivergence(P, Q):
    P = np.asarray(P).ravel()
    Q = np.asarray(Q).ravel()
    if P.size != Q.size:
        raise ValueError("KLdivergence: P and Q must have same length")
    mask = (P > 0) & (Q > 0)
    return float(np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask]))))


# --------------------------- plot_RJMCMC.m ---------------------------

def plot_RJMCMC(s, k, G, S):
    """
    Compute and plot the posterior PDF of log(rho) vs depth, along
    with KL divergence, layer-probability, and layer-count histogram.
    """
    nSamples = len(s)
    nZbins   = int(np.ceil((S.zMax - S.zMin) / G.dz))
    zPlot    = S.zMin + G.dz / 2.0 + np.arange(nZbins) * G.dz

    rhoBinEdges = np.arange(S.rhoMin, S.rhoMax + 1e-12, G.drho)
    nRhobins    = len(rhoBinEdges) - 1
    rhoPlot     = S.rhoMin + G.drho / 2.0 + np.arange(nRhobins) * G.drho

    rhoSamples = np.full((nZbins, nSamples), np.nan)
    kSamples   = np.full((nZbins, nSamples), np.nan)

    iProgress = 1
    landData = bool(_getfield(S, "landData", False))
    for iSample in range(nSamples):
        x  = s[iSample]
        xz = np.asarray(x.z,    dtype=float).ravel()
        xr = np.asarray(x.rhoh, dtype=float).ravel()
        pos_arr = np.where(xz >= S.regionBoundaryDepth)[0]
        if pos_arr.size == 0:
            continue
        pos = int(pos_arr[0])                       # 0-based
        xz_sub = np.unique(np.concatenate([xz[pos:], [S.zMax]]))
        iZbin = 0
        # iterate layers; iLayer in MATLAB is 1..length(x.z)-1
        for iLayer in range(len(xz_sub) - 1):
            while iZbin < nZbins and xz_sub[iLayer] >= S.zMin + G.dz * (iZbin + 1):
                # rho-array indexing for composite.rhoh:
                #   Marine: rhoh = [water_rhoh (nw+1), sub_rhoh (k2+1)]
                #     pos = nw (0-based), so ridx = iLayer + pos + 1
                #     skips the water-column entries.
                #   Land: rhoh = sub_rhoh directly, pos = 0
                #     ridx = iLayer + pos = iLayer.
                ridx = iLayer + pos + (0 if landData else 1)
                if ridx < 0:
                    ridx = 0
                if ridx >= len(xr):
                    ridx = len(xr) - 1
                if _getfield(S, "transform01_ab", False):
                    z_ = S.zMin + G.dz * (iZbin + 1)
                    zRL = np.asarray(S.zRhoLim).ravel()
                    pos1_arr = np.where(zRL >= z_)[0]
                    pos1 = int(pos1_arr[0]) if pos1_arr.size else len(zRL) - 1
                    mR = np.asarray(S.maxRho).ravel()
                    mN = np.asarray(S.minRho).ravel()
                    frac = (z_ - zRL[pos1 - 1]) / (zRL[pos1] - zRL[pos1 - 1])
                    lmx = mR[pos1 - 1] + frac * (mR[pos1] - mR[pos1 - 1])
                    lmn = mN[pos1 - 1] + frac * (mN[pos1] - mN[pos1 - 1])
                    rhoSamples[iZbin, iSample] = lmn + xr[ridx] * (lmx - lmn)
                else:
                    rhoSamples[iZbin, iSample] = xr[ridx]
                iZbin += 1
            if iZbin < nZbins:
                kSamples[iZbin, iSample] = 1
        if nSamples >= 10 and (iSample + 1) % max(1, nSamples // 10) == 0:
            print(f"{iProgress * 10}% complete...")
            iProgress += 1
    kSamples = kSamples[:-1, :]

    # histograms
    posteriorPDF = np.zeros((nZbins, nRhobins))
    p5  = np.zeros(nZbins)
    p95 = np.zeros(nZbins)
    KLd = np.zeros(nZbins)
    for iZbin in range(nZbins):
        data = rhoSamples[iZbin, :]
        data = data[~np.isnan(data)]
        if data.size > 0:
            h, _ = np.histogram(data, bins=rhoBinEdges, density=True)
            posteriorPDF[iZbin, :] = h
            p5[iZbin]  = np.percentile(data, 5)
            p95[iZbin] = np.percentile(data, 95)
        if _getfield(S, "transform01_ab", False):
            zRL = np.asarray(S.zRhoLim).ravel()
            ind_arr = np.where(zRL >= S.zMin + G.dz * (iZbin + 1))[0]
            ind = int(ind_arr[0]) if ind_arr.size else np.asarray(G.prior).shape[0] - 1
            KLd[iZbin] = KLdivergence(posteriorPDF[iZbin, :], np.asarray(G.prior)[ind, :])
        else:
            KLd[iZbin] = KLdivergence(posteriorPDF[iZbin, :], np.asarray(G.prior).ravel())

    p5  = np.concatenate([[p5[0]],  p5[:-1]])
    p95 = np.concatenate([[p95[0]], p95[:-1]])

    # plotting
    fig, axes = plt.subplots(1, 4, figsize=(14, 6))
    ax = axes[0]

    # Mask zero-PDF bins so they show as background, not dominate the
    # colour scale with log10(0)=-inf.
    pdf_log = np.log10(np.where(posteriorPDF > 0, posteriorPDF, np.nan))
    # use 1st..99th percentile of finite values for color limits
    finite = pdf_log[np.isfinite(pdf_log)]
    if finite.size:
        vmin = float(np.percentile(finite, 1))
        vmax = float(np.percentile(finite, 99))
    else:
        vmin, vmax = -3, 0
    h = ax.pcolormesh(rhoPlot, zPlot, pdf_log,
                      shading='auto', cmap='viridis',
                      vmin=vmin, vmax=vmax)
    ax.step(p5,  zPlot, '-r', linewidth=2, where='pre')
    ax.step(p95, zPlot, '-r', linewidth=2, where='pre')
    if _hasfield(S, "TrueModel"):
        y = SimpleNamespace(z=S.TrueModel.z, rho=S.TrueModel.rho)
        plotModel1D(y, ax=ax)
    ax.set_xlabel(r'$\log_{10}\rho$ (ohm$\cdot$m)')
    ax.set_ylabel(r'$\log_{10}$ depth (m)' if _getfield(S, "logZ", False)
                  else 'depth (m)')
    ax.set_title('posterior PDF + 5/95% percentile')
    ax.invert_yaxis()
    cbar = plt.colorbar(h, ax=ax)
    cbar.set_label(r'$\log_{10}$ PDF')

    axes[1].plot(KLd[:-1], zPlot[:-1], linewidth=2)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('KL divergence')
    axes[1].set_title('information vs depth')
    axes[1].grid(True, alpha=0.3)

    kPDF = np.nansum(kSamples, axis=1) / (np.nansum(kSamples) * G.dz + 1e-300)
    kPrior = np.mean(kPDF)
    axes[2].plot(kPDF, zPlot[:-1], linewidth=2)
    axes[2].axvline(kPrior, color='k', linestyle='--', label='prior')
    axes[2].invert_yaxis()
    axes[2].set_xlabel('probability density')
    axes[2].set_title('interface probability')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].hist(np.asarray(k).ravel(), density=True, edgecolor='white',
                 color='steelblue')
    axes[3].set_xlabel('# of subsurface layers'); axes[3].set_ylabel('pdf')
    axes[3].set_title('layer-count posterior')
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, posteriorPDF, p5, p95, KLd


# ---------------------------------------------------------------------
def plot_RJMCMC_linear(s, k, S,
                      depth_max_km   = 200.0,
                      depth_step_km  = 1.0,
                      rho_step       = 0.05,
                      show_water_column = True,
                      prior_pdf      = None,
                      shallow_zoom_km = None,
                      inversion_zMax_km = None):
    """
    Marginal posterior of log10(rho) vs LINEAR depth (km).

    Parameters
    ----------
    s, k                : ensemble samples (list) and layer-count array
                          (the thinned arrays returned by Step 4)
    S                   : main S namespace from Step 1
    depth_max_km        : max depth on the y-axis (km).  Default 200.
    depth_step_km       : depth bin size (km).  Default 1 km.
    rho_step            : log10(rho) bin size.  Default 0.05.
    show_water_column   : for marine data, include 0..seafloor in the plot.
                          No effect on landData runs.
    prior_pdf           : optional 1D prior over rho bins (for KL divergence)
    shallow_zoom_km     : if set, also add an extra subplot zooming into the
                          shallow part (0..shallow_zoom_km) so you can see
                          near-surface features that get squashed at 200 km.
    inversion_zMax_km   : optional - the deepest depth at which the MCMC was
                          allowed to place an interface (= S2.zMax / 1000).
                          Below this depth every sample is forced to be the
                          deepest halfspace rho, so the resulting "posterior"
                          is just the marginal of that one parameter and
                          doesn't reflect a depth-resolved structure.  When
                          set, bins below this depth are MASKED OUT (PDF set
                          to NaN, no 5/95% band drawn, no KL divergence) and
                          a horizontal black dashed line is drawn to mark the
                          inversion boundary.  STRONGLY RECOMMENDED for any
                          interpretation; otherwise the bottom of the plot
                          will look like a structure but it isn't.

    Returns
    -------
    fig, posteriorPDF, p5, p95, KLd
    """
    import numpy as _np
    from types import SimpleNamespace as _NS

    # ---- depth grid (linear, in metres) -----------------------------
    depth_max_m = float(depth_max_km) * 1000.0
    dz_m        = float(depth_step_km) * 1000.0
    nZbins      = int(_np.ceil(depth_max_m / dz_m))
    zEdges_m    = _np.arange(nZbins + 1) * dz_m
    zPlot_m     = 0.5 * (zEdges_m[:-1] + zEdges_m[1:])

    # mask for depths beyond the inversion zMax (no real info there)
    if inversion_zMax_km is not None:
        inv_zMax_m   = float(inversion_zMax_km) * 1000.0
        mask_beyond  = zPlot_m > inv_zMax_m
    else:
        inv_zMax_m   = None
        mask_beyond  = _np.zeros(nZbins, dtype=bool)

    # ---- rho grid ---------------------------------------------------
    rhoMin = float(S.rhoMin); rhoMax = float(S.rhoMax)
    rhoEdges = _np.arange(rhoMin, rhoMax + 1e-12, rho_step)
    nRhobins = len(rhoEdges) - 1
    rhoPlot  = rhoEdges[:-1] + rho_step / 2.0

    logZ     = bool(_getfield(S, "logZ", False))
    landData = bool(_getfield(S, "landData", False))

    rbd_field = float(S.regionBoundaryDepth)
    # NOTE: for marine, scalar `10**log10(3000) = 3000.0000000000014` and
    # vector `10**[log10(3000)] = 3000.000000000001`, a 1-ULP difference
    # that breaks `xz_m >= rbd_m` at the rbd index.  We do the locate-rbd
    # step in *log10 space* (bit-exact equality) and only then convert.
    rbd_m     = 10.0 ** rbd_field if logZ else rbd_field

    nSamples   = len(s)
    rhoSamples = _np.full((nZbins, nSamples), _np.nan)
    kSamples   = _np.zeros((nZbins, nSamples))

    iProgress = 1
    for iSample in range(nSamples):
        if (iSample + 1) / nSamples * 10 >= iProgress:
            print(f'{iProgress*10}% complete...')
            iProgress += 1

        x  = s[iSample]
        xz = _np.asarray(x.z,    dtype=float).ravel()
        xr = _np.asarray(x.rhoh, dtype=float).ravel()

        # locate rbd in the composite z array - bit-exact, in log10 space
        # if logZ is True
        if logZ:
            pos_arr = _np.where(xz >= rbd_field)[0]
        else:
            pos_arr = _np.where(xz >= rbd_m)[0]
        if pos_arr.size == 0:
            continue
        pos = int(pos_arr[0])

        xz_m = 10.0 ** xz if logZ else xz

        # Build (interface, rho) pairs in linear depth.  Convention:
        # for layer i, rho = rhos[i] applies in z in [ifaces[i], ifaces[i+1]).
        # composite.rhoh / composite.z layout:
        #   marine:  z=[w_z (nw), rbd, sub_z (k2)],  rhoh=[w_rhoh (nw+1), sub_rhoh (k2+1)]
        #   land:    z=[rbd, sub_z (k2)],            rhoh=[sub_rhoh (k2+1)]
        if landData:
            sub_z   = xz_m[pos + 1:]               # actual sub interfaces (rbd dropped)
            sub_rho = xr                           # entire composite.rhoh = sub_rhoh
            ifaces  = _np.concatenate([[0.0], sub_z, [depth_max_m + dz_m]])
            rhos    = sub_rho
            min_plot_m = 0.0
        elif show_water_column:
            # marine + water shown
            ifaces = _np.concatenate([[0.0], xz_m, [depth_max_m + dz_m]])
            rhos   = xr
            min_plot_m = 0.0
        else:
            # marine, water hidden
            sub_z   = xz_m[pos + 1:]
            sub_rho = xr[pos + 1:]
            ifaces  = _np.concatenate([[rbd_m], sub_z, [depth_max_m + dz_m]])
            rhos    = sub_rho
            min_plot_m = rbd_m

        # locate the layer for each depth-bin centre via searchsorted
        for iZbin in range(nZbins):
            z_c = zPlot_m[iZbin]
            if z_c < min_plot_m:
                continue
            if z_c > depth_max_m:
                break
            i_layer = _np.searchsorted(ifaces, z_c, side='right') - 1
            if i_layer < 0:
                i_layer = 0
            if i_layer >= len(rhos):
                i_layer = len(rhos) - 1
            rhoSamples[iZbin, iSample] = rhos[i_layer]

            # mark the bin as containing an interface if any (excluding the
            # top at 0 and the bottom sentinel) falls inside it
            z_lo, z_hi = zEdges_m[iZbin], zEdges_m[iZbin + 1]
            inside = (ifaces > z_lo) & (ifaces < z_hi) & (ifaces > 0) \
                     & (ifaces < depth_max_m)
            if _np.any(inside):
                kSamples[iZbin, iSample] = 1

    # ---- compute marginal PDF / percentiles / KL --------------------
    posteriorPDF = _np.zeros((nZbins, nRhobins))
    p5     = _np.full(nZbins, _np.nan)
    p50    = _np.full(nZbins, _np.nan)
    p95    = _np.full(nZbins, _np.nan)
    pMode  = _np.full(nZbins, _np.nan)
    KLd    = _np.zeros(nZbins)

    for iZbin in range(nZbins):
        if mask_beyond[iZbin]:
            continue                          # below inversion zMax -> no info
        col  = rhoSamples[iZbin, :]
        vals = col[~_np.isnan(col)]
        if vals.size == 0:
            continue
        h, _ = _np.histogram(vals, bins=rhoEdges)
        s_h  = h.sum()
        if s_h > 0:
            posteriorPDF[iZbin, :] = h / (s_h * rho_step)
            p5[iZbin]    = _np.percentile(vals, 5)
            p50[iZbin]   = _np.percentile(vals, 50)
            p95[iZbin]   = _np.percentile(vals, 95)
            pMode[iZbin] = rhoPlot[_np.argmax(h)]   # most probable rho per depth
            if prior_pdf is not None:
                KLd[iZbin] = KLdivergence(posteriorPDF[iZbin, :],
                                          _np.asarray(prior_pdf).ravel())

    # mask out the deep region in EVERY output so plot + interpretation match
    if mask_beyond.any():
        posteriorPDF[mask_beyond, :] = _np.nan
        kSamples[mask_beyond, :]     = _np.nan
        KLd[mask_beyond]             = _np.nan

    # ---- plotting ---------------------------------------------------
    nplots = 4 if shallow_zoom_km is None else 5
    fig, axes = plt.subplots(1, nplots, figsize=(3.4 * nplots, 6))
    ax = axes[0]
    zPlot_km = zPlot_m / 1000.0

    pdf_log = _np.log10(_np.where(posteriorPDF > 0, posteriorPDF, _np.nan))
    finite  = pdf_log[_np.isfinite(pdf_log)]
    if finite.size:
        vmin = float(_np.percentile(finite, 1))
        vmax = float(_np.percentile(finite, 99))
    else:
        vmin, vmax = -3.0, 0.0
    h_pcm = ax.pcolormesh(rhoPlot, zPlot_km, pdf_log,
                          shading='auto', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    ax.step(p5,  zPlot_km, '-r', linewidth=1.7, where='pre', label='5/95%')
    ax.step(p95, zPlot_km, '-r', linewidth=1.7, where='pre')
    ax.step(p50, zPlot_km, '-w', linewidth=1.7, where='pre', label='median')
    if not landData and show_water_column:
        ax.axhline(rbd_m / 1000.0, color='cyan', linewidth=1.2,
                   linestyle='--', alpha=0.9, label='seafloor')
    if inv_zMax_m is not None:
        ax.axhline(inv_zMax_m / 1000.0, color='black', linewidth=1.2,
                   linestyle='--', alpha=0.8, label='inversion zMax')
    ax.set_xlabel(r'$\log_{10}\rho$ (ohm$\cdot$m)')
    ax.set_ylabel('depth (km)')
    ax.set_title('posterior PDF + 5/50/95%')
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()
    cbar = plt.colorbar(h_pcm, ax=ax)
    cbar.set_label(r'$\log_{10}$ PDF')

    # optional shallow zoom
    if shallow_zoom_km is not None:
        axZ = axes[1]
        h_pcm2 = axZ.pcolormesh(rhoPlot, zPlot_km, pdf_log,
                                shading='auto', cmap='viridis',
                                vmin=vmin, vmax=vmax)
        axZ.step(p5,  zPlot_km, '-r', linewidth=1.7, where='pre')
        axZ.step(p95, zPlot_km, '-r', linewidth=1.7, where='pre')
        axZ.step(p50, zPlot_km, '-w', linewidth=1.7, where='pre')
        if not landData and show_water_column:
            axZ.axhline(rbd_m / 1000.0, color='cyan', linewidth=1.2,
                        linestyle='--', alpha=0.9)
        axZ.set_xlabel(r'$\log_{10}\rho$ (ohm$\cdot$m)')
        axZ.set_ylabel('depth (km)')
        axZ.set_title(f'shallow zoom: 0..{shallow_zoom_km:g} km')
        axZ.set_ylim(shallow_zoom_km, 0)        # inverted
        plt.colorbar(h_pcm2, ax=axZ)
        idx_offset = 2
    else:
        idx_offset = 1

    axes[idx_offset].plot(KLd, zPlot_km, linewidth=2)
    if inv_zMax_m is not None:
        axes[idx_offset].axhline(inv_zMax_m / 1000.0, color='black',
                                 linewidth=1.2, linestyle='--', alpha=0.8)
    axes[idx_offset].invert_yaxis()
    axes[idx_offset].set_xlabel('KL divergence')
    axes[idx_offset].set_ylabel('depth (km)')
    axes[idx_offset].set_title('information vs depth')
    axes[idx_offset].grid(True, alpha=0.3)

    # interface probability density per km - normalize only over the
    # depth range where the inversion was actually free to place interfaces
    kSum     = _np.nansum(kSamples, axis=1)
    valid    = ~mask_beyond
    norm     = _np.nansum(kSamples[valid, :]) * dz_m + 1e-300
    kPDF     = kSum / norm * 1000.0
    kPDF[mask_beyond] = _np.nan
    kPrior   = _np.nanmean(kPDF)
    axes[idx_offset + 1].plot(kPDF, zPlot_km, linewidth=2)
    axes[idx_offset + 1].axvline(kPrior, color='k', linestyle='--', label='prior')
    if inv_zMax_m is not None:
        axes[idx_offset + 1].axhline(inv_zMax_m / 1000.0, color='black',
                                     linewidth=1.2, linestyle='--', alpha=0.8)
    axes[idx_offset + 1].invert_yaxis()
    axes[idx_offset + 1].set_xlabel('probability density (per km)')
    axes[idx_offset + 1].set_ylabel('depth (km)')
    axes[idx_offset + 1].set_title('interface probability')
    axes[idx_offset + 1].legend()
    axes[idx_offset + 1].grid(True, alpha=0.3)

    axes[idx_offset + 2].hist(_np.asarray(k).ravel(), density=True,
                              edgecolor='white', color='steelblue')
    axes[idx_offset + 2].set_xlabel('# of subsurface layers')
    axes[idx_offset + 2].set_ylabel('pdf')
    axes[idx_offset + 2].set_title('layer-count posterior')
    axes[idx_offset + 2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, posteriorPDF, p5, p95, KLd


# ---------------------------------------------------------------------
# MATLAB-parula colormap (so paper-style plots match published figures
# from the marine-MT community without an extra dependency).
# ---------------------------------------------------------------------
_PARULA_RGB = [
    (0.2081, 0.1663, 0.5292), (0.2116, 0.1898, 0.5777),
    (0.2123, 0.2138, 0.6270), (0.2081, 0.2386, 0.6771),
    (0.1959, 0.2645, 0.7279), (0.1707, 0.2919, 0.7792),
    (0.1253, 0.3242, 0.8303), (0.0591, 0.3598, 0.8683),
    (0.0117, 0.3875, 0.8820), (0.0060, 0.4086, 0.8828),
    (0.0165, 0.4266, 0.8786), (0.0329, 0.4430, 0.8720),
    (0.0498, 0.4586, 0.8641), (0.0629, 0.4737, 0.8554),
    (0.0723, 0.4887, 0.8467), (0.0779, 0.5040, 0.8384),
    (0.0793, 0.5200, 0.8312), (0.0749, 0.5375, 0.8263),
    (0.0641, 0.5570, 0.8240), (0.0488, 0.5772, 0.8228),
    (0.0343, 0.5966, 0.8199), (0.0265, 0.6137, 0.8135),
    (0.0239, 0.6287, 0.8038), (0.0231, 0.6418, 0.7913),
    (0.0228, 0.6535, 0.7768), (0.0267, 0.6642, 0.7607),
    (0.0384, 0.6743, 0.7436), (0.0590, 0.6838, 0.7254),
    (0.0843, 0.6928, 0.7062), (0.1133, 0.7015, 0.6859),
    (0.1453, 0.7098, 0.6646), (0.1801, 0.7177, 0.6424),
    (0.2178, 0.7250, 0.6193), (0.2586, 0.7317, 0.5954),
    (0.3022, 0.7376, 0.5712), (0.3482, 0.7424, 0.5473),
    (0.3953, 0.7459, 0.5244), (0.4420, 0.7481, 0.5033),
    (0.4871, 0.7491, 0.4839), (0.5300, 0.7491, 0.4661),
    (0.5709, 0.7485, 0.4494), (0.6099, 0.7473, 0.4337),
    (0.6473, 0.7456, 0.4188), (0.6834, 0.7435, 0.4044),
    (0.7184, 0.7411, 0.3905), (0.7525, 0.7384, 0.3768),
    (0.7858, 0.7356, 0.3633), (0.8185, 0.7327, 0.3498),
    (0.8507, 0.7299, 0.3360), (0.8824, 0.7274, 0.3217),
    (0.9139, 0.7258, 0.3063), (0.9450, 0.7261, 0.2886),
    (0.9739, 0.7314, 0.2666), (0.9938, 0.7455, 0.2403),
    (0.9990, 0.7653, 0.2164), (0.9955, 0.7861, 0.1967),
    (0.9880, 0.8066, 0.1794), (0.9789, 0.8271, 0.1633),
    (0.9697, 0.8481, 0.1475), (0.9626, 0.8705, 0.1309),
    (0.9589, 0.8949, 0.1132), (0.9598, 0.9218, 0.0948),
    (0.9661, 0.9514, 0.0755), (0.9763, 0.9831, 0.0538),
]

try:
    from matplotlib.colors import ListedColormap
    PARULA_CMAP = ListedColormap(_PARULA_RGB, name='parula')
except ImportError:
    PARULA_CMAP = None


# ---------------------------------------------------------------------
def plot_RJMCMC_paper(s, k, S,
                     depth_min_km     = 0.0,
                     depth_max_km     = 130.0,
                     depth_step_km    = 0.5,
                     rho_step         = 0.05,
                     show_water_column= False,
                     inversion_zMax_km= None,
                     pdf_log_range    = (-3.0, -0.5),
                     overlay_depths_km= None,
                     title            = None,
                     figsize          = (4.5, 8.0),
                     cmap             = None,
                     rho_xlim         = None,
                     label_fontsize   = 14,
                     tick_fontsize    = 12,
                     smooth_sigma     = (1.0, 1.0),
                     zero_pdf_color   = 'low'):
    """
    Publication-style single-panel posterior plot.  Mimics the marine-MT
    convention used in e.g. Naif et al., Key et al. etc.:
      - single tall panel, depth on y (inverted), log10(rho) on x
      - parula colormap, FIXED PDF colour range  (default 10^-3 .. 10^-0.5)
      - bold white median, bold red 5 % and 95 % step lines
      - optional dashed white interpretive depth annotations
      - inversion-zMax cut-off as a black dashed line

    Returns (fig, posteriorPDF, p5, p50, p95, KLd).

    Parameters
    ----------
    s, k                : ensemble (samples list and k array, as from Step 4)
    S                   : main S namespace from Step 1
    depth_min_km        : top of the y-axis (default 0).  Set e.g. 5 to crop
                          out the water column for marine sites.
    depth_max_km        : bottom of the y-axis (default 130).
    depth_step_km       : depth bin (default 0.5 km - finer for paper-quality).
    rho_step            : log10(rho) bin (default 0.05).
    show_water_column   : if True (marine only), include 0..seafloor in the
                          binning.  False crops out the water for cleaner
                          subsurface-only figures.
    inversion_zMax_km   : mask depths > this (no info there).  Highly
                          recommended; pass `S2.zMax / 1000.0` from Step 1.
    pdf_log_range       : (vmin, vmax) for log10(PDF) colour scale.  Fixed
                          range gives consistent appearance across stations,
                          matching publication-figure convention.
    overlay_depths_km   : list of depth values (km) to draw as white dashed
                          horizontal lines (e.g. LAB, Moho, etc.).
    title               : optional small text shown inside the plot.
    figsize             : (w, h) inches.  Default tall+narrow for paper.
    cmap                : matplotlib colormap.  Defaults to MATLAB parula.
    rho_xlim            : (low, high) log10(rho) x-axis limits.  None = auto
                          from S.rhoMin/rhoMax.
    smooth_sigma        : (sig_z, sig_rho) Gaussian-filter standard deviation
                          IN BINS used to remove sampling-noise speckle from
                          the PDF heatmap (purely a visualisation tweak; the
                          5/50/95% lines are computed BEFORE smoothing).
                          (1.0, 1.0) is mild and is recommended; pass (0, 0)
                          to disable.  Larger values give a more "painted"
                          appearance like published papers.
    zero_pdf_color      : how to display histogram cells with zero samples.
                          'low'   - render as the lowest cmap colour (dark
                                    blue) so the figure looks filled in.
                                    RECOMMENDED for papers; sampling-noise
                                    zeros look like real zero-probability.
                          'white' - render as the axes background (NaN).
                                    Useful for diagnostics: white speckle
                                    flags bins where the posterior is truly
                                    zero or where sampling is too sparse.
    """
    import numpy as _np

    # depth grid
    depth_max_m = float(depth_max_km) * 1000.0
    dz_m        = float(depth_step_km) * 1000.0
    nZbins      = int(_np.ceil(depth_max_m / dz_m))
    zEdges_m    = _np.arange(nZbins + 1) * dz_m
    zPlot_m     = 0.5 * (zEdges_m[:-1] + zEdges_m[1:])

    if inversion_zMax_km is not None:
        inv_zMax_m  = float(inversion_zMax_km) * 1000.0
        mask_beyond = zPlot_m > inv_zMax_m
    else:
        inv_zMax_m  = None
        mask_beyond = _np.zeros(nZbins, dtype=bool)

    # rho grid
    rhoMin = float(S.rhoMin); rhoMax = float(S.rhoMax)
    rhoEdges = _np.arange(rhoMin, rhoMax + 1e-12, rho_step)
    nRhobins = len(rhoEdges) - 1
    rhoPlot  = rhoEdges[:-1] + rho_step / 2.0

    logZ     = bool(_getfield(S, "logZ", False))
    landData = bool(_getfield(S, "landData", False))
    rbd_field = float(S.regionBoundaryDepth)
    rbd_m     = 10.0 ** rbd_field if logZ else rbd_field

    nSamples   = len(s)
    rhoSamples = _np.full((nZbins, nSamples), _np.nan)

    # Bin each sample's resistivity profile onto the depth grid
    for iSample in range(nSamples):
        x  = s[iSample]
        xz = _np.asarray(x.z,    dtype=float).ravel()
        xr = _np.asarray(x.rhoh, dtype=float).ravel()

        # find rbd index in log10 space (bit-exact)
        if logZ:
            pos_arr = _np.where(xz >= rbd_field)[0]
        else:
            pos_arr = _np.where(xz >= rbd_m)[0]
        if pos_arr.size == 0:
            continue
        pos = int(pos_arr[0])
        xz_m = 10.0 ** xz if logZ else xz

        if landData:
            sub_z   = xz_m[pos + 1:]
            sub_rho = xr
            ifaces  = _np.concatenate([[0.0], sub_z, [depth_max_m + dz_m]])
            rhos    = sub_rho
            min_plot_m = 0.0
        elif show_water_column:
            ifaces = _np.concatenate([[0.0], xz_m, [depth_max_m + dz_m]])
            rhos   = xr
            min_plot_m = 0.0
        else:
            sub_z   = xz_m[pos + 1:]
            sub_rho = xr[pos + 1:]
            ifaces  = _np.concatenate([[rbd_m], sub_z, [depth_max_m + dz_m]])
            rhos    = sub_rho
            min_plot_m = rbd_m

        for iZbin in range(nZbins):
            z_c = zPlot_m[iZbin]
            if z_c < min_plot_m:
                continue
            if z_c > depth_max_m:
                break
            i_layer = _np.searchsorted(ifaces, z_c, side='right') - 1
            if i_layer < 0:
                i_layer = 0
            if i_layer >= len(rhos):
                i_layer = len(rhos) - 1
            rhoSamples[iZbin, iSample] = rhos[i_layer]

    # compute marginal PDF and summary curves
    # (summary lines use the RAW histogram, not the smoothed one)
    posteriorPDF = _np.zeros((nZbins, nRhobins))
    p5    = _np.full(nZbins, _np.nan)
    p50   = _np.full(nZbins, _np.nan)
    p95   = _np.full(nZbins, _np.nan)
    KLd   = _np.zeros(nZbins)

    for iZbin in range(nZbins):
        if mask_beyond[iZbin]:
            continue
        col  = rhoSamples[iZbin, :]
        vals = col[~_np.isnan(col)]
        if vals.size == 0:
            continue
        h, _ = _np.histogram(vals, bins=rhoEdges)
        s_h  = h.sum()
        if s_h > 0:
            posteriorPDF[iZbin, :] = h / (s_h * rho_step)
            p5[iZbin]  = _np.percentile(vals, 5)
            p50[iZbin] = _np.percentile(vals, 50)
            p95[iZbin] = _np.percentile(vals, 95)

    # Smooth the PDF along (z, rho) to remove sampling-noise speckle.
    # The summary lines above were computed BEFORE smoothing so they
    # remain statistically correct.
    pdf_display = posteriorPDF.copy()
    if smooth_sigma is not None and (smooth_sigma[0] > 0 or smooth_sigma[1] > 0):
        try:
            from scipy.ndimage import gaussian_filter
            valid = ~mask_beyond
            tmp = pdf_display[valid, :].copy()
            tmp = gaussian_filter(tmp, sigma=smooth_sigma, mode='nearest')
            pdf_display[valid, :] = tmp
        except ImportError:
            pass

    # mask out below-zMax for display + percentile arrays
    if mask_beyond.any():
        pdf_display[mask_beyond, :] = _np.nan
        posteriorPDF[mask_beyond, :] = _np.nan

    # plot
    if cmap is None:
        cmap = PARULA_CMAP if PARULA_CMAP is not None else 'viridis'

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    zPlot_km = zPlot_m / 1000.0
    vmin, vmax = pdf_log_range

    if zero_pdf_color == 'low':
        # treat 0-PDF cells as 10**vmin so they get the colormap floor
        # (looks "filled in" like published figures)
        pdf_for_plot = _np.where(pdf_display > 0, pdf_display, 10.0 ** vmin)
        pdf_log = _np.log10(pdf_for_plot)
        # but still mask the zMax region
        if mask_beyond.any():
            pdf_log[mask_beyond, :] = _np.nan
    else:
        pdf_log = _np.log10(_np.where(pdf_display > 0, pdf_display, _np.nan))

    h_pcm = ax.pcolormesh(rhoPlot, zPlot_km, pdf_log,
                          shading='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax)
    # 5/95% as bold red step lines
    ax.step(p5,  zPlot_km, '-r', linewidth=2.2, where='pre')
    ax.step(p95, zPlot_km, '-r', linewidth=2.2, where='pre')
    # median as bold white step line
    ax.step(p50, zPlot_km, '-w', linewidth=2.2, where='pre')

    # interpretive depth lines (LAB, Moho, etc.)
    if overlay_depths_km is not None:
        for zd in overlay_depths_km:
            ax.axhline(zd, color='white', linewidth=1.6,
                       linestyle='--', alpha=0.9)

    # mark seafloor for marine
    if not landData and show_water_column:
        ax.axhline(rbd_m / 1000.0, color='cyan', linewidth=1.2,
                   linestyle='--', alpha=0.8)
    # mark inversion zMax
    if inv_zMax_m is not None and (inv_zMax_m / 1000.0) <= depth_max_km:
        ax.axhline(inv_zMax_m / 1000.0, color='black', linewidth=1.2,
                   linestyle='--', alpha=0.7)

    ax.set_xlabel(r'log$_{10}(\rho)$ (ohm$\cdot$m)', fontsize=label_fontsize)
    ax.set_ylabel('Depth (km)', fontsize=label_fontsize)
    if rho_xlim is not None:
        ax.set_xlim(rho_xlim)
    ax.set_ylim(depth_max_km, depth_min_km)   # inverted
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    if title is not None:
        ax.text(0.05, 0.96, title, transform=ax.transAxes,
                color='white', fontsize=label_fontsize + 2,
                fontweight='bold', verticalalignment='top',
                bbox=dict(facecolor='black', edgecolor='white',
                          alpha=0.55, pad=4.0, boxstyle='round,pad=0.4'))

    cbar = plt.colorbar(h_pcm, ax=ax, pad=0.03)
    cbar.set_label(r'log$_{10}$(PDF)', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    fig.tight_layout()
    return fig, posteriorPDF, p5, p50, p95, KLd


# ---------------------------------------------------------------------
def plot_RJMCMC_waterColumn(s, k, G, S):
    """
    Like plot_RJMCMC but for the water-column portion of the model
    (everything above S.regionBoundaryDepth).
    """
    nSamples = len(s)
    nZbins   = int(np.ceil((S.zMax - S.zMin) / G.dz))
    zPlot    = S.zMin + G.dz / 2.0 + np.arange(nZbins) * G.dz

    rhoBinEdges = np.arange(S.rhoMin, S.rhoMax + 1e-12, G.drho)
    nRhobins    = len(rhoBinEdges) - 1
    rhoPlot     = S.rhoMin + G.drho / 2.0 + np.arange(nRhobins) * G.drho

    rhoSamples = np.zeros((nZbins, nSamples))
    kSamples   = np.zeros((nZbins, nSamples))

    for iSample in range(nSamples):
        x  = s[iSample]
        xz = np.asarray(x.z,    dtype=float).ravel()
        xr = np.asarray(x.rhoh, dtype=float).ravel()
        pos_arr = np.where(xz == S.regionBoundaryDepth)[0]
        if pos_arr.size == 0:
            continue
        pos = int(pos_arr[0])                        # 0-based index
        xz_top = xz[:pos + 1]
        iZbin = 0
        for iLayer in range(pos + 1):
            while iZbin < nZbins and xz_top[iLayer] >= S.zMin + G.dz * (iZbin + 1):
                if iLayer < len(xr):
                    rhoSamples[iZbin, iSample] = xr[iLayer]
                iZbin += 1
            if iZbin < nZbins:
                kSamples[iZbin, iSample] = 1
    kSamples = kSamples[:-1, :]

    # histograms
    posteriorPDF = np.zeros((nZbins, nRhobins))
    p5  = np.zeros(nZbins)
    p95 = np.zeros(nZbins)
    for iZbin in range(nZbins):
        data = rhoSamples[iZbin, :]
        data = data[data != 0]        # match MATLAB which uses zeros-init
        if data.size > 0:
            h, _ = np.histogram(data, bins=rhoBinEdges, density=True)
            posteriorPDF[iZbin, :] = h
            p5[iZbin]  = np.percentile(data, 5)
            p95[iZbin] = np.percentile(data, 95)
    p5  = np.concatenate([[p5[0]],  p5[:-1]])
    p95 = np.concatenate([[p95[0]], p95[:-1]])

    fig, axes = plt.subplots(1, 3, figsize=(11, 5))
    h = axes[0].pcolormesh(rhoPlot, zPlot,
                           np.log10(posteriorPDF + 1e-300), shading='auto')
    axes[0].step(p5,  zPlot, '-r', linewidth=2, where='pre')
    axes[0].step(p95, zPlot, '-r', linewidth=2, where='pre')
    axes[0].invert_yaxis(); axes[0].set_xlabel(r'$\log\rho$'); axes[0].set_ylabel('log(z)')
    plt.colorbar(h, ax=axes[0])

    kPDF = np.sum(kSamples, axis=1) / (np.sum(kSamples) * G.dz + 1e-300)
    axes[1].plot(kPDF, zPlot[:-1], linewidth=2)
    axes[1].invert_yaxis(); axes[1].set_xlabel('p.d.f.')

    axes[2].hist(np.asarray(k).ravel(), density=True, edgecolor='none')
    axes[2].set_xlabel('# water-column layers')
    fig.tight_layout()
    return fig, posteriorPDF, p5, p95


# ----------------- plot_convergence_PT_RJMCMC.m ----------------------

def plot_convergence_PT_RJMCMC(filePrefix, nFiles):
    """
    Plot per-chain RMS, interface counts, update/birth/death/move rates,
    and swap rates vs iteration number, across nFiles chains.  Reads
    pickle files produced by PT_RJMCMC.

    Behaves like MATLAB plot_convergence_PT_RJMCMC.m: if a chain was
    saved before the run finished, only the iterations that were
    actually executed are plotted (no zero tails).  We also defensively
    trim arrays in case an older pickle (pre-fix) is being read.
    """
    ARmain = [[None] * nFiles, [None] * nFiles]
    kMain  = [[None] * nFiles, [None] * nFiles]
    RMS    = [None] * nFiles
    indivRMS = [None] * nFiles

    for iFile in range(nFiles):
        conv_path = f"{filePrefix}_PT_RJMCMC_{iFile + 1}_conv.pkl"
        full_path = f"{filePrefix}_PT_RJMCMC_{iFile + 1}.pkl"
        if os.path.isfile(conv_path):
            with open(conv_path, "rb") as fh:
                U = pickle.load(fh)
        else:
            with open(full_path, "rb") as fh:
                U = pickle.load(fh)
        ARmain[0][iFile] = U["AR1_ll"]
        ARmain[1][iFile] = U["AR2_ll"]
        # k1_ll may be None (landData) -- replace with zeros of right length
        k1_raw = U.get("k1_ll")
        k2_arr = np.asarray(U["k2_ll"]).ravel()
        if k1_raw is None:
            kMain[0][iFile] = np.zeros_like(k2_arr)
        else:
            kMain[0][iFile] = np.asarray(k1_raw).ravel()
        kMain[1][iFile] = k2_arr
        RMS[iFile]      = np.asarray(U["en_ll"])
        ir              = U.get("indivRMSs_ll")
        indivRMS[iFile] = None if ir is None else np.asarray(ir)

    for m in range(2):   # model 1 (water) and model 2 (subsurface)
        fig, ax = plt.subplots(4, 2, figsize=(12, 12))
        for iFile in range(nFiles):
            AR_ll = ARmain[m][iFile]
            k_ll  = kMain[m][iFile]
            en_ll = RMS[iFile]

            # Defensive trim: in MATLAB and in the trimmed Python pickles
            # there should be no zero tail, but if we're reading an older
            # pickle that pre-allocated to N we still want the right answer.
            firstEmpty_arr = np.where(k_ll == 0)[0]
            firstEmpty = int(firstEmpty_arr[0]) if firstEmpty_arr.size else len(k_ll)
            k_ll  = k_ll[:firstEmpty]
            en_ll = en_ll[:firstEmpty]

            if len(en_ll) == 0:
                continue                          # all-zero (landData model 1)

            # ---- THIN the per-iteration arrays before plotting ------
            # Matplotlib with the Agg backend grinds to a halt past ~1e5
            # points per artist; for a 1e6-iteration run with nFiles=8
            # chains the naive ax.plot() call would render 8e6 points
            # and take 20+ minutes (or run out of memory).  Plot at most
            # MAX_POINTS samples per chain -- visually indistinguishable
            # from the full series for a Markov-chain diagnostic.
            MAX_POINTS = 5000
            n = len(en_ll)
            if n > MAX_POINTS:
                step = max(1, n // MAX_POINTS)
                idx  = np.arange(0, n, step)
                en_plot = en_ll[idx]
                k_plot  = k_ll[idx]
                x_plot  = idx
            else:
                en_plot = en_ll
                k_plot  = k_ll
                x_plot  = np.arange(n)

            ax[0, 0].plot(x_plot, en_plot[:, 1]); ax[0, 0].set_title('RMS per chain')
            ax[0, 1].plot(x_plot, k_plot);        ax[0, 1].set_title('interfaces per chain')

            if AR_ll is not None and len(AR_ll) > 0:
                # Also trim AR to whatever windows are real.  An AR entry is
                # real iff any of the rate fields is non-zero.  This makes us
                # robust to legacy pickles that stored _dummy_ar() in the tail.
                AR_real = [a for a in AR_ll
                           if any(a.get(k, 0) != 0
                                  for k in ('uAR', 'bAR', 'dAR', 'mAR',
                                            'TotalAR', 'evalCount'))]
                # The MATLAB convention: x-axis at (i+1)*ARwindowLength.
                # ARwindowLength = len(en_ll) / len(AR_real)
                nAR = len(AR_real)
                if nAR > 0:
                    ARw = len(en_ll) / nAR
                    uAR = np.array([a.get("uAR", 0) for a in AR_real])
                    bAR = np.array([a.get("bAR", 0) for a in AR_real])
                    dAR = np.array([a.get("dAR", 0) for a in AR_real])
                    mAR = np.array([a.get("mAR", 0) for a in AR_real])
                    swaps = np.array([a.get("swapRate", 0) for a in AR_real])
                    xx = (np.arange(nAR) + 1) * ARw
                    ax[1, 0].plot(xx, uAR); ax[1, 0].set_title('update rate')
                    ax[1, 1].plot(xx, bAR); ax[1, 1].set_title('birth rate')
                    ax[2, 0].plot(xx, dAR); ax[2, 0].set_title('death rate')
                    ax[2, 1].plot(xx, mAR); ax[2, 1].set_title('move rate')
                    ax[3, 0].plot(xx, swaps); ax[3, 0].set_title('swap rate')

        # individual data-type RMSs from the last chain
        if indivRMS[-1] is not None and indivRMS[-1].size:
            firstEmpty_arr = np.where(kMain[1][-1] == 0)[0]
            fe = int(firstEmpty_arr[0]) if firstEmpty_arr.size else len(kMain[1][-1])
            arr = indivRMS[-1][:fe, :]
            # Same thinning as above so a 1e6-iter run doesn't choke.
            MAX_POINTS = 5000
            n = arr.shape[0]
            if n > MAX_POINTS:
                step  = max(1, n // MAX_POINTS)
                idx   = np.arange(0, n, step)
                arr_p = arr[idx]
                x_p   = idx
            else:
                arr_p = arr
                x_p   = np.arange(n)
            for col in range(arr_p.shape[1]):
                ax[3, 1].plot(x_p, arr_p[:, col])
            ax[3, 1].set_title('individual data-type RMSs')

        # ensure x-axes share scale within a model
        for r in range(4):
            for c in range(2):
                ax[r, c].grid(True, alpha=0.3)

        fig.suptitle(f'Model {m + 1} convergence')
        fig.tight_layout()


# ------------- PlotCSEM_MT_ModelResponsesAndData.m -------------------

def PlotCSEM_MT_ModelResponsesAndData(FileName, MT=False, DCR=False,
                                      stCSEM=False, obCSEM=False,
                                      NtoCalc=50, NtoPlot=50,
                                      seed=None):
    """
    Plot sample-model forward responses against measured data, for
    whichever data types were inverted.  Input FileName is the prefix
    (without '_PT_RJMCMC_Combined.pkl' and without '.pkl').

    Loads `<FileName>.pkl` (preferred) or `<FileName>.mat` (legacy).
    Returns the list of created matplotlib Figures.
    """
    combined = FileName + "_PT_RJMCMC_Combined.pkl"
    with open(combined, "rb") as fh:
        U = pickle.load(fh)

    # Pull S from .pkl if available, else fall back to legacy .mat
    pkl_path = FileName + ".pkl"
    mat_path = FileName + ".mat"
    if os.path.exists(pkl_path):
        S = _load_input(pkl_path)
    elif loadmat is not None and os.path.exists(mat_path):
        S = _as_ns(_clean_matlab_dict(
            loadmat(mat_path, squeeze_me=True, struct_as_record=False)))
    else:
        raise FileNotFoundError(
            f"Neither {pkl_path} nor {mat_path} exists.")

    k2 = np.asarray(U["k2_ll"]).ravel()
    nz = np.where(k2 != 0)[0]
    iComputed = int(nz[-1]) + 1 if nz.size else len(k2)
    iComputed = max(iComputed, 1)

    skip = max(1, int(np.ceil(NtoCalc / NtoPlot)))
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, iComputed, NtoCalc)

    # storage
    if MT:
        freqs = np.asarray(S.MTdat.freqs).ravel()
        ModEnsAppRes = np.zeros((NtoCalc, len(freqs)))
        ModEnsPhase  = np.zeros((NtoCalc, len(freqs)))
    if DCR:
        es = np.asarray(S.dcrDat.es).ravel()
        ModEnsDCRappRes = np.zeros((NtoCalc, len(es)))
    if obCSEM:
        ob_freqs = np.asarray(S.obDat.Freqs).ravel()
        nVRx     = np.asarray(S.obRx.X).shape[0]    # virtual-Rx count (= 42 for L3)
        ModEnsObAmp = np.full((NtoCalc, len(ob_freqs), nVRx), np.nan)
    if stCSEM:
        st_freqs = np.asarray(S.stDat.Freqs).ravel()
        nStRx    = np.asarray(S.stRx.X).shape[0]
        nStSound = len(np.asarray(S.stTx.Soundings).ravel())
        ModEnsStAmp = np.full((NtoCalc, len(st_freqs), nStRx, nStSound), np.nan)

    landData = bool(_getfield(S, "landData", False))

    # CRITICAL: compare positions in *log10 space* (where the values were
    # actually stored), then only convert AFTER finding pos.  numpy uses
    # slightly different pow routines for scalar vs array exponentiation,
    # so for marine rbd = log10(3000) the scalar path gives 3000.0000000000014
    # but the array path gives 3000.000000000001 -- a 1-ULP mismatch that
    # would make `z == rbd_linear` always False and skip the entire loop.
    # Comparing in log10 space sidesteps this: smp.z[nw] is a *direct*
    # bit-copy of S.regionBoundaryDepth so `z_log == rbd_log` is exact.
    rbd_log = float(S.regionBoundaryDepth) if _getfield(S, "logZ", False) \
              else None
    if _getfield(S, "logZ", False):
        S.regionBoundaryDepth = 10.0 ** S.regionBoundaryDepth
        if _getfield(S, "transform01_ab", False):
            S.zMin = 10.0 ** S.zMin
            S.zMax = 10.0 ** S.zMax

    samples = U["s_ll"]
    n_skipped = 0
    for l in range(NtoCalc):
        smp = samples[indexes[l]]
        rho = np.asarray(smp.rhoh, dtype=float).ravel()
        z   = np.asarray(smp.z,    dtype=float).ravel()
        # find rbd index in the ORIGINAL (log10) space - bit-exact equality
        if rbd_log is not None:
            pos_arr = np.where(z == rbd_log)[0]
        else:
            pos_arr = np.where(z == S.regionBoundaryDepth)[0]
        # belt-and-braces fall-back for very-old pickles or weird FP edge:
        # accept anything within 1e-9 relative tolerance.
        if pos_arr.size == 0:
            ref = rbd_log if rbd_log is not None else S.regionBoundaryDepth
            pos_arr = np.where(np.isclose(z, ref, rtol=1e-9, atol=1e-12))[0]
        if pos_arr.size == 0:
            n_skipped += 1
            continue
        pos = int(pos_arr[0])
        # NOW we can do the log10 -> linear conversion for the slice
        if _getfield(S, "logZ", False):
            z = 10.0 ** z
        z_sub   = z[pos + 1:] - S.regionBoundaryDepth
        # same land-data rho-slicing fix as in get_fieldMT
        if landData:
            rho_sub = rho[pos:]
        else:
            rho_sub = rho[pos + 1:]

        if MT:
            z_pad  = np.concatenate([[0.0], z_sub])
            h      = np.diff(z_pad)
            rho_lin = 10.0 ** rho_sub
            appRes, phase, _ = MT1D(rho_lin, h, freqs)
            ModEnsAppRes[l, :] = np.log10(appRes)
            ModEnsPhase[l, :]  = phase

        if DCR:
            z_pad = np.concatenate([[0.0], z_sub])
            h     = np.diff(z_pad)
            ModEnsDCRappRes[l, :] = _sfilt(es, h, rho_sub)

        # ---- obCSEM forward for this sample ---------------------
        # We call get_field_obCSEM with the ORIGINAL sample (not the
        # z_sub/rho_sub slice; get_field_obCSEM does its own re-zeroing).
        # We need to make sure smp.z is in the same units as S expects
        # (smp is in log10 if S.logZ, get_field_obCSEM converts).
        if obCSEM:
            QQ = get_field_obCSEM(S, smp, yespause=False)
            # QQ shape (nFreq, nVRx, nSound=1).  Squeeze to (nFreq, nVRx).
            QQ = np.asarray(QQ)[..., 0] if QQ.ndim == 3 else np.asarray(QQ)
            if _getfield(S.obDat, 'observable', 'inline_E') == 'Pmax_E':
                ModEnsObAmp[l, :, :] = np.log10(np.maximum(QQ, 1e-300))
            else:
                ModEnsObAmp[l, :, :] = np.log10(np.abs(QQ))

        # ---- stCSEM forward for this sample ---------------------
        if stCSEM:
            QQ = get_field_stCSEM(S, smp, yespause=False)
            QQ = np.asarray(QQ)
            # QQ is (nFreq, nRx, nSound) complex
            ModEnsStAmp[l, ...] = np.log10(np.abs(QQ))

    if n_skipped:
        print(f"  [PlotCSEM] WARNING: skipped {n_skipped}/{NtoCalc} samples "
              f"that had no matching rbd in z.  This usually means the "
              f"`regionBoundaryDepth` in S.pkl was changed AFTER inversion.")

    figs = []
    # ---- plots ----
    if MT:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # MATLAB-style: thin grey lines with star markers at every frequency
        # so each individual posterior model is visible
        for l in range(0, NtoCalc, skip):
            ax[0].semilogx(1.0 / freqs, ModEnsPhase[l, :], '-*',
                           color=[0.65, 0.65, 0.65], alpha=0.6,
                           linewidth=0.8, markersize=4)
            ax[1].semilogx(1.0 / freqs, ModEnsAppRes[l, :], '-*',
                           color=[0.65, 0.65, 0.65], alpha=0.6,
                           linewidth=0.8, markersize=4)
        ax[0].errorbar(1.0 / freqs, S.MTdat.TEphase,
                       yerr=S.MTdat.TEphaseErr,
                       fmt='o', color='C3', mfc='none', mec='C3',
                       ecolor='k', capsize=3, markersize=8, linewidth=1.6,
                       label='data', zorder=10)
        ax[1].errorbar(1.0 / freqs, S.MTdat.TEappRes,
                       yerr=S.MTdat.TEappResErr,
                       fmt='o', color='C3', mfc='none', mec='C3',
                       ecolor='k', capsize=3, markersize=8, linewidth=1.6,
                       label='data', zorder=10)
        ax[0].set_xlabel('Period (s)'); ax[0].set_ylabel('Phase (deg)')
        ax[1].set_xlabel('Period (s)'); ax[1].set_ylabel(r'$\log_{10}\rho_a$ (Ohm-m)')
        # mark the physical range [0, 90] for phase so outliers stand out
        ax[0].axhspan(0, 90, color='lightblue', alpha=0.15,
                      label='physical range [0, 90 deg]')
        ax[0].grid(True, which='both', alpha=0.3)
        ax[1].grid(True, which='both', alpha=0.3)
        ax[0].set_title('Phase: posterior responses vs data')
        ax[1].set_title('Apparent resistivity: posterior responses vs data')
        ax[0].legend(loc='best', fontsize=9)
        ax[1].legend(loc='best', fontsize=9)
        fig.tight_layout()
        figs.append(fig)

    if DCR:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for l in range(0, NtoCalc, skip):
            ax2.loglog(es, ModEnsDCRappRes[l, :], '-',
                       color=[0.7, 0.7, 0.7], alpha=0.5)
        ax2.errorbar(es, S.dcrDat.appRes,
                     yerr=2 * S.dcrDat.appResErr,
                     fmt='o', color='C3', mfc='C3', mec='k', ecolor='k')
        ax2.set_xlabel('electrode spacing (m)')
        ax2.set_ylabel('apparent resistivity')
        figs.append(fig2)

    if obCSEM:
        # ----- obCSEM response plot: log10|E| vs source-receiver offset -----
        # After the reciprocity swap, S.obTx is the single seafloor Rx and
        # S.obRx is the (many) original moving-Tx positions.  Offset is the
        # horizontal distance between the seafloor Rx and each moving Tx.
        real_tx_x = float(np.asarray(S.obTx.X).ravel()[0])
        real_tx_y = float(np.asarray(S.obTx.Y).ravel()[0])
        vrx_x = np.asarray(S.obRx.X)[:, 0].astype(float)
        vrx_y = np.asarray(S.obRx.Y)[:, 0].astype(float)
        offsets = np.hypot(vrx_x - real_tx_x, vrx_y - real_tx_y)
        # sort by offset for cleaner ensemble lines
        order = np.argsort(offsets)

        Er    = np.asarray(S.obDat.Er,    dtype=float)
        ErErr = np.asarray(S.obDat.ErErr, dtype=float)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(ob_freqs)))
        for kf, fHz in enumerate(ob_freqs):
            # ensemble: thin grey star-marker curves (one per sample)
            for l in range(0, NtoCalc, skip):
                ax3.plot(offsets[order], ModEnsObAmp[l, kf, order], '-*',
                         color=[0.65, 0.65, 0.65], alpha=0.35,
                         linewidth=0.6, markersize=3, zorder=2)
            # data: coloured open circles with errorbars
            data_vals = Er[kf]
            data_errs = ErErr[kf]
            valid = ~np.isnan(data_vals)
            ax3.errorbar(offsets[valid], data_vals[valid],
                         yerr=data_errs[valid],
                         fmt='o', color=cmap[kf], mfc='none',
                         mec=cmap[kf], ecolor='k', capsize=3,
                         markersize=7, linewidth=1.4,
                         label=f'{fHz:.3g} Hz', zorder=10)
        ax3.set_xlabel('Source-receiver offset (m)')
        ax3.set_ylabel(r'$\log_{10}|E|$ (V/Am)')
        ax3.set_title(f'obCSEM: posterior responses vs data '
                      f'({_getfield(S.obDat, "observable", "inline_E")})')
        ax3.grid(True, which='both', alpha=0.3)
        ax3.legend(loc='best', fontsize=9)
        fig3.tight_layout()
        figs.append(fig3)

    if stCSEM:
        # ----- stCSEM response plot: |E| vs offset, one panel per sounding ---
        # If there are multiple soundings (Tx tow positions), show them as
        # subplot columns; otherwise a single panel.
        st_freqs = np.asarray(S.stDat.Freqs).ravel()
        soundings = np.asarray(S.stTx.Soundings).ravel()
        nS = len(soundings)
        # offset = horizontal distance between Tx(sounding j) and each Rx
        st_tx_x = np.asarray(S.stTx.X).ravel()
        st_tx_y = np.asarray(S.stTx.Y).ravel()
        st_rx_x = np.asarray(S.stRx.X)
        st_rx_y = np.asarray(S.stRx.Y)
        Er    = np.asarray(S.stDat.Er,    dtype=float)
        ErErr = np.asarray(S.stDat.ErErr, dtype=float)

        fig4, ax4 = plt.subplots(1, max(nS, 1), figsize=(6 * max(nS, 1), 5),
                                 squeeze=False)
        cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(st_freqs)))
        for js in range(nS):
            offs = np.hypot(st_rx_x[:, js] - st_tx_x[js],
                            st_rx_y[:, js] - st_tx_y[js])
            order = np.argsort(offs)
            for kf, fHz in enumerate(st_freqs):
                for l in range(0, NtoCalc, skip):
                    ax4[0, js].semilogy(offs[order],
                                        10.0 ** ModEnsStAmp[l, kf, order, js],
                                        '-', color=[0.65, 0.65, 0.65],
                                        alpha=0.35, linewidth=0.6, zorder=2)
                data_vals = Er[kf, :, js] if Er.ndim == 3 else Er[kf]
                data_errs = ErErr[kf, :, js] if ErErr.ndim == 3 else ErErr[kf]
                valid = ~np.isnan(data_vals)
                # plot data as log-space amplitude (the inversion uses log10
                # but we display linear |E| with a log y-axis so the visual
                # is comparable to the standard CSEM offset plot)
                ax4[0, js].errorbar(offs[valid], 10.0 ** data_vals[valid],
                                    yerr=(10.0 ** data_vals[valid])
                                         * np.log(10) * data_errs[valid],
                                    fmt='o', color=cmap[kf], mfc='none',
                                    mec=cmap[kf], ecolor='k', capsize=3,
                                    markersize=7, linewidth=1.4,
                                    label=f'{fHz:.3g} Hz', zorder=10)
            ax4[0, js].set_xlabel('Source-receiver offset (m)')
            ax4[0, js].set_ylabel(r'$|E|$ (V/Am)')
            ax4[0, js].set_title(f'stCSEM sounding {soundings[js]}')
            ax4[0, js].grid(True, which='both', alpha=0.3)
            ax4[0, js].legend(loc='best', fontsize=9)
        fig4.tight_layout()
        figs.append(fig4)

    return figs


# ------------------------ RMShistogramsPlotting.m --------------------

def RMShistogramsPlotting(MT_file, CSEM_file, Joint_file, burnIn=int(3e5),
                          out_goodfit_file=None):
    """
    Plot RMS-misfit histograms for MT-only / CSEM-only / Joint runs, and
    return the list of sample indices in the joint ensemble that fit
    both MT and CSEM better than the joint median of each.
    """
    with open(MT_file,   "rb") as fh: MT    = pickle.load(fh)
    with open(CSEM_file, "rb") as fh: CSEM  = pickle.load(fh)
    with open(Joint_file,"rb") as fh: Joint = pickle.load(fh)

    color1 = (0.4660, 0.6740, 0.1880)
    color2 = (0.8500, 0.3250, 0.0980)
    color3 = (0.0000, 0.4470, 0.7410)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(MT["indivRMSs_ll"][burnIn:, 0],  color=color1, label='MT-only')
    ax[0].hist(Joint["indivRMSs_ll"][burnIn:, 1], color=color3, alpha=0.6,
               label='Joint')
    ax[0].set_xlabel('RMS'); ax[0].legend()

    ax[1].hist(CSEM["indivRMSs_ll"][burnIn:, 0],  color=color2, label='CSEM-only')
    ax[1].hist(Joint["indivRMSs_ll"][burnIn:, 0], color=color3, alpha=0.6,
               label='Joint')
    ax[1].set_xlabel('RMS'); ax[1].legend()

    ax[2].hist(Joint["en_ll"][burnIn:, 1], color=color3, label='Joint total')
    ax[2].set_xlabel('RMS'); ax[2].legend()
    fig.tight_layout()

    # pick out good-fit indices (both CSEM & MT below joint medians)
    medMT   = np.median(Joint["indivRMSs_ll"][burnIn:, 1])
    medCSEM = np.median(Joint["indivRMSs_ll"][burnIn:, 0])
    N       = len(Joint["indivRMSs_ll"]) - burnIn
    rng = range(burnIn, burnIn + N)
    goodFit = [j for j in rng
               if Joint["indivRMSs_ll"][j, 0] < medCSEM and
                  Joint["indivRMSs_ll"][j, 1] < medMT]
    goodFit = np.asarray(goodFit, dtype=int)

    if out_goodfit_file is not None and goodFit.size:
        out = dict(
            s_ll=[Joint["s_ll"][j]    for j in goodFit],
            k1_ll=np.asarray(Joint["k1_ll"]).ravel()[goodFit]
                  if Joint.get("k1_ll") is not None else None,
            k2_ll=np.asarray(Joint["k2_ll"]).ravel()[goodFit],
        )
        with open(out_goodfit_file, "wb") as fh:
            pickle.dump(out, fh)

    return goodFit


# =====================================================================
# 5.  Quick self-test for the MT forward (no externals required)
# =====================================================================
if __name__ == "__main__":
    # three-layer MT test
    rho   = np.array([100.0, 10.0, 1000.0])
    h     = np.array([500.0, 2000.0])
    freqs = np.logspace(-3, 2, 20)
    appRes, phase, Z = MT1D(rho, h, freqs)
    print("MT1D self-test:")
    print("  periods (s) :", 1.0 / freqs[:4], "...")
    print("  app. res    :", appRes[:4], "...")
    print("  phase (deg) :", phase[:4],  "...")
    print("OK")
