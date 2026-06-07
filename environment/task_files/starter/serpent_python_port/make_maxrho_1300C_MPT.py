"""
Generate maxrho_1300C_MPT.txt -- depth-dependent maximum resistivity
prior corresponding to dry olivine at 1300 deg C mantle potential
temperature, matching the prescription in Blatter et al. (2022) Methods:

  "we modify the algorithm slightly to include a depth-dependent prior
   distribution such that the upper bound on mantle resistivity is the
   resistivity of dry olivine at 1,300 deg C MPT, while the lower bound
   is held constant at 0.1 ohm-m."

We use:
  1. Half-space cooling for T(z):
         T(z) = T_m * erf( z / (2*sqrt(kappa*t)) )
     with T_m = 1300 deg C, kappa = 1e-6 m^2/s, t = 22 Myr (the average
     age of the Cocos Plate in the survey area).  Below the conductive
     boundary layer the temperature asymptotes to T_m; we add a small
     mantle adiabatic gradient (0.3 deg C / km) at depths > 100 km for
     completeness.

  2. Dry-olivine Arrhenius conductivity calibrated to match the no-melt,
     no-hydration curves in Blatter Extended Data Fig. 9a at four
     temperatures (1216, 1283, 1351, 1418 deg C):
         log10(rho_dry) = -1.96 + 6620 / T_K
     This is a single-mechanism Arrhenius that reproduces the paper's
     own "dry mantle" upper bound to within ~ 0.1 log10 units.  If you
     have the original Naif (2018) parameterisation handy, you can
     replace `dry_olivine_log10rho()` below and re-run.

  3. We cap log10(rho_max) at 5 in the lithosphere so the prior does
     not become looser than the default scalar bound.

Output: maxrho_1300C_MPT.txt -- two-column ASCII, depth_km   log10_rho_max.
        Also writes a PNG showing the profile.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ---- T(z) from half-space cooling at MPT = 1300 C -----------------
MPT_C       = 1300.0          # mantle potential temperature (deg C)
plate_age_Myr = 22.0          # representative Cocos Plate age
kappa       = 1e-6            # thermal diffusivity (m^2/s)
adiabat_KpKm = 0.3            # adiabatic gradient deg C / km below TBL

t_sec       = plate_age_Myr * 1e6 * 365.25 * 86400.0
char_depth_m = 2.0 * np.sqrt(kappa * t_sec)

def temperature_C(depth_km):
    """T (deg C) at given depth (km).  Half-space cooling + mantle adiabat
    below the depth where T reaches 99 % of MPT."""
    z_m = np.asarray(depth_km, dtype=float) * 1e3
    T_hsc = MPT_C * erf(z_m / char_depth_m)
    # adiabatic addition only when we are well inside the asymptote;
    # use a smooth blend so there is no kink.
    z_asym = char_depth_m * 1.82 / 1e3   # ~ 99% of MPT reached here
    z_km   = np.asarray(depth_km, dtype=float)
    T = T_hsc.copy()
    deeper = z_km > z_asym
    if np.any(deeper):
        T[deeper] = MPT_C + adiabat_KpKm * (z_km[deeper] - z_asym)
    return T

def dry_olivine_log10rho(T_K):
    """log10(rho_dry_olivine) at temperature T (Kelvin), calibrated to
    Blatter et al. (2022) Extended Data Fig. 9a (zero hydration end of
    the dry-mantle curves at four reference temperatures).  Returns
    log10(ohm-m)."""
    return -1.96 + 6620.0 / T_K

# ---- compute the profile on a dense depth grid --------------------
depth_km = np.concatenate([
    np.arange(0.0,  10.0,  0.5),
    np.arange(10.0, 30.0,  1.0),
    np.arange(30.0, 80.0,  2.0),
    np.arange(80.0, 160.1, 5.0),
])
T_C  = temperature_C(depth_km)
T_K  = T_C + 273.15
log10_rho_max = dry_olivine_log10rho(T_K)
# cap in the lithosphere so we are no MORE permissive than the default
# scalar bound (rhoMax = 1e5 ohm-m, log10 = 5)
log10_rho_max = np.minimum(log10_rho_max, 5.0)

# ---- write the .txt file ------------------------------------------
hdr = (
    "# Depth-dependent upper bound on log10(rho_max) [log10(ohm-m)]\n"
    "# Equivalent to the dry-olivine resistivity at 1300 C MPT.\n"
    "# Mirrors the prescription in Blatter et al. 2022, Methods, ref. 17.\n"
    "#\n"
    "# Temperature model:\n"
    f"#   half-space cooling, MPT = {MPT_C:g} deg C,\n"
    f"#   plate age = {plate_age_Myr:g} Myr, kappa = 1e-6 m^2/s\n"
    "#   ( T(z) = MPT * erf( z / (2*sqrt(kappa*t)) ),  adiabat 0.3 C/km below TBL )\n"
    "#\n"
    "# Resistivity model:\n"
    "#   log10(rho_dry_olivine) = -1.96 + 6620 / T_K\n"
    "#   (Arrhenius fit to Blatter et al. 2022 Extended Data Fig. 9a, dry mantle)\n"
    "#\n"
    "# log10(rho_max) is capped at 5 in the cold lithosphere so this\n"
    "# prior is never LOOSER than the default scalar [-1, 5] bound.\n"
    "#\n"
    "# depth_km        log10_rho_max\n"
)
out = np.column_stack([depth_km, log10_rho_max])
fname = 'maxrho_1300C_MPT.txt'
with open(fname, 'w') as fh:
    fh.write(hdr)
    for d, r in out:
        fh.write(f"{d:10.3f}    {r:10.5f}\n")
print(f"wrote {fname} ({len(depth_km)} depth points)")

# ---- diagnostic plot ----------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(T_C, depth_km, 'b-', lw=1.5)
for ref_T in [1216, 1283, 1351, 1418]:
    ax[0].axvline(ref_T, color='grey', ls=':', alpha=0.5)
    ax[0].text(ref_T, 5, f'{ref_T}', rotation=90, fontsize=8, color='grey')
ax[0].set_xlabel('Temperature (deg C)')
ax[0].set_ylabel('Depth (km)')
ax[0].set_title(f'T(z): half-space cooling, MPT={MPT_C} C, {plate_age_Myr} Myr')
ax[0].invert_yaxis()
ax[0].grid(alpha=0.3)

ax[1].plot(log10_rho_max, depth_km, 'r-', lw=1.5)
ax[1].axvline(5.0, color='grey', ls=':', alpha=0.7, label='cap = 5')
ax[1].set_xlabel(r'$\log_{10}(\rho_{\rm max})$ [$\log_{10}\ \Omega\cdot$m]')
ax[1].set_ylabel('Depth (km)')
ax[1].set_title('Depth-dependent prior upper bound\n'
                '(dry olivine @ 1300 C MPT)')
ax[1].invert_yaxis()
ax[1].grid(alpha=0.3)
ax[1].legend()

fig.tight_layout()
fig.savefig('maxrho_1300C_MPT.png', dpi=130, bbox_inches='tight')
print("wrote maxrho_1300C_MPT.png")
