"""SABRE aging test: coagulation-only evolution of the Alaska 310-320 ppbv PSD.

Initializes TOMAS-JAX from the median SABRE size distribution in the
310-320 ppbv N2O bin (Alaska, >50N), at the median temperature and pressure
of that same bin, and integrates coagulation only for ~3 years.

The observed N2O-binned sequence (310-320 -> 220-230 ppbv) is the validation
target: from the observed stratospheric N2O trend, 315 -> 225 ppbv takes
~3 years, so each 10-ppbv step maps to ~1/3 year of aging. If the box model
is behaving, the modelled distribution at t = age(N2O) should track the
observed distribution in that N2O bin.

Units note: the AMP dN/dlog10Dp is reported for *dry* particles (<40% RH) at
STP (273.15 K, 1013 hPa). Coagulation rates go as (ambient number density)^2,
so the observed values are converted to ambient density before integration and
converted back to STP for comparison. At 142 hPa / 218 K that factor is ~0.176
(a 5.7x reduction) - skipping it would overestimate the coagulation rate by ~32x.

Phase: all initial mass goes into SO4 with no aerosol water, so the
coagulation-only run is dry throughout and directly comparable to the dry AMP
PSD. The nucl/cond runs call calc_equilibrium_water, which adds ~2% of the dry
mass as water at RH = 1.5% (reported as H2O/dry in the table below); the
diameters printed and saved here map *total* bin mass through `--density`, so
they are inflated by ~0.7% relative to a dry diameter in those runs. See the
"Dry, wet and STP" section of docs/sabre_aging_test.md.
"""
import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
# Ahead of any editable install of the sibling tomas-jax checkout.
sys.path.insert(0, _REPO)
_DEFAULT_PSD = os.path.join(_REPO, 'alaska_290-300_psd.csv')

import tomas_jax.core.config as config  # noqa: E402  (must precede JAX import)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tomas_jax.core.config import (  # noqa: E402
    ICOMP, ICOMP_NODIAG, SRTSO4, SRTSO2, SRTH2O, N_GAS_SPECIES, MW_H2SO4,
    MW_SO2, AVOGADRO, KB)
from tomas_jax.physics.so2_chemistry import (  # noqa: E402
    calc_solar_zenith_angle, calc_oh_concentration)
from tomas_jax.solvers.diffrax import coag_euler_step  # noqa: E402
from tomas_jax.physics.sedimentation import (  # noqa: E402
    sedimentation_step, calc_settling_velocity)
from tomas_jax.solvers.condensation import make_step  # noqa: E402
from run_box_model import load_measured_distribution, measured_to_Nk_Mk  # noqa: E402
from sabre_metrics import effective_dp, dry_scale  # noqa: E402

T_STP = 273.15      # K    (AMP file header)
P_STP = 1013.0e2    # Pa   (AMP file header)
# kg/m3, sulfate - matches the TOMAS grid construction (config.XK0). NOTE this
# is *not* the density the microphysics uses internally: physics/density.py
# (Tang 1997, Fortran aerodens.f) returns 1902 kg/m3 for pure sulfate and
# 1000 kg/m3 for any bin holding < 1e-15 kg/cell. See --density and the
# "Dry, wet and STP" section of docs/sabre_aging_test.md.
DENSITY = 1770.0
DENSITY_AERODENS = 1902.0   # what calc_density() returns for pure SO4
AERODENS_EMPTY_KG = 1e-15   # aerodens.f dry/empty clamp [kg/cell]
SEC_PER_YEAR = 365.25 * 86400.0

# Lower-stratosphere defaults (~100 hPa, 16 km, high latitude). These are NOT
# the run_box_model.py defaults, which are polluted-boundary-layer values and
# are 5-6 orders of magnitude too large here.
#
# H2SO4 production, calibrated against the observations rather than assumed.
#
# The a-priori estimate from SO2 + OH is ~50 molec/cm3/s (at 100 hPa / 219 K,
# background SO2 ~50 pptv = 1.7e8 molec/cm3, k(SO2+OH) ~ 1e-12 cm3/molec/s,
# diurnally averaged [OH] ~1e6 cm-3) and it is far too large for this box. The
# SABRE aerosol sulfur burden grows only +11% (to +28% at its 3.08 yr peak) over
# the 1.77 yr of the overworld sequence, i.e. a NET condensable supply of
# 1.4-7 molec/cm3/s. Forcing the box with 50 delivers 4x the entire existing
# burden in 1.77 yr, which the model disposes of as a spurious nucleation mode
# plus a condensational growth wave - two peaks below the observed mode that do
# not exist in the data. See the "Why 5 and not 50" section of
# docs/sabre_aging_test.md.
#
# The gap is not a contradiction: a box has no sink, whereas the real
# stratosphere removes sulfur by sedimentation and transport, so only the *net*
# accumulation is available to be constrained here.
#
# The default is set by ONE stated criterion: reproduce the observed net burden
# growth over the sequence (+11.0%). 1.4 gives +12.5%, a 1.01x match; 0 gives
# +2.1% and 5 gives +39%. The early-sequence growth would support up to ~7, so
# treat 1.4-7 as the bracket - `sabre_source_sweep.py` shows every conclusion
# here is insensitive across it, and that no value in 0-150 fits N and mass
# together.
H2SO4_PROD_STRAT = 1.4     # molec/cm3/s (net, matches observed burden growth)
# Finite-reservoir alternative to the constant production rate above. Background
# lower-stratospheric SO2 is ~20-70 pptv; oxidising a fixed reservoir is
# self-limiting, whereas a constant production rate is an infinite source.
SO2_STRAT_PPTV = 200.0     # pptv
OH_STRAT = 1.0e6           # molec/cm3 (noon peak with --diurnal, else constant)
RH_STRAT = 0.015           # 5 ppmv H2O at 219 K / 100 hPa, w.r.t. liquid
FION_STRAT = 20.0          # ion pairs/cm3/s, galactic cosmic rays at ~16 km
NH3_STRAT = 0.0            # NH3 is tropospheric; ~0 above the tropopause
ORG_STRAT = 0.0            # ditto oxidised organics -> leaves binary H2SO4-H2O
# Depth the settling flux is spread over. The aerosol layer scale height in the
# high-latitude lower stratosphere is a few km; 5 km is the round number.
LAYER_DEPTH_STRAT = 5.0e3  # m


def bin_edges_dp(xk, density=DENSITY):
    """Mass bin boundaries [kg] -> diameter boundaries [m]."""
    return np.cbrt(np.asarray(xk) / density * (6.0 / np.pi))


def nk_to_dndlog10dp(Nk, xk, boxvol, density=DENSITY):
    """Per-cell bin numbers -> dN/dlog10Dp [#/cm3] on the model grid.

    Returns (Dp_mid [m], dN/dlog10Dp).
    """
    Dp_edges = bin_edges_dp(xk, density)
    dlog10 = np.diff(np.log10(Dp_edges))
    Dp_mid = np.sqrt(Dp_edges[:-1] * Dp_edges[1:])
    return Dp_mid, np.asarray(Nk) / boxvol / dlog10


def integrate_number(Dp_mid, dndlog10dp, dp_lo=None, dp_hi=None):
    """Integrate dN/dlog10Dp over log10(Dp), optionally within [dp_lo, dp_hi]."""
    m = np.ones_like(Dp_mid, dtype=bool)
    if dp_lo is not None:
        m &= Dp_mid >= dp_lo
    if dp_hi is not None:
        m &= Dp_mid <= dp_hi
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return trapz(dndlog10dp[m], np.log10(Dp_mid[m]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--psd-csv', default=_DEFAULT_PSD,
                   help='Initial dN/dlog10Dp CSV [Dp nm, dN/dlog10Dp at STP]')
    p.add_argument('--temp', type=float, default=218.33,
                   help='Temperature [K] (median of the 310-320 ppbv bin)')
    p.add_argument('--pres', type=float, default=142.30e2,
                   help='Pressure [Pa] (median of the 310-320 ppbv bin)')
    p.add_argument('--years', type=float, default=3.0, help='Run length [years]')
    p.add_argument('--dt', type=float, default=3600.0, help='Timestep [s]')
    p.add_argument('--n-snapshots', type=int, default=10,
                   help='Evenly spaced output times (matches the 10 N2O bins)')
    p.add_argument('--substeps', type=int, default=3,
                   help='Euler substeps (+MNFIX) per timestep')
    p.add_argument('--no-stp-correction', action='store_true',
                   help='Treat the input as ambient number density (debug only)')
    p.add_argument('--density', type=float, default=DENSITY,
                   help=f'Density [kg/m3] used to map the observed dry diameters '
                        f'onto the mass grid and back (default: {DENSITY:g}, the '
                        f'grid-construction value). Pass '
                        f'{DENSITY_AERODENS:g} to match what the microphysics '
                        f'itself uses for pure sulfate (physics/density.py).')
    p.add_argument('--boxvol', type=float, default=1.0e6,
                   help='Box volume [cm3] (default: 1e6 = 1 m3). The physics is '
                        'boxvol-invariant except through the aerodens.f '
                        f'{AERODENS_EMPTY_KG:g} kg/cell dry clamp, which forces '
                        'rho=1000 in sparsely populated bins; raise this to test '
                        'that artefact.')
    p.add_argument('--processes',
                   choices=['coag', 'nucl_coag_cond', 'so2_nucl_coag_cond'],
                   default='coag',
                   help="'coag' (default) reproduces the coagulation-only test; "
                        "'nucl_coag_cond' adds binary nucleation + condensation "
                        'fed by a constant H2SO4 production rate; '
                        "'so2_nucl_coag_cond' instead starts from a finite SO2 "
                        'reservoir oxidised by OH (see --so2 / --oh / --diurnal), '
                        'which is self-limiting rather than an infinite source.')
    p.add_argument('--so2', type=float, default=SO2_STRAT_PPTV,
                   help=f'Initial SO2 mixing ratio [pptv] for so2_nucl_coag_cond '
                        f'(default: {SO2_STRAT_PPTV:g})')
    p.add_argument('--oh', type=float, default=OH_STRAT,
                   help=f'OH concentration [molec/cm3]. With --diurnal this is the '
                        f'noon peak; without it, the constant value '
                        f'(default: {OH_STRAT:g})')
    p.add_argument('--diurnal', action='store_true',
                   help='Modulate OH by max(0, cos(SZA)) instead of holding it '
                        'constant (see --lat / --doy)')
    p.add_argument('--lat', type=float, default=65.0,
                   help='Latitude [deg] for the SZA cycle (default: 65, Alaska)')
    p.add_argument('--doy', type=int, default=75,
                   help='Day of year for the SZA cycle (default: 75, mid-March)')
    p.add_argument('--h2so4-prod', type=float, default=H2SO4_PROD_STRAT,
                   help=f'H2SO4 production [molec/cm3/s] (default: '
                        f'{H2SO4_PROD_STRAT:g}, lower-stratosphere estimate)')
    p.add_argument('--rh', type=float, default=RH_STRAT,
                   help=f'Relative humidity as a fraction (default: {RH_STRAT})')
    p.add_argument('--fion', type=float, default=FION_STRAT,
                   help=f'Ion-pair production [pairs/cm3/s] (default: {FION_STRAT:g})')
    p.add_argument('--sediment', action='store_true',
                   help='Add gravitational settling as a first-order loss. NOTE: '
                        'pure loss is the wrong sign below the aerosol layer peak '
                        '(see docs/sedimentation.md); at 15-16 km the real flux '
                        'divergence is likely a coarse-mode source.')
    p.add_argument('--layer-depth', type=float, default=LAYER_DEPTH_STRAT,
                   help=f'Box depth [m] the settling flux is spread over '
                        f'(default: {LAYER_DEPTH_STRAT:g})')
    p.add_argument('--tp-track', action='store_true',
                   help='Follow the observed T and P along the N2O sequence '
                        'instead of holding the start-bin values. The parcel is '
                        'treated as fixed-mass, so it expands as P falls '
                        '(boxvol ~ T/P) and its number density drops, which slows '
                        'coagulation. Needs --start-bin, --env-csv, --airmass-csv.')
    p.add_argument('--start-bin', default='290-300',
                   help='N2O bin the run starts from; its mean age anchors the '
                        '--tp-track interpolation (default: 290-300)')
    p.add_argument('--tp-quantile', choices=['median', 'p25', 'p75', 'minrho'],
                   default='median',
                   help='Which per-bin T/P statistic --tp-track follows. Each N2O '
                        'bin spans a wide altitude range (P quartiles 82-162 hPa), '
                        'so the representative value is itself a choice. "minrho" '
                        'takes P at its 25th percentile and T at its 75th, the '
                        'lowest air density the data support and therefore the '
                        'slowest coagulation (default: median).')
    p.add_argument('--env-csv', default=None,
                   help='alaska_n2o_bin_env.csv (per-bin median T/P). '
                        'Default: alongside the repo')
    p.add_argument('--airmass-csv', default=None,
                   help='alaska_n2o_bin_airmass.csv (mean age, theta). '
                        'Default: alongside the repo')
    p.add_argument('--out', default='sabre_coag_output.npz', help='Output .npz')
    args = p.parse_args()

    boxvol = args.boxvol
    density = args.density
    xk = config.xk_boundaries()

    # ── optional T/P track along the observed N2O sequence ────────────────────
    # The air rises and cools/warms across the sequence: over the overworld bins
    # P falls 116.6 -> 92-98 hPa while T rises 219.6 -> 221.3 K, so the ambient
    # number density drops ~17-22%. Coagulation goes as (number density)^2, so
    # holding the start-bin values overstates it.
    track = None
    if args.tp_track:
        import pandas as pd
        _outside = _REPO
        env = pd.read_csv(args.env_csv
                          or os.path.join(_outside, 'alaska_n2o_bin_env.csv'))
        env['bin'] = env['n2o_lo'].astype(str) + '-' + env['n2o_hi'].astype(str)
        env = env.set_index('bin')
        am = pd.read_csv(args.airmass_csv
                         or os.path.join(_outside, 'alaska_n2o_bin_airmass.csv')
                         ).set_index('bin')
        if args.start_bin not in am.index:
            raise KeyError(f'--start-bin {args.start_bin} not in the airmass table')
        t0 = float(am.loc[args.start_bin, 'age_med'])
        # minrho: lowest density the quartiles allow (low P, high T) -> slowest
        # coagulation, i.e. the most generous case for the model.
        t_col, p_col = {
            'median': ('T_median_K', 'P_median_hPa'),
            'p25': ('T_p25_K', 'P_p25_hPa'),
            'p75': ('T_p75_K', 'P_p75_hPa'),
            'minrho': ('T_p75_K', 'P_p25_hPa'),
        }[args.tp_quantile]
        # Only the overworld branch at or after the start bin is one air mass.
        rows = [(float(am.loc[b, 'age_med']), float(env.loc[b, t_col]),
                 float(env.loc[b, p_col]) * 100.0)
                for b in am.index
                if b in env.index and float(am.loc[b, 'theta']) >= 400.0
                and float(am.loc[b, 'age_med']) >= t0]
        rows.sort()
        if len(rows) < 2:
            raise ValueError(
                f'Need >=2 overworld bins at or after {args.start_bin} for '
                f'--tp-track; got {len(rows)}.')
        ages_tr = jnp.asarray([r[0] for r in rows])
        temp_tr = jnp.asarray([r[1] for r in rows])
        pres_tr = jnp.asarray([r[2] for r in rows])
        # Start from the track, not from --temp/--pres, so the two are consistent
        args.temp, args.pres = rows[0][1], rows[0][2]
        track = (t0, ages_tr, temp_tr, pres_tr)

    # ── initial distribution ──────────────────────────────────────────────────
    Dp_m, dndlog10_stp = load_measured_distribution(args.psd_csv, 'nm')

    # STP -> ambient number density
    stp_to_amb = 1.0 if args.no_stp_correction else \
        (args.pres / P_STP) * (T_STP / args.temp)
    dndlog10_amb = dndlog10_stp * stp_to_amb

    Nk_cm3, Mk_cm3 = measured_to_Nk_Mk(Dp_m, dndlog10_amb, xk, density=density)
    Nk = jnp.asarray(Nk_cm3) * boxvol
    Mk = jnp.asarray(Mk_cm3) * boxvol

    Dp_mid, _ = nk_to_dndlog10dp(np.asarray(Nk), xk, boxvol, density)
    obs_lo, obs_hi = Dp_m.min(), Dp_m.max()

    # How much of the distribution sits under the aerodens.f dry clamp, where the
    # microphysics assumes rho = 1000 kg/m3 instead of 1902 and therefore sees a
    # collision diameter 1.24x larger than the one plotted here.
    m_bin = np.asarray(jnp.sum(Mk, axis=1))
    clamped = (m_bin > 0) & (m_bin < AERODENS_EMPTY_KG)
    n_clamped = float(np.sum(np.asarray(Nk)[clamped]))

    print('=' * 70)
    print('SABRE coagulation-only aging run')
    print('=' * 70)
    print(f'  Initial PSD : {args.psd_csv} ({len(Dp_m)} bins, '
          f'{obs_lo * 1e9:.1f}-{obs_hi * 1e9:.0f} nm)')
    if track is None:
        print(f'  T / P       : {args.temp:.2f} K / {args.pres / 100:.2f} hPa '
              f'(fixed)')
    else:
        t0, ages_tr, temp_tr, pres_tr = track
        print(f'  T / P track : observed ({args.tp_quantile}), {len(ages_tr)} '
              f'overworld bins, age {float(ages_tr[0]):.2f} -> '
              f'{float(ages_tr[-1]):.2f} yr')
        print('                ' + '  '.join(
            f'{float(a):.2f}yr:{float(T):.1f}K/{float(P) / 100:.0f}hPa'
            for a, T, P in zip(ages_tr, temp_tr, pres_tr)))
        n_end = (float(pres_tr[-1]) / float(pres_tr[0])) \
            * (float(temp_tr[0]) / float(temp_tr[-1]))
        print(f'                fixed-mass parcel expands; ambient number '
              f'density x{n_end:.3f} by the end '
              f'(min x{float(jnp.min(pres_tr / temp_tr)) / float(pres_tr[0] / temp_tr[0]):.3f})')
    print(f'  STP->ambient: x{stp_to_amb:.4f}'
          + ('  [DISABLED]' if args.no_stp_correction else ''))
    print(f'  Grid        : {len(xk) - 1} bins, '
          f'{bin_edges_dp(xk, density)[0] * 1e9:.2f} nm - '
          f'{bin_edges_dp(xk, density)[-1] * 1e6:.1f} um')
    print(f'  Density     : {density:.1f} kg/m3 (obs Dp <-> mass, dry). '
          f'Microphysics uses {DENSITY_AERODENS:.0f} above / 1000 below '
          f'{AERODENS_EMPTY_KG:g} kg/cell')
    print(f'  Box volume  : {boxvol:.3g} cm3; {int(clamped.sum())} populated bins '
          f'under the aerodens clamp, holding '
          f'{n_clamped / boxvol / stp_to_amb:.3f} #/cm3 STP '
          f'({100 * n_clamped / float(jnp.sum(Nk)):.2f}% of N)')
    print(f'  N_total     : {float(jnp.sum(Nk)) / boxvol:.3f} #/cm3 ambient  '
          f'= {float(jnp.sum(Nk)) / boxvol / stp_to_amb:.1f} #/cm3 STP')
    # kg/cm3 -> ug/m3 is x1e15 (1e6 cm3/m3 x 1e9 ug/kg)
    print(f'  Mass        : {float(jnp.sum(Mk)) / boxvol * 1e15:.4f} ug/m3 ambient '
          f'(dry: all in SO4, matching the AMP dry PSD)')
    print(f'  Run         : {args.years} yr, dt={args.dt:.0f} s, '
          f'{args.substeps} substeps/step')

    # ── time stepping: scan-fused coagulation, snapshots at even intervals ────
    total_steps = int(round(args.years * SEC_PER_YEAR / args.dt))
    steps_per_snap = total_steps // args.n_snapshots
    dt = jnp.float64(args.dt)
    temp, pres = jnp.float64(args.temp), jnp.float64(args.pres)
    boxvol_j = jnp.float64(boxvol)

    def state_at(istep):
        """(T, P, boxvol) at step `istep`. Constant unless --tp-track.

        Fixed-mass parcel: boxvol ~ T/P, so the per-cell Nk/Mk/Gc are untouched by
        expansion while the number density they represent falls. The STP
        diagnostics stay valid unchanged because boxvol(t) * stp_to_amb(t) is
        exactly boxvol(0) * stp_to_amb(0).
        """
        if track is None:
            return temp, pres, boxvol_j
        t0_, ages_, temp_, pres_ = track
        age = t0_ + istep * dt / SEC_PER_YEAR
        T = jnp.interp(age, ages_, temp_)      # clamps at the track endpoints
        P = jnp.interp(age, ages_, pres_)
        bv = boxvol_j * (pres_[0] / P) * (T / temp_[0])
        return T, P, bv

    so2_mode = args.processes == 'so2_nucl_coag_cond'
    full_physics = args.processes in ('nucl_coag_cond', 'so2_nucl_coag_cond')
    # H2SO4 production [molec/cm3/s] -> [kg/cell/s]. Only the nucl_coag_cond path
    # applies it: SO2 mode oxidises a finite reservoir instead, and coag-only has
    # no gas phase at all. It must be zero in those cases or the sulfur budget
    # below charges the run for production that never happened. Under --tp-track
    # the parcel expands, so a fixed per-volume rate makes more mass per cell over
    # time - hence a per-step function, and the produced total is accumulated in
    # the carry rather than assumed linear in t.
    prod_active = full_physics and not so2_mode

    def prod_rate(bv):
        if not prod_active:
            return jnp.float64(0.0)
        return args.h2so4_prod * bv * (MW_H2SO4 / 1000.0) / AVOGADRO

    Gc = jnp.zeros(N_GAS_SPECIES)

    # Initial SO2 reservoir: pptv -> molec/cm3 -> kg/cell
    n_air_cm3 = args.pres / (KB * args.temp) / 1.0e6
    so2_cm3 = args.so2 * 1.0e-12 * n_air_cm3
    if so2_mode:
        Gc = Gc.at[SRTSO2].set(so2_cm3 * boxvol * (MW_SO2 / 1000.0) / AVOGADRO)

    # Gravitational settling as a pure first-order loss (nothing above the box).
    # This is the wrong sign below the aerosol layer peak - see the flag help
    # and docs/sedimentation.md. Cumulative removed mass is tracked so the
    # sulfur budget still closes.
    depth = jnp.float64(args.layer_depth)

    def apply_sed(Nk_c, Mk_c, sed, T, P):
        if not args.sediment:
            return Nk_c, Mk_c, sed
        Nk_n, Mk_n, _, m_lost = sedimentation_step(
            Nk_c, Mk_c, xk, T, P, dt, depth,
            jnp.zeros_like(Nk_c), jnp.zeros_like(Mk_c), density=density)
        return Nk_n, Mk_n, sed + jnp.sum(m_lost, axis=0)

    if args.sediment:
        dp_probe = jnp.array([50e-9, 100e-9, 500e-9])
        v_probe = calc_settling_velocity(dp_probe, temp, pres, density)
        print(f'  Sedimentation: ON, layer depth {args.layer_depth / 1e3:.1f} km '
              f'(PURE LOSS - wrong sign below the layer peak)')
        print('    v_s @ 50/100/500 nm: '
              + ' / '.join(f'{v * SEC_PER_YEAR / 1e3:.2f}' for v in v_probe)
              + ' km/yr')

    if full_physics:
        procs = (['so2_chemistry'] if so2_mode else []) \
            + ['nucleation', 'coagulation', 'condensation']
        step_fn = make_step(procs, cond_method='ppm_jit')
        nucl_kw = dict(org_conc=jnp.float64(ORG_STRAT),
                       nh3_conc=jnp.float64(NH3_STRAT),
                       fion=jnp.float64(args.fion))
        print(f'  Processes   : {" + ".join(procs)}')
        if so2_mode:
            oh_mode = 'diurnal peak' if args.diurnal else 'constant'
            print(f'  SO2 initial : {args.so2:.4g} pptv = {so2_cm3:.3e} molec/cm3')
            print(f'  OH          : {args.oh:.3g} molec/cm3 ({oh_mode})'
                  + (f', lat {args.lat:.0f}N, doy {args.doy}' if args.diurnal else ''))
        else:
            print(f'  H2SO4 prod  : {args.h2so4_prod:.3g} molec/cm3/s')
        print(f'  RH / fion   : {args.rh:.3g} / {args.fion:.3g} pairs/cm3/s')
        print(f'  NH3 / org   : {NH3_STRAT:.3g} / {ORG_STRAT:.3g} molec/cm3 '
              f'(binary H2SO4-H2O nucleation)')

        use_diurnal = jnp.float64(1.0 if args.diurnal else 0.0)

        def one_step(carry, _):
            Nk_c, Mk_c, Gc_c, ovf, sed, prod, istep = carry
            T, P, bv = state_at(istep)
            if so2_mode:
                # OH follows the solar cycle at the flight latitude; the box is
                # fixed in space, so hour-of-day advances with the step index.
                hour = jnp.mod(istep * dt / 3600.0, 24.0)
                cos_sza = calc_solar_zenith_angle(
                    jnp.float64(args.lat), args.doy, hour)
                oh = calc_oh_concentration(
                    jnp.float64(args.oh), cos_sza, use_diurnal)
                kw = dict(nucl_kw, oh_conc=oh)
            else:
                # constant H2SO4 source, applied before the microphysics
                d_prod = prod_rate(bv) * dt
                Gc_c = Gc_c.at[SRTSO4].add(d_prod)
                prod = prod + d_prod
                kw = nucl_kw
            Nk_n, Mk_n, Gc_n = step_fn(
                Nk_c, Mk_c, Gc_c, xk, T, P, bv,
                jnp.float64(args.rh), jnp.float64(1.0), dt, **kw)
            Nk_n, Mk_n, sed = apply_sed(Nk_n, Mk_n, sed, T, P)
            return (Nk_n, Mk_n, Gc_n, ovf, sed, prod, istep + 1), None
    else:
        print(f'  Processes   : coagulation only')

        def one_step(carry, _):
            Nk_c, Mk_c, Gc_c, ovf, sed, prod, istep = carry
            T, P, bv = state_at(istep)
            Nk_n, Mk_n, d_ovf = coag_euler_step(
                Nk_c, Mk_c, xk, T, P, bv, dt,
                ICOMP_NODIAG, n_substeps=args.substeps, return_overflow=True)
            Nk_n, Mk_n, sed = apply_sed(Nk_n, Mk_n, sed, T, P)
            return (Nk_n, Mk_n, Gc_c, ovf + d_ovf, sed, prod, istep + 1), None

    @jax.jit
    def run_segment(Nk0, Mk0, Gc0, ovf0, sed0, prod0, istep0):
        carry, _ = jax.lax.scan(
            one_step, (Nk0, Mk0, Gc0, ovf0, sed0, prod0, istep0),
            None, length=steps_per_snap)
        return carry

    ovf = jnp.zeros(ICOMP)
    sed = jnp.zeros(ICOMP)
    prod = jnp.float64(0.0)
    istep = jnp.float64(0.0)
    M0 = float(jnp.sum(Mk))

    # Sulfur budget in moles of S, so SO2 -> H2SO4 (MW 64 -> 98) is conserved.
    mol_h2so4 = MW_H2SO4 / 1000.0
    mol_so2 = MW_SO2 / 1000.0

    def sulfur_moles(Mk_now, Gc_now, ovf_now, sed_now):
        return ((float(jnp.sum(Mk_now[:, SRTSO4])) + float(Gc_now[SRTSO4])
                 + float(ovf_now[SRTSO4]) + float(sed_now[SRTSO4])) / mol_h2so4
                + float(Gc_now[SRTSO2]) / mol_so2)

    S0 = sulfur_moles(Mk, Gc, ovf, sed)

    times, snaps_Nk, snaps_Mk = [0.0], [np.asarray(Nk)], [np.asarray(Mk)]
    snaps_Gc = [np.asarray(Gc)]
    # molec/cm3 per kg/cell, for reporting the gas phase
    gas_to_molec_cm3 = AVOGADRO / (MW_H2SO4 / 1000.0) / boxvol
    print(f'\n{"t [yr]":>7} | {"N (STP) [#/cm3]":>15} | {"Dp_peak [nm]":>12} | '
          f'{"Deff [nm]":>9} | {"H2SO4 [cm-3]":>12} | {"SO2 [cm-3]":>10} | '
          f'{"S budget":>9} | {"H2O/dry":>8}')
    print('-' * 107)

    def report(t_yr, Nk_now, Mk_now, Gc_now, ovf_now, sed_now, prod_now):
        _, dn = nk_to_dndlog10dp(np.asarray(Nk_now), xk, boxvol, density)
        # Read the model dry, like the AMP data: strip equilibrium water from the
        # diameter axis. Uniform factor, so dN/dlog10Dp is untouched.
        dp_dry = Dp_mid * dry_scale(Mk_now, SRTH2O)
        n_stp = integrate_number(dp_dry, dn, obs_lo, obs_hi) / stp_to_amb
        peak = dp_dry[np.argmax(dn)] * 1e9
        # effective diameter (M3/M2, Hansen & Travis 1974) over the measured range
        deff = effective_dp(dp_dry, dn, obs_lo, obs_hi) * 1e9
        h2so4 = float(Gc_now[SRTSO4]) * gas_to_molec_cm3
        # sulfur budget in moles of S: aerosol + gas (H2SO4 + SO2) + top-bin
        # overflow + settled out = initial + production
        s_in = S0 + float(prod_now) / mol_h2so4
        serr = (sulfur_moles(Mk_now, Gc_now, ovf_now, sed_now) - s_in) / s_in
        so2 = float(Gc_now[SRTSO2]) / mol_so2 * AVOGADRO / boxvol
        # Aerosol water relative to dry mass. The AMP PSD is *dry* (<40% RH), so
        # any water here makes the modelled diameters wet and the comparison
        # slightly unlike-for-like; condensation adds it via calc_equilibrium_water.
        m_dry = float(jnp.sum(Mk_now[:, :SRTH2O]))
        wet = float(jnp.sum(Mk_now[:, SRTH2O])) / max(m_dry, 1e-300)
        print(f'{t_yr:7.3f} | {n_stp:15.2f} | {peak:12.1f} | {deff:9.1f} | '
              f'{h2so4:12.3e} | {so2:10.3e} | {serr:9.2e} | {wet:8.4f}')

    report(0.0, Nk, Mk, Gc, ovf, sed, prod)
    t_wall = time.time()
    for i in range(args.n_snapshots):
        Nk, Mk, Gc, ovf, sed, prod, istep = run_segment(
            Nk, Mk, Gc, ovf, sed, prod, istep)
        t_yr = (i + 1) * steps_per_snap * args.dt / SEC_PER_YEAR
        times.append(t_yr)
        snaps_Nk.append(np.asarray(Nk))
        snaps_Mk.append(np.asarray(Mk))
        snaps_Gc.append(np.asarray(Gc))
        report(t_yr, Nk, Mk, Gc, ovf, sed, prod)
    print(f'\n  wall clock: {time.time() - t_wall:.1f} s '
          f'({total_steps} steps of {args.dt:.0f} s)')

    snaps_Nk = np.array(snaps_Nk)
    # model dN/dlog10Dp back in STP units, for direct comparison with the obs
    dndlog10_model_stp = np.array([
        nk_to_dndlog10dp(nk, xk, boxvol, density)[1] / stp_to_amb
        for nk in snaps_Nk])
    # Per-snapshot DRY diameter axis (identical to Dp_mid_m for a dry run). The
    # comparison scripts use this so the model is read dry, like the AMP PSD.
    Dp_mid_dry = np.array([Dp_mid * dry_scale(mk, SRTH2O) for mk in snaps_Mk])

    np.savez(args.out,
             time_years=np.array(times),
             Dp_mid_m=Dp_mid,
             Dp_mid_dry_m=Dp_mid_dry,
             Dp_edges_m=bin_edges_dp(xk, density),
             Nk=snaps_Nk,
             Mk=np.array(snaps_Mk),
             Gc=np.array(snaps_Gc),
             processes=args.processes,
             h2so4_prod=args.h2so4_prod, rh=args.rh, fion=args.fion,
             sediment=args.sediment, layer_depth=args.layer_depth,
             so2_pptv=args.so2, oh=args.oh, diurnal=args.diurnal,
             dndlog10dp_stp=dndlog10_model_stp,
             xk=np.asarray(xk),
             temp=args.temp, pres=args.pres, boxvol=boxvol, density=density,
             stp_to_amb=stp_to_amb, tp_track=args.tp_track,
             start_bin=args.start_bin,
             obs_dp_lo=obs_lo, obs_dp_hi=obs_hi,
             init_Dp_m=Dp_m, init_dndlog10_stp=dndlog10_stp)
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
