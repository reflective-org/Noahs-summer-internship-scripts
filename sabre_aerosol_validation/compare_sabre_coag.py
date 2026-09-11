"""Compare the coagulation-only TOMAS-JAX run against the SABRE aging sequence.

Reads the .npz written by run_sabre_coag.py plus the observed N2O-binned median
PSDs, maps each N2O bin onto an age (315 -> 225 ppbv over ~3 years, i.e. 1/3 yr
per 10 ppbv step), and produces:

  fig1  observed vs modelled dN/dlog10Dp, colour = age
  fig2  integrated number, peak/mean diameter and mass vs age
  plus a printed table and the model's "coagulation speed factor" relative to
  the observed aging rate.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from sabre_metrics import effective_dp

trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# Fallback N2O -> age when no measured mean age is supplied: the stratospheric
# descent from 315 to 225 ppbv takes ~3 yr. Prefer --ages measured, which reads
# the SABRE mean-age product (Ray et al. 2024) instead of assuming linearity.
N2O_START, N2O_END, AGE_SPAN_YR = 315.0, 225.0, 3.0

# Potential temperature separating the extratropical lowermost stratosphere
# (troposphere-ventilated across isentropes) from the overworld (tropical pipe).
# Bins below this are not part of the same aging sequence.
THETA_OVERWORLD = 400.0


def n2o_to_age(n2o_center):
    return (N2O_START - n2o_center) / (N2O_START - N2O_END) * AGE_SPAN_YR


def process_label(d):
    """Human-readable process list for the run stored in `d` (the .npz).

    The figures are generated for more than one case, so the model label must
    come from the run itself - a hardcoded 'coag only' silently mislabels a
    nucleation+condensation run.
    """
    procs = str(d['processes']) if 'processes' in d.files else 'coag'
    label = {
        'coag': 'coagulation only',
        'nucl_coag_cond': 'nucleation + coagulation + condensation',
        'so2_nucl_coag_cond': 'SO$_2$ + nucleation + coagulation + condensation',
    }.get(procs, procs)
    if 'sediment' in d.files and bool(d['sediment']):
        label += ' + sedimentation'
    return label


def model_dp(d, j):
    """Dry diameter axis [m] for snapshot j.

    Runs with condensation carry equilibrium water, and TOMAS bins are mass bins,
    so the grid diameter is a wet one while the AMP PSD is dry. run_sabre_coag.py
    writes the per-snapshot dry axis; fall back to the grid axis for older files.
    """
    if 'Dp_mid_dry_m' in d.files:
        return d['Dp_mid_dry_m'][j]
    return d['Dp_mid_m']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default='sabre_coag_3yr.npz')
    p.add_argument('--obs-psd', default=None,
                   help='alaska_n2o_binned_psd.csv (default: alongside the repo)')
    p.add_argument('--obs-env', default=None, help='alaska_n2o_bin_env.csv')
    p.add_argument('--airmass', default=None,
                   help='alaska_n2o_bin_airmass.csv (measured mean age, theta, '
                        'vortex fraction). Required for --ages measured.')
    p.add_argument('--ages', choices=['measured', 'linear'], default='measured',
                   help='measured: use the SABRE mean-age product (default). '
                        'linear: assume 315->225 ppbv spans 3 yr uniformly.')
    p.add_argument('--start-bin', default='310-320',
                   help='N2O bin the model was initialized from; its measured '
                        'age becomes the model t=0 offset.')
    p.add_argument('--outdir', default='figures_sabre')
    p.add_argument('--tag', default='', help='suffix appended to figure names')
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    obs_psd_path = args.obs_psd or os.path.join(root, 'alaska_n2o_binned_psd.csv')
    obs_env_path = args.obs_env or os.path.join(root, 'alaska_n2o_bin_env.csv')
    airmass_path = args.airmass or os.path.join(root, 'alaska_n2o_bin_airmass.csv')
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.npz)
    proc_label = process_label(d)
    t_yr = d['time_years']
    # dry diameter axis, one row per snapshot [nm]
    dp_mod_all = np.array([model_dp(d, j) * 1e9 for j in range(len(t_yr))])
    psd_mod = d['dndlog10dp_stp']                # [n_snap, nbins], STP units
    dp_lo, dp_hi = d['obs_dp_lo'] * 1e9, d['obs_dp_hi'] * 1e9

    obs = pd.read_csv(obs_psd_path)
    env = pd.read_csv(obs_env_path)
    dp_obs = obs['size (nm)'].values
    bin_cols = [c for c in obs.columns if c != 'size (nm)']

    # ── ages: measured mean age, or the linear N2O assumption ────────────────
    theta = np.full(len(bin_cols), np.nan)
    if args.ages == 'measured':
        if not os.path.exists(airmass_path):
            raise FileNotFoundError(
                f'Missing {airmass_path}. Run sabre_diagnose_bins.py to build it, '
                f'or pass --ages linear to fall back to the 3-yr assumption.')
        am = pd.read_csv(airmass_path).set_index('bin')
        missing = [c for c in bin_cols if c not in am.index]
        if missing:
            raise KeyError(f'Bins absent from {airmass_path}: {missing}')
        ages_obs = am.loc[bin_cols, 'age_med'].values
        theta = am.loc[bin_cols, 'theta'].values
    else:
        ages_obs = np.array(
            [n2o_to_age((int(c.split('-')[0]) + int(c.split('-')[1])) / 2)
             for c in bin_cols])

    # The model starts at the age of the bin it was initialized from, not at 0.
    if args.start_bin not in bin_cols:
        raise KeyError(f'--start-bin {args.start_bin} not among {bin_cols}')
    t0 = ages_obs[bin_cols.index(args.start_bin)]
    t_abs = t_yr + t0
    print(f'Ages: {args.ages}. Model initialized from {args.start_bin} '
          f'(age {t0:.2f} yr); model t=0 -> absolute age {t0:.2f} yr.')
    if args.ages == 'measured':
        lms = [c for c, th in zip(bin_cols, theta) if th < THETA_OVERWORLD]
        if lms:
            print(f'WARNING: bins {lms} sit below theta={THETA_OVERWORLD:.0f} K '
                  f'(lowermost stratosphere). These are ventilated across '
                  f'isentropes from the troposphere and are NOT the same air '
                  f'mass as the overworld bins - excluded from the fit below.')

    # Restrict the model to the measured size range so the integrals are
    # comparable. Size is reported as effective diameter (M3/M2, Hansen & Travis
    # 1974) - the area-weighted size the radiation sees - not as a number- or
    # mass-weighted mean; see sabre_metrics.py.
    def in_range(j):
        return (dp_mod_all[j] >= dp_lo) & (dp_mod_all[j] <= dp_hi)

    N_obs = np.array([trapz(np.nan_to_num(obs[c].values), np.log10(dp_obs))
                      for c in bin_cols])
    N_mod = np.array([trapz(psd[in_range(j)], np.log10(dp_mod_all[j][in_range(j)]))
                      for j, psd in enumerate(psd_mod)])
    de_obs = np.array([effective_dp(dp_obs, obs[c].values) for c in bin_cols])
    de_mod = np.array([effective_dp(dp_mod_all[j], psd, dp_lo, dp_hi)
                       for j, psd in enumerate(psd_mod)])
    pk_obs = np.array([dp_obs[np.nanargmax(obs[c].values)] for c in bin_cols])
    pk_mod = np.array([dp_mod_all[j][in_range(j)][np.argmax(psd[in_range(j)])]
                       for j, psd in enumerate(psd_mod)])

    # ── printed comparison at the observed ages ──────────────────────────────
    print(f'\n{"N2O bin":>9} {"theta":>6} {"age":>6} | {"N obs":>7} {"N mod":>7} '
          f'{"mod/obs":>8} | {"Deff obs":>8} {"Deff mod":>8} | '
          f'{"pk obs":>7} {"pk mod":>7}')
    print('-' * 92)
    for c, th, a, no, deo, pko in zip(bin_cols, theta, ages_obs, N_obs, de_obs, pk_obs):
        j = int(np.argmin(np.abs(t_abs - a)))
        flag = ' *' if th < THETA_OVERWORLD else '  '
        print(f'{c:>9}{flag}{th:6.0f} {a:6.2f} | {no:7.1f} {N_mod[j]:7.1f} '
              f'{N_mod[j] / no:8.2f} | {deo:8.1f} {de_mod[j]:8.1f} | '
              f'{pko:7.1f} {pk_mod[j]:7.1f}')
    if args.ages == 'measured':
        print('  * = lowermost stratosphere (theta < 400 K), different air mass')

    # elapsed model time needed to reach each observed number concentration,
    # compared against the elapsed observed age since the start bin
    print('\nModel time needed to reach each observed number concentration:')
    speeds = []
    for c, th, a, no in zip(bin_cols, theta, ages_obs, N_obs):
        dt_obs = a - t0
        if dt_obs <= 0:
            continue
        if no > N_mod[0] or no < N_mod[-1]:
            print(f'{c:>9}  +{dt_obs:4.2f} yr  N={no:6.1f}  -> outside the model range')
            continue
        t_match = np.interp(-no, -N_mod, t_yr)   # N_mod is decreasing
        skip = th < THETA_OVERWORLD
        note = '  [LMS, excluded]' if skip else f'   ({dt_obs / t_match:.1f}x too fast)'
        if not skip:
            speeds.append(dt_obs / t_match)
        print(f'{c:>9}  +{dt_obs:4.2f} yr  N={no:6.1f}  -> model +{t_match:5.2f} yr{note}')
    if speeds:
        print(f'\nMedian coagulation speed factor: {np.median(speeds):.1f}x '
              f'faster than the observed aging sequence '
              f'({len(speeds)} overworld bins)')

    # ── fig 1: PSD overlay ───────────────────────────────────────────────────
    # Colour maps onto measured age, so the observed and modelled panels share a
    # scale. LMS bins are dashed: same colour axis, different air mass.
    a_min, a_max = np.nanmin(ages_obs), np.nanmax(ages_obs)
    span = max(a_max - a_min, 1e-9)
    colors_obs = [cm.viridis((a - a_min) / span) for a in ages_obs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for c, th, a, col in zip(bin_cols, theta, ages_obs, colors_obs):
        is_lms = th < THETA_OVERWORLD
        ax1.plot(dp_obs, obs[c].values, color=col, lw=2,
                 ls=':' if is_lms else '-',
                 label=f'{c} ppbv, {a:.2f} yr' + (' (LMS)' if is_lms else ''))
    ax1.set_title('SABRE observed (Alaska, median by N$_2$O bin)\n'
                  'dotted = lowermost stratosphere, $\\theta$ < 400 K')
    ax1.legend(fontsize=7, frameon=False)

    for j, ta in enumerate(t_abs):
        col = cm.viridis(np.clip((ta - a_min) / span, 0, 1))
        ax2.plot(dp_mod_all[j], psd_mod[j], color=col, lw=2,
                 label=f'age = {ta:.2f} yr')
    ax2.set_title(f'TOMAS-JAX {proc_label}\n(init: {args.start_bin} ppbv PSD, '
                  f'{float(d["temp"]):.1f} K, {float(d["pres"]) / 100:.0f} hPa)')
    ax2.legend(fontsize=7, frameon=False)

    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xlim(2, 2e4)
        ax.set_ylim(0, 300)
        ax.set_xlabel('Diameter (nm)')
        ax.grid(alpha=0.25, lw=0.5)
    ax1.set_ylabel('dN/dlog$_{10}$D$_p$ (cm$^{-3}$ STP)')
    fig.tight_layout()
    fig.savefig(f'{args.outdir}/psd_obs_vs_model{args.tag}.png', dpi=150)

    # ── fig 1b: paired overlay at matched ages ───────────────────────────────
    # only overworld bins at or after the model start age are comparable
    plotted = [(c, a) for c, th, a in zip(bin_cols, theta, ages_obs)
               if a >= t0 and th >= THETA_OVERWORLD]
    if not plotted:
        raise ValueError(
            f'No overworld bins at or after the model start age ({t0:.2f} yr). '
            f'Check --start-bin (currently {args.start_bin}).')
    # rescaled to the plotted bins only, so the colour axis starts at the model
    # start bin instead of carrying dead space from the excluded LMS bins
    plotted_ages = [a for _, a in plotted]
    p_min, p_max = min(plotted_ages), max(plotted_ages)
    norm_p = Normalize(p_min, max(p_max, p_min + 1e-9))
    pairs = [(c, a, cm.viridis(norm_p(a))) for c, a in plotted]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for c, a, col in pairs:
        j = int(np.argmin(np.abs(t_abs - a)))
        ax.plot(dp_obs, obs[c].values, color=col, lw=2)
        ax.plot(dp_mod_all[j], psd_mod[j], color=col, lw=1.6, ls='--')
    ax.plot([], [], color='0.3', lw=2, label='observed (N$_2$O bin)')
    ax.plot([], [], color='0.3', lw=1.6, ls='--',
            label=f'model at same age ({proc_label})')
    ax.set_xscale('log')
    ax.set_xlim(2, 2e4)
    # 200 fits both cases now that the source is mass-consistent: the old 600 was
    # headroom for the spurious nucleation growth wave a 50 molec/cm3/s source
    # produced, and nothing here exceeds ~150 cm^-3.
    ax.set_ylim(0, 200)
    ax.set_xlabel('Diameter (nm)')
    ax.set_ylabel('dN/dlog$_{10}$D$_p$ (cm$^{-3}$ STP)')
    ax.set_title('SABRE Alaska (>50$\\degree$N), paired by measured mean age '
                 f'($\\theta$ > {THETA_OVERWORLD:.0f} K bins only)\n'
                 f'model initialized from {args.start_bin} ppbv N$_2$O '
                 f'(age {t0:.2f} yr)')
    ax.legend(fontsize=9, frameon=False)
    # colour axis spans only the plotted bins (see norm_p above)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm_p)
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label('measured mean age (yr)\nticks: age (N$_2$O bin, ppbv)')
    cb.set_ticks(plotted_ages)
    cb.set_ticklabels([f'{a:.2f}  ({c})' for c, a, _ in pairs], fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f'{args.outdir}/psd_paired_by_age{args.tag}.png', dpi=150)

    # ── fig 1c: same pairing, styled for slides ──────────────────────────────
    # Separate file so the analysis figure above keeps its full annotation. Large
    # type, minimal text: the N2O bin edges, the theta cut, the model start bin
    # and the process list live in the caption/docs, not on the axes.
    fig, ax = plt.subplots(figsize=(11, 7))
    for c, a, col in pairs:
        j = int(np.argmin(np.abs(t_abs - a)))
        ax.plot(dp_obs, obs[c].values, color=col, lw=3.2)
        ax.plot(dp_mod_all[j], psd_mod[j], color=col, lw=2.6, ls='--')
    ax.plot([], [], color='0.3', lw=3.2, label='Observed (SABRE)')
    ax.plot([], [], color='0.3', lw=2.6, ls='--', label='Model')
    ax.set_xscale('log')
    # empty decades above ~1 um only shrink the part of the axis that carries
    # the signal; the coarse tail is flat out to the 17.5 um grid top
    ax.set_xlim(2, 2e3)
    ax.set_ylim(0, 200)
    # dry on both sides: the AMP PSD is dry, and model_dp() returns the dry axis
    # (TOMAS bins are mass bins, so the grid diameter carries equilibrium water)
    ax.set_xlabel('Dry Diameter (nm)', fontsize=20)
    # STP, not ambient: both curves are cm^-3 at 273.15 K / 1013 hPa (the AMP file
    # convention). Keep the unit on the axis - at 142 hPa the two differ by ~5.7x.
    ax.set_ylabel('dN/dlog$_{10}$D$_p$ (cm$^{-3}$ STP)', fontsize=20)
    ax.set_title('Aerosol Size Distribution vs Air Age', fontsize=23, pad=14)
    ax.tick_params(labelsize=17, width=1.2, length=6)
    ax.legend(fontsize=19, frameon=False, loc='upper left')
    sm = cm.ScalarMappable(cmap='viridis', norm=norm_p)
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label('Air Age (years)', fontsize=20)
    cb.set_ticks(plotted_ages)
    cb.set_ticklabels([f'{a:.1f}' for _, a, _ in pairs], fontsize=17)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f'{args.outdir}/psd_paired_by_age{args.tag}_slide.png', dpi=150)

    # ── fig 2: bulk metrics vs age ───────────────────────────────────────────
    # Two panels: number and effective size. The mode of dN/dlog10Dp is still in
    # the printed table but not plotted - effective diameter is the size metric.
    # Only overworld bins are drawn: the theta < 400 K bins are a different air
    # mass, so plotting them alongside invites reading them as part of the sequence.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    obs_kw = dict(color='#B3472F', marker='o', ms=6, lw=2, label='SABRE observed')
    mod_kw = dict(color='#2F6FB3', marker='s', ms=5, lw=2,
                  label=f'TOMAS-JAX, {proc_label}')

    ow = theta >= THETA_OVERWORLD if args.ages == 'measured' \
        else np.ones(len(bin_cols), bool)

    for ax, yo, ym, ylab, title in [
        (axes[0], N_obs, N_mod, 'N (cm$^{-3}$ STP), 3 nm - 2.4 $\\mu$m',
         'Integrated number'),
        (axes[1], de_obs, de_mod, 'Effective diameter (nm)',
         'Effective size (M$_3$/M$_2$)'),
    ]:
        ax.plot(ages_obs[ow], yo[ow], **obs_kw)
        ax.plot(t_abs, ym, **mod_kw)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].set_yscale('log')

    xlab = ('Mean age of air (years, SABRE product)' if args.ages == 'measured'
            else 'Age since 315 ppbv (years, assumed linear)')
    for ax in axes:
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f'{args.outdir}/bulk_metrics_vs_age{args.tag}.png', dpi=150)

    print(f'\nWrote figures to {args.outdir}/')


if __name__ == '__main__':
    main()
