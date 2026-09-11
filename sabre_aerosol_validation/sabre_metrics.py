"""Shared size metrics for the SABRE comparison scripts.

Two conventions this module exists to enforce, both about comparing like with
like against the AMP data:

1. **Size is reported as effective diameter**, the third/second moment ratio
   (Hansen & Travis 1974) — the area-weighted size the radiation sees. Number- and
   mass-weighted means are not used: the first is dominated by whichever mode
   happens to carry the most particles (so a nucleation burst swamps it) and the
   second by the coarse tail, and neither is the quantity anyone downstream cares
   about.

2. **Diameters are dry.** The AMP PSD is dry (<40% RH), so the model must be read
   dry too. Condensation runs carry equilibrium water; `dry_scale` converts a
   total-mass diameter axis to a dry one.
"""
import numpy as np

trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def effective_dp(dp, dndlog10, dp_lo=None, dp_hi=None):
    """Effective diameter: M3/M2 of the distribution [same units as dp].

    Hansen & Travis (1974). Area-weighted, so it is far more sensitive to the
    coarse tail than a number-weighted mean and far less sensitive to an
    ultrafine mode.

    Args:
        dp: Bin diameters.
        dndlog10: dN/dlog10Dp on those bins.
        dp_lo, dp_hi: Optional integration limits, in the units of `dp`.
    """
    dp = np.asarray(dp, dtype=float)
    w = np.clip(np.nan_to_num(np.asarray(dndlog10, dtype=float)), 0.0, None)
    m = np.ones_like(dp, dtype=bool)
    if dp_lo is not None:
        m &= dp >= dp_lo
    if dp_hi is not None:
        m &= dp <= dp_hi
    d, w = dp[m], w[m]
    if d.size < 2:
        return np.nan
    lg = np.log10(d)
    m2 = trapz(w * d ** 2, lg)
    if m2 <= 0:
        return np.nan
    return trapz(w * d ** 3, lg) / m2


def dry_scale(Mk, i_h2o):
    """Factor converting a total-mass diameter axis to a dry-mass one.

    TOMAS bins are mass bins, so a particle carrying equilibrium water sits in a
    higher bin than its dry mass alone would place it, and the diameter read off
    the grid is a wet diameter. The water fraction is proportional to sulfate
    (calc_equilibrium_water scales with it, verified constant to 1.5% across
    populated bins), so a single factor (M_dry/M_total)^(1/3) converts the whole
    axis exactly - and because it is uniform, log-spacing and therefore
    dN/dlog10Dp are unchanged.

    Args:
        Mk: Mass per bin [kg/cell], shape (nbins, icomp).
        i_h2o: Water species index; species before it are the dry ones.

    Returns:
        Scale factor <= 1 (exactly 1 for a dry run).
    """
    Mk = np.asarray(Mk, dtype=float)
    tot = Mk.sum()
    if tot <= 0:
        return 1.0
    return float((Mk[:, :i_h2o].sum() / tot) ** (1.0 / 3.0))
