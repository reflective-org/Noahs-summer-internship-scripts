"""Gravitational settling — size-dependent first-order loss from a box of finite depth.

Physics:
    Stokes settling with the Cunningham slip correction (Seinfeld & Pandis 8.42):

        v_s(Dp) = rho_p * Dp^2 * g * Cc / (18 * mu)
        Cc      = 1 + Kn * (1.257 + 0.4 * exp(-1.1 / Kn)),   Kn = 2 * lambda / Dp

    For a box of depth H stacked under a layer holding C_above, the sedimentation
    flux divergence reduces to a per-bin relaxation:

        dC/dt = (v_s / H) * (C_above - C)

    which integrates exactly over dt to

        C(t+dt) = C_above + (C(t) - C_above) * exp(-v_s * dt / H)

    Number and mass in a bin decay with the same rate, so the mass/number ratio
    is preserved and no MNFIX is required afterwards.

IMPORTANT — the ``C_above = 0`` default is a *pure loss*, which is the right
limit only when the box sits at or above the peak of the aerosol layer. Below
the peak (e.g. the 15-16 km high-latitude lower stratosphere, which sits under
the Junge layer maximum) particles settling in from above outnumber those
settling out, the true flux divergence changes sign, and a pure-loss term is
wrong-signed. Pass ``Nk_above``/``Mk_above`` to represent the overlying layer
when that matters. See ``docs/sedimentation.md``.

All functions are JIT-compilable.
"""
import jax.numpy as jnp

from ..core.config import PI, GRAV, MOLAR_MASS_AIR, R_GAS, _DENS_INIT
from .gas_properties import calc_air_viscosity


def calc_air_mean_free_path(temp, pres):
    """Mean free path of air [m] (Seinfeld & Pandis 9.6).

    Args:
        temp: Temperature [K]
        pres: Pressure [Pa]

    Returns:
        Mean free path [m]
    """
    mu = calc_air_viscosity(temp)
    return 2.0 * mu / (pres * jnp.sqrt(8.0 * MOLAR_MASS_AIR / (PI * R_GAS * temp)))


def calc_settling_velocity(dp, temp, pres, density=_DENS_INIT):
    """Stokes settling velocity with Cunningham slip correction [m/s].

    Args:
        dp: Particle diameter [m] (scalar or array)
        temp: Temperature [K]
        pres: Pressure [Pa]
        density: Particle density [kg/m^3]

    Returns:
        Settling velocity [m/s], same shape as dp
    """
    mu = calc_air_viscosity(temp)
    mfp = calc_air_mean_free_path(temp, pres)
    kn = 2.0 * mfp / dp
    cc = 1.0 + kn * (1.257 + 0.4 * jnp.exp(-1.1 / kn))
    return density * dp**2 * GRAV * cc / (18.0 * mu)


def bin_diameters(Nk, Mk, xk, density=_DENS_INIT):
    """Volume-equivalent diameter of the average particle in each bin [m].

    Uses the actual mass/number ratio, clipped to the bin's mass boundaries so
    that empty or drifted bins still yield a diameter inside their own bin.

    Args:
        Nk: Number concentration [#/grid cell], shape (nbins,)
        Mk: Mass concentration [kg/grid cell], shape (nbins, icomp)
        xk: Mass bin boundaries [kg], shape (nbins+1,)
        density: Particle density [kg/m^3]

    Returns:
        Diameter [m], shape (nbins,)
    """
    avg_mass = jnp.sum(Mk, axis=1) / jnp.maximum(Nk, 1.0e-30)
    avg_mass = jnp.clip(avg_mass, xk[:-1], xk[1:])
    return jnp.cbrt(6.0 * avg_mass / (PI * density))


def sedimentation_step(Nk, Mk, xk, temp, pres, dt, layer_depth,
                       Nk_above, Mk_above, density=_DENS_INIT):
    """Apply one timestep of gravitational settling.

    Both ``*_above`` arrays are required (no None defaults) for JIT safety.
    Pass ``jnp.zeros_like(Nk)`` / ``jnp.zeros_like(Mk)`` for pure loss — but
    read the module docstring first, that limit is not always the right one.

    Args:
        Nk: Number concentration [#/grid cell], shape (nbins,)
        Mk: Mass concentration [kg/grid cell], shape (nbins, icomp)
        xk: Mass bin boundaries [kg], shape (nbins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        dt: Timestep [s]
        layer_depth: Box depth H [m] over which the settling flux is spread
        Nk_above: Number concentration in the overlying layer [#/grid cell]
        Mk_above: Mass concentration in the overlying layer [kg/grid cell]
        density: Particle density [kg/m^3]

    Returns:
        (Nk_new, Mk_new, Nk_lost, Mk_lost) where the ``*_lost`` arrays are the
        net amounts removed this step (negative if settling from above is a net
        source), for budget closure.
    """
    dp = bin_diameters(Nk, Mk, xk, density)
    v_s = calc_settling_velocity(dp, temp, pres, density)

    decay = jnp.exp(-v_s * dt / layer_depth)

    Nk_new = Nk_above + (Nk - Nk_above) * decay
    Mk_new = Mk_above + (Mk - Mk_above) * decay[:, None]

    return Nk_new, Mk_new, Nk - Nk_new, Mk - Mk_new
