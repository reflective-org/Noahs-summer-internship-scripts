# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research scripts from a summer internship on stratospheric aerosol and dynamics. Not a
package — no `pyproject.toml`, no test suite, no stable API. Scripts are run directly
from their own directory.

- `sabre_aerosol_validation/` — TOMAS-JAX aerosol microphysics box model validated
  against SABRE aircraft observations on the measured age-of-air axis. `run_sabre_coag.py`
  runs the model, `compare_sabre_coag.py` builds figures, `sabre_metrics.py` holds shared
  size metrics, `tomas_jax/` is the model package subset these runs touch. Its README has
  the exact reproduce commands.
- `aide_waccm_validation/` — AIDE vs. WACCM validation plots (notebook).
- `dynamics/` — age of air, diffusion coefficient, residual velocity (notebooks).

## Conventions that matter

- **JAX runs in float64.** `tomas_jax.core.config` must be imported before anything
  touches JAX, or the model silently runs in float32.
- **Diameters are dry, concentrations are at STP** (273.15 K, 1013 hPa) on both the model
  and observation side, matching the AMP data convention. Coagulation rates go as ambient
  number density squared, so observed values are converted to ambient before integration
  and back to STP for comparison — skipping that conversion is a ~30x error.
- **Size is reported as effective diameter** (M3/M2, Hansen & Travis 1974), not number- or
  mass-weighted means. See the module docstring in `sabre_metrics.py` for why.
- Notebooks read data from absolute paths local to the machine they were run on; they will
  not execute elsewhere without repointing those paths.
