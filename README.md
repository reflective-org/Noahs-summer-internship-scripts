# Noah's summer internship scripts

> [!WARNING]
> Research scripts, not a package. No install, no tests, no stable API.

Stratospheric aerosol and dynamics analysis. Needs `numpy`, `scipy`, `pandas`,
`matplotlib`, `jax`, `jaxlib`, `diffrax`.

| Directory | Contents |
|---|---|
| [`sabre_aerosol_validation/`](./sabre_aerosol_validation) | TOMAS-JAX box model vs. SABRE aircraft observations, compared on the measured age-of-air axis. See its [README](./sabre_aerosol_validation/README.md) to reproduce. |
| `aide_waccm_validation/` | AIDE vs. WACCM validation plots. |
| `dynamics/` | Age of air, diffusion coefficient, and residual velocity notebooks. |

Notebooks read data from paths local to the machine they were run on.

## License

Apache 2.0 — see [LICENSE](./LICENSE).
