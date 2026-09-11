# SABRE aerosol aging validation

TOMAS-JAX box model vs. SABRE aircraft observations (Alaska, >50°N), compared on the
measured mean-age-of-air axis. Two cases: coagulation only, and nucleation +
coagulation + condensation.

## Reproduce

Needs `numpy`, `scipy`, `matplotlib`, `pandas`, `jax`, `jaxlib`, `diffrax`.

```bash
# 1. simulations (~13 s and ~48 s wall clock)
python3 run_sabre_coag.py --psd-csv alaska_290-300_psd.csv \
    --years 1.8 --n-snapshots 6 --tp-track --start-bin 290-300 \
    --processes coag --out sabre_coag_from290.npz

python3 run_sabre_coag.py --psd-csv alaska_290-300_psd.csv \
    --years 1.8 --n-snapshots 6 --tp-track --start-bin 290-300 \
    --processes nucl_coag_cond --out sabre_full_from290.npz

# 2. figures -> figures_sabre/
python3 compare_sabre_coag.py --npz sabre_coag_from290.npz --start-bin 290-300 --tag _coag
python3 compare_sabre_coag.py --npz sabre_full_from290.npz --start-bin 290-300 --tag _full
```

`--years 1.8` is set by the observations, not chosen: the model is initialized from the
290-300 ppbv N₂O bin, whose measured mean age is 1.87 yr, and the oldest bin (220-230)
sits at 3.64 yr — so 1.77 yr of elapsed model time spans the observed aging sequence.
`--tp-track` lets temperature and pressure follow the measured bin-by-bin values along
that track instead of holding the start-bin conditions.

## Figures

Three per case (`_coag`, `_full`):

| File | Content |
|---|---|
| `psd_obs_vs_model_*.png` | observed and modelled dN/dlog₁₀Dp side by side, coloured by age |
| `psd_paired_by_age_*.png` | obs/model overlaid at matched age (θ > 400 K bins only) |
| `psd_paired_by_age_*_slide.png` | same pairing, styled for slides |
| `bulk_metrics_vs_age_*.png` | integrated number and effective diameter vs age |

## Result

Both cases coagulate ~3.8-3.9× too fast relative to the observed sequence: the model
sheds number in ~0.45 yr that the observations take ~1.8 yr to shed. Adding nucleation
and condensation barely moves it, so the discrepancy is in the coagulation rate or the box 
model limitations.

The 310-320 and 300-310 ppbv bins sit below θ = 400 K — lowermost stratosphere,
ventilated across isentropes from the troposphere — so they are a different air mass and
are excluded from the fit.
Conservation approach on 380 K θ threshold on stratospheric overworld. 
See https://acp.copernicus.org/articles/23/14375/2023/


## Files

`run_sabre_coag.py` runs the box model; `compare_sabre_coag.py` builds the figures;
`sabre_metrics.py` holds the shared size metrics (effective diameter M₃/M₂, dry-diameter
scaling); `run_box_model.py` is imported only for its PSD-loading helpers. `tomas_jax/`
is the subset of the model package these runs touch. `alaska_*.csv` are the
observations: the initial PSD, the N₂O-binned median PSDs, and the per-bin environment
and mean-age tables.

Diameters are dry and concentrations are at STP (273.15 K, 1013 hPa) on both sides,
matching the AMP data convention.
