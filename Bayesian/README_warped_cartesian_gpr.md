# Warped Cartesian ET plus residual GPR

`GPR_multi_cartesian_warped.py` predicts TCAM temperature fields using an
Eagar-Tsai (ET) prior, a learned Cartesian warp, and a Gaussian-process model
of the remaining temperature discrepancy. The current best tested variant
adds the Cartesian distance `r_3d` to the residual-GP inputs.

This implementation is not cylindrical. It does not use `theta`, Peclet
number, dimensionless spatial coordinates, or a Matérn kernel.

## Model

The final temperature prediction is

```text
delta_T_training = T_TCAM - T_warped_ET
T_pred_unconstrained = T_warped_ET + delta_T_pred
T_pred = clip(T_pred_unconstrained, T_ambient, T_evaporation)
```

The warp maps a requested Cartesian position back into the ET field:

```text
x_centered = x - x_shift
x_query = x_centered / rear_scale    when x_centered < 0
x_query = x_centered / front_scale   when x_centered >= 0
y_query = y / width_scale
z_query = z / depth_scale

T_warped_ET = T_ambient
             + amplitude * (T_ET(x_query, y_query, z_query) - T_ambient)
```

The six fitted warp parameters are therefore:

- amplitude;
- rear x scale;
- front x scale;
- y/width scale;
- z/depth scale; and
- x shift, represented internally relative to the ET beam radius.

A regularized quadratic response surface learns each warp parameter as a
function of power and velocity. This response surface supplies the warp for
the held-out process condition without using its TCAM field.

## Residual-GP inputs

The recommended `r_3d` model uses seven inputs:

| Input | Meaning |
|---|---|
| `P` | Laser power in W |
| `V` | Scan velocity in m/s |
| `x` | Cartesian coordinate along the scan direction in um |
| `y` | Cartesian transverse coordinate in um |
| `z` | Cartesian depth coordinate in um |
| `r_3d` | `sqrt(x^2 + y^2 + z^2)` in um |
| `T_warped_ET` | Warped ET prior temperature in K |

All seven inputs are normalized to `[0, 1]` before GP fitting. Fixed physical
ranges are used for `P` and `V`; the training-data ranges are used for the
spatial, radial, and temperature features.

The GP uses an anisotropic RBF kernel with one learned length scale per input,
plus a white-noise kernel. The production `r_3d` run learned:

```text
0.856^2 * RBF(length_scale=[0.451, 0.865, 0.05, 0.0915,
                            0.114, 0.05, 0.0864])
+ WhiteKernel(noise_level=0.0878)
```

The length scales follow the input order in the table. The `x` and `r_3d`
length scales reached the configured lower bound of `0.05`, so the fitted GP
uses both features strongly. This is worth monitoring in future validation
runs because a boundary-hitting length scale can also indicate that a smaller
scale or more training support is needed.

## Training and held-out prediction

The default experiment trains on 15 complete process cases and holds out:

```text
P = 250 W
V = 0.5 m/s
```

The workflow is deliberately divided into training, blind prediction, and
evaluation stages:

1. For each training case, the structured ET field is trilinearly evaluated
   at the TCAM mesh coordinates.
2. A Cartesian ET warp is fitted to that case using up to 900 warp samples.
3. The residual GP receives 500 sampled points from each case, for 7,500
   training points total. Sampling retains evaporation-plateau points and
   balances the remaining temperature regimes.
4. The six held-out warp parameters are predicted from only `P` and `V`.
5. The GP predicts the complete held-out field on a coarse lattice derived
   from the held-out ET grid.
6. The blind prediction VTI is written and frozen.
7. Only then is the held-out TCAM field opened for metrics, parity plots, and
   the comparison VTI.

The prediction lattice is independent of the held-out TCAM mesh:

```text
dimensions: 91 x 26 x 33
point count: 78,078
spacing: 8.0 x 8.0 x 7.8125 um
```

Thus, TCAM coordinates are used to align the 15 training cases, but they do
not determine the held-out prediction coordinates.

## Run the recommended r_3d model

From the repository root:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_3d
```

The output directory is:

```text
beamer/figures/bayesian/warped_cartesian_r3d_gpr/
  heldout_250W_0.5ms_15trainingcases/
```

The recorded production run took 1749.6 seconds, or approximately 29 minutes
10 seconds.

## Important output files

| File | Purpose |
|---|---|
| `heldout_prediction_on_coarse_et_grid.vti` | TCAM-blind prediction on the ET-derived lattice |
| `heldout_prediction_vs_tcam_on_coarse_et_grid.vti` | Post-prediction ParaView comparison containing interpolated TCAM and errors |
| `warped_cartesian_residual_gpr.joblib` | Fitted GP, scaler, warp response surface, feature list, and case split |
| `metrics.json` | Inputs, kernel, metrics, timings, grid metadata, and leakage statement |
| `heldout_parity.png` | Raw ET, warped ET, and final GPR parity plot with MAE and RMSE |
| `heldout_residual_parity.png` | Residual-model parity plot |
| `heldout_predictions_at_tcam_points.csv` | Evaluation values at the original held-out TCAM points |
| `training_case_warps.csv` | Fitted warp parameters and diagnostics for all 15 training cases |
| `warp_leave_one_case_out.csv` | Leave-one-training-case-out warp response-surface diagnostics |
| `training_sample.csv` | The 7,500 residual-GP training rows |
| `case_split.json` | Explicit training and held-out case records |

The blind prediction VTI contains:

| Array | Meaning |
|---|---|
| `T_ET` | Unwarped ET prior |
| `T_warped_ET` | Warped ET prior |
| `delta_T_pred` | Mean GP discrepancy prediction |
| `delta_T_std` | GP predictive standard deviation |
| `T_pred_unconstrained` | Warped prior plus mean discrepancy before physical clipping |
| `T_pred` | Final clipped mean prediction |
| `T_pred_lower_95` | Clipped `T_pred_unconstrained - 1.96 * delta_T_std` |
| `T_pred_upper_95` | Clipped `T_pred_unconstrained + 1.96 * delta_T_std` |

`T_pred` is the mean prediction. It is not a combination of the lower and
upper 95% fields. The comparison VTI additionally contains `T_TCAM`,
`T_pred_minus_TCAM`, `abs_error`, and `TCAM_valid_mask`.

Quantitative metrics are evaluated at the original held-out TCAM points, not
on TCAM values interpolated onto the coarse VTI.

## Held-out results

For the 250 W, 0.5 m/s held-out case:

| Model | Full RMSE (K) | Full MAE (K) | Hot-region RMSE (K) | Above-liquidus RMSE (K) |
|---|---:|---:|---:|---:|
| Raw ET | 1237.6 | 703.2 | 2414.0 | 2901.9 |
| Warped ET | 548.4 | 407.2 | 860.5 | 1073.0 |
| Warped Cartesian GP, no radius | 483.2 | 267.0 | 911.2 | 1080.9 |
| Warped Cartesian GP with `r_yz` | 500.4 | 270.8 | 954.9 | 1131.3 |
| Warped Cartesian GP with `r_3d` | **409.2** | **236.3** | **742.3** | **893.8** |

Relative to the six-input Cartesian GP, `r_3d` reduced full-field RMSE by
15.3%, hot-region RMSE by 18.5%, and above-liquidus RMSE by 17.3%.
The hot-region mask is defined from TCAM as temperatures at or above ambient
plus 75% of the ambient-to-liquidus temperature rise. The above-liquidus mask
is also defined from TCAM.

On the coarse ET lattice, hot-region depth-direction temperature reversals
decreased from 21.4% of adjacent hot pairs to 15.3%. The extracted liquidus
surface remained one connected component. The predicted melt-pool depth
improved from 100.8 um to 104.7 um compared with 112.8 um in TCAM. Predicted
length changed from 206.0 um to 201.1 um compared with 229.0 um in TCAM, so
`r_3d` improved the temperature errors and depth behavior but not every
geometric measure.

## Other controlled variants

The original six-input Cartesian model remains available:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py
```

Its inputs are `P, V, x, y, z, T_warped_ET`, and its outputs are written under
`warped_cartesian_gpr/`.

The cross-sectional-radius ablation is:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_yz
```

Here, `r_yz = sqrt(y^2 + z^2)`. It writes under
`warped_cartesian_ryz_gpr/`. It did not improve the held-out errors.

## Fast pipeline check

Use a separate temporary output directory so a smoke test cannot overwrite a
production model:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_3d \
  --points-per-case 20 \
  --warp-points-per-case 20 \
  --warp-max-iterations 1 \
  --n-restarts-optimizer 0 \
  --prediction-grid-stride 20 \
  --output-dir /tmp/ettokgt-warped-cartesian-r3d-smoke
```

## Re-export a saved model without TCAM

An existing fitted bundle can generate another ET-grid prediction without
retraining or opening TCAM:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --load-bundle beamer/figures/bayesian/warped_cartesian_r3d_gpr/heldout_250W_0.5ms_15trainingcases/warped_cartesian_residual_gpr.joblib \
  --prediction-grid-stride 4 \
  --prediction-output /tmp/r3d_prediction.vti
```

The saved bundle records its feature list, so the re-export automatically
reconstructs `r_3d` before prediction.

## Boundary handling and limitations

- Roundoff-sized warp excursions at ET boundaries are clipped back to the
  stored ET endpoint. This preserves nominal planes such as `z = 0`.
- Queries genuinely outside the finite ET support remain at ambient
  temperature; the code does not invent ET values outside its domain.
- `T_pred` is clipped to the ambient/evaporation interval, but monotonicity
  with depth is not explicitly enforced.
- `r_3d` is an engineered distance centered at the coordinate origin. It is
  useful for this split, but should be revalidated for other held-out process
  conditions and coordinate origins.
- The reported held-out TCAM metrics are evaluation results, not inputs to the
  blind prediction.
