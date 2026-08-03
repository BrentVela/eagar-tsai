# Warped Cartesian ET plus residual GPR

`GPR_multi_cartesian_warped.py` predicts TCAM temperature fields using an
Eagar-Tsai (ET) prior, a learned Cartesian warp, and a Gaussian-process model
of the remaining temperature discrepancy. The current best tested variant
adds the Cartesian distance `r_3d` to the residual-GP inputs and uses a
Matérn-5/2 covariance.

This implementation is not cylindrical. It does not use `theta`, Peclet
number, or dimensionless spatial coordinates.

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

The recommended GP uses an anisotropic Matérn-5/2 kernel with one learned
length scale per input, plus a white-noise kernel. The controlled production
run learned:

```text
1.39^2 * Matern(length_scale=[1.00, 1.61, 0.05, 0.172,
                              0.867, 0.0526, 0.188], nu=2.5)
+ WhiteKernel(noise_level=0.0476)
```

The length scales follow the input order in the table. The `x` length scale
reached the configured lower bound of `0.05`, while `r_3d` settled just above
it at `0.0526`. This is worth monitoring in future validation runs because a
boundary-hitting length scale can indicate that a smaller scale or more
training support is needed.

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

## Run the recommended r_3d + Matérn-5/2 model

From the repository root:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_3d \
  --kernel matern52
```

The output directory is:

```text
beamer/figures/bayesian/warped_cartesian_r3d_matern52_gpr/
  heldout_250W_0.5ms_15trainingcases/
```

The recorded production run used `n_restarts_optimizer=1` and took 2400.7
seconds, or approximately 40 minutes.

## Run the controlled Matérn-3/2 experiment

To change only the Matérn smoothness from `nu=2.5` to `nu=1.5`, while keeping
the recommended `r_3d` feature and using three optimizer restarts:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_3d \
  --kernel matern32 \
  --n-restarts-optimizer 3
```

The separate output directory is
`beamer/figures/bayesian/warped_cartesian_r3d_matern32_gpr/`.
In scikit-learn, three restarts means one initial optimizer run followed by
three restarted runs, for four sequential hyperparameter optimizations total.

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
| Warped Cartesian GP with `r_3d`, RBF | 409.2 | 236.3 | 742.3 | 893.8 |
| Warped Cartesian GP with `r_3d`, Matérn-5/2 | **319.1** | **189.7** | **575.0** | **699.3** |

Relative to the otherwise identical `r_3d` RBF GP, Matérn-5/2 reduced
full-field RMSE by 22.0%, full-field MAE by 19.7%, hot-region RMSE by 22.5%,
and above-liquidus RMSE by 21.8%. Its held-out R-squared was 0.9441, and its
nominal 95% residual interval covered 95.79% of the evaluation points.

The hot-region mask is defined from TCAM as temperatures at or above ambient
plus 75% of the ambient-to-liquidus temperature rise. The above-liquidus mask
is also defined from TCAM.

On the coarse ET lattice, hot-region depth-direction temperature reversals
decreased from 15.29% for `r_3d` RBF to 0.47% for `r_3d` Matérn-5/2. The
extracted liquidus surface remained one connected component. The predicted
melt-pool depth improved from 104.7 um to 106.9 um compared with 112.8 um in
TCAM. Predicted length changed from 201.1 um to 194.0 um compared with 229.0
um in TCAM, so Matérn-5/2 improved temperature accuracy, uncertainty
calibration, depth, and monotonic behavior but not every geometric measure.

The two `r_3d` runs used byte-identical `training_sample.csv` and
`training_case_warps.csv` files. Thus, the comparison changes the residual
kernel only. Matérn-5/2 cost more: 2400.7 seconds versus 1749.6 seconds for
RBF, an increase of about 37%.

## Rear-tip sampling experiment

Rear-focused sampling was tested because the recommended model underestimates
the liquidus length almost entirely at the trailing tip:

```text
TCAM:               x = -190.3 to +38.7 um
recommended model:  x = -157.4 to +36.6 um
```

The implemented sampler can reserve part of each case's fixed 500-point
budget for training-only rear-tip observations. It first finds the minimum-x
TCAM-liquidus point in that training case, then samples eligible observations
within 25 um ahead of that point. Eligible observations are within 300 K of
liquidus or are locations where TCAM is molten while warped ET is below
liquidus. Points closer to liquidus are preferred, with extra priority for
warped-prior false negatives. Cases with no molten TCAM points return the
unused quota to the original sampler.

The focused production command was:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_3d \
  --kernel matern52 \
  --rear-liquidus-points-per-case 30
```

It kept 7,500 total GP points and selected 358 dedicated tip points across the
eligible training cases. The training warp file was byte-identical to the
recommended Matérn run. Results were:

| Sampling | Full RMSE (K) | Rear tip (um) | Front tip (um) | Length (um) | Depth reversals |
|---|---:|---:|---:|---:|---:|
| Original Matérn sampling | **319.1** | **-157.4** | +36.6 | **194.0** | 0.47% |
| Broad rear-liquidus, up to 75/case | 427.7 | -152.4 | +40.8 | 193.2 | **0.02%** |
| Focused 25-um rear tip, up to 30/case | 324.0 | -149.5 | +40.8 | 190.3 | 0.33% |
| TCAM | - | -190.3 | +38.7 | 229.0 | - |

Both rear-sampling variants shortened rather than lengthened the held-out
pool. The focused points had the intended positive discrepancy—their median
`T_TCAM - T_warped_ET` was about +506 K—but reallocating observations changed
the single global GP kernel and its process interpolation rather than directly
constraining held-out geometry. Therefore rear-tip sampling is retained as a
reproducible negative result and is not enabled by default. The original
`r_3d` Matérn-5/2 model remains recommended.

Focused outputs are under
`warped_cartesian_r3d_matern52_rear_tip30_gpr/`. The earlier broad experiment
is under `warped_cartesian_r3d_matern52_rear_liquidus_gpr/`.

## Other controlled variants

The previous `r_3d` RBF model remains available by omitting `--kernel`:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_3d
```

It writes under `warped_cartesian_r3d_gpr/`. The command-line default remains
RBF for backward compatibility.

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

## Optimizer restarts

`--n-restarts-optimizer` controls repeated searches for kernel
hyperparameters; it does not resample the training data or average several
GPs. Scikit-learn first optimizes from the declared kernel values, then runs
the requested number of additional searches from randomized initial values.
It retains the solution with the highest log marginal likelihood.

```text
--n-restarts-optimizer 0  = 1 total optimization attempt
--n-restarts-optimizer 1  = 2 total optimization attempts (production default)
--n-restarts-optimizer 3  = 4 total optimization attempts
```

Use zero restarts for smoke tests, one for controlled model comparisons, and
consider three for a final selected model when the additional runtime is
acceptable. Runtime can grow approximately with the number of attempts. More
restarts can avoid a poor local optimum, but they do not resolve a length
scale consistently reaching its allowed lower bound.

## Fast pipeline check

Use a separate temporary output directory so a smoke test cannot overwrite a
production model:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --radial-feature r_3d \
  --kernel matern52 \
  --points-per-case 20 \
  --warp-points-per-case 20 \
  --warp-max-iterations 1 \
  --n-restarts-optimizer 0 \
  --prediction-grid-stride 20 \
  --output-dir /tmp/ettokgt-warped-cartesian-r3d-matern52-smoke
```

## Re-export a saved model without TCAM

An existing fitted bundle can generate another ET-grid prediction without
retraining or opening TCAM:

```bash
.venv/bin/python -u Bayesian/GPR_multi_cartesian_warped.py \
  --load-bundle beamer/figures/bayesian/warped_cartesian_r3d_matern52_gpr/heldout_250W_0.5ms_15trainingcases/warped_cartesian_residual_gpr.joblib \
  --prediction-grid-stride 4 \
  --prediction-output /tmp/r3d_matern52_prediction.vti
```

The saved bundle records its feature list and kernel family, so the re-export
automatically reconstructs `r_3d` and uses the fitted Matérn model.

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
