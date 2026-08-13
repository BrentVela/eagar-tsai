# SFF Symposium 2026 presentation

This folder contains the Beamer source and supporting assets for Kyle Swartz's presentation, “Rapid Microstructure Prediction in Laser Powder Bed Fusion Using Bayesian Updating of Eagar–Tsai Model,” presented August 3, 2026.

The deck now covers the full ET → TCAM → warped-prior Gaussian-process update → `G`–`R` → CET microstructure workflow. Most scientific figures can be regenerated from scripts in the repository; ParaView screenshots and the TCAM GUI image remain manual assets.

## Build the deck

Run from the `beamer/` directory because the source uses paths relative to it:

```bash
cd beamer
../.tools/bin/tectonic sff_beamer.tex
```

This is the build used for the checked-in `sff_beamer.pdf`. The source uses the repo-local Carlito fonts in `.tools/fonts/` when compiled with Tectonic/XeTeX. A sufficiently complete local TeX installation can also use:

```bash
latexmk -xelatex sff_beamer.tex
```

The linked warp animation is not embedded in the PDF. Keep its MP4 at the documented relative path and use a PDF viewer that supports `run:` links; the poster image is displayed when the link is unavailable.

Main files:

- `sff_beamer.tex`: presentation source
- `sff_theme.sty`: custom SFF theme
- `sff_beamer.pdf`: compiled presentation
- `figures/`: manual and generated assets plus ET/TCAM source data
- `mumatscale/`: MuMatScale inputs and rendering instructions for solidification-mode illustrations

## Install the figure-generation environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Thermo-Calc's separately installed `tc_python` package and a valid license are required for CET maps and microstructure classification. FFmpeg is required only to regenerate the MP4 warp animation. See the [root README](../README.md) for the broader project layout and data conventions.

## ET figures

`workflow_ET.py` regenerates the ET figures and intermediate data for Alloy 0 (Hf<sub>2</sub>Mo<sub>2</sub>Ta<sub>48</sub>W<sub>48</sub>) at 250 W and 0.5 m/s:

```bash
# From the repository root
python workflow_ET.py
```

The default output directory is `beamer/figures/250_0.5/`. Available steps run in pipeline order:

| Step | Principal output | Required existing input when run alone |
| --- | --- | --- |
| `temperature-field` | `temperature_field_*.png`, `ET_meta_*.csv` | none |
| `temperature-volume` | legacy `ET_*.csv` | none |
| `temperature-3d` | `ET_3D_temperature_*.png`, `.vti` | metadata CSV |
| `liquidus` | `liquidus_GR_*.csv` | metadata and temperature VTI |
| `gr-map` | `GR_map_overlayET_*.png` | liquidus CSV |
| `gr-projection` | `projected_GR_*.png` | liquidus CSV |
| `microstructure` | `projected_microstructure_*.png` | liquidus CSV |

Examples:

```bash
python workflow_ET.py --steps temperature-field temperature-3d liquidus
python workflow_ET.py --steps gr-map gr-projection microstructure
python workflow_ET.py --help
```

The `ET_3D_temperature_*.vti` output is a fine-resolution field cropped around the melt pool. It is used to extract the liquidus boundary and calculate `G` and `R` for the downstream ET figures. `temperature-volume` exists only for tools that require a large `x,y,z,T_ET` CSV. `gr-map` and `microstructure` require TC-Python. Composition row, material databases, CET parameters, resolution, process condition, and output paths are CLI-configurable.

The five ET images used directly in the deck are:

- `figures/250_0.5/temperature_field_alloy0_250_0.5.png`
- `figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.png`
- `figures/250_0.5/GR_map_overlayET_alloy0_250_0.5.png`
- `figures/250_0.5/projected_GR_alloy0_250_0.5.png`
- `figures/250_0.5/projected_microstructure_alloy0_250_0.5.png`

## TCAM figures

`workflow_TCAM.py` derives presentation figures from an exported native TCAM mesh and liquidus `G`–`R` table:

```bash
# From the repository root
python workflow_TCAM.py
```

The default output directory is `beamer/figures/TCAM/`. Each step is independent and can be run by itself when its source files are available:

| Step | Principal output | Required existing input when run alone |
| --- | --- | --- |
| `base-map` | `GR_map_alloy0_250_0.5.png` | Alloy spreadsheet; TC-Python, Thermo-Calc license, and configured thermodynamic/kinetic databases |
| `overlay-map` | `GR_map_overlayET_TCAM_alloy0_250_0.png` | Alloy spreadsheet, TCAM liquidus `G`–`R` CSV, and ET liquidus `G`–`R` CSV; TC-Python and Thermo-Calc license |
| `gr-projection` | `TCAM_projected_GR_alloy0_250_0.5.png` | Exported TCAM liquidus `G`–`R` CSV |
| `microstructure` | `TCAM_projected_microstructure_alloy0_250_0.5.png` | Alloy spreadsheet and exported TCAM liquidus `G`–`R` CSV; TC-Python and Thermo-Calc license |
| `temperature` | `ET_TC_temp_yz_<slices>um_inferno.png` | ET temperature VTI, ET metadata CSV, and native TCAM temperature mesh (`.vtu`, `.pvd`, or supported Exodus export) |

Examples:

```bash
python workflow_TCAM.py --steps gr-projection microstructure
python workflow_TCAM.py --steps temperature --slices 0 75 150
python workflow_TCAM.py --help
```

`base-map` calculates the alloy's CET map without overlay points. `overlay-map` places the TCAM and ET liquidus boundaries on that map, `gr-projection` projects TCAM `G` and `R` onto the melt-pool cross-section, and `microstructure` classifies those points using the CET map. The `temperature` step produces paired ET/TCAM YZ slices at the distances supplied with `--slices`.

Current defaults use:

- `figures/data/BU_TCAM/alloy0_250_0.5_row11/result.vtu` for the temperature mesh;
- `figures/data/BU_TCAM/alloy0_250_0.5_row11/TCAM_GR_alloy0_250_0.5.csv` for liquidus points;
- the matching ET VTI, metadata, and liquidus CSV under `figures/250_0.5/`; and
- `figures/TCAM/` as the canonical output directory used by the deck.

The TCAM liquidus CSV is a ParaView export containing the point coordinates, `Gradient_Magnitude`, and `R (m/s)`. The TCAM GUI screenshot is not generated by any workflow step.

The deck's TCAM assets include projected `G`–`R`, CET overlays, projected microstructure, and paired ET/TCAM temperature slices under `figures/TCAM/`. `figures/TCAM_alloy0_250_0.5.png` is a manual Thermo-Calc AM Module screenshot rather than a workflow output.

## Bayesian data folders

The Bayesian experiments use paired ET and TCAM fields for 16 power–velocity combinations: powers of 50, 150, 250, and 350 W crossed with velocities of 0.1, 0.5, 1.0, and 1.5 m/s.

`figures/data/BU_ET/` contains the ET side. `Bayesian/ET_prior.py` evaluated all 16 conditions on a common structured domain and wrote one directory per case containing `ET_temperature.vti`, `metadata.csv`, and an ET heatmap. The compact `ET_Alloy0_4x4.csv` file is the batch manifest.

```bash
# From the repository root
python Bayesian/ET_prior.py
```

`figures/data/BU_TCAM/` contains the corresponding higher-fidelity fields. `TCAM_solver.py` was configured to solve the same 16 conditions and archive each native TCAM result. Afterward, the temperature fields were cropped manually in ParaView to the shared Bayesian-update domain. The liquidus surface, temperature gradient, surface normals, and solidification rate were also calculated/exported manually for each case as `TCAM_GR_*.csv`. `TCAM_Alloy0_4x4.csv` is the TCAM batch manifest.

Each paired case therefore supplies:

- an ET structured temperature prior and metadata under `BU_ET/`;
- a native TCAM result and cropped temperature VTU under `BU_TCAM/`; and
- a ParaView-exported TCAM liquidus `G`–`R` CSV with `Points_0`–`Points_2`, `Gradient_Magnitude`, and `R (m/s)`.

The paired fields use the same spatial domain and coordinate convention needed for Bayesian updating. `ET_prior.py` writes positive `x` as the scan direction to match the raw TCAM frame. The current constants near the top of `TCAM_solver.py` may be configured for a different batch; set its input workbook, alloy, Excel-row range, and output paths before attempting to recreate the Alloy 0 dataset.

## Bayesian-updated figures

`workflow_bayesian.py` collects the Python-controlled parts of both presentation experiments. Run it from the repository root:

```bash
python workflow_bayesian.py
```

Some intermediate operations remain manual ParaView steps. Consequently, it is usually more practical to run selected stages:

| Step | Principal output | Required existing input when run alone |
| --- | --- | --- |
| `single-gpr` | Single-case corrected CSVs, diagnostic figures, and local model bundle | Default 250 W, 0.5 m/s ET VTI/metadata and cropped TCAM VTU |
| `single-gr-projection` | `single_training_case/G_projection_TCAM_vs_corrected.png`, `R_projection_TCAM_vs_corrected.png` | TCAM `G`–`R` CSV and manually exported `single_training_case/BU_GR_data.csv` |
| `single-microstructure` | `single_training_case/microstructure_TCAM_vs_corrected.png` | Same two `G`–`R` CSVs; TC-Python and Thermo-Calc license |
| `single-overlay-map` | `single_training_case/GR_map_overlayET_TCAM_BU_alloy0_250_0.png` | ET, TCAM, and corrected `G`–`R` CSVs; TC-Python and Thermo-Calc license |
| `warped-gpr` | Held-out warped-prior GPR fields, diagnostics, tables, and local model bundle | All 16 paired ET/TCAM cases |
| `warped-gr-projection` | Held-out `G_projection_TCAM_vs_corrected.png`, `R_projection_TCAM_vs_corrected.png` | TCAM `G`–`R` CSV and manually exported `heldout_GR_data.csv` |
| `warped-microstructure` | Held-out `microstructure_TCAM_vs_corrected.png` | Same two `G`–`R` CSVs; TC-Python and Thermo-Calc license |
| `warped-overlay-map` | Held-out `GR_map_overlayET_TCAM_BU_alloy0_250_0.png` | ET, TCAM, and corrected `G`–`R` CSVs; TC-Python and Thermo-Calc license |
| `warped-temperature` | `Corrected_TCAM_temp_yz_0_75_150um_inferno.png` | Held-out prediction VTI, ET metadata, and TCAM mesh |
| `warped-animation` | `warped_et_prior_animation.mp4` and poster PNG | Held-out ET prior; FFmpeg |

Examples:

```bash
python workflow_bayesian.py --steps single-gpr
python workflow_bayesian.py --steps single-gr-projection single-microstructure single-overlay-map
python workflow_bayesian.py --steps warped-gpr
python workflow_bayesian.py --steps warped-gr-projection warped-microstructure warped-overlay-map warped-temperature warped-animation
python workflow_bayesian.py --help
```

### Single-condition Cartesian figures

The presentation, `Bayesian/GPR.py`, and `workflow_bayesian.py` use `figures/bayesian/single_training_case/`. The script pairs `BU_ET/0_250W_0.5ms/ET_temperature.vti` with `BU_TCAM/alloy0_250_0.5_row11/TCAM_cropped_alloy0_250_0.5.vtu`, learns `T_TCAM - T_ET`, and writes `BU_corrected_data.csv`, `sampled_corrected_points.csv`, three diagnostic scatter plots, and the ignored local `.joblib` model.

`BU_GR_data.csv` was not generated by `GPR.py`. `BU_corrected_data.csv` was opened in ParaView, the corrected-temperature liquidus surface was extracted, and `Gradient_Magnitude`, normals, and `R (m/s)` were calculated before exporting the result. The required corrected CSV schema for the comparison scripts is either `x,y,z,G,R` or the ParaView form `Points_0,Points_1,Points_2,Gradient_Magnitude,R (m/s)`.

`Bayesian/CompareGR.py` consumes the corrected `BU_GR_data.csv` and the matching row11 TCAM `G`–`R` CSV to make the side-by-side `G` and `R` projections. `Bayesian/CompareMicrostructure.py` consumes those same two files plus Alloy 0 from `effective_cp_data.xlsx`; it calculates one shared CET map with TC-Python and classifies both liquidus surfaces. The three-way CET overlay was originally created with custom inputs in `microstructure/GR_Map_test.py`; `workflow_bayesian.py` now reproduces it directly using the ET, TCAM, and corrected liquidus CSVs. For the single-case map, TCAM is drawn at the bottom, corrected points in the middle, and ET on top.

`sampled_points_paraview.png` is a manual ParaView screenshot rather than a Python plot.

### Warped Cartesian multi-case figures

The presentation and new recommended-model runs use `figures/bayesian/multi_warped_gpr/heldout_250W_0.5ms_15cases/`. `Bayesian/GPR_multi_warped.py` fits the 15 training conditions, predicts a six-parameter ET warp for the held-out 250 W, 0.5 m/s condition, and models the remaining discrepancy with a Matérn-5/2 GP using `P`, `V`, `x`, `y`, `z`, `r_3d`, and warped ET temperature. The workflow uses `--n-restarts-optimizer 1` to reproduce the recorded presentation result. See [`Bayesian/README.md`](../Bayesian/README.md) for the full method, output schema, timings, metrics, variants, and leakage safeguards.

As in the single-case experiment, `heldout_GR_data.csv` was calculated manually in ParaView from the prediction VTI. It is the corrected input to `CompareGR.py` and `CompareMicrostructure.py`; the row11 TCAM `G`–`R` CSV is the reference input. The three-way overlay uses that corrected file together with the TCAM liquidus CSV and `figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv`. For the warped map, corrected points are drawn above both ET and TCAM so they remain visible.

`Bayesian/PlotTempYZStack.py` was run with the held-out prediction VTI as the left field, `T_pred` as its temperature array, the row11 TCAM VTU as the right field, and slices at 0, 75, and 150 µm behind the laser. `workflow_bayesian.py` supplies these custom inputs and interpolates only the plotted corrected YZ planes to 2 µm spacing in memory; it does not resample or rewrite the coarse prediction VTI. Use `--warped-temperature-spacing-um 4` to generate the figure with 4 µm slice spacing instead.

The following are manual ParaView screenshots and cannot be regenerated by the workflow:

- `et_paraview.png`
- `warped_et_paraview.png`
- `prediction_paraview.png`
- `prediction_paraview_no_colorbar.png`

`Bayesian/animate_warped_et.py` creates both `warped_et_prior_animation.mp4` and `warped_et_prior_animation_poster.png`.

## Other generated illustrations

- `python beamer/kgt_tip_radii.py` regenerates `figures/kgt_tip_radii.png`.
- `python beamer/generate_ca_microstructures.py` generates the local cellular-automaton morphology illustrations.
- `beamer/mumatscale/README.md` documents the MuMatScale planar, columnar, and equiaxed/branched examples rendered into `figures/ca_modes/`.

Several figures are sourced or manually composed and do not have a complete automated pipeline, including ParaView screenshots, the TCAM GUI screenshot, logos, literature figures, and the closing composite. Preserve attribution shown in the slide source when replacing these assets.

## Reproducibility notes

- Run Python workflows from the repository root and compile LaTeX from `beamer/`.
- `Bayesian/ET_prior.py` writes ET fields with positive `x` as the laser scan direction, matching raw TCAM coordinates. `workflow_ET.py` retains the native `eagar_tsai` convention: positive `x` points into the trailing wake, so the laser scans toward negative `x`.
- Trained GPR `.joblib` bundles are omitted because they are too large for ordinary GitHub storage. Rerun the corresponding GPR training workflow to recreate a model bundle before using prediction-only or re-export features.
- A clean scientific rerun needs the matching TCAM exports, Thermo-Calc databases (`TCHEA8` and `MOBHEA3` by default), and a valid license.
