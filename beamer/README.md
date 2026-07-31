# SFF Symposium Beamer Presentation

This is the presentation for SFF Symposium 2026 by Kyle Swartz titled, "Rapid Microstructure Prediction in Laser Powder Bed Fusion Using Bayesian Updating of Eagar–Tsai Model." This README contains information on figure creation. While some are already included in the repo, the raw data is provided to easily replicate most KGT, ET, and TCAM related figures in the slidedeck.

## Figures
### ET-Related Figures

`workflow_ET.py` regenerates the five ET figures used by the presentation and the intermediate data consumed by the TCAM comparison. By default it runs the VTI-based ET-to-microstructure pipeline for Alloy 0 (Hf<sub>2</sub>Mo<sub>2</sub>Ta<sub>48</sub>W<sub>48</sub>) at P = 250 W and V = 0.5 m/s. The large 3D temperature CSV is no longer part of the default run:

```python
DEFAULT_EXCEL = "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V", "Co", "Cr", "Fe", "Mn", "Ni"]
DEFAULT_CALC_ROOT = "beamer/figures"
```

The ET workflow reads only these exact columns from `effective_cp_data.xlsx`; it does not use alternate column names or fallbacks:

```text
Velocity (m/s)
Power (W)
Beam Diam (m)
Absorptivity
Liquidus (K)
THCD LT (W/mK)
RT Density (kg/m3)
Cp, Sheikh (J/kgK)
```

Run the complete workflow from the repository root:

```bash
python workflow_ET.py
```

Use `--steps` to generate individual figures or intermediate files:

```bash
python workflow_ET.py --steps temperature-field
python workflow_ET.py --steps temperature-field temperature-3d
python workflow_ET.py --steps gr-projection microstructure
```

Selected steps run in pipeline order and reuse prerequisite files already in the condition output directory. For example, `gr-projection` and `microstructure` reuse the existing liquidus CSV. Include prerequisite steps in the same command when those files do not exist.

| Step | Output | Existing prerequisite when run alone |
| --- | --- | --- |
| `temperature-field` | `temperature_field_*.png`, `ET_meta_*.csv` | None |
| `temperature-volume` (optional) | Legacy `ET_*.csv` | None |
| `temperature-3d` | `ET_3D_temperature_*.png`, `ET_3D_temperature_*.vti` | `ET_meta_*.csv` |
| `liquidus` | `liquidus_GR_*.csv` | `ET_meta_*.csv`, `ET_3D_temperature_*.vti` |
| `gr-map` | `GR_map_overlayET_*.png` | `liquidus_GR_*.csv` |
| `gr-projection` | `projected_GR_*.png` | `liquidus_GR_*.csv` |
| `microstructure` | `projected_microstructure_*.png` | `liquidus_GR_*.csv` |

The `gr-map` and `microstructure` steps require a working Thermo-Calc/TC-Python installation and license. Run `python workflow_ET.py --help` for process, resolution, database, and output-directory overrides.

The `ET_3D_temperature_*.vti` file is the canonical 3D ET temperature field.
It is consumed directly by the G/R extraction, GPR interpolation, and YZ
temperature plotting code. Use `--steps temperature-volume` only when a
legacy or external tool specifically requires an `x,y,z,T_ET` CSV.

`Bayesian/ET_prior.py` follows the same convention for batch cases: every
case writes `ET_temperature.vti`, `metadata.csv`, and (unless disabled) an
inferno heatmap. Add `--export-csv` only when a tabular copy is needed. The
parent `ET_Alloy#_4x4.csv` remains a compact manifest across the selected
Excel rows (for example, `ET_Alloy0_4x4.csv`).
ET-prior VTI, CSV, and heatmap outputs use positive x as the laser scanning
direction, matching the raw TCAM coordinate frame in ParaView.

### Multi-case Bayesian update

`Bayesian/GPR_multi.py` trains the ET-to-TCAM residual model on complete P-V
cases and evaluates one P-V case that was excluded from fitting. Its default
experiment uses all 15 remaining P-V conditions and holds out
`250 W, 0.5 m/s`:

```bash
python Bayesian/GPR_multi.py
```

The held-out TCAM VTU is not opened until after GPR fitting. It is then used
only to calculate whole-case test metrics and parity plots. To choose a
different split, repeat `--training-case` exactly once per training condition:

```bash
python Bayesian/GPR_multi.py \
  --held-out-case 350,1 \
  --training-case 50,0.1 \
  --training-case 50,1.5 \
  --training-case 150,0.5 \
  --training-case 150,1 \
  --training-case 250,0.1 \
  --training-case 250,1.5 \
  --training-case 350,0.5 \
  --training-case 350,1.5
```

Each case contributes the same maximum number of sampled points. A dedicated
fraction is reserved for evaporation-plateau points when they exist. Final
temperature predictions retain the unconstrained GPR result and a
physics-constrained result bounded by the 298 K ambient temperature and the
independently supplied `Boiling Onset (K)` value.

The CET model inputs can be overridden from the command line. Their workflow
defaults are a primary phase of `FCC_L12`, interfacial energy of `0.5 J/m2`,
nucleation-site density of `4.0e11 1/m3`, nucleation undercooling of `4.0 K`,
and equiaxed exponent of `3.13`:

```bash
python workflow_ET.py --steps gr-map microstructure \
  --primary-phase FCC_L12 \
  --interfacial-energy 0.5 \
  --cet-nucleation-sites 4.0e11 \
  --cet-nucleation-undercooling-k 4.0 \
  --cet-equiaxed-exponent 3.13
```

**`figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.png`**

- Description: 3D ET temperature field exported through VTI and rendered for the presentation.
- Workflow step: `temperature-3d`.

**`figures/250_0.5/GR_map_overlayET_alloy0_250_0.5.png`**

- Description: ET liquidus-boundary `G` and `R` points overlaid on the alloy's GR map calculated with TC-Python.
- Workflow step: `gr-map`.

**`figures/250_0.5/projected_GR_alloy0_250_0.5.png`**

- Description: `G` and `R` along the positive-`R` solidification front, projected onto a `yz` slice of the melt pool.
- Workflow step: `gr-projection`.

**`figures/250_0.5/projected_microstructure_alloy0_250_0.5.png`**

- Description: ET solidification-front points classified from the GR map and projected onto a `yz` slice of the melt pool.
- Workflow step: `microstructure`.

**`figures/250_0.5/temperature_field_alloy0_250_0.5.png`**

- Description: Temperature heatmaps of the melt pool showing length, width, and depth with the liquidus outlined in cyan.
- Workflow step: `temperature-field`. This step also writes `ET_meta_alloy0_250_0.5.csv`.

### TCAM-Related Figures
`workflow_TCAM.py` regenerates the five derived TCAM figures used by the presentation. It uses the same Alloy 0 row and TC-Python settings as `workflow_ET.py`:

```python
DEFAULT_EXCEL = "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V", "Co", "Cr", "Fe", "Mn", "Ni"]
DEFAULT_THERMO_DB = "TCHEA8"
DEFAULT_KINETIC_DB = "MOBHEA3"
DEFAULT_PRIMARY_PHASE = "BCC_B2"
DEFAULT_INTERFACIAL_ENERGY = 0.5
DEFAULT_TCAM_GR_CSV = (REPO_ROOT / "beamer/figures/data/TCAM_GR_alloy0_250_0.5.csv")
DEFAULT_ET_GR_CSV = (REPO_ROOT / "beamer/figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
DEFAULT_ET_TEMPERATURE_FIELD = (REPO_ROOT / "beamer/figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.vti")
DEFAULT_ET_METADATA_CSV = (REPO_ROOT / "beamer/figures/250_0.5/ET_meta_alloy0_250_0.5.csv")
DEFAULT_TCAM_MESH = REPO_ROOT / "beamer/figures/data/BU_TCAM/alloy0_P250_V0p5_row2/result.pvd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "beamer/figures/TCAM"
```

The workflow expects these exported inputs:

- `figures/data/TCAM_GR_alloy0_250_0.5.csv`: TCAM liquidus points exported through ParaView, including `Points_0`, `Points_1`, `Points_2`, `Gradient_Magnitude`, and `R (m/s)`.
- `figures/data/BU_TCAM/alloy0_P250_V0p5_row2/result.pvd`: the connected native TCAM mesh containing the `temperature` array. A GUI-exported `result.e` is also supported.
- `figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv`, `ET_3D_temperature_alloy0_250_0.5.vti`, and `ET_meta_alloy0_250_0.5.csv`: ET outputs created by `workflow_ET.py`.

Run all TCAM figure steps from the repository root:

```bash
python workflow_TCAM.py
```

Individual steps can be regenerated without running the entire workflow:

```bash
python workflow_TCAM.py --steps gr-projection microstructure
python workflow_TCAM.py --steps temperature --slices 0 80 160
```

The GR-map and microstructure steps require a working Thermo-Calc/TC-Python installation and license. Run `python workflow_TCAM.py --help` for input and output path overrides.

**`figures/TCAM/GR_map_alloy0_250_0.5.png`**

- Description: Thermal-gradient (`G`) versus solidification-rate (`R`) map calculated with the Thermo-Calc CET Property Model.
- Workflow step: `base-map`.

**`figures/TCAM/TCAM_projected_GR_alloy0_250_0.5.png`**

- Description: TCAM-derived `G` and `R` along the positive-`R` solidification front, projected onto a `yz` slice of the melt pool.
- Workflow step: `gr-projection`.

**`figures/TCAM/GR_map_overlayET_TCAM_alloy0_250_0.png`**

- Description: TCAM and ET liquidus-boundary points overlaid on the alloy's TC-Python GR map. TCAM points are lime and ET points are gold.
- Workflow step: `overlay-map`.

**`figures/TCAM/TCAM_projected_microstructure_alloy0_250_0.5.png`**

- Description: TCAM solidification-front points classified from the GR map and projected onto a `yz` slice of the melt pool.
- Workflow step: `microstructure`.

**`figures/TCAM/ET_TC_temp_yz_0_80_160um_inferno.png`**

- Description: Paired ET and TCAM temperature slices at `x = 0`, `80`, and `160 um`, using a shared inferno scale and cyan liquidus contours. The gray TCAM region is the keyhole vapor cavity.
- Workflow step: `temperature`.

**`figures/TCAM_alloy0_250_0.5.png`**

- Description: Thermo-Calc AM Module GUI view of the melt pool. This screenshot is not generated by `workflow_TCAM.py`.
- Manual step: In TCAM, use these parameters:

    ```text
    TCHEA8: Hf2Mo2Ta48W48 (at.%)
    Base plate temperature: 298 K
    Fluid flow: ON
    Height: 0.22 mm
    Mesh: Fine
    Power: 250 W
    Wavelength: 1070 nm
    Use keyhole model: ON
    Scanning speed: 500 mm/s
    Layer thickness: 30 um
    ```

## Files

- `sff_beamer.tex`: SFF Symposium presentation deck
- `sff_theme.sty`: reusable SFF Beamer theme

## Usage

Compile from this directory with a standard LaTeX workflow, for example:

```bash
pdflatex sff_beamer.tex
```

or:

```bash
latexmk -pdf sff_beamer.tex
```

## Notes

The presentation was compiled here with `tectonic` to produce `sff_beamer.pdf`.
