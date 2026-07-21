# SFF Symposium Beamer Presentation

This is the presentation for SFF Symposium 2026 by Kyle Swartz titled, "Rapid Microstructure Prediction in Laser Powder Bed Fusion Using Bayesian Updating of Eagar–Tsai Model." This README contains information on figure creation. While some are already included in the repo, the raw data is provided to easily replicate most KGT, ET, and TCAM related figures in the slidedeck.

## Figures
### ET-Related Figures

`workflow_ET.py` regenerates the five ET figures used by the presentation and the intermediate data consumed by the TCAM comparison. By default it runs the complete ET-to-microstructure pipeline for Alloy 0 (Hf<sub>2</sub>Mo<sub>2</sub>Ta<sub>48</sub>W<sub>48</sub>) at P = 250 W and V = 0.5 m/s:

```python
DEFAULT_EXCEL = "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
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
| `temperature-volume` | `ET_*.csv` | None |
| `temperature-3d` | `ET_3D_temperature_*.png`, `ET_3D_temperature_*.vti` | `ET_meta_*.csv` |
| `liquidus` | `liquidus_GR_*.csv` | `ET_meta_*.csv`, `ET_3D_temperature_*.vti` |
| `gr-map` | `GR_map_overlayET_*.png` | `liquidus_GR_*.csv` |
| `gr-projection` | `projected_GR_*.png` | `liquidus_GR_*.csv` |
| `microstructure` | `projected_microstructure_*.png` | `liquidus_GR_*.csv` |

The `gr-map` and `microstructure` steps require a working Thermo-Calc/TC-Python installation and license. Run `python workflow_ET.py --help` for process, resolution, database, and output-directory overrides.

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
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
DEFAULT_THERMO_DB = "TCHEA8"
DEFAULT_KINETIC_DB = "MOBHEA3"
DEFAULT_PRIMARY_PHASE = "BCC_B2"
DEFAULT_INTERFACIAL_ENERGY = 0.5
DEFAULT_TCAM_GR_CSV = (REPO_ROOT / "beamer/figures/data/TCAM_GR_alloy0_250_0.5.csv")
DEFAULT_ET_GR_CSV = (REPO_ROOT / "beamer/figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
DEFAULT_ET_TEMPERATURE_CSV = (REPO_ROOT / "beamer/figures/250_0.5/ET_alloy0_250_0.5.csv")
DEFAULT_ET_METADATA_CSV = (REPO_ROOT / "beamer/figures/250_0.5/ET_meta_alloy0_250_0.5.csv")
DEFAULT_TCAM_MESH = REPO_ROOT / "beamer/figures/data/result.e"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "beamer/figures/TCAM"
```

The workflow expects these exported inputs:

- `figures/data/TCAM_GR_alloy0_250_0.5.csv`: TCAM liquidus points exported through ParaView, including `Points_0`, `Points_1`, `Points_2`, `Gradient_Magnitude`, and `R (m/s)`.
- `figures/data/result.e`: the connected TCAM mesh containing the `temperature` array.
- `figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv`, `ET_alloy0_250_0.5.csv`, and `ET_meta_alloy0_250_0.5.csv`: ET outputs created by `workflow_ET.py`.

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
