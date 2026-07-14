# Texas A&M Beamer Template

This template started from the presentation styling in `/home/vela/proj/arl/arl_fellowship_presentation.tex`, then was refit into a reusable Texas A\&M-themed deck with slide-specific content removed.

## Figures
### workflow.py
The following 5 figures are automatically created by running `workflow.py`, which runs ET simulations for Alloy 0 (Hf<sub>2</sub>Mo<sub>2</sub>Ta<sub>48</sub>W<sub>48</sub>) at P = 250 W, V = 0.5 m/s. Ensure the following:<br/>
```
DEFAULT_EXCEL = "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
DEFAULT_CALC_ROOT = "beamer-template/figures"
```

**`figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.png`**
- Description: The 3D .vti temperature field.
- Manual step: Run `ETtoVTI.py`, which creates the .vti file and views it to output the figure. Ensure the following:
```
DEFAULT_OUTPUT_VTI = "beamer-template/figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.vti"
DEFAULT_OUTPUT_PNG = "beamer-template/figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.png"
```

**`figures/250_0.5/GR_map_overlayET_alloy0_250_0.5.png`**
- Description: Overlay of liquidus GR points onto its GR map, calculated with TC-Python.
- Manual step: I recommend just using `workflow.py` for this one, as it requires `liquidus_GR_alloy0_250_0.5.csv` and some other changes to get only the ET points overlaid.

**`figures/250_0.5/projected_GR_alloy0_250_0.5.png`**
- Description: G and R along the liquidus, projected onto a yz slice of the melt pool. Only points with positive R are valid, so the projection is the back of the melt pool, the solidification front.
- Manual step: Run `microstructure/PlotGR.py`, and ensure the following:<br/>
```
LIQUIDUS_CSV = ("beamer-template/figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
OUTPUT_PATH = ("beamer-template/figures/250_0.5/projected_GR_alloy0_250_0.5.png")
INTERPOLATE_PARAVIEW = False
INTERPOLATION_Y_POINTS = 75
INTERPOLATION_Z_POINTS = 115
SMOOTHING_SIGMA = 0.0
POINT_SIZE = 9
MICRON_SCALE_THRESHOLD = 1.0e-2
```
- Keep in mind that the input liquidus csv is first created by `workflow.py`.

**`figures/250_0.5/projected_microstructure_alloy0_250_0.5.png`**
- Description: Solidification microstructure along the liquidus, projected onto a yz slice of the melt pool. The microstructure along the back of the melt pool is classified based on the GR map.
- Manual step: Run `microstructure/PlotMicrostructure.py`, and ensure the following:<br/>
```
LIQUIDUS_CSV = ("beamer-template/figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
OUTPUT_PATH = ("beamer-template/figures/250_0.5/projected_microstructure_alloy0_250_0.5.png")
INTERPOLATE_PARAVIEW = False
INTERPOLATION_Y_POINTS = 75
INTERPOLATION_Z_POINTS = 115
SMOOTHING_SIGMA = 0.0
POINT_SIZE = 10
MICRON_SCALE_THRESHOLD = 1.0e-2
GRADIENT_K_PER_UM_THRESHOLD = 1.0e4
```
- Keep in mind that the input liquidus csv is first created by `workflow.py`.

**`figures/250_0.5/temperature_field_alloy0_250_0.5.png`**
- Description: Temperature heatmap of the melt pool showing length, width, and depth. The liquidus is outlined in red.
- Manual step: Run `eagar_tsai_test.py`, and ensure the following:<br/>
```
INPUT_XLSX = "effective_cp_data.xlsx"
ROW_INDEX = 2
OUTPUT_DIR = Path("beamer-template/figures/250_0.5")
OUTPUT_STEM = "alloy0_250_0.5"
```
- Note: this also generates the metadata csv `ET_meta_alloy0_250_0.5.csv`.

### Other Figures

**`figures/250_0.5/GR_map_alloy0_250_0.5.png`**
- Description: Thermal Gradient (G) - Solidification Rate (R) map calculated with TC-Python.
- Manual step: Run `microstructure/GR_Map_test.py`, and ensure the following:<br/>
```
output_path = "beamer-template/figures/250_0.5/GR_map_alloy0_250_0.5.png"
include_overlay = False
include_corrected_overlay = False
show_cet_lines = False
```

## Files

- `arl_beamer_template.tex`: starter deck
- `beamerthemeARLProposal.sty`: reusable Beamer theme

## Usage

Compile from this directory with a standard LaTeX workflow, for example:

```bash
pdflatex arl_beamer_template.tex
```

or:

```bash
latexmk -pdf arl_beamer_template.tex
```

## What was preserved from the source deck

- `Madrid` + `seahorse` Beamer base
- top headline banner with short title and author
- frame-number footline
- block, alert, and timeline slide patterns
- helper emphasis macros: `\stage{}`, `\deliverable{}`, `\risk{}`

## What changed

- Replaced the source palette with a Texas A\&M maroon, charcoal, tan, and gold palette
- Updated the headline prefix and sample institute metadata for Texas A\&M
- Removed presentation-specific slides, references, and images
- Replaced external title art with a self-contained TikZ graphic
- Added reusable example slides for a thesis slide, comparison slide, visual slide, and timeline

## Notes

The template was compiled here with `tectonic` to produce `arl_beamer_template.pdf`.
