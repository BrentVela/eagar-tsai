# SFF Symposium Beamer Template

This template started from the presentation styling in `/home/vela/proj/arl/arl_fellowship_presentation.tex`, then was refit into a reusable SFF Symposium themed deck with slide-specific content removed.

## Figures
### workflow.py - ET-Related Figures
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

2. **`figures/250_0.5/GR_map_overlayET_alloy0_250_0.5.png`**
- Description: Overlay of ET liquidus GR points onto its GR map, calculated with TC-Python.
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

### TCAM-Related Figures

**`figures/250_0.5/GR_map_alloy0_250_0.5.png`**
- Description: Thermal Gradient (G) - Solidification Rate (R) map calculated with TC-Python.
- Manual step: Run `microstructure/GR_Map_test.py`, and ensure the following:<br/>
    ```
    output_path = "beamer-template/figures/250_0.5/GR_map_alloy0_250_0.5.png"
    include_overlay = False
    include_corrected_overlay = False
    show_cet_lines = False
    ```

**`figures/TCAM/GR_map_overlayET_TCAM_alloy0_250_0.5.png`**
- Description: Overlay of ET and TCAM liquidus GR points onto the alloy's GR map, calculated with TC-Python.
- Manual step: Run `microstructure/GR_Map_test.py`, and ensure the following:
    ```
    excel_path = REPO_ROOT / "effective_cp_data.xlsx"
    row_index = 2  # EXCEL ROW - 2
    element_cols = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
    thermo_db = "TCHEA8"
    kinetic_db = "MOBHEA3"
    primary_phase = "BCC_B2"
    interfacial_energy = 0.5
    output_path = "beamer-template/figures/TCAM/GR_map_overlayET_TCAM_alloy0_250_0.png"
    include_overlay = True
    include_corrected_overlay = False
    show_cet_lines = False

    if include_overlay:
        overlay_csv = [REPO_ROOT / "beamer-template/figures/data/TCAM_GR_alloy0_250_0.5.csv"]
        overlay_label = ["TCAM liquidus boundary"]
        overlay_kwargs = [
            {"c": "lime", "s": 3, "alpha": 0.1},
        ]

    if include_overlay:
        overlay_csv.append(REPO_ROOT / "beamer-template/figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
        overlay_label.append("ET liquidus boundary")
        overlay_kwargs.append({"c": "gold", "s": 4, "alpha": 0.35})
    ```

**`figures/TCAM/TCAM_alloy0_250_0.5.png`**
- Description: Thermo-Calc AM Module GUI view of the melt pool.
- Manual step: In TCAM, use these parameters:<br/>
    ```
    TCHEA8: Hf<sub>2</sub>Mo<sub>2</sub>Ta<sub>48</sub>W<sub>48</sub>
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

**`figures/TCAM/TCAM_projected_GR_alloy0_250_0.5.png`**
- Description: Thermo-Calc AM Module GUI view of the melt pool.
- Manual step: Run `microstructure/PlotMicrostructure.py`, and ensure the following:<br/>
    ```
    LIQUIDUS_CSV = ("beamer-template/figures/data/TCAM_GR_alloy0_250_0.5.csv")
    OUTPUT_PATH = ("beamer-template/figures/TCAM/TCAM_projected_microstructure_alloy0_250_0.5.png")
    INTERPOLATE_PARAVIEW = False
    INTERPOLATION_Y_POINTS = 75
    INTERPOLATION_Z_POINTS = 115
    SMOOTHING_SIGMA = 0.0
    POINT_SIZE = 10
    MICRON_SCALE_THRESHOLD = 1.0e-2
    GRADIENT_K_PER_UM_THRESHOLD = 1.0e4
    ```

**`figures/250_0.5/TCAM_projected_GR_alloy0_250_0.5.png`**
- Description: TCAM-derived G and R along the liquidus, projected onto a yz slice of the melt pool. Only points with positive R are valid, so the projection is the back of the melt pool, the solidification front.
- Manual step: Run `microstructure/PlotGR.py`, and ensure the following:<br/>
    ```
    LIQUIDUS_CSV = ("beamer-template/figures/data/TCAM_GR_alloy0_250_0.5.csv")
    OUTPUT_PATH = ("beamer-template/figures/TCAM/TCAM_projected_GR_alloy0_250_0.5.png")
    INTERPOLATE_PARAVIEW = False
    INTERPOLATION_Y_POINTS = 75
    INTERPOLATION_Z_POINTS = 115
    SMOOTHING_SIGMA = 0.0
    POINT_SIZE = 7
    MICRON_SCALE_THRESHOLD = 1.0e-2
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

## Notes

The template was compiled here with `tectonic` to produce `arl_beamer_template.pdf`.
