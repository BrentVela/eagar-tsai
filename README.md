# ETtoKGT

Tools for rapid microstructure prediction in laser powder bed fusion. The repository combines the Eagar–Tsai (ET) analytical thermal model, Thermo-Calc Additive Manufacturing (TCAM) reference simulations, Gaussian-process discrepancy correction, thermal-gradient/solidification-rate (`G`–`R`) extraction, and CET/KGT-based microstructure analysis.

The current demonstration uses Hf<sub>2</sub>Mo<sub>2</sub>Ta<sub>48</sub>W<sub>48</sub> and supports the SFF Symposium 2026 presentation, “Rapid Microstructure Prediction in Laser Powder Bed Fusion Using Bayesian Updating of Eagar–Tsai Model.” See [`beamer/README.md`](beamer/README.md) for the presentation-specific reproduction guide and [`Bayesian/README_warped_cartesian_gpr.md`](Bayesian/README_warped_cartesian_gpr.md) for the recommended warped-prior GPR model and recorded results.

## What is in the repository?

| Area | Purpose |
| --- | --- |
| `workflow_ET.py` | End-to-end ET temperature, VTI, liquidus `G`–`R`, CET map, and microstructure workflow for one composition/process condition |
| `workflow_TCAM.py` | Regenerates the equivalent figures from exported TCAM data and compares ET with TCAM |
| `workflow_bayesian.py` | Regenerates the Python-controlled single-case and warped-GPR presentation products from prepared ET/TCAM data |
| `Bayesian/ET_prior.py` | Generates ET priors for a batch of power–velocity cases |
| `Bayesian/GPR_multi_warped.py` | Recommended model: learns a Cartesian ET warp and a residual Gaussian process, then predicts a held-out condition without opening its TCAM field |
| `Bayesian/GPR_multi.py`, `GPR_multi_cartesian.py`, `GPR_cylindrical_warped.py`, `GPR.py` | Earlier multicase, Cartesian, cylindrical-warped, and single-case discrepancy models retained for comparison; the active warped workflow is self-contained in `GPR_multi_warped.py` |
| `Bayesian/CompareGR.py`, `CompareMicrostructure.py` | Compare TCAM and Bayesian-updated `G`–`R` fields and resulting microstructures |
| `microstructure/` | `G`–`R` extraction, Thermo-Calc CET maps, projections, and classification utilities |
| `beamer/` | SFF 2026 Beamer deck, generated figures, CA illustrations, and presentation documentation |
| `et_melt_pool_script.py` | Original/refactored standalone ET melt-pool implementation with optional compiled C integrand |

Most scripts expose their current inputs with `python <script> --help`. Some older exploratory scripts still contain hard-coded paths and are best treated as research artifacts.

## Installation

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Two external tools are not installed by `requirements.txt`:

- `tc_python` is distributed with Thermo-Calc and requires a licensed Thermo-Calc installation. It is needed for CET-map and microstructure steps, but not for ET temperature generation or GPR fitting from existing data.
- A LaTeX engine is needed only for the presentation. This checkout is set up for the repo-local `.tools/bin/tectonic`; see the Beamer README for alternatives.

The main material/process table is `effective_cp_data.xlsx`. The current workflows expect column names such as `Velocity (m/s)`, `Power (W)`, `Beam Diam (m)`, `Absorptivity`, `Liquidus (K)`, `THCD LT (W/mK)`, `RT Density (kg/m3)`, and `Cp, Sheikh (J/kgK)`.

## Quick start

Generate the default ET products for Alloy 0 at 250 W and 0.5 m/s:

```bash
python workflow_ET.py
```

The default pipeline writes beneath `beamer/figures/250_0.5/`. It generates a 2D temperature field, a canonical 3D VTI field, liquidus `G`–`R` data, a CET overlay, a projected `G`–`R` field, and a projected microstructure. The CET-dependent steps require TC-Python. Run only selected stages with, for example:

```bash
python workflow_ET.py --steps temperature-field temperature-3d liquidus
python workflow_ET.py --steps gr-map gr-projection microstructure
```

Generate presentation figures from the exported TCAM mesh and `G`–`R` data:

```bash
python workflow_TCAM.py
```

Paths, composition row, process settings, grid resolution, databases, and output directories can be overridden through each workflow's CLI.

## Bayesian update workflow

Generate the 16 default ET priors (Excel rows 12–27) used by the multi-case experiments:

```bash
python Bayesian/ET_prior.py
```

Each case writes `ET_temperature.vti` and `metadata.csv` beneath `beamer/figures/data/BU_ET/`; heatmaps are enabled by default and CSV export is optional.

Run the recommended held-out model:

```bash
python -u Bayesian/GPR_multi_warped.py \
  --radial-feature r_3d \
  --kernel matern52 \
  --n-restarts-optimizer 1
```

The default split trains on 15 complete process cases and holds out 250 W, 0.5 m/s. It first learns a six-parameter Cartesian deformation of the ET field, then fits a Matérn-5/2 GP to the remaining TCAM–ET temperature discrepancy. The blind prediction is written before the held-out TCAM field is opened for evaluation. Outputs include VTI fields, parity plots, uncertainty, metrics, the fitted model bundle, training samples, and warp diagnostics beneath:

```text
beamer/figures/bayesian/multi_warped_gpr/
  heldout_250W_0.5ms_15cases/
```

The detailed model equations, leakage safeguards, controlled variants, output schema, and benchmark results are documented in [`Bayesian/README_warped_cartesian_gpr.md`](Bayesian/README_warped_cartesian_gpr.md).

## Data conventions

- Temperature is in kelvin; coordinates are stored in metres or micrometres as indicated by each file/column.
- Positive `x` is the laser scan direction. ET-prior outputs are aligned to the raw TCAM coordinate frame.
- `ET_temperature.vti` or `ET_3D_temperature_*.vti` is the canonical 3D ET representation. Use CSV export only for legacy or external consumers.
- The multi-case GPR expects paired case directories under `beamer/figures/data/BU_ET/` and `beamer/figures/data/BU_TCAM/`.
- Large raw and generated datasets are retained selectively. Paths under `beamer/figures/data/` may need to be replaced with local TCAM exports for a new experiment.

## Standalone Eagar–Tsai solver

`et_melt_pool_script.py` is the earlier standalone ET implementation. It evaluates the moving Gaussian heat-source integral over a semi-infinite solid and extracts melt length, width, depth, and extrema from the liquidus isotherm. Its assumptions include constant liquidus properties, constant absorptivity, steady motion, and no explicit melt flow, vaporization, or latent heat.

An optional C helper accelerates the numerical integrand. Build it from the repository root:

```bash
# Linux
gcc -O3 -fPIC -shared -o libeagar_tsai_integrand.so eagar_tsai_integrand.c

# macOS
clang -O3 -fPIC -shared -o libeagar_tsai_integrand.dylib eagar_tsai_integrand.c
```

On Windows, run `build_windows_dll.bat` from a Visual Studio Developer Command Prompt or MinGW shell, or use `build_windows_dll.ps1`. Without the library, the solver falls back to the slower interpreted integrator.

The main workflows now use the maintained [`eagar-tsai`](https://arroyavelab.github.io/eagar-tsai) Python package instead of this legacy implementation.

## Other utilities

- `PrintabilityMap_ET.py` calculates ET printability maps; `PrintabilityMap_og.py`, `PlotPrintabilityMap.py`, and `PlotMeltDepth.py` retain the earlier workflow.
- `ETtoVTI.py` and `ViewVTI.py` create and inspect ET VTI fields.
- `Bayesian/PlotTempYZ.py` and `PlotTempYZStack.py` compare temperature slices.
- `Bayesian/animate_warped_et.py` recreates the ET-warp animation used in the presentation (requires FFmpeg).
- `TCAM_solver.py` is experimental automation for TCAM simulations.

## References and attribution

The ET implementation follows T. W. Eagar and N.-S. Tsai, “Temperature Fields Produced by Traveling Distributed Heat Sources,” *Welding Journal* (1983); a copy is included as `original_ET_paper.pdf`. The C-integrand reformulation is based on work by Sasha Rubenchik (LLNL, 2015). The standalone ET code was originally implemented by Brent Vela and refactored in January 2026. `microstructure/GR_Map.py` was written by James Hanagan.
