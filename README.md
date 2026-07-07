# Rapid Microstructure Prediction in Laser Powder Bed Fusion Using Bayesian Updating of Eagar–Tsai Model 

This first section of the README details contributions by Kyle Swartz. To see the original Eagar-Tsai code implemented by Brent Vela, scroll past this section.

## Scripts 

Note: many of these files require you to hardcode the input and output file paths. Additionally, most ET functionality has been updated to use the [eagar-tsai Python library](https://arroyavelab.github.io/eagar-tsai).

- `workflow.py` Streamlines the ET outputs into one file. Outputs a temperature heatmap, metadata file, 3D temperature distribution (.csv and .vti), GR and liquidus data, GR projection, GR map overlay, and microstructure projection. 

microstructure/:
- `GR_Map_test.py` Run this file to calculate GR maps with GR_Map.py. This file requires alloy data as input and can overlay liquidus GR points onto the GR map if provided in a csv.
- `GR_Map.py` Code written by James Hanagan to calculate GR maps using TC-Python.
- `GRFrom3D.py` Calculates thermal gradient (G) and solidification rate (R) given a .vti file, and outputs a csv.
- `GRFromHeatmap.py` Calculates thermal gradient (G) and solidification rate (R) given a 2D ET temperature field. Mostly obsolete.
- `PlotGR.py` Plots G and R as a melt pool projection. Requires a csv as input.
- `PlotMicrostructure.py` Using a GR Map, classifies the microstructure of points along the liquidus and plots it as a melt pool projection. Uses GR_Map.py and requires a csv as input.

Bayesian/:
- `ET_prior.py` Given an excel file with alloy data, outputs a 3D temperature distribution and a temperature heatmap. This is the data used in the GPR.
- `GPR.py` Given an ET 3D temperature distribution csv and a TCAM 3D temperature distribution csv, will learn the error in temperature between ET and TCAM at a number of training points and predicts the corrected melt pool. Outputs a csv that can be viewed in ParaView. Keep in mind that the TCAM data csv must be cropped in ParaView to the exact domain as the ET data.
- `melt_geometry.py` Given the csv outputted by GPR.py, will print in the terminal the melt pool depth, width, and length for ET, TCAM, and Corrected melt pools. To run:
   ```bash
   python Bayesian/melt_geometry.py pathto.../matched_tcam_corrected_points.csv
   ```
- `PlotTempYZ.py` Can plot two melt pool temperature yz slices side by side. Requires temperature data from two different sources.

Miscellaneous:
- `PrintabilityMap_og.py` Uses old ET code to calculate melt pool geometry for large power-velocity spaces.
- `PlotMeltDepth.py` Uses data from PrintabilityMap_og.py to generate a melt depth map as a function of power and velocity.
- `PlotPrintabilityMap.py` Uses data from PrintabilityMap_og.py to generate a printability map.
- `PrintabilityMap_ET.py` Uses eagar-tsai Python library to calculate printability maps.
- `effective_heat_capacity.py` Old code to make calculations. Ignore.
- `TCAM_solver.py` Work in progress to run many TCAM simulations with TC-Python.
- `eagar_tsai_test.py` Used to run simple ET calculations for quick testing. Outputs a temperature heatmap.
- `ETtoVTI.py` Creates only a .vti file from an ET run.
- `ViewVTI.py` Views a .vti file to see 3D eagar-tsai temperature distribution.

Data:
- `effective_heat_capacity.xlsx` Actually important! This is where I have my updated alloy data using the TCHEA8 CALPHAD database. Includes boiling points and effective heat capacities using different methods. Still, happens to be the main source of data for the other scripts.
- `et_custom_input_data.xlsx` Shortened version of et_input_data_example.xlsx to navigate easier. Not really used anymore.
- `et_input_data_example.xlsx` Original refractory high entropy data sheet provided by Brent Vela. Uses TCHEA5 the CALPHAD database, so a little outdated compared to effective_heat_capacity.xlsx. 
- `TCAM_solver_data.xlsx` Work in progress. Works in tandem with TCAM_solver.py.

#
# ET Model (Eagar–Tsai)

This project implements the Eagar–Tsai moving heat source model to estimate melt pool dimensions for a scanning laser/beam over a semi‑infinite solid. The temperature field is computed from a 1D integral and evaluated numerically (optionally using a compiled C integrand for speed). Melt pool dimensions are extracted from the liquidus isotherm.

Attribution: The C integrand implementation is based on a reformulation by Sasha Rubenchik (LLNL, 2015).

Reference: T. W. Eagar and N.-S. Tsai, “Temperature Fields Produced by Traveling Distributed Heat Sources,” Welding Journal (Research Supplement), December 1983, pp. 346‑s–354‑s (see `original_ET_paper.pdf`).

Refactor note: Brent Vela refactored this code on January 21, 2026.

## Inputs (per row)

Beam/process:
- `Velocity_m/s` (scan speed, v) [m/s]
- `Power` (laser power, P) [W]
- `Beam_diameter_m` (beam diameter, 2σ) [m]
- `Absorptivity` (A) [unitless]

Material (liquidus properties):
- `T_liquidus` [K]
- `thermal_cond_liq` (k) [W/(m·K)]
- `Density_kg/m3` (ρ) [kg/m^3]
- `Cp_J/kg` (cp) [J/(kg·K)]

Fixed/implicit:
- Ambient temperature `t0 = 300 K`
- Domain size and spatial resolution (defaults in `compute_melt_pool`)

## Outputs (per row)

- `melt_length` [m]
- `melt_width` [m]
- `melt_depth` [m]
- `melt_length_um` [µm]
- `melt_width_um` [µm]
- `melt_depth_um` [µm]
- `peakT` [K]
- `minT` [K]

## Equations (as implemented)

Thermal diffusivity:
```
alpha = k / (rho * cp)
```

Non‑dimensional parameter:
```
p = alpha / (v * sigma)
```

Prefactor:
```
Ts = (A*P) / (pi*(k/alpha)*sqrt(pi*alpha*v*(sigma^3)))
```

Temperature field at (x,y,z):
```
T = t0 + Ts * ∫_0^∞ f(t, x, y, z, p) dt
```

Integrand:
```
f(t,x,y,z,p) = 1 / ((4 p t + 1) * sqrt(t))
               * exp( -z^2/(4t) - ((y^2 + (x - t)^2)/(4 p t + 1)) )
```

## Assumptions

- Semi‑infinite solid; domain only used for numerical evaluation.
- Constant material properties at liquidus.
- No melt flow, vaporization, or latent heat effects.
- Gaussian heat source with constant absorptivity.
- Steady‑state moving source; integral is evaluated numerically.

## Platform Build Notes (eagar_tsai_integrand)

This project uses a small C helper (`eagar_tsai_integrand.c`) that can be compiled into a shared library to speed up the Eagar–Tsai integration. The Python code will use the compiled library if it is present, and fall back to the slow interpreted integrator if not.

### Windows

#### Option A: MSVC (Visual Studio Developer Command Prompt)

1. Open the **Developer Command Prompt for VS**.
2. From the project folder, run:
   ```bat
   build_windows_dll.bat
   ```

#### Option B: MinGW-w64 (gcc)

1. Install MinGW-w64 and ensure `gcc` is on your PATH.
2. From the project folder, run:
   ```bat
   build_windows_dll.bat
   ```

#### PowerShell alternative

If you prefer PowerShell, run:
```powershell
.\build_windows_dll.ps1
```

This produces `libeagar_tsai_integrand.dll` in the same directory as `et_melt_pool_script.py`.

### Linux

From the project folder:
```bash
gcc -O3 -fPIC -shared -o libeagar_tsai_integrand.so eagar_tsai_integrand.c
```

This produces `libeagar_tsai_integrand.so` in the same directory as `et_melt_pool_script.py`.

### macOS

From the project folder:
```bash
clang -O3 -fPIC -shared -o libeagar_tsai_integrand.dylib eagar_tsai_integrand.c
```

This produces `libeagar_tsai_integrand.dylib` in the same directory as `et_melt_pool_script.py`.

### Notes

- The Python code looks for these files by name in the working directory:
  - Windows: `libeagar_tsai_integrand.dll`
  - Linux: `libeagar_tsai_integrand.so`
  - macOS: `libeagar_tsai_integrand.dylib`
- The exported C symbol name that Python loads is `eagar_tsai_integrand`.
- If the compiled library is not found, the code will fall back to the interpreted integrator (much slower).
- If you change the C function name or file name, rebuild the shared library before running Python.


## Validation Against Old Predictions

If you have legacy ET outputs (e.g., `ET_width_old`, `ET_depth_old`) in the input sheet, you can validate the new run by:

1. Run the model to generate fresh `melt_width`, `melt_depth`, and `melt_length`.
2. Compare the new outputs to the old columns using summary error metrics and parity plots.

Example (Python):
```python
import pandas as pd
from et_melt_pool_script import compute_melt_pool

# Load input with old ET columns (renamed to *_old)
df = pd.read_excel("et_input_data_example.xlsx")

# Map material columns used by the model
# (Adjust names if your sheet uses different headers)
df["T_liquidus"] = df["PROP LT (K)"]
df["thermal_cond_liq"] = df["PROP LT THCD (W/(mK))"]
df["Density_kg/m3"] = df["PROP RT Density (kg/m3)"]
df["Cp_J/kg"] = df["PROP LT C (J/(kg K))"]
df["Beam_diameter_m"] = df["Beam Diam (m)"]

# Run ET model
out = compute_melt_pool(df, workers=1, chunk_size=50)

# Simple error metrics vs. legacy columns
for col_new, col_old in [("melt_width", "ET_width_old"), ("melt_depth", "ET_depth_old")]:
    if col_old in out.columns:
        diff = out[col_new] - out[col_old]
        mae = diff.abs().mean()
        rmse = (diff.pow(2).mean()) ** 0.5
        print(col_new, "MAE:", mae, "RMSE:", rmse)
```

Recommended checks:
- Parity plots for width/depth.
- MAE/RMSE summary and max absolute error.
- Stratify errors by process parameters (power, speed) to spot systematic drift.
