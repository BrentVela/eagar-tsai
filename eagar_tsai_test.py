# from pathlib import Path

# import pandas as pd

# from eagar_tsai import compute_melt_pool, SimulationDomain


# INPUT_XLSX = "effective_cp_data.xlsx"
# OUTPUT_CSV = "CalcFiles/Test15/Cp-LT_results_alloy13.csv"
# START_EXCEL_ROW = 12


# # Placeholder input column names in the Excel sheet:
# #   Alloy, Composition,
# #   alloy_name, speed_m_s, power_w, beam_diameter_m, absorptivity,
# #   t_liquidus_k, k_liq_w_mk, density_kg_m3, cp_j_kgk
# #
# # The script renames those placeholders to the column names expected by
# # the eagar_tsai library.
# COLUMN_MAP = {
#     "Velocity (m/s)": "velocity_m_s",
#     "Power (W)": "power_w",
#     "Beam Diam (m)": "beam_diameter_m",
#     "Absorptivity": "absorptivity",
#     "Liquidus (K)": "liquidus_temperature_k",
#     "THCD LT (W/mK)": "thermal_conductivity_w_mk",
#     "RT Density (kg/m3)": "density_kg_m3",
#     "Cp, LT (J/kgK)": "specific_heat_j_kgk",
# }

# front_columns = ["Alloy", "Composition"]

# df = pd.read_excel(INPUT_XLSX)
# df = df.rename(columns=COLUMN_MAP)

# # Excel row numbers are 1-based and row 1 is the header.
# # Set START_EXCEL_ROW to the first data row you want to process.
# if START_EXCEL_ROW > 2:
#     df = df.iloc[START_EXCEL_ROW - 2 :].reset_index(drop=True)

# model_columns = [
#     "Alloy",
#     "Composition",
#     "velocity_m_s",
#     "power_w",
#     "beam_diameter_m",
#     "absorptivity",
#     "liquidus_temperature_k",
#     "thermal_conductivity_w_mk",
#     "density_kg_m3",
#     "specific_heat_j_kgk",
# ]
# missing = [col for col in model_columns if col not in df.columns]
# if missing:
#     raise ValueError(
#         f"Missing required columns in {INPUT_XLSX}: {', '.join(missing)}"
#     )

# domain = SimulationDomain(
#     x_length_um=600,
#     y_length_um=200,
#     z_depth_um=150,
#     spatial_resolution_um=1.0,
# )

# required_numeric_columns = [
#     "velocity_m_s",
#     "power_w",
#     "beam_diameter_m",
#     "absorptivity",
#     "liquidus_temperature_k",
#     "thermal_conductivity_w_mk",
#     "density_kg_m3",
#     "specific_heat_j_kgk",
# ]

# output_columns = [
#     "melt_length",
#     "melt_width",
#     "melt_depth",
#     "melt_length_um",
#     "melt_width_um",
#     "melt_depth_um",
#     "peak_temperature",
#     "min_temperature",
# ]

# good_mask = ~df[required_numeric_columns].isna().any(axis=1)
# good_df = df.loc[good_mask].copy()
# bad_df = df.loc[~good_mask].copy()

# # Rows with missing required numeric inputs bypass the solver and come back
# # with NaN in every non-identity column.
# if not bad_df.empty:
#     for col in bad_df.columns:
#         if col not in front_columns:
#             bad_df[col] = pd.NA

# if good_df.empty:
#     result = bad_df.reindex(columns=list(df.columns) + output_columns)
# else:
#     print("Starting calculation...")
#     good_result = compute_melt_pool(
#         good_df,
#         domain=domain,
#         workers=12,
#         chunk_size=50,
#         output_dir=None,
#         return_field=False,
#     )

#     if bad_df.empty:
#         result = good_result
#     else:
#         bad_df = bad_df.reindex(columns=good_result.columns)
#         result = pd.concat([good_result, bad_df], ignore_index=True)

# Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
# result = result[front_columns + [col for col in result.columns if col not in front_columns]]
# result.to_csv(OUTPUT_CSV, index=False)

# cols_to_print = [col for col in ["Alloy", "melt_length_um", "melt_width_um", "melt_depth_um"] if col in result.columns]
# print(result[cols_to_print])

import pandas as pd

from eagar_tsai import SimulationDomain, compute_melt_pool

df = pd.DataFrame({
    "velocity_m_s":              [0.10],
    "power_w":                   [400.0],
    "beam_diameter_m":           [8e-5],
    "absorptivity":              [0.590243085],
    "liquidus_temperature_k":    [3454.8464667112],
    "thermal_conductivity_w_mk": [36.27217],
    "density_kg_m3":             [18038.93493],
    "specific_heat_j_kgk":       [264.1654169],
})

domain = SimulationDomain(
    x_length_um=600,
    y_length_um=200,
    z_depth_um=150,
    spatial_resolution_um=1.0,
)

result = compute_melt_pool(
    df,
    domain=domain,
    workers=12,
    chunk_size=50,
    output_dir="CalcFiles/Test19/400_0.10",
    return_field=True,
)
print(result[["melt_length_um", "melt_width_um", "melt_depth_um"]])
result.loc[0, "temperature_field"].plot(output="CalcFiles/Test19/400_0.10/temperature_field_alloy0_400_0.10.png")
