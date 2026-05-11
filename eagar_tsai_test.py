from eagar_tsai import BeamParameters, MaterialProperties, SimulationDomain, compute_single_point

beam = BeamParameters(
        beam_diameter=8e-5,
        power=250.0,
        velocity=0.5,
        absorptivity=0.590243085
)
mat  = MaterialProperties(
        liquidus_temperature=3454.846467,
        thermal_conductivity=23.74950354,
        density=18038.92072,
        specific_heat=627.2569783511199
)

domain = SimulationDomain(
        x_length_um=600,
        y_length_um=200,
        z_depth_um=150,
        spatial_resolution_um=1.0,
)

result = compute_single_point(beam, mat, domain)
print("length (µm) =", result.length_um)                        # melt pool length in µm
print(result.temperature_field.T_xy.shape)     # (ny, nx) — surface plane in Kelvin
print(result.temperature_field.T_xz.shape)     # (nz, nx) — depth cross-section in Kelvin

fig = result.plot(output="CalcFiles/Test10/temperature_field_alloy0_Cp627.png") # equivalently: result.temperature_field.plot(output="temperature_field.png")



# import pandas as pd
# from eagar_tsai import compute_melt_pool

# df = pd.DataFrame({
#     "velocity_m_s":              [0.5],
#     "power_w":                   [250.0],
#     "beam_diameter_m":           [8e-5],
#     "absorptivity":              [0.590243085],
#     "liquidus_temperature_k":    [3454.846467],
#     "thermal_conductivity_w_mk": [23.74950354],
#     "density_kg_m3":             [18038.92072],
#     "specific_heat_j_kgk":       [2084.7584814613733],
# })

# result = compute_melt_pool(
#     df,
#     workers=12,                # parallel worker processes (default: None, serial)
#     chunk_size=50,            # rows per worker chunk (default: 50)
#     output_dir="CalcFiles/Test10/chunks_alloy0_Cp2084",  # write per-chunk CSVs (default: None)
# )
# print(result[["melt_length_um", "melt_width_um", "melt_depth_um"]])