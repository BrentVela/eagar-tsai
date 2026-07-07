import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

from eagar_tsai import (
    BeamParameters,
    MaterialProperties,
    SimulationDomain,
    compute_melt_pool,
    compute_temperature_volume,
)


INPUT_XLSX = "effective_cp_data.xlsx"
ROW_INDEX = 2
OUTPUT_DIR = Path("CalcFiles/Test20/eagar_tsai_library/effective_cp_row2_250W_0.5ms_5um")

DOMAIN = SimulationDomain(
    x_length_um=600.0,
    y_length_um=200.0,
    z_depth_um=250.0,
    spatial_resolution_um=5.0,
)


def load_case():
    df = pd.read_excel(INPUT_XLSX)
    row = df.iloc[ROW_INDEX]

    model_df = pd.DataFrame(
        {
            "velocity_m_s": [float(row["Velocity (m/s)"])],
            "power_w": [float(row["Power (W)"])],
            "beam_diameter_m": [float(row["Beam Diam (m)"])],
            "absorptivity": [float(row["Absorptivity"])],
            "liquidus_temperature_k": [float(row["Liquidus (K)"])],
            "thermal_conductivity_w_mk": [float(row["THCD LT (W/mK)"])],
            "density_kg_m3": [float(row["RT Density (kg/m3)"])],
            "specific_heat_j_kgk": [float(row["Cp, Sheikh (J/kgK)"])],
        }
    )

    beam = BeamParameters(
        beam_diameter=model_df.loc[0, "beam_diameter_m"],
        power=model_df.loc[0, "power_w"],
        velocity=model_df.loc[0, "velocity_m_s"],
        absorptivity=model_df.loc[0, "absorptivity"],
    )
    material = MaterialProperties(
        liquidus_temperature=model_df.loc[0, "liquidus_temperature_k"],
        thermal_conductivity=model_df.loc[0, "thermal_conductivity_w_mk"],
        density=model_df.loc[0, "density_kg_m3"],
        specific_heat=model_df.loc[0, "specific_heat_j_kgk"],
    )

    return row, model_df, beam, material


def volume_to_csv(volume, output_csv):
    xx, yy, zz = np.meshgrid(
        volume.x_range_um,
        volume.y_range_um,
        volume.z_range_um,
        indexing="ij",
    )
    out = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "T_ET": volume.T_xyz.ravel(order="C"),
        }
    )
    out.to_csv(output_csv, index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_row, model_df, beam, material = load_case()

    result = compute_melt_pool(
        model_df,
        domain=DOMAIN,
        workers=1,
        chunk_size=1,
        output_dir=None,
        return_field=True,
    )
    volume = compute_temperature_volume(
        beam=beam,
        material=material,
        domain=DOMAIN,
        workers=1,
        chunk_size=1,
    )

    stem = "alloy0_250W_0.5ms"
    heatmap_path = OUTPUT_DIR / f"temperature_field_{stem}.png"
    meta_path = OUTPUT_DIR / f"metadata_{stem}.csv"
    csv_path = OUTPUT_DIR / f"temperature_volume_{stem}.csv"
    vti_path = OUTPUT_DIR / f"temperature_volume_{stem}.vti"

    result.loc[0, "temperature_field"].plot(output=heatmap_path)
    volume_to_csv(volume, csv_path)
    volume.export_vti(vti_path)

    meta = result.drop(columns=["temperature_field"]).copy()
    meta.insert(0, "alloy", source_row["Alloy"])
    meta.insert(1, "composition", source_row["Composition"])
    meta.to_csv(meta_path, index=False)

    print(result[["melt_length_um", "melt_width_um", "melt_depth_um", "peak_temperature", "min_temperature"]])
    print(f"Wrote {heatmap_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {vti_path}")


if __name__ == "__main__":
    main()
