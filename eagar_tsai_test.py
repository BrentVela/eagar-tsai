import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd

from eagar_tsai import SimulationDomain, compute_melt_pool


INPUT_XLSX = "effective_cp_data.xlsx"
ROW_INDEX = 2
OUTPUT_DIR = Path("beamer-template/figures/250_0.5_2")
OUTPUT_STEM = "alloy0_250_0.5"

# Keep this aligned with workflow.py's DEFAULT_DOMAIN.
DOMAIN = SimulationDomain(
    x_length_um=600.0,
    y_length_um=200.0,
    z_depth_um=250.0,
    spatial_resolution_um=2.0,
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

    return row, model_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_row, model_df = load_case()

    result = compute_melt_pool(
        model_df,
        domain=DOMAIN,
        workers=1,
        chunk_size=1,
        output_dir=None,
        return_field=True,
    )
    heatmap_path = OUTPUT_DIR / f"temperature_field_{OUTPUT_STEM}.png"
    meta_path = OUTPUT_DIR / f"ET_meta_{OUTPUT_STEM}.csv"

    result.loc[0, "temperature_field"].plot(output=heatmap_path)

    meta = result.drop(columns=["temperature_field"]).copy()
    meta.insert(0, "alloy", source_row["Alloy"])
    meta.insert(1, "composition", source_row["Composition"])
    meta.to_csv(meta_path, index=False)

    print(result[["melt_length_um", "melt_width_um", "melt_depth_um", "peak_temperature", "min_temperature"]])
    print(f"Wrote {heatmap_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
