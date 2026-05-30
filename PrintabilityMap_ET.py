from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from eagar_tsai import (
    MaterialProperties,
    PrintabilityParameters,
    SimulationDomain,
    compute_printability_map,
)


INPUT_XLSX = "et_input_data_example.xlsx"
OUTPUT_DIR = "CalcFiles/Printability_ET"

DEFECT_ORDER = ["lack_of_fusion", "defect_free", "balling", "keyhole"]
DEFECT_COLORS = {
    "lack_of_fusion": "lightpink",
    "defect_free": "white",
    "balling": "turquoise",
    "keyhole": "skyblue",
}
DEFECT_LABELS = {
    "lack_of_fusion": "Lack of Fusion",
    "defect_free": "Defect-Free",
    "balling": "Balling",
    "keyhole": "Keyholing",
}


def material_from_excel(excel_path: str | Path, row_index: int) -> MaterialProperties:
    row = pd.read_excel(excel_path).iloc[row_index]
    return MaterialProperties(
        liquidus_temperature=float(row["PROP LT (K)"]),
        thermal_conductivity=float(row["PROP LT THCD (W/(mK))"]),
        density=float(row["PROP RT Density (kg/m3)"]),
        specific_heat=float(row["PROP LT C (J/(kg K))"]),
    )


def process_from_excel(
    excel_path: str | Path,
    row_index: int,
    layer_thickness_um: float,
    hatch_spacing_um: float,
) -> PrintabilityParameters:
    row = pd.read_excel(excel_path).iloc[row_index]
    return PrintabilityParameters(
        beam_diameter_m=float(row["Beam Diam (m)"]),
        absorptivity=float(row["Absorptivity"]),
        layer_thickness_m=layer_thickness_um * 1.0e-6,
        hatch_spacing_m=hatch_spacing_um * 1.0e-6,
    )


def add_geometry_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["W/D"] = df["melt_width_um"] / df["melt_depth_um"]
    df["L/W"] = df["melt_length_um"] / df["melt_width_um"]
    return df.replace([np.inf, -np.inf], np.nan)


def plot_printability_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    code_by_defect = {defect: idx for idx, defect in enumerate(DEFECT_ORDER)}
    plot_df = df.copy()
    plot_df["defect_code"] = plot_df["defect"].map(code_by_defect)

    grid = plot_df.pivot(
        index="power_w",
        columns="velocity_m_s",
        values="defect_code",
    ).sort_index().sort_index(axis=1)

    powers = grid.index.to_numpy(dtype=float)
    velocities = grid.columns.to_numpy(dtype=float)

    cmap = ListedColormap([DEFECT_COLORS[name] for name in DEFECT_ORDER])
    norm = BoundaryNorm(np.arange(len(DEFECT_ORDER) + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.imshow(
        grid.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        extent=[velocities.min(), velocities.max(), powers.min(), powers.max()],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    handles = [
        Patch(facecolor=DEFECT_COLORS[name], edgecolor="black", label=DEFECT_LABELS[name])
        for name in DEFECT_ORDER
        if name in set(plot_df["defect"])
    ]
    ax.legend(handles=handles, loc="best", frameon=True)
    ax.set_xlabel("Scan speed, v (m/s)")
    ax.set_ylabel("Laser power, P (W)")
    ax.set_title("Eagar-Tsai Printability Map")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a printability map with the eagar_tsai library."
    )
    parser.add_argument("--input", default=INPUT_XLSX, help="Excel file with material/process inputs.")
    parser.add_argument("--row", type=int, default=0, help="Zero-based Excel data row to use.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for CSV and PNG outputs.")
    parser.add_argument("--power-min", type=float, default=50.0, help="Minimum laser power in W.")
    parser.add_argument("--power-max", type=float, default=400.0, help="Maximum laser power in W.")
    parser.add_argument("--velocity-min", type=float, default=0.025, help="Minimum scan speed in m/s.")
    parser.add_argument("--velocity-max", type=float, default=2.0, help="Maximum scan speed in m/s.")
    parser.add_argument("--n-power", type=int, default=71, help="Number of power grid points.")
    parser.add_argument("--n-velocity", type=int, default=80, help="Number of velocity grid points.")
    parser.add_argument("--layer-thickness-um", type=float, default=30.0, help="Layer thickness in um.")
    parser.add_argument("--hatch-spacing-um", type=float, default=80.0, help="Hatch spacing in um.")
    parser.add_argument("--keyhole-wdr-threshold", type=float, default=2.5, help="Keyhole W/D threshold.")
    parser.add_argument("--workers", type=int, default=12, help="Worker processes. Use -1 for all cores.")
    parser.add_argument("--domain-x-um", type=float, default=1200.0, help="Eagar-Tsai x-domain length in um.")
    parser.add_argument("--domain-y-um", type=float, default=1200.0, help="Eagar-Tsai y half-domain in um.")
    parser.add_argument("--domain-z-um", type=float, default=1000.0, help="Eagar-Tsai z-domain depth in um.")
    parser.add_argument("--resolution-um", type=float, default=5.0, help="Spatial resolution in um.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    material = material_from_excel(args.input, args.row)
    process = process_from_excel(
        args.input,
        args.row,
        layer_thickness_um=args.layer_thickness_um,
        hatch_spacing_um=args.hatch_spacing_um,
    )
    domain = SimulationDomain(
        x_length_um=args.domain_x_um,
        y_length_um=args.domain_y_um,
        z_depth_um=args.domain_z_um,
        spatial_resolution_um=args.resolution_um,
    )

    result = compute_printability_map(
        process,
        material,
        power_range=(args.power_min, args.power_max),
        velocity_range=(args.velocity_min, args.velocity_max),
        n_power=args.n_power,
        n_velocity=args.n_velocity,
        keyhole_wdr_threshold=args.keyhole_wdr_threshold,
        domain=domain,
        workers=args.workers,
    )
    result = add_geometry_ratios(result)

    stem = f"printability_map_row{args.row}"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"

    result.to_csv(csv_path, index=False)
    plot_printability_dataframe(result, png_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(result["defect"].value_counts().to_string())


if __name__ == "__main__":
    main()
