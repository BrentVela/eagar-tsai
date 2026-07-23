import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eagar_tsai import (
    BeamParameters,
    MaterialProperties,
    SimulationDomain,
    TemperatureField,
    compute_temperature_volume,
)


DEFAULT_INPUT = Path("effective_cp_data.xlsx")
DEFAULT_OUTPUT_DIR = Path("beamer/figures/bayesian")
DEFAULT_ROW_INDEX = 2

COLUMN_ALIASES = {
    "power_w": ["Power (W)"],
    "velocity_m_s": ["Velocity (m/s)"],
    "beam_diameter_m": ["Beam Diam (m)"],
    "absorptivity": ["absorptivity", "Absorptivity"],
    "liquidus_temperature_k": ["Liquidus (K)"],
    "thermal_conductivity_w_mk": ["THCD LT (W/mK)"],
    "density_kg_m3": ["RT Density (kg/m3)"],
    "specific_heat_j_kgk": ["Cp, Sheikh (J/kgK)"],
}


def read_table(path):
    print("Read table...")
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {path.suffix}. Use .xlsx, .xls, or .csv.")


def first_existing_value(row, aliases, name):
    for column in aliases:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])
    raise ValueError(f"Missing required value for {name}. Tried columns: {aliases}")


def row_to_parameters(row):
    print("Getting parameters...")
    values = {
        name: first_existing_value(row, aliases, name)
        for name, aliases in COLUMN_ALIASES.items()
    }
    beam = BeamParameters(
        beam_diameter=values["beam_diameter_m"],
        power=values["power_w"],
        velocity=values["velocity_m_s"],
        absorptivity=values["absorptivity"],
    )
    material = MaterialProperties(
        liquidus_temperature=values["liquidus_temperature_k"],
        thermal_conductivity=values["thermal_conductivity_w_mk"],
        density=values["density_kg_m3"],
        specific_heat=values["specific_heat_j_kgk"],
    )
    return beam, material, values


def make_output_stem(row, row_index, values):
    for column in ("Alloy", "Name", "Composition", "Unnamed: 0"):
        if column in row.index and not pd.isna(row[column]):
            label = str(row[column])
            break
    else:
        label = f"row{row_index}"

    safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    power = f"{values['power_w']:g}"
    velocity = f"{values['velocity_m_s']:g}"
    return f"{safe_label}_{power}W_{velocity}ms"


def volume_to_dataframe(volume, mirror_y, flip_x):
    print("Getting volume...")
    x_um = volume.x_range_um
    y_um = volume.y_range_um
    z_um = volume.z_range_um
    temperature = volume.T_xyz

    if flip_x:
        x_um = -x_um[::-1]
        temperature = temperature[::-1, :, :]

    if mirror_y:
        temperature = np.concatenate([temperature[:, :0:-1, :], temperature], axis=1)
        y_um = np.concatenate([-y_um[:0:-1], y_um])

    xx, yy, zz = np.meshgrid(x_um, y_um, z_um, indexing="ij")
    return pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "T_ET": temperature.ravel(order="C"),
        }
    )


def dense_temperature_field(volume, mirror_y, flip_x):
    print("Getting temperature field...")
    x_range_m = volume.x_range_m
    y_range_m = volume.y_range_m
    z_range_m = volume.z_range_m
    temperature = volume.T_xyz

    if flip_x:
        x_range_m = -x_range_m[::-1]
        temperature = temperature[::-1, :, :]

    if mirror_y:
        temperature = np.concatenate([temperature[:, :0:-1, :], temperature], axis=1)
        y_range_m = np.concatenate([-y_range_m[:0:-1], y_range_m])

    z_surface_idx = int(np.argmin(np.abs(z_range_m)))
    y_center_idx = int(np.argmin(np.abs(y_range_m)))

    return TemperatureField(
        T_xy=temperature[:, :, z_surface_idx].T,
        T_xz=temperature[:, y_center_idx, :].T,
        x_range_m=x_range_m,
        y_range_m=y_range_m,
        z_range_m=z_range_m,
        liquidus_temperature_k=volume.liquidus_temperature_k,
        melt_width_m=volume.result.width,
        melt_depth_m=volume.result.depth,
    )


def metadata_dataframe(values, result):
    return pd.DataFrame(
        [
            {
                **values,
                "melt_length": result.length,
                "melt_width": result.width,
                "melt_depth": result.depth,
                "melt_length_um": result.length * 1.0e6,
                "melt_width_um": result.width * 1.0e6,
                "melt_depth_um": result.depth * 1.0e6,
                "peak_temperature": result.peak_temperature,
                "min_temperature": result.min_temperature,
                "temperature_field": result.temperature_field,
            }
        ]
    )


def warn_if_domain_expanded(volume, domain):
    expected = {
        "x_max_um": domain.x_length_um,
        "y_max_um": domain.y_length_um,
        "z_depth_um": domain.z_depth_um,
    }
    observed = {
        "x_max_um": float(volume.x_range_um[-1]),
        "y_max_um": float(volume.y_range_um[-1]),
        "z_depth_um": abs(float(volume.z_range_um[0])),
    }
    for name, expected_value in expected.items():
        observed_value = observed[name]
        if not np.isclose(observed_value, expected_value):
            print(
                "Warning: eagar_tsai expanded the domain "
                f"{name} from {expected_value:.3f} to {observed_value:.3f} um."
            )


def run(args):
    data = read_table(args.input)
    row = data.iloc[args.row_index]
    beam, material, values = row_to_parameters(row)

    domain = SimulationDomain(
        x_length_um=args.x_length_um,
        y_length_um=args.y_length_um,
        z_depth_um=args.z_depth_um,
        spatial_resolution_um=args.spatial_resolution_um,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or make_output_stem(row, args.row_index, values)

    volume = compute_temperature_volume(
        beam=beam,
        material=material,
        domain=domain,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    result = volume.result
    warn_if_domain_expanded(volume, domain)

    meta_path = output_dir / f"ET_meta_{stem}.csv"
    result_df = metadata_dataframe(values, result)
    result_df.to_csv(meta_path, index=False)
    print("Wrote metadata...")

    heatmap_path = None
    mirror_y = args.full_y
    if not args.no_heatmap:
        heatmap_path = output_dir / f"ET_{stem}.png"
        plot_field = dense_temperature_field(volume, mirror_y=False, flip_x=not args.no_flip_x)
        plot_field.plot(output=heatmap_path)
    print("Wrote heatmap...")

    csv_path = output_dir / f"ET_{stem}.csv"
    volume_to_dataframe(
        volume,
        mirror_y=mirror_y,
        flip_x=not args.no_flip_x,
    ).to_csv(csv_path, index=False)
    print("Wrote 3D csv...")

    vti_path = None
    if args.export_vti:
        vti_path = output_dir / f"ET_{stem}.vti"
        volume.export_vti(vti_path, mirror_y=mirror_y)

    print(f"Wrote {meta_path}")
    if heatmap_path is not None:
        print(f"Wrote {heatmap_path}")
    print(f"Wrote {csv_path}")
    if vti_path is not None:
        print(f"Wrote {vti_path}")
    print(
        "Volume shape: "
        f"nx={volume.x_range_um.size}, ny={volume.y_range_um.size}, nz={volume.z_range_um.size} "
        f"({'mirrored full y' if mirror_y else 'half y'} output)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute an Eagar-Tsai 3D temperature volume from a table row and export x,y,z,Temp CSV."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input .xlsx, .xls, or .csv file.")
    parser.add_argument(
        "--row-index",
        type=int,
        default=DEFAULT_ROW_INDEX,
        help="Zero-based data row index to use; default 2 skips the first two data rows.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=None, help="Optional output filename stem.")
    parser.add_argument("--x-length-um", type=float, default=600.0)
    parser.add_argument("--y-length-um", type=float, default=200.0)
    parser.add_argument("--z-depth-um", type=float, default=250.0)
    parser.add_argument("--spatial-resolution-um", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--full-y", action="store_true", help="Mirror y to export the full symmetric melt pool.")
    parser.add_argument(
        "--no-flip-x",
        action="store_true",
        help="Keep the native Eagar-Tsai x orientation instead of exporting positive-x scan coordinates.",
    )
    parser.add_argument("--export-vti", action="store_true", help="Also export a .vti file.")
    parser.add_argument("--no-heatmap", action="store_true", help="Skip the 2D temperature field heatmap PNG.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
