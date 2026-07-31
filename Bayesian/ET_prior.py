import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
DEFAULT_OUTPUT_DIR = Path("beamer/figures/data/BU_ET")
DEFAULT_START_EXCEL_ROW = 12 #inclusive
DEFAULT_END_EXCEL_ROW = 27 #inclusive

COLUMN_ALIASES = {
    "power_w": ["Power (W)"],
    "velocity_m_s": ["Velocity (m/s)"],
    "beam_diameter_m": ["Beam Diam (m)"],
    "absorptivity": ["absorptivity", "Absorptivity"],
    "liquidus_temperature_k": ["Liquidus (K)"],
    "evaporation_onset_temperature_k": ["Boiling Onset (K)"],
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


def make_manifest_filename(data, row_indices):
    """Build the requested ET_Alloy#_4x4.csv batch-manifest name."""
    if "Alloy" not in data.columns:
        return "ET_AlloyUnknown_4x4.csv"

    alloy_values = data.iloc[row_indices]["Alloy"].dropna().unique()
    if len(alloy_values) != 1:
        return "ET_MixedAlloys_4x4.csv"

    alloy = alloy_values[0]
    if isinstance(alloy, (int, float, np.integer, np.floating)) and float(
        alloy
    ).is_integer():
        alloy = str(int(alloy))
    else:
        alloy = str(alloy).strip()
    safe_alloy = "".join(
        character
        if character.isalnum() or character in "-_."
        else "_"
        for character in alloy
    )
    return f"ET_Alloy{safe_alloy}_4x4.csv"


def oriented_temperature_arrays(volume, mirror_y):
    """Return ET arrays with +x pointing in the laser scanning direction."""
    x_um = -volume.x_range_um[::-1]
    y_um = volume.y_range_um
    z_um = volume.z_range_um
    temperature = volume.T_xyz[::-1, :, :]

    if mirror_y:
        temperature = np.concatenate(
            [temperature[:, :0:-1, :], temperature],
            axis=1,
        )
        y_um = np.concatenate([-y_um[:0:-1], y_um])

    return x_um, y_um, z_um, temperature


def volume_to_dataframe(volume, mirror_y):
    print("Getting volume...")
    x_um, y_um, z_um, temperature = oriented_temperature_arrays(
        volume,
        mirror_y,
    )

    xx, yy, zz = np.meshgrid(x_um, y_um, z_um, indexing="ij")
    return pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "T_ET": temperature.ravel(order="C"),
        }
    )


def dense_temperature_field(volume, mirror_y):
    print("Getting temperature field...")
    x_um, y_um, z_um, temperature = oriented_temperature_arrays(
        volume,
        mirror_y,
    )
    x_range_m = x_um * 1.0e-6
    y_range_m = y_um * 1.0e-6
    z_range_m = z_um * 1.0e-6

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


def export_temperature_vti(volume, output_path, mirror_y):
    """Write a ParaView VTI with +x in the laser scanning direction."""
    import pyvista as pv

    x_um, y_um, z_um, temperature = oriented_temperature_arrays(
        volume,
        mirror_y,
    )
    spacing = tuple(
        float(axis[1] - axis[0]) if axis.size > 1 else 1.0
        for axis in (x_um, y_um, z_um)
    )
    grid = pv.ImageData(
        dimensions=temperature.shape,
        spacing=spacing,
        origin=(float(x_um[0]), float(y_um[0]), float(z_um[0])),
    )
    grid.point_data["Temperature_K"] = np.ascontiguousarray(
        temperature
    ).ravel(order="F")
    grid.field_data["scan_direction_x_sign"] = np.array([1], dtype=np.int8)
    grid.save(output_path)


def plot_temperature_field(temperature_field, output_path):
    """Render ET temperature panels with the workflow_ET inferno styling."""
    figure = temperature_field.plot(output=None)
    for axis in figure.axes:
        if not axis.images:
            continue
        for image in axis.images:
            image.set_cmap("inferno")
        for contour in axis.collections:
            contour.set_facecolor("none")
            contour.set_edgecolor("cyan")
            contour.set_linewidth(1.25)

    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


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


def select_row_indices(data, row_index, start_excel_row, end_excel_row):
    """
    Return zero-based dataframe indices for one row or an Excel-row range.

    Excel row 1 is the header, so Excel data row 2 maps to dataframe index 0.
    END_EXCEL_ROW is inclusive. The legacy row_index option takes precedence.
    """
    if row_index is not None:
        if row_index < 0 or row_index >= len(data):
            raise IndexError(
                f"row_index {row_index} is outside the available range "
                f"0 through {len(data) - 1}."
            )
        return [row_index]

    if start_excel_row < 2:
        raise ValueError("start_excel_row must be at least 2.")
    if end_excel_row is not None and end_excel_row < start_excel_row:
        raise ValueError(
            f"end_excel_row ({end_excel_row}) must be greater than or equal "
            f"to start_excel_row ({start_excel_row})."
        )

    final_excel_row = len(data) + 1
    effective_end = (
        final_excel_row
        if end_excel_row is None
        else min(end_excel_row, final_excel_row)
    )
    indices = list(range(start_excel_row - 2, effective_end - 1))
    if not indices:
        raise ValueError(
            f"Excel row range {start_excel_row} through "
            f"{end_excel_row} selected no data rows."
        )
    return indices


def run_case(args, data, row_index, output_dir, stem_override=None):
    """Calculate and write one self-contained ET case directory."""
    row = data.iloc[row_index]
    beam, material, values = row_to_parameters(row)
    source_excel_row = row_index + 2

    domain = SimulationDomain(
        x_length_um=args.x_length_um,
        y_length_um=args.y_length_um,
        z_depth_um=args.z_depth_um,
        spatial_resolution_um=args.spatial_resolution_um,
    )

    stem = stem_override or make_output_stem(row, row_index, values)
    case_dir = output_dir / stem
    case_dir.mkdir(parents=True, exist_ok=True)

    volume = compute_temperature_volume(
        beam=beam,
        material=material,
        domain=domain,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    result = volume.result
    warn_if_domain_expanded(volume, domain)

    heatmap_path = None
    mirror_y = args.full_y
    if not args.no_heatmap:
        heatmap_path = case_dir / "ET_heatmap.png"
        plot_field = dense_temperature_field(volume, mirror_y=False)
        plot_temperature_field(plot_field, heatmap_path)
        print("Wrote heatmap...")

    vti_path = case_dir / "ET_temperature.vti"
    export_temperature_vti(volume, vti_path, mirror_y=mirror_y)
    print("Wrote VTI...")

    csv_path = None
    if args.export_csv:
        csv_path = case_dir / "ET_temperature.csv"
        volume_to_dataframe(
            volume,
            mirror_y=mirror_y,
        ).to_csv(csv_path, index=False)
        print("Wrote optional 3D CSV...")

    meta_path = case_dir / "metadata.csv"
    result_df = metadata_dataframe(values, result)
    result_df.insert(0, "case_id", stem)
    result_df.insert(1, "source_excel_row", source_excel_row)
    if "Alloy" in row.index:
        result_df.insert(2, "alloy", row["Alloy"])
    if "Composition" in row.index:
        result_df.insert(3, "composition", row["Composition"])
    result_df["case_directory"] = str(case_dir)
    result_df["temperature_csv"] = (
        str(csv_path) if csv_path is not None else None
    )
    result_df["vti_file"] = str(vti_path)
    result_df["heatmap_file"] = (
        str(heatmap_path) if heatmap_path is not None else None
    )
    result_df["metadata_csv"] = str(meta_path)
    result_df.to_csv(meta_path, index=False)
    print("Wrote metadata...")

    print(f"Wrote {meta_path}")
    if heatmap_path is not None:
        print(f"Wrote {heatmap_path}")
    print(f"Wrote {vti_path}")
    if csv_path is not None:
        print(f"Wrote {csv_path}")
    print(
        "Volume shape: "
        f"nx={volume.x_range_um.size}, ny={volume.y_range_um.size}, nz={volume.z_range_um.size} "
        f"({'mirrored full y' if mirror_y else 'half y'} output)"
    )
    return result_df


def run(args):
    data = read_table(args.input)
    row_indices = select_row_indices(
        data,
        row_index=args.row_index,
        start_excel_row=args.start_excel_row,
        end_excel_row=args.end_excel_row,
    )
    if args.stem is not None and len(row_indices) != 1:
        raise ValueError("--stem can only be used when selecting one row.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / make_manifest_filename(data, row_indices)
    metadata_rows = []
    for case_number, row_index in enumerate(row_indices, start=1):
        print(
            f"\nET case {case_number}/{len(row_indices)}: "
            f"Excel row {row_index + 2}"
        )
        metadata_rows.append(
            run_case(
                args,
                data,
                row_index,
                output_dir,
                stem_override=args.stem,
            )
        )
        pd.concat(metadata_rows, ignore_index=True).to_csv(
            manifest_path,
            index=False,
        )

    manifest = pd.concat(metadata_rows, ignore_index=True)
    print(f"\nWrote batch manifest: {manifest_path}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Eagar-Tsai 3D temperature volumes for an inclusive "
            "Excel-row range and write self-contained case directories. "
            "Output coordinates use +x as the laser scanning direction."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input .xlsx, .xls, or .csv file.")
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help=(
            "Legacy zero-based data row index for a single case. When set, "
            "this overrides --start-excel-row and --end-excel-row."
        ),
    )
    parser.add_argument(
        "--start-excel-row",
        type=int,
        default=DEFAULT_START_EXCEL_ROW,
        help=(
            "First Excel row to process, including the header as row 1 "
            f"(default: {DEFAULT_START_EXCEL_ROW})."
        ),
    )
    parser.add_argument(
        "--end-excel-row",
        type=int,
        default=DEFAULT_END_EXCEL_ROW,
        help=(
            "Last Excel row to process, inclusive "
            f"(default: {DEFAULT_END_EXCEL_ROW})."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=None, help="Optional output filename stem.")
    parser.add_argument("--x-length-um", type=float, default=600.0)
    parser.add_argument("--y-length-um", type=float, default=200.0)
    parser.add_argument("--z-depth-um", type=float, default=250.0)
    parser.add_argument("--spatial-resolution-um", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--full-y", action="store_true", help="Mirror y to export the full symmetric melt pool.")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export the large x/y/z/T_ET CSV (VTI is always written).",
    )
    parser.add_argument(
        "--export-vti",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-heatmap", action="store_true", help="Skip the 2D temperature field heatmap PNG.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
