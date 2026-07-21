#!/usr/bin/env python3
"""Regenerate ET data products and presentation figures for one alloy row.

Run without ``--steps`` for the full ET-to-microstructure pipeline, or select
individual steps to reuse intermediate files already present in the output
directory.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eagar_tsai import (
    BeamParameters,
    MaterialProperties,
    SimulationDomain,
    compute_melt_pool,
    compute_temperature_volume,
)
from ETtoVTI import export_eagar_tsai_vti
DEFAULT_EXCEL = "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2 # EXCEL ROW - 2
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
DEFAULT_CALC_ROOT = "beamer/figures"

DEFAULT_THERMO_DB = "TCHEA8"
DEFAULT_KINETIC_DB = "MOBHEA3"
DEFAULT_PRIMARY_PHASE = "BCC_B2"
DEFAULT_INTERFACIAL_ENERGY = 0.5

ALL_STEPS = (
    "temperature-field",
    "temperature-volume",
    "temperature-3d",
    "liquidus",
    "gr-map",
    "gr-projection",
    "microstructure",
)

DEFAULT_DOMAIN = SimulationDomain(
    x_length_um=600.0,
    y_length_um=200.0,
    z_depth_um=250.0,
    spatial_resolution_um=2.0,
)
DEFAULT_VTI_LIMITS_UM = {
    "x": (-160.0, 380.0),
    "y": (0.0, 200.0),
    "z": (-60.0, 0.0),
}

INPUT_COLUMNS = {
    "velocity_m_s": "Velocity (m/s)",
    "power_w": "Power (W)",
    "beam_diameter_m": "Beam Diam (m)",
    "absorptivity": "Absorptivity",
    "liquidus_temperature_k": "Liquidus (K)",
    "thermal_conductivity_w_mk": "THCD LT (W/mK)",
    "density_kg_m3": "RT Density (kg/m3)",
    "specific_heat_j_kgk": "Cp, Sheikh (J/kgK)",
}


def _format_number(value):
    return f"{float(value):g}"


def _load_process_inputs(excel_path, row_index):
    df = pd.read_excel(excel_path)
    missing_columns = [column for column in INPUT_COLUMNS.values() if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{excel_path} is missing required column(s): {', '.join(missing_columns)}"
        )

    row = df.iloc[row_index]
    missing_values = [column for column in INPUT_COLUMNS.values() if pd.isna(row[column])]
    if missing_values:
        raise ValueError(
            f"Excel row {row_index} is missing value(s) for: {', '.join(missing_values)}"
        )

    values = {
        output_name: float(row[column])
        for output_name, column in INPUT_COLUMNS.items()
    }
    return pd.DataFrame([values]), row


def _alloy_id(row, row_index):
    if "Alloy" in row.index and not pd.isna(row["Alloy"]):
        return _format_number(row["Alloy"])
    if "Unnamed: 0" in row.index and not pd.isna(row["Unnamed: 0"]):
        return _format_number(row["Unnamed: 0"])
    return str(row_index)


def _default_output_dir(calc_root, row, row_index, process_df):
    alloy_id = _alloy_id(row, row_index)
    power = _format_number(process_df.loc[0, "power_w"])
    velocity = _format_number(process_df.loc[0, "velocity_m_s"])
    condition_dir = f"{power}_{velocity}"
    root = Path(calc_root)
    output_dir = root if root.name == condition_dir else root / condition_dir
    return output_dir, f"alloy{alloy_id}_{power}_{velocity}"


def _expanded_vti_limits(result, padding_um):
    row = result.iloc[0]
    melt_length_um = float(row["melt_length_um"])
    melt_width_um = float(row["melt_width_um"])
    melt_depth_um = float(row["melt_depth_um"])

    x_min, x_max = DEFAULT_VTI_LIMITS_UM["x"]
    y_min, y_max = DEFAULT_VTI_LIMITS_UM["y"]
    z_min, z_max = DEFAULT_VTI_LIMITS_UM["z"]

    return {
        "x": (x_min, max(x_max, melt_length_um + padding_um)),
        "y": (y_min, max(y_max, 0.5 * melt_width_um + padding_um)),
        "z": (min(z_min, -(melt_depth_um + padding_um)), z_max),
    }


def _temperature_volume_to_dataframe(volume):
    x_um = volume.x_range_um
    y_um = volume.y_range_um
    z_um = volume.z_range_um
    temperature = volume.T_xyz

    xx, yy, zz = np.meshgrid(x_um, y_um, z_um, indexing="ij")
    return pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "T_ET": temperature.ravel(order="C"),
        }
    )


def _compute_temperature_volume(process_df, domain, workers, chunk_size):
    row = process_df.iloc[0]
    beam = BeamParameters(
        beam_diameter=float(row["beam_diameter_m"]),
        power=float(row["power_w"]),
        velocity=float(row["velocity_m_s"]),
        absorptivity=float(row["absorptivity"]),
    )
    material = MaterialProperties(
        liquidus_temperature=float(row["liquidus_temperature_k"]),
        thermal_conductivity=float(row["thermal_conductivity_w_mk"]),
        density=float(row["density_kg_m3"]),
        specific_heat=float(row["specific_heat_j_kgk"]),
    )
    return compute_temperature_volume(
        beam=beam,
        material=material,
        domain=domain,
        workers=workers,
        chunk_size=chunk_size,
    )


def _plot_et_temperature_field(temperature_field, output_path):
    """Render the library temperature panels with workflow-specific styling."""
    figure = temperature_field.plot(output=None)
    for axis in figure.axes:
        # The two temperature panels contain AxesImage objects. Colorbar axes
        # do not, so this leaves their internal artists alone while their
        # linked mappables update automatically.
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


def _warn_if_liquidus_touches_vti_boundary(liquidus_df, vti_limits_um):
    boundary_checks = {
        "x_min": ("x", vti_limits_um["x"][0], liquidus_df["x"].min()),
        "x_max": ("x", vti_limits_um["x"][1], liquidus_df["x"].max()),
        "y_min": ("y", vti_limits_um["y"][0], liquidus_df["y"].min()),
        "y_max": ("y", vti_limits_um["y"][1], liquidus_df["y"].max()),
        "z_min": ("z", vti_limits_um["z"][0], liquidus_df["z"].min()),
        "z_max": ("z", vti_limits_um["z"][1], liquidus_df["z"].max()),
    }
    for name, (_, limit, observed) in boundary_checks.items():
        if abs(float(observed) - float(limit)) <= 1.0e-6:
            print(
                f"Warning: liquidus contour touches VTI {name} boundary "
                f"at {observed:.3f} um. Consider increasing --vti-padding-um."
            )


def _require_files(paths, step, prerequisite_steps):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        prerequisites = " ".join(prerequisite_steps)
        raise FileNotFoundError(
            f"Step '{step}' is missing input(s): {', '.join(missing)}. "
            f"Generate them first with --steps {prerequisites}, or include "
            "those prerequisite steps in the same run."
        )


def _workflow_paths(output_dir, stem):
    return {
        "et_csv": output_dir / f"ET_{stem}.csv",
        "vti": output_dir / f"ET_3D_temperature_{stem}.vti",
        "meta_csv": output_dir / f"ET_meta_{stem}.csv",
        "liquidus_csv": output_dir / f"liquidus_GR_{stem}.csv",
        "gr_map_png": output_dir / f"GR_map_overlayET_{stem}.png",
        "microstructure_png": output_dir / f"projected_microstructure_{stem}.png",
        "gr_projection_png": output_dir / f"projected_GR_{stem}.png",
        "temperature_png": output_dir / f"temperature_field_{stem}.png",
        "temperature_3d_png": output_dir / f"ET_3D_temperature_{stem}.png",
    }


def _generate_temperature_field(args, process_df, paths):
    print("Running Eagar-Tsai melt pool calculation...")
    result = compute_melt_pool(
        process_df,
        domain=DEFAULT_DOMAIN,
        workers=args.workers,
        chunk_size=args.chunk_size,
        output_dir=None,
        return_field=True,
    )
    result.to_csv(paths["meta_csv"], index=False)
    print(f"Wrote {paths['meta_csv']}")

    temperature_field = result.loc[0, "temperature_field"]
    if temperature_field is None:
        raise ValueError("The Eagar-Tsai calculation did not return a temperature field.")
    _plot_et_temperature_field(temperature_field, paths["temperature_png"])
    print(f"Wrote {paths['temperature_png']}")
    return result


def _generate_temperature_volume(args, process_df, paths):
    print("Writing matched ET-prior-style 3D temperature CSV...")
    volume = _compute_temperature_volume(
        process_df=process_df,
        domain=DEFAULT_DOMAIN,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    _temperature_volume_to_dataframe(volume).to_csv(paths["et_csv"], index=False)
    print(f"Wrote {paths['et_csv']}")


def _load_metadata(paths, step, prerequisite_steps):
    _require_files([paths["meta_csv"]], step, prerequisite_steps)
    return pd.read_csv(paths["meta_csv"])


def _generate_temperature_3d(args, process_df, paths, metadata=None):
    if metadata is None:
        metadata = _load_metadata(paths, "temperature-3d", ["temperature-field"])
    vti_limits_um = _expanded_vti_limits(metadata, args.vti_padding_um)
    print(
        "Using VTI limits: "
        f"x={vti_limits_um['x']} um, y={vti_limits_um['y']} um, "
        f"z={vti_limits_um['z']} um"
    )

    row = process_df.iloc[0]
    print("Exporting 3D temperature VTI and PNG...")
    export_eagar_tsai_vti(
        output_vti=paths["vti"],
        output_png=paths["temperature_3d_png"],
        power_w=float(row["power_w"]),
        scan_speed_m_s=float(row["velocity_m_s"]),
        beam_diameter_m=float(row["beam_diameter_m"]),
        absorptivity=float(row["absorptivity"]),
        liquidus_temperature_k=float(row["liquidus_temperature_k"]),
        thermal_conductivity_w_mk=float(row["thermal_conductivity_w_mk"]),
        density_kg_m3=float(row["density_kg_m3"]),
        heat_capacity_j_kgk=float(row["specific_heat_j_kgk"]),
        x_limits_um=vti_limits_um["x"],
        y_limits_um=vti_limits_um["y"],
        z_limits_um=vti_limits_um["z"],
        spatial_res_um=args.vti_resolution_um,
        workers=args.workers,
        chunk_size=args.chunk_size,
        mirror_y=False,
    )
    print(f"Wrote {paths['vti']}")
    print(f"Wrote {paths['temperature_3d_png']}")


def _generate_liquidus(args, paths, metadata=None):
    from microstructure.GRFrom3D import compute_gr_from_vti

    _require_files(
        [paths["vti"], paths["meta_csv"]],
        "liquidus",
        ["temperature-field", "temperature-3d"],
    )
    if metadata is None:
        metadata = pd.read_csv(paths["meta_csv"])
    vti_limits_um = _expanded_vti_limits(metadata, args.vti_padding_um)

    print("Extracting liquidus G/R from VTI...")
    _, _, _, liquidus_df = compute_gr_from_vti(
        vti_path=paths["vti"],
        meta_path=paths["meta_csv"],
        output_path=paths["liquidus_csv"],
    )
    _warn_if_liquidus_touches_vti_boundary(liquidus_df, vti_limits_um)
    print(f"Wrote {paths['liquidus_csv']}")


def _generate_gr_map(args, paths):
    from microstructure.GR_Map import request_GR_Grid_from_excel

    _require_files([paths["liquidus_csv"]], "gr-map", ["liquidus"])
    print("Generating GR map overlay...")
    request_GR_Grid_from_excel(
        excel_path=args.excel_path,
        row_index=args.row_index,
        element_cols=args.element_cols,
        thermodynamic_database=args.thermo_db,
        kinetic_database=args.kinetic_db,
        primary_phase=args.primary_phase,
        interfacial_energy=args.interfacial_energy,
        show_plot=False,
        disable_cache=args.disable_tc_cache,
        disable_output=True,
        save_path=paths["gr_map_png"],
        overlay_csv=paths["liquidus_csv"],
        overlay_label="3D liquidus boundary",
        legend_loc=args.gr_legend_loc,
    )
    print(f"Wrote {paths['gr_map_png']}")


def _generate_gr_projection(paths):
    from microstructure.PlotGR import plot_gr_projection

    _require_files([paths["liquidus_csv"]], "gr-projection", ["liquidus"])
    print("Generating projected G/R heatmap cross-section...")
    plot_gr_projection(
        liquidus_csv=paths["liquidus_csv"],
        output_path=paths["gr_projection_png"],
    )
    print(f"Wrote {paths['gr_projection_png']}")


def _generate_microstructure(args, paths):
    from microstructure.PlotMicrostructure import plot_projected_liquidus
    from tc_python import CompositionUnit

    _require_files([paths["liquidus_csv"]], "microstructure", ["liquidus"])
    print("Generating projected weld cross-section...")
    projection_map_configs = {
        "excel_alloy": {
            "thermo_db": args.thermo_db,
            "kinetic_db": args.kinetic_db,
            "primary_phase": args.primary_phase,
            "interfacial_energy": args.interfacial_energy,
            "composition_unit": CompositionUnit.MOLE_FRACTION,
        }
    }
    plot_projected_liquidus(
        liquidus_csv=paths["liquidus_csv"],
        output_path=paths["microstructure_png"],
        excel_path=args.excel_path,
        row_index=args.row_index,
        element_cols=args.element_cols,
        map_configs=projection_map_configs,
    )
    print(f"Wrote {paths['microstructure_png']}")


def run_workflow(args):
    process_df, source_row = _load_process_inputs(args.excel_path, args.row_index)
    output_dir, stem = _default_output_dir(args.calc_root, source_row, args.row_index, process_df)
    if args.output_dir:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _workflow_paths(output_dir, stem)
    selected = set(args.steps or ALL_STEPS)
    generated = {"output_dir": output_dir}
    metadata = None

    for step in ALL_STEPS:
        if step not in selected:
            continue
        print(f"Running ET workflow step: {step}")
        if step == "temperature-field":
            metadata = _generate_temperature_field(args, process_df, paths)
            generated["meta_csv"] = paths["meta_csv"]
            generated["temperature_png"] = paths["temperature_png"]
        elif step == "temperature-volume":
            _generate_temperature_volume(args, process_df, paths)
            generated["et_csv"] = paths["et_csv"]
        elif step == "temperature-3d":
            _generate_temperature_3d(args, process_df, paths, metadata)
            generated["vti"] = paths["vti"]
            generated["temperature_3d_png"] = paths["temperature_3d_png"]
        elif step == "liquidus":
            _generate_liquidus(args, paths, metadata)
            generated["liquidus_csv"] = paths["liquidus_csv"]
        elif step == "gr-map":
            _generate_gr_map(args, paths)
            generated["gr_map_png"] = paths["gr_map_png"]
        elif step == "gr-projection":
            _generate_gr_projection(paths)
            generated["gr_projection_png"] = paths["gr_projection_png"]
        elif step == "microstructure":
            _generate_microstructure(args, paths)
            generated["microstructure_png"] = paths["microstructure_png"]

    return generated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate ET data products and presentation figures for one Excel row."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        help="Run only the listed steps (default: all steps).",
    )
    parser.add_argument("--excel", dest="excel_path", default=DEFAULT_EXCEL)
    parser.add_argument("--row-index", type=int, default=DEFAULT_ROW_INDEX, help="Zero-based pandas row index.")
    parser.add_argument("--element-cols", nargs="+", default=DEFAULT_ELEMENT_COLS)
    parser.add_argument("--calc-root", default=DEFAULT_CALC_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--vti-resolution-um", type=float, default=1.0)
    parser.add_argument("--vti-padding-um", type=float, default=20.0)
    parser.add_argument("--thermo-db", default=DEFAULT_THERMO_DB)
    parser.add_argument("--kinetic-db", default=DEFAULT_KINETIC_DB)
    parser.add_argument("--primary-phase", default=DEFAULT_PRIMARY_PHASE)
    parser.add_argument("--interfacial-energy", type=float, default=DEFAULT_INTERFACIAL_ENERGY)
    parser.add_argument("--disable-tc-cache", action="store_true")
    parser.add_argument("--gr-legend-loc", default="upper left")
    return parser.parse_args()


if __name__ == "__main__":
    outputs = run_workflow(parse_args())
    print("ET workflow outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
