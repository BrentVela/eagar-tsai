import argparse
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
from ETtoVTI import export_eagar_tsai_vti
from microstructure.GRFrom3D import compute_gr_from_vti
from microstructure.GR_Map import request_GR_Grid_from_excel
from microstructure.PlotGR import plot_gr_projection
from microstructure.PlotMicrostructure import plot_projected_liquidus
from tc_python import CompositionUnit


DEFAULT_EXCEL = "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2 # EXCEL ROW - 2
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
DEFAULT_CALC_ROOT = "beamer-template/figures2"

DEFAULT_THERMO_DB = "TCHEA8"
DEFAULT_KINETIC_DB = "MOBHEA3"
DEFAULT_PRIMARY_PHASE = "BCC_B2"
DEFAULT_INTERFACIAL_ENERGY = 0.5

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
    "velocity_m_s": ["velocity_m_s", "Velocity_m/s", "Velocity (m/s)", "v"],
    "power_w": ["power_w", "Power", "Power (W)", "P"],
    "beam_diameter_m": ["beam_diameter_m", "Beam_diameter_m", "Beam Diam (m)"],
    "absorptivity": ["absorptivity", "Absorptivity"],
    "liquidus_temperature_k": ["liquidus_temperature_k", "Liquidus (K)", "TL", "PROP LT (K)"],
    "thermal_conductivity_w_mk": [
        "thermal_conductivity_w_mk",
        "THCD LT (W/mK)",
        "K_LT",
        "PROP LT THCD (W/(mK))",
        "EQ LT THCD (W/mK)",
    ],
    "density_kg_m3": ["density_kg_m3", "RT Density (kg/m3)", "PROP RT Density (kg/m3)"],
    "specific_heat_j_kgk": [
        "specific_heat_j_kgk",
        "Cp, Sheikh (J/kgK)",
        "Cp, LT (J/kgK)",
        "Cp_LT",
        "PROP LT C (J/(kg K))",
    ],
}


def _format_number(value):
    return f"{float(value):g}"


def _first_existing(row, candidates, output_name):
    for col in candidates:
        if col in row.index and not pd.isna(row[col]):
            return float(row[col])
    raise ValueError(f"Excel row is missing a value for {output_name}. Tried: {candidates}")


def _load_process_inputs(excel_path, row_index):
    df = pd.read_excel(excel_path)
    row = df.iloc[row_index]
    values = {
        output_name: _first_existing(row, candidates, output_name)
        for output_name, candidates in INPUT_COLUMNS.items()
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


def run_workflow(args):
    process_df, source_row = _load_process_inputs(args.excel_path, args.row_index)
    output_dir, stem = _default_output_dir(args.calc_root, source_row, args.row_index, process_df)
    if args.output_dir:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    et_csv = output_dir / f"ET_{stem}.csv"
    vti_path = output_dir / f"ET_3D_temperature_{stem}.vti"
    meta_path = output_dir / f"ET_meta_{stem}.csv"
    liquidus_csv = output_dir / f"liquidus_GR_{stem}.csv"
    gr_map_png = output_dir / f"GR_map_overlayET_{stem}.png"
    projected_png = output_dir / f"projected_microstructure_{stem}.png"
    gr_projected_png = output_dir / f"projected_GR_{stem}.png"
    temperature_png = output_dir / f"temperature_field_{stem}.png"
    temperature_3d_png = output_dir / f"ET_3D_temperature_{stem}.png"

    print("Running Eagar-Tsai melt pool calculation...")
    result = compute_melt_pool(
        process_df,
        domain=DEFAULT_DOMAIN,
        workers=args.workers,
        chunk_size=args.chunk_size,
        output_dir=None,
        return_field=True,
    )
    result.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path}")

    temperature_field = result.loc[0, "temperature_field"]
    if temperature_field is not None:
        temperature_field.plot(output=temperature_png)
        print(f"Wrote {temperature_png}")

    print("Writing matched ET-prior-style 3D temperature CSV...")
    volume = _compute_temperature_volume(
        process_df=process_df,
        domain=DEFAULT_DOMAIN,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    _temperature_volume_to_dataframe(volume).to_csv(et_csv, index=False)
    print(f"Wrote {et_csv}")

    vti_limits_um = _expanded_vti_limits(result, args.vti_padding_um)
    print(
        "Using VTI limits: "
        f"x={vti_limits_um['x']} um, y={vti_limits_um['y']} um, z={vti_limits_um['z']} um"
    )

    print("Exporting 3D temperature VTI...")
    export_eagar_tsai_vti(
        output_vti=vti_path,
        output_png=temperature_3d_png,
        power_w=float(process_df.loc[0, "power_w"]),
        scan_speed_m_s=float(process_df.loc[0, "velocity_m_s"]),
        beam_diameter_m=float(process_df.loc[0, "beam_diameter_m"]),
        absorptivity=float(process_df.loc[0, "absorptivity"]),
        liquidus_temperature_k=float(process_df.loc[0, "liquidus_temperature_k"]),
        thermal_conductivity_w_mk=float(process_df.loc[0, "thermal_conductivity_w_mk"]),
        density_kg_m3=float(process_df.loc[0, "density_kg_m3"]),
        heat_capacity_j_kgk=float(process_df.loc[0, "specific_heat_j_kgk"]),
        x_limits_um=vti_limits_um["x"],
        y_limits_um=vti_limits_um["y"],
        z_limits_um=vti_limits_um["z"],
        spatial_res_um=args.vti_resolution_um,
        workers=args.workers,
        chunk_size=args.chunk_size,
        mirror_y=False,
    )
    print(f"Wrote {vti_path}")
    print(f"Wrote {temperature_3d_png}")

    print("Extracting liquidus G/R from VTI...")
    _, _, _, liquidus_df = compute_gr_from_vti(
        vti_path=vti_path,
        meta_path=meta_path,
        output_path=liquidus_csv,
    )
    _warn_if_liquidus_touches_vti_boundary(liquidus_df, vti_limits_um)

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
        save_path=gr_map_png,
        overlay_csv=liquidus_csv,
        overlay_label="3D liquidus boundary",
        legend_loc=args.gr_legend_loc,
    )

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
        liquidus_csv=liquidus_csv,
        output_path=projected_png,
        excel_path=args.excel_path,
        row_index=args.row_index,
        element_cols=args.element_cols,
        map_configs=projection_map_configs,
    )

    print("Generating projected G/R heatmap cross-section...")
    plot_gr_projection(
        liquidus_csv=liquidus_csv,
        output_path=gr_projected_png,
    )

    return {
        "output_dir": output_dir,
        "et_csv": et_csv,
        "meta_csv": meta_path,
        "vti": vti_path,
        "temperature_3d_png": temperature_3d_png,
        "liquidus_csv": liquidus_csv,
        "gr_map_png": gr_map_png,
        "projected_png": projected_png,
        "gr_projected_png": gr_projected_png,
        "temperature_png": temperature_png,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Eagar-Tsai to GR-map workflow for one Excel row.")
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
    print("Workflow outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
