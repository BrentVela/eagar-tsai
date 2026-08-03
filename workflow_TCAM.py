#!/usr/bin/env python3
"""Regenerate the TCAM-derived figures used by the Beamer presentation.

The TCAM GUI screenshot and the ParaView liquidus export remain manual inputs.
This workflow turns those exported data, together with the ET results, into the
GR maps, projected solidification plots, and ET/TCAM temperature comparison.
"""

import argparse
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from microstructure.GR_Map import DEFAULT_PRIMARY_PHASE

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXCEL = REPO_ROOT / "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2  # Zero-based pandas row index (Excel row 4).
DEFAULT_ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V", "Co", "Cr", "Fe", "Mn", "Ni"]
DEFAULT_TCAM_GR_CSV = (REPO_ROOT / "beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/TCAM_GR_alloy0_250_0.5.csv")
DEFAULT_ET_GR_CSV = (REPO_ROOT / "beamer/figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
DEFAULT_ET_TEMPERATURE_FIELD = (
    REPO_ROOT
    / "beamer/figures/data/BU_ET/0_250W_0.5ms/ET_temperature.vti"
)
DEFAULT_ET_METADATA_CSV = (REPO_ROOT / "beamer/figures/250_0.5/ET_meta_alloy0_250_0.5.csv")
DEFAULT_TCAM_MESH = REPO_ROOT / "beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/result.vtu"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "beamer/figures/TCAM2"

DEFAULT_THERMO_DB = "TCHEA8"
DEFAULT_KINETIC_DB = "MOBHEA3"
DEFAULT_INTERFACIAL_ENERGY_OVERRIDE = None

ALL_STEPS = (
    "base-map",
    "overlay-map",
    "gr-projection",
    "microstructure",
    "temperature",
)


def _require_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing workflow input(s): " + ", ".join(missing))


def _map_kwargs(args):
    return {
        "excel_path": args.excel,
        "row_index": args.row_index,
        "element_cols": args.element_cols,
        "thermodynamic_database": args.thermo_db,
        "kinetic_database": args.kinetic_db,
        "primary_phase": args.primary_phase,
        "interfacial_energy": args.interfacial_energy,
        "show_plot": False,
        "disable_cache": args.disable_tc_cache,
        "disable_output": True,
        "legend_loc": "upper left",
        "show_cet_lines": False,
    }


def _generate_base_map(args):
    from microstructure.GR_Map import request_GR_Grid_from_excel

    output = args.output_dir / "GR_map_alloy0_250_0.5.png"
    request_GR_Grid_from_excel(save_path=output, **_map_kwargs(args))
    return output


def _generate_overlay_map(args):
    from microstructure.GR_Map import request_GR_Grid_from_excel

    output = args.output_dir / "GR_map_overlayET_TCAM_alloy0_250_0.png"
    request_GR_Grid_from_excel(
        save_path=output,
        overlay_csv=[args.tcam_gr_csv, args.et_gr_csv],
        overlay_label=["TCAM liquidus boundary", "ET liquidus boundary"],
        overlay_kwargs=[
            {"c": "lime", "s": 3, "alpha": 0.1},
            {"c": "gold", "s": 4, "alpha": 0.35},
        ],
        **_map_kwargs(args),
    )
    return output


def _generate_gr_projection(args):
    from microstructure.PlotGR import plot_gr_projection

    output = args.output_dir / "TCAM_projected_GR_alloy0_250_0.5.png"
    plot_gr_projection(
        liquidus_csv=args.tcam_gr_csv,
        output_path=output,
        point_size=7,
    )
    return output


def _generate_microstructure(args):
    from microstructure.PlotMicrostructure import plot_projected_liquidus
    from tc_python import CompositionUnit

    output = args.output_dir / "TCAM_projected_microstructure_alloy0_250_0.5.png"
    map_configs = {
        "excel_alloy": {
            "thermo_db": args.thermo_db,
            "kinetic_db": args.kinetic_db,
            "primary_phase": args.primary_phase,
            "interfacial_energy": args.interfacial_energy,
            "composition_unit": CompositionUnit.MOLE_FRACTION,
        }
    }
    plot_projected_liquidus(
        liquidus_csv=args.tcam_gr_csv,
        output_path=output,
        excel_path=args.excel,
        row_index=args.row_index,
        element_cols=args.element_cols,
        map_configs=map_configs,
    )
    return output


def _generate_temperature_comparison(args):
    # PlotTempYZStack currently exposes its data locations as module settings.
    # Set them here so this workflow's command-line paths remain authoritative.
    from Bayesian import PlotTempYZStack

    PlotTempYZStack.ET_TEMPERATURE_FIELD = args.et_temperature_field
    PlotTempYZStack.TCAM_MESH = args.tcam_mesh
    PlotTempYZStack.METADATA_CSV = args.et_metadata_csv
    PlotTempYZStack.OUTPUT_DIR = args.output_dir

    distances = "_".join(f"{value:g}" for value in args.slices)
    output = args.output_dir / f"ET_TC_temp_yz_{distances}um_inferno.png"
    PlotTempYZStack.main(args.slices, output)
    return output


def run_workflow(args):
    selected = set(args.steps or ALL_STEPS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if selected & {"base-map", "overlay-map", "microstructure"}:
        _require_files([args.excel])
    if selected & {"overlay-map", "gr-projection", "microstructure"}:
        _require_files([args.tcam_gr_csv])
    if "overlay-map" in selected:
        _require_files([args.et_gr_csv])
    if "temperature" in selected:
        _require_files(
            [args.et_temperature_field, args.et_metadata_csv, args.tcam_mesh]
        )

    generators = {
        "base-map": _generate_base_map,
        "overlay-map": _generate_overlay_map,
        "gr-projection": _generate_gr_projection,
        "microstructure": _generate_microstructure,
        "temperature": _generate_temperature_comparison,
    }
    outputs = {}
    for step in ALL_STEPS:
        if step not in selected:
            continue
        print(f"Running TCAM figure step: {step}")
        outputs[step] = generators[step](args)
        print(f"Wrote {outputs[step]}")
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate the TCAM-derived presentation figures."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        help="Run only the listed steps (default: all steps).",
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--row-index", type=int, default=DEFAULT_ROW_INDEX)
    parser.add_argument("--element-cols", nargs="+", default=DEFAULT_ELEMENT_COLS)
    parser.add_argument("--tcam-gr-csv", type=Path, default=DEFAULT_TCAM_GR_CSV)
    parser.add_argument("--et-gr-csv", type=Path, default=DEFAULT_ET_GR_CSV)
    parser.add_argument(
        "--et-temperature-field",
        "--et-temperature-csv",
        dest="et_temperature_field",
        type=Path,
        default=DEFAULT_ET_TEMPERATURE_FIELD,
        help=(
            "ET temperature .vti field. Legacy x/y/z/T_ET CSV files are "
            "also accepted; --et-temperature-csv remains as an alias."
        ),
    )
    parser.add_argument(
        "--et-metadata-csv", type=Path, default=DEFAULT_ET_METADATA_CSV
    )
    parser.add_argument("--tcam-mesh", type=Path, default=DEFAULT_TCAM_MESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--slices",
        nargs="+",
        type=float,
        default=[0.0, 80.0, 160.0],
        help="YZ-slice distances behind the laser in micrometers.",
    )
    parser.add_argument("--thermo-db", default=DEFAULT_THERMO_DB)
    parser.add_argument("--kinetic-db", default=DEFAULT_KINETIC_DB)
    parser.add_argument("--primary-phase", default=DEFAULT_PRIMARY_PHASE)
    parser.add_argument(
        "--interfacial-energy",
        type=float,
        default=DEFAULT_INTERFACIAL_ENERGY_OVERRIDE,
        help=(
            "Explicit interfacial energy in J/m^2. Otherwise, use the "
            "'Interfacial Energy (J/m^2)' spreadsheet value when present, "
            "or estimate it with Thermo-Calc at liquidus - 1 K."
        ),
    )
    parser.add_argument("--disable-tc-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generated = run_workflow(parse_args())
    print("TCAM workflow outputs:")
    for name, path in generated.items():
        print(f"  {name}: {path}")
