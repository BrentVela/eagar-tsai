#!/usr/bin/env python3
"""Regenerate Bayesian-update products and presentation figures.

The temperature-field GPR and Python plots are automated here. Liquidus G/R
extraction from the corrected VTI and ParaView screenshots remain manual steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MATERIAL_DATA = REPO_ROOT / "effective_cp_data.xlsx"
ET_LIQUIDUS_CSV = (REPO_ROOT / "beamer/figures/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
TCAM_GR_CSV = (REPO_ROOT / "beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/TCAM_GR_alloy0_250_0.5.csv")
TCAM_MESH = (REPO_ROOT / "beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/result.vtu")
ET_METADATA = (REPO_ROOT / "beamer/figures/data/BU_ET/0_250W_0.5ms/metadata.csv")

SINGLE_DIR = REPO_ROOT / "beamer/figures/bayesian/single_training_case"
SINGLE_GR_CSV = SINGLE_DIR / "BU_GR_data.csv"

WARPED_DIR = (REPO_ROOT / "beamer/figures/bayesian/multi_warped_gpr/heldout_250W_0.5ms_15cases")
WARPED_GR_CSV = WARPED_DIR / "heldout_GR_data.csv"
WARPED_PREDICTION_VTI = WARPED_DIR / "heldout_prediction_on_coarse_et_grid.vti"

STEP_ORDER = (
    "single-gpr",
    "single-gr-projection",
    "single-microstructure",
    "single-overlay-map",
    "warped-gpr",
    "warped-gr-projection",
    "warped-microstructure",
    "warped-overlay-map",
    "warped-temperature",
    "warped-animation",
)


def _require_files(paths: list[Path], step: str, note: str | None = None) -> None:
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    message = f"{step} is missing required input(s): " + ", ".join(
        str(path.relative_to(REPO_ROOT)) for path in missing
    )
    if note:
        message += f". {note}"
    raise FileNotFoundError(message)


def _run_script(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=REPO_ROOT,
        check=True,
    )


def _overlay_map(
    corrected_csv: Path,
    output_dir: Path,
    *,
    corrected_on_top: bool,
) -> None:
    _require_files(
        [MATERIAL_DATA, TCAM_GR_CSV, corrected_csv, ET_LIQUIDUS_CSV],
        "Bayesian G/R overlay map",
        "The corrected G/R CSV must first be exported manually from ParaView.",
    )
    from microstructure.GR_Map import request_GR_Grid_from_excel

    request_GR_Grid_from_excel(
        excel_path=MATERIAL_DATA,
        row_index=2,
        element_cols=[
            "W", "Re", "Nb", "Ta", "Mo", "Hf",
            "V", "Co", "Cr", "Fe", "Mn", "Ni",
        ],
        thermodynamic_database="TCHEA8",
        kinetic_database="MOBHEA3",
        primary_phase="BCC_B2",
        interfacial_energy=None,
        show_plot=False,
        disable_cache=False,
        disable_output=True,
        save_path=output_dir / "GR_map_overlayET_TCAM_BU_alloy0_250_0.png",
        overlay_csv=[TCAM_GR_CSV, corrected_csv, ET_LIQUIDUS_CSV],
        overlay_label=[
            "TCAM liquidus boundary",
            "Corrected liquidus points",
            "ET liquidus boundary",
        ],
        overlay_kwargs=[
            {"c": "lime", "s": 3, "alpha": 0.1, "zorder": 5},
            {
                "c": "deepskyblue",
                "s": 3,
                "alpha": 0.1,
                "zorder": 8 if corrected_on_top else 6,
            },
            {"c": "gold", "s": 4, "alpha": 0.35, "zorder": 7},
        ],
        legend_loc="upper left",
        show_cet_lines=False,
    )


def _warped_temperature(slice_spacing_um: float) -> None:
    _require_files(
        [WARPED_PREDICTION_VTI, ET_METADATA, TCAM_MESH],
        "warped-temperature",
    )
    from Bayesian import PlotTempYZStack

    PlotTempYZStack.ET_TEMPERATURE_FIELD = WARPED_PREDICTION_VTI
    PlotTempYZStack.ET_TEMPERATURE_ARRAY = "T_pred"
    PlotTempYZStack.ET_SCAN_DIRECTION_X_SIGN = 1
    PlotTempYZStack.LEFT_PANEL_LABEL = "Bayesian-Updated ET"
    PlotTempYZStack.TCAM_MESH = TCAM_MESH
    PlotTempYZStack.METADATA_CSV = ET_METADATA
    PlotTempYZStack.OUTPUT_DIR = WARPED_DIR
    PlotTempYZStack.main(
        (0.0, 75.0, 150.0),
        WARPED_DIR / "Corrected_TCAM_temp_yz_0_75_150um_inferno.png",
        et_slice_spacing_um=slice_spacing_um,
    )


def run_step(step: str, *, warped_temperature_spacing_um: float = 2.0) -> None:
    print(f"Running Bayesian figure step: {step}")
    if step == "single-gpr":
        _run_script("Bayesian/GPR.py")
    elif step == "single-gr-projection":
        _require_files(
            [TCAM_GR_CSV, SINGLE_GR_CSV],
            step,
            "Create BU_GR_data.csv from BU_corrected_data.csv in ParaView.",
        )
        from Bayesian.CompareGR import compare_gr_projections

        compare_gr_projections(TCAM_GR_CSV, SINGLE_GR_CSV, SINGLE_DIR)
    elif step == "single-microstructure":
        _require_files(
            [TCAM_GR_CSV, SINGLE_GR_CSV],
            step,
            "Create BU_GR_data.csv from BU_corrected_data.csv in ParaView.",
        )
        from Bayesian.CompareMicrostructure import compare_microstructures

        compare_microstructures(
            TCAM_GR_CSV,
            SINGLE_GR_CSV,
            SINGLE_DIR / "microstructure_TCAM_vs_corrected.png",
            excel_path=MATERIAL_DATA,
            row_index=2,
        )
    elif step == "single-overlay-map":
        # TCAM at the bottom, corrected in the middle, and ET on top.
        _overlay_map(SINGLE_GR_CSV, SINGLE_DIR, corrected_on_top=False)
    elif step == "warped-gpr":
        _run_script(
            "-u",
            "Bayesian/GPR_multi_warped.py",
            "--radial-feature", "r_3d",
            "--kernel", "matern52",
            "--n-restarts-optimizer", "1",
        )
    elif step == "warped-gr-projection":
        _require_files(
            [TCAM_GR_CSV, WARPED_GR_CSV],
            step,
            "Create heldout_GR_data.csv from the prediction VTI in ParaView.",
        )
        from Bayesian.CompareGR import compare_gr_projections

        compare_gr_projections(TCAM_GR_CSV, WARPED_GR_CSV, WARPED_DIR)
    elif step == "warped-microstructure":
        _require_files(
            [TCAM_GR_CSV, WARPED_GR_CSV],
            step,
            "Create heldout_GR_data.csv from the prediction VTI in ParaView.",
        )
        from Bayesian.CompareMicrostructure import compare_microstructures

        compare_microstructures(
            TCAM_GR_CSV,
            WARPED_GR_CSV,
            WARPED_DIR / "microstructure_TCAM_vs_corrected.png",
            excel_path=MATERIAL_DATA,
            row_index=2,
        )
    elif step == "warped-overlay-map":
        # Corrected points stay visible above both ET and TCAM.
        _overlay_map(WARPED_GR_CSV, WARPED_DIR, corrected_on_top=True)
    elif step == "warped-temperature":
        _warped_temperature(warped_temperature_spacing_um)
    elif step == "warped-animation":
        _run_script("Bayesian/animate_warped_et.py")
    else:
        raise ValueError(f"Unknown step: {step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate single-case and warped-GPR presentation products. "
            "ParaView G/R exports and screenshots remain manual inputs."
        )
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_ORDER,
        help="Run only the listed steps (default: all steps in workflow order).",
    )
    parser.add_argument(
        "--warped-temperature-spacing-um",
        type=float,
        default=2.0,
        help=(
            "In-memory interpolation spacing for the warped-temperature YZ "
            "figure only (default: 2 um)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.steps or STEP_ORDER)
    if args.warped_temperature_spacing_um <= 0.0:
        raise ValueError("Warped-temperature spacing must be positive.")
    for step in STEP_ORDER:
        if step in selected:
            run_step(
                step,
                warped_temperature_spacing_um=(
                    args.warped_temperature_spacing_um
                ),
            )


if __name__ == "__main__":
    main()
