#!/usr/bin/env python3
"""Compare TCAM and GPR-corrected microstructures on a mirrored YZ projection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microstructure.PlotMicrostructure import (
    ELEMENT_COLS,
    EQUIAXED_COLORBAR_MIN_FRACTION,
    EXCEL_PATH,
    MAP_CONFIGS,
    MAP_SOURCE,
    ROW_INDEX,
    _calculate_cet_grid,
    _classify_from_gr_map,
    _equiaxed_colormap,
    _load_projected_liquidus,
    _positive_r_projection_points,
)


TCAM_CSV = Path("beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/TCAM_GR_alloy0_250_0.5.csv")
CORRECTED_CSV = Path("beamer/figures/bayesian/warped_cartesian_r3d_gpr/heldout_250W_0.5ms_15trainingcases/heldout_GR_coarse_et.csv")
OUTPUT_PATH = Path("beamer/figures/bayesian/warped_cartesian_r3d_gpr/heldout_250W_0.5ms_15trainingcases/microstructure_TCAM_vs_corrected.png")

POINT_SIZE = 10
PLANAR_COLOR = "#b00000"

def _load_and_classify(
    csv_path: Path,
    r_grid: np.ndarray,
    g_grid: np.ndarray,
    equiaxed_fraction: np.ndarray,
) -> pd.DataFrame:
    liquidus, _ = _load_projected_liquidus(csv_path)
    projected = _positive_r_projection_points(liquidus)
    projected["equiaxed_fraction"] = _classify_from_gr_map(
        projected,
        r_grid,
        g_grid,
        equiaxed_fraction,
    )
    projected["y_plot"] = projected["y"].abs().round(0)
    projected["z_plot"] = projected["z"].round(0)
    return projected


def _plot_classified_points(
    axis,
    points: pd.DataFrame,
    side: float,
    point_size: float,
    equiaxed_cmap,
    equiaxed_norm,
) -> list:
    fraction = points["equiaxed_fraction"].to_numpy(dtype=float)
    planar = fraction == -1.0
    columnar = fraction == 0.0
    equiaxed = ~(planar | columnar)
    y = side * points["y_plot"].to_numpy(dtype=float)
    z = points["z_plot"].to_numpy(dtype=float)

    return [
        axis.scatter(
            y[columnar],
            z[columnar],
            color=equiaxed_cmap(equiaxed_norm(0.0)),
            marker="o",
            s=point_size,
            linewidths=0,
            rasterized=True,
        ),
        axis.scatter(
            y[equiaxed],
            z[equiaxed],
            c=fraction[equiaxed],
            cmap=equiaxed_cmap,
            norm=equiaxed_norm,
            marker="o",
            s=point_size,
            linewidths=0,
            rasterized=True,
        ),
        axis.scatter(
            y[planar],
            z[planar],
            color=PLANAR_COLOR,
            marker="o",
            s=point_size,
            linewidths=0,
            rasterized=True,
        ),
    ]


def _classification_summary(name: str, points: pd.DataFrame) -> None:
    fraction = points["equiaxed_fraction"].to_numpy(dtype=float)
    planar_count = int(np.count_nonzero(fraction == -1.0))
    columnar_count = int(np.count_nonzero(fraction == 0.0))
    equiaxed_count = len(fraction) - planar_count - columnar_count
    print(
        f"{name}: {len(points)} points; planar={planar_count}, "
        f"columnar={columnar_count}, equiaxed={equiaxed_count}."
    )


def compare_microstructures(
    tcam_csv: Path = TCAM_CSV,
    corrected_csv: Path = CORRECTED_CSV,
    output_path: Path = OUTPUT_PATH,
    point_size: float = POINT_SIZE,
    excel_path: Path = Path(EXCEL_PATH),
    row_index: int = ROW_INDEX,
) -> Path:
    tcam_csv = Path(tcam_csv)
    corrected_csv = Path(corrected_csv)
    output_path = Path(output_path)
    excel_path = Path(excel_path)

    print("Calculating shared CET classification map...")
    r_grid, g_grid, equiaxed_fraction, interfacial_energy = _calculate_cet_grid(
        excel_path=excel_path,
        row_index=row_index,
        element_cols=ELEMENT_COLS,
        map_source=MAP_SOURCE,
        map_configs=MAP_CONFIGS,
    )

    tcam = _load_and_classify(
        tcam_csv,
        r_grid,
        g_grid,
        equiaxed_fraction,
    )
    corrected = _load_and_classify(
        corrected_csv,
        r_grid,
        g_grid,
        equiaxed_fraction,
    )
    _classification_summary("TCAM", tcam)
    _classification_summary("Corrected", corrected)

    equiaxed_cmap = _equiaxed_colormap()
    equiaxed_norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    tcam_scatters = _plot_classified_points(
        axis,
        tcam,
        side=-1.0,
        point_size=point_size,
        equiaxed_cmap=equiaxed_cmap,
        equiaxed_norm=equiaxed_norm,
    )
    corrected_scatters = _plot_classified_points(
        axis,
        corrected,
        side=1.0,
        point_size=point_size,
        equiaxed_cmap=equiaxed_cmap,
        equiaxed_norm=equiaxed_norm,
    )

    max_y = max(float(tcam["y_plot"].max()), float(corrected["y_plot"].max()))
    z_min = min(float(tcam["z_plot"].min()), float(corrected["z_plot"].min()))
    z_max = max(float(tcam["z_plot"].max()), float(corrected["z_plot"].max()))
    axis.set_xlim(-max_y, max_y)
    axis.set_ylim(z_min, z_max)
    tcam_clip = Rectangle(
        (-max_y, z_min),
        max_y,
        z_max - z_min,
        transform=axis.transData,
    )
    corrected_clip = Rectangle(
        (0.0, z_min),
        max_y,
        z_max - z_min,
        transform=axis.transData,
    )
    for scatter in tcam_scatters:
        scatter.set_clip_path(tcam_clip)
    for scatter in corrected_scatters:
        scatter.set_clip_path(corrected_clip)
    axis.axvline(0.0, color="0.35", linewidth=0.8, zorder=3)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("y (um)", fontsize=12)
    axis.set_ylabel("z (um)", fontsize=12)
    axis.tick_params(axis="both", labelsize=10)
    axis.text(
        0.25,
        1.02,
        "TCAM",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    axis.text(
        0.75,
        1.02,
        "Bayesian-Updated ET",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    combined_fraction = np.concatenate(
        [
            tcam["equiaxed_fraction"].to_numpy(dtype=float),
            corrected["equiaxed_fraction"].to_numpy(dtype=float),
        ]
    )
    if np.any(combined_fraction >= EQUIAXED_COLORBAR_MIN_FRACTION):
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size=0.18, pad=0.15)
        equiaxed_mapper = cm.ScalarMappable(
            norm=equiaxed_norm,
            cmap=equiaxed_cmap,
        )
        equiaxed_mapper.set_array([])
        colorbar = figure.colorbar(equiaxed_mapper, cax=colorbar_axis)
        colorbar.set_label("Equiaxed fraction", fontsize=11)
        colorbar.ax.tick_params(labelsize=9)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=equiaxed_cmap(equiaxed_norm(0.0)),
            markeredgewidth=0,
            markersize=5,
            label="Columnar",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PLANAR_COLOR,
            markeredgewidth=0,
            markersize=5,
            label="Planar",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        borderpad=0.3,
        handletextpad=0.4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Interfacial energy used: {interfacial_energy:.6g} J/m^2."
    )
    print(f"Wrote {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify and compare TCAM and GPR-corrected microstructure "
            "on a mirrored YZ projection."
        )
    )
    parser.add_argument("--tcam-csv", type=Path, default=TCAM_CSV)
    parser.add_argument("--corrected-csv", type=Path, default=CORRECTED_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--point-size", type=float, default=POINT_SIZE)
    parser.add_argument("--excel", type=Path, default=Path(EXCEL_PATH))
    parser.add_argument("--row-index", type=int, default=ROW_INDEX)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        arguments = parse_args()
        compare_microstructures(
            tcam_csv=arguments.tcam_csv,
            corrected_csv=arguments.corrected_csv,
            output_path=arguments.output,
            point_size=arguments.point_size,
            excel_path=arguments.excel,
            row_index=arguments.row_index,
        )
    except Exception as exc:
        print(f"CompareMicrostructure failed: {exc}", file=sys.stderr)
        raise
