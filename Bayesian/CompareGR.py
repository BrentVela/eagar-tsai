#!/usr/bin/env python3
"""Compare TCAM and GPR-corrected G/R fields on mirrored YZ projections."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


TCAM_CSV = Path("beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/TCAM_GR_alloy0_250_0.5.csv")
CORRECTED_CSV = Path("beamer/figures/bayesian/warped_cartesian_r3d_matern32_gpr/heldout_250W_0.5ms_15trainingcases/heldout_GR_Data.csv")
OUTPUT_DIR = Path("beamer/figures/bayesian/warped_cartesian_r3d_matern32_gpr/heldout_250W_0.5ms_15trainingcases")
G_OUTPUT_NAME = "G_projection_TCAM_vs_corrected.png"
R_OUTPUT_NAME = "R_projection_TCAM_vs_corrected.png"

POINT_SIZE = 10
MICRON_SCALE_THRESHOLD = 1.0e-2
GRADIENT_K_PER_UM_THRESHOLD = 1.0e4

ET_COLUMNS = {
    "x": "x",
    "y": "y",
    "z": "z",
    "G": "G",
    "R": "R",
}
PARAVIEW_COLUMNS = {
    "Points_0": "x",
    "Points_1": "y",
    "Points_2": "z",
    "Gradient_Magnitude": "G",
    "R (m/s)": "R",
}


def _load_projection_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if set(ET_COLUMNS).issubset(df.columns):
        columns = ET_COLUMNS
    elif set(PARAVIEW_COLUMNS).issubset(df.columns):
        columns = PARAVIEW_COLUMNS
    else:
        et_missing = sorted(set(ET_COLUMNS).difference(df.columns))
        paraview_missing = sorted(set(PARAVIEW_COLUMNS).difference(df.columns))
        raise ValueError(
            f"{csv_path} does not match a supported G/R CSV format. "
            f"Missing ET columns: {et_missing}; "
            f"missing ParaView columns: {paraview_missing}"
        )

    points = df.loc[:, list(columns)].rename(columns=columns).copy()
    for column in ("x", "y", "z", "G", "R"):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points = points.replace([np.inf, -np.inf], np.nan).dropna()
    points = points[(points["G"] > 0.0) & (points["R"] >= 0.0)].copy()
    if points.empty:
        raise ValueError(f"No finite positive-G projection points found in {csv_path}.")

    max_abs_coordinate = points[["x", "y", "z"]].abs().to_numpy().max()
    if max_abs_coordinate < MICRON_SCALE_THRESHOLD:
        points.loc[:, ["x", "y", "z"]] *= 1.0e6
        points.attrs["input_coordinate_units"] = "m"
    else:
        points.attrs["input_coordinate_units"] = "um"

    positive_gradient = points.loc[points["G"] > 0.0, "G"].to_numpy()
    if np.median(positive_gradient) < GRADIENT_K_PER_UM_THRESHOLD:
        points.loc[:, "G"] *= 1.0e6
        points.attrs["input_gradient_units"] = "K/um"
    else:
        points.attrs["input_gradient_units"] = "K/m"

    points = points.sort_values("x").reset_index(drop=True)
    points["y_plot"] = points["y"].abs().round(0)
    points["z_plot"] = points["z"].round(0)
    return points


def _shared_normalization(
    tcam: pd.DataFrame,
    corrected: pd.DataFrame,
    field: str,
) -> Normalize:
    values = np.concatenate(
        [
            tcam[field].to_numpy(dtype=float),
            corrected[field].to_numpy(dtype=float),
        ]
    )
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    if value_min == value_max:
        padding = max(abs(value_min) * 0.01, 1.0)
        value_min -= padding
        value_max += padding
    return Normalize(vmin=value_min, vmax=value_max)


def _plot_comparison(
    tcam: pd.DataFrame,
    corrected: pd.DataFrame,
    field: str,
    output_path: Path,
    point_size: float,
) -> Path:
    plot_settings = {
        "G": {"cmap": "BuPu", "label": "G (K/m)"},
        "R": {"cmap": "YlOrRd", "label": "R (m/s)"},
    }
    settings = plot_settings[field]
    norm = _shared_normalization(tcam, corrected, field)

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    tcam_scatter = axis.scatter(
        -tcam["y_plot"],
        tcam["z_plot"],
        c=tcam[field],
        cmap=settings["cmap"],
        norm=norm,
        marker="o",
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    corrected_scatter = axis.scatter(
        corrected["y_plot"],
        corrected["z_plot"],
        c=corrected[field],
        cmap=settings["cmap"],
        norm=norm,
        marker="o",
        s=point_size,
        linewidths=0,
        rasterized=True,
    )

    max_y = max(float(tcam["y_plot"].max()), float(corrected["y_plot"].max()))
    z_min = min(float(tcam["z_plot"].min()), float(corrected["z_plot"].min()))
    z_max = max(float(tcam["z_plot"].max()), float(corrected["z_plot"].max()))
    axis.set_xlim(-max_y, max_y)
    axis.set_ylim(z_min, z_max)
    tcam_scatter.set_clip_path(
        Rectangle(
            (-max_y, z_min),
            max_y,
            z_max - z_min,
            transform=axis.transData,
        )
    )
    corrected_scatter.set_clip_path(
        Rectangle(
            (0.0, z_min),
            max_y,
            z_max - z_min,
            transform=axis.transData,
        )
    )
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

    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size=0.18, pad=0.15)
    colorbar = figure.colorbar(corrected_scatter, cax=colorbar_axis)
    colorbar.set_label(settings["label"], fontsize=11)
    colorbar.ax.tick_params(labelsize=9)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    colorbar.formatter = formatter
    colorbar.update_ticks()
    if field == "G":
        colorbar.ax.yaxis.get_offset_text().set_x(1.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")
    return output_path


def compare_gr_projections(
    tcam_csv: Path = TCAM_CSV,
    corrected_csv: Path = CORRECTED_CSV,
    output_dir: Path = OUTPUT_DIR,
    point_size: float = POINT_SIZE,
) -> tuple[Path, Path]:
    tcam_csv = Path(tcam_csv)
    corrected_csv = Path(corrected_csv)
    output_dir = Path(output_dir)

    tcam = _load_projection_points(tcam_csv)
    corrected = _load_projection_points(corrected_csv)

    print(
        f"TCAM: {len(tcam)} points; coordinates read as "
        f"{tcam.attrs['input_coordinate_units']}; G read as "
        f"{tcam.attrs['input_gradient_units']}."
    )
    print(
        f"Corrected: {len(corrected)} points; coordinates read as "
        f"{corrected.attrs['input_coordinate_units']}; G read as "
        f"{corrected.attrs['input_gradient_units']}."
    )

    g_output = _plot_comparison(
        tcam,
        corrected,
        "G",
        output_dir / G_OUTPUT_NAME,
        point_size,
    )
    r_output = _plot_comparison(
        tcam,
        corrected,
        "R",
        output_dir / R_OUTPUT_NAME,
        point_size,
    )
    return g_output, r_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create mirrored YZ comparisons of TCAM and GPR-corrected "
            "thermal gradient and solidification rate."
        )
    )
    parser.add_argument("--tcam-csv", type=Path, default=TCAM_CSV)
    parser.add_argument("--corrected-csv", type=Path, default=CORRECTED_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--point-size", type=float, default=POINT_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        arguments = parse_args()
        compare_gr_projections(
            tcam_csv=arguments.tcam_csv,
            corrected_csv=arguments.corrected_csv,
            output_dir=arguments.output_dir,
            point_size=arguments.point_size,
        )
    except Exception as exc:
        print(f"CompareGR failed: {exc}", file=sys.stderr)
        raise
