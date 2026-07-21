import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


LIQUIDUS_CSV = ("beamer/figures2/250_0.5/liquidus_GR_alloy0_250_0.5.csv")
OUTPUT_PATH = ("beamer/figures2/250_0.5/projected_GR_alloy0_250_0.5.png")
INTERPOLATE_PARAVIEW = False
INTERPOLATION_Y_POINTS = 75
INTERPOLATION_Z_POINTS = 115
SMOOTHING_SIGMA = 0.0
POINT_SIZE = 10
MICRON_SCALE_THRESHOLD = 1.0e-2

ET_COLUMNS = {"x", "y", "z", "G", "R"}
PARAVIEW_COLUMNS = {
    "Points_0": "x",
    "Points_1": "y",
    "Points_2": "z",
    "Gradient_Magnitude": "G",
    "R (m/s)": "R",
}


def _coordinates_to_microns(df):
    coord_cols = ["x", "y", "z"]
    max_abs_coord = df[coord_cols].abs().to_numpy().max()
    if max_abs_coord < MICRON_SCALE_THRESHOLD:
        df.loc[:, coord_cols] *= 1.0e6
        df.attrs["coordinate_units"] = "m"
    else:
        df.attrs["coordinate_units"] = "um"


def _load_liquidus_points(csv_path):
    df = pd.read_csv(csv_path)

    if ET_COLUMNS.issubset(df.columns):
        is_paraview = False
        df = df.loc[:, ["x", "y", "z", "G", "R"]].copy()
    elif set(PARAVIEW_COLUMNS).issubset(df.columns):
        is_paraview = True
        df = df.rename(columns=PARAVIEW_COLUMNS)
        df = df.loc[:, ["x", "y", "z", "G", "R"]].copy()
        _coordinates_to_microns(df)
    else:
        et_missing = sorted(ET_COLUMNS.difference(df.columns))
        paraview_missing = sorted(set(PARAVIEW_COLUMNS).difference(df.columns))
        raise ValueError(
            f"{csv_path} does not match a supported liquidus CSV format. "
            f"Missing ET columns: {et_missing}; "
            f"missing ParaView columns: {paraview_missing}"
        )

    df = df.dropna(subset=["x", "y", "z", "G", "R"]).copy()
    if df.empty:
        raise ValueError(f"No liquidus points found in {csv_path}.")

    df = df.sort_values("x").reset_index(drop=True)
    df.attrs["is_paraview"] = is_paraview
    return df


def _interpolate_projected_points(projected):
    points = projected.groupby(["y", "z"], as_index=False)[["G", "R"]].mean()
    y = np.linspace(points["y"].min(), points["y"].max(), INTERPOLATION_Y_POINTS)
    z = np.linspace(points["z"].min(), points["z"].max(), INTERPOLATION_Z_POINTS)
    yy, zz = np.meshgrid(y, z)

    grid_points = (points["y"], points["z"])
    g_grid = griddata(grid_points, points["G"], (yy, zz), method="linear")
    r_grid = griddata(grid_points, points["R"], (yy, zz), method="linear")
    valid = np.isfinite(g_grid) & np.isfinite(r_grid)
    g_grid = gaussian_filter(np.where(valid, g_grid, 0.0), sigma=SMOOTHING_SIGMA)
    r_grid = gaussian_filter(np.where(valid, r_grid, 0.0), sigma=SMOOTHING_SIGMA)
    weights = gaussian_filter(valid.astype(float), sigma=SMOOTHING_SIGMA)
    g_grid = np.divide(g_grid, weights, where=weights > 0.0)
    r_grid = np.divide(r_grid, weights, where=weights > 0.0)

    return pd.DataFrame(
        {
            "y_plot": yy[valid],
            "z_plot": zz[valid],
            "G": g_grid[valid],
            "R": r_grid[valid],
        }
    )


def plot_gr_projection(
    liquidus_csv=LIQUIDUS_CSV,
    output_path=OUTPUT_PATH,
    point_size=POINT_SIZE,
    label_fontsize=12,
    tick_fontsize=10,
    colorbar_label_fontsize=11,
    colorbar_tick_fontsize=9,
    colorbar_size=0.18,
    top_colorbar_pad=0.15,
    bottom_colorbar_pad=0.55,
):
    projected = _load_liquidus_points(liquidus_csv)
    if INTERPOLATE_PARAVIEW and projected.attrs["is_paraview"]:
        plot_points = _interpolate_projected_points(projected)
    else:
        plot_points = projected.copy()
        plot_points["y_plot"] = plot_points["y"].round(0)
        plot_points["z_plot"] = plot_points["z"].round(0)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    r_scatter = ax.scatter(
        -plot_points["y_plot"],
        plot_points["z_plot"],
        c=plot_points["R"],
        cmap="YlOrRd",
        marker="o",
        s=point_size,
        linewidths=0,
    )
    g_scatter = ax.scatter(
        plot_points["y_plot"],
        plot_points["z_plot"],
        c=plot_points["G"],
        cmap="BuPu",
        marker="o",
        s=point_size,
        linewidths=0,
    )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    r_scatter.set_clip_path(
        Rectangle(
            (x_min, y_min),
            -x_min,
            y_max - y_min,
            transform=ax.transData,
        )
    )
    g_scatter.set_clip_path(
        Rectangle(
            (0, y_min),
            x_max,
            y_max - y_min,
            transform=ax.transData,
        )
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y (um)", fontsize=label_fontsize)
    ax.set_ylabel("z (um)", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    divider = make_axes_locatable(ax)

    top_cax = divider.append_axes("top", size=colorbar_size, pad=top_colorbar_pad)
    top_colorbar = fig.colorbar(r_scatter, cax=top_cax, orientation="horizontal")
    top_colorbar.ax.xaxis.set_ticks_position("top")
    top_colorbar.ax.xaxis.set_label_position("top")
    top_colorbar.set_label("R (m/s)", fontsize=colorbar_label_fontsize)
    top_colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    bottom_cax = divider.append_axes("bottom", size=colorbar_size, pad=bottom_colorbar_pad)
    bottom_colorbar = fig.colorbar(g_scatter, cax=bottom_cax, orientation="horizontal")
    bottom_colorbar.set_label("G (K/m)", fontsize=colorbar_label_fontsize)
    bottom_colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path}")
    print(
        "Projected {count} liquidus points from x = {xmin:.3f} to {xmax:.3f} um.".format(
            count=len(projected),
            xmin=projected["x"].min(),
            xmax=projected["x"].max(),
        )
    )
    print(f"Plotted {len(plot_points)} points per half.")
    return output_path, projected


if __name__ == "__main__":
    try:
        plot_gr_projection()
    except Exception as exc:
        print(f"PlotGR failed: {exc}", file=sys.stderr)
        raise
