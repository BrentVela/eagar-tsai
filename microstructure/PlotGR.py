import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


LIQUIDUS_CSV = (
    "CalcFiles/Test17/TCAM_keyholing_alloy0_250_0.5/ParaView_fine0.15mm/subdivide4_fine0.15_data.csv"
)
OUTPUT_PATH = (
    "CalcFiles/Test17/TCAM_keyholing_alloy0_250_0.5/ParaView_fine0.15mm/GR_projection_fine0.15_subdivide4.png"
)
POINT_SIZE = 5

ET_COLUMNS = {"x", "y", "z", "G", "R"}
PARAVIEW_COLUMNS = {
    "Points_0": "x",
    "Points_1": "y",
    "Points_2": "z",
    "Gradient_Magnitude": "G",
    "R (m/s)": "R",
}


def _load_liquidus_points(csv_path):
    df = pd.read_csv(csv_path)

    if ET_COLUMNS.issubset(df.columns):
        df = df.loc[:, ["x", "y", "z", "G", "R"]].copy()
    elif set(PARAVIEW_COLUMNS).issubset(df.columns):
        df = df.rename(columns=PARAVIEW_COLUMNS)
        df = df.loc[:, ["x", "y", "z", "G", "R"]].copy()
        df.loc[:, ["x", "y", "z"]] *= 1.0e6
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

    return df.sort_values("x").reset_index(drop=True)


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
    projected["y_plot"] = projected["y"].round(0)
    projected["z_plot"] = projected["z"].round(0)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    r_scatter = ax.scatter(
        -projected["y_plot"],
        projected["z_plot"],
        c=projected["R"],
        cmap="YlOrRd",
        marker="o",
        s=point_size,
        linewidths=0,
    )
    g_scatter = ax.scatter(
        projected["y_plot"],
        projected["z_plot"],
        c=projected["G"],
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
    return output_path, projected


if __name__ == "__main__":
    try:
        plot_gr_projection()
    except Exception as exc:
        print(f"PlotGR failed: {exc}", file=sys.stderr)
        raise
