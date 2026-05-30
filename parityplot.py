import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def parity_plot(
    df,
    x_col,
    y_col,
    color_col=None,
    shape_col=None,
    title="Parity Plot",
    x_label=None,
    y_label=None,
    units="µm",
    color_map=None,
    marker_map=None,
    save_path=None,
):
    """
    Make a customizable parity plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Data table containing TCAM and ET results.
    x_col : str
        Column name for reference values, usually TCAM.
    y_col : str
        Column name for predicted values, usually ET.
    color_col : str, optional
        Column used to color points, e.g. "Alloy" or "Classification".
    shape_col : str, optional
        Column used to choose marker shape, e.g. "Classification" or "Cp_method".
    title : str
        Plot title.
    x_label : str, optional
        Custom x-axis label.
    y_label : str, optional
        Custom y-axis label.
    units : str
        Units shown in axis labels.
    color_map : dict, optional
        Dictionary mapping categories to colors.
    marker_map : dict, optional
        Dictionary mapping categories to marker styles.
    save_path : str, optional
        If provided, saves the figure to this path.
    """

    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    # Default axis labels
    if x_label is None:
        x_label = f"{x_col} [{units}]"
    if y_label is None:
        y_label = f"{y_col} [{units}]"

    # Default color map
    if color_col is not None:
        color_categories = sorted(plot_df[color_col].dropna().unique())
        if color_map is None:
            default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color_map = {
                cat: default_colors[i % len(default_colors)]
                for i, cat in enumerate(color_categories)
            }
    else:
        color_map = {None: "black"}

    # Default marker map
    if shape_col is not None:
        shape_categories = sorted(plot_df[shape_col].dropna().unique())
        if marker_map is None:
            default_markers = ["o", "s", "^", "v", "D", "X", "P", "*"]
            marker_map = {
                cat: default_markers[i % len(default_markers)]
                for i, cat in enumerate(shape_categories)
            }
    else:
        marker_map = {None: "o"}

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Plot by category combinations
    if color_col is not None and shape_col is not None:
        groups = plot_df.groupby([color_col, shape_col])
        for (color_cat, shape_cat), group in groups:
            ax.scatter(
                group[x_col],
                group[y_col],
                color=color_map[color_cat],
                marker=marker_map[shape_cat],
                edgecolor="black",
                linewidth=0.5,
                s=70,
                alpha=0.85,
            )

    elif color_col is not None:
        groups = plot_df.groupby(color_col)
        for color_cat, group in groups:
            ax.scatter(
                group[x_col],
                group[y_col],
                color=color_map[color_cat],
                marker="o",
                edgecolor="black",
                linewidth=0.5,
                s=70,
                alpha=0.85,
                label=str(color_cat),
            )

    elif shape_col is not None:
        groups = plot_df.groupby(shape_col)
        for shape_cat, group in groups:
            ax.scatter(
                group[x_col],
                group[y_col],
                color="black",
                marker=marker_map[shape_cat],
                edgecolor="black",
                linewidth=0.5,
                s=70,
                alpha=0.85,
                label=str(shape_cat),
            )

    else:
        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            color="black",
            marker="o",
            edgecolor="black",
            linewidth=0.5,
            s=70,
            alpha=0.85,
        )

    # Parity line
    min_val = min(plot_df[x_col].min(), plot_df[y_col].min())
    max_val = max(plot_df[x_col].max(), plot_df[y_col].max())
    padding = 0.05 * (max_val - min_val)

    min_axis = min_val - padding
    max_axis = max_val + padding

    ax.plot(
        [min_axis, max_axis],
        [min_axis, max_axis],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Parity, y = x",
    )

    ax.set_xlim(min_axis, max_axis)
    ax.set_ylim(min_axis, max_axis)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)

    # Error metrics
    error = plot_df[y_col] - plot_df[x_col]
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    mape = np.mean(np.abs(error / plot_df[x_col])) * 100
    bias = np.mean(error)

    text = (
        f"MAE = {mae:.2f} {units}\n"
        f"RMSE = {rmse:.2f} {units}\n"
        f"MAPE = {mape:.1f}%\n"
        f"Bias = {bias:.2f} {units}"
    )

    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    # Custom legends
    legend_elements = []

    if color_col is not None:
        for cat, color in color_map.items():
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"{color_col}: {cat}",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=8,
                )
            )

    if shape_col is not None:
        for cat, marker in marker_map.items():
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="black",
                    label=f"{shape_col}: {cat}",
                    linestyle="None",
                    markersize=8,
                )
            )

    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Parity, y = x",
        )
    )

    ax.legend(handles=legend_elements, fontsize=9, loc="best")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias,
        "N": len(plot_df),
    }