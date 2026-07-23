#!/usr/bin/env python3
"""Stack ET/TCAM YZ temperature comparisons at several melt-pool locations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, label


TEMPERATURE_CSV = Path(
    "beamer/figures/250_0.5/ET_alloy0_250_0.5.csv"
)
TCAM_MESH = Path("beamer/figures/data/result.e")
TCAM_TEMPERATURE_ARRAY = "temperature"
METADATA_CSV = Path(
    "beamer/figures/250_0.5/ET_meta_alloy0_250_0.5.csv"
)
OUTPUT_DIR = Path("beamer/figures/TCAM")

SLICE_X_UM = (0.0, 80.0, 160.0)
PLOT_Y_LIMIT_UM = 100.0
PLOT_Z_MIN_UM = -115.0
SAMPLE_GRID_NY = 241
SAMPLE_GRID_NZ = 231
KEYHOLE_COLOR = "#FFFFFF"  # Previous color: "#555555"
MIN_KEYHOLE_WIDTH_CELLS = 3
MIN_TCAM_CONTOUR_LENGTH_UM = 5.0


def load_et_slices():
    """Load all requested ET planes in one pass through the temperature CSV."""
    requested = np.asarray(SLICE_X_UM)
    parts = {x: [] for x in SLICE_X_UM}
    global_temperature_min = np.inf
    global_temperature_max = -np.inf
    for chunk in pd.read_csv(
        TEMPERATURE_CSV,
        usecols=["x", "y", "z", "T_ET"],
        chunksize=500_000,
    ):
        global_temperature_min = min(
            global_temperature_min,
            chunk["T_ET"].min(),
        )
        global_temperature_max = max(
            global_temperature_max,
            chunk["T_ET"].max(),
        )
        chunk_x = chunk["x"].to_numpy()
        for x in SLICE_X_UM:
            selected = np.isclose(chunk_x, x, atol=1.0e-6)
            if selected.any():
                parts[x].append(chunk.loc[selected, ["y", "z", "T_ET"]])

    slices = {}
    for x in SLICE_X_UM:
        if not parts[x]:
            raise ValueError(
                f"No ET data were found at x = {x:g} um. "
                f"Requested locations: {requested.tolist()}"
            )
        points = pd.concat(parts[x], ignore_index=True)
        points = points.loc[
            (points["y"] >= 0.0)
            & (points["y"] <= PLOT_Y_LIMIT_UM)
            & (points["z"] >= PLOT_Z_MIN_UM)
            & (points["z"] <= 0.0)
        ]
        plane = points.pivot(
            index="z",
            columns="y",
            values="T_ET",
        ).sort_index().sort_index(axis=1)
        y = plane.columns.to_numpy()
        z = plane.index.to_numpy()
        yy, zz = np.meshgrid(y, z)
        slices[x] = (yy, zz, plane.to_numpy())
    return slices, global_temperature_min, global_temperature_max


def load_tcam_source():
    """Read the connected TCAM element mesh and expose point temperature."""
    root = pv.read(TCAM_MESH)
    if isinstance(root, pv.MultiBlock):
        if "Element Blocks" not in root.keys():
            raise ValueError(
                f"Could not find 'Element Blocks' in {TCAM_MESH}. "
                f"Available blocks: {list(root.keys())}"
            )
        source = root["Element Blocks"].combine()
    else:
        source = root

    if TCAM_TEMPERATURE_ARRAY not in source.point_data:
        if TCAM_TEMPERATURE_ARRAY in source.cell_data:
            source = source.cell_data_to_point_data(pass_cell_data=True)
        else:
            available = sorted(
                set(source.point_data.keys()) | set(source.cell_data.keys())
            )
            raise ValueError(
                f"Could not find '{TCAM_TEMPERATURE_ARRAY}' in {TCAM_MESH}. "
                f"Available arrays: {available}"
            )
    return source


def mask_keyhole(temperature, invalid):
    """Retain top-connected cavities and repair thin probe-boundary artifacts."""
    components, _ = label(invalid)
    surface_components = np.unique(components[-1, :])
    surface_components = surface_components[surface_components != 0]
    keyhole_components = []
    for component in surface_components:
        columns = np.where(components == component)[1]
        if len(np.unique(columns)) >= MIN_KEYHOLE_WIDTH_CELLS:
            keyhole_components.append(component)
    keyhole = np.isin(components, keyhole_components)

    stray = invalid & ~keyhole
    if stray.any():
        nearest = distance_transform_edt(
            invalid,
            return_distances=False,
            return_indices=True,
        )
        temperature[stray] = temperature[tuple(nearest[:, stray])]
    return np.ma.masked_where(keyhole, temperature)


def probe_tcam_slices(source):
    """Probe each requested YZ plane using the original volume-cell topology."""
    y = np.linspace(0.0, PLOT_Y_LIMIT_UM, SAMPLE_GRID_NY)
    z = np.linspace(PLOT_Z_MIN_UM, 0.0, SAMPLE_GRID_NZ)
    yy, zz = np.meshgrid(y, z)
    slices = {}
    for x in SLICE_X_UM:
        query_points = np.column_stack(
            (
                np.full(yy.size, -x * 1.0e-6),
                yy.ravel() * 1.0e-6,
                zz.ravel() * 1.0e-6,
            )
        )
        sampled = pv.PolyData(query_points).sample(
            source,
            snap_to_closest_point=False,
        )
        valid = np.asarray(
            sampled["vtkValidPointMask"],
            dtype=bool,
        ).reshape(yy.shape)
        temperature = np.asarray(
            sampled[TCAM_TEMPERATURE_ARRAY],
            dtype=float,
        ).reshape(yy.shape)
        slices[x] = (yy, zz, mask_keyhole(temperature, ~valid))
        print(f"Probed TCAM plane at x = {x:g} um.")
    return slices


def draw_liquidus(axis, y, z, temperature, liquidus, filter_segments=False):
    """Draw the liquidus, optionally removing short TCAM mesh fragments."""
    valid_values = np.ma.compressed(np.ma.masked_invalid(temperature))
    if (
        valid_values.size == 0
        or liquidus < valid_values.min()
        or liquidus > valid_values.max()
    ):
        return

    if not filter_segments:
        axis.contour(
            y,
            z,
            temperature,
            levels=[liquidus],
            colors="cyan",
            linewidths=1.1,
        )
        return

    contour = axis.contour(
        y,
        z,
        temperature,
        levels=[liquidus],
        colors="none",
    )
    segments = [segment for segment in contour.allsegs[0] if len(segment) > 1]
    contour.remove()
    kept = []
    for segment in segments:
        arc_length = np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()
        if arc_length >= MIN_TCAM_CONTOUR_LENGTH_UM:
            kept.append(segment)
    if not kept and segments:
        kept = [max(segments, key=len)]
    for segment in kept:
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            color="cyan",
            linewidth=1.1,
            solid_capstyle="round",
            solid_joinstyle="round",
        )


def output_path_for_slices(slice_x_um):
    distances = "_".join(f"{x:g}" for x in slice_x_um)
    return OUTPUT_DIR / f"ET_TC_temp_yz_{distances}um_inferno.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stack ET/TCAM YZ comparisons at requested x locations."
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        type=float,
        default=SLICE_X_UM,
        metavar="X_UM",
        help="Slice locations in micrometers (default: 0 40 80 120 160).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path; otherwise the slice locations name it.",
    )
    return parser.parse_args()


def main(slice_x_um=SLICE_X_UM, output_png=None):
    global SLICE_X_UM
    SLICE_X_UM = tuple(slice_x_um)
    if output_png is None:
        output_png = output_path_for_slices(SLICE_X_UM)

    liquidus = pd.read_csv(
        METADATA_CSV,
        usecols=["liquidus_temperature_k"],
    ).iloc[0, 0]
    et_slices, et_global_min, et_global_max = load_et_slices()
    tcam_slices = probe_tcam_slices(load_tcam_source())

    # Anchor the shared scale to the full ET temperature range. Include a
    # lower TCAM minimum if one exists, but never extend the upper limit above
    # the maximum ET temperature.
    vmax = et_global_max
    valid_minima = [et_global_min] + [
        float(np.ma.min(values)) for _, _, values in tcam_slices.values()
    ]
    vmin = min(valid_minima)

    figure_height = 2.5 * len(SLICE_X_UM) + 0.5
    figure, axes = plt.subplots(
        len(SLICE_X_UM),
        1,
        figsize=(6.8, figure_height),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(KEYHOLE_COLOR)
    any_keyhole = False
    mappable = None

    for axis, x in zip(axes, SLICE_X_UM):
        et_y, et_z, et_temperature = et_slices[x]
        tcam_y, tcam_z, tcam_temperature = tcam_slices[x]

        axis.pcolormesh(
            -et_y,
            et_z,
            et_temperature,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        mappable = axis.pcolormesh(
            tcam_y,
            tcam_z,
            tcam_temperature,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        draw_liquidus(
            axis,
            -et_y,
            et_z,
            et_temperature,
            liquidus,
        )
        draw_liquidus(
            axis,
            tcam_y,
            tcam_z,
            tcam_temperature,
            liquidus,
            filter_segments=True,
        )
        keyhole_present = np.ma.getmaskarray(tcam_temperature).any()
        any_keyhole = any_keyhole or keyhole_present

        axis.text(
            0.015,
            0.08,
            rf"$x = {x:g}\,\mu\mathrm{{m}}$",
            transform=axis.transAxes,
            fontsize=14,
            color="white",
            bbox={
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 1.8,
            },
        )
        axis.set_xlim(-PLOT_Y_LIMIT_UM, PLOT_Y_LIMIT_UM)
        axis.set_ylim(PLOT_Z_MIN_UM, 0.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylabel("z (um)", fontsize=9)
        axis.tick_params(labelsize=8)

    axes[0].set_title("ET                                   TCAM", fontsize=12)
    axes[-1].set_xlabel("y (um)", fontsize=10)

    plot_bottom = 0.75 / figure_height if any_keyhole else 0.04
    plot_top = 0.965
    figure.subplots_adjust(
        left=0.13,
        right=0.80,
        top=plot_top,
        bottom=plot_bottom,
        hspace=0.08,
    )
    panel_right = max(axis.get_position().x1 for axis in axes)
    colorbar_axis = figure.add_axes(
        [panel_right + 0.018, plot_bottom, 0.05, plot_top - plot_bottom]
    )
    colorbar = figure.colorbar(mappable, cax=colorbar_axis)
    colorbar.set_label("Temperature (K)", fontsize=13, labelpad=12)
    colorbar.ax.tick_params(labelsize=11, length=6, width=1.2)

    if any_keyhole:
        figure.legend(
            handles=[
                Patch(
                    facecolor=KEYHOLE_COLOR,
                    edgecolor="black",
                    label="Keyhole (vapor cavity)",
                )
            ],
            loc="lower center",
            bbox_to_anchor=(0.48, 0.005),
            fontsize=9,
            frameon=True,
            framealpha=0.92,
        )

    figure.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(f"Shared temperature range: {vmin:g} to {vmax:g} K.")
    print(f"Saved stacked ET/TCAM comparison to {output_png}")


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.slices, arguments.output)
