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

try:
    from .et_temperature_field import load_et_temperature_field
except ImportError:
    from et_temperature_field import load_et_temperature_field


ET_TEMPERATURE_FIELD = Path(
    "beamer/figures/data/BU_ET/0_250W_0.5ms/ET_temperature.vti"
)
ET_TEMPERATURE_ARRAY = None
ET_SCAN_DIRECTION_X_SIGN = None
ET_SLICE_SPACING_UM = None
LEFT_PANEL_LABEL = "ET"
TCAM_MESH = Path("beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/result.vtu")
TCAM_TEMPERATURE_ARRAY = "temperature"
METADATA_CSV = Path(
    "beamer/figures/data/BU_ET/0_250W_0.5ms/metadata.csv"
)
OUTPUT_DIR = Path("beamer/figures/bayesian/multi_warped_gpr/heldout_250W_0.5ms_15cases")

SLICE_X_UM = (0.0, 75.0, 150.0)
PLOT_Y_LIMIT_UM = 100.0
PLOT_Z_MIN_UM = -115.0
SAMPLE_GRID_NY = 241
SAMPLE_GRID_NZ = 231
KEYHOLE_COLOR = "#555555"
MIN_KEYHOLE_WIDTH_CELLS = 3
MIN_TCAM_CONTOUR_LENGTH_UM = 5.0


def _resampled_axis(axis, limits_um, spacing_um):
    """Return a regularly spaced subset aligned to the source-grid origin."""
    lower = max(float(axis[0]), float(limits_um[0]))
    upper = min(float(axis[-1]), float(limits_um[1]))
    origin = float(axis[0])
    first_index = int(np.ceil((lower - origin) / spacing_um - 1.0e-12))
    last_index = int(np.floor((upper - origin) / spacing_um + 1.0e-12))
    values = origin + spacing_um * np.arange(first_index, last_index + 1)
    if values.size == 0:
        raise ValueError(
            f"Interpolation spacing {spacing_um:g} um selects no points "
            f"between {lower:g} and {upper:g} um."
        )
    return values


def load_et_slices(interpolation_spacing_um=None):
    """Load all requested ET planes from one structured temperature field."""
    field = load_et_temperature_field(
        ET_TEMPERATURE_FIELD,
        temperature_array=ET_TEMPERATURE_ARRAY,
        scan_direction_x_sign=ET_SCAN_DIRECTION_X_SIGN,
    )
    slices = {}
    if interpolation_spacing_um is not None:
        interpolation_spacing_um = float(interpolation_spacing_um)
        if interpolation_spacing_um <= 0.0:
            raise ValueError("ET slice interpolation spacing must be positive.")
        y = _resampled_axis(
            field.y_um,
            (0.0, PLOT_Y_LIMIT_UM),
            interpolation_spacing_um,
        )
        z = _resampled_axis(
            field.z_um,
            (PLOT_Z_MIN_UM, 0.0),
            interpolation_spacing_um,
        )
        yy, zz = np.meshgrid(y, z)
        interpolator = field.interpolator(bounds_error=True)

    for x in SLICE_X_UM:
        # SLICE_X_UM is distance behind the laser. The trailing direction is
        # opposite the field's recorded scan direction.
        field_x = -field.scan_direction_x_sign * x
        if interpolation_spacing_um is None:
            actual_x, yy, zz, temperature = field.yz_slice(
                field_x,
                y_limits_um=(0.0, PLOT_Y_LIMIT_UM),
                z_limits_um=(PLOT_Z_MIN_UM, 0.0),
            )
            if not np.isclose(actual_x, field_x):
                print(
                    f"Using nearest ET plane x={actual_x:g} um for requested "
                    f"trailing distance {x:g} um."
                )
        else:
            coordinates = np.column_stack(
                (
                    np.full(yy.size, field_x),
                    yy.ravel(),
                    zz.ravel(),
                )
            )
            temperature = interpolator(coordinates).reshape(yy.shape)
        slices[x] = (yy, zz, temperature)
    if interpolation_spacing_um is not None:
        print(
            "Interpolated plotted ET YZ planes at "
            f"{interpolation_spacing_um:g} um spacing."
        )
    return slices, field.temperature_min, field.temperature_max


def load_tcam_source():
    """Read the connected TCAM element mesh and expose point temperature."""
    root = pv.read(TCAM_MESH)
    if isinstance(root, pv.MultiBlock):
        if "Element Blocks" in root.keys():
            # Exodus-II result.e exported by the TCAM GUI.
            source = root["Element Blocks"].combine()
        else:
            # Native TC-Python result.pvd, normally containing Block-00.
            source = root.combine()
    else:
        source = root

    # Match TC-Python's AdditiveManufacturingResult.get_pyvista_mesh()
    # filtering when reading the native PVD bundle directly.
    if "subdomain_id" in source.array_names:
        source = source.threshold(
            value=3,
            scalars="subdomain_id",
            invert=True,
        )

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
        description=(
            "Stack ET/TCAM YZ comparisons at requested distances behind "
            "the laser."
        )
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        type=float,
        default=SLICE_X_UM,
        metavar="X_UM",
        help=(
            "Distances behind the laser in micrometers "
            "(default: 0 80 160)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path; otherwise the slice locations name it.",
    )
    parser.add_argument(
        "--et-temperature-field",
        type=Path,
        default=ET_TEMPERATURE_FIELD,
        help="Structured VTI or legacy CSV shown in the left half.",
    )
    parser.add_argument(
        "--et-temperature-array",
        default=ET_TEMPERATURE_ARRAY,
        help="Optional VTI point-data array for the left temperature field.",
    )
    parser.add_argument(
        "--et-scan-direction-x-sign",
        type=int,
        choices=(-1, 1),
        default=ET_SCAN_DIRECTION_X_SIGN,
        help="Override the left field's scan-direction sign.",
    )
    parser.add_argument(
        "--left-label",
        default=LEFT_PANEL_LABEL,
        help="Heading for the left half of the comparison.",
    )
    parser.add_argument(
        "--et-slice-spacing-um",
        type=float,
        default=ET_SLICE_SPACING_UM,
        help=(
            "Optionally interpolate only the plotted ET YZ planes at this "
            "spacing; the source VTI is not modified."
        ),
    )
    parser.add_argument("--tcam-mesh", type=Path, default=TCAM_MESH)
    return parser.parse_args()


def main(
    slice_x_um=SLICE_X_UM,
    output_png=None,
    et_slice_spacing_um=ET_SLICE_SPACING_UM,
):
    global SLICE_X_UM
    SLICE_X_UM = tuple(slice_x_um)
    if output_png is None:
        output_png = output_path_for_slices(SLICE_X_UM)

    liquidus = pd.read_csv(
        METADATA_CSV,
        usecols=["liquidus_temperature_k"],
    ).iloc[0, 0]
    et_slices, et_global_min, et_global_max = load_et_slices(
        et_slice_spacing_um
    )
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

    axes[0].text(
        0.25,
        1.025,
        LEFT_PANEL_LABEL,
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    axes[0].text(
        0.75,
        1.025,
        "TCAM",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
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
    ET_TEMPERATURE_FIELD = arguments.et_temperature_field
    ET_TEMPERATURE_ARRAY = arguments.et_temperature_array
    ET_SCAN_DIRECTION_X_SIGN = arguments.et_scan_direction_x_sign
    LEFT_PANEL_LABEL = arguments.left_label
    TCAM_MESH = arguments.tcam_mesh
    main(
        arguments.slices,
        arguments.output,
        arguments.et_slice_spacing_um,
    )
