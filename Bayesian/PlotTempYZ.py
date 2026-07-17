#!/usr/bin/env python3
"""Compare ParaView and ET temperature slices across a YZ projection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, label


TEMPERATURE_CSV = Path("beamer-template/figures/250_0.5/ET_alloy0_250_0.5.csv")
PARAVIEW_CSV = Path("beamer-template/figures/data/better_plane_resample.csv") # only need if USE_TCAM_MESH = False
TCAM_MESH = Path("beamer-template/figures/data/result.e")
TCAM_TEMPERATURE_ARRAY = "temperature"
USE_TCAM_MESH = True
METADATA_CSV = Path("beamer-template/figures/250_0.5/ET_meta_alloy0_250_0.5.csv")
ET_X_UM = -10.0
PARAVIEW_X_UM = -10.0
OUTPUT_PNG = Path(
    f"beamer-template/figures/TCAM/ET_TC_temp_yz_{ET_X_UM:g}um.png"
)
USE_TCAM_SLICE_MAX = False
PLOT_Y_LIMIT_UM = 100.0
PLOT_Z_MIN_UM = -115.0
INTERPOLATION_NY = 241
INTERPOLATION_NZ = 201
KEYHOLE_COLOR = "#202020"
MIN_KEYHOLE_WIDTH_CELLS = 3
MIN_TCAM_CONTOUR_LENGTH_UM = 5.0
#TEMPERATURE_VMAX_K = 6000 #Custom max temperature


def load_laser_center_slice():
    parts = []
    for chunk in pd.read_csv(
        TEMPERATURE_CSV,
        usecols=["x", "y", "z", "T_ET"],
        chunksize=500_000,
    ):
        on_slice = np.isclose(chunk["x"], ET_X_UM, atol=1.0e-6)
        parts.append(chunk.loc[on_slice, ["y", "z", "T_ET"]])

    plane = pd.concat(parts, ignore_index=True)
    temperature = plane.pivot(
        index="z",
        columns="y",
        values="T_ET",
    ).sort_index()
    y = temperature.columns.to_numpy()
    z = temperature.index.to_numpy()
    values = temperature.to_numpy()

    return y, z, values


def load_paraview_slice():
    points = pd.read_csv(
        PARAVIEW_CSV,
        usecols=["Points_0", "Points_1", "Points_2", "temperature"],
    )
    points["x"] = points["Points_0"] * 1.0e6
    points["y"] = points["Points_1"] * 1.0e6
    points["z"] = points["Points_2"] * 1.0e6
    return points.groupby(
        ["x", "y", "z"],
        as_index=False,
    )["temperature"].mean()


def mask_keyhole(temperature, invalid):
    """Mask the invalid region connected to the top material surface."""
    components, _ = label(invalid)
    surface_components = np.unique(components[-1, :])
    surface_components = surface_components[surface_components != 0]
    keyhole_components = []
    for component in surface_components:
        columns = np.where(components == component)[1]
        if len(np.unique(columns)) >= MIN_KEYHOLE_WIDTH_CELLS:
            keyhole_components.append(component)
    keyhole = np.isin(components, keyhole_components)

    # Fill isolated invalid probe/sampling pixels from the closest valid
    # material point. A one-column invalid strip on the y=0 mesh boundary is
    # a VTK probing artifact, not a keyhole. Only a top-connected cavity with
    # appreciable lateral width is retained.
    stray = invalid & ~keyhole
    if stray.any():
        nearest = distance_transform_edt(
            invalid,
            return_distances=False,
            return_indices=True,
        )
        temperature[stray] = temperature[tuple(nearest[:, stray])]
    return np.ma.masked_where(keyhole, temperature)


def load_tcam_mesh_slice():
    """Probe a regular YZ plane directly from the connected TCAM mesh."""
    import pyvista as pv

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
                f"Could not find temperature array "
                f"'{TCAM_TEMPERATURE_ARRAY}' in {TCAM_MESH}. "
                f"Available arrays: {available}"
            )

    y = np.linspace(0.0, PLOT_Y_LIMIT_UM, INTERPOLATION_NY)
    z = np.linspace(PLOT_Z_MIN_UM, 0.0, INTERPOLATION_NZ)
    yy, zz = np.meshgrid(y, z)
    query_points = np.column_stack(
        (
            np.full(yy.size, -PARAVIEW_X_UM * 1.0e-6),
            yy.ravel() * 1.0e-6,
            zz.ravel() * 1.0e-6,
        )
    )
    sampled = pv.PolyData(query_points).sample(
        source,
        snap_to_closest_point=False,
    )
    valid = np.asarray(sampled["vtkValidPointMask"], dtype=bool).reshape(
        yy.shape
    )
    temperature = np.asarray(
        sampled[TCAM_TEMPERATURE_ARRAY],
        dtype=float,
    ).reshape(yy.shape)
    temperature = mask_keyhole(temperature, ~valid)
    print(
        f"Probed {temperature.shape[1]}x{temperature.shape[0]} TCAM plane "
        f"from {TCAM_MESH} at x = {PARAVIEW_X_UM:g} um."
    )
    return yy, zz, temperature


def interpolate_paraview_slice(points):
    # A ParaView Slice export is planar, so all x coordinates are equal and
    # a regular Plane export can be reconstructed without any interpolation.
    planar = np.ptp(points["x"].to_numpy()) <= 1.0e-6
    display_points = points.loc[
        (points["y"] >= 0.0)
        & (points["y"] <= PLOT_Y_LIMIT_UM)
        & (points["z"] >= PLOT_Z_MIN_UM)
        & (points["z"] <= 0.0)
    ]
    unique_y = np.sort(display_points["y"].unique())
    unique_z = np.sort(display_points["z"].unique())
    regular_plane = (
        planar
        and len(display_points) == len(unique_y) * len(unique_z)
    )

    if regular_plane:
        plane = display_points.pivot(
            index="z",
            columns="y",
            values="temperature",
        )
        plane = plane.sort_index().sort_index(axis=1)
        plane = plane.loc[
            (plane.index >= PLOT_Z_MIN_UM) & (plane.index <= 0.0),
            (plane.columns >= 0.0) & (plane.columns <= PLOT_Y_LIMIT_UM),
        ]
        yy, zz = np.meshgrid(
            plane.columns.to_numpy(),
            plane.index.to_numpy(),
        )
        temperature = plane.to_numpy(copy=True)
        # ParaView uses zero temperature for samples outside the material
        # domain. The keyhole is the invalid component connected to the top
        # surface; isolated invalid boundary cells are sampling artifacts and
        # are filled from the nearest valid material point.
        invalid = ~np.isfinite(temperature) | (temperature <= 0.0)
        temperature = mask_keyhole(temperature, invalid)
        print(
            f"Using native {temperature.shape[1]}x{temperature.shape[0]} "
            f"TCAM plane at x = {points['x'].mean():g} um."
        )
        return yy, zz, temperature

    # Interpolate only over the displayed melt-pool neighborhood when the
    # input is an irregular slice or an unsliced volume export.
    y = np.linspace(0.0, PLOT_Y_LIMIT_UM, INTERPOLATION_NY)
    z = np.linspace(PLOT_Z_MIN_UM, 0.0, INTERPOLATION_NZ)
    yy, zz = np.meshgrid(y, z)

    if planar:
        source_points = (points["y"], points["z"])
        linear_query = (yy, zz)
        print(
            f"Using 2D y-z interpolation for TCAM slice at "
            f"x = {points['x'].mean():g} um."
        )
    else:
        xx = np.full_like(yy, -PARAVIEW_X_UM)
        source_points = (points["x"], points["y"], points["z"])
        linear_query = (xx, yy, zz)

    temperature = griddata(
        points=source_points,
        values=points["temperature"],
        xi=linear_query,
        method="linear",
    )
    # Linear interpolation can leave holes near the convex-hull boundary.
    # Fill only those holes with nearest-neighbor values so the displayed
    # temperature field remains continuous.
    missing = ~np.isfinite(temperature)
    if missing.any():
        if planar:
            nearest_query = (yy[missing], zz[missing])
        else:
            nearest_query = (xx[missing], yy[missing], zz[missing])
        nearest = griddata(
            points=source_points,
            values=points["temperature"],
            xi=nearest_query,
            method="nearest",
        )
        temperature[missing] = nearest
    valid_source = points["temperature"].to_numpy() > 0.0
    valid = griddata(
        points=source_points,
        values=valid_source.astype(np.uint8),
        xi=linear_query,
        method="nearest",
    ).astype(bool)
    return yy, zz, np.ma.masked_where(~valid, temperature)


def draw_tcam_liquidus(axis, y, z, temperature, liquidus):
    """Draw physically significant TCAM liquidus components only."""
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
            color="white",
            linewidth=1.5,
            solid_capstyle="round",
            solid_joinstyle="round",
        )


def main():
    liquidus = pd.read_csv(
        METADATA_CSV,
        usecols=["liquidus_temperature_k"],
    ).iloc[0, 0]
    y, z, temperature = load_laser_center_slice()
    if USE_TCAM_MESH:
        paraview_y, paraview_z, paraview_temperature = (
            load_tcam_mesh_slice()
        )
        tcam_sample_description = f"mesh {TCAM_MESH}"
    else:
        paraview = load_paraview_slice()
        paraview_y, paraview_z, paraview_temperature = (
            interpolate_paraview_slice(paraview)
        )
        tcam_sample_description = f"{len(paraview)} CSV samples"
    yy, zz = np.meshgrid(y, z)
    if USE_TCAM_SLICE_MAX:
        vmax = float(np.ma.max(paraview_temperature))
    else:
        vmax = np.nanmax(temperature)
        #vmax = TEMPERATURE_VMAX_K #Custom max temperature
    vmin = min(np.nanmin(temperature), float(np.ma.min(paraview_temperature)))

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(KEYHOLE_COLOR)
    paraview_image = axis.pcolormesh(
        paraview_y,
        paraview_z,
        paraview_temperature,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    paraview_image.set_clip_path(
        Rectangle(
            (0.0, PLOT_Z_MIN_UM),
            PLOT_Y_LIMIT_UM,
            -PLOT_Z_MIN_UM,
            transform=axis.transData,
        )
    )
    image = axis.pcolormesh(
        -yy,
        zz,
        temperature,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    image.set_clip_path(
        Rectangle(
            (-PLOT_Y_LIMIT_UM, PLOT_Z_MIN_UM),
            PLOT_Y_LIMIT_UM,
            -PLOT_Z_MIN_UM,
            transform=axis.transData,
        )
    )
    axis.contour(
        -yy,
        zz,
        temperature,
        levels=[liquidus],
        colors="white",
        linewidths=1.5,
    )
    draw_tcam_liquidus(
        axis,
        paraview_y,
        paraview_z,
        paraview_temperature,
        liquidus,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Temperature (K)")
    if np.ma.getmaskarray(paraview_temperature).any():
        axis.legend(
            handles=[
                Patch(
                    facecolor=KEYHOLE_COLOR,
                    edgecolor="black",
                    label="Keyhole (vapor cavity)",
                )
            ],
            loc="lower right",
            frameon=True,
            framealpha=0.92,
        )

    axis.set_xlabel("y (um)")
    axis.set_ylabel("z (um)")
    axis.set_title(
        f"ET x = {ET_X_UM:g} um | TCAM x = {PARAVIEW_X_UM:g} um"
    )
    axis.set_xlim(-PLOT_Y_LIMIT_UM, PLOT_Y_LIMIT_UM)
    axis.set_ylim(PLOT_Z_MIN_UM, 0.0)
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(
        f"Prepared TCAM slice from {tcam_sample_description} at "
        f"x = {PARAVIEW_X_UM:g} um."
    )
    print(f"Saved ParaView and ET temperature comparison to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
