import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from eagar_tsai import (
    BeamParameters,
    MaterialProperties,
    SimulationDomain,
    compute_melt_pool,
    compute_temperature_volume,
)

DEFAULT_EXCEL = "effective_cp_data.xlsx"
DEFAULT_ROW_INDEX = 2 # EXCEL ROW - 2
DEFAULT_OUTPUT_VTI = "beamer/figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.vti"
DEFAULT_OUTPUT_PNG = "beamer/figures/250_0.5/ET_3D_temperature_alloy0_250_0.5.png"
DEFAULT_LIMITS_UM = {
    "x": (-160.0, 380.0),
    "y": (0.0, 200.0),
    "z": (-60.0, 0.0),
}
DEFAULT_MELT_POOL_DOMAIN = SimulationDomain(
    x_length_um=600.0,
    y_length_um=200.0,
    z_depth_um=250.0,
    spatial_resolution_um=2.0,
)
DEFAULT_VTI_PADDING_UM = 20.0

INPUT_COLUMNS = {
    "scan_speed_m_s": ["velocity_m_s", "Velocity_m/s", "Velocity (m/s)", "v"],
    "power_w": ["power_w", "Power", "Power (W)", "P"],
    "beam_diameter_m": ["beam_diameter_m", "Beam_diameter_m", "Beam Diam (m)"],
    "absorptivity": ["absorptivity", "Absorptivity"],
    "liquidus_temperature_k": ["liquidus_temperature_k", "Liquidus (K)", "TL", "PROP LT (K)"],
    "thermal_conductivity_w_mk": [
        "thermal_conductivity_w_mk",
        "THCD LT (W/mK)",
        "K_LT",
        "PROP LT THCD (W/(mK))",
        "EQ LT THCD (W/mK)",
    ],
    "density_kg_m3": ["density_kg_m3", "RT Density (kg/m3)", "PROP RT Density (kg/m3)"],
    "heat_capacity_j_kgk": [
        "Cp, Sheikh (J/kgK)",
        "Cp, LT (J/kgK)",
        "Cp_LT",
        "PROP LT C (J/(kg K))",
    ],
}


def _first_existing(row, candidates, output_name):
    for col in candidates:
        if col in row.index and not pd.isna(row[col]):
            return float(row[col])
    raise ValueError(f"Excel row is missing a value for {output_name}. Tried: {candidates}")


def load_eagar_tsai_inputs_from_excel(excel_path=DEFAULT_EXCEL, row_index=DEFAULT_ROW_INDEX):
    """Load process and material properties for one alloy/condition row."""
    df = pd.read_excel(excel_path)
    row = df.iloc[int(row_index)]
    return {
        output_name: _first_existing(row, candidates, output_name)
        for output_name, candidates in INPUT_COLUMNS.items()
    }


def _domain_from_limits(x_limits_um, y_limits_um, z_limits_um, spatial_res_um):
    """Convert legacy explicit limits to the library's positive domain sizes."""
    x_length_um = max(float(x_limits_um[1]), abs(float(x_limits_um[0])))
    y_length_um = max(abs(float(y_limits_um[0])), abs(float(y_limits_um[1])))
    z_depth_um = max(abs(float(z_limits_um[0])), abs(float(z_limits_um[1])))
    return SimulationDomain(
        x_length_um=x_length_um,
        y_length_um=y_length_um,
        z_depth_um=z_depth_um,
        spatial_resolution_um=float(spatial_res_um),
    )


def _inputs_to_melt_pool_dataframe(inputs):
    """Build the column names expected by eagar_tsai.compute_melt_pool."""
    return pd.DataFrame(
        [
            {
                "velocity_m_s": inputs["scan_speed_m_s"],
                "power_w": inputs["power_w"],
                "beam_diameter_m": inputs["beam_diameter_m"],
                "absorptivity": inputs["absorptivity"],
                "liquidus_temperature_k": inputs["liquidus_temperature_k"],
                "thermal_conductivity_w_mk": inputs["thermal_conductivity_w_mk"],
                "density_kg_m3": inputs["density_kg_m3"],
                "specific_heat_j_kgk": inputs["heat_capacity_j_kgk"],
            }
        ]
    )


def _expanded_vti_limits(result, padding_um):
    """Match workflow_ET.py's melt-pool-based VTI domain expansion."""
    row = result.iloc[0]
    melt_length_um = float(row["melt_length_um"])
    melt_width_um = float(row["melt_width_um"])
    melt_depth_um = float(row["melt_depth_um"])

    x_min, x_max = DEFAULT_LIMITS_UM["x"]
    y_min, y_max = DEFAULT_LIMITS_UM["y"]
    z_min, z_max = DEFAULT_LIMITS_UM["z"]

    return {
        "x": (x_min, max(x_max, melt_length_um + padding_um)),
        "y": (y_min, max(y_max, 0.5 * melt_width_um + padding_um)),
        "z": (min(z_min, -(melt_depth_um + padding_um)), z_max),
    }


def workflow_vti_limits_from_inputs(inputs, padding_um, workers=None, chunk_size=10):
    """Compute workflow-style VTI limits from one ET input row."""
    result = compute_melt_pool(
        _inputs_to_melt_pool_dataframe(inputs),
        domain=DEFAULT_MELT_POOL_DOMAIN,
        workers=workers,
        chunk_size=chunk_size,
        output_dir=None,
        return_field=False,
    )
    return _expanded_vti_limits(result, float(padding_um))


def _trim_render_whitespace(fig, padding_px=18, white_threshold=248):
    """Trim white padding from the PyVista screenshot embedded in the figure."""
    if not fig.axes or not fig.axes[0].images:
        return

    image_ax = fig.axes[0]
    image = image_ax.images[0]
    arr = np.asarray(image.get_array())
    if arr.ndim < 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return

    rgb = arr[..., :3]
    nonwhite = np.any(rgb < white_threshold, axis=2)
    rows = np.flatnonzero(np.any(nonwhite, axis=1))
    cols = np.flatnonzero(np.any(nonwhite, axis=0))
    if rows.size == 0 or cols.size == 0:
        return

    y0 = max(int(rows[0]) - padding_px, 0)
    y1 = min(int(rows[-1]) + padding_px + 1, arr.shape[0])
    x0 = max(int(cols[0]) - padding_px, 0)
    x1 = min(int(cols[-1]) + padding_px + 1, arr.shape[1])
    cropped = arr[y0:y1, x0:x1]

    image.set_data(cropped)
    image.set_extent((0, cropped.shape[1], cropped.shape[0], 0))
    image_ax.set_xlim(0, cropped.shape[1])
    image_ax.set_ylim(cropped.shape[0], 0)


def _plot_temperature_volume(volume):
    """Render the library's 3D view with an inferno temperature scale."""
    plotter = volume.plot_3d(
        mirror_y=True,
        liquidus_contour=True,
        off_screen=True,
        return_plotter=True,
        show_scalar_bar=False,
    )
    for actor in plotter.actors.values():
        mapper = actor.mapper
        if getattr(mapper, "array_name", None) == "Temperature_K":
            mapper.lookup_table.apply_cmap("inferno")

    image = plotter.screenshot(return_img=True)
    plotter.close()
    if image is None:
        raise RuntimeError("PyVista did not return a temperature-volume image.")

    figure = plt.figure(figsize=(4.0, 3.2), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.085],
        hspace=0.04,
    )
    image_axis = figure.add_subplot(grid[0])
    colorbar_axis = figure.add_subplot(grid[1])
    image_axis.imshow(image, aspect="auto")
    image_axis.axis("off")

    scalar_mappable = ScalarMappable(
        cmap="inferno",
        norm=Normalize(vmin=298.0, vmax=float(volume.T_xyz.max())),
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.ax.tick_params(
        labelsize=8,
        length=1.7,
        width=0.4,
        pad=0.8,
    )
    colorbar.ax.xaxis.set_ticks_position("bottom")
    colorbar.ax.xaxis.set_label_position("bottom")
    colorbar.set_label("T (K)", fontsize=8, labelpad=4.0)
    colorbar.outline.set_linewidth(0.4)
    return figure


def compute_eagar_tsai_volume(
    power_w,
    scan_speed_m_s,
    beam_diameter_m,
    absorptivity,
    liquidus_temperature_k,
    thermal_conductivity_w_mk,
    density_kg_m3,
    heat_capacity_j_kgk,
    x_limits_um,
    y_limits_um,
    z_limits_um,
    spatial_res_um=1.0,
    workers=None,
    chunk_size=10,
):
    """Compute a 3-D Eagar-Tsai temperature volume using the eagar_tsai library.

    The older workflow accepted explicit x/y/z limits. The eagar_tsai library
    uses positive domain sizes and automatically expands if the melt pool
    touches a boundary, so the limits are interpreted as the minimum starting
    domain needed to cover the requested extents.
    """
    beam = BeamParameters(
        beam_diameter=float(beam_diameter_m),
        power=float(power_w),
        velocity=float(scan_speed_m_s),
        absorptivity=float(absorptivity),
    )
    material = MaterialProperties(
        liquidus_temperature=float(liquidus_temperature_k),
        thermal_conductivity=float(thermal_conductivity_w_mk),
        density=float(density_kg_m3),
        specific_heat=float(heat_capacity_j_kgk),
    )
    domain = _domain_from_limits(x_limits_um, y_limits_um, z_limits_um, spatial_res_um)
    return compute_temperature_volume(
        beam=beam,
        material=material,
        domain=domain,
        workers=workers,
        chunk_size=chunk_size,
    )


def export_eagar_tsai_vti(
    output_vti,
    power_w,
    scan_speed_m_s,
    beam_diameter_m,
    absorptivity,
    thermal_conductivity_w_mk,
    density_kg_m3,
    heat_capacity_j_kgk,
    x_limits_um,
    y_limits_um,
    z_limits_um,
    spatial_res_um=1.0,
    liquidus_temperature_k=None,
    output_png=None,
    workers=None,
    chunk_size=10,
    mirror_y=False,
    colorbar_gap=0.002,
    trim_render_whitespace=True,
    render_padding_px=18,
):
    """Compute and export an Eagar-Tsai 3-D temperature volume.

    Args:
        output_vti: Path to the VTI file to write.
        output_png: Optional path to a PyVista-rendered image of the volume.
        mirror_y: When ``True``, export/render the full symmetric melt pool.
            The workflow keeps this ``False`` for VTI output to match the
            previous half-domain convention used by downstream G/R extraction.
    """
    if liquidus_temperature_k is None:
        raise ValueError("liquidus_temperature_k is required for library-backed VTI export.")

    volume = compute_eagar_tsai_volume(
        power_w=power_w,
        scan_speed_m_s=scan_speed_m_s,
        beam_diameter_m=beam_diameter_m,
        absorptivity=absorptivity,
        liquidus_temperature_k=liquidus_temperature_k,
        thermal_conductivity_w_mk=thermal_conductivity_w_mk,
        density_kg_m3=density_kg_m3,
        heat_capacity_j_kgk=heat_capacity_j_kgk,
        x_limits_um=x_limits_um,
        y_limits_um=y_limits_um,
        z_limits_um=z_limits_um,
        spatial_res_um=spatial_res_um,
        workers=workers,
        chunk_size=chunk_size,
    )

    output_vti = Path(output_vti)
    vti_path = volume.export_vti(output_vti, mirror_y=mirror_y)

    png_path = None
    if output_png is not None:
        png_path = Path(output_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig = _plot_temperature_volume(volume)
        if trim_render_whitespace:
            _trim_render_whitespace(fig, padding_px=int(render_padding_px))
        if colorbar_gap is not None and len(fig.axes) >= 2:
            image_ax, colorbar_ax = fig.axes[:2]
            image_pos = image_ax.get_position()
            colorbar_pos = colorbar_ax.get_position()
            colorbar_ax.set_position(
                [
                    colorbar_pos.x0,
                    image_pos.y0 - float(colorbar_gap) - colorbar_pos.height,
                    colorbar_pos.width,
                    colorbar_pos.height,
                ]
            )
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

    return vti_path if png_path is None else (vti_path, png_path)


def export_eagar_tsai_vti_from_excel(
    excel_path=DEFAULT_EXCEL,
    row_index=DEFAULT_ROW_INDEX,
    output_vti=DEFAULT_OUTPUT_VTI,
    output_png=DEFAULT_OUTPUT_PNG,
    x_limits_um=DEFAULT_LIMITS_UM["x"],
    y_limits_um=DEFAULT_LIMITS_UM["y"],
    z_limits_um=DEFAULT_LIMITS_UM["z"],
    spatial_res_um=1.0,
    workers=None,
    chunk_size=10,
    mirror_y=False,
    auto_vti_limits=True,
    vti_padding_um=DEFAULT_VTI_PADDING_UM,
    colorbar_gap=0.002,
    trim_render_whitespace=True,
    render_padding_px=18,
):
    """Load one spreadsheet row and export ET 3-D VTI/PNG outputs."""
    inputs = load_eagar_tsai_inputs_from_excel(excel_path, row_index)
    if auto_vti_limits:
        limits = workflow_vti_limits_from_inputs(
            inputs,
            padding_um=vti_padding_um,
            workers=workers,
            chunk_size=chunk_size,
        )
        x_limits_um = limits["x"]
        y_limits_um = limits["y"]
        z_limits_um = limits["z"]
        print(
            "Using workflow VTI limits: "
            f"x={x_limits_um} um, y={y_limits_um} um, z={z_limits_um} um"
        )

    return export_eagar_tsai_vti(
        output_vti=output_vti,
        output_png=output_png,
        x_limits_um=x_limits_um,
        y_limits_um=y_limits_um,
        z_limits_um=z_limits_um,
        spatial_res_um=spatial_res_um,
        workers=workers,
        chunk_size=chunk_size,
        mirror_y=mirror_y,
        colorbar_gap=colorbar_gap,
        trim_render_whitespace=trim_render_whitespace,
        render_padding_px=render_padding_px,
        **inputs,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export an Eagar-Tsai 3-D temperature VTI from an Excel row.")
    parser.add_argument("--excel", default=DEFAULT_EXCEL)
    parser.add_argument("--row-index", type=int, default=DEFAULT_ROW_INDEX, help="Zero-based pandas row index.")
    parser.add_argument("--output-vti", default=DEFAULT_OUTPUT_VTI)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--x-limits-um", nargs=2, type=float, default=DEFAULT_LIMITS_UM["x"])
    parser.add_argument("--y-limits-um", nargs=2, type=float, default=DEFAULT_LIMITS_UM["y"])
    parser.add_argument("--z-limits-um", nargs=2, type=float, default=DEFAULT_LIMITS_UM["z"])
    parser.add_argument("--spatial-res-um", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--mirror-y", action="store_true")
    parser.add_argument(
        "--no-auto-vti-limits",
        action="store_true",
        help="Use the explicit axis limits instead of workflow-style melt-pool-expanded limits.",
    )
    parser.add_argument("--vti-padding-um", type=float, default=DEFAULT_VTI_PADDING_UM)
    parser.add_argument(
        "--colorbar-gap",
        type=float,
        default=0.002,
        help="Normalized figure-coordinate gap between the 3D render and colorbar.",
    )
    parser.add_argument("--no-trim-render-whitespace", action="store_true")
    parser.add_argument("--render-padding-px", type=int, default=18)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = export_eagar_tsai_vti_from_excel(
        excel_path=args.excel,
        row_index=args.row_index,
        output_vti=args.output_vti,
        output_png=args.output_png,
        x_limits_um=tuple(args.x_limits_um),
        y_limits_um=tuple(args.y_limits_um),
        z_limits_um=tuple(args.z_limits_um),
        spatial_res_um=args.spatial_res_um,
        workers=args.workers,
        chunk_size=args.chunk_size,
        mirror_y=args.mirror_y,
        auto_vti_limits=not args.no_auto_vti_limits,
        vti_padding_um=args.vti_padding_um,
        colorbar_gap=args.colorbar_gap,
        trim_render_whitespace=not args.no_trim_render_whitespace,
        render_padding_px=args.render_padding_px,
    )
    if isinstance(outputs, tuple):
        for output in outputs:
            print(f"Wrote {output}")
    else:
        print(f"Wrote {outputs}")
