import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from tc_python import CompositionUnit, TCPython

try:
    from .GR_Map import _load_alloy_from_excel, _tc_python_cache_dir, get_CET_grid
except ImportError:
    from GR_Map import _load_alloy_from_excel, _tc_python_cache_dir, get_CET_grid


LIQUIDUS_CSV = (
    "CalcFiles/Test17/TCAM_keyholing_alloy0_250_0.5/ParaView_fine0.15mm/subdivide_nosmooth_fine0.15_data.csv"
)
OUTPUT_PATH = (
    "CalcFiles/Test17/TCAM_keyholing_alloy0_250_0.5/ParaView_fine0.15mm/microstructure_projection_fine0.15_subdivide_nosmooth.png"
)
INTERPOLATE_PARAVIEW = False
INTERPOLATION_Y_POINTS = 75
INTERPOLATION_Z_POINTS = 115
SMOOTHING_SIGMA = 0.0
POINT_SIZE = 10

EXCEL_PATH = "et_custom_input_data.xlsx"
ROW_INDEX = 0  # Alloy 0; zero-based pandas row index.
ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
MAP_SOURCE = "excel_alloy"
ET_COLUMNS = {"x", "y", "z", "G", "R"}
PARAVIEW_COLUMNS = {
    "Points_0": "x",
    "Points_1": "y",
    "Points_2": "z",
    "Gradient_Magnitude": "G",
    "R (m/s)": "R",
}
MAP_CONFIGS = {
    "excel_alloy": {
        "thermo_db": "TCHEA8",
        "kinetic_db": "MOBHEA3",
        "primary_phase": "BCC_B2",
        "interfacial_energy": 0.5,
        "composition_unit": CompositionUnit.MOLE_FRACTION,
    },
    "cu70ni30": {
        "thermo_db": "TCHEA7",
        "kinetic_db": "MOBHEA3",
        "primary_phase": "FCC_L12",
        "interfacial_energy": 0.5,
        "composition_unit": CompositionUnit.MASS_PERCENT,
        "elements": ["Cu", "Ni"],
        "solutes": {"Ni": 30.0},
    },
}


def _load_projected_liquidus(csv_path):
    df = pd.read_csv(csv_path)

    if ET_COLUMNS.issubset(df.columns):
        is_paraview = False
        df = df.loc[:, ["x", "y", "z", "G", "R"]].copy()
    elif set(PARAVIEW_COLUMNS).issubset(df.columns):
        is_paraview = True
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

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "z", "G", "R"])
    df = df[(df["G"] > 0.0) & (df["R"] > 0.0)].copy()
    if df.empty:
        raise ValueError(f"No positive-G, positive-R liquidus points found in {csv_path}.")

    deepest = df.loc[df["z"].idxmin()]
    df.attrs["is_paraview"] = is_paraview
    return df, deepest


def _positive_r_projection_points(liquidus):
    return liquidus.sort_values("x").reset_index(drop=True)


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
            "y": yy[valid],
            "z": zz[valid],
            "G": g_grid[valid],
            "R": r_grid[valid],
        }
    )


def _format_title_value(value):
    return f"{float(value):g}"


def _first_row_value(row, columns):
    for col in columns:
        if col in row.index and not pd.isna(row[col]):
            return row[col]
    return None


def _projection_title(excel_path, row_index):
    row = pd.read_excel(excel_path).iloc[row_index]
    alloy = _first_row_value(row, ["Alloy", "Unnamed: 0"])
    power = _first_row_value(row, ["power_w", "Power", "Power (W)", "P"])
    velocity = _first_row_value(row, ["velocity_m_s", "Velocity_m/s", "Velocity (m/s)", "v"])

    if alloy is None or power is None or velocity is None:
        return "Melt Pool Liquidus Projection"
    return f"Alloy {_format_title_value(alloy)} (P = {_format_title_value(power)}, V = {_format_title_value(velocity)})"


def _calculate_cet_grid(
    excel_path=EXCEL_PATH,
    row_index=ROW_INDEX,
    element_cols=ELEMENT_COLS,
    map_source=MAP_SOURCE,
    map_configs=MAP_CONFIGS,
):
    config = map_configs[map_source]
    if map_source == "excel_alloy":
        elements, solutes = _load_alloy_from_excel(excel_path, row_index, element_cols)
        element_names = list(elements.keys())
    else:
        solutes = config["solutes"]
        element_names = config["elements"]

    with TCPython() as session:
        session.set_cache_folder(_tc_python_cache_dir())
        system = session.select_thermodynamic_and_kinetic_databases_with_elements(
            config["thermo_db"],
            config["kinetic_db"],
            element_names,
        ).get_system()

        r_grid, g_grid, equiaxed_fraction, _, _, _ = get_CET_grid(
            system,
            solutes,
            primary_phase=config["primary_phase"],
            interfacial_energy=config["interfacial_energy"],
            nb_nucleations_site=4.0e11,
            nucleation_undercooling=4.0,
            equiaxed_exponent=3.13,
            disable_output=True,
            composition_unit=config["composition_unit"],
        )

    return (
        np.asarray(r_grid),
        np.asarray(g_grid),
        np.asarray(equiaxed_fraction, dtype=float),
    )


def _classify_from_gr_map(df, r_grid, g_grid, equiaxed_fraction):
    log_r_values = np.log10(df["R"].to_numpy(dtype=float))
    log_g_values = np.log10(df["G"].to_numpy(dtype=float))

    log_r_grid = np.log10(r_grid)
    log_g_grid = np.log10(g_grid)
    grid_points = np.column_stack([log_r_grid, log_g_grid])

    query_points = np.column_stack([log_r_values, log_g_values])
    chunk_size = 2000
    nearest = np.empty(len(query_points), dtype=int)

    for start in range(0, len(query_points), chunk_size):
        stop = start + chunk_size
        deltas = query_points[start:stop, None, :] - grid_points[None, :, :]
        nearest[start:stop] = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)

    classified = equiaxed_fraction[nearest]
    classified = np.where(np.isnan(classified), -1.0, classified)

    return classified


def plot_projected_liquidus(
    liquidus_csv=LIQUIDUS_CSV,
    output_path=OUTPUT_PATH,
    excel_path=EXCEL_PATH,
    row_index=ROW_INDEX,
    element_cols=ELEMENT_COLS,
    map_source=MAP_SOURCE,
    map_configs=MAP_CONFIGS,
    label_fontsize=12,
    tick_fontsize=10,
    title_fontsize=13,
    legend_fontsize=8,
):
    title = _projection_title(excel_path, row_index)
    liquidus, deepest = _load_projected_liquidus(liquidus_csv)
    projected = _positive_r_projection_points(liquidus)
    if INTERPOLATE_PARAVIEW and liquidus.attrs["is_paraview"]:
        projected = _interpolate_projected_points(projected)

    r_grid, g_grid, equiaxed_fraction = _calculate_cet_grid(
        excel_path=excel_path,
        row_index=row_index,
        element_cols=element_cols,
        map_source=map_source,
        map_configs=map_configs,
    )
    projected["equiaxed_fraction"] = _classify_from_gr_map(
        projected,
        r_grid,
        g_grid,
        equiaxed_fraction,
    )
    projected["y_plot"] = projected["y"].round(0)
    projected["z_plot"] = projected["z"].round(0)

    planar = projected["equiaxed_fraction"] == -1.0
    cellular_dendritic = ~planar
    columnar_y = np.concatenate(
        [
            projected.loc[cellular_dendritic, "y_plot"].to_numpy(dtype=float),
            -projected.loc[cellular_dendritic, "y_plot"].to_numpy(dtype=float),
        ]
    )
    columnar_z = np.concatenate(
        [
            projected.loc[cellular_dendritic, "z_plot"].to_numpy(dtype=float),
            projected.loc[cellular_dendritic, "z_plot"].to_numpy(dtype=float),
        ]
    )
    planar_y = np.concatenate(
        [
            projected.loc[planar, "y_plot"].to_numpy(dtype=float),
            -projected.loc[planar, "y_plot"].to_numpy(dtype=float),
        ]
    )
    planar_z = np.concatenate(
        [
            projected.loc[planar, "z_plot"].to_numpy(dtype=float),
            projected.loc[planar, "z_plot"].to_numpy(dtype=float),
        ]
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(
        columnar_y,
        columnar_z,
        c="#dfe3ff",
        marker="o",
        s=POINT_SIZE,
        linewidths=0,
        alpha=1.0,
        label="Columnar",
    )
    ax.scatter(
        planar_y,
        planar_z,
        c="#b00000",
        marker="o",
        s=POINT_SIZE,
        linewidths=0,
        alpha=1.0,
        label="Planar",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y (um)", fontsize=label_fontsize)
    ax.set_ylabel("z (um)", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    legend = ax.legend(loc="lower right", fontsize=legend_fontsize, borderpad=0.3, handletextpad=0.4)
    legend_handles = legend.legend_handles if hasattr(legend, "legend_handles") else legend.legendHandles
    for handle in legend_handles:
        handle.set_sizes([20])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    planar_count = int((projected["equiaxed_fraction"] == -1.0).sum())
    print(f"Wrote {output_path}")
    print(
        "Projected {count} positive-R liquidus points from x = {xmin:.3f} to {xmax:.3f} um. "
        "Deepest point: y = {y:.3f} um, z = {z:.3f} um.".format(
            count=len(projected),
            xmin=liquidus["x"].min(),
            xmax=liquidus["x"].max(),
            y=deepest["y"],
            z=deepest["z"],
        )
    )
    print(f"Planar-classified points: {planar_count} / {len(projected)}")
    return output_path, deepest, projected


if __name__ == "__main__":
    try:
        plot_projected_liquidus()
    except Exception as exc:
        print(f"PlotMicrostructure failed: {exc}", file=sys.stderr)
        raise
