import os
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from tc_python import CompositionUnit, TCPython

try:
    from .GR_Map import (
        DEFAULT_CET_EQUIAXED_EXPONENT,
        DEFAULT_CET_NUCLEATION_SITES,
        DEFAULT_CET_NUCLEATION_UNDERCOOLING_K,
        DEFAULT_PRIMARY_PHASE,
        _load_alloy_from_excel,
        _load_interfacial_energy_from_excel,
        _load_liquidus_from_excel,
        _cet_grid_cache_metadata,
        _cet_grid_npz_path,
        _load_cet_grid_npz,
        _save_cet_grid_npz,
        _tc_python_cache_dir,
        get_CET_grid,
        resolve_interfacial_energy,
    )
except ImportError:
    from GR_Map import (
        DEFAULT_CET_EQUIAXED_EXPONENT,
        DEFAULT_CET_NUCLEATION_SITES,
        DEFAULT_CET_NUCLEATION_UNDERCOOLING_K,
        DEFAULT_PRIMARY_PHASE,
        _load_alloy_from_excel,
        _load_interfacial_energy_from_excel,
        _load_liquidus_from_excel,
        _cet_grid_cache_metadata,
        _cet_grid_npz_path,
        _load_cet_grid_npz,
        _save_cet_grid_npz,
        _tc_python_cache_dir,
        get_CET_grid,
        resolve_interfacial_energy,
    )


LIQUIDUS_CSV = (
    "beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/"
    "TCAM_GR_alloy0_250_0.5.csv"
)
OUTPUT_PATH = ("beamer/figures/TCAM/TCAM_projected_microstructure_alloy0_250_0.5.png")
INTERPOLATE_PARAVIEW = False
INTERPOLATION_Y_POINTS = 75
INTERPOLATION_Z_POINTS = 115
SMOOTHING_SIGMA = 0.0
POINT_SIZE = 10
MICRON_SCALE_THRESHOLD = 1.0e-2
GRADIENT_K_PER_UM_THRESHOLD = 1.0e4
GR_MAP_COLOR_RANGE = (-1.1, 1.0)
EQUIAXED_COLORBAR_MIN_FRACTION = 0.01

EXCEL_PATH = "effective_cp_data.xlsx"
ROW_INDEX = 2  # Alloy 0; zero-based pandas row index.
ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V", "Co", "Cr", "Fe", "Mn", "Ni"]
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
        "primary_phase": DEFAULT_PRIMARY_PHASE,
        "interfacial_energy": None,
        "composition_unit": CompositionUnit.MOLE_FRACTION,
    },
    "cu70ni30": {
        "thermo_db": "TCHEA7",
        "kinetic_db": "MOBHEA3",
        "primary_phase": "FCC_L12",
        "interfacial_energy": None,
        "liquidus_temperature": 1620.0,
        "composition_unit": CompositionUnit.MASS_PERCENT,
        "elements": ["Cu", "Ni"],
        "solutes": {"Ni": 30.0},
    },
}


def _coordinates_to_microns(df):
    coord_cols = ["x", "y", "z"]
    max_abs_coord = df[coord_cols].abs().to_numpy().max()
    if max_abs_coord < MICRON_SCALE_THRESHOLD:
        df.loc[:, coord_cols] *= 1.0e6
        df.attrs["coordinate_units"] = "m"
    else:
        df.attrs["coordinate_units"] = "um"


def _gradient_to_k_per_m(df):
    finite_g = df["G"].to_numpy(dtype=float)
    finite_g = finite_g[np.isfinite(finite_g) & (finite_g > 0)]
    if finite_g.size and np.nanmedian(finite_g) < GRADIENT_K_PER_UM_THRESHOLD:
        df.loc[:, "G"] *= 1.0e6
        df.attrs["gradient_units"] = "K/um"
    else:
        df.attrs["gradient_units"] = "K/m"


def _load_projected_liquidus(csv_path):
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

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "z", "G", "R"])
    df = df[(df["G"] > 0.0) & (df["R"] > 0.0)].copy()
    if df.empty:
        raise ValueError(f"No positive-G, positive-R liquidus points found in {csv_path}.")
    _gradient_to_k_per_m(df)

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


def _format_title_label(value):
    if isinstance(value, str):
        return value.strip()
    return _format_title_value(value)


def _format_composition_title(value):
    text = _format_title_label(value)
    return re.sub(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)", r"\1$_{\2}$", text)


def _first_row_value(row, columns):
    for col in columns:
        if col in row.index and not pd.isna(row[col]):
            return row[col]
    return None


def _projection_title(excel_path, row_index):
    row = pd.read_excel(excel_path).iloc[row_index]
    composition = _first_row_value(row, ["Composition", "composition"])
    alloy = _first_row_value(row, ["Alloy", "Unnamed: 0"])
    power = _first_row_value(row, ["power_w", "Power", "Power (W)", "P"])
    velocity = _first_row_value(row, ["velocity_m_s", "Velocity_m/s", "Velocity (m/s)", "v"])

    if composition is not None:
        title = _format_composition_title(composition)
    elif alloy is not None:
        title = f"Alloy {_format_title_value(alloy)}"
    else:
        return "Melt Pool Liquidus Projection"

    if power is None or velocity is None:
        return title
    return f"{title} P = {_format_title_value(power)}, V = {_format_title_value(velocity)}"


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
        liquidus_temperature = _load_liquidus_from_excel(excel_path, row_index)
        element_names = list(elements.keys())
    else:
        solutes = config["solutes"]
        element_names = config["elements"]
        liquidus_temperature = config["liquidus_temperature"]

    interfacial_energy = config.get("interfacial_energy")
    nb_nucleations_site = config.get(
        "nb_nucleations_site",
        DEFAULT_CET_NUCLEATION_SITES,
    )
    nucleation_undercooling = config.get(
        "nucleation_undercooling",
        DEFAULT_CET_NUCLEATION_UNDERCOOLING_K,
    )
    equiaxed_exponent = config.get(
        "equiaxed_exponent",
        DEFAULT_CET_EQUIAXED_EXPONENT,
    )
    spreadsheet_interfacial_energy = None
    if map_source == "excel_alloy" and interfacial_energy is None:
        spreadsheet_interfacial_energy = _load_interfacial_energy_from_excel(
            excel_path,
            row_index,
        )
        interfacial_energy = spreadsheet_interfacial_energy
        if (
            config.get("report_interfacial_energy", True)
            and interfacial_energy is not None
        ):
            print(
                "Using spreadsheet interfacial energy: "
                f"{interfacial_energy:.6g} J/m^2"
            )

    grid_metadata = None
    grid_npz_path = None
    if interfacial_energy is not None:
        grid_metadata = _cet_grid_cache_metadata(
            thermodynamic_database=config["thermo_db"],
            kinetic_database=config["kinetic_db"],
            element_names=element_names,
            solutes=solutes,
            primary_phase=config["primary_phase"],
            interfacial_energy=interfacial_energy,
            nb_nucleations_site=nb_nucleations_site,
            nucleation_undercooling=nucleation_undercooling,
            equiaxed_exponent=equiaxed_exponent,
            composition_unit=config["composition_unit"],
        )
        grid_npz_path = _cet_grid_npz_path(grid_metadata)
        cached_grid = _load_cet_grid_npz(grid_npz_path)
        if cached_grid is not None:
            print(f"Loaded CET grid NPZ: {grid_npz_path}")
            return (
                cached_grid[0],
                cached_grid[1],
                cached_grid[2],
                interfacial_energy,
            )

    with TCPython() as session:
        session.set_cache_folder(_tc_python_cache_dir())
        system = session.select_thermodynamic_and_kinetic_databases_with_elements(
            config["thermo_db"],
            config["kinetic_db"],
            element_names,
        ).get_system()
        interfacial_energy = resolve_interfacial_energy(
            system,
            solutes,
            config["primary_phase"],
            liquidus_temperature,
            interfacial_energy=interfacial_energy,
            composition_unit=config["composition_unit"],
            report=(
                config.get("report_interfacial_energy", True)
                and spreadsheet_interfacial_energy is None
            ),
        )

        if grid_metadata is None:
            grid_metadata = _cet_grid_cache_metadata(
                thermodynamic_database=config["thermo_db"],
                kinetic_database=config["kinetic_db"],
                element_names=element_names,
                solutes=solutes,
                primary_phase=config["primary_phase"],
                interfacial_energy=interfacial_energy,
                nb_nucleations_site=nb_nucleations_site,
                nucleation_undercooling=nucleation_undercooling,
                equiaxed_exponent=equiaxed_exponent,
                composition_unit=config["composition_unit"],
            )
            grid_npz_path = _cet_grid_npz_path(grid_metadata)
            cached_grid = _load_cet_grid_npz(grid_npz_path)
            if cached_grid is not None:
                print(f"Loaded CET grid NPZ: {grid_npz_path}")
                return (
                    cached_grid[0],
                    cached_grid[1],
                    cached_grid[2],
                    interfacial_energy,
                )

        (
            r_grid,
            g_grid,
            equiaxed_fraction,
            tip_radius,
            dendrite_tip_undercooling,
            nan_points,
        ) = get_CET_grid(
            system,
            solutes,
            primary_phase=config["primary_phase"],
            interfacial_energy=interfacial_energy,
            nb_nucleations_site=nb_nucleations_site,
            nucleation_undercooling=nucleation_undercooling,
            equiaxed_exponent=equiaxed_exponent,
            disable_output=True,
            composition_unit=config["composition_unit"],
        )
        _save_cet_grid_npz(
            grid_npz_path,
            grid_metadata,
            r_grid,
            g_grid,
            equiaxed_fraction,
            tip_radius,
            dendrite_tip_undercooling,
            nan_points,
        )
        print(f"Wrote CET grid NPZ: {grid_npz_path}")

    return (
        np.asarray(r_grid),
        np.asarray(g_grid),
        np.asarray(equiaxed_fraction, dtype=float),
        interfacial_energy,
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


def _equiaxed_colormap():
    """Return the nonnegative-fraction half of the GR-map color scale."""
    gr_norm = colors.Normalize(vmin=GR_MAP_COLOR_RANGE[0], vmax=GR_MAP_COLOR_RANGE[1])
    positive_start = gr_norm(0.0)
    sampled_colors = cm.seismic_r(np.linspace(positive_start, 1.0, 256))
    return colors.ListedColormap(sampled_colors, name="equiaxed_seismic_r")


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

    r_grid, g_grid, equiaxed_fraction, interfacial_energy = _calculate_cet_grid(
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
    projected.attrs["interfacial_energy_j_m2"] = interfacial_energy
    projected["y_plot"] = projected["y"].round(0)
    projected["z_plot"] = projected["z"].round(0)

    fraction = projected["equiaxed_fraction"].to_numpy(dtype=float)
    planar = fraction == -1.0
    columnar = fraction == 0.0
    equiaxed = ~(planar | columnar)
    equiaxed_cmap = _equiaxed_colormap()
    equiaxed_norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)

    columnar_y = np.concatenate(
        [
            projected.loc[columnar, "y_plot"].to_numpy(dtype=float),
            -projected.loc[columnar, "y_plot"].to_numpy(dtype=float),
        ]
    )
    columnar_z = np.concatenate(
        [
            projected.loc[columnar, "z_plot"].to_numpy(dtype=float),
            projected.loc[columnar, "z_plot"].to_numpy(dtype=float),
        ]
    )
    equiaxed_y = np.concatenate(
        [
            projected.loc[equiaxed, "y_plot"].to_numpy(dtype=float),
            -projected.loc[equiaxed, "y_plot"].to_numpy(dtype=float),
        ]
    )
    equiaxed_z = np.concatenate(
        [
            projected.loc[equiaxed, "z_plot"].to_numpy(dtype=float),
            projected.loc[equiaxed, "z_plot"].to_numpy(dtype=float),
        ]
    )
    equiaxed_values = np.concatenate([fraction[equiaxed], fraction[equiaxed]])
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
        color=equiaxed_cmap(equiaxed_norm(0.0)),
        marker="o",
        s=POINT_SIZE,
        linewidths=0,
        alpha=1.0,
        label="Columnar",
    )
    ax.scatter(
        equiaxed_y,
        equiaxed_z,
        c=equiaxed_values,
        cmap=equiaxed_cmap,
        norm=equiaxed_norm,
        marker="o",
        s=POINT_SIZE,
        linewidths=0,
        alpha=1.0,
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
    if np.any(fraction >= EQUIAXED_COLORBAR_MIN_FRACTION):
        equiaxed_mapper = cm.ScalarMappable(norm=equiaxed_norm, cmap=equiaxed_cmap)
        equiaxed_mapper.set_array([])
        colorbar = fig.colorbar(equiaxed_mapper, ax=ax, pad=0.02)
        colorbar.set_label("Equiaxed fraction", fontsize=label_fontsize)
        colorbar.ax.tick_params(labelsize=tick_fontsize)

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

    planar_count = int(planar.sum())
    columnar_count = int(columnar.sum())
    equiaxed_count = int(equiaxed.sum())
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
    print(
        "Microstructure-classified points: "
        f"planar={planar_count}, columnar={columnar_count}, "
        f"equiaxed={equiaxed_count} / {len(projected)}"
    )
    return output_path, deepest, projected


if __name__ == "__main__":
    try:
        plot_projected_liquidus()
    except Exception as exc:
        print(f"PlotMicrostructure failed: {exc}", file=sys.stderr)
        raise
